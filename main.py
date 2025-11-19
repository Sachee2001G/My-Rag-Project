from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import uvicorn
from datetime import datetime
import os

# Simple in-memory storage (for learning - in production use real databases)
documents_store = {}  # Stores processed documents
chat_history = {}  # Stores conversation history
bookings_store = []  # Stores interview bookings

app = FastAPI(title="Simple RAG API")

# DATA MODELS (defines the structure of our API requests/responses)

class ChunkResponse(BaseModel):
    """Response after uploading a document"""
    document_id: str
    filename: str
    num_chunks: int
    chunking_strategy: str


class QueryRequest(BaseModel):
    """Request to ask a question"""
    question: str
    document_id: str
    session_id: str = "default"


class QueryResponse(BaseModel):
    """Response with answer"""
    answer: str
    sources: List[str]
    session_id: str


class BookingRequest(BaseModel):
    """Request to book an interview"""
    name: str
    email: EmailStr
    date: str
    time: str


class BookingResponse(BaseModel):
    """Response after booking"""
    booking_id: int
    message: str

# HELPER FUNCTIONS (the actual logic)

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    
    if filename.endswith('.txt'):
        return file_content.decode('utf-8')
    elif filename.endswith('.pdf'):
        # For PDF, you'd need PyPDF2: pip install PyPDF2
        try:
            import PyPDF2
            import io
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except ImportError:
            raise HTTPException(
                status_code=400, 
                detail="PDF support requires PyPDF2. Install with: pip install PyPDF2"
            )
    else:
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files supported")
def chunk_text_simple(text: str, strategy: str = "fixed") -> List[str]:
    
    if strategy == "fixed":
        chunk_size = 500
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk.strip())
        return chunks
    
    elif strategy == "sentence":
        # Simple sentence splitting (split on '. ')
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < 500:
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    else:
        raise HTTPException(status_code=400, detail="Invalid chunking strategy")


def create_simple_embedding(text: str) -> List[float]:
   
    embedding = [0.0] * 384  # 384-dimensional vector
    
    for i, char in enumerate(text[:384]):
        embedding[i] = float(ord(char) % 100) / 100.0
    
    return embedding


def calculate_similarity(query_embedding: List[float], chunk_embedding: List[float]) -> float:
    
    # Dot product
    dot_product = sum(a * b for a, b in zip(query_embedding, chunk_embedding))
    
    # Magnitudes
    mag_a = sum(a * a for a in query_embedding) ** 0.5
    mag_b = sum(b * b for b in chunk_embedding) ** 0.5
    
    if mag_a == 0 or mag_b == 0:
        return 0.0
    
    return dot_product / (mag_a * mag_b)


def retrieve_relevant_chunks(question: str, document_id: str, top_k: int = 3) -> List[str]:
    
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_data = documents_store[document_id]
    chunks = doc_data['chunks']
    embeddings = doc_data['embeddings']
    
    # Create embedding for the question
    query_embedding = create_simple_embedding(question)
    
    # Calculate similarity scores for all chunks
    scores = []
    for i, chunk_embedding in enumerate(embeddings):
        similarity = calculate_similarity(query_embedding, chunk_embedding)
        scores.append((similarity, chunks[i]))
    
    # Sort by similarity (highest first) and get top_k
    scores.sort(reverse=True, key=lambda x: x[0])
    relevant_chunks = [chunk for _, chunk in scores[:top_k]]
    
    return relevant_chunks


def generate_answer(question: str, context_chunks: List[str], chat_history_list: List[str]) -> str:
    
    context = "\n\n".join(context_chunks)
    history = "\n".join(chat_history_list[-4:]) if chat_history_list else ""  # Last 2 turns
    
    # Simple template-based answer
    answer = f"Based on the document:\n\n{context[:500]}...\n\n"
    answer += f"To answer your question '{question}': Please review the context above. "
    answer += "This is a simple demo. In production, an LLM would generate a natural answer here."
    
    return answer



# API ENDPOINTS (the URLs your frontend/users will call)


@app.post("/upload", response_model=ChunkResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunking_strategy: str = "fixed"
) -> ChunkResponse:
    
    # Read file
    content = await file.read()
    
    # Extract text
    text = extract_text_from_file(content, file.filename)
    
    # Chunk text
    chunks = chunk_text_simple(text, chunking_strategy)
    
    # Create embeddings for each chunk
    embeddings = [create_simple_embedding(chunk) for chunk in chunks]
    
    # Generate unique ID
    doc_id = f"doc_{len(documents_store) + 1}_{datetime.now().timestamp()}"
    
    # Store everything
    documents_store[doc_id] = {
        'filename': file.filename,
        'chunks': chunks,
        'embeddings': embeddings,
        'chunking_strategy': chunking_strategy,
        'uploaded_at': datetime.now().isoformat()
    }
    
    return ChunkResponse(
        document_id=doc_id,
        filename=file.filename,
        num_chunks=len(chunks),
        chunking_strategy=chunking_strategy
    )


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest) -> QueryResponse:
    
    # Initialize chat history for this session if needed
    if request.session_id not in chat_history:
        chat_history[request.session_id] = []
    
    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(request.question, request.document_id)
    
    # Get chat history
    history = chat_history[request.session_id]
    
    # Generate answer
    answer = generate_answer(request.question, relevant_chunks, history)
    
    # Save to chat history
    chat_history[request.session_id].append(f"Q: {request.question}")
    chat_history[request.session_id].append(f"A: {answer}")
    
    return QueryResponse(
        answer=answer,
        sources=relevant_chunks,
        session_id=request.session_id
    )


@app.post("/book-interview", response_model=BookingResponse)
async def book_interview(booking: BookingRequest) -> BookingResponse:
    """
    Book an interview
    Simple storage in memory
    """
    booking_id = len(bookings_store) + 1
    
    booking_data = {
        'id': booking_id,
        'name': booking.name,
        'email': booking.email,
        'date': booking.date,
        'time': booking.time,
        'created_at': datetime.now().isoformat()
    }
    
    bookings_store.append(booking_data)
    
    return BookingResponse(
        booking_id=booking_id,
        message=f"Interview booked successfully for {booking.name} on {booking.date} at {booking.time}"
    )


@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    return {
        "documents": [
            {
                "id": doc_id,
                "filename": data['filename'],
                "chunks": len(data['chunks']),
                "uploaded_at": data['uploaded_at']
            }
            for doc_id, data in documents_store.items()
        ]
    }


@app.get("/bookings")
async def list_bookings():
    """List all bookings"""
    return {"bookings": bookings_store}


@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "Simple RAG API for Beginners",
        "endpoints": {
            "POST /upload": "Upload a document (.txt or .pdf)",
            "POST /query": "Ask questions about a document",
            "POST /book-interview": "Book an interview",
            "GET /documents": "List all documents",
            "GET /bookings": "List all bookings"
        }
    }



# RUN THE SERVER


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
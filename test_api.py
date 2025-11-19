

import requests
import json

# Base URL of your API
BASE_URL = "http://localhost:8000"


def test_upload_document():
    
    print("\n=== Testing Document Upload ===")
    
    # Create a simple test file
    test_content = """
    Python is a high-level programming language. 
    It was created by Guido van Rossum and first released in 1991.
    Python is known for its simplicity and readability.
    It is widely used in web development, data science, and automation.
    FastAPI is a modern web framework for building APIs with Python.
    """
    
    # Save to a file
    with open("test_doc.txt", "w") as f:
        f.write(test_content)
    
    # Upload the file
    with open("test_doc.txt", "rb") as f:
        files = {"file": ("test_doc.txt", f, "text/plain")}
        params = {"chunking_strategy": "fixed"}
        response = requests.post(f"{BASE_URL}/upload", files=files, params=params)
    
    if response.status_code == 200:
        data = response.json()
        print("✓ Upload successful!")
        print(f"  Document ID: {data['document_id']}")
        print(f"  Filename: {data['filename']}")
        print(f"  Number of chunks: {data['num_chunks']}")
        return data['document_id']
    else:
        print(f"✗ Upload failed: {response.text}")
        return None


def test_query_document(doc_id):
    """Test querying a document"""
    print("\n=== Testing Document Query ===")
    
    query_data = {
        "question": "What is Python?",
        "document_id": doc_id,
        "session_id": "test_session_1"
    }
    
    response = requests.post(f"{BASE_URL}/query", json=query_data)
    
    if response.status_code == 200:
        data = response.json()
        print("✓ Query successful!")
        print(f"  Question: {query_data['question']}")
        print(f"  Answer: {data['answer'][:200]}...")
        print(f"  Number of sources: {len(data['sources'])}")
        return True
    else:
        print(f"✗ Query failed: {response.text}")
        return False


def test_book_interview():
    """Test booking an interview"""
    print("\n=== Testing Interview Booking ===")
    
    booking_data = {
        "name": "John Doe",
        "email": "john@example.com",
        "date": "2025-12-01",
        "time": "10:00 AM"
    }
    
    response = requests.post(f"{BASE_URL}/book-interview", json=booking_data)
    
    if response.status_code == 200:
        data = response.json()
        print("✓ Booking successful!")
        print(f"  {data['message']}")
        print(f"  Booking ID: {data['booking_id']}")
        return True
    else:
        print(f"✗ Booking failed: {response.text}")
        return False


def test_list_documents():
    """Test listing all documents"""
    print("\n=== Testing List Documents ===")
    
    response = requests.get(f"{BASE_URL}/documents")
    
    if response.status_code == 200:
        data = response.json()
        print("✓ Retrieved documents list")
        print(f"  Total documents: {len(data['documents'])}")
        for doc in data['documents']:
            print(f"  - {doc['filename']} ({doc['chunks']} chunks)")
        return True
    else:
        print(f"✗ Failed to retrieve documents: {response.text}")
        return False


def test_list_bookings():
    """Test listing all bookings"""
    print("\n=== Testing List Bookings ===")
    
    response = requests.get(f"{BASE_URL}/bookings")
    
    if response.status_code == 200:
        data = response.json()
        print("✓ Retrieved bookings list")
        print(f"  Total bookings: {len(data['bookings'])}")
        for booking in data['bookings']:
            print(f"  - {booking['name']} on {booking['date']} at {booking['time']}")
        return True
    else:
        print(f"✗ Failed to retrieve bookings: {response.text}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Simple RAG API Test Suite")
    print("=" * 60)
    print("\nMake sure the server is running: python main.py")
    print("Press Enter to start tests...")
    input()
    
    # Test 1: Upload document
    doc_id = test_upload_document()
    
    if doc_id:
        # Test 2: Query document
        test_query_document(doc_id)
        
        # Test 3: List documents
        test_list_documents()
    
    # Test 4: Book interview
    test_book_interview()
    
    # Test 5: List bookings
    test_list_bookings()
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
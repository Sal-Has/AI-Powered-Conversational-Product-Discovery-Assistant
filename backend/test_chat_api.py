#!/usr/bin/env python3
"""
Test script for the /api/chat endpoint
"""

import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_chat_api():
    """Test the chat API endpoint."""
    print("🧪 Testing Chat API Endpoint")
    print("=" * 40)
    
    # API base URL
    base_url = "http://localhost:5000/api"
    
    # Test data
    test_queries = [
        "Find me a cheap Samsung phone with good battery",
        "iPhone with large storage under 50000",
        "Android phone with good camera",
        "Best smartphone for gaming"
    ]
    
    # First, we need to authenticate (assuming you have auth endpoints)
    # For testing, you might need to create a test user or use existing credentials
    
    print("📝 Note: This test requires:")
    print("1. Flask app running on localhost:5000")
    print("2. Valid JWT token for authentication")
    print("3. ChromaDB with product data")
    print("4. OpenAI API key configured")
    print()
    
    # Test without authentication (should fail)
    print("🔒 Testing without authentication...")
    try:
        response = requests.post(
            f"{base_url}/chat",
            json={"query": test_queries[0]},
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 401:
            print("✅ Authentication protection working correctly")
        else:
            print("⚠️ Expected 401 Unauthorized")
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    print("\n" + "=" * 40)
    print("💡 To fully test the API:")
    print("1. Start the Flask app: python app.py")
    print("2. Register/login to get JWT token")
    print("3. Run product pipeline to populate ChromaDB")
    print("4. Use the JWT token in Authorization header")
    print()
    print("Example curl command:")
    print('curl -X POST http://localhost:5000/api/chat \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -H "Authorization: Bearer YOUR_JWT_TOKEN" \\')
    print('  -d \'{"query": "Find me a Samsung phone"}\'')

def test_api_structure():
    """Test the API structure and imports."""
    print("\n🔍 Testing API Structure...")
    
    try:
        from chat_api import chat_bp, init_rag_components, generate_conversational_response
        print("✅ Chat API imports successful")
        
        from product_pipeline import JumiaProductPipeline
        print("✅ Product pipeline import successful")
        
        from openai import OpenAI
        print("✅ OpenAI import successful")
        
        print("✅ All required components available")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Test structure first
    if test_api_structure():
        # Then test API
        test_chat_api()
    else:
        print("❌ Cannot test API due to import errors")

#!/usr/bin/env python3
"""
Working example of OpenAI client initialization and usage.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def create_openai_client():
    """Create OpenAI client with proper initialization."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    # Simple, clean initialization
    client = OpenAI(api_key=api_key)
    return client

def test_client():
    """Test the OpenAI client."""
    try:
        client = create_openai_client()
        print("✅ OpenAI client created successfully")
        
        # Test API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful shopping assistant."},
                {"role": "user", "content": "Recommend a good smartphone under $500"}
            ],
            max_tokens=100
        )
        
        print("✅ API call successful")
        print(f"💬 Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_client()
    if success:
        print("\n🎉 OpenAI client is working correctly!")
    else:
        print("\n❌ OpenAI client test failed")

        
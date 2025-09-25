#!/usr/bin/env python3
"""
Simple test to verify OpenAI client initialization
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def test_openai_client():
    """Test OpenAI client initialization."""
    print("🧪 Testing OpenAI Client Initialization")
    print("=" * 40)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return False
    
    print(f"✅ API key loaded: {api_key[:20]}...")
    
    try:
        # Simple client initialization
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized successfully")
        
        # Test a simple completion
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful shopping assistant."},
                {"role": "user", "content": "Recommend a Samsung phone under 30000"}
            ],
            max_tokens=100
        )
        
        print("✅ OpenAI API call successful")
        print(f"💬 Response: {response.choices[0].message.content[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI client failed: {e}")
        return False

if __name__ == "__main__":
    success = test_openai_client()
    if success:
        print("\n🎉 OpenAI Client: WORKING!")
    else:
        print("\n❌ OpenAI Client: FAILED")

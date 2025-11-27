"""Test script to verify both LLM providers work correctly."""
import os
from dotenv import load_dotenv

load_dotenv()

from llm_service import LLMService

def test_provider(provider_name: str):
    print(f"\n{'='*60}\nTesting {provider_name.upper()}\n{'='*60}")
    
    try:
        service = LLMService(provider=provider_name)
        info = service.get_provider_info()
        print(f"✅ Provider: {info}")
        
        result = service.generate(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Recommend a smartphone under 15000 KES."}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        print(f"✅ Latency: {result['latency_ms']:.2f} ms")
        print(f"✅ Tokens: {result['output_tokens']}")
        print(f"✅ Response: {result['text'][:150]}...")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    # Test OpenAI
    # if os.getenv('OPENAI_API_KEY'):
    #     test_provider('openai')
    
    # Test Gemma 3
    test_provider('local_gemma3')
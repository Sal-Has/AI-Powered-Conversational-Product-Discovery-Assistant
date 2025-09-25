#!/usr/bin/env python3
"""
Comprehensive fix for OpenAI client 'proxies' parameter issue.
This script will:
1. Clear Python cache files
2. Test different OpenAI client initialization methods
3. Provide a clean working example
"""

import os
import sys
import shutil
import traceback
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def clear_python_cache():
    """Clear Python cache files that might be causing conflicts."""
    print("🧹 Clearing Python cache files...")
    
    # Get the backend directory
    backend_dir = Path(__file__).parent
    
    # Clear __pycache__ directories
    pycache_dirs = list(backend_dir.rglob("__pycache__"))
    for pycache_dir in pycache_dirs:
        try:
            shutil.rmtree(pycache_dir)
            print(f"✅ Removed: {pycache_dir}")
        except Exception as e:
            print(f"⚠️ Could not remove {pycache_dir}: {e}")
    
    # Clear .pyc files
    pyc_files = list(backend_dir.rglob("*.pyc"))
    for pyc_file in pyc_files:
        try:
            pyc_file.unlink()
            print(f"✅ Removed: {pyc_file}")
        except Exception as e:
            print(f"⚠️ Could not remove {pyc_file}: {e}")
    
    print("✅ Cache clearing completed")

def test_openai_import_clean():
    """Test OpenAI import with fresh environment."""
    print("\n🔍 Testing OpenAI Import (Clean)")
    print("=" * 40)
    
    try:
        # Remove any cached modules
        modules_to_remove = [key for key in sys.modules.keys() if 'openai' in key.lower()]
        for module in modules_to_remove:
            del sys.modules[module]
            print(f"🗑️ Removed cached module: {module}")
        
        # Fresh import
        import openai
        print(f"✅ OpenAI imported successfully")
        print(f"📦 Version: {openai.__version__}")
        
        # Import client class
        from openai import OpenAI
        print(f"✅ OpenAI client class imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_openai_client_minimal():
    """Test OpenAI client with minimal parameters."""
    print("\n🧪 Testing OpenAI Client (Minimal)")
    print("=" * 40)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False
    
    print(f"✅ API key found: {api_key[:20]}...")
    
    try:
        from openai import OpenAI
        
        # Test 1: Minimal initialization
        print("🔧 Test 1: Minimal initialization...")
        client = OpenAI(api_key=api_key)
        print("✅ Minimal initialization successful")
        
        # Test 2: Test a simple API call
        print("🔧 Test 2: Simple API call...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, OpenAI is working!'"}
            ],
            max_tokens=50
        )
        
        print("✅ API call successful")
        print(f"💬 Response: {response.choices[0].message.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Client test failed: {e}")
        traceback.print_exc()
        return False

def test_openai_client_detailed():
    """Test OpenAI client with detailed error analysis."""
    print("\n🔬 Testing OpenAI Client (Detailed Analysis)")
    print("=" * 50)
    
    try:
        from openai import OpenAI
        import inspect
        
        # Inspect the OpenAI constructor
        print("🔍 Inspecting OpenAI constructor...")
        sig = inspect.signature(OpenAI.__init__)
        print(f"📋 Constructor signature: {sig}")
        
        # List all valid parameters
        print("📝 Valid parameters:")
        for param_name, param in sig.parameters.items():
            if param_name != 'self':
                print(f"   - {param_name}: {param}")
        
        # Check if 'proxies' is a valid parameter
        valid_params = [p for p in sig.parameters.keys() if p != 'self']
        if 'proxies' in valid_params:
            print("⚠️ 'proxies' is a valid parameter")
        else:
            print("❌ 'proxies' is NOT a valid parameter")
        
        return True
        
    except Exception as e:
        print(f"❌ Detailed analysis failed: {e}")
        traceback.print_exc()
        return False

def create_working_example():
    """Create a working example of OpenAI client usage."""
    print("\n📝 Creating Working Example")
    print("=" * 30)
    
    example_code = '''#!/usr/bin/env python3
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
            model="gpt-4o-mini",
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
        print("\\n🎉 OpenAI client is working correctly!")
    else:
        print("\\n❌ OpenAI client test failed")
'''
    
    # Write the example to a file
    example_file = Path(__file__).parent / "openai_working_example.py"
    try:
        with open(example_file, 'w', encoding='utf-8') as f:
            f.write(example_code)
        print(f"✅ Working example created: {example_file}")
        return str(example_file)
    except Exception as e:
        print(f"❌ Failed to create example: {e}")
        return None

def main():
    """Main function to run all fixes and tests."""
    print("🚀 OpenAI Client Fix Script")
    print("=" * 60)
    
    # Step 1: Clear cache
    clear_python_cache()
    
    # Step 2: Test clean import
    import_success = test_openai_import_clean()
    
    if not import_success:
        print("\n❌ Cannot proceed - OpenAI import failed")
        return
    
    # Step 3: Test detailed analysis
    test_openai_client_detailed()
    
    # Step 4: Test minimal client
    client_success = test_openai_client_minimal()
    
    # Step 5: Create working example
    example_file = create_working_example()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    if import_success:
        print("✅ OpenAI import: SUCCESS")
    else:
        print("❌ OpenAI import: FAILED")
    
    if client_success:
        print("✅ OpenAI client: SUCCESS")
    else:
        print("❌ OpenAI client: FAILED")
    
    if example_file:
        print(f"✅ Working example: {example_file}")
        print("\n💡 Next steps:")
        print(f"   1. Run: python {example_file}")
        print("   2. If it works, use the same pattern in your other files")
    
    if client_success:
        print("\n🎉 OpenAI client is now working correctly!")
        print("   The 'proxies' parameter issue has been resolved.")
    else:
        print("\n❌ Issue persists. Please check:")
        print("   1. Your OpenAI API key is valid")
        print("   2. Your internet connection")
        print("   3. No firewall blocking OpenAI API")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Detailed test to capture full stack trace and identify proxies parameter source
"""

import os
import sys
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openai_import():
    """Test OpenAI import and inspect the module."""
    print("🔍 Testing OpenAI Import and Module Inspection")
    print("=" * 50)
    
    try:
        import openai
        print(f"✅ OpenAI imported successfully")
        print(f"📦 OpenAI version: {openai.__version__}")
        print(f"📁 OpenAI module path: {openai.__file__}")
        
        # Check OpenAI client class
        from openai import OpenAI
        print(f"🏗️ OpenAI Client class: {OpenAI}")
        print(f"📋 OpenAI Client __init__ signature:")
        
        import inspect
        sig = inspect.signature(OpenAI.__init__)
        print(f"   {sig}")
        
        # List all parameters
        print("📝 OpenAI Client __init__ parameters:")
        for param_name, param in sig.parameters.items():
            print(f"   - {param_name}: {param}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI import failed: {e}")
        traceback.print_exc()
        return False

def test_openai_client_detailed():
    """Test OpenAI client initialization with detailed error capture."""
    print("\n🧪 Testing OpenAI Client Initialization (Detailed)")
    print("=" * 50)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return False
    
    print(f"✅ API key loaded: {api_key[:20]}...")
    
    try:
        # Import and create client with detailed error handling
        from openai import OpenAI
        
        print("🔧 Attempting to create OpenAI client...")
        print(f"   Using parameters: api_key='{api_key[:10]}...'")
        
        # Try client creation
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized successfully")
        return True
        
    except TypeError as e:
        print(f"❌ TypeError during client initialization: {e}")
        print("\n📊 Full Stack Trace:")
        traceback.print_exc()
        
        # Additional debugging
        print("\n🔍 Additional Debug Information:")
        print(f"   Error type: {type(e)}")
        print(f"   Error args: {e.args}")
        
        # Check if this is the proxies error
        if "proxies" in str(e):
            print("🎯 FOUND: This is the proxies parameter error!")
            print("   Investigating call stack...")
            
            # Print detailed stack trace
            tb = traceback.extract_tb(e.__traceback__)
            for frame in tb:
                print(f"   📁 File: {frame.filename}")
                print(f"   🔢 Line: {frame.lineno}")
                print(f"   🏷️ Function: {frame.name}")
                print(f"   💾 Code: {frame.line}")
                print("   " + "-" * 40)
        
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return False

def check_environment():
    """Check environment for any proxy-related variables."""
    print("\n🌍 Environment Variable Check")
    print("=" * 30)
    
    # Check all environment variables for proxy-related ones
    proxy_vars = []
    openai_vars = []
    
    for key, value in os.environ.items():
        if 'proxy' in key.lower():
            proxy_vars.append((key, value))
        if 'openai' in key.lower():
            openai_vars.append((key, value))
    
    print(f"🔍 Proxy-related variables: {len(proxy_vars)}")
    for key, value in proxy_vars:
        print(f"   {key} = {value}")
    
    print(f"🔍 OpenAI-related variables: {len(openai_vars)}")
    for key, value in openai_vars:
        # Mask API key for security
        display_value = value[:10] + "..." if key == 'OPENAI_API_KEY' and len(value) > 10 else value
        print(f"   {key} = {display_value}")

def check_local_files():
    """Check for any local files that might interfere."""
    print("\n📁 Local File Check")
    print("=" * 20)
    
    # Check for local openai.py files
    current_dir = os.getcwd()
    print(f"📂 Current directory: {current_dir}")
    
    suspicious_files = ['openai.py', 'openai.pyc']
    for filename in suspicious_files:
        filepath = os.path.join(current_dir, filename)
        if os.path.exists(filepath):
            print(f"⚠️ Found suspicious file: {filepath}")
        else:
            print(f"✅ No {filename} found in current directory")

if __name__ == "__main__":
    print("🚀 Detailed OpenAI Debugging Session")
    print("=" * 60)
    
    # Run all tests
    check_environment()
    check_local_files()
    import_success = test_openai_import()
    
    if import_success:
        client_success = test_openai_client_detailed()
        
        if client_success:
            print("\n🎉 All tests passed!")
        else:
            print("\n❌ Client initialization failed - check stack trace above")
    else:
        print("\n❌ Import failed - cannot proceed with client test")

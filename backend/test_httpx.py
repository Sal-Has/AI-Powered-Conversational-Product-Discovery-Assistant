import httpx
import inspect
import os
from openai import OpenAI

print("🔍 HTTPX and OpenAI Compatibility Test")
print("=" * 50)

# Check HTTPX version
print(f"📦 HTTPX version: {httpx.__version__}")

# Check what parameters HTTPX Client accepts
sig = inspect.signature(httpx.Client.__init__)
httpx_params = list(sig.parameters.keys())
print(f"🔧 HTTPX Client parameters: {httpx_params}")
print(f"🔍 'proxies' in HTTPX params: {'proxies' in httpx_params}")

# Try creating HTTPX client directly
print("\n🧪 Testing HTTPX Client Creation:")
try:
    client = httpx.Client()
    print("✅ HTTPX Client created successfully")
    client.close()
except Exception as e:
    print(f"❌ HTTPX Client failed: {e}")

# Test with explicit proxy settings
print("\n🧪 Testing HTTPX Client with explicit proxy settings:")
try:
    # Test what happens when we pass proxies
    client = httpx.Client(proxies=None)
    print("✅ HTTPX Client with proxies=None created successfully")
    client.close()
except Exception as e:
    print(f"❌ HTTPX Client with proxies=None failed: {e}")

# Clear environment just in case
print("\n🧹 Clearing any proxy environment variables...")
proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY', 
              'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy']
for var in proxy_vars:
    if var in os.environ:
        print(f"   Removing {var}")
        del os.environ[var]
    else:
        print(f"   {var} not set")

# Try OpenAI with custom HTTP client
print("\n🤖 Testing OpenAI with custom HTTP client:")
try:
    # Create custom HTTP client
    http_client = httpx.Client()
    
    # Create OpenAI client
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY', 'test-key'),
        http_client=http_client
    )
    print("✅ OpenAI client with custom HTTP client created successfully")
    http_client.close()
except Exception as e:
    print(f"❌ OpenAI with custom HTTP client failed: {e}")
    import traceback
    traceback.print_exc()

# Try basic OpenAI client creation
print("\n🤖 Testing basic OpenAI client:")
try:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', 'test-key'))
    print("✅ Basic OpenAI client created successfully")
except Exception as e:
    print(f"❌ Basic OpenAI client failed: {e}")
    import traceback
    traceback.print_exc()
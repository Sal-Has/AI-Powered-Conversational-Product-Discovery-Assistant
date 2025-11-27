#!/usr/bin/env python3
"""
Run the Flask application with the chat API
"""

import os
from dotenv import load_dotenv
from app import create_app

# Load environment variables
load_dotenv()

def main():
    """Run the Flask application."""
    print("🚀 Starting AI-Powered Product Discovery API...")
    print("=" * 50)
    
    # Check required environment variables
    required_vars = ['OPENAI_API_KEY', 'SECRET_KEY', 'JWT_SECRET_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these in your .env file")
        return
    
    print("✅ Environment variables configured")
    
    # Create and run the app
    try:
        app = create_app()
        print("\n🌐 Available endpoints:")
        print("   - POST /api/chat - Main conversational endpoint")
        print("   - POST /api/search - Direct semantic search")
        print("   - GET  /api/status - System status")
        print("   - GET  /api/health - Health check")
        print("   - POST /api/auth/register - User registration")
        print("   - POST /api/auth/login - User login")
        print("\n🔑 Authentication required for chat endpoints")
        print("📊 Make sure ChromaDB has product data")
        print("\n🎯 Starting server on http://localhost:5000")
        print("=" * 50)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return

if __name__ == '__main__':
    main()

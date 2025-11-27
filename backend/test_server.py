#!/usr/bin/env python3
"""
Test Flask server for the chat API without authentication
"""

import os
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_test_app():
    """Create a test Flask app with the no-auth chat API."""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
    
    # Enable CORS
    CORS(app)
    
    # Import and register chat blueprint (no auth)
    try:
        from chat_api_no_auth import chat_no_auth_bp
        app.register_blueprint(chat_no_auth_bp, url_prefix='/api')
        print("✅ No-Auth Chat API registered successfully")
    except Exception as e:
        print(f"❌ Failed to register chat API: {e}")
        return None
    
    # Health check endpoint
    @app.route('/api/health')
    def health_check():
        return jsonify({
            'status': 'healthy', 
            'message': 'Test Chat API Server is running (NO AUTH)',
            'endpoints': [
                'POST /api/chat - NO AUTH REQUIRED',
                'POST /api/search - NO AUTH REQUIRED', 
                'GET /api/status - System status',
                'GET /api/health - Health check'
            ]
        }), 200
    
    # Test endpoint
    @app.route('/api/test')
    def test():
        return jsonify({
            'message': 'Test chat server is working!',
            'note': 'Chat endpoints work WITHOUT authentication',
            'test_query': 'curl -X POST http://localhost:5000/api/chat -H "Content-Type: application/json" -d \'{"query": "Samsung phone"}\''
        }), 200
    
    return app

def main():
    """Run the test chat server."""
    print("🚀 Starting Test Chat API Server (NO AUTH)...")
    print("=" * 50)
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️ Warning: OPENAI_API_KEY not found in environment")
        print("   Chat functionality may not work properly")
    else:
        print("✅ OpenAI API key found")
    
    # Create app
    app = create_test_app()
    if not app:
        print("❌ Failed to create app")
        return
    
    print("\n🌐 Available endpoints (NO AUTHENTICATION REQUIRED):")
    print("   - GET  /api/health - Health check")
    print("   - GET  /api/test - Test endpoint with instructions")
    print("   - POST /api/chat - Chat endpoint (NO AUTH)")
    print("   - POST /api/search - Search endpoint (NO AUTH)")
    print("   - GET  /api/status - System status")
    
    print("\n💡 Test commands:")
    print('   curl -X POST http://localhost:5000/api/chat -H "Content-Type: application/json" -d \'{"query": "Samsung phone"}\'')
    print('   curl -X POST http://localhost:5000/api/search -H "Content-Type: application/json" -d \'{"query": "iPhone", "k": 3}\'')
    
    print(f"\n🎯 Starting server on http://localhost:5000")
    print("=" * 50)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ Server failed to start: {e}")

if __name__ == '__main__':
    main()

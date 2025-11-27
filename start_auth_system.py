#!/usr/bin/env python3
"""
Startup script for the AI-Powered Product Discovery Assistant with Authentication
Starts Flask backend with JWT authentication and React frontend
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def print_banner():
    """Print startup banner."""
    print("🔐" + "=" * 60 + "🔐")
    print("   AI-POWERED PRODUCT DISCOVERY ASSISTANT")
    print("   Step 5: Full Authentication System")
    print("🔐" + "=" * 60 + "🔐")
    print()

def check_requirements():
    """Check if required directories and files exist."""
    print("🔍 Checking system requirements...")
    
    # Check backend
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found!")
        return False
    
    # Check frontend
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found!")
        return False
    
    # Check key files
    required_files = [
        "backend/app.py",
        "backend/auth.py",
        "backend/chat_api.py",
        "frontend/package.json",
        "frontend/src/components/ChatbotRAG.js",
        "frontend/src/components/Login.js",
        "frontend/src/components/Register.js"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ Required file missing: {file_path}")
            return False
    
    print("✅ All requirements satisfied!")
    return True

def start_backend():
    """Start the Flask backend server with authentication."""
    print("\n🔧 Starting Flask Backend with Authentication...")
    print("   - JWT Authentication System")
    print("   - RAG Pipeline with ChromaDB")
    print("   - OpenAI GPT Integration") 
    print("   - Product Semantic Search")
    print("   - User Registration & Login")
    
    backend_dir = Path("backend")
    
    try:
        # Start the full app with authentication
        process = subprocess.Popen(
            [sys.executable, "app.py"],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print("✅ Backend server starting on http://localhost:5000")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def start_frontend():
    """Start the React frontend development server."""
    print("\n⚛️  Starting React Frontend with Authentication...")
    print("   - Login & Registration Pages")
    print("   - JWT Token Management")
    print("   - Protected Chat Interface")
    print("   - Product Card Display")
    print("   - Real-time RAG Integration")
    
    frontend_dir = Path("frontend")
    
    try:
        # Check if node_modules exists
        if not (frontend_dir / "node_modules").exists():
            print("📦 Installing npm dependencies...")
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
        
        # Start React development server
        process = subprocess.Popen(
            ["npm", "start"],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print("✅ Frontend server starting on http://localhost:3000")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return None

def wait_for_servers():
    """Wait for servers to be ready."""
    print("\n⏳ Waiting for servers to initialize...")
    
    # Wait a bit for servers to start
    for i in range(10, 0, -1):
        print(f"   Opening browser in {i} seconds...", end='\r')
        time.sleep(1)
    
    print("\n🌐 Opening browser...")
    
    try:
        webbrowser.open("http://localhost:3000")
        print("✅ Browser opened to http://localhost:3000")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print("   Please manually open: http://localhost:3000")

def print_instructions():
    """Print usage instructions."""
    print("\n" + "=" * 60)
    print("🎯 AUTHENTICATED SYSTEM READY!")
    print("=" * 60)
    print()
    print("📱 Frontend (React): http://localhost:3000")
    print("🔧 Backend (Flask):  http://localhost:5000")
    print()
    print("🔑 Authentication Flow:")
    print("   1. Register a new account at /register")
    print("   2. Login with your credentials at /login")
    print("   3. Access the protected chat interface")
    print("   4. JWT tokens handled automatically")
    print()
    print("💬 Try these sample queries after login:")
    print('   - "Find me a Samsung phone under 30000"')
    print('   - "iPhone with good camera and battery"')
    print('   - "Android phone for gaming"')
    print('   - "Budget smartphone with dual SIM"')
    print()
    print("🔒 Security Features:")
    print("   - JWT access & refresh tokens")
    print("   - Automatic token refresh")
    print("   - Protected API endpoints")
    print("   - Session management")
    print()
    print("🛑 To stop: Press Ctrl+C")
    print("=" * 60)

def main():
    """Main startup function."""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ System requirements not met. Please check the setup.")
        return 1
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("\n❌ Failed to start backend server.")
        return 1
    
    # Wait a moment for backend to initialize
    time.sleep(5)
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("\n❌ Failed to start frontend server.")
        backend_process.terminate()
        return 1
    
    # Wait for servers and open browser
    wait_for_servers()
    
    # Print instructions
    print_instructions()
    
    try:
        # Keep both processes running
        while True:
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("\n❌ Backend process stopped unexpectedly")
                break
            
            if frontend_process.poll() is not None:
                print("\n❌ Frontend process stopped unexpectedly")
                break
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
        
        # Terminate processes
        if backend_process:
            backend_process.terminate()
            print("✅ Backend server stopped")
        
        if frontend_process:
            frontend_process.terminate()
            print("✅ Frontend server stopped")
        
        print("👋 Thank you for using the AI Product Discovery Assistant!")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

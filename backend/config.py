import os
from datetime import timedelta
from dotenv import load_dotenv
from pathlib import Path

# Load .env from parent directory (Final_Project folder)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-dev-secret-key'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    JWT_TOKEN_LOCATION = ['headers']
    JWT_HEADER_NAME = 'Authorization'
    JWT_HEADER_TYPE = 'Bearer'
    JWT_COOKIE_CSRF_PROTECT = False  # Add this line
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///chatbot_auth.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # CORS configuration
    CORS_ORIGINS = ['http://localhost:3000']  # React dev server
    
    # LLM Configuration
    LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'openai')  # 'openai' or 'local_gemma3'
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')  # Replace with your actual key
    
    # Model configurations
    OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-3.5-turbo')
    LOCAL_GEMMA_MODEL = "gemma-3-1b-it-qat"  # Local LM Studio model (Gemma 3 1B quantized)
    LOCAL_GEMMA_BASE_URL = "http://127.0.0.1:1234/v1"  # LM Studio OpenAI-compatible API

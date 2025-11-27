from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.set_password(password)

    def set_password(self, password):
        """Hash and set the password"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Check if provided password matches the hash"""
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        """Convert user object to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active
        }

    def __repr__(self):
        return f'<User {self.username}>'


class TokenBlocklist(db.Model):
    """Model to store blacklisted JWT tokens"""
    id = db.Column(db.Integer, primary_key=True)
    jti = db.Column(db.String(36), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<TokenBlocklist {self.jti}>'


class ChatSession(db.Model):
    """Model to store chat sessions"""
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120), nullable=False)
    title = db.Column(db.String(255), default='New Chat')  # ADD THIS LINE
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationship to chat messages
    messages = db.relationship('ChatMessage', backref='session', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        """Convert session to dictionary"""
        return {
            'id': self.id,
            'user_email': self.user_email,
            'title': self.title,  # ADD THIS LINE
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'is_active': self.is_active,
            'message_count': len(self.messages)
        }


class ChatMessage(db.Model):
    """Model to store individual chat messages"""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    query = db.Column(db.Text)  # Original user query
    products_json = db.Column(db.Text)  # JSON string of retrieved products
    action = db.Column(db.String(50))  # 'search', 'compare', 'clarify'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert message to dictionary"""
        import json
        return {
            'id': self.id,
            'session_id': self.session_id,
            'role': self.role,
            'content': self.content,
            'query': self.query,
            'products': json.loads(self.products_json) if self.products_json else [],
            'action': self.action,
            'created_at': self.created_at.isoformat()
        }
    
    def __repr__(self):
        return f'<ChatMessage {self.id} - {self.role}>'


class ProductIndex(db.Model):
    """Minimal index to track live metadata for products scraped into Chroma/Jumia DB.

    This does NOT replace the main product storage in Chroma; it only stores
    the latest known price/availability for a given product ID and URL so that
    background jobs and on-demand checks can update it.
    """

    id = db.Column(db.Integer, primary_key=True)
    # This should match the product "id" used in the Chroma documents / pipeline
    product_id = db.Column(db.String(100), unique=True, nullable=False)
    url = db.Column(db.String(500), nullable=False)

    last_seen_price_text = db.Column(db.String(100))
    last_seen_price_numeric = db.Column(db.Integer)
    last_seen_status = db.Column(db.String(50), default='unknown')  # in_stock, out_of_stock, not_found, unknown
    last_checked_at = db.Column(db.DateTime)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'product_id': self.product_id,
            'url': self.url,
            'last_seen_price_text': self.last_seen_price_text,
            'last_seen_price_numeric': self.last_seen_price_numeric,
            'last_seen_status': self.last_seen_status,
            'last_checked_at': self.last_checked_at.isoformat() if self.last_checked_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self):
        return f'<ProductIndex {self.product_id}>'


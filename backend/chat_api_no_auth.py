import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from flask import Blueprint, request, jsonify
import openai
from openai import OpenAI

from product_pipeline import JumiaProductPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
chat_no_auth_bp = Blueprint('chat_no_auth', __name__)

# Initialize RAG components
pipeline = None
openai_client = None

def init_rag_components():
    """Initialize RAG pipeline and OpenAI client."""
    global pipeline, openai_client
    
    try:
        # Initialize product pipeline for semantic search
        pipeline = JumiaProductPipeline(
            chroma_persist_directory="./chroma_db",
            collection_name="jumia_products"
        )
        logger.info("✅ RAG Pipeline initialized")
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        openai_client = OpenAI(api_key=api_key)
        logger.info("✅ OpenAI client initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG components: {e}")
        raise

# Initialize components when module loads
init_rag_components()

def generate_conversational_response(query: str, products: List[Dict[str, Any]]) -> str:
    """
    Generate conversational response using OpenAI GPT.
    
    Args:
        query: User's original query
        products: List of relevant products from semantic search
        
    Returns:
        Conversational response from GPT
    """
    if not products:
        return "I couldn't find any products matching your query. Could you try rephrasing your search or being more specific about what you're looking for?"
    
    # Prepare product information for the prompt
    product_info = []
    for i, product in enumerate(products, 1):
        specs_text = ""
        if product.get('specs') and isinstance(product['specs'], dict):
            key_specs = list(product['specs'].items())[:3]  # Top 3 specs
            specs_text = ", ".join([f"{k}: {v}" for k, v in key_specs])
        
        product_text = f"""
{i}. {product.get('name', 'Unknown Product')}
   Price: {product.get('price_text', 'Price not available')}
   Rating: {product.get('rating', 'No rating')}
   {f"Key specs: {specs_text}" if specs_text else ""}
   URL: {product.get('url', 'URL not available')}
"""
        product_info.append(product_text.strip())
    
    products_text = "\n\n".join(product_info)
    
    # Create system prompt
    system_prompt = """You are a helpful shopping assistant for Jumia Kenya, specializing in smartphones and electronics. 
Your role is to help customers find the perfect products based on their needs and budget.

Guidelines:
- Be conversational, friendly, and helpful
- Provide specific product recommendations from the search results
- Highlight key features that match the user's query
- Mention prices and help with budget considerations
- If comparing products, point out pros and cons
- Always encourage the user to check the product page for full details
- Keep responses concise but informative (max 300 words)
"""

    user_prompt = f"""
User Query: "{query}"

Here are the most relevant products I found:

{products_text}

Please provide a helpful, conversational response that:
1. Acknowledges the user's request
2. Recommends the most suitable products from the list
3. Explains why these products match their needs
4. Mentions key features, prices, and any standout characteristics
5. Encourages them to explore the product pages for more details
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=400,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"❌ OpenAI API error: {e}")
        return f"I found {len(products)} relevant products for your query, but I'm having trouble generating a detailed response right now. Please check the product list below for options that might interest you."

@chat_no_auth_bp.route('/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint for conversational product discovery (NO AUTHENTICATION).
    
    Expected JSON payload:
    {
        "query": "Find me a cheap Samsung phone with good battery"
    }
    
    Returns:
    {
        "success": true,
        "answer": "LLM-generated conversational response",
        "products": [list of relevant products],
        "query": "original user query",
        "timestamp": "2024-01-01T12:00:00Z"
    }
    """
    try:
        logger.info(f"🔍 Chat request received")
        
        # Validate request
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON'
            }), 400
        
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query is required'
            }), 400
        
        if len(query) > 500:
            return jsonify({
                'success': False,
                'error': 'Query too long (max 500 characters)'
            }), 400
        
        logger.info(f"🔍 Processing query: '{query}'")
        
        # Perform semantic search
        try:
            products = pipeline.semantic_search(query, k=5, score_threshold=0.0)
            logger.info(f"✅ Found {len(products)} products")
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            return jsonify({
                'success': False,
                'error': 'Search service temporarily unavailable'
            }), 500
        
        # Generate conversational response
        try:
            answer = generate_conversational_response(query, products)
            logger.info("✅ Generated conversational response")
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            # Fallback response
            if products:
                answer = f"I found {len(products)} products that might interest you. Here are the top recommendations based on your search."
            else:
                answer = "I couldn't find any products matching your query. Could you try rephrasing your search?"
        
        # Format products for response (remove embeddings and internal fields)
        formatted_products = []
        for product in products:
            formatted_product = {
                'id': product.get('id'),
                'name': product.get('name'),
                'title': product.get('title', product.get('name')),
                'price_text': product.get('price_text'),
                'price_numeric': product.get('price_numeric'),
                'rating': product.get('rating'),
                'url': product.get('url'),
                'image_url': product.get('image_url'),
                'description': product.get('description', '')[:200] + "..." if len(product.get('description', '')) > 200 else product.get('description', ''),
                'specs': product.get('specs', {}),
                'similarity_score': round(product.get('similarity_score', 0), 3)
            }
            formatted_products.append(formatted_product)
        
        # Return response
        response_data = {
            'success': True,
            'answer': answer,
            'products': formatted_products,
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'results_count': len(formatted_products)
        }
        
        logger.info(f"✅ Chat response generated successfully")
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'timestamp': datetime.now().isoformat()
        }), 500

@chat_no_auth_bp.route('/search', methods=['POST'])
def search():
    """
    Direct semantic search endpoint (without conversational response, NO AUTHENTICATION).
    
    Expected JSON payload:
    {
        "query": "Samsung smartphone",
        "k": 5,
        "score_threshold": 0.0
    }
    """
    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        query = data.get('query', '').strip()
        k = data.get('k', 5)
        score_threshold = data.get('score_threshold', 0.0)
        
        if not query:
            return jsonify({'success': False, 'error': 'Query is required'}), 400
        
        # Validate parameters
        k = max(1, min(k, 20))  # Limit between 1 and 20
        score_threshold = max(0.0, min(score_threshold, 1.0))  # Between 0 and 1
        
        # Perform search
        products = pipeline.semantic_search(query, k=k, score_threshold=score_threshold)
        
        # Format response
        formatted_products = []
        for product in products:
            formatted_product = {
                'id': product.get('id'),
                'name': product.get('name'),
                'price_text': product.get('price_text'),
                'url': product.get('url'),
                'image_url': product.get('image_url'),
                'rating': product.get('rating'),
                'similarity_score': round(product.get('similarity_score', 0), 3)
            }
            formatted_products.append(formatted_product)
        
        return jsonify({
            'success': True,
            'products': formatted_products,
            'query': query,
            'results_count': len(formatted_products),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Search endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': 'Search failed',
            'timestamp': datetime.now().isoformat()
        }), 500

@chat_no_auth_bp.route('/status', methods=['GET'])
def status():
    """Get RAG system status."""
    try:
        # Check pipeline status
        pipeline_status = "healthy"
        collection_stats = {}
        
        try:
            collection_stats = pipeline.get_collection_stats()
        except:
            pipeline_status = "error"
        
        # Check OpenAI status
        openai_status = "healthy" if openai_client else "error"
        
        return jsonify({
            'success': True,
            'status': {
                'pipeline': pipeline_status,
                'openai': openai_status,
                'collection': collection_stats
            },
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging
from product_pipeline import JumiaProductPipeline
import os

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
product_bp = Blueprint('products', __name__)

# Initialize pipeline (singleton pattern)
pipeline = None

def get_pipeline():
    """Get or create pipeline instance."""
    global pipeline
    if pipeline is None:
        # Initialize with ChromaDB parameters
        pipeline = JumiaProductPipeline(
            chroma_persist_directory="./chroma_db"
        )
    return pipeline

@product_bp.route('/scrape', methods=['POST'])
@jwt_required()
def scrape_products():
    """
    Scrape products from specified categories.
    Requires authentication.
    """
    try:
        current_user = get_jwt_identity()
        logger.info(f"User {current_user} initiated product scraping")
        
        data = request.get_json()
        
        # Validate input
        if not data or 'category_urls' not in data:
            return jsonify({
                'error': 'category_urls is required',
                'example': {
                    'category_urls': [
                        'https://www.jumia.com.eg/phones-tablets/',
                        'https://www.jumia.com.eg/electronics/'
                    ],
                    'max_products_per_category': 20
                }
            }), 400
        
        category_urls = data['category_urls']
        max_products_per_category = data.get('max_products_per_category', 20)
        
        if not isinstance(category_urls, list) or not category_urls:
            return jsonify({'error': 'category_urls must be a non-empty list'}), 400
        
        # Validate URLs
        for url in category_urls:
            if not isinstance(url, str) or not url.startswith('http'):
                return jsonify({'error': f'Invalid URL: {url}'}), 400
        
        # Run pipeline
        pipeline_instance = get_pipeline()
        result = pipeline_instance.run_pipeline(category_urls, max_products_per_category)
        
        return jsonify({
            'success': True,
            'message': 'Product scraping completed',
            'result': result,
            'user': current_user
        }), 200
        
    except Exception as e:
        logger.error(f"Error in scrape_products: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@product_bp.route('/search', methods=['POST'])
@jwt_required()
def search_products():
    """
    Perform semantic search for products.
    Requires authentication.
    """
    try:
        current_user = get_jwt_identity()
        data = request.get_json()
        
        # Validate input
        if not data or 'query' not in data:
            return jsonify({
                'error': 'query is required',
                'example': {
                    'query': 'smartphone android',
                    'k': 5
                }
            }), 400
        
        query = data['query']
        k = data.get('k', 5)
        
        if not isinstance(query, str) or not query.strip():
            return jsonify({'error': 'query must be a non-empty string'}), 400
        
        if not isinstance(k, int) or k < 1 or k > 50:
            return jsonify({'error': 'k must be an integer between 1 and 50'}), 400
        
        # Perform search
        pipeline_instance = get_pipeline()
        results = pipeline_instance.semantic_search(query.strip(), k)
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'count': len(results),
            'user': current_user
        }), 200
        
    except Exception as e:
        logger.error(f"Error in search_products: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@product_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_collection_stats():
    """
    Get statistics about the product collection.
    Requires authentication.
    """
    try:
        current_user = get_jwt_identity()
        pipeline_instance = get_pipeline()
        
        # Get collection count
        try:
            collection_info = pipeline_instance.collection.count()
            stats = {
                'total_products': collection_info,
                'collection_name': 'jumia_products',
                'status': 'active'
            }
        except Exception as e:
            stats = {
                'total_products': 0,
                'collection_name': 'jumia_products',
                'status': 'empty or error',
                'error': str(e)
            }
        
        return jsonify({
            'success': True,
            'stats': stats,
            'user': current_user
        }), 200
        
    except Exception as e:
        logger.error(f"Error in get_collection_stats: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@product_bp.route('/categories', methods=['GET'])
def get_sample_categories():
    """
    Get sample Jumia category URLs for testing.
    Public endpoint (no authentication required).
    """
    sample_categories = {
        'electronics': [
            'https://www.jumia.com.eg/phones-tablets/',
            'https://www.jumia.com.eg/electronics/',
            'https://www.jumia.com.eg/computers/',
            'https://www.jumia.com.eg/televisions/'
        ],
        'fashion': [
            'https://www.jumia.com.eg/fashion/',
            'https://www.jumia.com.eg/mens-fashion/',
            'https://www.jumia.com.eg/womens-fashion/',
            'https://www.jumia.com.eg/shoes/'
        ],
        'home': [
            'https://www.jumia.com.eg/home-kitchen/',
            'https://www.jumia.com.eg/furniture/',
            'https://www.jumia.com.eg/appliances/'
        ],
        'beauty': [
            'https://www.jumia.com.eg/beauty/',
            'https://www.jumia.com.eg/health-beauty/',
            'https://www.jumia.com.eg/perfumes/'
        ]
    }
    
    return jsonify({
        'success': True,
        'message': 'Sample Jumia category URLs',
        'categories': sample_categories,
        'usage': 'Use these URLs with the /scrape endpoint to populate the product database'
    }), 200

@product_bp.route('/health', methods=['GET'])
def pipeline_health():
    """
    Check if the product pipeline is healthy.
    Public endpoint (no authentication required).
    """
    try:
        pipeline_instance = get_pipeline()
        
        # Test if we can access the collection
        collection_accessible = True
        try:
            pipeline_instance.collection.count()
        except:
            collection_accessible = False
        
        # Test if embedding model is loaded
        model_loaded = hasattr(pipeline_instance, 'embedding_model') and pipeline_instance.embedding_model is not None
        
        health_status = {
            'pipeline_initialized': pipeline_instance is not None,
            'collection_accessible': collection_accessible,
            'embedding_model_loaded': model_loaded,
            'overall_status': 'healthy' if all([pipeline_instance, collection_accessible, model_loaded]) else 'degraded'
        }
        
        status_code = 200 if health_status['overall_status'] == 'healthy' else 503
        
        return jsonify({
            'success': True,
            'health': health_status
        }), status_code
        
    except Exception as e:
        logger.error(f"Error in pipeline_health: {str(e)}")
        return jsonify({
            'success': False,
            'health': {
                'overall_status': 'unhealthy',
                'error': str(e)
            }
        }), 503

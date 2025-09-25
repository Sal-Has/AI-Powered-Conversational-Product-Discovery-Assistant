from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging
import os
from rag_pipeline import RAGPipeline

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
rag_bp = Blueprint('rag', __name__)

# Initialize RAG pipeline (singleton pattern)
rag_pipeline = None

def get_rag_pipeline():
    """Get or create RAG pipeline instance."""
    global rag_pipeline
    if rag_pipeline is None:
        try:
            rag_pipeline = RAGPipeline(
                chroma_persist_directory="./chroma_db",
                collection_name="jumia_products",
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                llm_model="gpt-4o-mini"
            )
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    return rag_pipeline

@rag_bp.route('/recommend', methods=['POST'])
@jwt_required()
def generate_recommendation():
    """
    Generate product recommendations using the complete RAG pipeline.
    Combines semantic search with LLM-generated conversational responses.
    """
    try:
        current_user = get_jwt_identity()
        logger.info(f"User {current_user} requested RAG recommendation")
        
        data = request.get_json()
        
        # Validate input
        if not data or 'query' not in data:
            return jsonify({
                'error': 'query is required',
                'example': {
                    'query': 'Samsung smartphone with good camera under 50000',
                    'k': 5,
                    'score_threshold': 0.0,
                    'include_reasoning': True
                }
            }), 400
        
        query = data['query']
        k = data.get('k', 5)
        score_threshold = data.get('score_threshold', 0.0)
        include_reasoning = data.get('include_reasoning', True)
        
        # Validate parameters
        if not isinstance(query, str) or not query.strip():
            return jsonify({'error': 'query must be a non-empty string'}), 400
        
        if not isinstance(k, int) or k < 1 or k > 20:
            return jsonify({'error': 'k must be an integer between 1 and 20'}), 400
        
        if not isinstance(score_threshold, (int, float)) or score_threshold < 0 or score_threshold > 1:
            return jsonify({'error': 'score_threshold must be a number between 0 and 1'}), 400
        
        # Generate recommendation using RAG pipeline
        pipeline = get_rag_pipeline()
        result = pipeline.generate_recommendation(
            user_query=query.strip(),
            k=k,
            score_threshold=score_threshold,
            include_reasoning=include_reasoning
        )
        
        # Add user info to response
        result['user'] = current_user
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in generate_recommendation: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e),
            'llm_response': 'I apologize, but I encountered an error while processing your request. Please try again later.'
        }), 500

@rag_bp.route('/compare', methods=['POST'])
@jwt_required()
def compare_products():
    """
    Compare multiple products using the RAG pipeline.
    """
    try:
        current_user = get_jwt_identity()
        data = request.get_json()
        
        # Validate input
        if not data or 'products' not in data:
            return jsonify({
                'error': 'products list is required',
                'example': {
                    'products': ['Samsung Galaxy S21', 'iPhone 13'],
                    'comparison_criteria': 'features and value'
                }
            }), 400
        
        products = data['products']
        comparison_criteria = data.get('comparison_criteria', 'features and value')
        
        if not isinstance(products, list) or len(products) < 2:
            return jsonify({'error': 'products must be a list with at least 2 items'}), 400
        
        if len(products) > 5:
            return jsonify({'error': 'maximum 5 products can be compared at once'}), 400
        
        # Generate comparison using RAG pipeline
        pipeline = get_rag_pipeline()
        result = pipeline.compare_products(
            product_names=products,
            comparison_criteria=comparison_criteria
        )
        
        # Add user info to response
        result['user'] = current_user
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in compare_products: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@rag_bp.route('/batch', methods=['POST'])
@jwt_required()
def batch_recommendations():
    """
    Generate recommendations for multiple queries in batch.
    """
    try:
        current_user = get_jwt_identity()
        data = request.get_json()
        
        # Validate input
        if not data or 'queries' not in data:
            return jsonify({
                'error': 'queries list is required',
                'example': {
                    'queries': [
                        'Samsung smartphone under 50000',
                        'iPhone with good camera',
                        'budget Android phone'
                    ],
                    'k': 3
                }
            }), 400
        
        queries = data['queries']
        k = data.get('k', 3)
        
        if not isinstance(queries, list) or not queries:
            return jsonify({'error': 'queries must be a non-empty list'}), 400
        
        if len(queries) > 10:
            return jsonify({'error': 'maximum 10 queries can be processed in batch'}), 400
        
        # Validate each query
        for query in queries:
            if not isinstance(query, str) or not query.strip():
                return jsonify({'error': 'all queries must be non-empty strings'}), 400
        
        # Generate batch recommendations
        pipeline = get_rag_pipeline()
        results = pipeline.batch_recommendations(queries=queries, k=k)
        
        return jsonify({
            'success': True,
            'results': results,
            'query_count': len(queries),
            'user': current_user
        }), 200
        
    except Exception as e:
        logger.error(f"Error in batch_recommendations: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@rag_bp.route('/status', methods=['GET'])
@jwt_required()
def get_rag_status():
    """
    Get the status of the RAG pipeline components.
    """
    try:
        current_user = get_jwt_identity()
        
        pipeline = get_rag_pipeline()
        status = pipeline.get_pipeline_status()
        
        # Add user info to response
        status['user'] = current_user
        
        return jsonify(status), 200 if status['success'] else 503
        
    except Exception as e:
        logger.error(f"Error in get_rag_status: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e),
            'pipeline_status': 'unhealthy'
        }), 503

@rag_bp.route('/health', methods=['GET'])
def rag_health():
    """
    Check if the RAG pipeline is healthy.
    Public endpoint (no authentication required).
    """
    try:
        pipeline = get_rag_pipeline()
        
        # Test basic functionality
        health_checks = {
            'pipeline_initialized': pipeline is not None,
            'retriever_available': hasattr(pipeline, 'retriever') and pipeline.retriever is not None,
            'llm_service_available': hasattr(pipeline, 'llm_service') and pipeline.llm_service is not None,
            'openai_api_key_set': bool(os.getenv('OPENAI_API_KEY'))
        }
        
        # Check retriever collection
        try:
            retriever_stats = pipeline.retriever.get_collection_stats()
            health_checks['retriever_collection_accessible'] = retriever_stats.get('status') == 'active'
            health_checks['products_available'] = retriever_stats.get('total_products', 0) > 0
        except:
            health_checks['retriever_collection_accessible'] = False
            health_checks['products_available'] = False
        
        overall_healthy = all(health_checks.values())
        
        return jsonify({
            'success': True,
            'health_checks': health_checks,
            'overall_status': 'healthy' if overall_healthy else 'degraded',
            'message': 'RAG pipeline is ready for recommendations' if overall_healthy else 'RAG pipeline has some issues'
        }), 200 if overall_healthy else 503
        
    except Exception as e:
        logger.error(f"Error in rag_health: {str(e)}")
        return jsonify({
            'success': False,
            'health_checks': {'pipeline_initialized': False},
            'overall_status': 'unhealthy',
            'error': str(e)
        }), 503

@rag_bp.route('/models', methods=['GET'])
def get_available_models():
    """
    Get information about available models and configuration.
    Public endpoint (no authentication required).
    """
    return jsonify({
        'success': True,
        'available_models': {
            'llm_models': ['gpt-4o-mini', 'gpt-4'],
            'embedding_models': ['multi-qa-MiniLM-L6-cos-v1'],
            'current_llm': 'gpt-4o-mini',
            'current_embedding': 'multi-qa-MiniLM-L6-cos-v1'
        },
        'features': {
            'semantic_search': True,
            'conversational_recommendations': True,
            'product_comparison': True,
            'batch_processing': True,
            'real_time_responses': True
        },
        'limits': {
            'max_products_per_query': 20,
            'max_products_for_comparison': 5,
            'max_batch_queries': 10,
            'max_query_length': 500
        }
    }), 200

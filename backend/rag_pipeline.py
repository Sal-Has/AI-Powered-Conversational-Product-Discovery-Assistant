import os
import sys
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from product_pipeline import JumiaProductPipeline
from llm_service import LLMService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation pipeline for product recommendations.
    Combines semantic search (Step 2) with LLM generation (Step 3).
    """
    
    def __init__(self, 
                 chroma_persist_directory: str = "./chroma_db",
                 collection_name: str = "jumia_products",
                 openai_api_key: Optional[str] = None,
                 llm_model: str = "gpt-4o-mini",
                 embedding_model: str = "multi-qa-MiniLM-L6-cos-v1"):
        """
        Initialize the RAG pipeline.
        
        Args:
            chroma_persist_directory: ChromaDB persistence directory
            collection_name: ChromaDB collection name
            openai_api_key: OpenAI API key (if None, reads from environment)
            llm_model: OpenAI model to use (gpt-4o-mini or gpt-4)
            embedding_model: SentenceTransformer model for embeddings
        """
        self.collection_name = collection_name
        
        # Initialize retriever (Step 2)
        logger.info("🔄 Initializing semantic retriever...")
        self.retriever = JumiaProductPipeline(
            chroma_persist_directory=chroma_persist_directory,
            collection_name=collection_name,
            embedding_model=embedding_model
        )
        logger.info("✅ Semantic retriever initialized")
        
        # Initialize LLM service (Step 3)
        logger.info("🔄 Initializing LLM service...")
        self.llm_service = LLMService(
            api_key=openai_api_key,
            model=llm_model
        )
        logger.info("✅ LLM service initialized")
        
        logger.info("🎉 RAG Pipeline ready!")
    
    def generate_recommendation(self, 
                              user_query: str, 
                              k: int = 5,
                              score_threshold: float = 0.0,
                              include_reasoning: bool = True) -> Dict[str, Any]:
        """
        Generate product recommendations using the complete RAG pipeline.
        
        Args:
            user_query: User's search query
            k: Number of products to retrieve
            score_threshold: Minimum similarity score for retrieval
            include_reasoning: Whether to include LLM reasoning
            
        Returns:
            Dictionary with LLM response and structured product data
        """
        logger.info(f"🔍 Processing query: '{user_query}'")
        
        try:
            # Step 1: Semantic retrieval
            logger.info("📚 Retrieving relevant products...")
            retrieved_products = self.retriever.semantic_search(
                query=user_query,
                k=k,
                score_threshold=score_threshold
            )
            
            if not retrieved_products:
                logger.info("⚠️ No products found matching the query")
                return self.llm_service.generate_shopping_recommendation(
                    user_query=user_query,
                    retrieved_products=[]
                )
            
            logger.info(f"✅ Retrieved {len(retrieved_products)} relevant products")
            
            # Step 2: LLM generation
            if include_reasoning:
                logger.info("🤖 Generating conversational recommendation...")
                recommendation = self.llm_service.generate_shopping_recommendation(
                    user_query=user_query,
                    retrieved_products=retrieved_products
                )
            else:
                # Return just the products without LLM processing
                recommendation = {
                    "success": True,
                    "query": user_query,
                    "llm_response": "Products retrieved successfully (LLM reasoning disabled)",
                    "products": retrieved_products,
                    "product_count": len(retrieved_products),
                    "model_used": "retrieval_only",
                    "generated_at": datetime.now().isoformat()
                }
            
            logger.info("✅ RAG pipeline completed successfully")
            return recommendation
            
        except Exception as e:
            logger.error(f"❌ RAG pipeline failed: {e}")
            return {
                "success": False,
                "query": user_query,
                "error": str(e),
                "llm_response": "I'm experiencing technical difficulties. Please try again later.",
                "products": [],
                "product_count": 0,
                "generated_at": datetime.now().isoformat()
            }
    
    def compare_products(self, 
                        product_names: List[str],
                        comparison_criteria: str = "features and value") -> Dict[str, Any]:
        """
        Compare specific products using the RAG pipeline.
        
        Args:
            product_names: List of product names to compare
            comparison_criteria: What to focus on in comparison
            
        Returns:
            Dictionary with comparison response
        """
        logger.info(f"🔍 Comparing products: {product_names}")
        
        all_products = []
        
        # Retrieve each product
        for product_name in product_names:
            products = self.retriever.semantic_search(query=product_name, k=3)
            if products:
                # Take the most relevant match
                all_products.append(products[0])
        
        if len(all_products) < 2:
            return {
                "success": False,
                "error": "Could not find enough products for comparison",
                "llm_response": "I couldn't find enough products to make a meaningful comparison. Please check the product names and try again.",
                "products": all_products
            }
        
        # Generate comparison using LLM
        return self.llm_service.generate_product_comparison(
            products=all_products,
            comparison_criteria=comparison_criteria
        )
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get the status of the RAG pipeline components."""
        try:
            # Check retriever status
            retriever_stats = self.retriever.get_collection_stats()
            
            # Check LLM service status
            llm_status = {
                "model": self.llm_service.model,
                "max_tokens": self.llm_service.max_tokens,
                "temperature": self.llm_service.temperature,
                "status": "active"
            }
            
            return {
                "success": True,
                "retriever": retriever_stats,
                "llm_service": llm_status,
                "pipeline_status": "healthy",
                "checked_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "pipeline_status": "unhealthy",
                "checked_at": datetime.now().isoformat()
            }
    
    def batch_recommendations(self, 
                            queries: List[str],
                            k: int = 3) -> List[Dict[str, Any]]:
        """
        Generate recommendations for multiple queries in batch.
        
        Args:
            queries: List of user queries
            k: Number of products per query
            
        Returns:
            List of recommendation responses
        """
        logger.info(f"🔄 Processing {len(queries)} queries in batch...")
        
        results = []
        for query in queries:
            try:
                recommendation = self.generate_recommendation(
                    user_query=query,
                    k=k
                )
                results.append(recommendation)
            except Exception as e:
                logger.error(f"❌ Batch processing failed for query '{query}': {e}")
                results.append({
                    "success": False,
                    "query": query,
                    "error": str(e),
                    "llm_response": "Failed to process this query in batch.",
                    "products": []
                })
        
        logger.info(f"✅ Batch processing completed: {len(results)} results")
        return results


def main():
    """Test the RAG pipeline."""
    print("🧪 Testing Complete RAG Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    try:
        rag = RAGPipeline()
        print("✅ RAG Pipeline initialized")
    except Exception as e:
        print(f"❌ Failed to initialize RAG pipeline: {e}")
        return
    
    # Test queries
    test_queries = [
        "Samsung smartphone with good camera under 50000",
        "iPhone with large storage",
        "budget Android phone",
        "smartphone for gaming",
        "phone with long battery life"
    ]
    
    print(f"\n🔍 Testing {len(test_queries)} queries...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: '{query}' ---")
        
        try:
            result = rag.generate_recommendation(query, k=3)
            
            if result['success']:
                print(f"✅ Found {result['product_count']} products")
                print(f"🤖 LLM Response:")
                print(f"   {result['llm_response'][:200]}...")
                
                if result['products']:
                    print(f"📱 Top Product: {result['products'][0]['name']}")
                    print(f"💰 Price: {result['products'][0]['price_text']}")
            else:
                print(f"❌ Query failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    # Test pipeline status
    print(f"\n📊 Pipeline Status:")
    status = rag.get_pipeline_status()
    if status['success']:
        print(f"✅ Pipeline: {status['pipeline_status']}")
        print(f"📚 Products in DB: {status['retriever']['total_products']}")
        print(f"🤖 LLM Model: {status['llm_service']['model']}")
    else:
        print(f"❌ Status check failed: {status.get('error')}")


if __name__ == "__main__":
    main()

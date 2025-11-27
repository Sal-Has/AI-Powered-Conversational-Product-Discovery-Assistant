import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import uuid

import requests
from bs4 import BeautifulSoup
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from basic_scraper_with_storage import BasicJumiaScraper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JumiaProductPipeline:
    """
    Enhanced Jumia product pipeline using ChromaDB vector database.
    Combines web scraping with semantic search capabilities.
    """
    
    def __init__(self, 
                 chroma_persist_directory: str = "./chroma_db",
                 collection_name: str = "jumia_products",
                 embedding_model: str = "multi-qa-MiniLM-L6-cos-v1"):
        """
        Initialize the pipeline with ChromaDB client and embedding model.
        
        Args:
            chroma_persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            embedding_model: SentenceTransformer model name
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.chroma_persist_directory = chroma_persist_directory
        
        # Initialize ChromaDB client
        try:
            os.makedirs(chroma_persist_directory, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(
                path=chroma_persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info(f"✅ Connected to ChromaDB at {chroma_persist_directory}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB: {e}")
            raise
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"✅ Loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
        
        # Initialize scraper
        self.scraper = BasicJumiaScraper()
        
        # Initialize collection
        self._init_collection()
    
    def _init_collection(self):
        """Initialize ChromaDB collection if it doesn't exist."""
        try:
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"✅ Using ChromaDB collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize collection: {e}")
            raise
    
    def generate_embeddings(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for product descriptions.
        
        Args:
            products: List of product dictionaries
            
        Returns:
            List of products with embeddings
        """
        if not products:
            return []
        
        logger.info(f"🔄 Generating embeddings for {len(products)} products...")
        
        # Prepare texts for embedding
        texts = []
        for product in products:
            # Combine title, description, and key specs for embedding
            text_parts = []
            if product.get('name'):
                text_parts.append(product['name'])
            if product.get('description'):
                text_parts.append(product['description'])
            
            # Add key specs if available
            if product.get('specs'):
                try:
                    specs = json.loads(product['specs']) if isinstance(product['specs'], str) else product['specs']
                    for key, value in specs.items():
                        if key and value:
                            text_parts.append(f"{key}: {value}")
                except:
                    pass
            
            combined_text = " ".join(text_parts)
            texts.append(combined_text)
        
        # Generate embeddings
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            logger.info(f"✅ Generated {len(embeddings)} embeddings")
            
            # Add embeddings to products
            for i, product in enumerate(products):
                product['embedding'] = embeddings[i].tolist()
            
            return products
            
        except Exception as e:
            logger.error(f"❌ Failed to generate embeddings: {e}")
            raise
    
    def upsert_products(self, products: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Upsert products into ChromaDB collection.
        
        Args:
            products: List of product dictionaries with embeddings
            
        Returns:
            Dictionary with insertion/update statistics
        """
        if not products:
            return {'inserted': 0, 'updated': 0}
        
        logger.info(f"🔄 Upserting {len(products)} products to ChromaDB...")
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for product in products:
            if 'embedding' not in product:
                logger.warning(f"⚠️ Product {product.get('id', 'unknown')} missing embedding")
                continue
            
            # Use product ID or generate one
            product_id = product.get('id', str(uuid.uuid4()))
            ids.append(product_id)
            
            # Add embedding
            embeddings.append(product['embedding'])
            
            # Prepare metadata
            metadata = {
                'id': product.get('id', ''),
                'name': product.get('name', ''),
                'title': product.get('title', ''),
                'price_text': product.get('price_text', ''),
                'price_numeric': product.get('price_numeric', 0),
                'rating': product.get('rating', ''),
                'url': product.get('url', ''),
                'image_url': product.get('image_url', ''),
                'description': product.get('description', ''),
                'specs': product.get('specs', '{}'),
                'category': product.get('category', ''),
                'scraped_at': product.get('scraped_at', datetime.now().isoformat())
            }
            metadatas.append(metadata)
            
            # Create document text for ChromaDB
            doc_text = f"{product.get('name', '')} {product.get('description', '')}"
            documents.append(doc_text)
        
        try:
            # Upsert to ChromaDB
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            result = {'inserted': len(ids), 'updated': 0}
            logger.info(f"✅ Upserted {len(ids)} products to ChromaDB")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to upsert products: {e}")
            raise
    
    def semantic_search(self, query: str, k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Perform semantic search for products.
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of matching products with similarity scores
        """
        if not query.strip():
            return []
        
        logger.info(f"🔍 Searching for: '{query}' (k={k})")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Search in ChromaDB
            search_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            # Format results
            results = []
            if search_results['ids'] and len(search_results['ids']) > 0:
                for i in range(len(search_results['ids'][0])):
                    metadata = search_results['metadatas'][0][i]
                    distance = search_results['distances'][0][i]
                    
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance
                    
                    # Skip results below threshold
                    if similarity_score < score_threshold:
                        continue
                    
                    product = {
                        'id': metadata.get('id', ''),
                        'name': metadata.get('name', ''),
                        'title': metadata.get('title', ''),
                        'price_text': metadata.get('price_text', ''),
                        'price_numeric': metadata.get('price_numeric'),
                        'rating': metadata.get('rating', ''),
                        'url': metadata.get('url', ''),
                        'image_url': metadata.get('image_url', ''),
                        'description': metadata.get('description', ''),
                        'specs': metadata.get('specs', '{}'),
                        'category': metadata.get('category', ''),
                        'similarity_score': float(similarity_score),
                        'scraped_at': metadata.get('scraped_at', '')
                    }
                    
                    # Parse specs if it's a JSON string
                    try:
                        if isinstance(product['specs'], str):
                            product['specs'] = json.loads(product['specs'])
                    except:
                        product['specs'] = {}
                    
                    results.append(product)
            
            logger.info(f"✅ Found {len(results)} matching products")
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            raise
    
    def hybrid_search(self, query: str, k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Enhanced hybrid search with intelligent re-ranking.
        Combines semantic similarity with metadata-based boosting.
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of re-ranked products with boosted scores
        """
        if not query.strip():
            return []
        
        logger.info(f"🔍 Hybrid search for: '{query}' (k={k})")
        
        # Get more results than needed for re-ranking
        initial_k = min(k * 3, 30)
        results = self.semantic_search(query, k=initial_k, score_threshold=score_threshold)
        
        if not results:
            return []
        
        # Parse query for boosting hints
        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        
        # Extract query features
        has_price_constraint = any(word in query_lower for word in ['under', 'below', 'less than', 'budget', 'cheap', 'affordable'])
        has_ram_query = any(word in query_lower for word in ['ram', 'memory'])
        has_storage_query = any(word in query_lower for word in ['storage', 'gb', 'rom'])
        has_camera_query = any(word in query_lower for word in ['camera', 'photo', 'photography', 'mp'])
        has_battery_query = any(word in query_lower for word in ['battery', 'mah'])
        has_gaming_query = any(word in query_lower for word in ['gaming', 'game', 'performance'])
        
        # Extract brand preference
        brands = ['samsung', 'iphone', 'apple', 'xiaomi', 'redmi', 'oppo', 'tecno', 'infinix', 'nokia', 'huawei']
        query_brand = None
        for brand in brands:
            if brand in query_lower:
                query_brand = brand
                break
        
        # Extract price constraint
        price_limit = None
        try:
            import re
            price_match = re.search(r'under\s+(\d+)|below\s+(\d+)|less\s+than\s+(\d+)', query_lower)
            if price_match:
                price_limit = int(price_match.group(1) or price_match.group(2) or price_match.group(3))
        except:
            pass
        
        # Re-rank results with optimized boost factors
        for product in results:
            boost_score = 1.0
            product_name_lower = product['name'].lower()
            
            # 1. Exact brand match boost (strong signal) - Increased from 1.3 to 1.35
            if query_brand and query_brand in product_name_lower:
                boost_score *= 1.35
                logger.debug(f"Brand boost for {product['name'][:30]}")
            
            # 2. Keyword overlap boost
            product_tokens = set(product_name_lower.split())
            overlap = len(query_tokens & product_tokens)
            if overlap > 0:
                keyword_boost = 1.0 + (overlap * 0.05)  # +5% per matching keyword
                boost_score *= keyword_boost
            
            # 3. Price constraint boost - STRENGTHENED
            if has_price_constraint and price_limit and product.get('price_numeric'):
                product_price = product['price_numeric']
                if product_price <= price_limit:
                    # Boost products under the limit, more boost for lower prices
                    # Changed from 1.2 to 1.4 for stronger price match signal
                    price_ratio = product_price / price_limit
                    # Products well under budget get up to 40% boost
                    price_boost = 1.4 * (1.2 - price_ratio * 0.4)
                    boost_score *= price_boost
                elif product_price > price_limit:
                    # Stronger penalty for over-budget: 0.7 -> 0.55
                    boost_score *= 0.55
            
            # 4. Feature-specific boosts - STRENGTHENED
            if has_camera_query:
                # Increased from 1.15 to 1.3 for high-MP cameras
                if any(x in product_name_lower for x in ['64mp', '108mp', '200mp']):
                    boost_score *= 1.35  # Premium cameras
                elif any(x in product_name_lower for x in ['50mp', '48mp']):
                    boost_score *= 1.3   # Good cameras
                elif 'camera' in product_name_lower:
                    boost_score *= 1.15  # Camera mentioned
            
            if has_ram_query:
                # Increased RAM boost: 1.2 -> 1.25 for 8GB
                if '12gb' in product_name_lower or '12 gb' in product_name_lower:
                    boost_score *= 1.3   # Premium RAM
                elif '8gb' in product_name_lower or '8 gb' in product_name_lower:
                    boost_score *= 1.25  # High RAM
                elif '6gb' in product_name_lower or '6 gb' in product_name_lower:
                    boost_score *= 1.15  # Good RAM
                elif '4gb' in product_name_lower or '4 gb' in product_name_lower:
                    boost_score *= 1.05  # Minimum RAM
            
            if has_storage_query:
                # Increased storage boost
                if '512gb' in product_name_lower or '512 gb' in product_name_lower:
                    boost_score *= 1.25
                elif '256gb' in product_name_lower or '256 gb' in product_name_lower:
                    boost_score *= 1.2
                elif '128gb' in product_name_lower or '128 gb' in product_name_lower:
                    boost_score *= 1.1
            
            if has_battery_query:
                # Tiered battery boost
                if any(x in product_name_lower for x in ['6000mah', '7000mah']):
                    boost_score *= 1.25  # Large batteries
                elif '5000mah' in product_name_lower:
                    boost_score *= 1.15  # Standard large battery
            
            if has_gaming_query:
                # Gaming phones need high RAM - strengthened boost
                if '12gb' in product_name_lower:
                    boost_score *= 1.3
                elif '8gb' in product_name_lower:
                    boost_score *= 1.25
            
            # 5. Apply boost to similarity score
            product['original_score'] = product['similarity_score']
            product['boost_factor'] = boost_score
            product['similarity_score'] = product['similarity_score'] * boost_score
        
        # Sort by boosted score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Return top k
        final_results = results[:k]
        
        logger.info(f"✅ Hybrid search returned {len(final_results)} re-ranked products")
        return final_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            return {
                'total_products': count,
                'collection_name': self.collection_name,
                'distance_metric': 'cosine',
                'status': 'active'
            }
        except Exception as e:
            logger.error(f"❌ Failed to get collection stats: {e}")
            return {
                'total_products': 0,
                'collection_name': self.collection_name,
                'status': 'error',
                'error': str(e)
            }
    
    def run_pipeline(self, category_urls: List[str], max_products_per_category: int = 20) -> Dict[str, Any]:
        """
        Run the complete pipeline: scrape, embed, and store products.
        
        Args:
            category_urls: List of Jumia category URLs to scrape
            max_products_per_category: Maximum products per category
            
        Returns:
            Pipeline execution results
        """
        logger.info(f"🚀 Starting pipeline for {len(category_urls)} categories...")
        
        all_products = []
        
        # For now, use the smartphone scraper as a base
        # In a full implementation, you'd extend this to handle different categories
        try:
            max_pages = max(1, max_products_per_category // 20)

            for url in category_urls:
                if "ios-phones" in url:
                    # Fixed 5 pages for iOS as requested
                    logger.info("📱 Scraping iOS phones...")
                    products = self.scraper.scrape_ios_phones(max_pages=5)
                else:
                    logger.info("🤖 Scraping smartphones...")
                    products = self.scraper.scrape_smartphones(max_pages=max_pages)

                all_products.extend(products)            
        except Exception as e:
            logger.error(f"❌ Scraping failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_products': 0,
                'inserted': 0,
                'updated': 0
            }
        
        if not all_products:
            logger.warning("⚠️ No products scraped")
            return {
                'success': True,
                'total_products': 0,
                'inserted': 0,
                'updated': 0
            }
        
        try:
            # Deduplicate products by ID before generating embeddings / upsert
            if all_products:
                unique_by_id = {}
                for p in all_products:
                    pid = p.get('id')
                    if pid:
                        unique_by_id[pid] = p  # last one wins; that’s fine
                all_products = list(unique_by_id.values())
                logger.info(f"🧹 Deduplicated products by ID: {len(all_products)} unique products")
            # Generate embeddings
            products_with_embeddings = self.generate_embeddings(all_products)
            
            # Store in ChromaDB
            storage_result = self.upsert_products(products_with_embeddings)
            
            # Also store in SQLite for backup
            self.scraper.store_products(all_products)
            
            result = {
                'success': True,
                'total_products': len(all_products),
                'inserted': storage_result['inserted'],
                'updated': storage_result['updated'],
                'collection_stats': self.get_collection_stats()
            }
            
            logger.info(f"✅ Pipeline completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_products': len(all_products),
                'inserted': 0,
                'updated': 0
            }
    
    @property
    def collection(self):
        """Compatibility property for existing code."""
        return self._collection
    
    @collection.setter
    def collection(self, value):
        """Setter for collection property."""
        self._collection = value


def main():
    """Test the pipeline."""
    print("🧪 Testing ChromaDB Product Pipeline...")
    
    # Initialize pipeline
    pipeline = JumiaProductPipeline()
    
    # Test scraping and storage
    # Increase max_products_per_category to get more products (10 → 50 or 100)
    result = pipeline.run_pipeline(
    category_urls=[
        "https://www.jumia.co.ke/smartphones/",
        "https://www.jumia.co.ke/ios-phones/",
    ],
    max_products_per_category=50
)
    
    print(f"Pipeline result: {result}")
    
    # Test search
    if result['total_products'] > 0:
        print("\n🔍 Testing semantic search...")
        search_queries = [
            "Samsung smartphone with good camera",
            "iPhone with large storage",
            "Android phone under 20000",
            "smartphone with long battery life"
        ]
        
        for query in search_queries:
            print(f"\nSearching: '{query}'")
            results = pipeline.hybrid_search(query, k=3)
            
            for i, product in enumerate(results, 1):
                print(f"  {i}. {product['name']} (Score: {product['similarity_score']:.3f})")
                print(f"     Price: {product['price_text']}")
                print(f"     URL: {product['url']}")


if __name__ == "__main__":
    main()

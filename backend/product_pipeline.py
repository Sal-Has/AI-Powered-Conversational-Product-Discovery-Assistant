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
            # Calculate total pages based on products per category
            max_pages = max(1, max_products_per_category // 20)  # ~20 products per page
            
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
    result = pipeline.run_pipeline(
        category_urls=["https://www.jumia.co.ke/smartphones/"],
        max_products_per_category=10
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
            results = pipeline.semantic_search(query, k=3)
            
            for i, product in enumerate(results, 1):
                print(f"  {i}. {product['name']} (Score: {product['similarity_score']:.3f})")
                print(f"     Price: {product['price_text']}")
                print(f"     URL: {product['url']}")


if __name__ == "__main__":
    main()

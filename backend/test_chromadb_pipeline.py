#!/usr/bin/env python3
"""
Test suite for ChromaDB Product Pipeline
Tests all functionality of the JumiaProductPipeline with ChromaDB backend.
"""

import os
import sys
import tempfile
import shutil
import json
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import time
import gc

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from product_pipeline import JumiaProductPipeline


class TestChromaDBPipeline(unittest.TestCase):
    """Test cases for ChromaDB Product Pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for ChromaDB
        self.temp_dir = tempfile.mkdtemp()
        self.chroma_persist_directory = os.path.join(self.temp_dir, "test_chroma_db")
        self.pipeline = None
        
        # Sample product data
        self.sample_products = [
            {
                'id': 'test_product_1',
                'name': 'Samsung Galaxy S21',
                'title': 'Samsung Galaxy S21 128GB',
                'price_text': 'KSh 45,000',
                'price_numeric': 45000,
                'rating': '4.5/5',
                'url': 'https://www.jumia.co.ke/samsung-galaxy-s21',
                'image_url': 'https://example.com/image1.jpg',
                'description': 'Latest Samsung smartphone with excellent camera',
                'specs': '{"RAM": "8GB", "Storage": "128GB", "Camera": "64MP"}',
                'category': 'smartphones',
                'scraped_at': datetime.now().isoformat()
            },
            {
                'id': 'test_product_2',
                'name': 'iPhone 13',
                'title': 'Apple iPhone 13 256GB',
                'price_text': 'KSh 85,000',
                'price_numeric': 85000,
                'rating': '4.8/5',
                'url': 'https://www.jumia.co.ke/iphone-13',
                'image_url': 'https://example.com/image2.jpg',
                'description': 'Premium Apple smartphone with A15 Bionic chip',
                'specs': '{"RAM": "6GB", "Storage": "256GB", "Camera": "12MP"}',
                'category': 'smartphones',
                'scraped_at': datetime.now().isoformat()
            }
        ]
    
    def tearDown(self):
        """Clean up test fixtures with proper ChromaDB cleanup."""
        # Properly close ChromaDB connections
        if self.pipeline:
            try:
                # Close ChromaDB client connections
                if hasattr(self.pipeline, 'chroma_client'):
                    # Reset the client to close connections
                    self.pipeline.chroma_client.reset()
                    del self.pipeline.chroma_client
                del self.pipeline
            except:
                pass
        
        # Force garbage collection to release file handles
        gc.collect()
        
        # Wait a bit for Windows to release file handles
        time.sleep(0.1)
        
        # Retry cleanup with error handling
        if os.path.exists(self.temp_dir):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(self.temp_dir)
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(0.2)  # Wait longer between retries
                        gc.collect()  # Force another garbage collection
                    else:
                        # On final attempt, try to remove what we can
                        try:
                            self._force_remove_tree(self.temp_dir)
                        except:
                            pass  # Ignore final cleanup errors
    
    def _force_remove_tree(self, path):
        """Force remove directory tree on Windows."""
        def handle_remove_readonly(func, path, exc):
            """Handle readonly files on Windows."""
            try:
                os.chmod(path, 0o777)
                func(path)
            except:
                pass
        
        shutil.rmtree(path, onerror=handle_remove_readonly)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization with ChromaDB."""
        self.pipeline = JumiaProductPipeline(
            chroma_persist_directory=self.chroma_persist_directory,
            collection_name="test_collection"
        )
        
        self.assertIsNotNone(self.pipeline.chroma_client)
        self.assertIsNotNone(self.pipeline.embedding_model)
        self.assertIsNotNone(self.pipeline.collection)
        self.assertEqual(self.pipeline.collection_name, "test_collection")
    
    def test_collection_creation(self):
        """Test ChromaDB collection creation."""
        self.pipeline = JumiaProductPipeline(
            chroma_persist_directory=self.chroma_persist_directory,
            collection_name="test_collection"
        )
        
        # Collection should be created and accessible
        self.assertIsNotNone(self.pipeline.collection)
        self.assertEqual(self.pipeline.collection.name, "test_collection")
    
    def test_embedding_generation(self):
        """Test embedding generation for products."""
        self.pipeline = JumiaProductPipeline(
            chroma_persist_directory=self.chroma_persist_directory
        )
        
        # Generate embeddings
        products_with_embeddings = self.pipeline.generate_embeddings(self.sample_products)
        
        self.assertEqual(len(products_with_embeddings), 2)
        for product in products_with_embeddings:
            self.assertIn('embedding', product)
            self.assertIsInstance(product['embedding'], list)
            self.assertGreater(len(product['embedding']), 0)
    
    def test_product_upsert(self):
        """Test upserting products to ChromaDB."""
        self.pipeline = JumiaProductPipeline(
            chroma_persist_directory=self.chroma_persist_directory
        )
        
        # Generate embeddings first
        products_with_embeddings = self.pipeline.generate_embeddings(self.sample_products)
        
        # Upsert products
        result = self.pipeline.upsert_products(products_with_embeddings)
        
        self.assertIn('inserted', result)
        self.assertIn('updated', result)
        self.assertEqual(result['inserted'], 2)
        
        # Verify products are in collection
        count = self.pipeline.collection.count()
        self.assertEqual(count, 2)
    
    def test_semantic_search(self):
        """Test semantic search functionality."""
        self.pipeline = JumiaProductPipeline(
            chroma_persist_directory=self.chroma_persist_directory
        )
        
        # Add products first
        products_with_embeddings = self.pipeline.generate_embeddings(self.sample_products)
        self.pipeline.upsert_products(products_with_embeddings)
        
        # Test search
        results = self.pipeline.semantic_search("Samsung smartphone", k=5)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Check result structure
        for result in results:
            self.assertIn('name', result)
            self.assertIn('similarity_score', result)
            self.assertIn('price_text', result)
            self.assertIsInstance(result['similarity_score'], float)
    
    def test_search_with_threshold(self):
        """Test semantic search with similarity threshold."""
        self.pipeline = JumiaProductPipeline(
            chroma_persist_directory=self.chroma_persist_directory
        )
        
        # Add products first
        products_with_embeddings = self.pipeline.generate_embeddings(self.sample_products)
        self.pipeline.upsert_products(products_with_embeddings)
        
        # Test search with high threshold
        results = self.pipeline.semantic_search("Samsung smartphone", k=5, score_threshold=0.9)
        
        # Should return fewer or no results due to high threshold
        self.assertIsInstance(results, list)
        for result in results:
            self.assertGreaterEqual(result['similarity_score'], 0.9)
    
    def test_collection_stats(self):
        """Test collection statistics retrieval."""
        self.pipeline = JumiaProductPipeline(
            chroma_persist_directory=self.chroma_persist_directory
        )
        
        # Initially empty
        stats = self.pipeline.get_collection_stats()
        self.assertEqual(stats['total_products'], 0)
        self.assertEqual(stats['status'], 'active')
        
        # Add products
        products_with_embeddings = self.pipeline.generate_embeddings(self.sample_products)
        self.pipeline.upsert_products(products_with_embeddings)
        
        # Check updated stats
        stats = self.pipeline.get_collection_stats()
        self.assertEqual(stats['total_products'], 2)
        self.assertEqual(stats['collection_name'], 'jumia_products')
        self.assertEqual(stats['distance_metric'], 'cosine')
    
    def test_empty_product_list(self):
        """Test handling of empty product lists."""
        self.pipeline = JumiaProductPipeline(
            chroma_persist_directory=self.chroma_persist_directory
        )
        
        # Test empty embedding generation
        result = self.pipeline.generate_embeddings([])
        self.assertEqual(result, [])
        
        # Test empty upsert
        result = self.pipeline.upsert_products([])
        self.assertEqual(result, {'inserted': 0, 'updated': 0})
    
    def test_empty_search_query(self):
        """Test handling of empty search queries."""
        self.pipeline = JumiaProductPipeline(
            chroma_persist_directory=self.chroma_persist_directory
        )
        
        # Test empty query
        results = self.pipeline.semantic_search("", k=5)
        self.assertEqual(results, [])
        
        # Test whitespace-only query
        results = self.pipeline.semantic_search("   ", k=5)
        self.assertEqual(results, [])
    
    def test_specs_parsing(self):
        """Test JSON specs parsing in search results."""
        self.pipeline = JumiaProductPipeline(
            chroma_persist_directory=self.chroma_persist_directory
        )
        
        # Add products
        products_with_embeddings = self.pipeline.generate_embeddings(self.sample_products)
        self.pipeline.upsert_products(products_with_embeddings)
        
        # Search and check specs parsing
        results = self.pipeline.semantic_search("Samsung", k=1)
        
        self.assertGreater(len(results), 0)
        result = results[0]
        self.assertIn('specs', result)
        self.assertIsInstance(result['specs'], dict)
        self.assertIn('RAM', result['specs'])
    
    @patch('product_pipeline.BasicJumiaScraper')
    def test_run_pipeline_success(self, mock_scraper_class):
        """Test successful pipeline execution."""
        # Mock scraper
        mock_scraper = Mock()
        mock_scraper.scrape_smartphones.return_value = self.sample_products
        mock_scraper.store_products.return_value = None
        mock_scraper_class.return_value = mock_scraper
        
        self.pipeline = JumiaProductPipeline(
            chroma_persist_directory=self.chroma_persist_directory
        )
        
        # Run pipeline
        result = self.pipeline.run_pipeline(
            category_urls=["https://www.jumia.co.ke/smartphones/"],
            max_products_per_category=10
        )
        
        self.assertTrue(result['success'])
        self.assertEqual(result['total_products'], 2)
        self.assertEqual(result['inserted'], 2)
        self.assertIn('collection_stats', result)
    
    @patch('product_pipeline.BasicJumiaScraper')
    def test_run_pipeline_scraping_failure(self, mock_scraper_class):
        """Test pipeline handling of scraping failures."""
        # Mock scraper to raise exception
        mock_scraper = Mock()
        mock_scraper.scrape_smartphones.side_effect = Exception("Scraping failed")
        mock_scraper_class.return_value = mock_scraper
        
        self.pipeline = JumiaProductPipeline(
            chroma_persist_directory=self.chroma_persist_directory
        )
        
        # Run pipeline
        result = self.pipeline.run_pipeline(
            category_urls=["https://www.jumia.co.ke/smartphones/"],
            max_products_per_category=10
        )
        
        self.assertFalse(result['success'])
        self.assertIn('error', result)
        self.assertEqual(result['total_products'], 0)
    
    @patch('product_pipeline.BasicJumiaScraper')
    def test_run_pipeline_no_products(self, mock_scraper_class):
        """Test pipeline handling when no products are scraped."""
        # Mock scraper to return empty list
        mock_scraper = Mock()
        mock_scraper.scrape_smartphones.return_value = []
        mock_scraper_class.return_value = mock_scraper
        
        self.pipeline = JumiaProductPipeline(
            chroma_persist_directory=self.chroma_persist_directory
        )
        
        # Run pipeline
        result = self.pipeline.run_pipeline(
            category_urls=["https://www.jumia.co.ke/smartphones/"],
            max_products_per_category=10
        )
        
        self.assertTrue(result['success'])
        self.assertEqual(result['total_products'], 0)
        self.assertEqual(result['inserted'], 0)
    
    def test_compatibility_property(self):
        """Test collection compatibility property."""
        self.pipeline = JumiaProductPipeline(
            chroma_persist_directory=self.chroma_persist_directory
        )
        
        # Test that collection property works
        self.assertIsNotNone(self.pipeline.collection)
        
        # Test count method
        count = self.pipeline.collection.count()
        self.assertIsInstance(count, int)
        self.assertEqual(count, 0)


def run_comprehensive_test():
    """Run comprehensive test suite with detailed output."""
    print("🧪 Running ChromaDB Pipeline Test Suite...")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestChromaDBPipeline)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'✅ PASSED' if success else '❌ FAILED'}")
    
    return success


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)

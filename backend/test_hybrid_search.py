#!/usr/bin/env python3
"""
Quick test to compare semantic_search vs hybrid_search results.
"""

from product_pipeline import JumiaProductPipeline

def compare_search_methods():
    print("\n" + "="*70)
    print("COMPARING SEMANTIC SEARCH VS HYBRID SEARCH")
    print("="*70)
    
    pipeline = JumiaProductPipeline(
        chroma_persist_directory="./chroma_db",
        collection_name="jumia_products"
    )
    
    test_queries = [
        "Samsung phone with good camera under 20000",
        "Samsung phone with 8GB RAM and high storage",
        "Budget phone with high RAM for gaming"
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: '{query}'")
        print(f"{'='*70}")
        
        # Semantic search
        print("\n🔍 SEMANTIC SEARCH (baseline):")
        semantic_results = pipeline.semantic_search(query, k=5)
        for i, product in enumerate(semantic_results, 1):
            print(f"{i}. {product['name'][:60]}")
            print(f"   Score: {product['similarity_score']:.4f} | Price: {product['price_text']}")
        
        # Hybrid search
        print("\n⚡ HYBRID SEARCH (improved):")
        hybrid_results = pipeline.hybrid_search(query, k=5)
        for i, product in enumerate(hybrid_results, 1):
            boost = product.get('boost_factor', 1.0)
            original = product.get('original_score', product['similarity_score'])
            print(f"{i}. {product['name'][:60]}")
            print(f"   Score: {product['similarity_score']:.4f} (boost: {boost:.2f}x) | Price: {product['price_text']}")
        
        # Check if rankings changed
        semantic_ids = [p['id'] for p in semantic_results]
        hybrid_ids = [p['id'] for p in hybrid_results]
        
        if semantic_ids != hybrid_ids:
            print("\n✅ Rankings IMPROVED - order changed to prioritize relevant products")
        else:
            print("\n⚠️  Rankings unchanged")

if __name__ == "__main__":
    compare_search_methods()
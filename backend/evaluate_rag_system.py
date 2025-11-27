#!/usr/bin/env python3
"""
Complete RAG Evaluation Script
Evaluates both retrieval and generation quality using:
1. ROUGE score - Text generation quality
2. BERTScore - Semantic similarity
3. MRR (Mean Reciprocal Rank) - Retrieval quality
4. Precision@K - Retrieval accuracy

Usage:
    python evaluate_rag_system.py
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Set
import numpy as np
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# RAG components
from product_pipeline import JumiaProductPipeline
from llm_service import LLMService
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Evaluation metrics
try:
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    import nltk
    nltk.download('punkt', quiet=True)
    METRICS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: {e}")
    print("Please install: pip install rouge-score bert-score nltk")
    METRICS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Comprehensive RAG system evaluation."""
    
    def __init__(self, pipeline: JumiaProductPipeline, llm_service: LLMService):
        self.pipeline = pipeline
        self.llm_service = llm_service
        
        if METRICS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
    
    # ========== RETRIEVAL METRICS ==========
    
    def precision_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Precision@K: % relevant in top K."""
        if k == 0 or not relevant:
            return 0.0
        top_k = retrieved[:k]
        relevant_retrieved = sum(1 for item in top_k if item in relevant)
        return relevant_retrieved / k
    
    def mean_reciprocal_rank(self, retrieved: List[str], relevant: Set[str]) -> float:
        """MRR: 1/rank of first relevant item."""
        for i, item in enumerate(retrieved, 1):
            if item in relevant:
                return 1.0 / i
        return 0.0
    
    def recall_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Recall@K: % of relevant items found in top K."""
        if not relevant:
            return 0.0
        top_k = retrieved[:k]
        relevant_retrieved = sum(1 for item in top_k if item in relevant)
        return relevant_retrieved / len(relevant)
    
    def evaluate_retrieval(self, query: str, relevant_ids: Set[str], 
                          k_values: List[int] = [1, 3, 5, 10]) -> tuple:
        """Evaluate retrieval for a query using hybrid search."""
        # Use hybrid_search for better ranking
        results = self.pipeline.hybrid_search(query, k=max(k_values))
        retrieved_ids = [product['id'] for product in results]
        
        metrics = {
            'query': query,
            'num_retrieved': len(retrieved_ids),
            'num_relevant': len(relevant_ids),
            'mrr': self.mean_reciprocal_rank(retrieved_ids, relevant_ids)
        }
        
        for k in k_values:
            metrics[f'precision@{k}'] = self.precision_at_k(retrieved_ids, relevant_ids, k)
            metrics[f'recall@{k}'] = self.recall_at_k(retrieved_ids, relevant_ids, k)
        
        if results:
            metrics['avg_similarity'] = np.mean([r['similarity_score'] for r in results])
        else:
            metrics['avg_similarity'] = 0.0
        
        return metrics, results
    
    # ========== GENERATION METRICS ==========
    
    def calculate_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        if not METRICS_AVAILABLE:
            return {}
        
        scores = self.rouge_scorer.score(reference, hypothesis)
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge1_p': scores['rouge1'].precision,
            'rouge1_r': scores['rouge1'].recall,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rouge2_p': scores['rouge2'].precision,
            'rouge2_r': scores['rouge2'].recall,
            'rougeL_f': scores['rougeL'].fmeasure,
            'rougeL_p': scores['rougeL'].precision,
            'rougeL_r': scores['rougeL'].recall,
        }
    
    def calculate_bertscore(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """Calculate BERTScore."""
        if not METRICS_AVAILABLE:
            return {}
        
        try:
            P, R, F1 = bert_score(
                hypotheses, references, 
                lang='en', verbose=False, device='cpu'
            )
            return {
                'bertscore_precision': P.mean().item(),
                'bertscore_recall': R.mean().item(),
                'bertscore_f1': F1.mean().item()
            }
        except Exception as e:
            logger.error(f"BERTScore error: {e}")
            return {'bertscore_precision': 0, 'bertscore_recall': 0, 'bertscore_f1': 0}
    
    def generate_rag_response(self, query: str, products: List[Dict[str, Any]]) -> str:
        """Generate response using LLMService."""
        product_info = ""
        for i, product in enumerate(products[:5], 1):
            product_info += f"\n{i}. {product.get('name', 'Unknown')}\n"
            product_info += f"   Price: {product.get('price_text', 'N/A')}\n"
            product_info += f"   Rating: {product.get('rating', 'N/A')}\n"
            
            # Add specs
            specs = product.get('specs', {})
            if isinstance(specs, str):
                try:
                    specs = json.loads(specs)
                except:
                    specs = {}
            
            if specs:
                spec_items = list(specs.items())[:3]
                spec_text = ", ".join([f"{k}: {v}" for k, v in spec_items])
                product_info += f"   Specs: {spec_text}\n"
        
        system_prompt = """You are a helpful e-commerce assistant. Based on the user's query and the product information provided, give a concise, informative response recommending suitable products."""
        
        user_prompt = f"""User Query: {query}

Available Products:
{product_info}

Please provide a helpful response recommending the best products that match the user's needs."""
        
        try:
            result = self.llm_service.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            return result['text']
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"Error generating response: {e}"
    
    def evaluate_generation(self, generated: str, reference: str) -> Dict[str, Any]:
        """Evaluate generation quality."""
        metrics = {}
        
        # ROUGE scores
        rouge_scores = self.calculate_rouge(reference, generated)
        metrics.update(rouge_scores)
        
        # BERTScore
        bert_scores = self.calculate_bertscore([reference], [generated])
        metrics.update(bert_scores)
        
        return metrics
    
    # ========== END-TO-END EVALUATION ==========
    
    def evaluate_rag_query(self, test_case: Dict[str, Any], 
                          k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """Evaluate both retrieval and generation for a query."""
        query = test_case['query']
        relevant_ids = test_case['relevant_ids']
        reference_response = test_case['reference_response']
        
        logger.info(f"\n📝 Evaluating: '{query}'")
        
        # 1. Evaluate Retrieval
        retrieval_metrics, retrieved_products = self.evaluate_retrieval(
            query, relevant_ids, k_values
        )
        
        # 2. Generate Response
        generated_response = self.generate_rag_response(query, retrieved_products)
        
        # 3. Evaluate Generation
        generation_metrics = self.evaluate_generation(generated_response, reference_response)
        
        return {
            'query': query,
            'retrieval': retrieval_metrics,
            'generation': generation_metrics,
            'generated_response': generated_response,
            'reference_response': reference_response,
            'num_products': len(retrieved_products)
        }
    
    def evaluate_test_set(self, test_cases: List[Dict[str, Any]], 
                         k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """Evaluate multiple test cases."""
        logger.info(f"\n🚀 Evaluating {len(test_cases)} test cases...\n")
        
        all_results = []
        
        for i, test_case in enumerate(test_cases, 1):
            try:
                print(f"\n{'='*70}")
                print(f"Test Case {i}/{len(test_cases)}")
                print(f"{'='*70}")
                
                result = self.evaluate_rag_query(test_case, k_values)
                all_results.append(result)
                
                # Print summary
                ret = result['retrieval']
                gen = result['generation']
                print(f"\n📊 Query: '{test_case['query'][:60]}...'")
                print(f"\n🔍 Retrieval Metrics:")
                print(f"   MRR: {ret['mrr']:.4f}")
                print(f"   P@1: {ret.get('precision@1', 0):.4f} | P@5: {ret.get('precision@5', 0):.4f}")
                print(f"   R@1: {ret.get('recall@1', 0):.4f} | R@5: {ret.get('recall@5', 0):.4f}")
                
                if METRICS_AVAILABLE and gen:
                    print(f"\n📝 Generation Metrics:")
                    print(f"   ROUGE-1: {gen.get('rouge1_f', 0):.4f}")
                    print(f"   ROUGE-L: {gen.get('rougeL_f', 0):.4f}")
                    print(f"   BERTScore F1: {gen.get('bertscore_f1', 0):.4f}")
                
                print(f"\n💬 Generated Response:")
                print(f"   {result['generated_response'][:200]}...")
                
            except Exception as e:
                logger.error(f"Error evaluating '{test_case['query']}': {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate averages
        avg_metrics = self._calculate_averages(all_results, k_values)
        
        return {
            'summary': avg_metrics,
            'detailed_results': all_results,
            'total_cases': len(test_cases),
            'successful': len(all_results)
        }
    
    def _calculate_averages(self, results: List[Dict], k_values: List[int]) -> Dict:
        """Calculate average metrics."""
        if not results:
            return {}
        
        avg = {
            'retrieval': {
                'avg_mrr': np.mean([r['retrieval']['mrr'] for r in results])
            },
            'generation': {}
        }
        
        # Retrieval averages
        for k in k_values:
            avg['retrieval'][f'avg_precision@{k}'] = np.mean(
                [r['retrieval'][f'precision@{k}'] for r in results]
            )
            avg['retrieval'][f'avg_recall@{k}'] = np.mean(
                [r['retrieval'][f'recall@{k}'] for r in results]
            )
        
        # Generation averages
        if METRICS_AVAILABLE and results[0]['generation']:
            for metric in ['rouge1_f', 'rouge2_f', 'rougeL_f', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1']:
                values = [r['generation'].get(metric, 0) for r in results if r['generation'].get(metric) is not None]
                if values:
                    avg['generation'][f'avg_{metric}'] = np.mean(values)
        
        return avg
    
    def print_summary(self, results: Dict[str, Any]):
        """Print formatted summary."""
        summary = results['summary']
        
        print("\n" + "="*70)
        print("📊 RAG SYSTEM EVALUATION SUMMARY")
        print("="*70)
        print(f"\nTest Cases: {results['successful']}/{results['total_cases']} successful")
        
        print("\n" + "-"*70)
        print("🔍 RETRIEVAL METRICS (Average)")
        print("-"*70)
        ret = summary['retrieval']
        print(f"Mean Reciprocal Rank (MRR): {ret['avg_mrr']:.4f}")
        print()
        
        for k in [1, 3, 5, 10]:
            if f'avg_precision@{k}' in ret:
                prec = ret[f'avg_precision@{k}']
                rec = ret[f'avg_recall@{k}']
                print(f"K={k:2d} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        
        if METRICS_AVAILABLE and summary['generation']:
            print("\n" + "-"*70)
            print("📝 GENERATION METRICS (Average)")
            print("-"*70)
            gen = summary['generation']
            print(f"\nROUGE Scores:")
            print(f"  ROUGE-1 F1: {gen.get('avg_rouge1_f', 0):.4f}")
            print(f"  ROUGE-2 F1: {gen.get('avg_rouge2_f', 0):.4f}")
            print(f"  ROUGE-L F1: {gen.get('avg_rougeL_f', 0):.4f}")
            
            print(f"\nBERTScore:")
            print(f"  Precision:  {gen.get('avg_bertscore_precision', 0):.4f}")
            print(f"  Recall:     {gen.get('avg_bertscore_recall', 0):.4f}")
            print(f"  F1:         {gen.get('avg_bertscore_f1', 0):.4f}")
        
        print("\n" + "="*70)


def create_test_dataset() -> List[Dict[str, Any]]:
    """
    Create test dataset with queries, relevant products, and reference answers.
    Updated with actual product IDs from Jumia Kenya database (Nov 22, 2025).
    """
    return [
        {
            'query': 'Samsung phone with good camera under 20000',
            'relevant_ids': {
                '318518c3aff7c8a4494e689f86eb3a25',  # Samsung GALAXY A16, 4GB+128GB, 50MP - KSh 16,490
                '66cbdfc55838603315728d01e54fbce5',  # Samsung Galaxy A07, 4GB+128GB - KSh 12,520
                'f5f28ed3d886ad53f33e6822fd58e94d',  # Samsung Galaxy A17 5G, 4GB+128GB - KSh 18,999
                '50ef6375e0cef271c2cf14e7b8d68db9',  # Samsung GALAXY A06, 4GB+64GB - KSh 10,100
            },
            'reference_response': 'I recommend the Samsung Galaxy A16 at KES 16,490. It features a 50MP camera with excellent photo quality, 4GB RAM, and 128GB storage, perfect for photography within your budget. For a more affordable option, the Samsung Galaxy A06 at KES 10,100 also offers good camera quality and value.'
        },
        {
            'query': 'iPhone 256GB storage',
            'relevant_ids': {
                '8e0bfad15aeea1489418786735c997fa',  # Apple IPHONE 13 PRO 256GB - KSh 63,500
                'a9c1c1758fcdcce50bde591242ebb2b6',  # Apple IPHONE 13 PRO MAX 256GB - KSh 75,999
                '0ac055f5a73500f65550c519acef8722',  # Apple IPHONE 13 PRO 256GB - KSh 65,999
                'acf1149d1851d5c19eff8cd899b59948',  # Apple IPHONE 12 PRO 256GB - KSh 54,999
            },
            'reference_response': 'For an iPhone with 256GB storage, I recommend the iPhone 13 Pro at KES 63,500. It offers excellent performance, ProMotion display, advanced camera system, and ample storage for all your photos, videos, and apps. For a more premium option, the iPhone 13 Pro Max at KES 75,999 provides a larger screen and better battery life.'
        },
        {
            'query': 'Budget phone with high RAM for gaming',
            'relevant_ids': {
                '185c53d0f61a0e9bdb42406f9a7b3f02',  # Oukitel C2, 16GB RAM+128GB ROM - KSh 11,122
                '318518c3aff7c8a4494e689f86eb3a25',  # Samsung GALAXY A16, 4GB RAM+128GB - KSh 16,490
                '42d7d802e31d39f2a09d2096992197db',  # Oukitel C57 Pro, 4GB RAM 128GB ROM - KSh 10,317
                'f5f28ed3d886ad53f33e6822fd58e94d',  # Samsung Galaxy A17 5G, 4GB RAM+128GB - KSh 18,999
            },
            'reference_response': 'For gaming on a budget, I recommend the Oukitel C2 with 16GB RAM at KES 11,122. It offers excellent multitasking and gaming performance with 128GB storage. For a more reliable brand, the Samsung Galaxy A16 at KES 16,490 with 4GB RAM provides smooth gaming with a 5000mAh battery for long gaming sessions.'
        },
    ]


def get_product_ids_helper():
    """Helper function to get product IDs for test cases."""
    print("\n🔍 Getting product IDs from database...\n")
    
    pipeline = JumiaProductPipeline(
        chroma_persist_directory="./chroma_db",
        collection_name="jumia_products"
    )
    
    queries = [
        ("Samsung camera phone", "Samsung Phones"),
        ("iPhone 256GB storage", "iPhones"),
        ("budget gaming phone high RAM", "Gaming Phones")
    ]
    
    for query, category in queries:
        print(f"\n{'='*60}")
        print(f"{category}: '{query}'")
        print(f"{'='*60}")
        results = pipeline.semantic_search(query, k=5)
        
        if results:
            print(f"\nFound {len(results)} products:")
            for i, r in enumerate(results, 1):
                print(f"\n{i}. ID: {r['id']}")
                print(f"   Name: {r['name']}")
                print(f"   Price: {r['price_text']}")
                print(f"   Similarity: {r['similarity_score']:.3f}")
        else:
            print("No results found")
    
    print("\n" + "="*60)
    print("Copy the relevant IDs and add them to create_test_dataset()")
    print("="*60 + "\n")


def main():
    """Main evaluation function."""
    print("\n🚀 RAG SYSTEM EVALUATION")
    print("="*70)
    print("\nThis script evaluates:")
    print("  📍 RETRIEVAL: MRR, Precision@K, Recall@K")
    print("  📝 GENERATION: ROUGE scores, BERTScore")
    print()
    
    if not METRICS_AVAILABLE:
        print("⚠️  Install metrics: pip install rouge-score bert-score nltk")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Check for helper mode
    if len(sys.argv) > 1 and sys.argv[1] == '--get-ids':
        get_product_ids_helper()
        return
    
    # Initialize
    logger.info("Initializing RAG components...")
    
    pipeline = JumiaProductPipeline(
        chroma_persist_directory="./chroma_db",
        collection_name="jumia_products"
    )
    
    # Check collection
    stats = pipeline.get_collection_stats()
    print(f"\n✅ Collection: {stats['total_products']} products")
    
    if stats['total_products'] == 0:
        print("\n❌ No products in database. Run scraper first.")
        return
    
    # Initialize LLM Service (will use provider from .env LLM_PROVIDER)
    try:
        llm_service = LLMService()
        provider_info = llm_service.get_provider_info()
        print(f"✅ LLM Provider: {provider_info['provider']} / {provider_info['model']}")
    except Exception as e:
        print(f"\n❌ Failed to initialize LLM service: {e}")
        print("Make sure OPENAI_API_KEY or HF_TOKEN is set in .env")
        return
    
    # Initialize evaluator
    evaluator = RAGEvaluator(pipeline, llm_service)
    
    # Load test cases
    test_cases = create_test_dataset()
    
    # Check if test cases have ground truth
    has_ground_truth = any(len(tc['relevant_ids']) > 0 for tc in test_cases)
    
    if not has_ground_truth:
        print("\n⚠️  No ground truth product IDs in test cases")
        print("Run with --get-ids flag to get product IDs:")
        print("  python evaluate_rag_system.py --get-ids")
        print("\nThen update create_test_dataset() with the IDs")
        return
    
    # Confirm before running
    print(f"\nReady to evaluate {len(test_cases)} test cases")
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Run evaluation
    print("\n" + "="*70)
    print("STARTING EVALUATION")
    print("="*70)
    
    results = evaluator.evaluate_test_set(test_cases, k_values=[1, 3, 5, 10])
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"rag_evaluation_{timestamp}.json"
    
    # Prepare JSON-serializable results
    json_results = {
        'timestamp': timestamp,
        'summary': results['summary'],
        'total_cases': results['total_cases'],
        'successful_cases': results['successful'],
        'test_cases': []
    }
    
    for result in results['detailed_results']:
        json_results['test_cases'].append({
            'query': result['query'],
            'retrieval_metrics': result['retrieval'],
            'generation_metrics': result['generation'],
            'generated_response': result['generated_response'],
            'reference_response': result['reference_response']
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}\n")


if __name__ == "__main__":
    main()
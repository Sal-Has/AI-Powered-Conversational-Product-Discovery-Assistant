            rag_pipeline = RAGPipeline(
                chroma_persist_directory="./chroma_db",
                collection_name="jumia_products",
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                llm_model="gpt-3.5"
            )
{{ ... }}
     return jsonify({
         'success': True,
         'available_models': {
             'llm_models': ['gpt-3.5'],
             'embedding_models': ['multi-qa-MiniLM-L6-cos-v1'],
             'current_llm': 'gpt-3.5',
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

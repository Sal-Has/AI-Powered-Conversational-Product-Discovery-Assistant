# run_scheduled_scrape.py

from product_pipeline import JumiaProductPipeline

def main():
    # 1) Configure which categories and how deep to scrape
    category_urls = [
        "https://www.jumia.co.ke/smartphones/",
        "https://www.jumia.co.ke/ios-phones/",
    ]
    max_products_per_category = 25  # or whatever you want

    # 2) Initialize pipeline (uses same ChromaDB as the app)
    pipeline = JumiaProductPipeline(
        chroma_persist_directory="./chroma_db",
        collection_name="jumia_products",
    )

    # 3) Run the full scrape + embed + index pipeline
    result = pipeline.run_pipeline(category_urls, max_products_per_category)

    print("=== Scheduled scrape finished ===")
    print(result)

if __name__ == "__main__":
    main()
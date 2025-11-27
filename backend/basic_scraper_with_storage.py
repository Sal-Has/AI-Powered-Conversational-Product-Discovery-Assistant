from bs4 import BeautifulSoup
import requests
import time
import pandas as pd
import re
import json
import sqlite3
from datetime import datetime
import hashlib

class BasicJumiaScraper:
    """
    Basic Jumia scraper that stores products in SQLite database.
    No complex dependencies - works with basic packages.
    """
    
    def __init__(self, db_path="jumia_products.db"):
        """Initialize the scraper with SQLite database."""
        self.headers = {
            "User-Agent": "JuaFindBot/1.0 (contact: devbot@test.dev)"
        }
        self.official_url = "https://www.jumia.co.ke"
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with products table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id TEXT PRIMARY KEY,
                name TEXT,
                title TEXT,
                price_text TEXT,
                price_numeric REAL,
                rating TEXT,
                url TEXT,
                image_url TEXT,
                description TEXT,
                specs TEXT,
                category TEXT,
                scraped_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✅ Database initialized: {self.db_path}")
    
    def _generate_product_id(self, url: str) -> str:
        """Generate a unique product ID from URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def clean_text(self, text):
        """Clean up whitespace, newlines, non-breaking spaces."""
        if not text:
            return ""
        return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()

    def extract_product_description(self, container):
        """Extracts clean text + structured specs from product description container."""
        if not container:
            return {"text": "", "specs": {}}

        # Remove unwanted tags (images, scripts, styles)
        for tag in container.find_all(["img", "script", "style"]):
            tag.decompose()

        details = []
        specs_dict = {}

        # ---- Handle tables ----
        for table in container.find_all("table"):
            for row in table.find_all("tr"):
                th = row.find("th")
                tds = row.find_all("td")

                if th and tds:  # Case: <th> + <td>
                    key = self.clean_text(th.get_text(" ", strip=True))
                    val = self.clean_text(" ".join(td.get_text(" ", strip=True) for td in tds))
                    details.append(f"{key}: {val}")
                    specs_dict[key] = val

                elif len(tds) >= 2:  # Case: <td> + <td>
                    key = self.clean_text(tds[0].get_text(" ", strip=True))
                    val = self.clean_text(" ".join(td.get_text(" ", strip=True) for td in tds[1:]))
                    details.append(f"{key}: {val}")
                    specs_dict[key] = val

                elif len(tds) == 1:  # Case: only <td>
                    cell = tds[0]
                    lis = cell.find_all("li")
                    if lis:
                        val = ", ".join(self.clean_text(li.get_text(" ", strip=True)) for li in lis)
                        details.append(val)
                        specs_dict.setdefault("Features", val)
                    else:
                        val = self.clean_text(cell.get_text(" ", strip=True))
                        details.append(val)
                        specs_dict.setdefault("Other", val)

            table.decompose()

        # ---- Handle ul/li outside tables ----
        for ul in container.find_all("ul"):
            lis = [self.clean_text(li.get_text(" ", strip=True)) for li in ul.find_all("li")]
            if lis:
                val = ", ".join(lis)
                details.append(val)
                specs_dict.setdefault("Features", val)
            ul.decompose()

        # ---- Handle plain text (p tags, etc.) ----
        description_text = self.clean_text(container.get_text(" ", strip=True))

        # Combine all parts, separate with commas
        parts = []
        if description_text:
            parts.append(description_text)
        if details:
            parts.extend(details)

        combined_text = self.clean_text(", ".join(parts))

        return {"text": combined_text, "specs": specs_dict}

    def _extract_price_numeric(self, price_text):
        """Extract numeric price from price text."""
        if not price_text:
            return 0
        
        # Remove currency symbols and extract numbers
        import re
        numbers = re.findall(r'[\d,]+', price_text.replace(',', ''))
        if numbers:
            try:
                return float(numbers[0])
            except:
                return 0
        return 0

    def scrape_smartphones(self, max_pages=2):
        """Scrape Android/general smartphones from Jumia."""
        return self._scrape_category(
            start_url="https://www.jumia.co.ke/smartphones/",
            max_pages=max_pages,
            category_name="smartphones"
        )

    def scrape_ios_phones(self, max_pages=5):
        """Scrape iOS phones from Jumia using the same logic as smartphones."""
        return self._scrape_category(
            start_url="https://www.jumia.co.ke/ios-phones/",
            max_pages=max_pages,
            category_name="ios_phones"
        )

    def _scrape_category(self, start_url: str, max_pages: int, category_name: str):
        """Generic category scraper reused by both smartphones and iOS phones."""
        print(f"🔄 Starting scraping for {category_name} (max {max_pages} pages)...")
        
        url = start_url
        all_products = []
        
        for page in range(max_pages):
            print(f"📄 Scraping page {page + 1} of {category_name}...")
            
            try:
                response_content = requests.get(url, headers=self.headers)
                response_content.raise_for_status()
                soup = BeautifulSoup(response_content.text, "html.parser")

                product_cards = soup.find(name="div", class_="-phs -pvxs row _no-g _4cl-3cm-shs")
                if not product_cards:
                    print(f"⚠️ No product cards found on page {page + 1} for {category_name}")
                    break
                    
                articles = product_cards.find_all(name="article", class_="prd _fb col c-prd")
                
                if not articles:
                    print(f"⚠️ No articles found on page {page + 1} for {category_name}")
                    break

                for article in articles:
                    try:
                        anchors = article.find_all(name="a")
                        if len(anchors) < 2:
                            continue
                            
                        link = anchors[1]["href"]
                        info = article.find(name="div", class_="info")
                        if not info:
                            continue
                            
                        name = info.find(name="h3", class_="name")
                        price = info.find(name="div", class_="prc")
                        rev = info.find(name="div", class_="rev")
                        
                        # Extract image URL from listing page (efficient approach)
                        image_div = article.find("div", class_="img-c")
                        img_tag = image_div.find("img", class_="img") if image_div else None
                        image_url = img_tag["data-src"] if img_tag and img_tag.has_attr("data-src") else ""

                        if not name or not price:
                            continue

                        if rev:
                            rating = rev.find(name="div", class_="stars _s")
                            rating_text = rating.text if rating else "No rating"
                        else:
                            rating_text = "No rating"

                        # Fetch product page for detailed description
                        time.sleep(0.5)  # Be respectful to the server
                        product_url = self.official_url + link
                        product_page = requests.get(product_url, headers=self.headers)
                        product_page.raise_for_status()
                        
                        another_soup = BeautifulSoup(product_page.text, "html.parser")
                        product_description_container = another_soup.find(
                            name="div", class_="markup -mhm -pvl -oxa -sc"
                        )

                        description_data = self.extract_product_description(product_description_container)

                        # Parse price to numeric value
                        price_text = self.clean_text(price.text)
                        price_numeric = self._extract_price_numeric(price_text)

                        # Generate product ID
                        product_id = self._generate_product_id(product_url)

                        product_data = {
                            "id": product_id,
                            "name": self.clean_text(name.text),
                            "title": self.clean_text(name.text),  # Use name as title
                            "price_text": price_text,
                            "price_numeric": price_numeric,
                            "rating": self.clean_text(rating_text),
                            "url": product_url,
                            "image_url": image_url,
                            "description": description_data["text"],
                            "specs": json.dumps(description_data["specs"]),
                            "category": category_name,
                            "scraped_at": datetime.now().isoformat()
                        }

                        all_products.append(product_data)
                        print(f"✅ Scraped ({category_name}): {product_data['name']}")
                        print(f"🖼️ Image URL: {image_url}")

                    except Exception as e:
                        print(f"⚠️ Error scraping individual product in {category_name}: {e}")
                        continue

                # Go to next page
                try:
                    pages_div = soup.find(name="div", class_="pg-w -ptm -pbxl")
                    if pages_div:
                        anchor = pages_div.find("a", attrs={"aria-label": "Next Page"})
                        if anchor and anchor.get("href"):
                            next_page_link = url[:-1] + anchor["href"]
                            url = next_page_link
                            time.sleep(0.5)  # Be respectful between pages
                        else:
                            print(f"📄 No more pages found after page {page + 1} for {category_name}")
                            break
                    else:
                        print(f"📄 No pagination found after page {page + 1} for {category_name}")
                        break
                        
                except Exception as e:
                    print(f"⚠️ Error finding next page for {category_name}: {e}")
                    break

            except Exception as e:
                print(f"❌ Error scraping page {page + 1} for {category_name}: {e}")
                break

        print(f"✅ Scraping completed for {category_name}. Found {len(all_products)} products")
        return all_products

    def store_products(self, products):
        """Store products in SQLite database."""
        if not products:
            print("⚠️ No products to store")
            return {'inserted': 0, 'updated': 0}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        inserted = 0
        updated = 0
        
        for product in products:
            # Check if product exists
            cursor.execute("SELECT id FROM products WHERE id = ?", (product['id'],))
            exists = cursor.fetchone()
            
            if exists:
                # Update existing product
                cursor.execute('''
                    UPDATE products SET 
                    name=?, title=?, price_text=?, price_numeric=?, rating=?, 
                    url=?, image_url=?, description=?, specs=?, category=?, scraped_at=?
                    WHERE id=?
                ''', (
                    product['name'], product['title'], product['price_text'], 
                    product['price_numeric'], product['rating'], product['url'],
                    product['image_url'], product['description'], product['specs'],
                    product['category'], product['scraped_at'], product['id']
                ))
                updated += 1
            else:
                # Insert new product
                cursor.execute('''
                    INSERT INTO products 
                    (id, name, title, price_text, price_numeric, rating, url, image_url, 
                     description, specs, category, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    product['id'], product['name'], product['title'], product['price_text'],
                    product['price_numeric'], product['rating'], product['url'],
                    product['image_url'], product['description'], product['specs'],
                    product['category'], product['scraped_at']
                ))
                inserted += 1
        
        conn.commit()
        conn.close()
        
        result = {'inserted': inserted, 'updated': updated}
        print(f"💾 Database storage completed: {result}")
        return result

    def search_products(self, query: str, limit: int = 5):
        """Simple text search in products."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simple text search in name, description
        search_query = f"%{query.lower()}%"
        cursor.execute('''
            SELECT id, name, price_text, url, image_url, rating, description, specs
            FROM products 
            WHERE LOWER(name) LIKE ? OR LOWER(description) LIKE ?
            ORDER BY price_numeric ASC
            LIMIT ?
        ''', (search_query, search_query, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        # Format results
        products = []
        for row in results:
            specs = {}
            try:
                specs = json.loads(row[7]) if row[7] else {}
            except:
                pass
            
            product = {
                'id': row[0],
                'name': row[1],
                'price_text': row[2],
                'url': row[3],
                'image_url': row[4],
                'rating': row[5],
                'description': row[6][:200] + "..." if len(row[6]) > 200 else row[6],
                'specs': specs
            }
            products.append(product)
        
        print(f"🔍 Found {len(products)} products for query: '{query}'")
        return products

    def get_stats(self):
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM products")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(price_numeric) FROM products WHERE price_numeric IS NOT NULL")
        avg_price = cursor.fetchone()[0]
        
        cursor.execute("SELECT category, COUNT(*) FROM products GROUP BY category")
        categories = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_products': total,
            'average_price': round(avg_price, 2) if avg_price else 0,
            'categories': dict(categories)
        }

    def run_complete_pipeline(self, max_pages: int = 2, save_excel: bool = True):
        """Run the complete scraping and storage pipeline for smartphones + iOS phones."""
        print("🚀 Starting Basic Jumia Scraper Pipeline...")
        
        # Scrape Android/general smartphones
        smartphones = self.scrape_smartphones(max_pages)
        
        # Scrape iOS phones (fixed 5 pages as requested)
        ios_phones = self.scrape_ios_phones(max_pages=5)
        
        # Combine all products
        products = smartphones + ios_phones
        
        if not products:
            print("❌ No products scraped")
            return {'total_products': 0, 'inserted': 0, 'updated': 0}
        
        # Save to Excel if requested
        if save_excel:
            df = pd.DataFrame(products)
            df.to_excel("jumia_smartphones_basic.xlsx", index=False)
            print("📊 Products saved to jumia_smartphones_basic.xlsx")
        
        # Store in database
        storage_result = self.store_products(products)
        
        # Get final stats
        stats = self.get_stats()
        
        summary = {
            'total_products': len(products),
            'inserted': storage_result['inserted'],
            'updated': storage_result['updated'],
            'database_stats': stats
        }
        
        print(f"✅ Pipeline completed: {summary}")
        return summary


def main():
    """Main function to run the basic scraper."""
    scraper = BasicJumiaScraper()
    
    # Run the complete pipeline
    # max_pages: How many pages to scrape (each page ~8-10 products)
    # 25 pages = ~200-250 products (from current 80 with 10 pages)
    result = scraper.run_complete_pipeline(max_pages=25, save_excel=True)
    print(f"\n🎉 Scraping completed: {result}")
    
    # Test search functionality
    if result['total_products'] > 0:
        print("\n🔍 Testing search functionality...")
        search_queries = [
            "Samsung",
            "iPhone",
            "Android",
            "camera"
        ]
        
        for query in search_queries:
            print(f"\nSearching for: '{query}'")
            results = scraper.search_products(query, limit=3)
            for i, product in enumerate(results, 1):
                print(f"  {i}. {product['name']} - {product['price_text']}")
                if product['specs']:
                    key_specs = list(product['specs'].items())[:1]
                    if key_specs:
                        print(f"     {key_specs[0][0]}: {key_specs[0][1]}")


if __name__ == "__main__":
    main()

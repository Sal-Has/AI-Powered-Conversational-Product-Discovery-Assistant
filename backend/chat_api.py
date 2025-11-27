from __future__ import annotations

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import requests
import openai
from openai import OpenAI

from product_pipeline import JumiaProductPipeline
from models import db, ChatSession, ChatMessage, ProductIndex
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
chat_bp = Blueprint('chat', __name__)

# Initialize RAG components
pipeline = None
openai_client = None

def init_rag_components():
    """Initialize RAG pipeline and OpenAI client."""
    global pipeline, openai_client
    
    try:
        # Initialize product pipeline for semantic search
        pipeline = JumiaProductPipeline(
            chroma_persist_directory="./chroma_db",
            collection_name="jumia_products"
        )
        logger.info("✅ RAG Pipeline initialized")
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        openai_client = OpenAI(api_key=api_key)
        logger.info("✅ OpenAI client initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG components: {e}")
        raise

# Initialize components when module loads
init_rag_components()


def fetch_live_status(url: str) -> Tuple[Optional[int], Optional[str], str]:
    """Fetch latest price and availability status for a Jumia product URL.

    Returns (price_numeric, price_text, status)
    status is one of: 'in_stock', 'out_of_stock', 'not_found', 'unknown'.

    This is a best-effort HTML scrape and may need selector tuning if Jumia changes
    its layout. It is kept intentionally simple and defensive.
    """
    if not url:
        return None, None, 'unknown'

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; JumiaPriceChecker/1.0)"
        }
        resp = requests.get(url, headers=headers, timeout=6)

        # Basic HTTP checks
        if resp.status_code == 404:
            return None, None, 'not_found'
        if resp.status_code >= 500:
            return None, None, 'unknown'

        # Detect redirects to a different product or generic page.
        # Many Jumia product URLs end with a numeric code like ...-326022109.html.
        # If the final response URL no longer contains that code, we treat it
        # as the original product being unavailable / removed.
        import re
        product_code_match = re.search(r"(\d+)\.html?", url)
        if product_code_match:
            original_code = product_code_match.group(1)
            final_url = resp.url or url
            if original_code not in final_url:
                # Redirected away from the specific product -> treat as not found
                logger.info(
                    f"🔁 Live check redirect detected. Original code {original_code} "
                    f"not in final URL {final_url}. Marking as not_found."
                )
                return None, None, 'not_found'

        html = resp.text
        html_lower = html.lower()

        # --- Availability heuristics ---
        # We prefer strong positive signals ("add to cart", explicit stock count)
        # over generic "out of stock" phrases that might appear elsewhere on the page.

        status = 'unknown'

        try:
            # Positive indicators: product can be added to cart and/or shows remaining items
            positive_phrases = [
                'add to cart',
                'add to basket',
                'add to bag',
                'items left',
                'only',  # often appears in "only 3 items left"
            ]

            has_positive = any(p in html_lower for p in positive_phrases)

            # Negative indicators: explicit out-of-stock messaging
            negative_phrases = [
                'out of stock',
                'currently unavailable',
                'this item is unavailable',
            ]
            has_negative = any(p in html_lower for p in negative_phrases)

            if has_positive:
                # If we see clear purchase signals, treat as in stock even if
                # a generic "out of stock" phrase appears elsewhere on the page.
                status = 'in_stock'
            elif has_negative:
                status = 'out_of_stock'
            else:
                # Default assumption when we don't see strong signals either way
                status = 'in_stock'
        except Exception as e:
            logger.warning(f"⚠️ Availability heuristic failed for URL {url}: {e}")
            status = 'unknown'

        # Try to find a price pattern like "KSh 18,999" or "KSH18,999"
        price_match = re.search(r'(ksh|ksh\.|kshs)\s*([\d,]+)', html_lower)
        price_text = None
        price_numeric: Optional[int] = None

        if price_match:
            # Rebuild a nicer price_text with original case if possible
            digits = price_match.group(2)
            price_text = f"KSh {digits}"
            try:
                price_numeric = int(digits.replace(',', ''))
            except ValueError:
                price_numeric = None

        return price_numeric, price_text, status

    except Exception as e:
        logger.error(f"❌ Live status fetch failed for URL {url}: {e}")
        return None, None, 'unknown'


@chat_bp.route('/products/<product_id>/check_live', methods=['GET'])
@jwt_required()
def check_live_product(product_id: str):
    """Check latest price and availability for a product by ID.

    Uses ProductIndex as a cache. On the first call, the frontend MUST pass
    ?url=<jumia_product_url>. Later calls can omit url and will reuse the stored one.
    """
    current_user = get_jwt_identity()
    logger.info(f"🔍 Live check requested by user {current_user} for product_id={product_id}")

    # URL can be provided by frontend on first call or when changed
    url = request.args.get('url', '').strip()

    # Look up existing index row (if any)
    index = db.session.query(ProductIndex).filter_by(product_id=product_id).first()

    if not url:
        # If no URL in query, we must already have one stored
        if not index or not index.url:
            return jsonify({
                'success': False,
                'error': 'Product URL is required for live check (use ?url=...)'
            }), 400
        url = index.url
    else:
        # If URL is provided, ensure we have an index row
        if not index:
            index = ProductIndex(product_id=product_id, url=url)
            db.session.add(index)

    # Fetch live status
    price_num, price_text, status = fetch_live_status(url)

    from datetime import datetime as _dt
    index.last_checked_at = _dt.utcnow()
    index.url = url

    # If the product is no longer found or the page is gone, clear any
    # previously cached price to avoid showing misleading stale prices.
    if status == 'not_found':
        index.last_seen_price_text = None
        index.last_seen_price_numeric = None
    else:
        if price_text:
            index.last_seen_price_text = price_text
        if price_num is not None:
            index.last_seen_price_numeric = price_num

    if status:
        index.last_seen_status = status

    try:
        db.session.commit()
    except Exception as e:
        logger.error(f"❌ Failed to update ProductIndex for {product_id}: {e}")
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': 'Failed to update live status'
        }), 500

    return jsonify({
        'success': True,
        'product_id': product_id,
        'url': index.url,
        'latest_price_text': index.last_seen_price_text,
        'latest_price_numeric': index.last_seen_price_numeric,
        'status': index.last_seen_status,
        'last_checked_at': index.last_checked_at.isoformat() if index.last_checked_at else None,
    }), 200


# ========== CONVERSATION MANAGEMENT CLASSES ==========

class ConversationMemory:
    """
    Manages conversation memory for each user session.
    Stores recent queries and product recommendations for comparison.
    """
    
    def __init__(self, max_history: int = 5, ttl_minutes: int = 30):
        """
        Args:
            max_history: Maximum number of conversation turns to store
            ttl_minutes: Time-to-live for session data in minutes
        """
        self.sessions: Dict[str, Dict[str, Any]] = {}  # user_id -> session_data
        self.max_history = max_history
        self.ttl_minutes = ttl_minutes
    
    def get_session(self, user_id: str) -> Dict[str, Any]:
        """Get or create a session for a user."""
        self._cleanup_expired_sessions()
        
        if user_id not in self.sessions:
            self.sessions[user_id] = {
                'history': [],
                'last_products': [],
                'last_query': '',
                # Tracks the specific products that were presented as options
                # in the last assistant answer (typically top 2-3 items).
                'last_options': [],
                'last_updated': datetime.now()
            }
        
        return self.sessions[user_id]
    
    def add_conversation_turn(self, user_id: str, query: str, products: List[Dict[str, Any]], answer: str, action: str = 'search'):
        """Add a conversation turn to the session and save to database."""
        session = self.get_session(user_id)
        
        turn = {
            'query': query,
            'products': products,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        }
        
        session['history'].append(turn)
        session['last_products'] = products
        session['last_query'] = query
        # By default, treat the first few products as the "options" explicitly
        # presented to the user in the last answer. This lets us later honor
        # phrases like "compare the two options" or "compare the three options".
        session['last_options'] = products[:3] if products else []
        session['last_updated'] = datetime.now()
        
        # Keep only max_history recent turns
        if len(session['history']) > self.max_history:
            session['history'] = session['history'][-self.max_history:]
        
        # Save to database
        self._save_to_database(user_id, query, products, answer, action)
    
    def get_last_products(self, user_id: str) -> List[Dict[str, Any]]:
        """Get the last retrieved products for a user."""
        session = self.get_session(user_id)
        return session.get('last_products', [])
    
    def get_conversation_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get the conversation history for a user."""
        session = self.get_session(user_id)
        return session.get('history', [])
    
    def clear_session(self, user_id: str):
        """Clear session data for a user."""
        if user_id in self.sessions:
            del self.sessions[user_id]
    
    def _get_or_create_db_session(self, user_id: str) -> ChatSession:
        """Get or create database session for user."""
        # Get active session
        db_session = db.session.query(ChatSession).filter_by(
            user_email=user_id,
            is_active=True
        ).first()
        
        if not db_session:
            db_session = ChatSession(user_email=user_id)
            db.session.add(db_session)
            db.session.commit()
            logger.info(f"✅ Created new database session for {user_id}")
        
        return db_session
    
    def _save_to_database(self, user_id: str, query: str, products: List[Dict[str, Any]], answer: str, action: str):
        """Save conversation turn to database."""
        try:
            db_session = self._get_or_create_db_session(user_id)
            
            # Save user message
            user_msg = ChatMessage(
                session_id=db_session.id,
                role='user',
                content=query,
                query=query,
                action=action
            )
            db.session.add(user_msg)
            
            # Save assistant message with products
            assistant_msg = ChatMessage(
                session_id=db_session.id,
                role='assistant',
                content=answer,
                query=query,
                products_json=json.dumps(products[:5]) if products else '[]',  # Save top 5 products
                action=action
            )
            db.session.add(assistant_msg)
            
            # Update session timestamp
            db_session.updated_at = datetime.utcnow()
            
            db.session.commit()
            logger.info(f"💾 Saved conversation turn to database for {user_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save to database: {e}")
            db.session.rollback()
    
    def get_all_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all chat sessions for a user from database."""
        sessions = db.session.query(ChatSession).filter_by(user_email=user_id).order_by(ChatSession.created_at.desc()).all()
        return [s.to_dict() for s in sessions]
    
    def get_session_messages(self, session_id: int) -> List[Dict[str, Any]]:
        """Get all messages for a specific session."""
        messages = db.session.query(ChatMessage).filter_by(session_id=session_id).order_by(ChatMessage.created_at.asc()).all()
        return [m.to_dict() for m in messages]
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions based on TTL."""
        now = datetime.now()
        expired_users = []
        
        for user_id, session in self.sessions.items():
            last_updated = session.get('last_updated', now)
            if (now - last_updated).total_seconds() > (self.ttl_minutes * 60):
                expired_users.append(user_id)
        
        for user_id in expired_users:
            del self.sessions[user_id]
            logger.info(f"🧹 Cleaned up expired session for user {user_id}")


class QueryAnalyzer:
    """
    Analyzes user queries to detect vagueness, ambiguity, and comparison intent.
    """
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
    
    def is_off_topic(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Detect if the query is completely off-topic (not related to products/shopping).
        
        Returns:
            (is_off_topic, off_topic_response)
            - is_off_topic: True if query is unrelated to products
            - off_topic_response: Polite rejection message
        """
        query_lower = query.lower()
        
        # Product/shopping related keywords - if ANY of these are present, it's ON topic
        # Include core product types, common attributes (price, rating, battery, warranty, specs),
        # and shopping verbs so short questions like "does it have a warranty" are not rejected.
        product_keywords = [
            'phone', 'smartphone', 'mobile', 'iphone', 'samsung', 'xiaomi', 'oppo', 'tecno',
            'laptop', 'computer', 'tablet', 'ipad', 'macbook',
            'headphone', 'earphone', 'earbuds', 'airpods',
            'tv', 'television', 'monitor', 'screen', 'display',
            'camera', 'watch', 'smartwatch',
            'charger', 'cable', 'adapter', 'accessory',
            'speaker', 'bluetooth', 'wireless',
            'product', 'buy', 'purchase', 'price', 'cost', 'cheap', 'expensive', 'affordable',
            'ksh', 'kes', 'shilling', 'budget',
            'gb', 'ram', 'storage', 'battery', 'mah',
            'jumia', 'shop', 'shopping', 'store',
            'android', 'ios', 'windows',
            'electronic', 'device', 'gadget',
            'spec', 'specs', 'specifications', 'feature', 'features', 'detail', 'review', 'rating',
            'warranty', 'guarantee'
        ]
                # Comparison/refinement related keywords – still on-topic even without product words
        comparison_keywords = [
            'compare', 'comparison', 'versus', 'vs',
            'cheaper', 'more expensive', 'less expensive', 'lower price', 'higher price', 'pricier',
            'better option', 'better options', 'better one',
            'first two', 'first 2', 'second one', 'third one',
            'other options', 'more options', 'different options',
            'this one', 'that one', 'these options'
        ]

        if any(keyword in query_lower for keyword in comparison_keywords):
            # User is refining or comparing previously shown products
            return False, None
        
        # If ANY product keyword is found, it's ON topic
        if any(keyword in query_lower for keyword in product_keywords):
            return False, None
        
        # Off-topic categories to detect
        off_topic_keywords = {
            'politics': ['president', 'minister', 'government', 'congress', 'parliament', 'election', 'vote', 'political'],
            'geography': ['capital', 'country', 'continent', 'ocean', 'mountain', 'river'],
            'weather': ['weather', 'temperature', 'rain', 'sunny', 'cloudy', 'forecast'],
            'sports': ['football', 'basketball', 'tennis', 'soccer', 'cricket', 'match', 'game', 'score'],
            'entertainment': ['movie', 'film', 'actor', 'actress', 'song', 'music', 'concert'],
            'math': ['calculate', 'solve', 'equation', 'formula', 'mathematics'],
            'general_knowledge': ['who is', 'what is', 'when was', 'where is', 'how many people']
        }
        
        # Check if query matches off-topic categories
        for category, keywords in off_topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                logger.info(f"🚫 Off-topic query detected: {category}")
                return True, "I'm a shopping assistant specialized in helping you find electronics and products on Jumia Kenya. I can only help with product searches, comparisons, and recommendations. Please ask me about phones, laptops, accessories, or any other products you're interested in!"
        
        # If query is a simple greeting, allow it
        generic_patterns = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if query_lower.strip() in generic_patterns:
            return False, None  # Greetings are OK

        # Default: if it's not clearly product/shopping related, treat as off-topic
        return True, (
            "I'm a shopping assistant specialized in helping you find electronics and products on "
            "Jumia Kenya. I can only help with product searches, comparisons, and recommendations. "
            "Please ask me about phones, laptops, accessories, or other products you're interested in."
        )
    
    def detect_product_reference(self, query: str, has_previous_products: bool) -> Tuple[bool, Optional[str]]:
        """
        Detect if the user is referring to a previous product using pronouns or references.
        E.g., "give me its specs", "tell me more about it", "show me the first one"
        
        Returns:
            (is_reference, reference_type)
            - is_reference: True if query refers to previous product
            - reference_type: 'pronoun', 'ordinal', 'details' or None
        """
        if not has_previous_products:
            return False, None
        
        query_lower = query.lower()
        
        # Pronoun references
        pronoun_patterns = [
            'its specs', 'its spec', 'its detail', 'its features', 'its price',
            'it spec', 'it detail', 'it features',
            'that one', 'this one', 'the one',
            'about it', 'more about', 'tell me more',
            'details of it', 'info about it'
        ]
        
        # Ordinal references
        ordinal_patterns = [
            'first one', 'first option', 'second one', 'second option',
            'option 1', 'option 2', 'number 1', 'number 2'
        ]
        
        # Detail request keywords
        detail_keywords = ['spec', 'detail', 'feature', 'info', 'information', 'more']
        
        # Check for pronoun patterns
        for pattern in pronoun_patterns:
            if pattern in query_lower:
                logger.info(f"🔗 Detected pronoun reference: '{pattern}'")
                return True, 'pronoun'
        
        # Check for ordinal patterns
        for pattern in ordinal_patterns:
            if pattern in query_lower:
                logger.info(f"🔗 Detected ordinal reference: '{pattern}'")
                return True, 'ordinal'
        
        # Check if it's a detail request with pronouns
        has_pronoun = any(word in query_lower.split() for word in ['it', 'its', 'that', 'this'])
        has_detail_keyword = any(keyword in query_lower for keyword in detail_keywords)
        
        if has_pronoun and has_detail_keyword:
            logger.info(f"🔗 Detected detail request with pronoun reference")
            return True, 'details'
        
        return False, None
    
    def analyze_query_specificity(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Analyze if a query is specific enough to perform retrieval.
        
        Returns:
            (is_specific, clarification_question)
            - is_specific: True if query is specific enough, False if too vague
            - clarification_question: Question to ask user if query is vague, None otherwise
        """
        
        # Quick pre-check: If query has product keywords, it's specific enough
        product_keywords = [
            'phone', 'smartphone', 'mobile', 'iphone', 'samsung', 'xiaomi', 'oppo', 'tecno',
            'laptop', 'computer', 'tablet', 'ipad', 'macbook',
            'headphone', 'earphone', 'earbuds', 'airpods',
            'tv', 'television', 'monitor', 'screen',
            'camera', 'watch', 'smartwatch',
            'charger', 'cable', 'adapter',
            'speaker', 'bluetooth', 'wireless',
            'ksh', 'kes', 'shilling', 'price', 'budget', 'cheap', 'expensive', 'affordable',
            'gb', 'ram', 'storage', 'battery', 'mah'
        ]
        
        query_lower = query.lower()
        
        # If any product keyword is found, consider it specific
        if any(keyword in query_lower for keyword in product_keywords):
            logger.info(f"📊 Query contains product keywords, considered specific: '{query}'")
            return True, None
        
        # If query is very short (less than 3 words) and has no product keywords, it's vague
        word_count = len(query.split())
        if word_count < 3:
            logger.info(f"📊 Query too short and no product keywords: '{query}'")
            return False, "Could you please be more specific? What type of product are you looking for? (e.g., phone, laptop, headphones) and what's your budget?"
        
        # For other cases, use AI analysis
        analysis_prompt = f"""Analyze the following product search query and determine if it's specific enough to return relevant results.

Query: "{query}"

A query is considered TOO VAGUE ONLY if it lacks ALL of these:
- NO product category mentioned (e.g., "phone", "laptop", "headphones")
- NO specific requirements (e.g., price range, brand, features, use case)
- Only contains extremely generic terms like "good", "best", "nice", "something"

Examples of VAGUE queries (mark as vague):
- "I need something good"
- "Show me products"
- "What do you have?"
- "I want to buy something"
- "best product"

Examples of SPECIFIC queries (mark as specific):
- "Samsung phone under 20000 KES"
- "Laptop for gaming with good graphics"
- "Wireless headphones with noise cancellation"
- "iPhone with good camera below 80000 ksh"
- "cheap phone with good battery"
- "best gaming laptop"
- "phone under 50000"

Respond in JSON format:
{{
    "is_specific": true/false,
    "reason": "brief explanation",
    "clarification_question": "question to ask user if vague, or null if specific"
}}

CRITICAL RULES:
- If the query mentions ANY product type AT ALL, mark it as specific (is_specific: true)
- If the query mentions ANY price, brand, or feature, mark it as specific (is_specific: true)
- Only mark as vague if it's EXTREMELY generic with ZERO product information
- When in doubt, mark as SPECIFIC
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a lenient query analysis assistant. You should only mark queries as vague if they are EXTREMELY generic. Always respond with valid JSON only, no additional text."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent behavior
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON from markdown code blocks if present
            if content.startswith("```"):
                # Extract JSON from code block
                lines = content.split('\n')
                content = '\n'.join([line for line in lines if not line.startswith('```') and line.strip() != 'json'])
                content = content.strip()
            
            result = json.loads(content)
            
            is_specific = result.get('is_specific', True)  # Default to True if not specified
            clarification = result.get('clarification_question')
            
            logger.info(f"📊 Query analysis: is_specific={is_specific}, query='{query}'")
            
            return is_specific, clarification
            
        except json.JSONDecodeError as je:
            logger.error(f"❌ Query analysis JSON parse failed: {je}. Response: {content[:200] if 'content' in locals() else 'N/A'}")
            # Default to considering query specific on JSON error
            return True, None
        except Exception as e:
            logger.error(f"❌ Query analysis failed: {e}")
            # Default to considering query specific on error
            return True, None
    
    def extract_products_from_comparison(self, query: str) -> Optional[Tuple[str, str]]:
        """
        Extract two product names from direct comparison queries.
        E.g., "compare iphone 12 and iphone 11" -> ("iphone 12", "iphone 11")
        E.g., "compare the specs of iphone 12 and iphone 13" -> ("iphone 12", "iphone 13")
        """
        import re
        
        query_lower = query.lower()
        
        # Pattern 1: "compare [the] [attribute] [of] Product1 and Product2"
        pattern1 = r'compare\s+(?:the\s+)?(?:specs?|specifications?|features?|price|battery|camera|performance)?\s*(?:of\s+)?(?:the\s+)?(.+?)\s+(?:and|with|vs|versus)\s+(?:the\s+)?(.+?)$'
        
        # Pattern 2: "difference between Product1 and Product2"
        pattern2 = r'(?:difference|differences)\s+(?:in\s+)?(?:specs?|specifications?|features?|price)?\s*(?:between\s+)?(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)$'
        
        # Pattern 3: "Product1 vs Product2"
        pattern3 = r'(.+?)\s+(?:vs|versus)\s+(.+?)(?:\s*comparison)?$'
        
        for pattern in [pattern1, pattern2, pattern3]:
            match = re.search(pattern, query_lower)
            if match:
                product1 = match.group(1).strip()
                product2 = match.group(2).strip()
                
                # Clean up: remove common words
                for word in ['the', 'a', 'an']:
                    product1 = product1.replace(f' {word} ', ' ').strip()
                    product2 = product2.replace(f' {word} ', ' ').strip()
                    product1 = product1.replace(f'{word} ', '').strip()
                    product2 = product2.replace(f'{word} ', '').strip()
                
                if product1 and product2 and len(product1) > 1 and len(product2) > 1:
                    logger.info(f"📱 Extracted products: '{product1}' and '{product2}'")
                    return (product1, product2)
        
        return None
    
    def detect_comparison_intent(self, query: str, has_previous_products: bool) -> Tuple[bool, str]:
        """
        Detect if the query is asking for comparison or refinement.
        
        Returns:
            (is_comparison, comparison_type)
            - is_comparison: True if user wants comparison
            - comparison_type: 'explicit' (direct request) or 'refinement' (query refinement)
        """
        
        query_lower = query.lower()
        
        # Map keywords to attribute names
        attribute_keywords = {
            'rating': ['rating', 'rated', 'review', 'star'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'affordable'],
            'battery': ['battery', 'battery life', 'mah', 'power'],
            'camera': ['camera', 'photo', 'picture', 'megapixel', 'mp'],
            'storage': ['storage', 'gb', 'memory', 'space', 'capacity'],
            'ram': ['ram', 'memory'],
            'screen': ['screen', 'display', 'size', 'inch'],
            'performance': ['performance', 'speed', 'fast', 'processor'],
            'specs': ['specs', 'specifications', 'features', 'specification']
        }
        
        # Explicit comparison keywords
        explicit_keywords = [
            'compare', 'comparison', 'versus', 'vs', 'difference between',
            'which is better', 'which one', 'better than', 'compared to',
            'previous', 'last one', 'the one before', 'earlier',
            'option 1', 'option 2', 'first phone', 'second phone'
        ]
        
        for keyword in explicit_keywords:
            if keyword in query_lower:
                logger.info(f"🔍 Explicit comparison detected: '{keyword}' in query")
                return True, 'explicit'
        
        
        # Refinement keywords (implies user wants to compare with previous)
        refinement_keywords = [
            'better', 'cheaper', 'more expensive', 'faster', 'larger',
            'smaller', 'newer', 'higher', 'lower', 'improved',
            'with more', 'with less', 'with better'
        ]
        
        if has_previous_products:
            for keyword in refinement_keywords:
                if keyword in query_lower:
                    logger.info(f"🔍 Refinement detected: '{keyword}' in query")
                    return True, 'refinement'
        
        return False, 'none'
   
    
    def detect_comparison_count(self, query: str) -> Optional[int]:
        """
        Detect how many products the user wants to compare.
        E.g., "compare the two options" -> 2, "compare all three" -> 3
        
        Returns:
            number of products to compare, or None for all
        """
        query_lower = query.lower()
        
        # Patterns for specific numbers
        number_patterns = [
            ('two', 2), ('both', 2), ('these two', 2), ('those two', 2), 
            ('the two', 2), ('2', 2),
            ('three', 3), ('all three', 3), ('these three', 3), ('3', 3),
            ('four', 4), ('4', 4),
            ('first two', 2), ('top two', 2), ('last two', 2)
        ]
        
        for pattern, count in number_patterns:
            if pattern in query_lower:
                logger.info(f"🔢 Detected comparison count: {count} products")
                return count
        
        return None  # Compare all available products
    
    def detect_comparison_attribute(self, query: str) -> Optional[str]:
        """
        Detect if user is asking to compare specific attributes.
        E.g., "compare the battery", "which has better camera"
        
        Returns:
            attribute name if detected, None otherwise
        """
        query_lower = query.lower()
        
        # Map attributes to their keywords
        attribute_keywords = {
            'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'cheaper', 'costlier'],
            'battery': ['battery', 'battery life', 'mah', 'power', 'charge'],
            'camera': ['camera', 'photo', 'picture', 'megapixel', 'mp', 'photography'],
            'storage': ['storage', 'gb', 'space', 'capacity'],
            'ram': ['ram', 'memory'],
            'screen': ['screen', 'display', 'size', 'inch'],
            'performance': ['performance', 'speed', 'fast', 'faster', 'processor', 'chipset'],
            'rating': ['rating', 'rated', 'review', 'star', 'reviews']
        }
        
        # Check if any attribute keywords are in the query
        for attribute, keywords in attribute_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                logger.info(f"🎯 Detected attribute comparison: {attribute}")
                return attribute
        
        return None




class ComparisonGenerator:
    """
    Generates structured product comparisons using LLM.
    """
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
    
    def generate_comparison(self, 
                          current_products: List[Dict[str, Any]], 
                          previous_products: List[Dict[str, Any]],
                          current_query: str,
                          previous_query: str,
                          comparison_type: str,
                          attribute: Optional[str] = None) -> str:
        """
        Generate a structured comparison between current and previous products.
        
        Args:
            current_products: Current search results
            previous_products: Previous search results
            current_query: User's current query
            previous_query: User's previous query
            comparison_type: Type of comparison ('explicit', 'refinement', 'direct')
            attribute: If specified, focus comparison on this specific attribute
        """
        
        # Format previous products
        previous_info = self._format_products_for_comparison(previous_products, limit=3)
        
        # Format current products
        current_info = self._format_products_for_comparison(current_products, limit=3)
        
        # Adjust prompts based on whether specific attribute is requested
        if attribute:
            system_prompt = f"""You are a helpful shopping assistant that creates focused product comparisons.
The user wants to compare {attribute} specifically.
Focus your comparison primarily on {attribute}, but mention other relevant factors briefly if they're important.
Provide a clear recommendation based on the {attribute} comparison."""
            
            user_prompt = f"""The user asked: "{current_query}"

Previous products:
{previous_info}

Current products:
{current_info}

Provide a focused comparison of {attribute} between these products, highlighting which one(s) are better in terms of {attribute}, and give a recommendation."""
        
        elif comparison_type == 'explicit' or comparison_type == 'direct':
            system_prompt = """You are a helpful shopping assistant that creates detailed product comparisons.
Provide clear, structured comparisons highlighting key differences in:
- Price
- Key features and specifications
- Battery life (if applicable)
- Camera quality (if applicable)
- Performance/RAM/Storage
- User ratings
- Value for money

End with a clear recommendation based on the user's needs."""
            
            user_prompt = f"""The user asked: "{current_query}"

Previous search (from query: "{previous_query}"):
{previous_info}

Current search results:
{current_info}

Provide a detailed comparison between these products and recommend which one(s) would be best for the user's needs."""
        
        else:  # refinement
            system_prompt = """You are a helpful shopping assistant that helps users refine their product search.
When a user is looking for improvements over previous recommendations, help them understand:
- What's different in the new recommendations
- Whether the new options meet their refined requirements
- Trade-offs between old and new options
- A clear recommendation"""
            
            user_prompt = f"""The user initially searched for: "{previous_query}"
And now refined their search to: "{current_query}"

Previous recommendations:
{previous_info}

New recommendations based on refinement:
{current_info}

Explain how these new options compare to the previous ones, highlighting improvements that match their refined criteria."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            comparison_text = response.choices[0].message.content.strip()
            logger.info("✅ Generated product comparison")
            
            return comparison_text
            
        except Exception as e:
            logger.error(f"❌ Comparison generation failed: {e}")
            return "I found some new products for you, but I'm having trouble generating a detailed comparison right now. Please review both sets of results to see which better matches your needs."
    def _format_products_for_comparison(self, products: List[Dict[str, Any]], limit: int = 3) -> str:
        """Format product list for LLM consumption."""
        if not products:
            return "No products found."
        
        formatted = []
        for i, product in enumerate(products[:limit], 1):
            specs_text = ""
            if product.get('specs') and isinstance(product['specs'], dict):
                key_specs = list(product['specs'].items())[:3]
                specs_text = ", ".join([f"{k}: {v}" for k, v in key_specs])
            
            product_text = f"""{i}. {product.get('name', 'Unknown')}
   - Price: {product.get('price_text', 'N/A')}
   - Rating: {product.get('rating', 'N/A')}
   - Key specs: {specs_text if specs_text else 'N/A'}"""
            
            formatted.append(product_text)
        
        return "\n\n".join(formatted)


class ConversationManager:
    """
    Main conversation manager that orchestrates all conversational intelligence features.
    """
    
    def __init__(self, openai_client: OpenAI, max_history: int = 5, session_ttl_minutes: int = 30):
        self.memory = ConversationMemory(max_history=max_history, ttl_minutes=session_ttl_minutes)
        self.query_analyzer = QueryAnalyzer(openai_client)
        self.comparison_generator = ComparisonGenerator(openai_client)
        self.openai_client = openai_client
    
    def process_query(self, 
                     user_id: str,
                     query: str) -> Dict[str, Any]:
        """
        Process a user query with full conversational intelligence.
        
        Returns a dict with:
        - action: 'clarify', 'compare', or 'search'
        - message: Optional message to user
        - should_search: Whether to proceed with retrieval
        - comparison_context: Data needed for comparison if action is 'compare'
        """
        
        # Get user session
        session = self.memory.get_session(user_id)
        last_products = session.get('last_products', [])
        last_query = session.get('last_query', '')
        last_options = session.get('last_options', [])
        
        # Check for comparison intent
        is_comparison, comparison_type = self.query_analyzer.detect_comparison_intent(
            query, 
            has_previous_products=len(last_products) > 0
        )
        
        # Handle direct product comparison (e.g., "compare iphone 12 and iphone 11")
        if is_comparison and not last_products:
            products_to_compare = self.query_analyzer.extract_products_from_comparison(query)
            if products_to_compare:
                # Also detect if specific attribute is requested
                attribute = self.query_analyzer.detect_comparison_attribute(query)
                logger.info(f"🔄 Direct comparison mode: {products_to_compare[0]} vs {products_to_compare[1]}")
                if attribute:
                    logger.info(f"🎯 Attribute-specific comparison requested: {attribute}")
                return {
                    'action': 'direct_compare',
                    'should_search': True,
                    'product1': products_to_compare[0],
                    'product2': products_to_compare[1],
                    'comparison_type': 'explicit',
                    'attribute': attribute
                }
        
        # Handle comparison with previous products
        if is_comparison and last_products:
            # Detect how many products user wants to compare
            comparison_count = self.query_analyzer.detect_comparison_count(query)
            logger.info(f"🔄 Comparison mode activated: {comparison_type}")
            if comparison_count:
                logger.info(f"📊 User wants to compare {comparison_count} products")
            return {
                'action': 'compare',
                'should_search': True,
                'comparison_type': comparison_type,
                'previous_products': last_products,
                'previous_query': last_query,
                # Explicit options list from last turn (typically top 2-3
                # products that were highlighted to the user). This lets us
                # honor phrases like "compare the two options" by using these
                # exact products when possible.
                'last_options': last_options,
                'comparison_count': comparison_count  # None means all products
            }
        
        # Analyze query specificity
        is_specific, clarification = self.query_analyzer.analyze_query_specificity(query)
        
        if not is_specific and clarification:
            logger.info(f"❓ Query too vague, requesting clarification")
            return {
                'action': 'clarify',
                'should_search': False,
                'message': clarification
            }
        
        # Query is specific enough, proceed with search
        return {
            'action': 'search',
            'should_search': True
        }
    
    def save_conversation_turn(self, 
                              user_id: str,
                              query: str,
                              products: List[Dict[str, Any]],
                              answer: str,
                              action: str = 'search'):
        """Save a conversation turn to memory and database."""
        self.memory.add_conversation_turn(user_id, query, products, answer, action)
    
    def generate_comparison_response(self,
                                   current_products: List[Dict[str, Any]],
                                   current_query: str,
                                   comparison_type: str,
                                   previous_products: List[Dict[str, Any]],
                                   previous_query: str,
                                   attribute: Optional[str] = None) -> str:
        """Generate comparison response."""
        return self.comparison_generator.generate_comparison(
            current_products=current_products,
            previous_products=previous_products,
            current_query=current_query,
            previous_query=previous_query,
            comparison_type=comparison_type,
            attribute=attribute
        )
    
    def clear_user_session(self, user_id: str):
        """Clear session data for a user."""
        self.memory.clear_session(user_id)


# Initialize conversation manager
conversation_manager = None

def init_conversation_manager():
    """Initialize conversation manager with OpenAI client."""
    global conversation_manager
    if openai_client:
        conversation_manager = ConversationManager(
            openai_client=openai_client,
            max_history=5,
            session_ttl_minutes=30
        )
        logger.info("✅ Conversation Manager initialized")
    else:
        logger.error("❌ Cannot initialize Conversation Manager without OpenAI client")

# Initialize after components are ready
if openai_client:
    init_conversation_manager()

def generate_conversational_response(query: str, products: List[Dict[str, Any]], conversation_context: str = None) -> str:
    """
    Generate conversational response using OpenAI GPT.
    
    Args:
        query: User's original query
        products: List of relevant products from semantic search
        conversation_context: Optional context from previous conversation turns
        
    Returns:
        Conversational response from GPT
    """
    if not products:
        return "I couldn't find any products matching your query. Could you try rephrasing your search or being more specific about what you're looking for?"
    
    # Prepare product information for the prompt
    product_info = []
    for i, product in enumerate(products, 1):
        specs_text = ""
        if product.get('specs') and isinstance(product['specs'], dict):
            key_specs = list(product['specs'].items())[:3]  # Top 3 specs
            specs_text = ", ".join([f"{k}: {v}" for k, v in key_specs])
        
        product_text = f"""
{i}. {product.get('name', 'Unknown Product')}
   Price: {product.get('price_text', 'Price not available')}
   Rating: {product.get('rating', 'No rating')}
   {f"Key specs: {specs_text}" if specs_text else ""}
   URL: {product.get('url', 'URL not available')}
"""
        product_info.append(product_text.strip())
    
    products_text = "\n\n".join(product_info)
    
    # Create system prompt
    system_prompt = """You are a helpful shopping assistant for Jumia Kenya, specializing in smartphones and electronics. 
Your role is to help customers find the perfect products based on their needs and budget.

Guidelines:
- Be conversational, friendly, and helpful
- Provide specific product recommendations from the search results
- Highlight key features that match the user's query
- Mention prices and help with budget considerations
- If comparing products, point out pros and cons
- Always encourage the user to check the product page for full details
- Keep responses concise but informative (max 300 words)
"""

    # Build user prompt with optional conversation context
    context_text = ""
    if conversation_context:
        context_text = f"\n\nConversation Context:\n{conversation_context}\n"
    
    user_prompt = f"""
User Query: "{query}"{context_text}

Here are the most relevant products I found:

{products_text}

Please provide a helpful, conversational response that:
1. Acknowledges the user's request (and references previous context if provided)
2. Recommends the most suitable products from the list
3. Explains why these products match their needs
4. Mentions key features, prices, and any standout characteristics
5. Encourages them to explore the product pages for more details
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=400,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"❌ OpenAI API error: {e}")
        return f"I found {len(products)} relevant products for your query, but I'm having trouble generating a detailed response right now. Please check the product list below for options that might interest you."


def filter_products_by_live_status(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out products that are known to be unavailable based on live status.

    Rules:
    - If ProductIndex exists and last_seen_status is 'unknown', 'not_found', or 'out_of_stock',
      the product is excluded from recommendations.
    - If there is no ProductIndex row for a product ID, we keep it (no live check has been done yet).
    """
    if not products:
        return products

    # Collect product IDs that have an id field
    product_ids = [p.get('id') for p in products if p.get('id')]
    if not product_ids:
        return products

    try:
        index_rows = db.session.query(ProductIndex).filter(ProductIndex.product_id.in_(product_ids)).all()
    except Exception as e:
        logger.error(f"❌ Failed to fetch ProductIndex rows for live filtering: {e}")
        return products

    status_by_id = {row.product_id: (row.last_seen_status or '').lower() for row in index_rows}

    filtered: List[Dict[str, Any]] = []
    dropped = 0

    for p in products:
        pid = p.get('id')
        status = status_by_id.get(pid)

        # If we have a stored status and it's clearly not available, skip it
        if status in {'unknown', 'not_found', 'out_of_stock'}:
            dropped += 1
            continue

        filtered.append(p)

    if dropped:
        logger.info(
            f"🧹 Live status filter removed {dropped} products (status in ['unknown','not_found','out_of_stock']) "
            f"out of {len(products)} candidates"
        )

    return filtered
def answer_direct_product_question(query: str, products: List[Dict[str, Any]]) -> Optional[str]:
    """
    Try to answer direct questions like:
      - 'what is the rating of the samsung a17 5G'
      - 'what is the price of the samsung a17 5G'
      - 'how much is the samsung a17'
      - 'what is the battery capacity of the samsung a17'
    using only the current products list (no extra GPT call).
    """
    if not products:
        return None

    q = query.lower()

    # Which attribute is being asked about?
    ask_rating = 'rating' in q or 'rate' in q
    # Treat "how much" as a price question only when it's not clearly tied to
    # RAM or storage. For example:
    #   - "how much is the samsung a16"  -> price
    #   - "how much ram does the samsung a05 have" -> RAM only
    ask_price = (
        'price' in q
        or ('how much' in q and 'ram' not in q and 'storage' not in q and 'rom' not in q)
        or 'cost' in q
    )
    ask_battery = 'battery' in q or 'mah' in q or 'battery life' in q
    ask_storage = 'storage' in q or 'rom' in q or 'gb' in q
    ask_ram = 'ram' in q
    ask_warranty = 'warranty' in q or 'guarantee' in q
    ask_features = 'key features' in q or 'features' in q
    ask_specs = 'specs' in q or 'specifications' in q or 'specification' in q
    # Camera-related questions ("camera quality", "megapixel", "mp", etc.)
    ask_camera = (
        'camera' in q
        or 'photo quality' in q
        or 'picture quality' in q
        or 'megapixel' in q
        or 'mp ' in q
        or q.strip().endswith(' mp')
    )

    if not (
        ask_rating or ask_price or ask_battery or ask_storage or ask_ram
        or ask_warranty or ask_features or ask_specs or ask_camera
    ):
        return None  # Not a simple attribute question

    # Try to find the mentioned product by name in the query
    def match_score(prod: Dict[str, Any]) -> int:
        name = (prod.get('name') or '').lower()
        title = (prod.get('title') or '').lower()
        score = 0

        # Strong bonus if the full name/title appears as a substring in the
        # query (helps distinguish similar models like A05 vs A16).
        full_label = (name or title).strip()
        if full_label and full_label in q:
            score += 5

        # Token overlap as a softer signal
        for token in name.split():
            if token and token in q:
                score += 1
        for token in title.split():
            if token and token in q:
                score += 1
        return score

    best_product = None
    best_score = 0
    for p in products:
        s = match_score(p)
        if s > best_score:
            best_score = s
            best_product = p

    # If no explicit name match but only one product is in context,
    # assume the user is asking about that single product (e.g. "what is the price?").
    if (not best_product or best_score == 0) and len(products) == 1:
        best_product = products[0]

    # If we still couldn't confidently pick a product, bail out
    if not best_product:
        return None

    name = best_product.get('name', 'this product')
    rating = best_product.get('rating')
    price_text = best_product.get('price_text')
    description = best_product.get('description') or ''
    specs = best_product.get('specs') or {}

    # Extract some likely fields from specs
    specs_lower = {str(k).lower(): str(v) for k, v in specs.items()} if isinstance(specs, dict) else {}
    battery_text = None
    storage_text = None
    ram_text = None
    warranty_text = None
    camera_text = None

    for k, v in specs_lower.items():
        if battery_text is None and ('battery' in k or 'mah' in v):
            battery_text = v
        if storage_text is None and ('rom' in k or 'storage' in k or 'gb' in v):
            storage_text = v
        if ram_text is None and 'ram' in k:
            ram_text = v
        if warranty_text is None and 'warranty' in k:
            warranty_text = v
        # Look for camera-related specs: keys mentioning camera, rear/front camera,
        # or values containing MP which usually indicate megapixels.
        if camera_text is None and (
            'camera' in k
            or 'rear camera' in k
            or 'front camera' in k
            or 'mp' in v.lower()
        ):
            camera_text = v

    # Build a concise answer
    parts = []

    if ask_rating and rating is not None:
        parts.append(f"The rating of {name} is {rating} out of 5.")

    if ask_price and price_text:
        parts.append(f"The price of {name} is {price_text}.")

    if ask_battery and battery_text:
        parts.append(f"It has a battery of {battery_text}.")

    if ask_storage and storage_text:
        parts.append(f"It comes with storage of {storage_text}.")

    if ask_ram and ram_text:
        parts.append(f"It has {ram_text} of RAM.")

    if ask_warranty:
        if warranty_text:
            parts.append(f"It comes with a warranty: {warranty_text}.")
        else:
            parts.append("I don't see warranty information in the specs. Please check the product page for exact warranty details.")

    if ask_camera:
        if camera_text:
            parts.append(f"The camera details for {name} are: {camera_text}.")
        else:
            # Fallback: try to pull a short snippet from the description mentioning camera/MP
            desc_lower = description.lower()
            if 'camera' in desc_lower or 'mp' in desc_lower:
                parts.append(
                    f"I couldn't find a dedicated camera spec field, but the description mentions: {description[:180]}..."
                )
            else:
                parts.append("I don't see clear camera specifications in the data I have. Please check the product page for detailed camera information.")

    # Key features / specs: summarise from specs dict
    if (ask_features or ask_specs) and specs_lower:
        # Take up to 5 key spec entries for a short summary
        key_items = list(specs.items())[:5]
        summary = "; ".join([f"{k}: {v}" for k, v in key_items])
        if ask_features:
            parts.append(f"Some key features of {name} are: {summary}.")
        if ask_specs and not ask_features:
            parts.append(f"Here are the main specifications of {name}: {summary}.")

    if not parts:
        # Could not answer specifically – let normal flow handle it
        return None

    return " ".join(parts)

@chat_bp.route('/chat', methods=['POST'])
@jwt_required()
def chat():
    """
    Main chat endpoint with conversational intelligence features.
    
    Features:
    - Query clarification for vague queries
    - Product comparison with conversation memory
    - Context-aware refinement suggestions
    """
    try:
        # Get current user
        current_user = get_jwt_identity()
        logger.info(f"🔍 Chat request from user: {current_user}")

        # Validate request
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400

        data = request.get_json()
        query = data.get('query', '').strip()

        if not query:
            return jsonify({'success': False, 'error': 'Query is required'}), 400

        if len(query) > 500:
            return jsonify({'success': False, 'error': 'Query too long (max 500 characters)'}), 400

        logger.info(f"🔍 Processing query: '{query}'")

        # Process query with conversation intelligence
        if not conversation_manager:
            logger.error("❌ Conversation manager not initialized")
            return jsonify({
                'success': False,
                'error': 'Conversation service unavailable'
            }), 503
        
        # Check if query is off-topic (not related to products)
        is_off_topic, off_topic_response = conversation_manager.query_analyzer.is_off_topic(query)
        if is_off_topic:
            logger.info(f"🚫 Rejected off-topic query: '{query}'")
            return jsonify({
                'success': True,
                'action': 'off_topic',
                'answer': off_topic_response,
                'products': [],
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'user': current_user
            }), 200

        # Check conversation history for context
        history = conversation_manager.memory.get_conversation_history(current_user)
        enhanced_query = query
        is_refinement = False
        previous_products = []
        
        if history and len(history) > 0:
            last_turn = history[-1]
            
            # Check if this is a follow-up to a clarification request (no products)
            if last_turn.get('products') is not None and len(last_turn.get('products', [])) == 0:
                # Combine previous vague query with current clarification
                previous_query = last_turn.get('query', '')
                if previous_query:
                    enhanced_query = f"{previous_query} {query}"
                    logger.info(f"🔗 Combined queries: '{previous_query}' + '{query}' = '{enhanced_query}'")
            
            # Check if this is a refinement of previous results (user had products before)
            elif last_turn.get('products') and len(last_turn.get('products', [])) > 0:
                # Detect refinement keywords
                refinement_keywords = ['that has', 'with', 'one that', 'which has', 'having', 'including', 'featuring']
                spec_keywords = ['gb', 'ram', 'rom', 'storage', 'battery', 'camera', 'mah', 'inch', 'display']
                
                query_lower = query.lower()
                has_refinement = any(keyword in query_lower for keyword in refinement_keywords)
                has_spec = any(keyword in query_lower for keyword in spec_keywords)
                
                if has_refinement or has_spec:
                    is_refinement = True
                    previous_products = last_turn.get('products', [])
                    # Combine with previous context for better search
                    previous_query = last_turn.get('query', '')
                    enhanced_query = f"{previous_query} {query}"
                    logger.info(f"🔍 Refinement detected: '{query}' refining previous '{previous_query}'")
                    logger.info(f"📦 Previous results available: {len(previous_products)} products")

        query_analysis = conversation_manager.process_query(current_user, enhanced_query)
        action = query_analysis.get('action')

        logger.info(f"📋 Query action determined: {action}")

        # Handle clarification request (vague query)
        if action == 'clarify':
            clarification_message = query_analysis.get('message')
            
            # Save clarification to memory so we can reference it later
            conversation_manager.save_conversation_turn(
                user_id=current_user,
                query=query,
                products=[],  # No products for clarification
                answer=clarification_message,
                action='clarify'
            )
            
            return jsonify({
                'success': True,
                'action': 'clarify',
                'answer': clarification_message,
                'products': [],
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'user': current_user,
                'needs_clarification': True
            }), 200

        # Handle product detail request (e.g., "give me its specs")
        if action == 'product_detail':
            previous_products = query_analysis.get('previous_products', [])
            previous_query = query_analysis.get('previous_query', '')
            reference_type = query_analysis.get('reference_type')
            
            logger.info(f"📊 Showing details for {len(previous_products)} previous products")
            
            if not previous_products:
                return jsonify({
                    'success': True,
                    'action': 'clarify',
                    'answer': "I don't have any previous products to show details for. Could you please search for a product first?",
                    'products': [],
                    'query': query,
                    'timestamp': datetime.now().isoformat(),
                    'user': current_user
                }), 200
            
            # Try to narrow down to the specific product(s) the user referred to
            # in the previous message. For example, if the user said something like
            # "I prefer the Samsung Galaxy A16 at 15,699" and then asks
            # "does it have warranty", we want to focus on that particular option
            # instead of all previous products.

            products = previous_products[:3]

            try:
                pq_lower = (previous_query or '').lower().strip()
                matched_products = []

                if pq_lower and previous_products:
                    for prod in previous_products:
                        name = (prod.get('name') or '').lower()
                        title = (prod.get('title') or '').lower()

                        # Simple substring heuristics: if the previous user
                        # message contains the product name or title, treat it
                        # as the selected option.
                        if name and name in pq_lower:
                            matched_products.append(prod)
                            continue
                        if title and title in pq_lower:
                            matched_products.append(prod)

                # If we found one or more clear matches, prefer those products
                # for the detail view. This makes follow-up questions like
                # "does it have warranty" apply to the explicitly chosen option.
                if matched_products:
                    products = matched_products[:3]
                    logger.info(
                        f"🎯 Narrowed product_detail to {len(products)} product(s) based on previous_query match"
                    )

            except Exception as e:
                # If anything goes wrong, fall back gracefully to the original
                # top-3 behavior.
                logger.warning(f"⚠️ Failed to refine product_detail selection from previous_query: {e}")
            
            # Generate detailed response focusing on specs
            try:
                products_text = "\n\n".join([
                    f"{i+1}. {p.get('name')} - {p.get('price_text', 'N/A')}\n" +
                    f"   Rating: {p.get('rating', 'N/A')} | URL: {p.get('url', 'N/A')}\n" +
                    f"   Specs: {p.get('specs', {})}\n" +
                    f"   Description: {p.get('description', 'N/A')}"
                    for i, p in enumerate(products)
                ])
                
                detail_prompt = f"""The user previously searched for: "{previous_query}"

They are now asking: "{query}"

Provide detailed specifications and features for these products:

{products_text}

Focus on technical specifications, features, and what makes each product unique. Be comprehensive and informative."""

                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful shopping assistant. Provide detailed product specifications and features when asked."},
                        {"role": "user", "content": detail_prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                
                answer = response.choices[0].message.content.strip()
                logger.info("✅ Generated product detail response")
                
            except Exception as e:
                logger.error(f"❌ Detail generation failed: {e}")
                answer = "Here are the detailed specifications for the products I showed you earlier. Please check the product information below."
            
            # Format products for response
            formatted_products = []
            for product in products:
                formatted_products.append({
                    'id': product.get('id'),
                    'name': product.get('name'),
                    'title': product.get('title', product.get('name')),
                    'price_text': product.get('price_text'),
                    'price_numeric': product.get('price_numeric'),
                    'rating': product.get('rating'),
                    'url': product.get('url'),
                    'image_url': product.get('image_url'),
                    'description': product.get('description', ''),
                    'specs': product.get('specs', {}),
                    'similarity_score': round(product.get('similarity_score', 0), 3)
                })
            
            # Save to conversation
            conversation_manager.save_conversation_turn(
                user_id=current_user,
                query=query,
                products=products,
                answer=answer,
                action='product_detail'
            )
            
            return jsonify({
                'success': True,
                'action': 'product_detail',
                'answer': answer,
                'products': formatted_products,
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'user': current_user,
                'results_count': len(formatted_products)
            }), 200

        # Handle direct product comparison (e.g., "compare iphone 12 and iphone 11")
        if action == 'direct_compare':
            product1_name = query_analysis.get('product1')
            product2_name = query_analysis.get('product2')
            
            logger.info(f"🔍 Searching for product 1: {product1_name}")
            logger.info(f"🔍 Searching for product 2: {product2_name}")
            
            try:
                # Search for both products
                products1 = pipeline.hybrid_search(product1_name, k=3, score_threshold=0.0)
                products2 = pipeline.hybrid_search(product2_name, k=3, score_threshold=0.0)
                
                logger.info(f"✅ Found {len(products1)} results for '{product1_name}'")
                logger.info(f"✅ Found {len(products2)} results for '{product2_name}'")
                
                # Combine both product lists
                all_products = products1 + products2
                
                # Generate comparison response
                answer = conversation_manager.generate_comparison_response(
                    current_products=products1,
                    current_query=query,
                    comparison_type='direct',
                    previous_products=products2,
                    previous_query=product2_name
                )
                logger.info("✅ Generated direct comparison response")
                
                # Format products
                formatted_products = []
                for product in all_products:
                    formatted_products.append({
                        'name': product.get('name', 'Unknown Product'),
                        'price': product.get('price', 'N/A'),
                        'rating': product.get('rating', 'N/A'),
                        'reviews': product.get('reviews_count', 0),
                        'url': product.get('url', ''),
                        'image_url': product.get('image_url', ''),
                        'specs': product.get('key_specs', '')
                    })
                
                # Save to conversation memory using the standard helper
                conversation_manager.save_conversation_turn(
                    user_id=current_user,
                    query=query,
                    products=all_products,
                    answer=answer,
                    action='compare'
                )
                
                return jsonify({
                    'success': True,
                    'action': 'compare',
                    'answer': answer,
                    'products': formatted_products,
                    'query': query,
                    'results_count': len(all_products),
                    'timestamp': datetime.now().isoformat(),
                    'user': current_user
                }), 200
                
            except Exception as e:
                logger.error(f"❌ Direct comparison failed: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Failed to perform comparison'
                }), 500

       
        # Perform semantic search (SKIP if in comparison mode with previous products)
        # Perform semantic search
        comparison_type = query_analysis.get('comparison_type')

        # SKIP search only for explicit/direct comparisons that just reuse previous
        # products or the explicit options that were highlighted in the last turn.
        if (
            action == 'compare'
            and query_analysis.get('previous_products')
            and comparison_type in ['explicit', 'direct']
        ):
            # Prefer the explicit "options" (typically top 2-3) stored from the
            # last assistant answer when the user says things like
            # "compare the two options" or "compare the three options".
            last_options = query_analysis.get('last_options') or []
            comparison_count = query_analysis.get('comparison_count')

            if comparison_count:
                # User specified how many products to compare
                if last_options and len(last_options) >= comparison_count:
                    products = last_options[:comparison_count]
                    logger.info(
                        f"🔄 Comparing {len(products)} products from last_options as requested by user"
                    )
                else:
                    # Fallback: slice previous_products if we don't have enough
                    products = query_analysis.get('previous_products', [])[:comparison_count]
                    logger.info(
                        f"🔄 Comparing {len(products)} previous products as requested by user (no suitable last_options)"
                    )
            else:
                # No specific count requested – compare all available products
                if last_options:
                    products = last_options[:5]
                    logger.info(
                        f"🔄 Comparing {len(products)} products from last_options (no explicit count)"
                    )
                else:
                    products = query_analysis.get('previous_products', [])[:5]
                    logger.info(
                        f"🔄 Comparing all {len(products)} previous products (no explicit count, no last_options)"
                    )
        else:
            try:
                # Use enhanced query for search if it was combined with context
                search_query = enhanced_query if enhanced_query != query else query
                products = pipeline.hybrid_search(search_query, k=10, score_threshold=0.0)  # Hybrid search with improved ranking
                logger.info(f"✅ Found {len(products)} products for query: '{search_query}'")
                
                # Apply brand filtering if specific brand mentioned
                brands = ['samsung', 'xiaomi', 'apple', 'iphone', 'oppo', 'tecno', 'infinix', 'nokia', 'huawei', 'itel', 'realme', 'vivo']
                query_lower = search_query.lower()
                mentioned_brand = None
                
                for brand in brands:
                    if brand in query_lower:
                        mentioned_brand = brand
                        logger.info(f"🏷️ Brand detected in query: {brand}")
                        break
                
                # Filter products by brand if mentioned
                if mentioned_brand:
                    brand_filtered = []
                    other_products = []
                    
                    for product in products:
                        product_name = product.get('name', '').lower()
                        product_title = product.get('title', '').lower()
                        
                        if mentioned_brand in product_name or mentioned_brand in product_title:
                            brand_filtered.append(product)
                        else:
                            other_products.append(product)
                    
                    # Prioritize brand matches, but include others if not enough matches
                    if len(brand_filtered) > 0:
                        products = brand_filtered  # Use brand filtered products
                        logger.info(f"✅ Filtered to {len(products)} {mentioned_brand} products")
                    else:
                        logger.warning(f"⚠️ No exact {mentioned_brand} matches found, returning similar products")
                
                # Apply storage/RAM filtering if specified
                import re

                # Detect storage specification (e.g., "256 GB", "512GB", "256 ROM", "1TB")
                storage_match = re.search(r'(\d+)\s*(gb|tb|GB|TB|rom|ROM)(?!\s*ram)', query_lower)
                ram_match = re.search(r'(\d+)\s*(gb|GB)\s*(ram|RAM)', query_lower)

                if storage_match:
                    storage_value = storage_match.group(1)
                    storage_unit = storage_match.group(2).upper()
                    # Normalize ROM to GB for consistency
                    if storage_unit == 'ROM':
                        storage_unit = 'GB'
                    storage_spec = f"{storage_value}{storage_unit}"

                    logger.info(f"💾 Detected storage specification: {storage_spec} (from user query)")

                    spec_filtered = []
                    for product in products:
                        product_text = (
                            product.get('name', '') + ' ' +
                            product.get('title', '') + ' ' +
                            product.get('description', '') + ' ' +
                            str(product.get('specs', {}))
                        ).lower()

                        # Check if the exact storage is mentioned
                        if storage_spec.lower() in product_text or f"{storage_value} {storage_unit.lower()}" in product_text:
                            spec_filtered.append(product)

                    if len(spec_filtered) > 0:
                        products = spec_filtered
                        logger.info(f"✅ Filtered to {len(products)} products with {storage_spec} storage")
                    else:
                        # No products match the storage spec - clear products list
                        logger.warning(f"⚠️ No products with {storage_spec} storage found")
                        products = []

                if ram_match:
                    ram_value = ram_match.group(1)
                    ram_spec = f"{ram_value}GB"

                    logger.info(f"💻 Detected RAM specification: {ram_spec} RAM")

                    spec_filtered = []
                    for product in products:
                        product_text = (
                            product.get('name', '') + ' ' +
                            product.get('title', '') + ' ' +
                            product.get('description', '') + ' ' +
                            str(product.get('specs', {}))
                        ).lower()

                        # Check if the exact RAM is mentioned
                        if (
                            f"{ram_spec.lower()} ram" in product_text or
                            f"{ram_value}gb ram" in product_text or
                            f"{ram_value} gb ram" in product_text
                        ):
                            spec_filtered.append(product)

                    if len(spec_filtered) > 0:
                        products = spec_filtered
                        logger.info(f"✅ Filtered to {len(products)} products with {ram_spec} RAM")
                    else:
                        # No products match the RAM spec - clear products list
                        logger.warning(f"⚠️ No products with {ram_spec} RAM found")
                        products = []

                # Apply refinement filtering whenever we have previous products
                previous_products = query_analysis.get('previous_products', [])
                if previous_products:

                    # Filter out previously shown products if user wants "better", "other", or "different" options
                    if (
                        'better' in query_lower or
                        'other' in query_lower or
                        'different' in query_lower or
                        'more options' in query_lower or
                        'alternatives' in query_lower
                    ):
                        previous_product_ids = {p.get('id') for p in previous_products if p.get('id')}

                        # Filter out products already shown
                        new_products = [p for p in products if p.get('id') not in previous_product_ids]

                        if len(new_products) > 0:
                            products = new_products
                            logger.info(
                                f"✅ Filtered out {len(previous_product_ids)} previously shown products, "
                                f"showing {len(products)} new alternatives"
                            )
                        else:
                            logger.warning("⚠️ No new alternatives found - all current results were already shown")

                    # Cheaper options
                    if 'cheaper' in query_lower or 'less expensive' in query_lower or 'lower price' in query_lower:
                        previous_prices = []
                        for prod in previous_products:
                            price_text = prod.get('price_text', '')
                            price_match = re.search(r'[Kk][Ss][Hh]?\s*([\d,]+)', price_text)
                            if price_match:
                                price_num = int(price_match.group(1).replace(',', ''))
                                previous_prices.append(price_num)

                        if previous_prices:
                            max_previous_price = min(previous_prices)  # cheapest previous
                            logger.info(
                                f"💰 User wants cheaper option. Previous lowest price: KSh {max_previous_price}"
                            )

                            price_filtered = []
                            for product in products:
                                price_text = product.get('price_text', '')
                                price_match = re.search(r'[Kk][Ss][Hh]?\s*([\d,]+)', price_text)
                                if price_match:
                                    price_num = int(price_match.group(1).replace(',', ''))
                                    if price_num < max_previous_price:
                                        price_filtered.append(product)

                            if len(price_filtered) > 0:
                                products = price_filtered
                                logger.info(
                                    f"✅ Filtered to {len(products)} cheaper products (< KSh {max_previous_price})"
                                )
                            else:
                                logger.warning(
                                    "⚠️ No cheaper options found. Previous recommendation already the cheapest."
                                )
                                products = []

                    # More expensive options
                    elif (
                        'more expensive' in query_lower or
                        'higher price' in query_lower or
                        'pricier' in query_lower
                    ):
                        previous_prices = []
                        for prod in previous_products:
                            price_text = prod.get('price_text', '')
                            price_match = re.search(r'[Kk][Ss][Hh]?\s*([\d,]+)', price_text)
                            if price_match:
                                price_num = int(price_match.group(1).replace(',', ''))
                                previous_prices.append(price_num)

                        if previous_prices:
                            min_previous_price = max(previous_prices)  # most expensive previous
                            logger.info(
                                f"💰 User wants more expensive option. Previous highest price: KSh {min_previous_price}"
                            )

                            price_filtered = []
                            for product in products:
                                price_text = product.get('price_text', '')
                                price_match = re.search(r'[Kk][Ss][Hh]?\s*([\d,]+)', price_text)
                                if price_match:
                                    price_num = int(price_match.group(1).replace(',', ''))
                                    if price_num > min_previous_price:
                                        price_filtered.append(product)

                            if len(price_filtered) > 0:
                                products = price_filtered
                                logger.info(
                                    f"✅ Filtered to {len(products)} more expensive products (> KSh {min_previous_price})"
                                )
                            else:
                                logger.warning("⚠️ No more expensive options found in current results.")

                # Limit to top 5 results
                products = products[:5]
                    
            except Exception as e:
                logger.error(f"❌ Semantic search failed: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Search service temporarily unavailable'
                }), 500
        # Handle comparison mode
        if action == 'compare':
            comparison_type = query_analysis.get('comparison_type')
            attribute = query_analysis.get('attribute')  # None for full, specific for attribute-only
            previous_products = query_analysis.get('previous_products', [])
            previous_query = query_analysis.get('previous_query', '')

            # Special handling for price refinement queries with no results
            if comparison_type == 'refinement' and len(products) == 0:
                query_lower = query.lower()
                if 'cheaper' in query_lower or 'less expensive' in query_lower:
                    # Extract the brand if mentioned
                    brand_mentioned = None
                    brands = ['samsung', 'xiaomi', 'apple', 'iphone', 'oppo', 'tecno', 'infinix', 'nokia', 'huawei', 'itel', 'realme', 'vivo']
                    for brand in brands:
                        if brand in query_lower:
                            brand_mentioned = brand.capitalize()
                            break
                    
                    # Get the cheapest previous product price
                    import re
                    cheapest_price = None
                    cheapest_product = None
                    for prod in previous_products:
                        price_text = prod.get('price_text', '')
                        price_match = re.search(r'[Kk][Ss][Hh]?\s*([\d,]+)', price_text)
                        if price_match:
                            price_num = int(price_match.group(1).replace(',', ''))
                            if cheapest_price is None or price_num < cheapest_price:
                                cheapest_price = price_num
                                cheapest_product = prod
                    
                    if cheapest_product and brand_mentioned:
                        answer = f"The {cheapest_product.get('name', 'product')} at {cheapest_product.get('price_text', 'KSh ' + str(cheapest_price))} is already the most affordable {brand_mentioned} option I have in our database. There are no cheaper {brand_mentioned} products available.\n\nWould you like me to:\n1. Show you products from other brands at a lower price?\n2. Show you more {brand_mentioned} products with better features at similar prices?\n3. Help you find something else?"
                    elif cheapest_product:
                        answer = f"The {cheapest_product.get('name', 'product')} at {cheapest_product.get('price_text', 'KSh ' + str(cheapest_price))} is already the most affordable option matching your criteria. There are no cheaper alternatives available.\n\nWould you like me to show you products with better features at similar prices, or help you search for something different?"
                    else:
                        answer = "I couldn't find any cheaper options than what I previously recommended. That appears to be the best budget option available. Would you like me to help you find something else?"
                    
                    logger.info("🚫 No cheaper products available - sent helpful message")
                    # Don't save empty products to memory for this case
                elif 'more expensive' in query_lower or 'higher price' in query_lower:
                    answer = "I couldn't find more expensive options matching your criteria in the current results. Would you like me to search for premium alternatives or help you with something else?"
                    logger.info("� No more expensive products available")
                else:
                    answer = "I couldn't find products matching your refined criteria. Could you try adjusting your requirements or let me know what specific features you're looking for?"
            elif attribute:
                logger.info(f"�🔄 Generating {comparison_type} comparison for attribute: {attribute}")
                try:
                    answer = conversation_manager.generate_comparison_response(
                        current_products=products,
                        current_query=query,
                        comparison_type=comparison_type,
                        previous_products=previous_products,
                        previous_query=previous_query,
                        attribute=attribute
                    )
                    logger.info("✅ Generated comparison response")
                except Exception as e:
                    logger.error(f"❌ Comparison generation failed: {e}")
                    answer = "I found some new products for you. Let me know if you'd like me to compare them with your previous results."
            else:
                logger.info(f"🔄 Generating {comparison_type} full comparison")
                try:
                    answer = conversation_manager.generate_comparison_response(
                        current_products=products,
                        current_query=query,
                        comparison_type=comparison_type,
                        previous_products=previous_products,
                        previous_query=previous_query,
                        attribute=attribute
                    )
                    logger.info("✅ Generated comparison response")
                except Exception as e:
                    logger.error(f"❌ Comparison generation failed: {e}")
                    answer = "I found some new products for you. Let me know if you'd like me to compare them with your previous results."

        # Handle regular search
        else:
            # Check if products list is empty after all filtering (brand, specs, price)
            if len(products) == 0:
                import re
                # Check if storage spec was requested
                storage_match = re.search(r'(\d+)\s*(gb|tb|GB|TB|rom|ROM)(?!\s*ram)', query_lower)
                ram_match = re.search(r'(\d+)\s*(gb|GB)\s*(ram|RAM)', query_lower)
                
                if storage_match:
                    storage_value = storage_match.group(1)
                    storage_unit = storage_match.group(2).upper()
                    if storage_unit == 'ROM':
                        storage_unit = 'GB'
                    
                    # Extract brand if mentioned
                    brand_mentioned = None
                    brands = ['samsung', 'xiaomi', 'apple', 'iphone', 'oppo', 'tecno', 'infinix', 'nokia', 'huawei', 'itel', 'realme', 'vivo']
                    for brand in brands:
                        if brand in query_lower:
                            brand_mentioned = brand.capitalize()
                            break
                    
                    if brand_mentioned:
                        answer = (
                            f"I couldn't find any {brand_mentioned} phones with {storage_value}{storage_unit} storage in our database.\n\n"
                            f"Would you like me to:\n"
                            f"1. Show you {brand_mentioned} phones with different storage capacities (64GB, 128GB)?\n"
                            f"2. Show you phones from other brands with {storage_value}{storage_unit} storage?\n"
                            f"3. Help you find something else?"
                        )
                    else:
                        answer = (
                            f"I couldn't find any phones with {storage_value}{storage_unit} storage matching your criteria. "
                            f"Would you like me to show you phones with different storage capacities or adjust your search?"
                        )
                    
                    logger.info(f"🚫 No products found with {storage_value}{storage_unit} storage - sent helpful message")
                elif ram_match:
                    ram_value = ram_match.group(1)
                    answer = (
                        f"I couldn't find any phones with {ram_value}GB RAM matching your criteria. "
                        f"Would you like me to show you phones with different RAM capacities or adjust your search?"
                    )
                    logger.info(f"🚫 No products found with {ram_value}GB RAM - sent helpful message")
                else:
                    answer = (
                        "I couldn't find any products matching your search criteria. Could you try:\n"
                        "1. Broadening your search (remove specific specs)\n"
                        "2. Trying a different brand\n"
                        "3. Adjusting your budget\n\n"
                        "What would you like to do?"
                    )
            else:
                # Try to answer direct product attribute questions (rating, price, etc.) without GPT
                direct_answer = answer_direct_product_question(query, products)
                if direct_answer:
                    answer = direct_answer
                    logger.info("✅ Answered direct product attribute question without GPT")
                else:
                    try:
                        # Get conversation history for context
                        history = conversation_manager.memory.get_conversation_history(current_user)
                        
                        # Build context from recent conversation (last 2 turns)
                        context = None
                        if history and len(history) > 0:
                            recent_turns = history[-2:]  # Last 2 turns
                            context_parts = []
                            for turn in recent_turns:
                                context_parts.append(f"User asked: '{turn['query']}'")
                                if turn.get('answer'):
                                    # Truncate long answers
                                    answer_preview = (
                                        turn['answer'][:150] + "..."
                                        if len(turn['answer']) > 150
                                        else turn['answer']
                                    )
                                    context_parts.append(f"Bot responded: '{answer_preview}'")
                            context = "\n".join(context_parts)
                        
                        # Normal GPT conversational response
                        answer = generate_conversational_response(query, products, context)
                        logger.info(" Generated conversational response")
                    except Exception as e:
                        logger.error(f" Response generation failed: {e}")
                        if products:
                            answer = (
                                f"I found {len(products)} products that might interest you. "
                                f"Here are the top recommendations based on your search."
                            )
                        else:
                            answer = (
                                "I couldn't find any products matching your query. "
                                "Could you try rephrasing your search?"
                            )

        # Apply live status filter so we do not recommend products that are
        # known to be unavailable (unknown/not_found/out_of_stock).
        products = filter_products_by_live_status(products)

        # Format products
        formatted_products = []
        for product in products:
            formatted_products.append({
                'id': product.get('id'),
                'name': product.get('name'),
                'title': product.get('title', product.get('name')),
                'price_text': product.get('price_text'),
                'price_numeric': product.get('price_numeric'),
                'rating': product.get('rating'),
                'url': product.get('url'),
                'image_url': product.get('image_url'),
                'description': product.get('description', '')[:200] + "..." if len(product.get('description', '')) > 200 else product.get('description', ''),
                'specs': product.get('specs', {}),
                'similarity_score': round(product.get('similarity_score', 0), 3)
            })

        # Save conversation turn to memory
        conversation_manager.save_conversation_turn(
            user_id=current_user,
            query=query,
            products=products,  # Store original products with all metadata
            answer=answer,
            action=action
        )

        # Build response
        response_data = {
            'success': True,
            'action': action,
            'answer': answer,
            'products': formatted_products,
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'user': current_user,
            'results_count': len(formatted_products)
        }

        # Add comparison context if applicable
        if action == 'compare':
            response_data['comparison'] = {
                'type': comparison_type,
                'previous_query': previous_query
            }

        logger.info(f"✅ Chat response generated successfully")
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f" Chat endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': 'Chat request failed',
            'timestamp': datetime.now().isoformat()
        }), 500


@chat_bp.route('/search', methods=['POST'])
@jwt_required()
def search():
    """
    Direct semantic search endpoint (without conversational response).
    
    Expected JSON payload:
    {
        "query": "Samsung smartphone",
        "k": 5,
        "score_threshold": 0.0
    }
    """
    try:
        current_user = get_jwt_identity()
        
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        query = data.get('query', '').strip()
        k = data.get('k', 5)
        score_threshold = data.get('score_threshold', 0.0)
        
        if not query:
            return jsonify({'success': False, 'error': 'Query is required'}), 400
        
        # Validate parameters
        k = max(1, min(k, 20))  # Limit between 1 and 20
        score_threshold = max(0.0, min(score_threshold, 1.0))  # Between 0 and 1
        
        # Perform search with hybrid ranking
        products = pipeline.hybrid_search(query, k=k, score_threshold=score_threshold)
        
        # Format response
        formatted_products = []
        for product in products:
            formatted_product = {
                'id': product.get('id'),
                'name': product.get('name'),
                'price_text': product.get('price_text'),
                'url': product.get('url'),
                'image_url': product.get('image_url'),
                'rating': product.get('rating'),
                'similarity_score': round(product.get('similarity_score', 0), 3)
            }
            formatted_products.append(formatted_product)
        
        return jsonify({
            'success': True,
            'products': formatted_products,
            'query': query,
            'results_count': len(formatted_products),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Search endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': 'Search failed',
            'timestamp': datetime.now().isoformat()
        }), 500

@chat_bp.route('/chats', methods=['GET'])
@jwt_required()
def get_all_chats():
    """Get all chat sessions for the current user."""
    try:
        current_user = get_jwt_identity()
        
        if not conversation_manager:
            return jsonify({
                'success': False,
                'error': 'Conversation manager not available'
            }), 503
        
        sessions = conversation_manager.memory.get_all_sessions(current_user)
        
        return jsonify({
            'success': True,
            'sessions': sessions,
            'count': len(sessions),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting chat sessions: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@chat_bp.route('/chats/<int:session_id>', methods=['GET'])
@jwt_required()
def get_chat_session(session_id):
    """Get all messages from a specific chat session."""
    try:
        current_user = get_jwt_identity()
        
        if not conversation_manager:
            return jsonify({
                'success': False,
                'error': 'Conversation manager not available'
            }), 503
        
        # Verify session belongs to user
        session = db.session.get(ChatSession, session_id)
        if not session or session.user_email != current_user:
            return jsonify({
                'success': False,
                'error': 'Session not found or access denied'
            }), 404
        
        messages = conversation_manager.memory.get_session_messages(session_id)
        
        return jsonify({
            'success': True,
            'session': session.to_dict(),
            'messages': messages,
            'count': len(messages),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting chat session: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@chat_bp.route('/chats', methods=['POST'])
@jwt_required()
def create_chat():
    """Create a new chat session."""
    try:
        current_user = get_jwt_identity()
        data = request.get_json() or {}
        title = data.get('title', 'New Chat')
        
        # Create new session
        new_session = ChatSession(
            user_email=current_user,
            title=title,
            is_active=True
        )
        db.session.add(new_session)
        db.session.commit()
        
        logger.info(f"✅ New chat session created for user: {current_user}")
        
        return jsonify({
            'success': True,
            'message': 'New chat session created',
            'session': new_session.to_dict(),
            'timestamp': datetime.now().isoformat()
        }), 201
        
    except Exception as e:
        logger.error(f"❌ Error creating chat session: {e}")
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@chat_bp.route('/chats/<int:session_id>', methods=['DELETE'])
@jwt_required()
def delete_chat_session(session_id):
    """Delete a chat session."""
    try:
        current_user = get_jwt_identity()
        
        # Verify session belongs to user
        session = db.session.get(ChatSession, session_id)
        if not session or session.user_email != current_user:
            return jsonify({
                'success': False,
                'error': 'Session not found or access denied'
            }), 404
        
        # Delete session (cascade will delete messages)
        db.session.delete(session)
        db.session.commit()
        
        logger.info(f"✅ Deleted chat session {session_id} for user: {current_user}")
        
        return jsonify({
            'success': True,
            'message': 'Chat session deleted',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error deleting chat session: {e}")
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@chat_bp.route('/conversation/history', methods=['GET'])
@jwt_required()
def get_conversation_history():
    """Get conversation history for current user."""
    try:
        current_user = get_jwt_identity()
        
        if not conversation_manager:
            return jsonify({
                'success': False,
                'error': 'Conversation manager not available'
            }), 503
        
        history = conversation_manager.memory.get_conversation_history(current_user)
        
        return jsonify({
            'success': True,
            'history': history,
            'count': len(history),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting conversation history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@chat_bp.route('/conversation/clear', methods=['POST'])
@jwt_required()
def clear_conversation():
    """Clear conversation history for current user."""
    try:
        current_user = get_jwt_identity()
        
        if not conversation_manager:
            return jsonify({
                'success': False,
                'error': 'Conversation manager not available'
            }), 503
        
        conversation_manager.memory.clear_session(current_user)
        logger.info(f"✅ Conversation cleared for user: {current_user}")
        
        return jsonify({
            'success': True,
            'message': 'Conversation history cleared',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error clearing conversation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@chat_bp.route('/status', methods=['GET'])
def status():
    """Get RAG system status."""
    try:
        # Check pipeline status
        pipeline_status = "healthy"
        collection_stats = {}
        
        try:
            collection_stats = pipeline.get_collection_stats()
        except:
            pipeline_status = "error"
        
        # Check OpenAI status
        openai_status = "healthy" if openai_client else "error"
        
        # Check conversation manager
        conversation_status = "healthy" if conversation_manager else "not_initialized"
        
        return jsonify({
            'success': True,
            'status': {
                'pipeline': pipeline_status,
                'openai': openai_status,
                'conversation_manager': conversation_status,
                'collection': collection_stats
            },
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

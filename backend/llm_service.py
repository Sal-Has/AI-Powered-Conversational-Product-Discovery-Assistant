import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    """
    LLM service for generating conversational product recommendations
    using OpenAI's GPT models in a RAG pipeline.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini",
                 max_tokens: int = 1000,
                 temperature: float = 0.7):
        """
        Initialize the LLM service.
        
        Args:
            api_key: OpenAI API key (optional, will use OPENAI_API_KEY env var)
            model: OpenAI model to use
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0.0-1.0)
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Load API key from environment or parameter
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Simple client initialization without extra parameters
        self.client = OpenAI(api_key=api_key)
        logger.info(f"✅ LLM service initialized with model: {model}")
    
    def generate_shopping_recommendation(self, 
                                       user_query: str, 
                                       retrieved_products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a conversational shopping recommendation using retrieved products.
        
        Args:
            user_query: User's original search query
            retrieved_products: List of products from semantic search
            
        Returns:
            Dictionary with LLM response and metadata
        """
        if not retrieved_products:
            return self._generate_no_results_response(user_query)
        
        # Create the prompt
        prompt = self._create_shopping_prompt(user_query, retrieved_products)
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            llm_response = response.choices[0].message.content.strip()
            
            # Prepare response with both LLM text and structured data
            result = {
                "success": True,
                "query": user_query,
                "llm_response": llm_response,
                "products": retrieved_products,
                "product_count": len(retrieved_products),
                "model_used": self.model,
                "generated_at": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Generated recommendation for query: '{user_query}'")
            return result
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            return {
                "success": False,
                "query": user_query,
                "error": str(e),
                "llm_response": "I apologize, but I'm having trouble generating a recommendation right now. Please try again later.",
                "products": retrieved_products,
                "product_count": len(retrieved_products),
                "generated_at": datetime.now().isoformat()
            }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the shopping assistant."""
        return """You are a helpful and knowledgeable shopping assistant for Jumia Kenya, an e-commerce platform. Your role is to:

1. Analyze user queries and recommend the most suitable products from the provided list
2. Explain why each recommended product fits the user's needs
3. Highlight key features, specifications, and value propositions
4. Compare products when multiple options are available
5. Provide practical shopping advice and considerations
6. Use a friendly, conversational tone that builds trust

Guidelines:
- Always mention specific product names, prices, and key features
- Explain the reasoning behind your recommendations
- If products have different price ranges, explain the value at each level
- Mention any standout features or specifications
- Be honest about trade-offs between different options
- End with a clear recommendation or next steps
- Keep responses concise but informative (aim for 150-300 words)
- Use Kenyan Shillings (KSh) for prices"""
    
    def _create_shopping_prompt(self, user_query: str, products: List[Dict[str, Any]]) -> str:
        """Create the prompt with user query and retrieved products."""
        
        prompt_parts = [
            f"User Query: {user_query}",
            "",
            "Retrieved Products:"
        ]
        
        for i, product in enumerate(products, 1):
            # Extract key information
            name = product.get('name', 'Unknown Product')
            price = product.get('price_text', 'Price not available')
            description = product.get('description', '')[:200] + "..." if len(product.get('description', '')) > 200 else product.get('description', '')
            rating = product.get('rating', 'No rating')
            similarity_score = product.get('similarity_score', 0)
            
            # Parse specs if available
            specs_info = ""
            try:
                specs = product.get('specs', {})
                if isinstance(specs, str):
                    specs = json.loads(specs)
                if specs and isinstance(specs, dict):
                    key_specs = []
                    for key, value in list(specs.items())[:3]:  # Top 3 specs
                        if key and value:
                            key_specs.append(f"{key}: {value}")
                    if key_specs:
                        specs_info = f" | Specs: {', '.join(key_specs)}"
            except:
                pass
            
            product_info = f"""
{i}. {name}
   Price: {price}
   Rating: {rating} | Relevance: {similarity_score:.2f}
   Description: {description}{specs_info}
   URL: {product.get('url', '')}
"""
            prompt_parts.append(product_info)
        
        prompt_parts.extend([
            "",
            "Please provide a helpful recommendation based on the user's query and these products. Focus on the most relevant options and explain why they would be good choices."
        ])
        
        return "\n".join(prompt_parts)
    
    def _generate_no_results_response(self, user_query: str) -> Dict[str, Any]:
        """Generate response when no products are found."""
        return {
            "success": True,
            "query": user_query,
            "llm_response": f"I couldn't find any products matching '{user_query}' in our current inventory. You might want to try different search terms or browse our categories. For example, try searching for broader terms like 'smartphone', 'laptop', or 'headphones'.",
            "products": [],
            "product_count": 0,
            "model_used": self.model,
            "generated_at": datetime.now().isoformat()
        }
    
    def generate_product_comparison(self, 
                                  products: List[Dict[str, Any]], 
                                  comparison_criteria: str = "features and value") -> Dict[str, Any]:
        """
        Generate a detailed comparison between multiple products.
        
        Args:
            products: List of products to compare
            comparison_criteria: What to focus on in comparison
            
        Returns:
            Dictionary with comparison response
        """
        if len(products) < 2:
            return {
                "success": False,
                "error": "At least 2 products required for comparison",
                "llm_response": "I need at least two products to make a comparison.",
                "products": products
            }
        
        prompt = f"""Compare these products based on {comparison_criteria}:

"""
        
        for i, product in enumerate(products, 1):
            name = product.get('name', 'Unknown Product')
            price = product.get('price_text', 'Price not available')
            description = product.get('description', '')[:150] + "..." if len(product.get('description', '')) > 150 else product.get('description', '')
            
            prompt += f"{i}. {name} - {price}\n   {description}\n\n"
        
        prompt += "Provide a detailed comparison highlighting the strengths and weaknesses of each product, and recommend which one offers the best value for different user needs."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return {
                "success": True,
                "llm_response": response.choices[0].message.content.strip(),
                "products": products,
                "comparison_criteria": comparison_criteria,
                "model_used": self.model,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Comparison generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "llm_response": "I'm having trouble generating a comparison right now. Please try again later.",
                "products": products
            }

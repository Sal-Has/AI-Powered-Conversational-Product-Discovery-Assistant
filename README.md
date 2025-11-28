# AI-Powered Conversational Product Discovery Assistant

An end-to-end AI assistant that helps users discover products through natural language conversation.

The system combines:

- **RAG (Retrieval-Augmented Generation)** over a product index
- **Flask** backend with JWT authentication
- **React** frontend chat interface
- **ChromaDB** vector store for product embeddings
- **Web scraping pipeline** for building the product catalog

---

## Features

- **Conversational product discovery**
  - Ask free-form questions about products (e.g. "Best budget phones under $300 with good battery").
  - Assistant interprets vague queries and asks clarification questions.
  - Supports follow‑up questions with short‑term **conversation memory**.

- **Product comparison**
  - Detects comparison intent (e.g. "compare these two", "which is better").
  - Returns structured comparisons (price, features, battery, camera, performance, ratings, and value).

- **RAG-based retrieval**
  - Products stored in **ChromaDB** with rich metadata (name, price, rating, image URL, link, etc.).
  - Backend retrieves relevant products and passes them to the LLM.

- **Authentication (JWT)**
  - User registration and login.
  - Protected `/api/chat` endpoint with `Authorization: Bearer <token>` header.
  - Optional direct test chat endpoint (no auth) for debugging.

- **React frontend**
  - Chat-style UI with user/assistant messages.
  - Product recommendation cards with:
    - Image
    - Name
    - Price and rating (if available)
    - Link to product page
  - Loading states and error handling (e.g. invalid token, empty results).

---

## Project Structure

```text
Final_Project/
├── backend/
│   ├── app.py                 # Flask app entry point
│   ├── rag_routes.py          # RAG/chat API routes
│   ├── rag_pipeline.py        # RAG pipeline and ChromaDB integration
│   ├── models.py              # DB / models (e.g. users, product index)
│   ├── fix_openai_client.py   # OpenAI client / API handling
│   ├── test_rag_pipeline.py   # RAG testing script
│   └── ...                    # other backend modules
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatbotRAG.jsx      # Authenticated chat interface
│   │   │   ├── DirectChat.jsx      # Direct (no-auth) chat interface
│   │   │   ├── Login.jsx
│   │   │   ├── Register.jsx
│   │   │   └── ...
│   │   └── App.jsx
│   └── ...
└── README.md




import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ProductCard from './ProductCard';

const DirectChat = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! I'm your AI shopping assistant for Jumia Kenya. I can help you find smartphones and electronics based on your needs and budget. What are you looking for today?",
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || isTyping) return;

    const userMessage = {
      id: Date.now(),
      text: inputMessage,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const queryText = inputMessage;
    setInputMessage('');
    setIsTyping(true);
    setError(null);

    try {
      // Direct API call to localhost:5000 (no auth required)
      const response = await axios.post('http://localhost:5000/api/chat', {
        query: queryText
      }, {
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (response.data.success) {
        const botResponse = {
          id: Date.now() + 1,
          text: response.data.answer,
          sender: 'bot',
          timestamp: new Date(),
          products: response.data.products || [],
          query: response.data.query,
          results_count: response.data.results_count || 0
        };
        
        setMessages(prev => [...prev, botResponse]);
      } else {
        throw new Error(response.data.error || 'Failed to get response');
      }
    } catch (error) {
      console.error('Chat API Error:', error);
      
      let errorMessage = 'Sorry, I encountered an error while processing your request.';
      
      if (error.response?.status === 500) {
        errorMessage = 'The search service is temporarily unavailable. Please try again later.';
      } else if (error.message) {
        errorMessage = `Error: ${error.message}`;
      }
      
      const errorResponse = {
        id: Date.now() + 1,
        text: errorMessage,
        sender: 'bot',
        timestamp: new Date(),
        isError: true
      };
      
      setMessages(prev => [...prev, errorResponse]);
      setError(errorMessage);
    } finally {
      setIsTyping(false);
    }
  };

  const formatTime = (date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

const renderMessage = (message) => {
  const isUser = message.sender === 'user';
  const hasProducts = message.products && message.products.length > 0;
  
  return (
    <div key={message.id} className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-4xl ${isUser ? 'w-auto' : 'w-full'}`}>

        {/* Product Grid – shown ABOVE the text for bot messages */}
        {!isUser && hasProducts && (
          <div className="mb-3 space-y-3">
            <h4 className="text-sm font-medium text-gray-700 px-2">
              Product Recommendations:
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {message.products.map((product, index) => (
                <ProductCard key={product.id || index} product={product} />
              ))}
            </div>
          </div>
        )}

        {/* Message Bubble */}
        <div className={`px-4 py-2 rounded-lg ${
          isUser
            ? 'bg-blue-500 text-white rounded-br-none ml-12'
            : `${message.isError ? 'bg-red-50 border border-red-200 text-red-800' : 'bg-white text-gray-800 shadow-sm border border-gray-200'} rounded-bl-none`
        }`}>
          <p className="text-sm whitespace-pre-wrap">{message.text}</p>
          <p className={`text-xs mt-1 ${
            isUser ? 'text-blue-100' : message.isError ? 'text-red-600' : 'text-gray-500'
          }`}>
            {formatTime(message.timestamp)}
            {message.results_count > 0 && (
              <span className="ml-2">• {message.results_count} products found</span>
            )}
          </p>
        </div>

      </div>
    </div>
  );
};

  return (
    <div className="h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200 px-6 py-4 flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z" />
            </svg>
          </div>
          <div>
            <h1 className="text-xl font-semibold text-gray-800">AI Shopping Assistant</h1>
            <p className="text-sm text-gray-500">Direct connection to RAG backend - No auth required</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          {/* System Status Indicator */}
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${error ? 'bg-red-400' : 'bg-green-400'}`}></div>
            <span className="text-xs text-gray-500">
              {error ? 'Service Error' : 'Connected'}
            </span>
          </div>
        </div>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map(renderMessage)}
        
        {isTyping && (
          <div className="flex justify-start">
            <div className="bg-white text-gray-800 shadow-sm border border-gray-200 rounded-lg rounded-bl-none px-4 py-2 max-w-xs">
              <div className="flex items-center space-x-2">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
                <span className="text-xs text-gray-500">AI is thinking...</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <div className="bg-white border-t border-gray-200 p-4">
        <form onSubmit={handleSendMessage} className="flex space-x-4">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Ask me about smartphones, electronics, or specific products..."
            className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={isTyping}
            maxLength={500}
          />
          <button
            type="submit"
            disabled={!inputMessage.trim() || isTyping}
            className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg transition-colors duration-200 flex items-center space-x-2"
          >
            {isTyping ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Searching...</span>
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
                <span>Send</span>
              </>
            )}
          </button>
        </form>
        
        {/* Input Helper */}
        <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
          <span>Try: "Samsung phone under 30000" or "iPhone with good camera"</span>
          <span>{inputMessage.length}/500</span>
        </div>
      </div>
    </div>
  );
};

export default DirectChat;

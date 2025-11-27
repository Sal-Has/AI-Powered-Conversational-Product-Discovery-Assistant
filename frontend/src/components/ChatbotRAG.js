import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import { chatbotAPI } from '../services/api';
import ProductCard from './ProductCard';
import ChatSidebar from './ChatSidebar';
import { Menu } from 'lucide-react';
import './ChatbotRAG.css';

const ChatbotRAG = () => {
  // Chat messages state
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
  
  // Session management state
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  
  const messagesEndRef = useRef(null);
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  // Load chat sessions on mount
  useEffect(() => {
    loadChatSessions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Load messages when session changes
  useEffect(() => {
    if (currentSessionId) {
      loadSessionMessages(currentSessionId);
    }
  }, [currentSessionId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load all chat sessions
  const loadChatSessions = async () => {
    try {
      const response = await chatbotAPI.getChatSessions();
      if (response.success) {
        setSessions(response.sessions);
        // If no current session, create one
        if (response.sessions.length === 0) {
          await createNewChat();
        } else if (!currentSessionId) {
          // Load the most recent session
          setCurrentSessionId(response.sessions[0].id);
        }
      }
    } catch (error) {
      console.error('Failed to load chat sessions:', error);
    }
  };

  // Load messages for a specific session
  const loadSessionMessages = async (sessionId) => {
    try {
      const response = await chatbotAPI.getChatSession(sessionId);
      if (response.success) {
        const sessionMessages = response.messages || [];  
        // Convert to our message format
        const formattedMessages = sessionMessages.map(msg => ({
          id: msg.id,
          text: msg.content,  
          sender: msg.role,   
          timestamp: new Date(msg.created_at),  
          products: msg.products || []
        }));
        
        // Add welcome message if empty
        if (formattedMessages.length === 0) {
          setMessages([{
            id: 1,
            text: "Hello! I'm your AI shopping assistant for Jumia Kenya. I can help you find smartphones and electronics based on your needs and budget. What are you looking for today?",
            sender: 'bot',
            timestamp: new Date()
          }]);
        } else {
          setMessages(formattedMessages);
        }
      }
    } catch (error) {
      console.error('Failed to load session messages:', error);
    }
  };

  // Create new chat session
  const createNewChat = async () => {
    try {
      const response = await chatbotAPI.createChatSession();
      if (response.success) {
        setSessions(prev => [response.session, ...prev]);
        setCurrentSessionId(response.session.id);
        setMessages([{
          id: 1,
          text: "Hello! I'm your AI shopping assistant for Jumia Kenya. I can help you find smartphones and electronics based on your needs and budget. What are you looking for today?",
          sender: 'bot',
          timestamp: new Date()
        }]);
      }
    } catch (error) {
      console.error('Failed to create new chat:', error);
    }
  };

  // Select a chat session
  const handleSelectSession = (sessionId) => {
    setCurrentSessionId(sessionId);
  };

  // Delete a chat session
  const handleDeleteSession = async (sessionId) => {
    try {
      const response = await chatbotAPI.deleteChatSession(sessionId);
      if (response.success) {
        setSessions(prev => prev.filter(s => s.id !== sessionId));
        
        // If deleted current session, switch to another or create new
        if (sessionId === currentSessionId) {
          const remaining = sessions.filter(s => s.id !== sessionId);
          if (remaining.length > 0) {
            setCurrentSessionId(remaining[0].id);
          } else {
            await createNewChat();
          }
        }
      }
    } catch (error) {
      console.error('Failed to delete session:', error);
    }
  };

  // Save message to current session
  const saveMessageToSession = async (messageData) => {
    if (!currentSessionId) return;
    
    try {
      await chatbotAPI.addMessageToSession(currentSessionId, messageData);
      // Refresh sessions to update titles and timestamps
      loadChatSessions();
    } catch (error) {
      console.error('Failed to save message:', error);
    }
  };

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

    // Save user message to session
    await saveMessageToSession({
      sender: 'user',
      message: queryText,
      products: []
    });

    try {
      const response = await chatbotAPI.sendQuery(queryText);
      
      if (response.success) {
        const botResponse = {
          id: Date.now() + 1,
          text: response.answer,
          sender: 'bot',
          timestamp: new Date(),
          products: response.products || [],
          query: response.query,
          results_count: response.results_count || 0
        };
        
        setMessages(prev => [...prev, botResponse]);
        
        // Save bot response to session
        await saveMessageToSession({
          sender: 'bot',
          message: response.answer,
          products: response.products || []
        });
      } else {
        throw new Error(response.error || 'Failed to get response');
      }
    } catch (error) {
      console.error('Chat API Error:', error);
      
      let errorMessage = 'Sorry, I encountered an error while processing your request.';
      
      if (error.response?.status === 401) {
        errorMessage = 'Your session has expired. Please log in again.';
        setTimeout(() => {
          handleLogout();
        }, 2000);
      } else if (error.response?.status === 500) {
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

  const handleLogout = async () => {
    await logout();
    navigate('/login');
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
          {!isUser && hasProducts && (
            <div className="mb-3 space-y-3">
              <h4 className="text-sm font-medium text-gray-700 px-2">
                Product Recommendations:
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {message.products.map((product, index) => (
                  <ProductCard key={product.id || index} product={product} enableLiveCheck />
                ))}
              </div>
            </div>
          )}
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
    <div className="chat-container">
      {/* Sidebar */}
      <ChatSidebar
        sessions={sessions}
        currentSessionId={currentSessionId}
        onSelectSession={handleSelectSession}
        onNewChat={createNewChat}
        onDeleteSession={handleDeleteSession}
        isOpen={sidebarOpen}
      />

      {/* Main Chat Area */}
      <div className="chat-main">
        {/* Header */}
        <div className="chat-header">
          <div className="flex items-center space-x-4">
            <button 
              className="sidebar-toggle"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              <Menu size={20} />
            </button>
            <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-semibold text-gray-800">AI Shopping Assistant</h1>
              <p className="text-sm text-gray-500">Welcome, {user?.username}!</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${error ? 'bg-red-400' : 'bg-green-400'}`}></div>
              <span className="text-xs text-gray-500">
                {error ? 'Service Error' : 'Online'}
              </span>
            </div>
            
            <button onClick={handleLogout} className="logout-btn">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
              <span>Logout</span>
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="chat-messages">
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

        {/* Input */}
        <div className="chat-input-container">
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
          
          <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
            <span>Try: "Samsung phone under 30000" or "iPhone with good camera"</span>
            <span>{inputMessage.length}/500</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatbotRAG;
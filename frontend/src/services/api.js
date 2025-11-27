import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle token expiration
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      const refreshToken = localStorage.getItem('refresh_token');
      if (refreshToken) {
        try {
          const response = await axios.post(`${API_BASE_URL}/auth/refresh`, {}, {
            headers: {
              Authorization: `Bearer ${refreshToken}`,
            },
          });

          const { access_token } = response.data;
          localStorage.setItem('access_token', access_token);

          // Retry original request with new token
          originalRequest.headers.Authorization = `Bearer ${access_token}`;
          return api(originalRequest);
        } catch (refreshError) {
          // Refresh failed, redirect to login
          localStorage.removeItem('access_token');
          localStorage.removeItem('refresh_token');
          localStorage.removeItem('user');
          window.location.href = '/login';
          return Promise.reject(refreshError);
        }
      } else {
        // No refresh token, redirect to login
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        localStorage.removeItem('user');
        window.location.href = '/login';
      }
    }

    return Promise.reject(error);
  }
);

// Auth API functions
export const authAPI = {
  register: async (userData) => {
    const response = await api.post('/auth/register', userData);
    return response.data;
  },

  login: async (credentials) => {
    const response = await api.post('/auth/login', credentials);
    return response.data;
  },

  logout: async () => {
    const response = await api.post('/auth/logout');
    return response.data;
  },

  getCurrentUser: async () => {
    const response = await api.get('/auth/me');
    return response.data;
  },

  refreshToken: async () => {
    const refreshToken = localStorage.getItem('refresh_token');
    const response = await axios.post(`${API_BASE_URL}/auth/refresh`, {}, {
      headers: {
        Authorization: `Bearer ${refreshToken}`,
      },
    });
    return response.data;
  },
};

// RAG Chatbot API functions
export const chatbotAPI = {
  // Send query to RAG chat endpoint
  sendQuery: async (query) => {
    const response = await api.post('/chat', { query });
    return response.data;
  },

  // Direct semantic search without conversation
  searchProducts: async (query, k = 5, score_threshold = 0.0) => {
    const response = await api.post('/search', { 
      query, 
      k, 
      score_threshold 
    });
    return response.data;
  },

  // Get system status
  getStatus: async () => {
    const response = await api.get('/status');
    return response.data;
  },

  // Health check
  getHealth: async () => {
    const response = await api.get('/health');
    return response.data;
  },

  // Chat session management
  createChatSession: async (title = 'New Chat') => {
    const response = await api.post('/chats', { title });
    return response.data;
  },

  getChatSessions: async () => {
    const response = await api.get('/chats');
    return response.data;
  },

  getChatSession: async (sessionId) => {
    const response = await api.get(`/chats/${sessionId}`);
    return response.data;
  },

  deleteChatSession: async (sessionId) => {
    const response = await api.delete(`/chats/${sessionId}`);
    return response.data;
  },

  addMessageToSession: async (sessionId, messageData) => {
    const response = await api.post(`/chats/${sessionId}/messages`, messageData);
    return response.data;
  },

  // Check live price and availability for a product
  checkLiveProduct: async (productId, url) => {
    if (!productId && !url) {
      throw new Error('productId or url is required for live check');
    }

    const config = {};
    if (url) {
      config.params = { url };
    }

    const response = await api.get(`/products/${productId}/check_live`, config);
    return response.data;
  },
};

export default api;

import React, { useState } from 'react';
import { MessageSquare, Plus, Trash2, Search } from 'lucide-react';
import './ChatSidebar.css';

const ChatSidebar = ({ 
  sessions, 
  currentSessionId, 
  onSelectSession, 
  onNewChat, 
  onDeleteSession,
  isOpen 
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [hoveredSessionId, setHoveredSessionId] = useState(null);

const filteredSessions = sessions.filter(session => 
  session && session.title && session.title.toLowerCase().includes(searchQuery.toLowerCase())
);

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  const handleDelete = (e, sessionId) => {
    e.stopPropagation();
    if (window.confirm('Delete this chat? This cannot be undone.')) {
      onDeleteSession(sessionId);
    }
  };

  return (
    <div className={`chat-sidebar ${isOpen ? 'open' : ''}`}>
      {/* Header */}
      <div className="sidebar-header">
        <button className="new-chat-btn" onClick={onNewChat}>
          <Plus size={18} />
          <span>New chat</span>
        </button>
      </div>

      {/* Search */}
      <div className="sidebar-search">
        <Search size={16} className="search-icon" />
        <input
          type="text"
          placeholder="Search chats..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="search-input"
        />
      </div>

      {/* Chat List */}
      <div className="sidebar-content">
        {filteredSessions.length === 0 ? (
          <div className="empty-state">
            <MessageSquare size={48} className="empty-icon" />
            <p>No chats yet</p>
            <p className="empty-subtitle">Start a new conversation</p>
          </div>
        ) : (
          <div className="chat-list">
            {filteredSessions.map((session) => (
              <div
                key={session.id}
                className={`chat-item ${
                  session.id === currentSessionId ? 'active' : ''
                }`}
                onClick={() => onSelectSession(session.id)}
                onMouseEnter={() => setHoveredSessionId(session.id)}
                onMouseLeave={() => setHoveredSessionId(null)}
              >
                <div className="chat-item-content">
                  <MessageSquare size={16} className="chat-icon" />
                  <div className="chat-item-text">
                    <div className="chat-title">{session.title}</div>
                    <div className="chat-date">
                      {formatDate(session.updated_at)} · {session.message_count} messages
                    </div>
                  </div>
                </div>
                {hoveredSessionId === session.id && (
                  <button
                    className="delete-btn"
                    onClick={(e) => handleDelete(e, session.id)}
                    title="Delete chat"
                  >
                    <Trash2 size={16} />
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="sidebar-footer">
        <div className="sidebar-stats">
          {sessions.length} {sessions.length === 1 ? 'chat' : 'chats'}
        </div>
      </div>
    </div>
  );
};

export default ChatSidebar;
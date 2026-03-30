/**
 * Conversation Manager
 * In-memory store for active SMS conversations.
 * Each phone number has one session with:
 *  - history: [{role, content}] for Claude
 *  - state:   arbitrary bot state object
 *  - botType: which bot is currently handling this conversation
 *  - lastActivity: Date
 */

const SESSION_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes

class ConversationManager {
  constructor() {
    this._sessions = new Map();
    // Clean up stale sessions every 10 minutes
    setInterval(() => this._cleanup(), 10 * 60 * 1000);
  }

  /**
   * Get or create a session for a phone number.
   */
  getSession(phone) {
    if (!this._sessions.has(phone)) {
      this._sessions.set(phone, {
        phone,
        history: [],
        state: {},
        botType: null,
        startedAt: new Date(),
        lastActivity: new Date()
      });
    }
    return this._sessions.get(phone);
  }

  /**
   * Add a message to the conversation history.
   * @param {string} phone
   * @param {'user'|'assistant'} role
   * @param {string} content
   */
  addMessage(phone, role, content) {
    const session = this.getSession(phone);
    session.history.push({ role, content });
    session.lastActivity = new Date();
    // Keep last 20 messages to avoid huge context
    if (session.history.length > 20) {
      session.history = session.history.slice(-20);
    }
  }

  /**
   * Update arbitrary state on the session.
   */
  setState(phone, updates) {
    const session = this.getSession(phone);
    session.state = { ...session.state, ...updates };
    session.lastActivity = new Date();
  }

  /**
   * Set which bot type is handling this conversation.
   */
  setBotType(phone, botType) {
    const session = this.getSession(phone);
    session.botType = botType;
  }

  /**
   * Clear/reset a session (after booking confirmed, etc.)
   */
  clearSession(phone) {
    this._sessions.delete(phone);
  }

  _cleanup() {
    const now = Date.now();
    for (const [phone, session] of this._sessions) {
      if (now - session.lastActivity.getTime() > SESSION_TIMEOUT_MS) {
        this._sessions.delete(phone);
      }
    }
  }

  get activeCount() {
    return this._sessions.size;
  }
}

module.exports = new ConversationManager();

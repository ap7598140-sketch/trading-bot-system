/**
 * Bot Router
 * Central dispatcher for all inbound SMS messages.
 *
 * Routing priority (highest to lowest):
 *  1. EMERGENCY — keyword detected → EmergencyBot
 *  2. "QUOTE" / "PRICE" keyword → QuoteBot
 *  3. "BOOK" / "APPOINTMENT" keyword → BookingBot
 *  4. Active session continues with same bot → respective bot
 *  5. No session / ambiguous → Claude intent classification → route accordingly
 *  6. Default → BookingBot (most valuable action for a trade business)
 */

const claudeService = require('../services/claudeService');
const conversationManager = require('./conversationManager');
const logger = require('../services/logger');

const INTENTS = {
  EMERGENCY: 'emergency',
  QUOTE: 'quote',
  BOOKING: 'booking'
};

// Fast keyword overrides — checked before any AI
const KEYWORD_ROUTES = {
  emergency: INTENTS.EMERGENCY,
  urgent:    INTENTS.EMERGENCY,
  sos:       INTENTS.EMERGENCY,
  quote:     INTENTS.QUOTE,
  price:     INTENTS.QUOTE,
  cost:      INTENTS.QUOTE,
  how much:  INTENTS.QUOTE,
  book:      INTENTS.BOOKING,
  appointment: INTENTS.BOOKING,
  schedule:  INTENTS.BOOKING
};

class BotRouter {
  constructor(bots, clientConfig) {
    this.emergency = bots.emergency;
    this.quote     = bots.quote;
    this.booking   = bots.booking;
    this.config    = clientConfig;
  }

  /**
   * Main dispatch method.
   * @param {object} msg - { from: string, body: string }
   */
  async handle({ from, body }) {
    const lower = body.toLowerCase().trim();

    // 1. Check emergency keywords (fastest path — safety first)
    if (this.emergency.isEmergency(body)) {
      logger.info('Routing → Emergency', { from });
      return this.emergency.handle(from, body);
    }

    // 2. Check explicit keyword overrides
    const keywordIntent = this._checkKeywords(lower);
    if (keywordIntent) {
      logger.info('Routing → keyword match', { from, intent: keywordIntent });
      return this._dispatchIntent(keywordIntent, from, body);
    }

    // 3. Continue existing session with the same bot
    const session = conversationManager.getSession(from);
    if (session.botType && session.history.length > 0) {
      logger.info('Routing → active session', { from, botType: session.botType });
      return this._dispatchIntent(session.botType, from, body);
    }

    // 4. AI intent classification for first-contact messages
    const intent = await this._classifyIntent(body);
    logger.info('Routing → AI classified', { from, intent });
    return this._dispatchIntent(intent, from, body);
  }

  _checkKeywords(lower) {
    for (const [kw, intent] of Object.entries(KEYWORD_ROUTES)) {
      if (lower.includes(kw)) return intent;
    }
    return null;
  }

  async _classifyIntent(message) {
    const serviceKeywords = this.config.services
      .flatMap(s => s.keywords)
      .slice(0, 20)
      .join(', ');

    const systemPrompt = `You are an SMS routing classifier for a ${this.config.tradeType} business.
Classify the customer's message into one of these intents:
- booking: wants to schedule an appointment
- quote: wants a price estimate
- emergency: has an urgent problem needing immediate help

Known services: ${serviceKeywords}

Reply with ONLY one word: booking | quote | emergency`;

    const result = await claudeService.complete(systemPrompt, message, 20);
    const trimmed = result.trim().toLowerCase();
    if (['booking', 'quote', 'emergency'].includes(trimmed)) return trimmed;
    return INTENTS.BOOKING; // default
  }

  async _dispatchIntent(intent, from, body) {
    switch (intent) {
      case INTENTS.EMERGENCY:
        return this.emergency.handle(from, body);
      case INTENTS.QUOTE:
        return this.quote.handle(from, body);
      case INTENTS.BOOKING:
      default:
        return this.booking.handle(from, body);
    }
  }
}

module.exports = BotRouter;

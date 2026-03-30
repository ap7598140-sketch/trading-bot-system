/**
 * Quote Bot
 * Gives instant price estimates via SMS based on the service config.
 * - Detects which service the customer is asking about
 * - Returns formatted price range from config (no AI markup needed for prices)
 * - Uses Claude to write a natural, helpful message around the price
 * - Logs all quotes to Google Sheets
 */

const { v4: uuidv4 } = require('uuid');
const claudeService = require('../services/claudeService');
const twilioService = require('../services/twilioService');
const sheetsService = require('../services/sheetsService');
const conversationManager = require('../shared/conversationManager');
const logger = require('../services/logger');

class QuoteBot {
  constructor(clientConfig) {
    this.config = clientConfig;
  }

  /**
   * Match a message to a service using keyword matching + Claude fallback.
   */
  async _matchService(message) {
    const msgLower = message.toLowerCase();

    // Try keyword match first (fast, free)
    for (const service of this.config.services) {
      if (service.keywords.some(kw => msgLower.includes(kw.toLowerCase()))) {
        return service;
      }
    }

    // Claude fallback — ask it to identify the service
    const serviceNames = this.config.services.map(s => s.name).join(' | ');
    const systemPrompt = `You are a ${this.config.tradeType} service classifier.
Given a customer message, respond with EXACTLY the service name that matches, or "unknown" if none match.
Available services: ${serviceNames}`;
    const match = await claudeService.complete(systemPrompt, message, 80);
    return this.config.services.find(
      s => s.name.toLowerCase() === match.trim().toLowerCase()
    ) || null;
  }

  /**
   * Format a price object into a human-readable string.
   */
  _formatPrice(service) {
    const p = service.price;
    if (!p) return 'price on request';
    switch (p.unit) {
      case 'fixed':
        return p.to
          ? `$${p.from}–$${p.to} (fixed)`
          : `$${p.from} (fixed)`;
      case 'from':
        return `from $${p.from}`;
      case 'range':
        return `$${p.from}–$${p.to}`;
      case 'callout':
        return `$${p.from} callout fee + labour`;
      case 'per_point':
        return `$${p.from} per point`;
      default:
        return `from $${p.from}`;
    }
  }

  /**
   * Check if current time is within business hours.
   */
  _isAfterHours() {
    const now = new Date();
    const dayNames = ['sunday','monday','tuesday','wednesday','thursday','friday','saturday'];
    const day = dayNames[now.getDay()];
    const hours = this.config.hours[day];
    if (!hours) return true; // closed day

    const [openH, openM] = hours.open.split(':').map(Number);
    const [closeH, closeM] = hours.close.split(':').map(Number);
    const nowMins = now.getHours() * 60 + now.getMinutes();
    const openMins = openH * 60 + openM;
    const closeMins = closeH * 60 + closeM;
    return nowMins < openMins || nowMins > closeMins;
  }

  /**
   * Build the Claude prompt for writing the quote message.
   */
  _buildQuotePrompt(service, priceStr) {
    const surcharge = this.config.hours.emergencySurcharge;
    const afterHours = this._isAfterHours();

    return `You are ${this.config.bot.name} for ${this.config.businessName}.
A customer asked for a quote on: ${service.name}

Our price for this service is: ${priceStr}
${afterHours ? `After-hours surcharge: +$${surcharge}` : ''}
Duration: ${service.duration ? service.duration + ' mins approx' : 'varies by job'}

Write a SHORT (2-3 sentences) friendly SMS reply that:
1. States the price clearly
2. ${afterHours ? 'Mentions the after-hours surcharge applies' : 'Notes this is an estimate and final price given on inspection'}
3. Offers to book them in (say "Reply BOOK to book")
4. Ends with: ${this.config.bot.signOff}

Keep it under 320 characters total.`;
  }

  /**
   * Main handler — called by BotRouter.
   */
  async handle(from, body) {
    conversationManager.addMessage(from, 'user', body);
    conversationManager.setBotType(from, 'quote');

    await this._logConversation(from, 'inbound', body, 'quote');

    const service = await this._matchService(body);

    let reply;
    if (!service) {
      reply = `Hi! I can give you a quote for any of our services. What do you need help with? (e.g. leaking tap, blocked drain, hot water). — ${this.config.bot.signOff}`;
    } else {
      const priceStr = this._formatPrice(service);
      const prompt = this._buildQuotePrompt(service, priceStr);
      reply = await claudeService.complete(prompt, body, 200);

      // Save quote to Sheets
      try {
        await sheetsService.logQuote(this.config, {
          id: uuidv4(),
          customerPhone: from,
          service: service.name,
          priceFrom: service.price?.from || '',
          priceTo: service.price?.to || '',
          priceUnit: service.price?.unit || '',
          description: body
        });
      } catch (e) {
        logger.warn('Failed to log quote', { error: e.message });
      }
    }

    await twilioService.send(from, reply);
    conversationManager.addMessage(from, 'assistant', reply);
    await this._logConversation(from, 'outbound', reply, 'quote', service?.id || 'unknown_service');

    logger.info('Quote bot replied', { from, service: service?.name || 'unknown' });
  }

  async _logConversation(phone, direction, message, botType, intent = '') {
    try {
      await sheetsService.logConversation(this.config, {
        id: uuidv4(),
        customerPhone: phone,
        direction,
        message,
        botType,
        intent
      });
    } catch (e) {
      logger.warn('Failed to log conversation', { error: e.message });
    }
  }
}

module.exports = QuoteBot;

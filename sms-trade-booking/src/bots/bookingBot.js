/**
 * Booking Bot
 * Handles multi-turn SMS conversations to collect:
 *   1. Customer name
 *   2. Service needed
 *   3. Preferred date & time
 *   4. Confirmation
 * Then logs to Google Sheets and sends a confirmation SMS.
 */

const { v4: uuidv4 } = require('uuid');
const claudeService = require('../services/claudeService');
const twilioService = require('../services/twilioService');
const sheetsService = require('../services/sheetsService');
const conversationManager = require('../shared/conversationManager');
const logger = require('../services/logger');

class BookingBot {
  constructor(clientConfig) {
    this.config = clientConfig;
  }

  /**
   * Build the system prompt dynamically from client config.
   */
  _buildSystemPrompt() {
    const { businessName, tradeType, bot, services, hours, location } = this.config;

    const serviceList = services
      .map(s => `- ${s.name}`)
      .join('\n');

    const hoursText = Object.entries(hours)
      .filter(([day, val]) => val && typeof val === 'object')
      .map(([day, val]) => `${day}: ${val.open}–${val.close}`)
      .join(', ');

    return `You are ${bot.name}, the SMS booking assistant for ${businessName} (${tradeType}).
Your job is to book appointments for customers via SMS.

SERVICES WE OFFER:
${serviceList}

BUSINESS HOURS: ${hoursText}
LOCATION: ${location.suburb}, ${location.city}

BOOKING FLOW — collect these in order, naturally:
1. Customer's name
2. What service they need (match to our service list above)
3. Preferred date (ask for day + date)
4. Preferred time (within business hours)
5. Confirm all details and tell them we'll confirm within 1 hour

RULES:
- Keep responses SHORT (under 160 chars when possible, 2 sentences max)
- Be ${bot.tone}
- If the service is an emergency, tell them to reply EMERGENCY for urgent help
- Do NOT confirm prices — only the quote bot does that
- Once you have name, service, date, time → output a JSON summary block like:
  BOOKING_COMPLETE:{"name":"...","service":"...","date":"...","time":"...","phone":"..."}
- Sign off with: ${bot.signOff}`;
  }

  /**
   * Main handler — called by BotRouter when this bot is active.
   * @param {string} from  - Customer phone
   * @param {string} body  - Inbound message text
   */
  async handle(from, body) {
    const session = conversationManager.getSession(from);
    conversationManager.addMessage(from, 'user', body);
    conversationManager.setBotType(from, 'booking');

    // Log inbound to Sheets
    await this._logConversation(from, 'inbound', body, 'booking');

    const systemPrompt = this._buildSystemPrompt();
    const reply = await claudeService.chat(systemPrompt, session.history);

    // Check if Claude has completed the booking
    const bookingMatch = reply.match(/BOOKING_COMPLETE:(\{.*?\})/s);
    if (bookingMatch) {
      await this._finaliseBooking(from, bookingMatch[1], reply);
    } else {
      conversationManager.addMessage(from, 'assistant', reply);
      await twilioService.send(from, reply);
      await this._logConversation(from, 'outbound', reply, 'booking');
    }

    logger.info('Booking bot replied', { from, replyLength: reply.length });
  }

  async _finaliseBooking(from, jsonStr, fullReply) {
    let bookingData;
    try {
      bookingData = JSON.parse(jsonStr);
    } catch (e) {
      logger.error('Failed to parse booking JSON', { jsonStr });
      return;
    }

    const booking = {
      id: uuidv4(),
      customerName: bookingData.name,
      customerPhone: from,
      service: bookingData.service,
      preferredDate: bookingData.date,
      preferredTime: bookingData.time,
      status: 'pending',
      notes: ''
    };

    // Save to Sheets
    try {
      await sheetsService.logBooking(this.config, booking);
      logger.info('Booking saved to Sheets', { id: booking.id });
    } catch (e) {
      logger.error('Failed to save booking', { error: e.message });
    }

    // Send clean confirmation (strip the JSON marker from Claude's reply)
    const cleanReply = fullReply.replace(/BOOKING_COMPLETE:\{.*?\}/s, '').trim() ||
      `Great! Booking received for ${bookingData.service} on ${bookingData.date} at ${bookingData.time}. We'll confirm shortly. — ${this.config.bot.signOff}`;

    await twilioService.send(from, cleanReply);
    await this._logConversation(from, 'outbound', cleanReply, 'booking', 'booking_complete');

    // Alert the owner
    await twilioService.send(
      this.config.ownerPhone,
      `📅 NEW BOOKING [${this.config.businessName}]\n` +
      `Name: ${booking.customerName}\n` +
      `Phone: ${from}\n` +
      `Service: ${booking.service}\n` +
      `Date: ${booking.preferredDate} ${booking.preferredTime}`
    );

    // Clear session after booking
    conversationManager.clearSession(from);
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

module.exports = BookingBot;

/**
 * Twilio Service
 * Wraps all Twilio SMS operations: send messages, validate webhooks.
 */

const twilio = require('twilio');
const logger = require('./logger');

class TwilioService {
  constructor() {
    this.client = twilio(
      process.env.TWILIO_ACCOUNT_SID,
      process.env.TWILIO_AUTH_TOKEN
    );
    this.from = process.env.TWILIO_PHONE_NUMBER;
  }

  /**
   * Send an SMS message
   * @param {string} to   - Recipient phone number (E.164 format)
   * @param {string} body - Message text
   * @returns {Promise<object>} Twilio message object
   */
  async send(to, body) {
    try {
      const msg = await this.client.messages.create({
        from: this.from,
        to,
        body
      });
      logger.info('SMS sent', { to, sid: msg.sid, length: body.length });
      return msg;
    } catch (err) {
      logger.error('SMS send failed', { to, error: err.message });
      throw err;
    }
  }

  /**
   * Send a long message by splitting into 160-char chunks
   * @param {string} to
   * @param {string} body
   */
  async sendLong(to, body) {
    const MAX = 1550; // Twilio's practical concatenated SMS limit
    if (body.length <= MAX) return this.send(to, body);

    const chunks = [];
    let remaining = body;
    while (remaining.length > 0) {
      chunks.push(remaining.slice(0, MAX));
      remaining = remaining.slice(MAX);
    }
    for (const chunk of chunks) {
      await this.send(to, chunk);
    }
  }

  /**
   * Validate that an incoming request is genuinely from Twilio
   * @param {string} signature - X-Twilio-Signature header
   * @param {string} url       - Full webhook URL
   * @param {object} params    - POST body params
   */
  validateRequest(signature, url, params) {
    return twilio.validateRequest(
      process.env.TWILIO_AUTH_TOKEN,
      signature,
      url,
      params
    );
  }
}

module.exports = new TwilioService();

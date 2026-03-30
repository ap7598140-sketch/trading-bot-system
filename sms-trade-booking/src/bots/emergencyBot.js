/**
 * Emergency Callout Bot
 * Detects urgent messages and:
 *  1. Immediately alerts the owner via SMS
 *  2. Sends a reassuring reply to the customer
 *  3. Logs the emergency to Google Sheets
 *
 * Detection: keyword matching (instant) + Claude confirmation (async, not blocking)
 */

const { v4: uuidv4 } = require('uuid');
const claudeService = require('../services/claudeService');
const twilioService = require('../services/twilioService');
const sheetsService = require('../services/sheetsService');
const conversationManager = require('../shared/conversationManager');
const logger = require('../services/logger');

class EmergencyBot {
  constructor(clientConfig) {
    this.config = clientConfig;
  }

  /**
   * Fast keyword-based emergency detection.
   * Returns true if any emergency keyword is found.
   */
  isEmergency(message) {
    const lower = message.toLowerCase();
    return this.config.emergencyKeywords.some(kw => lower.includes(kw.toLowerCase()));
  }

  /**
   * Ask Claude to confirm emergency and extract details.
   */
  async _analyseWithClaude(from, message) {
    const prompt = `You are an emergency triage assistant for ${this.config.businessName} (${this.config.tradeType}).
A customer sent this SMS: "${message}"

Respond in JSON only:
{
  "isEmergency": true/false,
  "urgencyLevel": "critical" | "high" | "medium",
  "issue": "brief description of the problem",
  "safetyRisk": true/false,
  "suggestedAction": "what the tradesperson should do first"
}`;
    try {
      const raw = await claudeService.complete(prompt, message, 200);
      const jsonMatch = raw.match(/\{[\s\S]*\}/);
      if (jsonMatch) return JSON.parse(jsonMatch[0]);
    } catch (e) {
      logger.warn('Emergency analysis failed', { error: e.message });
    }
    return { isEmergency: true, urgencyLevel: 'high', issue: message, safetyRisk: false };
  }

  /**
   * Build owner alert SMS.
   */
  _buildOwnerAlert(from, analysis) {
    const urgencyEmoji = {
      critical: '🚨🚨🚨',
      high: '🚨',
      medium: '⚠️'
    }[analysis.urgencyLevel] || '🚨';

    return `${urgencyEmoji} EMERGENCY CALLOUT [${this.config.businessName}]
From: ${from}
Issue: ${analysis.issue}
Urgency: ${analysis.urgencyLevel.toUpperCase()}
${analysis.safetyRisk ? '⚠️ SAFETY RISK — respond immediately' : ''}
Reply to customer or call them directly.`;
  }

  /**
   * Build customer reassurance SMS.
   */
  _buildCustomerReply(isAfterHours) {
    const surcharge = this.config.hours.emergencySurcharge;
    const afterHoursNote = isAfterHours
      ? ` After-hours emergency surcharge of $${surcharge} applies.`
      : '';

    return `🚨 We've received your emergency! ${this.config.ownerName} has been alerted and will contact you within 15 minutes.${afterHoursNote} — ${this.config.businessName}`;
  }

  _isAfterHours() {
    const now = new Date();
    const dayNames = ['sunday','monday','tuesday','wednesday','thursday','friday','saturday'];
    const day = dayNames[now.getDay()];
    const hours = this.config.hours[day];
    if (!hours) return true;
    const [openH, openM] = hours.open.split(':').map(Number);
    const [closeH, closeM] = hours.close.split(':').map(Number);
    const nowMins = now.getHours() * 60 + now.getMinutes();
    return nowMins < openH * 60 + openM || nowMins > closeH * 60 + closeM;
  }

  /**
   * Main handler — called by BotRouter when an emergency is detected.
   */
  async handle(from, body) {
    conversationManager.setBotType(from, 'emergency');
    logger.warn('EMERGENCY DETECTED', { from, body: body.substring(0, 100) });

    // 1. Send customer reassurance FIRST (critical — don't wait for analysis)
    const customerReply = this._buildCustomerReply(this._isAfterHours());
    await twilioService.send(from, customerReply);

    // 2. Analyse with Claude (non-blocking for customer, but we await for owner alert)
    const analysis = await this._analyseWithClaude(from, body);

    // 3. Alert the owner
    const ownerAlert = this._buildOwnerAlert(from, analysis);
    await twilioService.send(this.config.ownerPhone, ownerAlert);

    // 4. Log to Sheets (conversation + lead as emergency)
    const logId = uuidv4();
    await Promise.allSettled([
      sheetsService.logConversation(this.config, {
        id: logId,
        customerPhone: from,
        direction: 'inbound',
        message: body,
        botType: 'emergency',
        intent: `emergency_${analysis.urgencyLevel}`
      }),
      sheetsService.logConversation(this.config, {
        id: uuidv4(),
        customerPhone: from,
        direction: 'outbound',
        message: customerReply,
        botType: 'emergency',
        intent: 'emergency_response'
      }),
      // Also log as a lead so it shows in the leads tab
      sheetsService.logLead(this.config, {
        id: uuidv4(),
        source: 'SMS Emergency',
        phone: from,
        serviceNeeded: analysis.issue,
        postedText: body,
        status: 'urgent',
        notes: `Urgency: ${analysis.urgencyLevel}. Safety risk: ${analysis.safetyRisk}`
      })
    ]);

    logger.warn('Emergency handled', { from, urgency: analysis.urgencyLevel });
  }
}

module.exports = EmergencyBot;

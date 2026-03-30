/**
 * Claude Service
 * Wraps Anthropic Claude Haiku API for all AI response generation.
 */

const Anthropic = require('@anthropic-ai/sdk');
const logger = require('./logger');

const MODEL = 'claude-haiku-4-5-20251001';
const MAX_TOKENS = 1024;

class ClaudeService {
  constructor() {
    this.client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
  }

  /**
   * Generate a single AI response given a system prompt and message history.
   * @param {string}   systemPrompt - Instruction context for the bot
   * @param {Array}    messages     - [{role:'user'|'assistant', content:'...'}]
   * @param {number}   maxTokens    - Override default max tokens
   * @returns {Promise<string>} Assistant reply text
   */
  async chat(systemPrompt, messages, maxTokens = MAX_TOKENS) {
    try {
      const response = await this.client.messages.create({
        model: MODEL,
        max_tokens: maxTokens,
        system: systemPrompt,
        messages
      });
      const text = response.content[0]?.text || '';
      logger.info('Claude response', {
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens
      });
      return text;
    } catch (err) {
      logger.error('Claude API error', { error: err.message });
      throw err;
    }
  }

  /**
   * Quick single-turn completion — no conversation history needed.
   * @param {string} systemPrompt
   * @param {string} userMessage
   */
  async complete(systemPrompt, userMessage, maxTokens = MAX_TOKENS) {
    return this.chat(systemPrompt, [{ role: 'user', content: userMessage }], maxTokens);
  }

  /**
   * Classify an intent from a fixed set of labels.
   * Returns the matching label or 'unknown'.
   * @param {string}   message - Incoming SMS text
   * @param {string[]} labels  - Possible intent labels
   */
  async classify(message, labels) {
    const systemPrompt = `You are an intent classifier for a trade services SMS bot.
Given the user's message, respond with EXACTLY ONE of these labels (no other text):
${labels.join(' | ')}

If unsure, respond with: unknown`;
    const result = await this.complete(systemPrompt, message, 50);
    const trimmed = result.trim().toLowerCase();
    const match = labels.find(l => l.toLowerCase() === trimmed);
    return match || 'unknown';
  }
}

module.exports = new ClaudeService();

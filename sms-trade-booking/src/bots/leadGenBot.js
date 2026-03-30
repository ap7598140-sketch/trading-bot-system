/**
 * Lead Generator Bot
 * Scrapes Google Maps and Facebook (via public search) for people
 * actively looking for trade services in the client's service area.
 *
 * Sources:
 *  - Google Maps: Searches for competitors + extracts recent reviews mentioning
 *    "looking for", "need a", "recommend" etc (signals of someone needing the service)
 *  - Facebook: Searches public posts in local community groups
 *    (uses Puppeteer to scrape public group posts)
 *
 * All found leads are saved to Google Sheets and the owner is alerted.
 *
 * USAGE: Run standalone via `npm run lead-gen` or scheduled via n8n.
 */

const puppeteer = require('puppeteer');
const { v4: uuidv4 } = require('uuid');
const claudeService = require('../services/claudeService');
const twilioService = require('../services/twilioService');
const sheetsService = require('../services/sheetsService');
const logger = require('../services/logger');

// Keywords that suggest someone is looking for a tradesperson
const INTENT_KEYWORDS = [
  'looking for', 'need a', 'anyone recommend', 'can anyone suggest',
  'who is good', 'urgent', 'asap', 'recommend a', 'need help with',
  'does anyone know', 'good plumber', 'good electrician', 'good builder',
  'need tradesman', 'anyone know a', 'anyone got a', 'need someone to'
];

class LeadGenBot {
  constructor(clientConfig) {
    this.config = clientConfig;
    this.browser = null;
  }

  async _launchBrowser() {
    if (!this.browser) {
      this.browser = await puppeteer.launch({
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
      });
    }
    return this.browser;
  }

  async _closeBrowser() {
    if (this.browser) {
      await this.browser.close();
      this.browser = null;
    }
  }

  /**
   * Search Google Maps for recent activity (people looking for trades).
   * Strategy: Search "{tradeType} near {suburb}" and extract Q&A / reviews
   * that indicate someone is actively looking.
   */
  async scrapeGoogleMaps() {
    const leads = [];
    const { tradeType, location } = this.config;

    logger.info('Scraping Google Maps', { tradeType, areas: location.serviceAreas });

    const browser = await this._launchBrowser();

    for (const area of location.serviceAreas.slice(0, 3)) { // limit to 3 areas per run
      const page = await browser.newPage();
      await page.setUserAgent(
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
      );

      try {
        const searchQuery = encodeURIComponent(`${tradeType} ${area} ${location.city}`);
        await page.goto(
          `https://www.google.com/maps/search/${searchQuery}`,
          { waitUntil: 'networkidle2', timeout: 30000 }
        );
        await page.waitForTimeout(2000);

        // Extract visible text that might contain lead signals
        const pageText = await page.evaluate(() => document.body.innerText);
        const leadSignals = this._extractLeadSignals(pageText, area, 'Google Maps');
        leads.push(...leadSignals);

      } catch (e) {
        logger.warn('Google Maps scrape failed for area', { area, error: e.message });
      } finally {
        await page.close();
      }
    }

    return leads;
  }

  /**
   * Search Facebook public posts in local community groups.
   * Uses public search (no login required for public posts).
   */
  async scrapeFacebook() {
    const leads = [];
    const { tradeType, location } = this.config;

    logger.info('Scraping Facebook', { tradeType, city: location.city });

    const browser = await this._launchBrowser();
    const page = await browser.newPage();

    await page.setUserAgent(
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
    );

    try {
      // Facebook public search for local posts
      for (const area of location.serviceAreas.slice(0, 2)) {
        const query = encodeURIComponent(
          `${tradeType} ${area} recommend looking for`
        );
        await page.goto(
          `https://www.facebook.com/search/posts/?q=${query}`,
          { waitUntil: 'domcontentloaded', timeout: 20000 }
        );
        await page.waitForTimeout(3000);

        const posts = await page.evaluate(() => {
          const elements = document.querySelectorAll('[data-ad-comet-preview="message"]');
          return Array.from(elements).slice(0, 10).map(el => ({
            text: el.innerText,
            timestamp: new Date().toISOString()
          }));
        });

        for (const post of posts) {
          if (this._hasLeadIntent(post.text)) {
            const lead = await this._qualifyLead(post.text, area, 'Facebook');
            if (lead) leads.push(lead);
          }
        }
      }
    } catch (e) {
      logger.warn('Facebook scrape failed', { error: e.message });
    } finally {
      await page.close();
    }

    return leads;
  }

  /**
   * Check if text contains lead intent signals.
   */
  _hasLeadIntent(text) {
    if (!text) return false;
    const lower = text.toLowerCase();
    return INTENT_KEYWORDS.some(kw => lower.includes(kw));
  }

  /**
   * Extract potential lead signals from raw scraped text.
   */
  _extractLeadSignals(text, area, source) {
    const leads = [];
    const lines = text.split('\n').filter(l => l.length > 20 && l.length < 500);

    for (const line of lines) {
      if (this._hasLeadIntent(line)) {
        leads.push({
          id: uuidv4(),
          source,
          location: area,
          serviceNeeded: this.config.tradeType,
          postedText: line.trim(),
          status: 'new',
          notes: `Auto-detected intent: ${INTENT_KEYWORDS.find(kw => line.toLowerCase().includes(kw))}`
        });
      }
    }
    return leads;
  }

  /**
   * Use Claude to qualify and enrich a lead from raw post text.
   */
  async _qualifyLead(postText, area, source) {
    const prompt = `You are a lead qualifier for a ${this.config.tradeType} business.
Analyse this Facebook post and respond in JSON only:
{
  "isValidLead": true/false,
  "serviceNeeded": "what trade service they need",
  "urgency": "low" | "medium" | "high",
  "location": "suburb/area if mentioned",
  "summary": "one sentence summary"
}

Post: "${postText.substring(0, 300)}"`;

    try {
      const raw = await claudeService.complete(prompt, postText, 200);
      const jsonMatch = raw.match(/\{[\s\S]*\}/);
      if (!jsonMatch) return null;
      const data = JSON.parse(jsonMatch[0]);
      if (!data.isValidLead) return null;

      return {
        id: uuidv4(),
        source,
        location: data.location || area,
        serviceNeeded: data.serviceNeeded || this.config.tradeType,
        postedText: postText.substring(0, 500),
        status: 'new',
        notes: `Urgency: ${data.urgency}. ${data.summary}`
      };
    } catch (e) {
      logger.warn('Lead qualification failed', { error: e.message });
      return null;
    }
  }

  /**
   * Save all new leads to Google Sheets and alert owner if high-value leads found.
   */
  async _saveLeads(leads) {
    if (leads.length === 0) return;

    for (const lead of leads) {
      try {
        await sheetsService.logLead(this.config, lead);
      } catch (e) {
        logger.warn('Failed to save lead', { error: e.message });
      }
    }

    // Alert owner if any new leads found
    const highUrgency = leads.filter(l => l.notes?.includes('high'));
    if (highUrgency.length > 0) {
      await twilioService.send(
        this.config.ownerPhone,
        `🔥 ${highUrgency.length} urgent lead(s) found for ${this.config.businessName}!\n` +
        `Check your dashboard: ${process.env.WEBHOOK_BASE_URL || 'http://localhost:3000'}`
      );
    } else {
      await twilioService.send(
        this.config.ownerPhone,
        `📊 Lead Gen: ${leads.length} new lead(s) found for ${this.config.businessName}. Check your dashboard.`
      );
    }

    logger.info('Leads saved', { count: leads.length });
  }

  /**
   * Run full lead generation cycle.
   * Called by n8n workflow or npm run lead-gen.
   */
  async run() {
    logger.info('Lead Gen Bot starting', { client: this.config.businessName });

    try {
      const [gmLeads, fbLeads] = await Promise.allSettled([
        this.scrapeGoogleMaps(),
        this.scrapeFacebook()
      ]);

      const allLeads = [
        ...(gmLeads.status === 'fulfilled' ? gmLeads.value : []),
        ...(fbLeads.status === 'fulfilled' ? fbLeads.value : [])
      ];

      logger.info('Lead Gen complete', {
        googleMaps: gmLeads.status === 'fulfilled' ? gmLeads.value.length : 0,
        facebook: fbLeads.status === 'fulfilled' ? fbLeads.value.length : 0,
        total: allLeads.length
      });

      await this._saveLeads(allLeads);
      return allLeads;
    } finally {
      await this._closeBrowser();
    }
  }
}

// Allow standalone execution: node src/bots/leadGenBot.js
if (require.main === module) {
  require('dotenv').config({ path: require('path').join(__dirname, '../../.env') });
  const config = require('../../config/clientConfig');
  const bot = new LeadGenBot(config);
  bot.run()
    .then(leads => {
      logger.info('Lead Gen finished', { total: leads.length });
      process.exit(0);
    })
    .catch(err => {
      logger.error('Lead Gen failed', { error: err.message });
      process.exit(1);
    });
}

module.exports = LeadGenBot;

/**
 * Google Sheets Service
 * Handles all read/write operations to the client's Google Spreadsheet.
 *
 * Sheet tabs expected (auto-created if missing):
 *   Bookings | Quotes | Leads | Conversations
 */

const { google } = require('googleapis');
const logger = require('./logger');

// Column headers for each tab
const HEADERS = {
  Bookings: [
    'ID', 'Timestamp', 'CustomerName', 'CustomerPhone',
    'Service', 'PreferredDate', 'PreferredTime',
    'Status', 'Notes', 'ConfirmedAt'
  ],
  Quotes: [
    'ID', 'Timestamp', 'CustomerName', 'CustomerPhone',
    'Service', 'PriceFrom', 'PriceTo', 'PriceUnit',
    'Description', 'ConvertedToBooking'
  ],
  Leads: [
    'ID', 'Timestamp', 'Source', 'Name', 'Phone', 'Email',
    'Location', 'ServiceNeeded', 'PostedText', 'Status', 'Notes'
  ],
  Conversations: [
    'ID', 'Timestamp', 'CustomerPhone', 'Direction',
    'Message', 'BotType', 'Intent'
  ]
};

class SheetsService {
  constructor() {
    this._auth = null;
    this._sheets = null;
  }

  async _getSheets() {
    if (this._sheets) return this._sheets;

    const auth = new google.auth.JWT({
      email: process.env.GOOGLE_SERVICE_ACCOUNT_EMAIL,
      key: (process.env.GOOGLE_PRIVATE_KEY || '').replace(/\\n/g, '\n'),
      scopes: ['https://www.googleapis.com/auth/spreadsheets']
    });
    await auth.authorize();
    this._sheets = google.sheets({ version: 'v4', auth });
    return this._sheets;
  }

  /**
   * Ensure all expected tabs and header rows exist.
   * Safe to call on startup each time.
   */
  async ensureSheetTabs(spreadsheetId, tabConfig) {
    const sheets = await this._getSheets();

    // Get existing sheets
    const meta = await sheets.spreadsheets.get({ spreadsheetId });
    const existing = meta.data.sheets.map(s => s.properties.title);

    const toCreate = Object.values(tabConfig).filter(t => !existing.includes(t));

    if (toCreate.length > 0) {
      await sheets.spreadsheets.batchUpdate({
        spreadsheetId,
        requestBody: {
          requests: toCreate.map(title => ({
            addSheet: { properties: { title } }
          }))
        }
      });
      logger.info('Created sheet tabs', { tabs: toCreate });
    }

    // Ensure headers on each tab
    for (const [key, tabName] of Object.entries(tabConfig)) {
      const headers = HEADERS[key] || HEADERS[tabName];
      if (!headers) continue;
      try {
        const check = await sheets.spreadsheets.values.get({
          spreadsheetId,
          range: `${tabName}!A1:Z1`
        });
        if (!check.data.values || check.data.values.length === 0) {
          await sheets.spreadsheets.values.update({
            spreadsheetId,
            range: `${tabName}!A1`,
            valueInputOption: 'RAW',
            requestBody: { values: [headers] }
          });
          logger.info('Added headers', { tab: tabName });
        }
      } catch (e) {
        logger.warn('Could not check/set headers', { tab: tabName, error: e.message });
      }
    }
  }

  /**
   * Append a row to a sheet tab.
   * @param {string}   spreadsheetId
   * @param {string}   tabName  - Sheet tab name
   * @param {Array}    row      - Array of cell values
   */
  async appendRow(spreadsheetId, tabName, row) {
    const sheets = await this._getSheets();
    try {
      await sheets.spreadsheets.values.append({
        spreadsheetId,
        range: `${tabName}!A:Z`,
        valueInputOption: 'USER_ENTERED',
        insertDataOption: 'INSERT_ROWS',
        requestBody: { values: [row] }
      });
      logger.info('Row appended', { tab: tabName });
    } catch (err) {
      logger.error('Sheet append failed', { tab: tabName, error: err.message });
      throw err;
    }
  }

  /**
   * Read all rows from a tab (returns array of arrays, skipping header row).
   */
  async readRows(spreadsheetId, tabName) {
    const sheets = await this._getSheets();
    const res = await sheets.spreadsheets.values.get({
      spreadsheetId,
      range: `${tabName}!A2:Z`
    });
    return res.data.values || [];
  }

  /**
   * Read all rows and return as array of objects (using header row as keys).
   */
  async readAsObjects(spreadsheetId, tabName) {
    const sheets = await this._getSheets();
    const headerRes = await sheets.spreadsheets.values.get({
      spreadsheetId,
      range: `${tabName}!A1:Z1`
    });
    const headers = (headerRes.data.values || [[]])[0];
    const rows = await this.readRows(spreadsheetId, tabName);
    return rows.map(row =>
      Object.fromEntries(headers.map((h, i) => [h, row[i] || '']))
    );
  }

  // ─── Convenience write methods ─────────────────────────────────────────────

  async logBooking(config, booking) {
    const { spreadsheetId, tabs } = config.sheets;
    await this.appendRow(spreadsheetId, tabs.bookings, [
      booking.id,
      new Date().toISOString(),
      booking.customerName,
      booking.customerPhone,
      booking.service,
      booking.preferredDate || '',
      booking.preferredTime || '',
      booking.status || 'pending',
      booking.notes || '',
      booking.confirmedAt || ''
    ]);
  }

  async logQuote(config, quote) {
    const { spreadsheetId, tabs } = config.sheets;
    await this.appendRow(spreadsheetId, tabs.quotes, [
      quote.id,
      new Date().toISOString(),
      quote.customerName || '',
      quote.customerPhone,
      quote.service,
      quote.priceFrom || '',
      quote.priceTo || '',
      quote.priceUnit || '',
      quote.description || '',
      'No'
    ]);
  }

  async logLead(config, lead) {
    const { spreadsheetId, tabs } = config.sheets;
    await this.appendRow(spreadsheetId, tabs.leads, [
      lead.id,
      new Date().toISOString(),
      lead.source,
      lead.name || '',
      lead.phone || '',
      lead.email || '',
      lead.location || '',
      lead.serviceNeeded || '',
      lead.postedText || '',
      lead.status || 'new',
      lead.notes || ''
    ]);
  }

  async logConversation(config, entry) {
    const { spreadsheetId, tabs } = config.sheets;
    await this.appendRow(spreadsheetId, tabs.conversations, [
      entry.id,
      new Date().toISOString(),
      entry.customerPhone,
      entry.direction,   // 'inbound' | 'outbound'
      entry.message,
      entry.botType || '',
      entry.intent || ''
    ]);
  }
}

module.exports = new SheetsService();

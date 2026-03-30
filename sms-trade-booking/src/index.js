/**
 * SMS Trade Booking System — Main Entry Point
 *
 * Boots the system:
 *  1. Loads client config
 *  2. Ensures Google Sheets tabs exist
 *  3. Instantiates all bots
 *  4. Creates bot router
 *  5. Starts Express webhook server
 */

require('dotenv').config();
const logger = require('./services/logger');
const clientConfig = require('../config/clientConfig');
const sheetsService = require('./services/sheetsService');
const createWebhookServer = require('./services/webhookServer');

const BookingBot   = require('./bots/bookingBot');
const QuoteBot     = require('./bots/quoteBot');
const EmergencyBot = require('./bots/emergencyBot');
const BotRouter    = require('./shared/botRouter');

async function main() {
  logger.info('=== SMS Trade Booking System Starting ===');
  logger.info('Client', { name: clientConfig.businessName, type: clientConfig.tradeType });

  // 1. Ensure Google Sheets tabs are set up
  try {
    await sheetsService.ensureSheetTabs(
      clientConfig.sheets.spreadsheetId,
      clientConfig.sheets.tabs
    );
    logger.info('Google Sheets ready');
  } catch (e) {
    logger.warn('Google Sheets setup failed (continuing without Sheets)', { error: e.message });
  }

  // 2. Instantiate bots
  const bots = {
    booking:   new BookingBot(clientConfig),
    quote:     new QuoteBot(clientConfig),
    emergency: new EmergencyBot(clientConfig)
  };
  logger.info('Bots initialised', { bots: Object.keys(bots) });

  // 3. Create router
  const router = new BotRouter(bots, clientConfig);

  // 4. Start webhook server
  const app = createWebhookServer(router, clientConfig);
  const port = process.env.PORT || 3000;

  app.listen(port, () => {
    logger.info(`Webhook server listening on port ${port}`);
    logger.info(`Twilio webhook URL: ${process.env.WEBHOOK_BASE_URL || `http://localhost:${port}`}/sms/inbound`);
    logger.info(`Dashboard API: http://localhost:${port}/api/stats`);
    logger.info('=== System Ready ===');
  });

  // 5. Graceful shutdown
  process.on('SIGINT', () => {
    logger.info('Shutting down...');
    process.exit(0);
  });
}

main().catch(err => {
  logger.error('Fatal startup error', { error: err.message, stack: err.stack });
  process.exit(1);
});

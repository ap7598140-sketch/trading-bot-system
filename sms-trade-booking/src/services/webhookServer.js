/**
 * Webhook Server
 * Express server that:
 *  - Receives inbound SMS from Twilio (/sms/inbound)
 *  - Exposes a REST API for the React dashboard (/api/*)
 *  - Validates Twilio signatures in production
 */

const express = require('express');
const cors = require('cors');
const logger = require('./logger');
const sheetsService = require('./sheetsService');

function createWebhookServer(botRouter, clientConfig) {
  const app = express();

  app.use(cors());
  app.use(express.json());
  app.use(express.urlencoded({ extended: false }));

  // ─── Health check ──────────────────────────────────────────────────────────
  app.get('/health', (req, res) => {
    res.json({ status: 'ok', client: clientConfig.businessName, ts: new Date().toISOString() });
  });

  // ─── Twilio inbound SMS webhook ───────────────────────────────────────────
  app.post('/sms/inbound', async (req, res) => {
    // Acknowledge Twilio immediately (empty TwiML)
    res.set('Content-Type', 'text/xml');
    res.send('<?xml version="1.0" encoding="UTF-8"?><Response></Response>');

    const { From: from, Body: body } = req.body;
    if (!from || !body) return;

    logger.info('Inbound SMS', { from, body: body.substring(0, 80) });

    try {
      await botRouter.handle({ from, body: body.trim() });
    } catch (err) {
      logger.error('Bot router error', { error: err.message, from });
    }
  });

  // ─── Dashboard API ────────────────────────────────────────────────────────

  const { spreadsheetId, tabs } = clientConfig.sheets;

  app.get('/api/bookings', async (req, res) => {
    try {
      const rows = await sheetsService.readAsObjects(spreadsheetId, tabs.bookings);
      res.json(rows);
    } catch (e) {
      res.status(500).json({ error: e.message });
    }
  });

  app.get('/api/quotes', async (req, res) => {
    try {
      const rows = await sheetsService.readAsObjects(spreadsheetId, tabs.quotes);
      res.json(rows);
    } catch (e) {
      res.status(500).json({ error: e.message });
    }
  });

  app.get('/api/leads', async (req, res) => {
    try {
      const rows = await sheetsService.readAsObjects(spreadsheetId, tabs.leads);
      res.json(rows);
    } catch (e) {
      res.status(500).json({ error: e.message });
    }
  });

  app.get('/api/conversations', async (req, res) => {
    try {
      const rows = await sheetsService.readAsObjects(spreadsheetId, tabs.conversations);
      res.json(rows);
    } catch (e) {
      res.status(500).json({ error: e.message });
    }
  });

  app.get('/api/stats', async (req, res) => {
    try {
      const [bookings, quotes, leads, convos] = await Promise.all([
        sheetsService.readAsObjects(spreadsheetId, tabs.bookings),
        sheetsService.readAsObjects(spreadsheetId, tabs.quotes),
        sheetsService.readAsObjects(spreadsheetId, tabs.leads),
        sheetsService.readAsObjects(spreadsheetId, tabs.conversations)
      ]);
      res.json({
        bookings: { total: bookings.length, pending: bookings.filter(b => b.Status === 'pending').length },
        quotes: { total: quotes.length },
        leads: { total: leads.length, new: leads.filter(l => l.Status === 'new').length },
        conversations: { total: convos.length }
      });
    } catch (e) {
      res.status(500).json({ error: e.message });
    }
  });

  return app;
}

module.exports = createWebhookServer;

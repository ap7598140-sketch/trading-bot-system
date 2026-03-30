# SMS Trade Booking Bot System

Universal SMS automation for trade businesses (plumbers, electricians, builders, roofers, etc.)

## Architecture

```
Twilio SMS
    │
    ▼
Webhook Server (Express :3000)
    │
    ▼
Bot Router ──► Emergency Bot  ──► Owner SMS alert (instant)
    │
    ├──► Booking Bot    ──► Multi-turn SMS conversation
    │                        └── Google Sheets (Bookings tab)
    │
    ├──► Quote Bot      ──► Instant price reply
    │                        └── Google Sheets (Quotes tab)
    │
    └──► (all bots log) ──► Google Sheets (Conversations tab)

Lead Gen Bot (standalone / n8n scheduled)
    ├── Google Maps scraper
    ├── Facebook public posts scraper
    └── Google Sheets (Leads tab) + Owner SMS alert
```

## Quick Start

### 1. Clone & Install
```bash
cd sms-trade-booking
npm install          # backend deps
cd dashboard && npm install && cd ..
```

### 2. Configure Environment
```bash
cp .env.example .env
# Fill in:
#  TWILIO_ACCOUNT_SID / AUTH_TOKEN / PHONE_NUMBER
#  ANTHROPIC_API_KEY
#  GOOGLE_SERVICE_ACCOUNT_EMAIL / PRIVATE_KEY / SPREADSHEET_ID
#  WEBHOOK_BASE_URL (your ngrok/server URL)
#  ACTIVE_CLIENT (e.g. example-plumber)
```

### 3. Set Up Google Sheets
1. Create a new Google Spreadsheet
2. Share it with your service account email (Editor)
3. Copy the spreadsheet ID to `GOOGLE_SPREADSHEET_ID` in `.env`
4. The system auto-creates: **Bookings | Quotes | Leads | Conversations** tabs on startup

### 4. Configure Your Client
```bash
cp config/clients/example-plumber.config.js config/clients/my-client.config.js
# Edit my-client.config.js — set businessName, services, prices, hours, ownerPhone
# Set ACTIVE_CLIENT=my-client in .env
```

### 5. Expose Webhook (local dev)
```bash
# Install ngrok: https://ngrok.com
ngrok http 3000
# Copy the https URL to WEBHOOK_BASE_URL in .env
# Set Twilio webhook URL to: https://YOUR_URL/sms/inbound
```

### 6. Run
```bash
npm start            # starts backend + SMS webhook server
npm run dashboard    # starts React dashboard on :3001
npm run lead-gen     # run lead generator manually
```

## Bots

### Booking Bot
- Multi-turn SMS conversation to collect name → service → date → time
- Sends confirmation to customer + alert to owner
- Logs all bookings to Sheets

### Quote Bot
- Instant price estimates based on `services[]` in client config
- Keyword matching first (fast), Claude fallback for ambiguous requests
- Logs all quotes to Sheets

### Emergency Bot
- Detects emergency keywords in under 100ms (no AI wait)
- Sends customer reassurance immediately
- Alerts owner with urgency level, issue summary, safety risk flag
- After-hours surcharge mentioned automatically

### Lead Generator Bot
- Scrapes Google Maps for trade-related search results in service areas
- Scrapes public Facebook posts in local community groups
- Claude qualifies and scores each lead (urgency, service needed)
- Alerts owner via SMS when high-urgency leads are found
- Run standalone: `npm run lead-gen` or schedule via n8n (workflow included)

## Adding a New Client

1. Copy `config/clients/example-plumber.config.js` → `config/clients/<slug>.config.js`
2. Fill in all fields (businessName, tradeType, services, hours, ownerPhone, etc.)
3. Set `ACTIVE_CLIENT=<slug>` in `.env`
4. Restart the server

One server = one client. To run multiple clients simultaneously, clone the repo and run separate instances on different ports with different `.env` files.

## n8n Automation Workflows

Import these from `n8n/workflows/`:

| Workflow | Trigger | Action |
|----------|---------|--------|
| `lead-gen-workflow.json` | Every 12h | Runs lead gen bot |
| `booking-reminder-workflow.json` | Every 1h | Sends 24h reminders |

## Google Sheets Schema

| Tab | Columns |
|-----|---------|
| Bookings | ID, Timestamp, CustomerName, CustomerPhone, Service, PreferredDate, PreferredTime, Status, Notes, ConfirmedAt |
| Quotes | ID, Timestamp, CustomerName, CustomerPhone, Service, PriceFrom, PriceTo, PriceUnit, Description, ConvertedToBooking |
| Leads | ID, Timestamp, Source, Name, Phone, Email, Location, ServiceNeeded, PostedText, Status, Notes |
| Conversations | ID, Timestamp, CustomerPhone, Direction, Message, BotType, Intent |

## Dashboard

React app at `http://localhost:3001`

- **Stats cards** — live totals for bookings, quotes, leads, messages
- **Activity chart** — daily activity for last 7 days
- **Bookings tab** — sortable/filterable table with status badges
- **Quotes tab** — all price enquiries with conversion tracking
- **Leads tab** — Google Maps + Facebook leads with urgency indicators
- **Conversations tab** — per-customer SMS thread view with bot type labelling

## Tech Stack

| Layer | Tech |
|-------|------|
| SMS | Twilio |
| AI | Claude Haiku (`claude-haiku-4-5-20251001`) |
| Scraping | Puppeteer |
| Data | Google Sheets API v4 |
| Automation | n8n |
| Backend | Node.js + Express |
| Frontend | React + Recharts |

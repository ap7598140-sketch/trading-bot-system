/**
 * Client Config: Mike's Plumbing Services
 * Copy this file to create a new client. Rename it <slug>.config.js
 * and set ACTIVE_CLIENT=<slug> in your .env file.
 */

module.exports = {
  // ─── Business Identity ──────────────────────────────────────────────────────
  businessName: "Mike's Plumbing Services",
  tradeType: 'plumber',           // plumber | electrician | builder | roofer | etc.
  ownerName: 'Mike Johnson',
  ownerPhone: '+15551234567',     // Owner's mobile — receives emergency alerts

  // ─── Location ───────────────────────────────────────────────────────────────
  location: {
    suburb: 'Bondi Beach',
    city: 'Sydney',
    state: 'NSW',
    postcode: '2026',
    serviceRadius: 20,            // km radius from suburb centre
    serviceAreas: [               // List of covered suburbs for lead gen
      'Bondi', 'Bondi Beach', 'Bondi Junction', 'Bronte', 'Coogee',
      'Randwick', 'Maroubra', 'Kingsford', 'Mascot', 'Rosebery'
    ]
  },

  // ─── Services & Pricing ─────────────────────────────────────────────────────
  services: [
    {
      id: 'leaking_tap',
      name: 'Leaking Tap Repair',
      keywords: ['leaking tap', 'dripping tap', 'tap repair', 'tap fix'],
      price: { from: 120, to: 180, unit: 'fixed' },
      duration: 60,               // minutes
      emergency: false
    },
    {
      id: 'blocked_drain',
      name: 'Blocked Drain',
      keywords: ['blocked drain', 'clogged drain', 'blocked sink', 'drain blocked'],
      price: { from: 150, to: 350, unit: 'fixed' },
      duration: 90,
      emergency: true
    },
    {
      id: 'burst_pipe',
      name: 'Burst Pipe',
      keywords: ['burst pipe', 'broken pipe', 'pipe burst', 'flooding'],
      price: { from: 250, unit: 'from' },
      duration: 120,
      emergency: true
    },
    {
      id: 'hot_water',
      name: 'Hot Water System',
      keywords: ['hot water', 'no hot water', 'water heater', 'hot water system'],
      price: { from: 180, to: 1200, unit: 'range' },
      duration: 180,
      emergency: true
    },
    {
      id: 'toilet_repair',
      name: 'Toilet Repair',
      keywords: ['toilet', 'toilet repair', 'toilet broken', 'running toilet', 'toilet flush'],
      price: { from: 130, to: 250, unit: 'fixed' },
      duration: 60,
      emergency: false
    },
    {
      id: 'bathroom_reno',
      name: 'Bathroom Renovation',
      keywords: ['bathroom renovation', 'bathroom reno', 'new bathroom', 'bathroom install'],
      price: { from: 3000, unit: 'from' },
      duration: null,             // null = quote required
      emergency: false
    },
    {
      id: 'general_callout',
      name: 'General Callout',
      keywords: ['plumber', 'plumbing', 'help', 'issue', 'problem'],
      price: { from: 120, unit: 'callout' },
      duration: null,
      emergency: false
    }
  ],

  // ─── Business Hours ─────────────────────────────────────────────────────────
  hours: {
    monday:    { open: '07:00', close: '17:00' },
    tuesday:   { open: '07:00', close: '17:00' },
    wednesday: { open: '07:00', close: '17:00' },
    thursday:  { open: '07:00', close: '17:00' },
    friday:    { open: '07:00', close: '17:00' },
    saturday:  { open: '08:00', close: '13:00' },
    sunday:    null,                              // null = closed
    publicHolidays: null,                         // null = closed
    emergencyCallout: true,                       // 24/7 emergency available
    emergencySurcharge: 100                       // $ added to price after hours
  },

  // ─── Booking Settings ───────────────────────────────────────────────────────
  booking: {
    slotDurationMinutes: 60,
    advanceBookingDays: 30,
    confirmationMessage: true,
    reminderHoursBefore: 24
  },

  // ─── Bot Personality ────────────────────────────────────────────────────────
  bot: {
    name: 'PlumbBot',             // Name the bot introduces itself as
    greeting: "Hi! I'm PlumbBot from Mike's Plumbing. How can I help you today? 🔧",
    tone: 'friendly and professional',
    signOff: "Mike's Plumbing — Reliable. Fast. Affordable."
  },

  // ─── Emergency Keywords ─────────────────────────────────────────────────────
  // Any message containing these triggers immediate owner alert
  emergencyKeywords: [
    'emergency', 'urgent', 'burst pipe', 'flooding', 'flood',
    'no water', 'water everywhere', 'gas leak', 'sewage', 'overflow',
    'help asap', 'help now', 'immediately', 'critical', 'disaster'
  ],

  // ─── Google Sheets ──────────────────────────────────────────────────────────
  sheets: {
    spreadsheetId: process.env.GOOGLE_SPREADSHEET_ID,
    tabs: {
      bookings:      'Bookings',
      quotes:        'Quotes',
      leads:         'Leads',
      conversations: 'Conversations'
    }
  }
};

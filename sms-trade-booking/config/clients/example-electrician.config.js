/**
 * Client Config: Sparky's Electrical Services
 * Example config for an electrician. Set ACTIVE_CLIENT=example-electrician
 */

module.exports = {
  businessName: "Sparky's Electrical",
  tradeType: 'electrician',
  ownerName: 'Dave Sparks',
  ownerPhone: '+15559876543',

  location: {
    suburb: 'Newtown',
    city: 'Sydney',
    state: 'NSW',
    postcode: '2042',
    serviceRadius: 25,
    serviceAreas: [
      'Newtown', 'Glebe', 'Leichhardt', 'Annandale', 'Petersham',
      'Marrickville', 'Sydenham', 'Tempe', 'St Peters', 'Erskineville'
    ]
  },

  services: [
    {
      id: 'power_point',
      name: 'Power Point Installation',
      keywords: ['power point', 'outlet', 'power outlet', 'power socket'],
      price: { from: 150, to: 250, unit: 'fixed' },
      duration: 60,
      emergency: false
    },
    {
      id: 'switchboard',
      name: 'Switchboard Upgrade',
      keywords: ['switchboard', 'fuse box', 'circuit breaker', 'main switch'],
      price: { from: 800, to: 2500, unit: 'range' },
      duration: 240,
      emergency: false
    },
    {
      id: 'no_power',
      name: 'No Power / Fault Finding',
      keywords: ['no power', 'power out', 'lights out', 'tripped', 'blackout', 'power gone'],
      price: { from: 150, unit: 'from' },
      duration: 90,
      emergency: true
    },
    {
      id: 'safety_inspection',
      name: 'Safety Inspection / Cert',
      keywords: ['safety check', 'inspection', 'electrical certificate', 'pre-sale', 'compliance'],
      price: { from: 200, to: 400, unit: 'range' },
      duration: 120,
      emergency: false
    },
    {
      id: 'lighting',
      name: 'Lighting Installation',
      keywords: ['lights', 'lighting', 'downlights', 'led', 'pendant', 'light fitting'],
      price: { from: 120, to: 80, unit: 'per_point' },
      duration: 60,
      emergency: false
    },
    {
      id: 'ev_charger',
      name: 'EV Charger Installation',
      keywords: ['ev charger', 'electric car', 'car charger', 'tesla charger', 'ev point'],
      price: { from: 900, to: 2000, unit: 'range' },
      duration: 240,
      emergency: false
    }
  ],

  hours: {
    monday:    { open: '07:30', close: '17:30' },
    tuesday:   { open: '07:30', close: '17:30' },
    wednesday: { open: '07:30', close: '17:30' },
    thursday:  { open: '07:30', close: '17:30' },
    friday:    { open: '07:30', close: '17:30' },
    saturday:  { open: '08:00', close: '14:00' },
    sunday:    null,
    publicHolidays: null,
    emergencyCallout: true,
    emergencySurcharge: 120
  },

  booking: {
    slotDurationMinutes: 60,
    advanceBookingDays: 21,
    confirmationMessage: true,
    reminderHoursBefore: 24
  },

  bot: {
    name: 'SparkBot',
    greeting: "Hi! I'm SparkBot from Sparky's Electrical ⚡ What can I help you with today?",
    tone: 'friendly, knowledgeable and safety-focused',
    signOff: "Sparky's Electrical — Licensed. Reliable. Safe."
  },

  emergencyKeywords: [
    'emergency', 'urgent', 'no power', 'power out', 'sparking', 'sparks',
    'burning smell', 'smoke', 'fire', 'shock', 'electric shock', 'tripped',
    'all power gone', 'help asap', 'immediately', 'dangerous', 'electrocuted'
  ],

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

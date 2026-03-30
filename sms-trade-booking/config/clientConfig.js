/**
 * Client Config Loader
 * Loads the active client config from config/clients/<ACTIVE_CLIENT>.config.js
 * All bots consume this single config object at runtime.
 */

const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '../.env') });

function loadClientConfig() {
  const clientName = process.env.ACTIVE_CLIENT || 'example-plumber';
  const configPath = path.join(__dirname, 'clients', `${clientName}.config.js`);

  let config;
  try {
    config = require(configPath);
  } catch (e) {
    throw new Error(
      `Cannot load client config "${clientName}". ` +
      `Expected file at: ${configPath}\n` +
      `Set ACTIVE_CLIENT in .env to match a file in config/clients/`
    );
  }

  validateConfig(config);
  return config;
}

function validateConfig(config) {
  const required = [
    'businessName', 'tradeType', 'ownerPhone',
    'services', 'hours', 'location'
  ];
  for (const key of required) {
    if (!config[key]) {
      throw new Error(`Client config missing required field: "${key}"`);
    }
  }
}

module.exports = loadClientConfig();

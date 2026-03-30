const BASE = process.env.REACT_APP_API_URL || '';

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`API ${path} failed: ${res.status}`);
  return res.json();
}

export const api = {
  stats:         () => get('/api/stats'),
  bookings:      () => get('/api/bookings'),
  quotes:        () => get('/api/quotes'),
  leads:         () => get('/api/leads'),
  conversations: () => get('/api/conversations')
};

const COLORS = {
  pending:   { bg: '#78350f', text: '#fcd34d' },
  confirmed: { bg: '#14532d', text: '#86efac' },
  completed: { bg: '#1e3a5f', text: '#93c5fd' },
  cancelled: { bg: '#3b1f1f', text: '#fca5a5' },
  new:       { bg: '#1e3a5f', text: '#93c5fd' },
  urgent:    { bg: '#7f1d1d', text: '#fca5a5' },
  contacted: { bg: '#14532d', text: '#86efac' },
  inbound:   { bg: '#1e293b', text: '#94a3b8' },
  outbound:  { bg: '#0f2a1e', text: '#6ee7b7' }
};

export function StatusBadge({ value }) {
  const lower = (value || '').toLowerCase();
  const c = COLORS[lower] || { bg: '#1e293b', text: '#94a3b8' };
  return (
    <span style={{
      background: c.bg,
      color: c.text,
      borderRadius: 4,
      padding: '2px 8px',
      fontSize: 11,
      fontWeight: 600,
      textTransform: 'uppercase',
      letterSpacing: '0.05em'
    }}>
      {value || '—'}
    </span>
  );
}

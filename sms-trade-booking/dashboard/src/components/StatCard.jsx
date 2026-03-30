export function StatCard({ label, value, sub, color = '#3b82f6', icon }) {
  return (
    <div style={{
      background: '#1e293b',
      border: `1px solid ${color}33`,
      borderRadius: 12,
      padding: '20px 24px',
      display: 'flex',
      flexDirection: 'column',
      gap: 6,
      minWidth: 160
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        {icon && <span style={{ fontSize: 20 }}>{icon}</span>}
        <span style={{ color: '#94a3b8', fontSize: 13, fontWeight: 500 }}>{label}</span>
      </div>
      <div style={{ fontSize: 36, fontWeight: 700, color }}>{value ?? '—'}</div>
      {sub && <div style={{ color: '#64748b', fontSize: 12 }}>{sub}</div>}
    </div>
  );
}

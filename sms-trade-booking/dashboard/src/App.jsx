import { useState, useCallback } from 'react';
import { usePolling } from './hooks/usePolling';
import { api } from './services/api';
import { StatCard } from './components/StatCard';
import { BookingsPanel } from './components/BookingsPanel';
import { QuotesPanel } from './components/QuotesPanel';
import { LeadsPanel } from './components/LeadsPanel';
import { ConversationsPanel } from './components/ConversationsPanel';
import { ActivityChart } from './components/ActivityChart';

const TABS = [
  { id: 'bookings',      label: 'Bookings',      icon: '📅' },
  { id: 'quotes',        label: 'Quotes',        icon: '💰' },
  { id: 'leads',         label: 'Leads',         icon: '🔍' },
  { id: 'conversations', label: 'Conversations', icon: '💬' }
];

const styles = {
  root: { minHeight: '100vh', background: '#0f172a', color: '#e2e8f0', fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif' },
  header: { borderBottom: '1px solid #1e293b', padding: '16px 32px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  logo: { fontSize: 18, fontWeight: 700, color: '#f8fafc', letterSpacing: '-0.01em' },
  live: { display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: '#10b981' },
  dot: { width: 8, height: 8, borderRadius: '50%', background: '#10b981', animation: 'pulse 2s infinite' },
  main: { padding: '24px 32px', maxWidth: 1400, margin: '0 auto' },
  statsRow: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16, marginBottom: 28 },
  chartCard: { background: '#1e293b', borderRadius: 12, padding: '20px 24px', marginBottom: 28 },
  chartTitle: { color: '#94a3b8', fontSize: 13, fontWeight: 600, marginBottom: 16 },
  tabBar: { display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #1e293b', paddingBottom: 0 },
  tab: (active) => ({
    padding: '10px 18px',
    fontSize: 13,
    fontWeight: active ? 600 : 400,
    color: active ? '#f8fafc' : '#64748b',
    cursor: 'pointer',
    border: 'none',
    background: 'none',
    borderBottom: active ? '2px solid #3b82f6' : '2px solid transparent',
    marginBottom: -1,
    transition: 'color 0.15s'
  }),
  panel: { background: '#1e293b', borderRadius: 12, padding: 24 }
};

export default function App() {
  const [activeTab, setActiveTab] = useState('bookings');

  const statsFn = useCallback(() => api.stats(), []);
  const bookingsFn = useCallback(() => api.bookings(), []);
  const quotesFn = useCallback(() => api.quotes(), []);
  const leadsFn = useCallback(() => api.leads(), []);

  const { data: stats } = usePolling(statsFn, 30000);
  const { data: bookings } = usePolling(bookingsFn, 30000);
  const { data: quotes } = usePolling(quotesFn, 30000);
  const { data: leads } = usePolling(leadsFn, 30000);

  return (
    <div style={styles.root}>
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>

      {/* Header */}
      <header style={styles.header}>
        <div>
          <div style={styles.logo}>🔧 Trade Bot Dashboard</div>
          <div style={{ color: '#64748b', fontSize: 12, marginTop: 2 }}>SMS Automation System</div>
        </div>
        <div style={styles.live}>
          <div style={styles.dot} />
          Live — auto-refreshes every 30s
        </div>
      </header>

      <main style={styles.main}>
        {/* Stats row */}
        <div style={styles.statsRow}>
          <StatCard
            label="Total Bookings"
            value={stats?.bookings?.total ?? '—'}
            sub={`${stats?.bookings?.pending ?? 0} pending`}
            color="#3b82f6"
            icon="📅"
          />
          <StatCard
            label="Quotes Sent"
            value={stats?.quotes?.total ?? '—'}
            color="#8b5cf6"
            icon="💰"
          />
          <StatCard
            label="Leads Found"
            value={stats?.leads?.total ?? '—'}
            sub={`${stats?.leads?.new ?? 0} new`}
            color="#10b981"
            icon="🔍"
          />
          <StatCard
            label="SMS Messages"
            value={stats?.conversations?.total ?? '—'}
            color="#f59e0b"
            icon="💬"
          />
        </div>

        {/* Activity chart */}
        <div style={styles.chartCard}>
          <div style={styles.chartTitle}>Activity — Last 7 Days</div>
          <ActivityChart
            bookings={bookings || []}
            quotes={quotes || []}
            leads={leads || []}
          />
          <div style={{ display: 'flex', gap: 20, marginTop: 12, fontSize: 11, color: '#64748b' }}>
            <span><span style={{ color: '#3b82f6' }}>■</span> Bookings</span>
            <span><span style={{ color: '#8b5cf6' }}>■</span> Quotes</span>
            <span><span style={{ color: '#10b981' }}>■</span> Leads</span>
          </div>
        </div>

        {/* Tab panels */}
        <div style={styles.tabBar}>
          {TABS.map(tab => (
            <button
              key={tab.id}
              style={styles.tab(activeTab === tab.id)}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.icon} {tab.label}
            </button>
          ))}
        </div>

        <div style={styles.panel}>
          {activeTab === 'bookings'      && <BookingsPanel />}
          {activeTab === 'quotes'        && <QuotesPanel />}
          {activeTab === 'leads'         && <LeadsPanel />}
          {activeTab === 'conversations' && <ConversationsPanel />}
        </div>
      </main>
    </div>
  );
}

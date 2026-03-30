import { useCallback, useState } from 'react';
import { usePolling } from '../hooks/usePolling';
import { api } from '../services/api';
import { StatusBadge } from './StatusBadge';

const BOT_COLORS = {
  booking:   '#3b82f6',
  quote:     '#8b5cf6',
  emergency: '#ef4444'
};

function ConversationThread({ phone, messages }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {messages.map((msg, i) => {
        const isOutbound = msg.Direction === 'outbound';
        const botColor = BOT_COLORS[msg.BotType] || '#64748b';
        return (
          <div key={i} style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: isOutbound ? 'flex-end' : 'flex-start',
          }}>
            <div style={{
              background: isOutbound ? '#1e3a5f' : '#1e293b',
              border: `1px solid ${isOutbound ? '#3b82f633' : '#334155'}`,
              borderRadius: 8,
              padding: '8px 12px',
              maxWidth: '80%',
              fontSize: 13
            }}>
              <div style={{ color: '#e2e8f0', lineHeight: 1.5 }}>{msg.Message}</div>
              <div style={{ display: 'flex', gap: 8, marginTop: 4, alignItems: 'center' }}>
                <span style={{ color: '#475569', fontSize: 11 }}>
                  {msg.Timestamp ? new Date(msg.Timestamp).toLocaleTimeString() : ''}
                </span>
                {msg.BotType && (
                  <span style={{ color: botColor, fontSize: 10, fontWeight: 600, textTransform: 'uppercase' }}>
                    {msg.BotType}
                  </span>
                )}
                {msg.Intent && (
                  <span style={{ color: '#475569', fontSize: 10 }}>{msg.Intent}</span>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function ConversationsPanel() {
  const fetchFn = useCallback(() => api.conversations(), []);
  const { data, loading, error } = usePolling(fetchFn, 15000);
  const [selected, setSelected] = useState(null);

  if (loading) return <div style={{ color: '#475569', padding: 24 }}>Loading conversations...</div>;
  if (error)   return <div style={{ color: '#f87171', padding: 24 }}>Error: {error}</div>;

  const rows = data || [];

  // Group by phone number
  const grouped = rows.reduce((acc, row) => {
    const phone = row.CustomerPhone || 'unknown';
    if (!acc[phone]) acc[phone] = [];
    acc[phone].push(row);
    return acc;
  }, {});

  const phones = Object.keys(grouped).sort((a, b) => {
    const aLast = grouped[a][grouped[a].length - 1];
    const bLast = grouped[b][grouped[b].length - 1];
    return new Date(bLast.Timestamp) - new Date(aLast.Timestamp);
  });

  const activePhone = selected || phones[0];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: 16, height: 500 }}>
      {/* Sidebar — conversation list */}
      <div style={{
        background: '#0f172a',
        borderRadius: 8,
        border: '1px solid #1e293b',
        overflowY: 'auto'
      }}>
        {phones.map(phone => {
          const msgs = grouped[phone];
          const lastMsg = msgs[msgs.length - 1];
          const isActive = phone === activePhone;
          const hasEmergency = msgs.some(m => m.BotType === 'emergency');
          return (
            <div
              key={phone}
              onClick={() => setSelected(phone)}
              style={{
                padding: '12px 14px',
                cursor: 'pointer',
                borderBottom: '1px solid #1e293b',
                background: isActive ? '#1e293b' : 'transparent',
                borderLeft: isActive ? '3px solid #3b82f6' : '3px solid transparent'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: '#e2e8f0', fontSize: 13, fontWeight: isActive ? 600 : 400 }}>
                  {hasEmergency ? '🚨 ' : ''}{phone}
                </span>
                <span style={{ color: '#475569', fontSize: 11 }}>{msgs.length} msgs</span>
              </div>
              <div style={{ color: '#64748b', fontSize: 11, marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {lastMsg?.Message?.substring(0, 50) || ''}
              </div>
              <div style={{ color: '#475569', fontSize: 10, marginTop: 2 }}>
                {lastMsg?.Timestamp ? new Date(lastMsg.Timestamp).toLocaleString() : ''}
              </div>
            </div>
          );
        })}
        {phones.length === 0 && (
          <div style={{ color: '#475569', padding: 24, textAlign: 'center', fontSize: 13 }}>
            No conversations yet
          </div>
        )}
      </div>

      {/* Main thread */}
      <div style={{
        background: '#0f172a',
        borderRadius: 8,
        border: '1px solid #1e293b',
        padding: 16,
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: 8
      }}>
        {activePhone && grouped[activePhone] ? (
          <>
            <div style={{ color: '#64748b', fontSize: 12, marginBottom: 8 }}>
              Thread: {activePhone} — {grouped[activePhone].length} messages
            </div>
            <ConversationThread
              phone={activePhone}
              messages={grouped[activePhone]}
            />
          </>
        ) : (
          <div style={{ color: '#475569', textAlign: 'center', marginTop: 60 }}>
            Select a conversation
          </div>
        )}
      </div>
    </div>
  );
}

import { useCallback } from 'react';
import { usePolling } from '../hooks/usePolling';
import { api } from '../services/api';
import { DataTable } from './DataTable';
import { StatusBadge } from './StatusBadge';

const SOURCE_ICONS = {
  'Google Maps': '🗺️',
  'Facebook': '📘',
  'SMS Emergency': '🚨'
};

const COLUMNS = [
  { key: 'Timestamp',     label: 'Found',        render: v => v ? new Date(v).toLocaleString() : '—' },
  { key: 'Source',        label: 'Source',       render: v => `${SOURCE_ICONS[v] || '📌'} ${v}` },
  { key: 'Name',          label: 'Name' },
  { key: 'Phone',         label: 'Phone' },
  { key: 'Location',      label: 'Location' },
  { key: 'ServiceNeeded', label: 'Service' },
  { key: 'PostedText',    label: 'Post',         render: v => v ? v.substring(0, 80) + (v.length > 80 ? '…' : '') : '—' },
  { key: 'Status',        label: 'Status',       render: v => <StatusBadge value={v} /> },
  { key: 'Notes',         label: 'Notes',        render: v => v ? v.substring(0, 60) : '—' }
];

export function LeadsPanel() {
  const fetchFn = useCallback(() => api.leads(), []);
  const { data, loading, error } = usePolling(fetchFn, 30000);

  if (loading) return <div style={{ color: '#475569', padding: 24 }}>Loading leads...</div>;
  if (error)   return <div style={{ color: '#f87171', padding: 24 }}>Error: {error}</div>;

  const rows = data || [];
  const urgent = rows.filter(r => r.Status === 'urgent').length;
  const newLeads = rows.filter(r => r.Status === 'new').length;

  return (
    <div>
      <div style={{ display: 'flex', gap: 16, marginBottom: 16 }}>
        <span style={{ color: '#64748b', fontSize: 13 }}>{rows.length} total leads</span>
        {newLeads > 0 && <span style={{ color: '#93c5fd', fontSize: 13 }}>🔵 {newLeads} new</span>}
        {urgent > 0  && <span style={{ color: '#fca5a5', fontSize: 13 }}>🚨 {urgent} urgent</span>}
      </div>
      <DataTable columns={COLUMNS} rows={rows} filterKey="ServiceNeeded" />
    </div>
  );
}

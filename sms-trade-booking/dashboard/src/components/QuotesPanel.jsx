import { useCallback } from 'react';
import { usePolling } from '../hooks/usePolling';
import { api } from '../services/api';
import { DataTable } from './DataTable';
import { StatusBadge } from './StatusBadge';

const COLUMNS = [
  { key: 'Timestamp',          label: 'Date',     render: v => v ? new Date(v).toLocaleString() : '—' },
  { key: 'CustomerPhone',      label: 'Phone' },
  { key: 'Service',            label: 'Service' },
  { key: 'PriceFrom',          label: 'From',     render: v => v ? `$${v}` : '—' },
  { key: 'PriceTo',            label: 'To',       render: v => v ? `$${v}` : '—' },
  { key: 'PriceUnit',          label: 'Unit' },
  { key: 'Description',        label: 'Request',  render: v => v ? v.substring(0, 60) + (v.length > 60 ? '…' : '') : '—' },
  { key: 'ConvertedToBooking', label: 'Booked?',  render: v => <StatusBadge value={v === 'Yes' ? 'confirmed' : 'pending'} /> }
];

export function QuotesPanel() {
  const fetchFn = useCallback(() => api.quotes(), []);
  const { data, loading, error } = usePolling(fetchFn, 30000);

  if (loading) return <div style={{ color: '#475569', padding: 24 }}>Loading quotes...</div>;
  if (error)   return <div style={{ color: '#f87171', padding: 24 }}>Error: {error}</div>;

  return (
    <div>
      <div style={{ marginBottom: 12, color: '#64748b', fontSize: 13 }}>
        {(data || []).length} total quotes
      </div>
      <DataTable columns={COLUMNS} rows={data || []} />
    </div>
  );
}

import { useCallback } from 'react';
import { usePolling } from '../hooks/usePolling';
import { api } from '../services/api';
import { DataTable } from './DataTable';
import { StatusBadge } from './StatusBadge';

const COLUMNS = [
  { key: 'Timestamp',     label: 'Date',         render: v => v ? new Date(v).toLocaleString() : '—' },
  { key: 'CustomerName',  label: 'Customer' },
  { key: 'CustomerPhone', label: 'Phone' },
  { key: 'Service',       label: 'Service' },
  { key: 'PreferredDate', label: 'Appt Date' },
  { key: 'PreferredTime', label: 'Time' },
  { key: 'Status',        label: 'Status', render: v => <StatusBadge value={v} /> },
  { key: 'Notes',         label: 'Notes' }
];

export function BookingsPanel() {
  const fetchFn = useCallback(() => api.bookings(), []);
  const { data, loading, error } = usePolling(fetchFn, 30000);

  if (loading) return <div style={{ color: '#475569', padding: 24 }}>Loading bookings...</div>;
  if (error)   return <div style={{ color: '#f87171', padding: 24 }}>Error: {error}</div>;

  return (
    <div>
      <div style={{ marginBottom: 12, color: '#64748b', fontSize: 13 }}>
        {(data || []).length} total bookings
      </div>
      <DataTable columns={COLUMNS} rows={data || []} filterKey="CustomerName" />
    </div>
  );
}

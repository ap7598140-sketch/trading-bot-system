import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

/**
 * Shows bookings/quotes/leads per day for the last 7 days.
 */
export function ActivityChart({ bookings = [], quotes = [], leads = [] }) {
  // Build last 7 days
  const days = [];
  for (let i = 6; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    days.push(d.toISOString().slice(0, 10));
  }

  function countByDay(rows) {
    const counts = {};
    days.forEach(d => { counts[d] = 0; });
    rows.forEach(row => {
      const day = (row.Timestamp || '').slice(0, 10);
      if (counts[day] !== undefined) counts[day]++;
    });
    return counts;
  }

  const bCounts = countByDay(bookings);
  const qCounts = countByDay(quotes);
  const lCounts = countByDay(leads);

  const chartData = days.map(d => ({
    day: d.slice(5),   // MM-DD
    Bookings: bCounts[d],
    Quotes:   qCounts[d],
    Leads:    lCounts[d]
  }));

  const tooltipStyle = {
    background: '#1e293b',
    border: '1px solid #334155',
    borderRadius: 8,
    color: '#e2e8f0',
    fontSize: 12
  };

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis dataKey="day" tick={{ fill: '#64748b', fontSize: 11 }} />
        <YAxis tick={{ fill: '#64748b', fontSize: 11 }} allowDecimals={false} />
        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: '#1e293b' }} />
        <Bar dataKey="Bookings" fill="#3b82f6" radius={[3, 3, 0, 0]} />
        <Bar dataKey="Quotes"   fill="#8b5cf6" radius={[3, 3, 0, 0]} />
        <Bar dataKey="Leads"    fill="#10b981" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

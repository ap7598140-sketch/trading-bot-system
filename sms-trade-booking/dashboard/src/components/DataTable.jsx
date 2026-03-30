/**
 * Generic sortable/filterable data table.
 */
import { useState } from 'react';

const styles = {
  wrapper: { overflowX: 'auto' },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 13 },
  th: {
    background: '#0f172a',
    color: '#64748b',
    padding: '10px 14px',
    textAlign: 'left',
    fontWeight: 600,
    borderBottom: '1px solid #1e293b',
    cursor: 'pointer',
    whiteSpace: 'nowrap',
    userSelect: 'none'
  },
  td: {
    padding: '10px 14px',
    borderBottom: '1px solid #1e293b',
    color: '#e2e8f0',
    maxWidth: 260,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap'
  },
  trHover: { background: '#1e293b' },
  filterInput: {
    background: '#1e293b',
    border: '1px solid #334155',
    borderRadius: 6,
    color: '#e2e8f0',
    padding: '6px 10px',
    fontSize: 13,
    marginBottom: 12,
    width: '100%',
    maxWidth: 320,
    outline: 'none'
  },
  empty: { textAlign: 'center', color: '#475569', padding: 32 }
};

export function DataTable({ columns, rows, filterKey }) {
  const [filter, setFilter] = useState('');
  const [sortCol, setSortCol] = useState(null);
  const [sortAsc, setSortAsc] = useState(true);
  const [hovered, setHovered] = useState(null);

  const filtered = filter
    ? rows.filter(r => {
        const val = filterKey ? r[filterKey] : Object.values(r).join(' ');
        return val.toLowerCase().includes(filter.toLowerCase());
      })
    : rows;

  const sorted = sortCol
    ? [...filtered].sort((a, b) => {
        const av = a[sortCol] || '';
        const bv = b[sortCol] || '';
        return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
      })
    : filtered;

  function handleSort(col) {
    if (sortCol === col) setSortAsc(!sortAsc);
    else { setSortCol(col); setSortAsc(true); }
  }

  return (
    <div>
      <input
        style={styles.filterInput}
        placeholder="Search..."
        value={filter}
        onChange={e => setFilter(e.target.value)}
      />
      <div style={styles.wrapper}>
        <table style={styles.table}>
          <thead>
            <tr>
              {columns.map(col => (
                <th
                  key={col.key}
                  style={styles.th}
                  onClick={() => handleSort(col.key)}
                >
                  {col.label} {sortCol === col.key ? (sortAsc ? '↑' : '↓') : ''}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.length === 0 ? (
              <tr><td colSpan={columns.length} style={styles.empty}>No data</td></tr>
            ) : sorted.map((row, i) => (
              <tr
                key={i}
                style={hovered === i ? styles.trHover : {}}
                onMouseEnter={() => setHovered(i)}
                onMouseLeave={() => setHovered(null)}
              >
                {columns.map(col => (
                  <td key={col.key} style={styles.td} title={row[col.key] || ''}>
                    {col.render ? col.render(row[col.key], row) : (row[col.key] || '—')}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

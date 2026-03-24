/**
 * Output formatting utilities for the CLI.
 * Supports JSON, CSV, and table output formats.
 */

export type OutputFormat = 'json' | 'csv' | 'table';

export function outputData(
  data: unknown,
  format: OutputFormat,
  columns?: string[],
): string {
  switch (format) {
    case 'json':
      return formatJson(data);
    case 'csv':
      return formatCsv(data, columns);
    case 'table':
    default:
      return formatTable(data, columns);
  }
}

function formatJson(data: unknown): string {
  return JSON.stringify(data, null, 2);
}

function formatCsv(data: unknown, columns?: string[]): string {
  const rows = normalizeToArray(data);
  if (rows.length === 0) return '';
  const cols = columns || Object.keys(rows[0] as Record<string, unknown>);
  const header = cols.join(',');
  const body = rows
    .map((row) => {
      const record = row as Record<string, unknown>;
      return cols.map((col) => escapeCsvValue(record[col])).join(',');
    })
    .join('\n');
  return `${header}\n${body}`;
}

function escapeCsvValue(value: unknown): string {
  if (value === null || value === undefined) return '';
  const str = String(value);
  if (str.includes(',') || str.includes('"') || str.includes('\n')) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

function formatTable(data: unknown, columns?: string[]): string {
  const rows = normalizeToArray(data);
  if (rows.length === 0) return 'No data.';

  const cols = columns || Object.keys(rows[0] as Record<string, unknown>);
  const widths: Record<string, number> = {};

  for (const col of cols) {
    widths[col] = col.length;
  }

  const stringRows = rows.map((row) => {
    const record = row as Record<string, unknown>;
    const stringRow: Record<string, string> = {};
    for (const col of cols) {
      const val = truncate(formatValue(record[col]), 40);
      stringRow[col] = val;
      widths[col] = Math.max(widths[col], val.length);
    }
    return stringRow;
  });

  const header = cols.map((col) => col.padEnd(widths[col])).join('  ');
  const separator = cols.map((col) => '-'.repeat(widths[col])).join('  ');
  const body = stringRows
    .map((row) => cols.map((col) => (row[col] || '').padEnd(widths[col])).join('  '))
    .join('\n');

  return `${header}\n${separator}\n${body}`;
}

function normalizeToArray(data: unknown): unknown[] {
  if (Array.isArray(data)) return data;
  if (data && typeof data === 'object') {
    const obj = data as Record<string, unknown>;
    // Handle paginated responses
    if ('results' in obj && Array.isArray(obj.results)) return obj.results;
    if ('data' in obj && Array.isArray(obj.data)) return obj.data;
    if ('items' in obj && Array.isArray(obj.items)) return obj.items;
    return [data];
  }
  return [{ value: data }];
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function truncate(str: string, maxLen: number): string {
  if (str.length <= maxLen) return str;
  return str.slice(0, maxLen - 3) + '...';
}

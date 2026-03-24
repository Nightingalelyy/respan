/**
 * Pagination utilities for CLI list commands.
 */

export interface PaginationInfo {
  page: number;
  totalPages?: number;
  totalCount?: number;
  hasNext: boolean;
  hasPrevious: boolean;
}

export function extractPagination(response: unknown, currentPage: number): PaginationInfo {
  if (!response || typeof response !== 'object') {
    return { page: currentPage, hasNext: false, hasPrevious: currentPage > 1 };
  }

  const obj = response as Record<string, unknown>;
  const count = typeof obj.count === 'number' ? obj.count : undefined;
  const next = obj.next != null;
  const previous = obj.previous != null;
  const totalPages =
    typeof obj.total_pages === 'number'
      ? obj.total_pages
      : count && typeof obj.page_size === 'number'
        ? Math.ceil(count / obj.page_size)
        : undefined;

  return {
    page: currentPage,
    totalPages,
    totalCount: count,
    hasNext: next,
    hasPrevious: previous,
  };
}

export function formatPaginationInfo(info: PaginationInfo): string {
  const parts: string[] = [`Page ${info.page}`];
  if (info.totalPages) parts[0] += ` of ${info.totalPages}`;
  if (info.totalCount !== undefined) parts.push(`Total: ${info.totalCount}`);
  if (info.hasNext) parts.push('(more results available)');
  return parts.join('  |  ');
}

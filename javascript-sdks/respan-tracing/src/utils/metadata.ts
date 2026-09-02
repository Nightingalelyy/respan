type MetadataRecord = Record<string, unknown>;

/** Merge canonical metadata JSON without emitting respan.metadata.* aliases. */
export function mergeCanonicalMetadata(
  existing: unknown,
  propagated: unknown,
): string {
  return safeJson({
    ...metadataRecord(propagated),
    ...metadataRecord(existing),
  });
}

function metadataRecord(value: unknown): MetadataRecord {
  if (isRecord(value)) return { ...value };
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value) as unknown;
      return isRecord(parsed) ? { ...parsed } : { value: parsed };
    } catch {
      return { value };
    }
  }
  return value === undefined || value === null ? {} : { value };
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return JSON.stringify({ value: String(value) });
  }
}

function isRecord(value: unknown): value is MetadataRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

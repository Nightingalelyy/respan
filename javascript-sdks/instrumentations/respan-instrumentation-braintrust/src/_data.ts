import { types } from "node:util";

export type Row = Record<string, any>;

/** Read data properties only; instrumentation never calls customer accessors. */
export function data(value: unknown, key: string): any {
  if (!value || typeof value !== "object" || types.isProxy(value))
    return undefined;
  const descriptor = Object.getOwnPropertyDescriptor(value, key);
  return descriptor && "value" in descriptor ? descriptor.value : undefined;
}

export function primitive(value: unknown): value is string | number | boolean {
  return (
    typeof value === "string" ||
    typeof value === "boolean" ||
    (typeof value === "number" && Number.isFinite(value))
  );
}

export function string(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

/** JSON-compatible snapshot, with no arbitrary size or collection limits. */
export function snapshot(value: unknown, seen = new WeakSet<object>()): any {
  if (value === null || primitive(value)) return value;
  if (!value || typeof value !== "object" || types.isProxy(value))
    return undefined;
  if (seen.has(value)) return "[Circular]";
  seen.add(value);
  let result: any;
  if (Array.isArray(value)) {
    result = Array.from(
      { length: value.length },
      (_, index) => snapshot(data(value, String(index)), seen) ?? null,
    );
  } else {
    result = Object.create(null);
    for (const key of Object.keys(value)) {
      const item = snapshot(data(value, key), seen);
      if (item !== undefined) result[key] = item;
    }
  }
  seen.delete(value);
  return result;
}

export function json(value: unknown): string {
  return JSON.stringify(value) ?? "null";
}

export function merge(
  into: Row,
  from: Row,
  path: string[] = [],
  replace?: Set<string>,
): Row {
  const paths =
    replace ??
    new Set((from._merge_paths ?? []).map((item: unknown) => json(item)));
  for (const key of Object.keys(from)) {
    const existing = data(into, key);
    if (
      path.length === 0 &&
      [
        "id",
        "span_id",
        "root_span_id",
        "span_parents",
        "created",
        "_parent_id",
      ].includes(key) &&
      existing !== undefined
    )
      continue;
    const value = from[key];
    const next = [...path, key];
    if (
      key === "tags" &&
      path.length === 0 &&
      Array.isArray(existing) &&
      Array.isArray(value) &&
      !paths.has(json(next))
    ) {
      into[key] = [...new Set([...existing, ...value])];
    } else if (
      value &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      existing &&
      typeof existing === "object" &&
      !Array.isArray(existing) &&
      !paths.has(json(next))
    ) {
      merge(existing, value, next, paths);
    } else if (value !== undefined)
      Object.defineProperty(into, key, {
        value,
        configurable: true,
        enumerable: true,
        writable: true,
      });
  }
  return into;
}

export function time(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value))
    return value < 10_000_000_000 ? value * 1000 : value;
  if (typeof value === "string") {
    const result = Date.parse(value);
    return Number.isFinite(result) ? result : undefined;
  }
  return undefined;
}

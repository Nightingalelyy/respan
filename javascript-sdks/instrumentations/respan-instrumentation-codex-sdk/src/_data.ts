import { types } from "node:util";
export type Row = Record<string, unknown>;
export function data(value: unknown, key: string): unknown {
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
export function record(value: unknown): value is Row {
  return Boolean(
    value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    !types.isProxy(value),
  );
}
export function snapshot(
  value: unknown,
  seen = new WeakSet<object>(),
): unknown {
  if (primitive(value) || value === null) return value;
  if (typeof value === "bigint") return value.toString();
  if (!value || typeof value !== "object" || types.isProxy(value))
    return undefined;
  if (seen.has(value)) return "[Circular]";
  seen.add(value);
  const result: unknown = Array.isArray(value)
    ? Array.from({ length: Number(data(value, "length")) }, (_, index) =>
        snapshot(data(value, String(index)), seen),
      )
    : Object.fromEntries(
        Object.keys(value).map((k) => [k, snapshot(data(value, k), seen)]),
      );
  seen.delete(value);
  return result;
}
export function json(value: unknown): string {
  return JSON.stringify(snapshot(value)) ?? "null";
}

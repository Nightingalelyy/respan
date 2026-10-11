import { data, string, type Row } from "./_data.js";
import { types } from "node:util";

/** Reuse current native export customizers; retain the exact minimum's field semantics. */
export function maskMergedRow(logger: unknown, row: Row): Row {
  const customizers = data(logger, "exportCustomizers");
  if (Array.isArray(customizers)) {
    let result = row;
    for (const customizer of customizers) {
      const apply = data(customizer, "onSpanExport");
      if (typeof apply === "function") result = apply.call(customizer, result);
    }
    return result;
  }
  const mask = data(logger, "maskingFunction");
  if (typeof mask !== "function") return row;
  const result = { ...row };
  for (const field of [
    "input",
    "output",
    "expected",
    "metadata",
    "context",
    "scores",
    "metrics",
  ]) {
    if (row[field] === undefined) continue;
    try {
      result[field] = mask(row[field]);
    } catch (error) {
      const constructor =
        error instanceof Error && !types.isProxy(error)
          ? data(Object.getPrototypeOf(error), "constructor")
          : undefined;
      const type =
        typeof constructor === "function"
          ? (string(
              Object.getOwnPropertyDescriptor(constructor, "name")?.value,
            ) ?? "Error")
          : "Error";
      const message = `ERROR: Failed to mask field '${field}' - ${type}`;
      if (field === "scores" || field === "metrics") {
        delete result[field];
        result.error = result.error ? `${result.error}; ${message}` : message;
      } else
        result[field] = field === "metadata" ? { error: message } : message;
    }
  }
  return result;
}

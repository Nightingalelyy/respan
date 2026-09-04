/**
 * Best-effort detection of the pi version running in this process, without
 * importing pi. Walks up from the pi entry script (`process.argv[1]`) and the
 * working directory looking for `@earendil-works/pi-coding-agent/package.json`.
 * Read-only, cached, and silent on failure.
 */

import * as fs from "node:fs";
import * as path from "node:path";

const PI_PACKAGE_NAME = "@earendil-works/pi-coding-agent";
const MAX_DEPTH = 24;

let cached: string | undefined;
let resolved = false;

export function detectPiVersion(startPaths?: string[]): string | undefined {
  if (startPaths) {
    return findFromPaths(startPaths);
  }
  if (!resolved) {
    resolved = true;
    cached = findFromPaths([process.argv[1], process.cwd()]);
  }
  return cached;
}

function findFromPaths(startPaths: Array<string | undefined>): string | undefined {
  for (const start of startPaths) {
    if (!start) {
      continue;
    }
    try {
      const version = findPiVersionFrom(start);
      if (version) {
        return version;
      }
    } catch {
      // Ignore and try the next candidate.
    }
  }
  return undefined;
}

function findPiVersionFrom(start: string): string | undefined {
  let dir = start;
  try {
    if (!fs.statSync(dir).isDirectory()) {
      dir = path.dirname(dir);
    }
  } catch {
    dir = path.dirname(dir);
  }
  for (let depth = 0; depth < MAX_DEPTH; depth += 1) {
    const own = readPackageVersion(path.join(dir, "package.json"), true);
    if (own) {
      return own;
    }
    const nested = readPackageVersion(
      path.join(dir, "node_modules", ...PI_PACKAGE_NAME.split("/"), "package.json"),
      false,
    );
    if (nested) {
      return nested;
    }
    const parent = path.dirname(dir);
    if (parent === dir) {
      break;
    }
    dir = parent;
  }
  return undefined;
}

function readPackageVersion(file: string, requireName: boolean): string | undefined {
  try {
    const raw = JSON.parse(fs.readFileSync(file, "utf8")) as unknown;
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
      return undefined;
    }
    const record = raw as Record<string, unknown>;
    if (requireName && record.name !== PI_PACKAGE_NAME) {
      return undefined;
    }
    return typeof record.version === "string" && record.version ? record.version : undefined;
  } catch {
    return undefined;
  }
}

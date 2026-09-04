import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { normalizeBaseUrl, parseTraceScope, resolvePiRespanConfig } from "../dist/_config.js";

const tempDirs = [];

function makeTempDir(prefix) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  tempDirs.push(dir);
  return dir;
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2));
}

test.after(() => {
  for (const dir of tempDirs) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("precedence: defaults < global json < project json < env", () => {
  const homeDir = makeTempDir("respan-pi-home-");
  const cwd = makeTempDir("respan-pi-cwd-");
  writeJson(path.join(homeDir, ".pi", "agent", "respan.json"), {
    enabled: true,
    workflow_name: "global-wf",
    agent_name: "global-agent",
    customer_id: "global-cust",
    base_url: "https://global.example.com",
    metadata: { team: "platform", env: "global" },
  });
  writeJson(path.join(cwd, ".pi", "respan.json"), {
    workflow_name: "project-wf",
    base_url: "https://eu.respan.ai/",
    metadata: { env: "project", numeric: 3 },
  });

  const config = resolvePiRespanConfig({
    cwd,
    homeDir,
    env: { RESPAN_API_KEY: "sk-env", RESPAN_CUSTOMER_ID: "env-cust", RESPAN_PROJECT_ID: "proj-1" },
  });

  assert.equal(config.enabled, true);
  assert.equal(config.apiKey, "sk-env");
  assert.equal(config.workflowName, "project-wf");
  assert.equal(config.agentName, "global-agent");
  assert.equal(config.customerIdentifier, "env-cust");
  assert.equal(config.baseURL, "https://eu.respan.ai/api");
  assert.equal(config.projectId, "proj-1");
  assert.deepEqual(config.metadata, {
    team: "platform",
    env: "project",
    numeric: "3",
    project_id: "proj-1",
  });
  assert.deepEqual(config.sources, ["~/.pi/agent/respan.json", ".pi/respan.json", "env"]);
  assert.equal(config.debug, false);
});

test("span_name is an alias of workflow_name and RESPAN_PI_DEBUG enables debug", () => {
  const homeDir = makeTempDir("respan-pi-home-");
  const cwd = makeTempDir("respan-pi-cwd-");
  writeJson(path.join(cwd, ".pi", "respan.json"), { span_name: "legacy-name" });
  const config = resolvePiRespanConfig({
    cwd,
    homeDir,
    env: { RESPAN_API_KEY: "sk", RESPAN_PI_DEBUG: "1" },
  });
  assert.equal(config.workflowName, "legacy-name");
  assert.equal(config.debug, true);
  const off = resolvePiRespanConfig({ cwd, homeDir, env: { RESPAN_API_KEY: "sk", RESPAN_PI_DEBUG: "false" } });
  assert.equal(off.debug, false);
});

test("RESPAN_PI_TRACING and enabled flags disable/enable tracing", () => {
  const homeDir = makeTempDir("respan-pi-home-");
  const cwd = makeTempDir("respan-pi-cwd-");

  const envDisabled = resolvePiRespanConfig({
    cwd,
    homeDir,
    env: { RESPAN_API_KEY: "sk-test", RESPAN_PI_TRACING: "false" },
  });
  assert.equal(envDisabled.enabled, false);
  assert.equal(envDisabled.apiKey, "sk-test");

  for (const value of ["0", "off", "no", "FALSE"]) {
    assert.equal(
      resolvePiRespanConfig({ cwd, homeDir, env: { RESPAN_API_KEY: "sk", RESPAN_PI_TRACING: value } }).enabled,
      false,
      `RESPAN_PI_TRACING=${value}`,
    );
  }

  writeJson(path.join(cwd, ".pi", "respan.json"), { enabled: false });
  const jsonDisabled = resolvePiRespanConfig({ cwd, homeDir, env: { RESPAN_API_KEY: "sk-test" } });
  assert.equal(jsonDisabled.enabled, false);

  const envWins = resolvePiRespanConfig({
    cwd,
    homeDir,
    env: { RESPAN_API_KEY: "sk-test", RESPAN_PI_TRACING: "1" },
  });
  assert.equal(envWins.enabled, true);

  const projectWins = makeTempDir("respan-pi-cwd-");
  writeJson(path.join(homeDir, ".pi", "agent", "respan.json"), { enabled: false });
  writeJson(path.join(projectWins, ".pi", "respan.json"), { enabled: true });
  assert.equal(
    resolvePiRespanConfig({ cwd: projectWins, homeDir, env: { RESPAN_API_KEY: "sk-test" } }).enabled,
    true,
  );
});

test("trace_scope: global json < project json < env, unknown values ignored", () => {
  const homeDir = makeTempDir("respan-pi-home-");
  const cwd = makeTempDir("respan-pi-cwd-");
  const env = { RESPAN_API_KEY: "sk" };
  assert.equal(resolvePiRespanConfig({ cwd, homeDir, env }).traceScope, undefined);

  writeJson(path.join(homeDir, ".pi", "agent", "respan.json"), { trace_scope: "session" });
  assert.equal(resolvePiRespanConfig({ cwd, homeDir, env }).traceScope, "session");

  writeJson(path.join(cwd, ".pi", "respan.json"), { trace_scope: "Run" });
  assert.equal(resolvePiRespanConfig({ cwd, homeDir, env }).traceScope, "run");

  const fromEnv = resolvePiRespanConfig({ cwd, homeDir, env: { ...env, RESPAN_PI_TRACE_SCOPE: "session" } });
  assert.equal(fromEnv.traceScope, "session");
  assert.ok(fromEnv.sources.includes("env"));

  // Unknown values are ignored at every level (the previous level's value stands).
  writeJson(path.join(cwd, ".pi", "respan.json"), { trace_scope: "both" });
  assert.equal(resolvePiRespanConfig({ cwd, homeDir, env }).traceScope, "session");
  assert.equal(
    resolvePiRespanConfig({ cwd, homeDir, env: { ...env, RESPAN_PI_TRACE_SCOPE: "nope" } }).traceScope,
    "session",
  );
  assert.equal(parseTraceScope(" SESSION "), "session");
  assert.equal(parseTraceScope("run"), "run");
  assert.equal(parseTraceScope(1), undefined);
  assert.equal(parseTraceScope(undefined), undefined);
});

test("credentials fallback uses the active profile and its base URL", () => {
  const homeDir = makeTempDir("respan-pi-home-");
  const cwd = makeTempDir("respan-pi-cwd-");
  writeJson(path.join(homeDir, ".respan", "credentials.json"), {
    default: { apiKey: "sk-default", baseUrl: "https://api.respan.ai/api" },
    work: { accessToken: "tok-work", baseUrl: "https://work.example.com" },
  });

  const defaultProfile = resolvePiRespanConfig({ cwd, homeDir, env: {} });
  assert.equal(defaultProfile.enabled, true);
  assert.equal(defaultProfile.apiKey, "sk-default");
  assert.equal(defaultProfile.baseURL, "https://api.respan.ai/api");
  assert.deepEqual(defaultProfile.sources, ["~/.respan/credentials.json"]);

  writeJson(path.join(homeDir, ".respan", "config.json"), { activeProfile: "work" });
  const workProfile = resolvePiRespanConfig({ cwd, homeDir, env: {} });
  assert.equal(workProfile.apiKey, "tok-work");
  assert.equal(workProfile.baseURL, "https://work.example.com/api");

  // Env base URL wins over the credential's base URL; env key wins over the file.
  const envBase = resolvePiRespanConfig({ cwd, homeDir, env: { RESPAN_BASE_URL: "https://custom.example.com/" } });
  assert.equal(envBase.apiKey, "tok-work");
  assert.equal(envBase.baseURL, "https://custom.example.com/api");
  const envKey = resolvePiRespanConfig({ cwd, homeDir, env: { RESPAN_API_KEY: "sk-env" } });
  assert.equal(envKey.apiKey, "sk-env");
  assert.ok(!envKey.sources.includes("~/.respan/credentials.json"));

  // A missing profile means no key.
  writeJson(path.join(homeDir, ".respan", "config.json"), { activeProfile: "missing" });
  assert.equal(resolvePiRespanConfig({ cwd, homeDir, env: {} }).apiKey, undefined);
});

test("base URL normalization", () => {
  assert.equal(normalizeBaseUrl("https://x.example.com"), "https://x.example.com/api");
  assert.equal(normalizeBaseUrl("https://x.example.com/"), "https://x.example.com/api");
  assert.equal(normalizeBaseUrl("https://x.example.com/api"), "https://x.example.com/api");
  assert.equal(normalizeBaseUrl("https://x.example.com/api///"), "https://x.example.com/api");
  assert.equal(normalizeBaseUrl("   "), undefined);
  assert.equal(normalizeBaseUrl(undefined), undefined);
  const config = resolvePiRespanConfig({
    cwd: makeTempDir("respan-pi-cwd-"),
    homeDir: makeTempDir("respan-pi-home-"),
    env: { RESPAN_API_KEY: "sk", RESPAN_BASE_URL: "http://localhost:8000" },
  });
  assert.equal(config.baseURL, "http://localhost:8000/api");
  const unset = resolvePiRespanConfig({
    cwd: makeTempDir("respan-pi-cwd-"),
    homeDir: makeTempDir("respan-pi-home-"),
    env: { RESPAN_API_KEY: "sk" },
  });
  assert.equal(unset.baseURL, undefined);
});

test("enabled is false without an API key (fail-open) and malformed files are ignored", () => {
  const homeDir = makeTempDir("respan-pi-home-");
  const cwd = makeTempDir("respan-pi-cwd-");
  const config = resolvePiRespanConfig({ cwd, homeDir, env: {} });
  assert.equal(config.enabled, false);
  assert.equal(config.apiKey, undefined);
  assert.deepEqual(config.sources, []);

  fs.mkdirSync(path.join(cwd, ".pi"), { recursive: true });
  fs.writeFileSync(path.join(cwd, ".pi", "respan.json"), "{ not json");
  const malformed = resolvePiRespanConfig({ cwd, homeDir, env: { RESPAN_API_KEY: "sk" } });
  assert.equal(malformed.enabled, true);
  assert.deepEqual(malformed.sources, ["env"]);

  const explicitlyEnabledWithoutKey = resolvePiRespanConfig({ cwd, homeDir, env: { RESPAN_PI_TRACING: "true" } });
  assert.equal(explicitlyEnabledWithoutKey.enabled, false);
});

test("readFile injection keeps the resolver pure", () => {
  const files = new Map([
    ["/home/u/.pi/agent/respan.json", JSON.stringify({ workflow_name: "injected" })],
    ["/home/u/.respan/credentials.json", JSON.stringify({ default: { apiKey: "sk-injected" } })],
  ]);
  const config = resolvePiRespanConfig({
    cwd: "/proj",
    homeDir: "/home/u",
    env: {},
    readFile: (filePath) => files.get(filePath),
  });
  assert.equal(config.workflowName, "injected");
  assert.equal(config.apiKey, "sk-injected");
  assert.equal(config.enabled, true);
});

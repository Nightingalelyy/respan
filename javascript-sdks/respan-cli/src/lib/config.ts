import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';

const CONFIG_DIR = path.join(os.homedir(), '.respan');
const CREDENTIALS_FILE = path.join(CONFIG_DIR, 'credentials.json');
const CONFIG_FILE = path.join(CONFIG_DIR, 'config.json');

export interface ApiKeyCredential {
  type: 'api_key';
  apiKey: string;
  baseUrl: string;
}

export interface JwtCredential {
  type: 'jwt';
  accessToken: string;
  refreshToken: string;
  email: string;
  baseUrl: string;
}

export type Credential = ApiKeyCredential | JwtCredential;

export interface Config {
  activeProfile?: string;
  defaults?: Record<string, string>;
}

function ensureConfigDir(): void {
  if (!fs.existsSync(CONFIG_DIR)) {
    fs.mkdirSync(CONFIG_DIR, { recursive: true });
  }
}

function readJson<T>(filePath: string, fallback: T): T {
  try {
    if (!fs.existsSync(filePath)) return fallback;
    const raw = fs.readFileSync(filePath, 'utf-8');
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function writeJson(filePath: string, data: unknown): void {
  ensureConfigDir();
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2) + '\n', 'utf-8');
}

// --- Credentials ---

export function getAllCredentials(): Record<string, Credential> {
  return readJson<Record<string, Credential>>(CREDENTIALS_FILE, {});
}

export function getCredential(profile?: string): Credential | undefined {
  const creds = getAllCredentials();
  const name = profile || getActiveProfile();
  return creds[name];
}

export function setCredential(profile: string, credential: Credential): void {
  const creds = getAllCredentials();
  creds[profile] = credential;
  writeJson(CREDENTIALS_FILE, creds);
  // Set active profile if this is the first credential
  const config = getConfig();
  if (!config.activeProfile) {
    setActiveProfile(profile);
  }
}

export function deleteCredential(profile: string): void {
  const creds = getAllCredentials();
  delete creds[profile];
  writeJson(CREDENTIALS_FILE, creds);
}

// --- Config ---

export function getConfig(): Config {
  return readJson<Config>(CONFIG_FILE, {});
}

export function getActiveProfile(): string {
  const config = getConfig();
  return config.activeProfile || 'default';
}

export function setActiveProfile(profile: string): void {
  const config = getConfig();
  config.activeProfile = profile;
  writeJson(CONFIG_FILE, config);
}

export function getConfigValue(key: string): string | undefined {
  const config = getConfig();
  if (key === 'activeProfile') return config.activeProfile;
  return config.defaults?.[key];
}

export function setConfigValue(key: string, value: string): void {
  const config = getConfig();
  if (key === 'activeProfile') {
    config.activeProfile = value;
  } else {
    if (!config.defaults) config.defaults = {};
    config.defaults[key] = value;
  }
  writeJson(CONFIG_FILE, config);
}

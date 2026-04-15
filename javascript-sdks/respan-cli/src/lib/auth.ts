import { getCredential, Credential } from './config.js';

export const DEFAULT_BASE_URL = 'https://api.respan.ai';
export const ENTERPRISE_BASE_URL = 'https://endpoint.respan.ai';

export interface AuthConfig {
  apiKey?: string;
  accessToken?: string;
  refreshToken?: string;
  baseUrl: string;
}

function normalizeBaseUrl(baseUrl?: string): string {
  return (baseUrl || DEFAULT_BASE_URL).replace(/\/+$/, '');
}

function resolveConfiguredBaseUrl(credential?: Credential, flagBaseUrl?: string): string {
  return normalizeBaseUrl(flagBaseUrl || credential?.baseUrl || DEFAULT_BASE_URL);
}

export function resolveAuth(flags: { 'api-key'?: string; 'base-url'?: string; profile?: string }): AuthConfig {
  const credential = getCredential(flags.profile);
  const baseUrl = resolveConfiguredBaseUrl(credential, flags['base-url']);

  if (flags['api-key']) {
    return { apiKey: flags['api-key'], baseUrl };
  }
  if (process.env.RESPAN_API_KEY) {
    return {
      apiKey: process.env.RESPAN_API_KEY,
      baseUrl,
    };
  }
  if (credential) {
    return credentialToAuth(credential, baseUrl);
  }
  throw new Error('Not authenticated. Run `respan auth login` or set RESPAN_API_KEY.');
}

function credentialToAuth(cred: Credential, baseUrl: string): AuthConfig {
  if (cred.type === 'api_key') {
    return { apiKey: cred.apiKey, baseUrl };
  }
  return {
    accessToken: cred.accessToken,
    refreshToken: cred.refreshToken,
    baseUrl,
  };
}

export async function refreshJwtToken(credential: Credential & { type: 'jwt' }): Promise<{ access: string }> {
  const origin = credential.baseUrl.replace(/\/api\/?$/, '');
  const response = await fetch(`${origin}/auth/jwt/refresh/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh: credential.refreshToken }),
  });
  if (!response.ok) {
    throw new Error('Token refresh failed. Please login again with `respan auth login`.');
  }
  return response.json() as Promise<{ access: string }>;
}

import { Flags } from '@oclif/core';
import { select, input } from '@inquirer/prompts';
import * as http from 'node:http';
import * as open from 'node:child_process';
import { BaseCommand } from '../../lib/base-command.js';
import { setCredential, setActiveProfile } from '../../lib/config.js';
import { printBanner, printLoginSuccess } from '../../lib/banner.js';
import { DEFAULT_BASE_URL, ENTERPRISE_BASE_URL } from '../../lib/auth.js';

const CALLBACK_PORT = 18392;
const CALLBACK_PATH = '/callback';
const LOGIN_TIMEOUT_MS = 120_000;
const LOGIN_URL_BASE = 'https://platform.respan.ai/login';
const ENTERPRISE_LOGIN_URL_BASE = 'https://enterprise.respan.ai/login';

interface BrowserLoginResult {
  token: string;
  refreshToken: string;
  email?: string;
}

function openBrowser(url: string): void {
  const cmd = process.platform === 'darwin' ? 'open' : process.platform === 'win32' ? 'start' : 'xdg-open';
  open.exec(`${cmd} "${url}"`);
}

function successHtml(): string {
  return `<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Respan CLI</title>
<style>body{font-family:-apple-system,system-ui,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;background:#0a0a0f;color:#fff}
.card{text-align:center;padding:2rem}.check{font-size:3rem;margin-bottom:1rem;color:#6483F0}p{color:#aaa;margin-top:0.5rem}</style></head>
<body><div class="card"><div class="check">&#10003;</div><h2>Login successful!</h2><p>You can close this window and return to the terminal.</p></div></body></html>`;
}

function errorHtml(msg: string): string {
  return `<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Respan CLI</title>
<style>body{font-family:-apple-system,system-ui,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;background:#0a0a0f;color:#fff}
.card{text-align:center;padding:2rem}.icon{font-size:3rem;margin-bottom:1rem;color:#f06464}p{color:#aaa;margin-top:0.5rem}</style></head>
<body><div class="card"><div class="icon">&#10007;</div><h2>Login failed</h2><p>${msg}</p></div></body></html>`;
}

function waitForBrowserLogin(enterprise: boolean): Promise<BrowserLoginResult> {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      const url = new URL(req.url || '/', `http://localhost:${CALLBACK_PORT}`);
      if (url.pathname !== CALLBACK_PATH) {
        res.writeHead(404);
        res.end('Not found');
        return;
      }

      const token = url.searchParams.get('token');
      const refreshToken = url.searchParams.get('refresh_token');
      const email = url.searchParams.get('email') || undefined;

      if (!token || !refreshToken) {
        res.writeHead(400, { 'Content-Type': 'text/html' });
        res.end(errorHtml('Missing token. Please try again.'));
        cleanup();
        reject(new Error('Login callback missing token or refresh_token.'));
        return;
      }

      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(successHtml());
      cleanup();
      resolve({ token, refreshToken, email });
    });

    const timeout = setTimeout(() => {
      cleanup();
      reject(new Error('Login timed out after 120 seconds. Please try again.'));
    }, LOGIN_TIMEOUT_MS);

    function cleanup() {
      clearTimeout(timeout);
      server.close();
    }

    server.on('error', (err: NodeJS.ErrnoException) => {
      if (err.code === 'EADDRINUSE') {
        reject(new Error(`Port ${CALLBACK_PORT} is in use. Close the other process and try again.`));
      } else {
        reject(err);
      }
    });

    server.listen(CALLBACK_PORT, '127.0.0.1', () => {
      const redirectUri = `http://localhost:${CALLBACK_PORT}${CALLBACK_PATH}`;
      const loginUrlBase = enterprise ? ENTERPRISE_LOGIN_URL_BASE : LOGIN_URL_BASE;
      const loginUrl = new URL(loginUrlBase);
      loginUrl.searchParams.set('mode', 'cli');
      loginUrl.searchParams.set('redirect_uri', redirectUri);

      console.log('');
      console.log('  Opening browser to log in...');
      console.log(`  If the browser doesn't open, visit:`);
      console.log(`  ${loginUrl.toString()}`);
      console.log('');
      console.log('  Waiting for login (timeout: 120s)...');
      openBrowser(loginUrl.toString());
    });
  });
}

export default class AuthLogin extends BaseCommand {
  static description = 'Log in to Respan';
  static flags = {
    ...BaseCommand.baseFlags,
    'api-key': Flags.string({ description: 'API key to store (skips interactive prompt)' }),
    'profile-name': Flags.string({ description: 'Profile name', default: 'default' }),
    'base-url': Flags.string({ description: 'API base URL (auto-detected from login method if not set)' }),
    enterprise: Flags.boolean({ description: 'Use enterprise SSO login', default: false }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(AuthLogin);
    this.globalFlags = flags;
    const profile = flags['profile-name'] || 'default';
    setActiveProfile(profile);

    await printBanner();

    // Step 1: Determine environment (skip if --enterprise flag or --api-key with flag)
    const enterprise = flags.enterprise || (!flags['api-key'] && await select({
      message: 'Select your environment:',
      choices: [
        { name: 'Respan Platform', value: false },
        { name: 'Enterprise', value: true },
      ],
    }));
    const baseUrl = flags['base-url'] || (enterprise ? ENTERPRISE_BASE_URL : DEFAULT_BASE_URL);

    // If --api-key passed directly, skip auth method prompt
    if (flags['api-key']) {
      setCredential(profile, { type: 'api_key', apiKey: flags['api-key'], baseUrl });
      await printLoginSuccess(undefined, profile);
      return;
    }

    // Step 2: Choose auth method
    const method = await select({
      message: 'How would you like to authenticate?',
      choices: [
        { name: 'API key (recommended)', value: 'api_key' },
        { name: 'Browser login', value: 'browser' },
      ],
    });

    if (method === 'api_key') {
      const apiKey = await input({ message: 'Enter your Respan API key:' });
      setCredential(profile, { type: 'api_key', apiKey, baseUrl });
      await printLoginSuccess(undefined, profile);
      return;
    }

    const result = await waitForBrowserLogin(enterprise);

    setCredential(profile, {
      type: 'jwt',
      accessToken: result.token,
      refreshToken: result.refreshToken,
      email: result.email || '',
      baseUrl,
    });
    await printLoginSuccess(result.email, profile);
  }
}

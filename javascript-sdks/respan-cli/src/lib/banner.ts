const RESET = '\x1b[0m';

const GRADIENT_STOPS: [number, number, number][] = [
  [170, 190, 255], // #AABEFF bright
  [100, 131, 240], // #6483F0 primary
  [40, 65, 180],   // #2841B4 deep
];

function lerp(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}

function gradientColor(x: number, maxLen: number): string {
  const t = Math.min(x / Math.max(maxLen - 1, 1), 1);
  const seg = t * (GRADIENT_STOPS.length - 1);
  const idx = Math.min(Math.floor(seg), GRADIENT_STOPS.length - 2);
  const lt = seg - idx;
  const r = lerp(GRADIENT_STOPS[idx][0], GRADIENT_STOPS[idx + 1][0], lt);
  const g = lerp(GRADIENT_STOPS[idx][1], GRADIENT_STOPS[idx + 1][1], lt);
  const b = lerp(GRADIENT_STOPS[idx][2], GRADIENT_STOPS[idx + 1][2], lt);
  return `\x1b[38;2;${r};${g};${b}m`;
}

// Layer 2: near black shadow
const SHADOW_COLOR = '\x1b[38;2;5;5;10m';
// Layer 3: gray shadow
const SHADOW_BG = '\x1b[38;2;177;179;188m';

// ASCII art from official Respan SVG logo: [.] Respan
// Half-blocks for correct aspect ratio, fits 80-col Apple Terminal
const BANNER_LINES = [
  '▄████       █████     ▄▄▄▄▄▄',
  '███           ███     ███▀████▄',
  '███           ███     ███   ███  ▄███▄▄  ▄▄███▄  ██▄▄██▄▄  ▄▄███▄  ██▄▄███▄',
  '███           ███     ███▄▄▄██▀ ██▀  ███ ██  ▀▀▀ ███  ▀██▄ ▀▀  ▀██ ███▀ ▀██',
  '███           ███     ███▀▀██  █████████ ▀▀████▄ ██▄   ███ ▄██████ ███   ██',
  '███      ██▄  ███     ███  ▀██  ██▄  ▄▄▄ ▄▄  ▄██ ███▄ ▄██▀███  ▄██ ███   ██',
  '███▄▄     ▀ ▄▄███     ▀█▀   ▀██  ▀▀██▀▀  ▀▀███▀▀ ██▀▀██▀▀  ▀██▀▀██ ▀█▀   ▀█',
  ' ▀▀▀▀       ▀▀▀▀▀                                ██',
  '                                                 ▀▀',
];

function renderBanner(lines: string[]): string[] {
  const grid = lines.map(l => [...l]);
  const maxLen = Math.max(...grid.map(r => r.length));
  for (const row of grid) while (row.length < maxLen) row.push(' ');
  const H = grid.length, W = maxLen;

  // Layer 2: offset (1, 0) — right edge shadow
  const off2X = 1, off2Y = 0;
  // Layer 3: offset (2, 1) — further right and down
  const off3X = 2, off3Y = 1;
  const outW = W + Math.max(off2X, off3X);
  const outH = H + Math.max(off2Y, off3Y);
  const result: string[] = [];

  for (let y = 0; y < outH; y++) {
    let line = '';
    let lastColor = '';
    for (let x = 0; x < outW; x++) {
      const hasFg = y < H && x < W && grid[y][x] !== ' ';

      const s2R = y - off2Y, s2C = x - off2X;
      const hasL2 = s2R >= 0 && s2R < H && s2C >= 0 && s2C < W && grid[s2R][s2C] !== ' ';

      const s3R = y - off3Y, s3C = x - off3X;
      const hasL3 = s3R >= 0 && s3R < H && s3C >= 0 && s3C < W && grid[s3R][s3C] !== ' ';

      if (hasFg) {
        const c = gradientColor(x, maxLen);
        if (c !== lastColor) { line += c; lastColor = c; }
        line += grid[y][x];
      } else if (hasL2) {
        if (lastColor !== 'L2') { line += SHADOW_COLOR; lastColor = 'L2'; }
        line += grid[s2R][s2C];
      } else if (hasL3) {
        if (lastColor !== 'L3') { line += SHADOW_BG; lastColor = 'L3'; }
        line += grid[s3R][s3C];
      } else {
        line += ' ';
      }
    }
    result.push(line.replace(/\s+$/, '') + RESET);
  }
  while (result.length && result[result.length - 1].replace(/\x1b\[[^m]*m/g, '').trim() === '') result.pop();
  return result;
}

const PC = '\x1b[38;2;100;131;240m'; // primary color

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export async function printBanner(): Promise<void> {
  if (!process.stdout.isTTY || process.env.NO_COLOR) return;
  const lines = renderBanner(BANNER_LINES);
  console.log('');
  for (const line of lines) {
    console.log(line);
    await sleep(80);
  }
  console.log('');
}

export async function printLoginSuccess(email?: string, profile?: string): Promise<void> {
  if (!process.stdout.isTTY || process.env.NO_COLOR) {
    const msg = email ? `Logged in as ${email}.` : 'Logged in.';
    const profileMsg = profile ? ` Profile "${profile}" saved.` : '';
    console.log(`${msg}${profileMsg}`);
    return;
  }
  console.log('');
  if (email) console.log(`  ${PC}\u2713${RESET} Logged in as ${email}`);
  else console.log(`  ${PC}\u2713${RESET} Logged in`);
  if (profile) console.log(`  ${PC}\u2713${RESET} Profile "${profile}" saved`);
  console.log('');
}

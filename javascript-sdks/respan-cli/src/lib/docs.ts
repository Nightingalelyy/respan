/**
 * Docs fetcher — downloads relevant documentation pages from the Respan
 * llms.txt index and writes them as skill reference files.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { createSpinner } from './spinner.js';
import { ensureDir } from './integrate.js';

const LLMS_TXT_URL = 'https://www.respan.ai/docs/llms.txt';

interface DocEntry {
  title: string;
  url: string;
  description?: string;
  section: string;
}

/**
 * Workflow categories — which docs to fetch for each workflow.
 */
const WORKFLOW_DOC_FILTERS: Record<string, string[]> = {
  instrument: [
    'get-started',
    'quickstart',
    'python-sdk/overview',
    'python-sdk/initialize',
    'python-sdk/decorators',
    'python-sdk/instrumentations',
    'python-sdk/instrumentation-protocol',
    'typescript-sdk/overview',
    'typescript-sdk/initialize',
    'typescript-sdk/methods',
    'typescript-sdk/instrumentations',
    'integrations/openai-sdk',
    'integrations/anthropic',
    'integrations/openai-agents-sdk',
    'integrations/claude-agents-sdk',
    'integrations/vercel-ai-sdk',
    'integrations/pydantic-ai',
  ],
  observe: [
    'tracing/concepts',
    'tracing/spans',
    'tracing/traces',
    'tracing/advanced',
    'monitoring/metrics',
    'monitoring/views',
    'monitoring/monitors',
    'monitoring/webhooks',
  ],
  evaluate: [
    'evals/concepts',
    'evals/evaluators',
    'evals/datasets',
    'evals/experiments',
    'evals/online-evals',
    'eval-quickstart',
  ],
  reference: [
    'reference/span-attributes',
    'reference/error-handling',
    'reference/api-rate-limits',
    'cli',
  ],
};

/**
 * Parse the llms.txt index into structured entries.
 */
function parseLlmsTxt(content: string): DocEntry[] {
  const entries: DocEntry[] = [];
  let currentSection = 'Docs';

  for (const line of content.split('\n')) {
    const trimmed = line.trim();

    // Section headers
    const sectionMatch = trimmed.match(/^##\s+(.+)/);
    if (sectionMatch) {
      currentSection = sectionMatch[1];
      continue;
    }

    // Doc entries: - [Title](url): Description
    const entryMatch = trimmed.match(/^-\s+(?:.+?\s+)?\[(.+?)\]\((.+?)\)(?::\s*(.+))?$/);
    if (entryMatch) {
      entries.push({
        title: entryMatch[1],
        url: entryMatch[2],
        description: entryMatch[3],
        section: currentSection,
      });
    }
  }

  return entries;
}

/**
 * Filter docs by workflow categories.
 */
function filterByWorkflows(entries: DocEntry[], workflows: string[]): DocEntry[] {
  if (workflows.length === 0 || workflows.includes('all')) {
    // For 'all', return a curated subset (not the entire docs site)
    const allFilters = Object.values(WORKFLOW_DOC_FILTERS).flat();
    return entries.filter((e) =>
      allFilters.some((filter) => e.url.includes(filter)),
    );
  }

  const filters = workflows.flatMap((w) => WORKFLOW_DOC_FILTERS[w] || []);
  return entries.filter((e) =>
    filters.some((filter) => e.url.includes(filter)),
  );
}

/**
 * Download a single doc page and extract markdown content.
 */
async function fetchDocPage(url: string): Promise<string> {
  // Convert .mdx URL to raw content — fetch the page and extract text
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status}`);
  }
  return response.text();
}

/**
 * Generate a filename from a URL.
 * e.g. https://respan.ai/docs/sdks/python-sdk/overview.mdx → python-sdk-overview.md
 */
function urlToFilename(url: string): string {
  const parsed = new URL(url);
  const pathParts = parsed.pathname
    .replace(/^\/docs\//, '')
    .replace(/\.mdx$/, '')
    .split('/')
    .filter(Boolean);

  // Take last 2-3 meaningful segments
  const relevant = pathParts.slice(-Math.min(3, pathParts.length));
  return relevant.join('-') + '.md';
}

/**
 * Fetch docs from llms.txt and write them to the output directory.
 *
 * @param outputDir - Directory to write docs to (e.g. .respan/skills/docs/)
 * @param workflows - Which workflow categories to fetch (default: all)
 * @param refresh - Clear output dir before downloading
 * @param concurrency - Number of parallel downloads
 */
export async function fetchDocs(options: {
  outputDir: string;
  workflows?: string[];
  refresh?: boolean;
  concurrency?: number;
  verbose?: boolean;
}): Promise<{ fetched: number; failed: number }> {
  const {
    outputDir,
    workflows = ['all'],
    refresh = false,
    concurrency = 5,
    verbose = false,
  } = options;

  // 1. Fetch llms.txt index
  const spinner = createSpinner('Fetching docs index');
  spinner.start();

  let indexContent: string;
  try {
    const response = await fetch(LLMS_TXT_URL);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    indexContent = await response.text();
    spinner.succeed('Fetched docs index');
  } catch (error) {
    spinner.fail('Failed to fetch docs index');
    throw error;
  }

  // 2. Parse and filter
  const allEntries = parseLlmsTxt(indexContent);
  const filtered = filterByWorkflows(allEntries, workflows);

  if (verbose) {
    process.stderr.write(`  Found ${filtered.length} docs to fetch\n`);
  }

  // 3. Prepare output directory
  if (refresh && fs.existsSync(outputDir)) {
    fs.rmSync(outputDir, { recursive: true });
  }
  ensureDir(outputDir);

  // 4. Write index file
  const indexLines = ['# Respan Documentation\n'];
  const sections = new Map<string, DocEntry[]>();
  for (const entry of filtered) {
    const section = sections.get(entry.section) || [];
    section.push(entry);
    sections.set(entry.section, section);
  }
  for (const [section, entries] of sections) {
    indexLines.push(`\n## ${section}\n`);
    for (const entry of entries) {
      const filename = urlToFilename(entry.url);
      const desc = entry.description ? ` — ${entry.description}` : '';
      indexLines.push(`- [${entry.title}](${filename})${desc}`);
    }
  }
  fs.writeFileSync(path.join(outputDir, '_index.md'), indexLines.join('\n') + '\n');

  // 5. Download docs in batches
  let fetched = 0;
  let failed = 0;

  for (let i = 0; i < filtered.length; i += concurrency) {
    const batch = filtered.slice(i, i + concurrency);
    const results = await Promise.allSettled(
      batch.map(async (entry) => {
        const filename = urlToFilename(entry.url);
        const filePath = path.join(outputDir, filename);

        // Skip if already exists and not refreshing
        if (!refresh && fs.existsSync(filePath)) {
          fetched++;
          return;
        }

        try {
          const content = await fetchDocPage(entry.url);
          // Write as-is (MDX content) — agents can read it
          const header = `# ${entry.title}\n\n${entry.description ? `> ${entry.description}\n\n` : ''}Source: ${entry.url}\n\n---\n\n`;
          fs.writeFileSync(filePath, header + content);
          fetched++;
        } catch {
          failed++;
          if (verbose) {
            process.stderr.write(`  Failed: ${entry.title}\n`);
          }
        }
      }),
    );
  }

  return { fetched, failed };
}

/**
 * Write the llms.txt index as a skill file for quick reference.
 */
export async function writeLlmsTxtSkill(outputDir: string): Promise<void> {
  const response = await fetch(LLMS_TXT_URL);
  if (!response.ok) throw new Error(`Failed to fetch llms.txt: ${response.status}`);
  const content = await response.text();

  ensureDir(outputDir);
  const header = `# Respan Documentation Index

This is the full Respan documentation index. Use it to find relevant docs
when helping users set up or troubleshoot Respan.

Source: ${LLMS_TXT_URL}

---

`;
  fs.writeFileSync(path.join(outputDir, 'llms.txt'), header + content);
}

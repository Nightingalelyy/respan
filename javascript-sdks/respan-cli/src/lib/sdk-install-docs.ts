/**
 * Per-language SDK install docs — written as skill reference files
 * so the coding agent knows exactly how to instrument the project.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { ensureDir } from './integrate.js';

const PYTHON_DOCS = `# Respan Python SDK Installation Guide

## Package: \`respan-ai\`

### Installation

Look up the latest version:
\`\`\`bash
pip index versions respan-ai 2>/dev/null | head -1 || pip install respan-ai 2>&1 | grep -oP 'respan-ai==\\K[0-9.]+'
\`\`\`

Install with exact version:

| Package manager | Command |
|----------------|---------|
| pip | \`pip install respan-ai==<VERSION>\` |
| poetry | \`poetry add respan-ai==<VERSION>\` |
| uv | \`uv add respan-ai==<VERSION>\` |

### Instrumentation Packages

Install only for libraries detected in the project:

| LLM library | Package | Import |
|-------------|---------|--------|
| \`openai\` | \`respan-instrumentation-openai\` | \`from respan_instrumentation_openai import OpenAIInstrumentor\` |
| \`anthropic\` | \`respan-instrumentation-anthropic\` | \`from respan_instrumentation_anthropic import AnthropicInstrumentor\` |
| \`openai-agents\` | \`respan-instrumentation-openai-agents\` | \`from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor\` |
| \`pydantic-ai\` | \`respan-instrumentation-pydantic-ai\` | \`from respan_instrumentation_pydantic_ai import PydanticAIInstrumentor\` |
| \`claude-agent-sdk\` | \`respan-instrumentation-claude-agent-sdk\` | \`from respan_instrumentation_claude_agent_sdk import ClaudeAgentSDKInstrumentor\` |

### Initialization

Add this to your entrypoint **before** any LLM client is created:

\`\`\`python
from respan import Respan
# Import the instrumentors for your LLM libraries:
from respan_instrumentation_openai import OpenAIInstrumentor

respan = Respan(
    app_name="<project-name>",
    instrumentations=[OpenAIInstrumentor()],
)
\`\`\`

### Decorators

\`\`\`python
from respan import workflow, task, agent, tool

@workflow(name="my_pipeline")
def run_pipeline(input: str):
    result = step_one(input)
    return step_two(result)

@task(name="step_one")
def step_one(input: str):
    return client.chat.completions.create(...)

@task(name="step_two")
def step_two(input: str):
    return client.chat.completions.create(...)
\`\`\`

### Environment Variables

\`\`\`
RESPAN_API_KEY=your-api-key
\`\`\`

### Verification

\`\`\`bash
# Run your app, then check traces:
respan traces list --limit 5
\`\`\`

Or visit https://platform.respan.ai
`;

const TYPESCRIPT_DOCS = `# Respan TypeScript SDK Installation Guide

## Package: \`@respan/respan\`

### Installation

Look up the latest version:
\`\`\`bash
npm view @respan/respan version
\`\`\`

Install with exact version:

| Package manager | Command |
|----------------|---------|
| npm | \`npm install --save-exact @respan/respan@<VERSION> --no-audit --no-fund\` |
| yarn | \`yarn add --exact @respan/respan@<VERSION>\` |
| pnpm | \`pnpm add --save-exact @respan/respan@<VERSION>\` |

### Instrumentation Packages

Install only for libraries detected in the project:

| LLM library | Package | Import |
|-------------|---------|--------|
| \`openai\` | \`@respan/instrumentation-openai\` | \`import { OpenAIInstrumentor } from "@respan/instrumentation-openai"\` |
| \`@anthropic-ai/sdk\` | \`@respan/instrumentation-anthropic\` | \`import { AnthropicInstrumentor } from "@respan/instrumentation-anthropic"\` |
| \`@openai/agents\` | \`@respan/instrumentation-openai-agents\` | \`import { OpenAIAgentsInstrumentor } from "@respan/instrumentation-openai-agents"\` |
| \`ai\` (Vercel) | \`@respan/instrumentation-vercel\` | \`import { VercelInstrumentor } from "@respan/instrumentation-vercel"\` |

### Initialization

Add this to your entrypoint **before** any LLM client is created:

\`\`\`typescript
import { Respan } from "@respan/respan";
// Import the instrumentors for your LLM libraries:
import { OpenAIInstrumentor } from "@respan/instrumentation-openai";

const respan = new Respan({
  appName: "<project-name>",
  instrumentations: [new OpenAIInstrumentor()],
});
await respan.initialize();
\`\`\`

### Wrappers

\`\`\`typescript
import { withWorkflow, withTask, withAgent, withTool } from "@respan/respan";

const result = await withWorkflow({ name: "my_pipeline" }, async () => {
  const step1 = await withTask({ name: "step_one" }, async () => {
    return await client.chat.completions.create({ ... });
  });

  const step2 = await withTask({ name: "step_two" }, async () => {
    return await client.chat.completions.create({ ... });
  });

  return step2;
});
\`\`\`

### Environment Variables

\`\`\`
RESPAN_API_KEY=your-api-key
\`\`\`

### Verification

\`\`\`bash
# Run your app, then check traces:
respan traces list --limit 5
\`\`\`

Or visit https://platform.respan.ai
`;

const INDEX = `# SDK Install Docs

Per-language SDK installation guides. Read the file for the detected language.

- [Python](python.md)
- [TypeScript](typescript.md)
`;

/**
 * Write all SDK install docs to \`<outputDir>/sdk-install/\`.
 */
export function writeSdkInstallDocs(outputDir: string): void {
  const dir = path.join(outputDir, 'sdk-install');
  ensureDir(dir);
  fs.writeFileSync(path.join(dir, 'python.md'), PYTHON_DOCS);
  fs.writeFileSync(path.join(dir, 'typescript.md'), TYPESCRIPT_DOCS);
  fs.writeFileSync(path.join(dir, '_index.md'), INDEX);
}

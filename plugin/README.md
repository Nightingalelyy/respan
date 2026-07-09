# Respan — Claude Code Plugin

Packages Respan's `/respan` skill and the hosted Respan MCP server into a single
installable Claude Code plugin. One install gives a user the Respan know-how
(tracing, gateway, prompts, evals, datasets) **and** live access to the Respan
platform via MCP tools — no separate `claude mcp add` step.

## What's inside

```
plugin/
├── .claude-plugin/
│   ├── plugin.json         # manifest: name, version, api_key user config
│   └── marketplace.json    # single-plugin marketplace (source: "./")
├── .mcp.json               # connects the hosted MCP server at mcp.respan.ai
├── scripts/
│   └── build-plugin.mjs    # copies the shared skill into skills/ (see below)
└── skills/
    └── respan/             # GENERATED — do not hand-edit
        ├── SKILL.md
        └── references/*.md
```

## Single source of truth

The skill lives in exactly one place: `respan/skills/` at the monorepo root.
`scripts/build-plugin.mjs` copies it into `plugin/skills/respan/` at build time
and prepends the YAML frontmatter a plugin skill needs. This mirrors the CLI's
`generate:skill-refs` step — one skill, assembled into each distribution (CLI
bundle and this plugin). **Never edit `plugin/skills/` by hand**; edit
`respan/skills/` and re-run the build. The generated files are committed so the
published plugin is self-contained (marketplaces copy the plugin directory).

```bash
node plugin/scripts/build-plugin.mjs
```

Wire this into the release process so the plugin skill can never drift from the
CLI's copy.

## Test locally

From the monorepo root, load the plugin without publishing anything:

```bash
claude --plugin-dir ./plugin
```

You'll be prompted for a Respan API key (create one at https://platform.respan.ai).
Then invoke the skill and confirm the MCP tools are live (`mcp__respan__*`).

Validate the manifest before publishing:

```bash
claude plugin validate ./plugin --strict
```

## Authentication

The manifest declares a `userConfig.api_key` (marked `sensitive`), so Claude Code
prompts for it at install time and stores it in the OS keychain. `.mcp.json`
injects it as `Authorization: Bearer ${user_config.api_key}` against the hosted
server `https://mcp.respan.ai/mcp`. The server also supports OAuth browser
sign-in; API-key config is what this plugin ships with first.

## Publishing to the community marketplace

1. `node plugin/scripts/build-plugin.mjs` and commit the result.
2. `claude plugin validate ./plugin --strict`.
3. Submit the plugin to Anthropic's community marketplace via
   https://platform.claude.com/plugins/submit (the catalog pins to a commit SHA;
   CI bumps the pin as you push).

Users then install with:

```
/plugin marketplace add anthropics/claude-plugins-community
/plugin install respan
```

Or, to self-host from this repo, point at the bundled `marketplace.json`:

```
/plugin marketplace add respanai/respan
/plugin install respan
```

# CI/CD

This repository now treats package publishing as an explicit, checked-in inventory instead of ad hoc per-package logic.
Only the selected core SDK packages are included in automation; other packages can remain in the repo without being part of CI publish decisions.

## What The Pipeline Does

- `ci.yml` runs on pull requests and pushes to `main`.
- It validates the package inventory in `.github/release-packages.json`.
- It builds every publishable Python package.
- It installs the JavaScript workspace once per job and then builds or tests each publishable JavaScript package through Yarn workspaces.

## What Publishes On `main`

- `publish.yml` runs after the `CI` workflow completes successfully for `main`, or by manual dispatch.
- A package is published only when its manifest version changed between the previous and current ref.
- JavaScript packages publish to npm from the Yarn workspace.
- Python packages build first, then publish from built artifacts with PyPI Trusted Publishing.

This means merges to `main` can publish automatically, but only if the PR included an actual package version bump.

## Included Packages

Current release automation covers:

- Python: `respan-ai`, `respan-sdk`, `respan-tracing`, `respan-instrumentation-openai`, `respan-instrumentation-openai-agents`, `respan-instrumentation-openinference`
- JavaScript: `@respan/respan`, `@respan/cli`, `@respan/respan-sdk`, `@respan/tracing`, `@respan/instrumentation-openai`, `@respan/instrumentation-openai-agents`, `@respan/instrumentation-openinference`, `@respan/instrumentation-vercel`

Exporters and other packages are intentionally excluded from release automation.

## Release Metadata Contract

Every publishable package must have:

- one entry in `.github/release-packages.json`
- a manifest name that matches that inventory entry
- a single authoritative manifest version
- public publish metadata for npm packages

For JavaScript packages, the root workspace in `javascript-sdks/package.json` must include each release-managed package path. Extra workspaces are allowed.

## Required Registry Setup

Before the publish workflow can succeed, configure Trusted Publishing for each registry package:

- npm: add this repository and the `publish.yml` workflow as a Trusted Publisher for each npm package
- PyPI: add this repository and the `publish.yml` workflow as a Trusted Publisher for each PyPI project

Recommended GitHub environments:

- `npm`
- `pypi`

Use those environments for approval gates if you want a manual checkpoint before packages are released.

## Local Validation

Run:

```bash
python3 scripts/release_inventory.py --validate
```

List publishable packages:

```bash
python3 scripts/release_inventory.py --ecosystem all
```

List packages that would publish between two refs:

```bash
python3 scripts/release_inventory.py --ecosystem all --changed-from <base> --changed-to <head> --version-changed
```

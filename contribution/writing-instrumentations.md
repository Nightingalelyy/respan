# Writing Instrumentations

This guide is for adding a new package under:

- `python-sdks/instrumentations/`
- `javascript-sdks/instrumentations/`

It does not apply to legacy packages under `*/legacy/`.

## Choose The Right Shape

There are only two acceptable patterns:

1. Thin wrapper around an existing OTEL/OpenInference/Traceloop instrumentor
2. Native instrumentation package that translates vendor events into the Respan tracing model

Prefer the thin wrapper when a mature upstream instrumentor already exists. Only build a native integration when you actually need custom event translation or patching.

## Package Placement

New active instrumentations must live here:

- Python: `python-sdks/instrumentations/respan-instrumentation-<name>`
- JavaScript: `javascript-sdks/instrumentations/respan-instrumentation-<name>`

Do not add new instrumentation work under `legacy/`.

## Package Naming

### Python

- Distribution name: `respan-instrumentation-<name>`
- Import package: `respan_instrumentation_<name>`

### JavaScript

- Package name: `@respan/instrumentation-<name>`
- Directory name: `respan-instrumentation-<name>`

Use the same `<name>` across both languages where possible.

## Minimum Package Structure

### Python

```text
python-sdks/instrumentations/respan-instrumentation-<name>/
├── pyproject.toml
├── README.md
├── src/
│   └── respan_instrumentation_<name>/
│       ├── __init__.py
│       ├── _instrumentation.py
│       └── ...
└── tests/
```

### JavaScript

```text
javascript-sdks/instrumentations/respan-instrumentation-<name>/
├── package.json
├── tsconfig.json
├── README.md
├── src/
│   └── index.ts
└── tests/
```

## Required Integration Steps

For every new active instrumentation:

1. Add the package directory
2. Add tests
3. Add the package to the release-managed inventory if it should publish
4. Add it to the JS root workspace if it is a JS package
5. Add examples only if they add real coverage or onboarding value

### Inventory

Release-managed instrumentations must be added to:

- [.github/release-packages.json](/Users/chensihan/Documents/github/respan/.github/release-packages.json)

If the package is not in release inventory, CI/CD will not treat it as part of the active release surface.

### JavaScript Workspace

Release-managed JS instrumentations must be listed in:

- [package.json](/Users/chensihan/Documents/github/respan/javascript-sdks/package.json)

Legacy packages must stay out of that workspace list.

## Python Guidance

Python instrumentation packages should generally depend on:

- `respan-tracing`
- `respan-sdk` only when they need shared constants or types
- the vendor SDK or upstream OTEL instrumentor they wrap

Good defaults:

- keep package metadata in `pyproject.toml`
- expose one clear instrumentor entrypoint from `__init__.py`
- keep translation/serialization helpers private
- put unit tests under `tests/`

If the package is intended to be loaded as a plugin, make the plugin entrypoint explicit in `pyproject.toml`.

## JavaScript Guidance

JS instrumentation packages should generally:

- compile with `tsc`
- expose one clear entrypoint from `src/index.ts`
- keep package metadata and `repository.directory` accurate
- avoid coupling themselves to `legacy/` packages

If a JS package has a real test suite, wire it through the package `test` script so CI picks it up automatically.

## Testing Expectations

Minimum expectation for a new instrumentation:

- it builds
- it can be packaged
- it has at least one focused unit or smoke test for its core mapping logic

Current CI behavior already gives you:

- build validation
- package smoke validation
- affected-package execution

If your integration has subtle event translation, add direct unit tests for those mappings. Do not rely only on end-to-end examples.

## Release Expectations

If you touch a release-managed instrumentation package in a PR, you must add one release intent file under:

- `.release-intents/`

Use one of:

- `none`
- `new`
- `patch`
- `minor`
- `major`

See [publish.md](/Users/chensihan/Documents/github/respan/contribution/publish.md) for the release workflow.

## Anti-Patterns

Do not do these:

- add new active packages under `legacy/`
- bypass `.github/release-packages.json`
- keep duplicate contributor docs in package subtrees
- introduce circular dependencies between core packages and instrumentations
- rely on manual post-merge version editing

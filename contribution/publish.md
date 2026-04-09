# Publishing And Release Intents

This repository does not use hand-maintained final version bumps in each PR.

Instead, release-managed changes are driven by:

- [.github/release-packages.json](../.github/release-packages.json)
- `.release-intents/*.json`

## Why Release Intents Exist

Parallel PRs against the same package make manual version bumps noisy and error-prone.

Example:

- PR A changes `@respan/cli`
- PR B also changes `@respan/cli`

If both PRs guess the next version up front, the second PR has to rebase and re-bump after the first merges.

Release intents remove that problem:

- PRs describe the release level they need
- publish computes the next concrete version later

## Where Intents Live

Intent files live at repo root:

- `.release-intents/`
- use [.release-intents/20260409-example.json](../.release-intents/20260409-example.json) as the template

Recommended naming:

- `YYYYMMDD-short-description.json`

Examples:

- `20260409-cli-auth.json`
- `20260409-openai-instrumentation.json`

The filename is for humans, not for automation. CI only requires that the file lives under `.release-intents/` and contains valid JSON for changed release-managed packages.

Use one file per PR.

Example:

```json
{
  "summary": "Patch CLI formatting and tracing docs",
  "packages": {
    "@respan/cli": "patch",
    "respan-tracing": "minor"
  }
}
```

Valid values:

- `none`
- `new`
- `patch`
- `minor`
- `major`

Meaning:

- `none`
  - package changed, but do not publish it
- `new`
  - first release for a package that is not yet in the registry
  - publish uses the manifest version directly
- `patch` / `minor` / `major`
  - publish bumps from the current registry-aware version base

`none` is allowed when a release-managed package changed but should not cut a release.
`new` must only be used for packages that do not already exist in npm or PyPI.

## What CI Enforces

CI now does all of the following:

1. Validates the release inventory
2. Validates all release intent files
3. Fails if a release-managed package changed without a matching release intent
4. Runs only affected release-managed packages and their dependents
5. Runs lightweight build and packaging checks

This means contributors no longer need to hand-edit final package versions just to satisfy CI.

## What Publish Does

Publish now works like this:

1. Read changed release intent files for the merge range
2. Aggregate bump levels per package
3. Compute the next concrete version at publish time
4. Build and publish the package
5. Run a registry smoke test from npm or PyPI

The important part is step 3: the version is decided late, which avoids the parallel-PR collision problem.

## Manifest Sync

Publish computes the final concrete versions at publish time and then writes those exact versions back into the repository.

So the current flow is:

- release intent drives publishing
- publish chooses the final version late, using the registry-aware bump logic
- successful publishes record the exact released versions
- the workflow syncs those versions back into package manifests on `main`

This keeps the repository manifests aligned with what was actually published without forcing parallel PRs to guess final versions up front.

## What About Release Notes And Changelogs

Those are not automated yet.

At the moment, release intent is used only for:

- deciding whether a package should release
- deciding the bump level

It is not yet used to generate:

- changelog entries
- GitHub release notes
- package-level release summaries

If we want that later, the natural next step is:

1. require a short human summary field in each intent file
2. aggregate those summaries during publish
3. generate per-package release notes from the merged intents

That is straightforward, but it is separate from the version-conflict problem.

## Contributor Rules

If your PR changes a release-managed package:

1. update the code
2. add one intent file in `.release-intents/`
3. do not manually guess the final version just to satisfy CI

If your PR only touches legacy packages under `*/legacy/`, it should not affect the main release flow.

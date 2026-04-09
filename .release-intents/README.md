Each pull request that changes a release-managed package should add one JSON file
to `.release-intents/`.

Minimal example:

```json
{
  "summary": "Patch CLI output formatting",
  "packages": {
    "@respan/cli": "patch",
    "respan-tracing": "none"
  }
}
```

Valid bump values are:

- `none`
- `patch`
- `minor`
- `major`

Guidelines:

- Use one intent file per PR.
- Only list packages that are part of the release-managed inventory.
- Use `none` when a release-managed package changed but should not cut a release.
- CI validates that changed release-managed packages are covered by changed intent files.

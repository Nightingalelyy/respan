## Summary

- describe the main code change
- describe the main user or developer impact

## Release Intent

If this PR changes any release-managed package, add one `.release-intents/*.json` file.

Rules:

- use one intent file per PR
- use one of `none`, `new`, `patch`, `minor`, or `major`
- do not hand-edit final package versions just to satisfy CI
- prefer `YYYYMMDD-short-description.json` naming
- use [.release-intents/20260409-example.json](/Users/chensihan/Documents/github/respan/.release-intents/20260409-example.json) as the starting shape

## Validation

- [ ] I updated or added a `.release-intents/*.json` file if this PR changes a release-managed package
- [ ] I ran the relevant local checks for the packages I touched

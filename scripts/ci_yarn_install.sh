#!/usr/bin/env bash
set -euo pipefail

# Yarn 4.9.2 can crash in its HTTP retry handler before installation finishes.
# Retry only that exact crash; dependency, lockfile, and build failures stay fatal.
readonly retryable_error='Error: The `onCancel` handler was attached after the promise settled.'
install_log="$(mktemp "${TMPDIR:-/tmp}/respan-yarn-install.XXXXXX")"
trap 'rm -f "$install_log"' EXIT

for attempt in 1 2 3; do
  if yarn install "$@" 2>&1 | tee "$install_log"; then
    exit 0
  else
    pipeline_status=("${PIPESTATUS[@]}")
  fi

  install_status="${pipeline_status[0]}"
  if [ "${pipeline_status[1]}" -ne 0 ]; then
    exit "${pipeline_status[1]}"
  fi
  # Full-line matching avoids matching the error string in Node's source dump.
  if [ "$attempt" -eq 3 ] || ! grep -Fxq "$retryable_error" "$install_log"; then
    exit "$install_status"
  fi

  delay="$((attempt * 5))"
  printf '::warning::Yarn request cancellation crash (attempt %s/3); retrying in %s seconds.\n' "$attempt" "$delay"
  sleep "$delay"
done

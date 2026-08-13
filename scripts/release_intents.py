#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import release_inventory


REPO_ROOT = Path(__file__).resolve().parents[1]
INTENTS_DIR = REPO_ROOT / ".release-intents"
BUMP_ORDER = {"none": 0, "patch": 1, "minor": 2, "major": 3, "new": 4}
SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$")


def load_intent(path: Path) -> dict:
    return json.loads(path.read_text())


def list_intent_paths() -> list[Path]:
    if not INTENTS_DIR.exists():
        return []
    return sorted(path for path in INTENTS_DIR.glob("*.json") if path.is_file())


def changed_intent_paths(base: str | None, head: str | None) -> list[Path]:
    if not base or not head:
        return list_intent_paths()

    output = release_inventory.git_stdout_or_none(
        "diff",
        "--name-only",
        "--diff-filter=AM",
        base,
        head,
        "--",
        str(INTENTS_DIR.relative_to(REPO_ROOT)),
    )
    if output is None:
        return []

    paths: list[Path] = []
    for line in output.splitlines():
        candidate = (REPO_ROOT / line.strip()).resolve()
        if candidate.is_file() and candidate.suffix == ".json":
            paths.append(candidate)
    return sorted(paths)


def inventory_entries_by_name() -> dict[str, dict]:
    return {entry["name"]: entry for entry in release_inventory.load_inventory()}


def validate_intent(path: Path, entries_by_name: dict[str, dict]) -> list[str]:
    errors: list[str] = []
    try:
        payload = load_intent(path)
    except json.JSONDecodeError as exc:
        return [f"{path.relative_to(REPO_ROOT)} is not valid JSON: {exc}"]

    packages = payload.get("packages")
    if not isinstance(packages, dict) or not packages:
        return [f"{path.relative_to(REPO_ROOT)} must define a non-empty packages object"]

    for package_name, bump in packages.items():
        if package_name not in entries_by_name:
            errors.append(
                f"{path.relative_to(REPO_ROOT)} references unknown release-managed package {package_name}"
            )
            continue
        if bump not in BUMP_ORDER:
            errors.append(
                f"{path.relative_to(REPO_ROOT)} uses invalid bump '{bump}' for {package_name}"
            )
    return errors


def validate_all_intents() -> list[str]:
    entries_by_name = inventory_entries_by_name()
    errors: list[str] = []
    for path in list_intent_paths():
        errors.extend(validate_intent(path, entries_by_name))
    return errors


def aggregate_bumps(intent_paths: list[Path], ecosystem: str) -> dict[str, str]:
    entries_by_name = inventory_entries_by_name()
    resolved: dict[str, str] = {}

    for path in intent_paths:
        payload = load_intent(path)
        for package_name, bump in payload["packages"].items():
            entry = entries_by_name.get(package_name)
            if entry is None:
                continue
            if ecosystem != "all" and entry["ecosystem"] != ecosystem:
                continue
            previous = resolved.get(package_name)
            if previous is None or BUMP_ORDER[bump] > BUMP_ORDER[previous]:
                resolved[package_name] = bump

    return resolved


def changed_release_package_names(base: str | None, head: str | None, ecosystem: str) -> set[str]:
    entries = release_inventory.load_inventory()
    selected = [entry for entry in entries if ecosystem == "all" or entry["ecosystem"] == ecosystem]
    if not base or not head:
        return {entry["name"] for entry in selected}

    files = release_inventory.changed_files(base, head)
    return {
        entry["name"]
        for entry in selected
        if release_inventory.package_changed(entry, files)
    }


def missing_intents(base: str | None, head: str | None, ecosystem: str) -> list[str]:
    changed_packages = changed_release_package_names(base, head, ecosystem)
    covered = set(aggregate_bumps(changed_intent_paths(base, head), ecosystem).keys())
    return sorted(changed_packages - covered)


def planned_records(base: str | None, head: str | None, ecosystem: str) -> list[dict]:
    entries = inventory_entries_by_name()
    bumps = aggregate_bumps(changed_intent_paths(base, head), ecosystem)
    records: list[dict] = []

    for package_name in sorted(bumps):
        bump = bumps[package_name]
        if bump == "none":
            continue
        entry = entries[package_name]
        manifest = release_inventory.load_manifest(entry)
        record = release_inventory.build_record(entry, manifest)
        record["bump"] = bump
        records.append(record)

    return records


def parse_semver(version: str) -> tuple[int, int, int]:
    match = SEMVER_RE.fullmatch(version)
    if not match:
        raise ValueError(f"unsupported version format: {version}")
    return tuple(int(part) for part in match.groups())


def version_string(parts: tuple[int, int, int]) -> str:
    return f"{parts[0]}.{parts[1]}.{parts[2]}"


def bump_version(version: str, bump: str) -> str:
    major, minor, patch = parse_semver(version)
    if bump == "new":
        return version
    if bump == "major":
        return version_string((major + 1, 0, 0))
    if bump == "minor":
        return version_string((major, minor + 1, 0))
    if bump == "patch":
        return version_string((major, minor, patch + 1))
    if bump == "none":
        return version
    raise ValueError(f"unsupported bump type: {bump}")


def max_version(left: str, right: str) -> str:
    return left if parse_semver(left) >= parse_semver(right) else right


def latest_registry_version(entry: dict) -> str | None:
    if entry["ecosystem"] == "javascript":
        package_name = urllib.parse.quote(entry["name"], safe="@")
        url = f"https://registry.npmjs.org/{package_name}/latest"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise
        return payload.get("version")

    package_name = urllib.parse.quote(entry["name"], safe="")
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    return payload.get("info", {}).get("version")


def replace_python_version(text: str, version: str) -> str:
    in_poetry = False
    lines: list[str] = []
    replaced = False

    for raw_line in text.splitlines():
        line = raw_line
        stripped = line.strip()

        if stripped.startswith("[") and stripped.endswith("]"):
            in_poetry = stripped == "[tool.poetry]"
            lines.append(line)
            continue

        if in_poetry and stripped.startswith("version = "):
            indent = line[: len(line) - len(line.lstrip())]
            lines.append(f'{indent}version = "{version}"')
            replaced = True
            continue

        lines.append(line)

    if not replaced:
        raise ValueError("could not locate [tool.poetry] version field")

    return "\n".join(lines) + "\n"


def set_manifest_version(entry: dict, version: str) -> None:
    path = release_inventory.manifest_path(entry)
    if entry["ecosystem"] == "javascript":
        manifest = json.loads(path.read_text())
        manifest["version"] = version
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        return

    text = path.read_text()
    path.write_text(replace_python_version(text, version))


def set_versions(payload: dict[str, str]) -> list[str]:
    entries = inventory_entries_by_name()
    updated: list[str] = []

    for package_name in sorted(payload):
        entry = entries.get(package_name)
        if entry is None:
            raise KeyError(package_name)
        set_manifest_version(entry, payload[package_name])
        updated.append(entry["path"])

    return updated


def resolved_base_version(entry: dict, prefer_registry: bool) -> str:
    manifest = release_inventory.load_manifest(entry)
    current_version = release_inventory.manifest_version(entry, manifest)
    if not prefer_registry:
        return current_version

    registry_version = latest_registry_version(entry)
    if not registry_version:
        return current_version
    return max_version(current_version, registry_version)


def resolved_release_version(entry: dict, bump: str, prefer_registry: bool) -> str:
    if bump == "new":
        manifest = release_inventory.load_manifest(entry)
        current_version = release_inventory.manifest_version(entry, manifest)
        registry_version = latest_registry_version(entry)
        if registry_version is not None:
            raise ValueError(
                f"package {entry['name']} already exists in the registry at {registry_version}; "
                "cannot use bump type 'new'"
            )
        return current_version

    base_version = resolved_base_version(entry, prefer_registry=prefer_registry)
    return bump_version(base_version, bump)


def resolve_versions(records: list[dict], prefer_registry: bool) -> dict[str, str]:
    entries = inventory_entries_by_name()
    return {
        record["name"]: resolved_release_version(
            entries[record["name"]],
            record["bump"],
            prefer_registry=prefer_registry,
        )
        for record in records
    }


def validate_version_mapping(payload: object) -> dict[str, str]:
    if not isinstance(payload, dict) or not payload:
        raise ValueError("version mapping must contain a non-empty object")

    for package_name, version in payload.items():
        if not isinstance(package_name, str) or not isinstance(version, str):
            raise ValueError("version mapping must map package names to version strings")
        parse_semver(version)

    return payload


def apply_resolved_versions(payload: dict[str, str], package_name: str) -> str:
    if package_name not in payload:
        raise KeyError(package_name)
    set_versions(payload)
    return payload[package_name]


def apply_resolved_versions_for_plan(
    payload: dict[str, str],
    package_names: list[str],
) -> dict[str, str]:
    planned_names = set(package_names)
    resolved_names = set(payload)
    if planned_names != resolved_names:
        missing = sorted(planned_names - resolved_names)
        unexpected = sorted(resolved_names - planned_names)
        details: list[str] = []
        if missing:
            details.append(f"missing resolved versions for {missing}")
        if unexpected:
            details.append(f"unexpected resolved versions for {unexpected}")
        raise ValueError("javascript publication plan/version mismatch: " + "; ".join(details))

    set_versions(payload)
    return payload


def collect_published_versions(artifact_root: Path) -> dict[str, str]:
    versions: dict[str, str] = {}

    def add_version(package_name: object, version: object, source: Path) -> None:
        if not isinstance(package_name, str) or not isinstance(version, str):
            raise ValueError(f"invalid published version payload in {source}")
        parse_semver(version)
        previous = versions.get(package_name)
        if previous is not None and previous != version:
            raise ValueError(
                f"conflicting published versions for {package_name}: {previous} != {version}"
            )
        versions[package_name] = version

    for payload_path in sorted(artifact_root.glob("*/version.json")):
        payload = json.loads(payload_path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"invalid published version payload in {payload_path}")
        add_version(payload.get("name"), payload.get("version"), payload_path)

    for payload_path in sorted(artifact_root.glob("*/versions.json")):
        payload = json.loads(payload_path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"invalid published versions payload in {payload_path}")
        for package_name, version in payload.items():
            add_version(package_name, version, payload_path)

    if not versions:
        raise ValueError(f"no published version artifacts found under {artifact_root}")
    return dict(sorted(versions.items()))


def write_published_version_mapping(artifact_root: Path, output_path: Path) -> dict[str, str]:
    versions = collect_published_versions(artifact_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(versions, indent=2, sort_keys=True) + "\n")
    return versions


def print_records(records: list[dict], fmt: str, count: bool) -> int:
    if count:
        print(len(records))
        return 0
    if fmt == "github-matrix":
        print(json.dumps(records, separators=(",", ":")))
        return 0
    if fmt == "names":
        for record in records:
            print(record["name"])
        return 0
    print(json.dumps(records, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")

    common_selection = argparse.ArgumentParser(add_help=False)
    common_selection.add_argument(
        "--ecosystem",
        choices=["all", "javascript", "python"],
        default="all",
    )
    common_selection.add_argument("--changed-from")
    common_selection.add_argument("--changed-to")

    common_plan = argparse.ArgumentParser(add_help=False, parents=[common_selection])
    common_plan.add_argument("--count", action="store_true")
    common_plan.add_argument("--format", choices=["json", "github-matrix", "names"], default="json")

    subparsers.add_parser("plan", parents=[common_plan])
    subparsers.add_parser("missing", parents=[common_plan])

    resolve_versions_parser = subparsers.add_parser("resolve-versions", parents=[common_selection])
    resolve_versions_parser.add_argument("--prefer-registry", action="store_true")

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--package", required=True)
    apply_parser.add_argument("--bump", choices=sorted(BUMP_ORDER.keys()), required=True)
    apply_parser.add_argument("--prefer-registry", action="store_true")

    set_versions_parser = subparsers.add_parser("set-versions")
    set_versions_parser.add_argument("--file", required=True)

    apply_resolved_parser = subparsers.add_parser("apply-resolved-versions")
    apply_resolved_parser.add_argument("--versions-json", required=True)
    apply_target = apply_resolved_parser.add_mutually_exclusive_group(required=True)
    apply_target.add_argument("--package")
    apply_target.add_argument("--plan-file", type=Path)

    merge_versions_parser = subparsers.add_parser("merge-version-artifacts")
    merge_versions_parser.add_argument("--artifact-root", type=Path, required=True)
    merge_versions_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "validate":
        errors = validate_all_intents()
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print("release intents validation passed")
        return 0

    if args.command == "plan":
        records = planned_records(args.changed_from, args.changed_to, args.ecosystem)
        return print_records(records, args.format, args.count)

    if args.command == "missing":
        records = [{"name": name} for name in missing_intents(args.changed_from, args.changed_to, args.ecosystem)]
        return print_records(records, args.format, args.count)

    if args.command == "resolve-versions":
        records = planned_records(args.changed_from, args.changed_to, args.ecosystem)
        try:
            versions = resolve_versions(records, prefer_registry=args.prefer_registry)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(json.dumps(versions, separators=(",", ":"), sort_keys=True))
        return 0

    if args.command == "apply":
        entries = inventory_entries_by_name()
        entry = entries.get(args.package)
        if entry is None:
            print(f"unknown release-managed package: {args.package}", file=sys.stderr)
            return 1

        try:
            next_version = resolved_release_version(
                entry,
                args.bump,
                prefer_registry=args.prefer_registry,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        set_manifest_version(entry, next_version)
        print(next_version)
        return 0

    if args.command == "set-versions":
        try:
            payload = validate_version_mapping(json.loads(Path(args.file).read_text()))
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        try:
            updated_paths = set_versions(payload)
        except KeyError as exc:
            print(f"unknown release-managed package: {exc.args[0]}", file=sys.stderr)
            return 1

        for updated_path in updated_paths:
            print(updated_path)
        return 0

    if args.command == "apply-resolved-versions":
        try:
            payload = validate_version_mapping(json.loads(args.versions_json))
            if args.plan_file:
                package_names = release_inventory.javascript_plan_package_names(args.plan_file)
                applied_versions = apply_resolved_versions_for_plan(payload, package_names)
                print(json.dumps(applied_versions, separators=(",", ":"), sort_keys=True))
                return 0

            release_version = apply_resolved_versions(payload, args.package)
        except json.JSONDecodeError as exc:
            print(f"version mapping is not valid JSON: {exc}", file=sys.stderr)
            return 1
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except KeyError as exc:
            print(f"unknown package in resolved version mapping: {exc.args[0]}", file=sys.stderr)
            return 1

        print(release_version)
        return 0

    if args.command == "merge-version-artifacts":
        try:
            write_published_version_mapping(args.artifact_root, args.output)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(args.output)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

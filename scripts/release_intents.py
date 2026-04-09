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

    common_plan = argparse.ArgumentParser(add_help=False)
    common_plan.add_argument("--ecosystem", choices=["all", "javascript", "python"], default="all")
    common_plan.add_argument("--changed-from")
    common_plan.add_argument("--changed-to")
    common_plan.add_argument("--count", action="store_true")
    common_plan.add_argument("--format", choices=["json", "github-matrix", "names"], default="json")

    subparsers.add_parser("plan", parents=[common_plan])
    subparsers.add_parser("missing", parents=[common_plan])

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--package", required=True)
    apply_parser.add_argument("--bump", choices=sorted(BUMP_ORDER.keys()), required=True)
    apply_parser.add_argument("--prefer-registry", action="store_true")

    set_versions_parser = subparsers.add_parser("set-versions")
    set_versions_parser.add_argument("--file", required=True)

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

    if args.command == "apply":
        entries = inventory_entries_by_name()
        entry = entries.get(args.package)
        if entry is None:
            print(f"unknown release-managed package: {args.package}", file=sys.stderr)
            return 1

        if args.bump == "new":
            manifest = release_inventory.load_manifest(entry)
            current_version = release_inventory.manifest_version(entry, manifest)
            registry_version = latest_registry_version(entry)
            if registry_version is not None:
                print(
                    f"package {args.package} already exists in the registry at {registry_version}; "
                    "cannot use bump type 'new'",
                    file=sys.stderr,
                )
                return 1
            set_manifest_version(entry, current_version)
            print(current_version)
            return 0

        base_version = resolved_base_version(entry, prefer_registry=args.prefer_registry)
        next_version = bump_version(base_version, args.bump)
        set_manifest_version(entry, next_version)
        print(next_version)
        return 0

    if args.command == "set-versions":
        payload = json.loads(Path(args.file).read_text())
        if not isinstance(payload, dict) or not payload:
            print("version mapping file must contain a non-empty object", file=sys.stderr)
            return 1

        for package_name, version in payload.items():
            if not isinstance(package_name, str) or not isinstance(version, str):
                print("version mapping file must map package names to version strings", file=sys.stderr)
                return 1
            try:
                parse_semver(version)
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

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

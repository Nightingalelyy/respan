from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "scripts" / "ci_yarn_install.sh"
RETRYABLE_ERROR = (
    "Error: The `onCancel` handler was attached after the promise settled."
)


class CiYarnInstallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.root = Path(self.temp_directory.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log_directory = self.root / "logs"
        self.log_directory.mkdir()
        self.calls = self.root / "calls.jsonl"
        self.sleeps = self.root / "sleeps.jsonl"
        self.plan = self.root / "plan.json"
        self.environment = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "TMPDIR": str(self.log_directory),
            "YARN_TEST_CALLS": str(self.calls),
            "YARN_TEST_SLEEPS": str(self.sleeps),
            "YARN_TEST_PLAN": str(self.plan),
        }
        self.write_executable(
            "yarn",
            """
import json
import os
from pathlib import Path
import sys

calls = Path(os.environ["YARN_TEST_CALLS"])
attempt = len(calls.read_text().splitlines()) if calls.exists() else 0
with calls.open("a") as stream:
    stream.write(json.dumps({
        "args": sys.argv[1:],
        "cwd": os.getcwd(),
        "hardened_mode": os.environ.get("YARN_ENABLE_HARDENED_MODE"),
        "checksum_behavior": os.environ.get("YARN_CHECKSUM_BEHAVIOR"),
        "network_concurrency": os.environ.get("YARN_NETWORK_CONCURRENCY"),
    }) + "\\n")
plan = json.loads(Path(os.environ["YARN_TEST_PLAN"]).read_text())
result = plan[min(attempt, len(plan) - 1)]
print(result.get("stdout", ""), end="")
print(result.get("stderr", ""), end="", file=sys.stderr)
sys.exit(result["exit"])
""",
        )
        self.write_executable(
            "sleep",
            """
import json
import os
from pathlib import Path
import sys

with Path(os.environ["YARN_TEST_SLEEPS"]).open("a") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
""",
        )

    def write_executable(self, name: str, source: str) -> None:
        path = self.bin / name
        path.write_text(f"#!{sys.executable}\n" + source)
        path.chmod(0o755)

    def run_helper(self, plan: list[dict], *args: str) -> subprocess.CompletedProcess:
        self.plan.write_text(json.dumps(plan))
        result = subprocess.run(
            ["bash", str(HELPER), *args],
            cwd=self.root,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(list(self.log_directory.iterdir()), [], "temporary log leaked")
        return result

    def call_records(self) -> list[dict]:
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def sleep_records(self) -> list[list[str]]:
        if not self.sleeps.exists():
            return []
        return [json.loads(line) for line in self.sleeps.read_text().splitlines()]

    def test_success_exits_without_retry(self) -> None:
        result = self.run_helper([{"exit": 0, "stdout": "installed successfully\n"}])

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("installed successfully", result.stdout)
        self.assertEqual(len(self.call_records()), 1)
        self.assertEqual(self.sleep_records(), [])

    def test_targeted_crash_then_success_retries_once(self) -> None:
        result = self.run_helper(
            [
                {"exit": 1, "stderr": RETRYABLE_ERROR + "\n"},
                {"exit": 0, "stdout": "second attempt succeeded\n"},
            ]
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(RETRYABLE_ERROR, result.stdout)
        self.assertIn("second attempt succeeded", result.stdout)
        self.assertEqual(len(self.call_records()), 2)
        self.assertEqual(self.sleep_records(), [["5"]])

    def test_persistent_crash_stops_after_three_attempts(self) -> None:
        result = self.run_helper([{"exit": 23, "stderr": RETRYABLE_ERROR + "\n"}])

        self.assertEqual(result.returncode, 23)
        self.assertEqual(len(self.call_records()), 3)
        self.assertEqual(self.sleep_records(), [["5"], ["10"]])
        self.assertEqual(result.stdout.count(RETRYABLE_ERROR), 3)

    def test_ordinary_install_failure_is_not_retried(self) -> None:
        result = self.run_helper(
            [{"exit": 42, "stderr": "YN0028: The lockfile would have been modified\n"}]
        )

        self.assertEqual(result.returncode, 42)
        self.assertEqual(len(self.call_records()), 1)
        self.assertEqual(self.sleep_records(), [])

    def test_source_dump_containing_error_string_is_not_a_match(self) -> None:
        result = self.run_helper(
            [
                {
                    "exit": 2,
                    "stderr": (
                        f'const source = "{RETRYABLE_ERROR}";\n'
                        "Error: A different installation failure\n"
                    ),
                }
            ]
        )

        self.assertEqual(result.returncode, 2)
        self.assertEqual(len(self.call_records()), 1)
        self.assertEqual(self.sleep_records(), [])

    def test_deterministic_failure_after_retry_stops_immediately(self) -> None:
        result = self.run_helper(
            [
                {"exit": 1, "stderr": RETRYABLE_ERROR + "\n"},
                {"exit": 17, "stderr": "YN0018: Remote archive checksum mismatch\n"},
            ]
        )

        self.assertEqual(result.returncode, 17)
        self.assertEqual(len(self.call_records()), 2)
        self.assertEqual(self.sleep_records(), [["5"]])

    def test_arguments_directory_and_yarn_settings_are_preserved(self) -> None:
        self.environment.update(
            {
                "YARN_ENABLE_HARDENED_MODE": "1",
                "YARN_CHECKSUM_BEHAVIOR": "throw",
                "YARN_NETWORK_CONCURRENCY": "7",
            }
        )
        args = ["--immutable", "--inline-builds", "argument with spaces"]
        result = self.run_helper(
            [
                {"exit": 1, "stderr": RETRYABLE_ERROR + "\n"},
                {"exit": 0},
            ],
            *args,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = self.call_records()
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(record["args"], ["install", *args])
            self.assertEqual(Path(record["cwd"]).resolve(), self.root.resolve())
            self.assertEqual(record["hardened_mode"], "1")
            self.assertEqual(record["checksum_behavior"], "throw")
            self.assertEqual(record["network_concurrency"], "7")


if __name__ == "__main__":
    unittest.main()

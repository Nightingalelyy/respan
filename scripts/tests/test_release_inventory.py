from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import release_inventory  # noqa: E402


class JavaScriptPublicationOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entries = [
            {
                "name": "@respan/respan-sdk",
                "ecosystem": "javascript",
                "path": "javascript-sdks/respan-sdk",
            },
            {
                "name": "@respan/tracing",
                "ecosystem": "javascript",
                "path": "javascript-sdks/respan-tracing",
            },
            {
                "name": "@respan/instrumentation-vercel",
                "ecosystem": "javascript",
                "path": "javascript-sdks/instrumentations/respan-instrumentation-vercel",
            },
        ]
        self.graph = {
            "@respan/respan-sdk": set(),
            "@respan/tracing": {"@respan/respan-sdk"},
            "@respan/instrumentation-vercel": {"@respan/tracing"},
        }

    def test_publish_order_is_topological_but_excludes_unplanned_dependencies(self) -> None:
        with mock.patch.object(
            release_inventory,
            "js_internal_dependency_names",
            return_value=self.graph,
        ):
            ordered = release_inventory.javascript_publish_order(
                self.entries,
                ["@respan/instrumentation-vercel", "@respan/tracing"],
            )

        self.assertEqual(
            [entry["name"] for entry in ordered],
            ["@respan/tracing", "@respan/instrumentation-vercel"],
        )

    def test_build_order_includes_transitive_internal_dependencies(self) -> None:
        with mock.patch.object(
            release_inventory,
            "js_internal_dependency_names",
            return_value=self.graph,
        ):
            ordered = release_inventory.javascript_build_order_for_packages(
                self.entries,
                ["@respan/instrumentation-vercel"],
            )

        self.assertEqual(
            [entry["name"] for entry in ordered],
            [
                "@respan/respan-sdk",
                "@respan/tracing",
                "@respan/instrumentation-vercel",
            ],
        )

    def test_publish_order_rejects_dependency_cycles(self) -> None:
        cyclic_graph = {
            **self.graph,
            "@respan/tracing": {"@respan/instrumentation-vercel"},
        }
        with mock.patch.object(
            release_inventory,
            "js_internal_dependency_names",
            return_value=cyclic_graph,
        ):
            with self.assertRaisesRegex(ValueError, "cyclic javascript dependency"):
                release_inventory.javascript_publish_order(
                    self.entries,
                    ["@respan/tracing", "@respan/instrumentation-vercel"],
                )


if __name__ == "__main__":
    unittest.main()

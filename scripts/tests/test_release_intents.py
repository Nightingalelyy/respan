from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import release_intents  # noqa: E402


class ResolveVersionsTests(unittest.TestCase):
    def test_resolves_one_atomic_map_for_runtime_and_dependent_plugin(self) -> None:
        tracing = {
            "name": "@respan/tracing",
            "ecosystem": "javascript",
        }
        plugin = {
            "name": "@respan/instrumentation-vercel",
            "ecosystem": "javascript",
        }
        entries = {
            tracing["name"]: tracing,
            plugin["name"]: plugin,
        }
        manifests = {
            tracing["name"]: {"version": "1.1.6"},
            plugin["name"]: {"version": "0.1.0"},
        }
        registry_versions = {
            tracing["name"]: "1.1.7",
            plugin["name"]: "0.1.0",
        }
        records = [
            {"name": tracing["name"], "bump": "minor"},
            {"name": plugin["name"], "bump": "patch"},
        ]

        with mock.patch.object(
            release_intents,
            "inventory_entries_by_name",
            return_value=entries,
        ):
            with mock.patch.object(
                release_intents.release_inventory,
                "load_manifest",
                side_effect=lambda entry: manifests[entry["name"]],
            ):
                with mock.patch.object(
                    release_intents,
                    "latest_registry_version",
                    side_effect=lambda entry: registry_versions[entry["name"]],
                ):
                    versions = release_intents.resolve_versions(records, prefer_registry=True)

        self.assertEqual(
            versions,
            {
                "@respan/tracing": "1.2.0",
                "@respan/instrumentation-vercel": "0.1.1",
            },
        )

    def test_applies_every_resolved_version_and_returns_matrix_target(self) -> None:
        versions = {
            "@respan/tracing": "1.2.0",
            "@respan/instrumentation-vercel": "0.1.1",
        }

        with mock.patch.object(release_intents, "set_versions") as set_versions:
            selected = release_intents.apply_resolved_versions(
                versions,
                "@respan/instrumentation-vercel",
            )

        set_versions.assert_called_once_with(versions)
        self.assertEqual(selected, "0.1.1")

    def test_rejects_unknown_matrix_target_before_mutating_manifests(self) -> None:
        with mock.patch.object(release_intents, "set_versions") as set_versions:
            with self.assertRaises(KeyError):
                release_intents.apply_resolved_versions(
                    {"@respan/tracing": "1.2.0"},
                    "@respan/instrumentation-vercel",
                )

        set_versions.assert_not_called()

    def test_applies_one_resolved_map_for_the_complete_publication_plan(self) -> None:
        versions = {
            "@respan/tracing": "1.2.0",
            "@respan/instrumentation-vercel": "0.1.1",
        }

        with mock.patch.object(release_intents, "set_versions") as set_versions:
            applied = release_intents.apply_resolved_versions_for_plan(
                versions,
                ["@respan/instrumentation-vercel", "@respan/tracing"],
            )

        set_versions.assert_called_once_with(versions)
        self.assertEqual(applied, versions)

    def test_rejects_plan_version_mismatch_before_mutating_manifests(self) -> None:
        with mock.patch.object(release_intents, "set_versions") as set_versions:
            with self.assertRaisesRegex(ValueError, "missing resolved versions"):
                release_intents.apply_resolved_versions_for_plan(
                    {"@respan/tracing": "1.2.0"},
                    ["@respan/tracing", "@respan/instrumentation-vercel"],
                )

        set_versions.assert_not_called()


class PublishedVersionArtifactTests(unittest.TestCase):
    def test_merges_aggregate_javascript_and_single_python_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_root = Path(temporary_directory)
            javascript = artifact_root / "published-versions-javascript"
            python = artifact_root / "published-version-python-respan-ai"
            javascript.mkdir()
            python.mkdir()
            (javascript / "versions.json").write_text(
                json.dumps(
                    {
                        "@respan/tracing": "1.2.0",
                        "@respan/instrumentation-vercel": "0.1.1",
                    }
                )
            )
            (python / "version.json").write_text(
                json.dumps({"name": "respan-ai", "version": "0.3.0"})
            )

            versions = release_intents.collect_published_versions(artifact_root)

        self.assertEqual(
            versions,
            {
                "@respan/instrumentation-vercel": "0.1.1",
                "@respan/tracing": "1.2.0",
                "respan-ai": "0.3.0",
            },
        )

    def test_rejects_conflicting_artifact_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_root = Path(temporary_directory)
            aggregate = artifact_root / "published-versions-javascript"
            singleton = artifact_root / "published-version-python-collision"
            aggregate.mkdir()
            singleton.mkdir()
            (aggregate / "versions.json").write_text(
                json.dumps({"collision": "1.0.0"})
            )
            (singleton / "version.json").write_text(
                json.dumps({"name": "collision", "version": "2.0.0"})
            )

            with self.assertRaisesRegex(ValueError, "conflicting published versions"):
                release_intents.collect_published_versions(artifact_root)


class PublishWorkflowTests(unittest.TestCase):
    def test_javascript_publication_is_preflighted_and_dependency_ordered(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "publish.yml").read_text()
        self.assertIn(
            "ref: ${{ github.event_name == 'workflow_run' && "
            "github.event.workflow_run.head_sha || inputs.head_ref || github.sha }}",
            workflow,
        )
        javascript_job = workflow.split("  publish-javascript:\n", 1)[1].split(
            "  build-python:\n",
            1,
        )[0]

        apply_position = javascript_job.index("apply-resolved-versions")
        install_position = javascript_job.index("yarn install")
        pack_position = javascript_job.index("yarn pack --out")
        publish_position = javascript_job.index("npm publish")
        smoke_position = javascript_job.index("npm view")

        self.assertLess(apply_position, install_position)
        self.assertLess(install_position, pack_position)
        self.assertLess(pack_position, publish_position)
        self.assertLess(publish_position, smoke_position)
        self.assertEqual(javascript_job.count("yarn pack --out"), 1)
        self.assertEqual(javascript_job.count("npm publish"), 1)
        self.assertNotIn("strategy:", javascript_job)
        self.assertNotIn("matrix.", javascript_job)
        self.assertNotIn("release_intents.py apply --package", javascript_job)
        self.assertIn("--plan-file .release-sync/javascript-plan.json", javascript_job)
        self.assertIn("--javascript-build-order-file", javascript_job)
        self.assertIn("--javascript-publish-order-file", javascript_job)
        self.assertIn(
            "Publish and smoke JavaScript packages in dependency order",
            javascript_job,
        )
        self.assertIn("id-token: write", javascript_job)
        self.assertIn("name: npm", javascript_job)
        self.assertIn("ref: ${{ needs.discover.outputs.head_ref }}", javascript_job)
        self.assertIn(
            "JAVASCRIPT_VERSIONS: ${{ needs.discover.outputs.javascript_versions }}",
            javascript_job,
        )
        self.assertIn("path: .release-sync/versions.json", javascript_job)
        self.assertIn("name: published-versions-javascript", javascript_job)
        self.assertIn("pattern: published-version*", workflow)
        self.assertIn("merge-version-artifacts", workflow)
        self.assertIn(
            "python3 scripts/release_intents.py set-versions --file "
            ".release-sync/versions.json",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Audit the isolated MUNO21 official graph-metric runtime and data inputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

PINNED_COMMIT = "4245b2a46485294878d3c26ccb252f5aac01acd7"


def _command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=cwd,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def audit_graph_eval(project_root: Path, storage_root: Path) -> dict[str, Any]:
    graph_env = storage_root / "envs/activemap-graph"
    python = graph_env / "bin/python"
    go = graph_env / "bin/go"
    official = project_root / "external/muno21-official"
    data = storage_root / "datasets/muno21/extracted/mapupdate"
    checks: dict[str, dict[str, Any]] = {}

    checks["graph_python"] = {"passed": python.is_file(), "path": str(python)}
    checks["go_binary"] = {"passed": go.is_file(), "path": str(go)}
    if python.is_file():
        passed, detail = _command(
            [
                str(python),
                "-c",
                "import numpy,scipy,networkx,skimage; print(skimage.__version__)",
            ]
        )
        checks["python_dependencies"] = {"passed": passed, "detail": detail}
    else:
        checks["python_dependencies"] = {"passed": False, "detail": "python missing"}
    if go.is_file():
        passed, detail = _command([str(go), "version"])
        checks["go_runtime"] = {"passed": passed, "detail": detail}
    else:
        checks["go_runtime"] = {"passed": False, "detail": "go missing"}

    if (official / ".git").is_dir():
        passed, detail = _command(["git", "-C", str(official), "rev-parse", "HEAD"])
        commit = detail.splitlines()[-1] if passed and detail else None
        checks["official_commit"] = {
            "passed": passed and commit == PINNED_COMMIT,
            "actual": commit,
            "expected": PINNED_COMMIT,
        }
    else:
        checks["official_commit"] = {
            "passed": False,
            "actual": None,
            "expected": PINNED_COMMIT,
        }
    for relative in ("go/metrics/apls.py", "go/metrics/geo.go", "go/metrics/error_rate.go"):
        checks[f"official_file:{relative}"] = {
            "passed": (official / relative).is_file(),
            "path": str(official / relative),
        }
    discoverlib = official / "python/discoverlib"
    checks["official_discoverlib"] = {
        "passed": discoverlib.is_dir(),
        "path": str(discoverlib),
    }
    if python.is_file() and discoverlib.is_dir():
        passed, detail = _command(
            [
                str(python),
                "-c",
                (
                    "import sys; "
                    f"sys.path.insert(0, {str((official / 'python').resolve())!r}); "
                    "from discoverlib import geom, graph; print('discoverlib ready')"
                ),
            ]
        )
        checks["official_apls_import"] = {"passed": passed, "detail": detail}
    else:
        checks["official_apls_import"] = {
            "passed": False,
            "detail": "graph Python or official discoverlib missing",
        }
    go_root = official / "go"
    if go.is_file() and (go_root / "go.mod").is_file():
        passed, detail = _command(
            [str(go), "list", "-mod=readonly", "./..."],
            cwd=go_root,
            env={**os.environ, "GOPROXY": "off"},
        )
        checks["official_go_modules_offline"] = {
            "passed": passed,
            "detail": detail,
        }
    else:
        checks["official_go_modules_offline"] = {
            "passed": False,
            "detail": "Go runtime or official go.mod missing",
        }
    for relative in ("annotations.json", "graphs/graphs"):
        path = data / relative
        checks[f"data:{relative}"] = {
            "passed": path.is_file() if path.suffix else path.is_dir(),
            "path": str(path),
        }
    infrastructure_keys = [
        key for key in checks if not key.startswith("data:")
    ]
    return {
        "schema_version": "muno21-graph-eval-readiness-v1",
        "pinned_commit": PINNED_COMMIT,
        "integration": "external_subprocess_only",
        "third_party_source_imported": False,
        "test_assets_read": False,
        "infrastructure_ready": all(checks[key]["passed"] for key in infrastructure_keys),
        "ready": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("storage_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = audit_graph_eval(args.project_root, args.storage_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

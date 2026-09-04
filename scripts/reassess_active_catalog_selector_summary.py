from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.evaluate_active_catalog_selector import promotion_gate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    source = json.loads(args.summary.read_text(encoding="utf-8"))
    gate = promotion_gate(
        float(source["valid_action_rate"]),
        source["metrics"],
        source["aoi_bootstrap"],
    )
    payload = {
        "schema_version": "active-catalog-selector-promotion-reassessment-v2",
        "reason": (
            "align implementation with the predeclared >=0.99 schema/executable "
            "validity threshold; retain exact-1 validity as an audit field"
        ),
        "source_summary": {
            "path": str(args.summary.resolve()),
            "sha256": _sha256(args.summary),
            "original_gate": source["promotion_gate"],
        },
        "valid_action_rate": source["valid_action_rate"],
        "promotion_gate": gate,
        "test_assets_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

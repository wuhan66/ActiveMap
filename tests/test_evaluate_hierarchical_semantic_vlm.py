import hashlib
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from scripts.evaluate_hierarchical_semantic_vlm import load_gate_probabilities


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_gate_probabilities_verifies_frozen_validation_features(tmp_path):
    feature_root = tmp_path / "features"
    gate_root = tmp_path / "gate"
    feature_root.mkdir()
    gate_root.mkdir()
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float16)
    labels = np.asarray([0, 0, 1, 1])
    np.save(feature_root / "features.npy", features)
    records = [
        {"example_id": f"e{index}", "split": "val", "oracle_use_tool": bool(label)}
        for index, label in enumerate(labels)
    ]
    (feature_root / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    feature_summary_path = feature_root / "summary.json"
    feature_summary_path.write_text(
        json.dumps({"test_assets_read": False}) + "\n", encoding="utf-8"
    )
    model = LogisticRegression().fit(features.astype(np.float32), labels)
    joblib.dump(model, gate_root / "gate.joblib")
    gate_summary = {
        "test_assets_read": False,
        "selected": {"threshold": 0.4},
        "sources": {"val": {"summary_sha256": _sha256(feature_summary_path)}},
    }
    (gate_root / "summary.json").write_text(
        json.dumps(gate_summary) + "\n", encoding="utf-8"
    )
    probabilities, threshold, loaded = load_gate_probabilities(feature_root, gate_root)
    assert set(probabilities) == {"e0", "e1", "e2", "e3"}
    assert threshold == 0.4
    assert loaded == gate_summary


def test_load_gate_probabilities_rejects_changed_feature_manifest(tmp_path):
    feature_root = tmp_path / "features"
    gate_root = tmp_path / "gate"
    feature_root.mkdir()
    gate_root.mkdir()
    np.save(feature_root / "features.npy", np.asarray([[0.0], [1.0]], dtype=np.float16))
    (feature_root / "records.jsonl").write_text(
        json.dumps({"example_id": "a"}) + "\n" + json.dumps({"example_id": "b"}) + "\n"
    )
    (feature_root / "summary.json").write_text(
        json.dumps({"test_assets_read": False}) + "\n", encoding="utf-8"
    )
    (gate_root / "summary.json").write_text(
        json.dumps(
            {
                "test_assets_read": False,
                "selected": {"threshold": 0.5},
                "sources": {"val": {"summary_sha256": "0" * 64}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        load_gate_probabilities(feature_root, gate_root)
    except ValueError as error:
        assert "not calibrated" in str(error)
    else:
        raise AssertionError("changed feature manifests must be rejected")


def test_load_gate_probabilities_supports_predicted_utility(tmp_path):
    feature_root = tmp_path / "features"
    gate_root = tmp_path / "gate"
    feature_root.mkdir()
    gate_root.mkdir()
    features = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float16)
    np.save(feature_root / "features.npy", features)
    (feature_root / "records.jsonl").write_text(
        "".join(
            json.dumps({"example_id": f"e{index}"}) + "\n" for index in range(3)
        ),
        encoding="utf-8",
    )
    feature_summary_path = feature_root / "summary.json"
    feature_summary_path.write_text(
        json.dumps({"test_assets_read": False}) + "\n", encoding="utf-8"
    )
    model = Ridge(alpha=1.0).fit(features.astype(np.float32), [-1.0, 0.0, 1.0])
    joblib.dump(model, gate_root / "gate.joblib")
    gate_summary = {
        "test_assets_read": False,
        "score_type": "predicted_utility",
        "selected": {"threshold": 0.1},
        "sources": {"val": {"summary_sha256": _sha256(feature_summary_path)}},
    }
    (gate_root / "summary.json").write_text(
        json.dumps(gate_summary) + "\n", encoding="utf-8"
    )

    scores, threshold, _ = load_gate_probabilities(feature_root, gate_root)

    assert threshold == 0.1
    assert scores["e0"] < scores["e1"] < scores["e2"]

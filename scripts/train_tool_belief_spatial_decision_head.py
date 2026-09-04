#!/usr/bin/env python3
"""Train a causal decision head over belief features and real change rasters."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scripts.train_tool_belief_decision_head import (
    FEATURE_DIM,
    KEEP_INDEX,
    UPDATE_OPERATIONS,
    DecisionTrajectoryDataset,
    _calibrate,
    _feature_names,
    _json_metrics,
    _operation_weights,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def temporal_artifact_index(records: Path, artifact_dir: Path) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for row in _read_jsonl(records):
        result = row.get("tool_result", {})
        if result.get("tool") != "TEMPORAL_CHANGE":
            continue
        key = (str(row["episode_id"]), str(row["evidence_id"]))
        path = artifact_dir / f"{result['call_id']}_change.npy"
        if key in index and index[key] != path:
            raise ValueError(f"duplicate temporal artifact mapping for {key}")
        if not path.is_file():
            raise FileNotFoundError(path)
        index[key] = path
    if not index:
        raise ValueError(f"no temporal artifacts indexed from {records}")
    return index


def causal_spatial_stack(
    artifacts: list[torch.Tensor], stage: int, *, spatial_size: int
) -> torch.Tensor:
    if len(artifacts) != 3 or stage not in {0, 1, 2, 3}:
        raise ValueError("three artifacts and a stage in 0,1,2,3 are required")
    zero = torch.zeros((spatial_size, spatial_size), dtype=torch.float32)
    return torch.stack(
        [artifacts[index] if index < stage else zero for index in range(3)], dim=0
    )


class SpatialDecisionDataset(Dataset):
    def __init__(
        self,
        details: Path,
        records: Path,
        artifact_dir: Path,
        expected_split: str,
        *,
        spatial_size: int = 64,
        stages: tuple[int, ...] = (0, 1, 2, 3),
    ) -> None:
        self.base = DecisionTrajectoryDataset(details, expected_split, stages=stages)
        self.examples = self.base.examples
        self.class_counts = self.base.class_counts
        self.spatial_size = spatial_size
        artifact_index = temporal_artifact_index(records, artifact_dir)
        grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        for row in _read_jsonl(details):
            if row.get("split", expected_split) != expected_split:
                raise ValueError(f"trajectory row is not from {expected_split}")
            grouped.setdefault(str(row["sequence_id"]), []).append(row)
        paths_by_sequence: dict[str, list[Path]] = {}
        required_paths: set[Path] = set()
        for sequence_id, rows in grouped.items():
            rows.sort(key=lambda item: int(item["step"]))
            if [int(row["step"]) for row in rows] != [1, 2, 3]:
                raise ValueError(f"{sequence_id} does not contain steps 1, 2, 3")
            paths = []
            for row in rows:
                key = (str(row["episode_id"]), str(row["evidence_id"]))
                if key not in artifact_index:
                    raise ValueError(f"trajectory evidence has no temporal artifact: {key}")
                paths.append(artifact_index[key])
            paths_by_sequence[sequence_id] = paths
            required_paths.update(paths)
        self.artifacts = {
            path: self._load_artifact(path, spatial_size) for path in sorted(required_paths)
        }
        self.paths_by_sequence = paths_by_sequence

    @staticmethod
    def _load_artifact(path: Path, spatial_size: int) -> torch.Tensor:
        array = np.load(path, allow_pickle=False).astype(np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError(f"invalid temporal artifact: {path} {array.shape}")
        tensor = torch.from_numpy(array)[None, None]
        if tensor.shape[-2:] != (spatial_size, spatial_size):
            tensor = F.interpolate(
                tensor, size=(spatial_size, spatial_size), mode="bilinear", align_corners=False
            )
        return tensor[0, 0].clamp(0.0, 1.0)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        features, target, baseline, residual, stage = self.base[index]
        example = self.examples[index]
        artifacts = [
            self.artifacts[path] for path in self.paths_by_sequence[example["sequence_id"]]
        ]
        spatial = causal_spatial_stack(
            artifacts, int(stage.item()), spatial_size=self.spatial_size
        )
        return features, spatial, target, baseline, residual, stage


class SpatialHierarchicalDecisionHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        spatial_base_channels: int = 16,
        dropout: float = 0.15,
        use_spatial: bool = True,
        fusion_mode: str = "concat",
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if fusion_mode not in {"concat", "gated_residual"}:
            raise ValueError(f"unsupported fusion mode: {fusion_mode}")
        self.use_spatial = use_spatial
        self.fusion_mode = fusion_mode
        base = spatial_base_channels
        self.belief_encoder = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(3, base, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(4, base),
            nn.GELU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, base * 2),
            nn.GELU(),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, base * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(base * 16, hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if fusion_mode == "gated_residual":
            self.spatial_residual = nn.Linear(hidden_dim, hidden_dim)
            self.spatial_gate = nn.Linear(hidden_dim * 2, hidden_dim)
            self.residual_fusion = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            nn.init.zeros_(self.spatial_residual.weight)
            nn.init.zeros_(self.spatial_residual.bias)
            nn.init.zeros_(self.spatial_gate.weight)
            nn.init.constant_(self.spatial_gate.bias, gate_bias)
        self.update_head = nn.Linear(hidden_dim, 1)
        self.operation_head = nn.Linear(hidden_dim, len(UPDATE_OPERATIONS))

    def forward(
        self, belief_features: torch.Tensor, spatial_evidence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        belief = self.belief_encoder(belief_features)
        spatial = self.spatial_encoder(spatial_evidence)
        if not self.use_spatial:
            spatial = torch.zeros_like(spatial)
        if self.fusion_mode == "gated_residual":
            gate = torch.sigmoid(self.spatial_gate(torch.cat((belief, spatial), dim=1)))
            residual = self.spatial_residual(spatial)
            if not self.use_spatial:
                residual = torch.zeros_like(residual)
            hidden = self.residual_fusion(belief + gate * residual)
        else:
            hidden = self.fusion(torch.cat((belief, spatial), dim=1))
        return self.update_head(hidden).squeeze(-1), self.operation_head(hidden)


def _run_epoch(
    loader: DataLoader,
    model: SpatialHierarchicalDecisionHead,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    operation_weights: torch.Tensor,
    keep_weight: float,
    update_weight: float,
    operation_loss_weight: float,
    safety_margin: float,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    collected: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "target",
            "baseline",
            "residual",
            "update_probability",
            "update_operation",
            "stage",
        )
    }
    for features, spatial, target, baseline, residual, stage in tqdm(
        loader, leave=False, unit="batch", desc="train" if training else "val"
    ):
        features = features.to(device)
        spatial = spatial.to(device)
        target_device = target.to(device)
        update_target = (target_device != KEEP_INDEX).float()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            update_logit, operation_logit = model(features, spatial)
            binary = F.binary_cross_entropy_with_logits(
                update_logit, update_target, reduction="none"
            )
            weights = torch.where(
                update_target > 0.5,
                torch.full_like(update_target, update_weight),
                torch.full_like(update_target, keep_weight),
            )
            binary_loss = torch.mean(binary * weights)
            update_mask = target_device != KEEP_INDEX
            operation_loss = (
                F.cross_entropy(
                    operation_logit[update_mask],
                    target_device[update_mask] - 1,
                    weight=operation_weights,
                )
                if bool(update_mask.any())
                else update_logit.sum() * 0.0
            )
            loss = binary_loss + operation_loss_weight * operation_loss
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        count = int(features.shape[0])
        total_loss += float(loss.detach()) * count
        sample_count += count
        collected["target"].append(target.numpy())
        collected["baseline"].append(baseline.numpy())
        collected["residual"].append(residual.numpy())
        collected["stage"].append(stage.numpy())
        collected["update_probability"].append(
            torch.sigmoid(update_logit).detach().cpu().numpy()
        )
        collected["update_operation"].append(
            (torch.argmax(operation_logit, dim=1) + 1).detach().cpu().numpy()
        )
    values = {name: np.concatenate(parts) for name, parts in collected.items()}
    stages = sorted(int(value) for value in np.unique(values["stage"]))
    calibrated_by_stage = {
        str(stage): _calibrate(
            values["target"][values["stage"] == stage],
            values["baseline"][values["stage"] == stage],
            values["update_probability"][values["stage"] == stage],
            values["update_operation"][values["stage"] == stage],
            safety_margin=safety_margin,
        )
        for stage in stages
    }
    residual_counts = Counter(values["residual"].tolist())
    return {
        "loss": total_loss / sample_count,
        "calibrated": None,
        "calibrated_by_stage": calibrated_by_stage,
        "residual_metrics": {"class_counts": dict(residual_counts)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_details", type=Path)
    parser.add_argument("val_details", type=Path)
    parser.add_argument("train_records", type=Path)
    parser.add_argument("val_records", type=Path)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--spatial-base-channels", type=int, default=16)
    parser.add_argument("--spatial-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--keep-weight", type=float, default=2.0)
    parser.add_argument("--update-weight", type=float, default=1.0)
    parser.add_argument("--operation-loss-weight", type=float, default=1.0)
    parser.add_argument("--safety-margin", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--no-spatial", action="store_true")
    parser.add_argument(
        "--fusion-mode", choices=("concat", "gated_residual"), default="concat"
    )
    parser.add_argument("--gate-bias", type=float, default=-2.0)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    stages = (0, 1, 2, 3)
    train_data = SpatialDecisionDataset(
        args.train_details,
        args.train_records,
        args.artifact_dir,
        "train",
        spatial_size=args.spatial_size,
        stages=stages,
    )
    val_data = SpatialDecisionDataset(
        args.val_details,
        args.val_records,
        args.artifact_dir,
        "val",
        spatial_size=args.spatial_size,
        stages=stages,
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    model = SpatialHierarchicalDecisionHead(
        hidden_dim=args.hidden_dim,
        spatial_base_channels=args.spatial_base_channels,
        dropout=args.dropout,
        use_spatial=not args.no_spatial,
        fusion_mode=args.fusion_mode,
        gate_bias=args.gate_bias,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(args.patience // 4, 1)
    )
    operation_weights = _operation_weights(train_data, device)
    config = {
        "protocol": (
            "spatial_hierarchical_decision_head_v2_gated_residual"
            if args.fusion_mode == "gated_residual"
            else "spatial_hierarchical_decision_head_v1"
        ),
        "feature_dim": FEATURE_DIM,
        "feature_names": _feature_names(),
        "spatial_channels": 3,
        "spatial_size": args.spatial_size,
        "causal_stages": list(stages),
        "use_spatial": not args.no_spatial,
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "hidden_dim": args.hidden_dim,
        "spatial_base_channels": args.spatial_base_channels,
        "dropout": args.dropout,
        "fusion_mode": args.fusion_mode,
        "gate_bias": args.gate_bias,
        "seed": args.seed,
        "test_assets_read": False,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    if args.verify_only:
        features, spatial, target, _baseline, _residual, stage = next(iter(val_loader))
        with torch.inference_mode():
            update_logit, operation_logit = model(features.to(device), spatial.to(device))
        preflight = {
            **config,
            "mode": "verify_only",
            "batch_size": int(features.shape[0]),
            "belief_shape": list(features.shape),
            "spatial_shape": list(spatial.shape),
            "target_shape": list(target.shape),
            "stage_values": sorted(set(int(value) for value in stage.tolist())),
            "update_logit_shape": list(update_logit.shape),
            "operation_logit_shape": list(operation_logit.shape),
            "finite_outputs": bool(
                torch.isfinite(update_logit).all() and torch.isfinite(operation_logit).all()
            ),
        }
        (args.output_dir / "preflight.json").write_text(
            json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(preflight, indent=2))
        return
    best_f1 = -math.inf
    best_epoch: int | None = None
    best_val: dict[str, Any] | None = None
    stale = 0
    history_path = args.output_dir / "history.jsonl"
    with history_path.open("w", encoding="utf-8") as history:
        for epoch in range(1, args.epochs + 1):
            train_result = _run_epoch(
                train_loader,
                model,
                device=device,
                optimizer=optimizer,
                operation_weights=operation_weights,
                keep_weight=args.keep_weight,
                update_weight=args.update_weight,
                operation_loss_weight=args.operation_loss_weight,
                safety_margin=args.safety_margin,
            )
            with torch.inference_mode():
                val_result = _run_epoch(
                    val_loader,
                    model,
                    device=device,
                    optimizer=None,
                    operation_weights=operation_weights,
                    keep_weight=args.keep_weight,
                    update_weight=args.update_weight,
                    operation_loss_weight=args.operation_loss_weight,
                    safety_margin=args.safety_margin,
                )
            scheduler.step(val_result["loss"])
            train_json = _json_metrics(train_result)
            val_json = _json_metrics(val_result)
            history.write(json.dumps({"epoch": epoch, "train": train_json, "val": val_json}) + "\n")
            history.flush()
            stage_values = val_result["calibrated_by_stage"]
            all_safe = all(value is not None for value in stage_values.values())
            val_f1 = (
                float(np.mean([value["metrics"]["macro_f1"] for value in stage_values.values()]))
                if all_safe
                else -math.inf
            )
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "config": config,
                "decision_thresholds": {
                    stage: value["threshold"]
                    for stage, value in stage_values.items()
                    if value is not None
                },
                "val_metrics": val_json,
            }
            torch.save(checkpoint, args.output_dir / "last.pt")
            if val_f1 > best_f1 + 1e-8:
                best_f1 = val_f1
                best_epoch = epoch
                best_val = val_json
                stale = 0
                torch.save(checkpoint, args.output_dir / "best_safety.pt")
            else:
                stale += 1
            print(
                f"epoch={epoch}/{args.epochs} train_loss={train_result['loss']:.6f} "
                f"val_loss={val_result['loss']:.6f} val_safe_macro_f1={val_f1:.6f}",
                flush=True,
            )
            if stale >= args.patience:
                break
    summary = {
        **config,
        "best_epoch": best_epoch,
        "best_safe_macro_f1": best_f1 if best_epoch is not None else None,
        "best_val": best_val,
        "epochs_completed": epoch,
        "passed_hard_safety": best_epoch is not None,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

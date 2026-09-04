#!/usr/bin/env python3
"""Evaluate zero-shot Direct VLM actions on leakage-separated MUNO21 visuals."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


SYSTEM_PROMPTS = {
    "minimal_v1": """You are a structured editable-map update policy.
Inspect the current aerial image and the old editable road map. Return exactly
one JSON action and no prose. Valid outputs are:
{"action":"REJECT"}
{"action":"COMMIT","edit":"ADD"}
{"action":"COMMIT","edit":"DELETE"}
{"action":"COMMIT","edit":"RESHAPE"}
Reject means keep the old map unchanged.""",
    "operational_v2": """You are a conservative structured editable-map update
policy. Compare the current aerial RGB image with the old editable road map.
Use these operation definitions:
- REJECT: keep the old map when it matches the visible road, or when evidence
  is unclear, occluded, or ambiguous.
- ADD: a road is clearly visible in the current RGB image but is absent from
  the old map.
- DELETE: an old-map road is clearly absent from the current RGB image.
- RESHAPE: the same road exists in both, but the old-map geometry is clearly
  displaced or has the wrong shape.
Do not choose RESHAPE merely because of line thickness, rasterization, or a
small overlay misalignment. First decide presence versus absence, then geometry.
Return exactly one JSON action and no prose. Valid outputs are:
{"action":"REJECT"}
{"action":"COMMIT","edit":"ADD"}
{"action":"COMMIT","edit":"DELETE"}
{"action":"COMMIT","edit":"RESHAPE"}""",
}


def _load_jsonl(path: Path, expected_split: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    if any(row.get("split") != expected_split for row in rows):
        raise ValueError(
            f"Direct VLM expected {expected_split} records only: {path}"
        )
    return rows


def _terminal_prediction(text: str) -> tuple[str, bool, str | None]:
    from activemap.agent.records import AgentAction
    from activemap.agent.vlm_evaluation import extract_json_object

    try:
        action = AgentAction.model_validate(extract_json_object(text))
    except Exception as exc:
        return "REJECT", False, str(exc)
    if action.action.value == "REJECT":
        return "REJECT", True, None
    if action.action.value == "COMMIT" and action.edit is not None:
        return f"COMMIT:{action.edit.value}", True, None
    return "REJECT", False, "Direct VLM emitted a non-terminal action"


def _operation(prediction: str) -> str:
    return "KEEP" if prediction == "REJECT" else prediction.removeprefix("COMMIT:")


def _prompt_text(prompt_version: str, image_mode: str = "triplet") -> str:
    if prompt_version not in SYSTEM_PROMPTS:
        raise ValueError(f"unknown prompt version: {prompt_version}")
    if image_mode == "composite":
        return (
            "The single composite contains three aligned panels: left is the "
            "current aerial RGB image, center is the old editable road-map "
            "mask, and right overlays that exact old map in yellow on the "
            "current image. Judge only whether the old editable road geometry "
            "should remain unchanged or receive one ADD, DELETE, or RESHAPE edit."
        )
    if image_mode != "triplet":
        raise ValueError(f"unknown image mode: {image_mode}")
    return (
        "Image 1 is the current aerial RGB image. Image 2 is the old editable "
        "road-map mask. Image 3 overlays that exact old map in yellow on the "
        "current image. Judge only whether the old editable road geometry "
        "should remain unchanged or receive one ADD, DELETE, or RESHAPE edit."
    )


def _prompt(
    record: dict[str, Any],
    prompt_version: str,
    demonstrations: list[tuple[dict[str, Any], str]],
    image_mode: str = "triplet",
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    image_names = (
        ("composite",)
        if image_mode == "composite"
        else ("rgb", "prior_mask", "overlay")
    )
    images = []
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[prompt_version]}]
    for demonstration, operation in demonstrations:
        demo_images = [
            Image.open(demonstration["images"][name]).convert("RGB")
            for name in image_names
        ]
        images.extend(demo_images)
        messages.extend(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image} for image in demo_images
                    ]
                    + [
                        {
                            "type": "text",
                            "text": "Labeled training demonstration. "
                            + _prompt_text(prompt_version, image_mode),
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": (
                        '{"action":"REJECT"}'
                        if operation == "KEEP"
                        else f'{{"action":"COMMIT","edit":"{operation}"}}'
                    ),
                },
            ]
        )
    query_images = [
        Image.open(record["images"][name]).convert("RGB")
        for name in image_names
    ]
    images.extend(query_images)
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image} for image in query_images
            ]
            + [{"type": "text", "text": _prompt_text(prompt_version, image_mode)}],
        }
    )
    return messages, images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("inputs", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--image-mode", choices=("triplet", "composite"), default="triplet"
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--prompt-version",
        choices=tuple(SYSTEM_PROMPTS),
        default="operational_v2",
    )
    parser.add_argument("--demonstration-inputs", type=Path)
    parser.add_argument("--demonstration-labels", type=Path)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    inputs = _load_jsonl(args.inputs, "val")
    labels = _load_jsonl(args.labels, "val")
    labels_by_id = {str(row["example_id"]): str(row["edit"]) for row in labels}
    if set(labels_by_id) != {str(row["example_id"]) for row in inputs}:
        raise ValueError("Direct VLM input and label supports differ")
    if args.limit is not None:
        inputs = inputs[: args.limit]
    if (args.demonstration_inputs is None) != (
        args.demonstration_labels is None
    ):
        raise ValueError("both demonstration files are required")
    demonstrations = []
    if args.demonstration_inputs is not None:
        demo_inputs = _load_jsonl(args.demonstration_inputs, "train")
        demo_labels = _load_jsonl(args.demonstration_labels, "train")
        demo_labels_by_id = {
            str(row["example_id"]): str(row["edit"]) for row in demo_labels
        }
        if set(demo_labels_by_id) != {
            str(row["example_id"]) for row in demo_inputs
        }:
            raise ValueError("demonstration input and label supports differ")
        demonstrations = [
            (row, demo_labels_by_id[str(row["example_id"])]) for row in demo_inputs
        ]

    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from activemap.agent.identifiers import public_task_id
    from activemap.agent.vlm_evaluation import multiclass_metrics
    from activemap.models import EditOperation

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device)
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    traces = []
    targets = []
    predictions = []
    rollout_rows = []
    valid_count = 0
    budgets = (1.5, 3.0, 4.5)
    for record in tqdm(inputs, desc="MUNO21 Direct VLM"):
        messages, opened_images = _prompt(
            record, args.prompt_version, demonstrations, args.image_mode
        )
        try:
            encoded = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(args.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            continuation = generated[:, encoded["input_ids"].shape[1] :]
            raw = processor.tokenizer.batch_decode(
                continuation, skip_special_tokens=True
            )[0]
        finally:
            for image in opened_images:
                image.close()
        prediction, valid, error = _terminal_prediction(raw)
        valid_count += int(valid)
        target = labels_by_id[str(record["example_id"])]
        predicted_operation = _operation(prediction)
        targets.append(target)
        predictions.append(predicted_operation)
        traces.append(
            {
                "example_id": record["example_id"],
                "split": "val",
                "target_operation": target,
                "prediction": prediction,
                "predicted_operation": predicted_operation,
                "schema_and_executable_valid": valid,
                "fallback_used": not valid,
                "raw_generation": raw,
                "parse_error": error,
            }
        )
        target_action = "REJECT" if target == "KEEP" else f"COMMIT:{target}"
        for budget in budgets:
            rollout_rows.append(
                {
                    "task_id": public_task_id(str(record["example_id"])),
                    "aoi_id": record["observation"]["aoi_id"],
                    "budget": budget,
                    "target": target_action,
                    "prediction": prediction,
                    "selected_evidence_ids": [
                        record["observation"]["anchor_evidence_id"]
                    ],
                    "source_example_id": record["example_id"],
                    "method": "direct_vlm",
                    "test_assets_read": False,
                }
            )

    labels_order = [operation.value for operation in EditOperation]
    metrics = multiclass_metrics(targets, predictions, labels_order)
    false_edits = sum(
        target == "KEEP" and prediction != "KEEP"
        for target, prediction in zip(targets, predictions, strict=True)
    )
    missed_edits = sum(
        target != "KEEP" and prediction == "KEEP"
        for target, prediction in zip(targets, predictions, strict=True)
    )
    false_edit_rate = false_edits / max(sum(item == "KEEP" for item in targets), 1)
    missed_edit_rate = missed_edits / max(sum(item != "KEEP" for item in targets), 1)
    results = [
        {
            "method": "direct_vlm",
            "budget": budget,
            "sample_count": len(inputs),
            "terminal_accuracy": metrics["accuracy"],
            "false_edit_rate": false_edit_rate,
            "missed_edit_rate": missed_edit_rate,
            "mean_acquisitions": 0.0,
            "mean_cost": 0.0,
            "mean_evidence_quality_gain": 0.0,
            "mean_quality_cost_utility": 0.0,
        }
        for budget in budgets
    ]
    summary = {
        "schema_version": "muno21-direct-vlm-evaluation-v1",
        "protocol": {
            "split": "val",
            "test_assets_read": False,
            "visual_inputs": ["current_rgb", "old_map_mask", "old_map_overlay"],
            "image_mode": args.image_mode,
            "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
            "prompt_version": args.prompt_version,
            "demonstration_split": "train" if demonstrations else None,
            "demonstration_count": len(demonstrations),
            "demonstration_ids": [
                row["example_id"] for row, _ in demonstrations
            ],
            "initial_anchor_cost_excluded": True,
            "invalid_action_fallback": "REJECT",
            "final_map_writeback_quality_available": False,
        },
        "model": args.model,
        "adapter": str(args.adapter.resolve()) if args.adapter is not None else None,
        "sample_count": len(inputs),
        "schema_valid_rate": valid_count / len(inputs),
        "executable_valid_rate": valid_count / len(inputs),
        "operation_metrics": metrics,
        "false_edit_rate": false_edit_rate,
        "missed_edit_rate": missed_edit_rate,
        "fallback_count": len(inputs) - valid_count,
        "prediction_counts": dict(sorted(Counter(predictions).items())),
        "results": results,
        "test_assets_read": False,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (args.output_dir / "rollouts.jsonl").open("w", encoding="utf-8") as handle:
        for row in rollout_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

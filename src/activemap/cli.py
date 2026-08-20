"""Command-line entry points for the first ActiveMap data gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from activemap.data.audit import audit_manifest as run_manifest_audit
from activemap.data.manifest import read_manifest, write_manifest
from activemap.data.sn7 import build_sn7_manifest
from activemap.data.splits import assign_group_splits, write_split_files
from activemap.validation import validate_jsonl as run_jsonl_validation

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
console = Console()


@app.command("index-sn7")
def index_sn7(
    raw_root: Annotated[Path, typer.Argument(exists=True, file_okay=False, resolve_path=True)],
    output: Annotated[Path, typer.Argument()],
    metadata: Annotated[
        bool, typer.Option(help="Read raster dimensions, CRS, and transform.")
    ] = True,
    strict: Annotated[bool, typer.Option(help="Fail on unparseable or unreadable assets.")] = True,
) -> None:
    """Index monthly SpaceNet 7 images, labels, and UDM masks."""
    frame = build_sn7_manifest(raw_root, read_raster_metadata=metadata, strict=strict)
    write_manifest(frame, output)
    message = f"Wrote {len(frame)} rows across {frame['aoi_id'].nunique()} AOIs to {output}"
    console.print(f"[green]{message}[/green]")


@app.command("make-splits")
def make_splits(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Argument()],
    seed: Annotated[int, typer.Option()] = 20260710,
    train_ratio: Annotated[float, typer.Option()] = 0.70,
    val_ratio: Annotated[float, typer.Option()] = 0.15,
    group_column: Annotated[str, typer.Option()] = "aoi_id",
    manifest_output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Assign complete AOIs to train, validation, and test splits."""
    frame = read_manifest(manifest)
    result, groups = assign_group_splits(
        frame,
        group_column=group_column,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    write_split_files(groups, output_dir, seed)
    default_output = manifest.with_name(f"{manifest.stem}_split{manifest.suffix}")
    output_manifest = manifest_output or default_output
    write_manifest(result, output_manifest)

    table = Table(title="AOI split summary")
    table.add_column("split")
    table.add_column("AOIs", justify="right")
    table.add_column("rows", justify="right")
    for split_name in ("train", "val", "test"):
        table.add_row(
            split_name,
            str(len(groups[split_name])),
            str(int((result["split"] == split_name).sum())),
        )
    console.print(table)
    console.print(f"[green]Wrote split manifest to {output_manifest}[/green]")


@app.command("validate-jsonl")
def validate_jsonl(
    jsonl_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    schema_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Validate every non-empty JSONL row against a JSON Schema."""
    valid_count, errors = run_jsonl_validation(jsonl_path, schema_path)
    if errors:
        for error in errors:
            console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]Validated {valid_count} records[/green]")


@app.command("audit-manifest")
def audit_manifest_command(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    check_paths: Annotated[bool, typer.Option()] = True,
) -> None:
    """Check duplicate keys, timestamp format, split leakage, and asset paths."""
    frame = read_manifest(manifest)
    issues = run_manifest_audit(frame, check_paths=check_paths)
    if issues:
        for issue in issues:
            console.print(f"[red]{issue}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]Manifest audit passed for {len(frame)} rows[/green]")


@app.command("audit-external-datasets")
def audit_external_datasets_command(
    registry: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Validate the license-aware external dataset registry."""
    from activemap.data.external_datasets import audit_external_dataset_registry

    report = audit_external_dataset_registry(registry)
    console.print_json(data=report)
    if not report["valid"]:
        raise typer.Exit(code=1)


@app.command("scaffold-external-dataset")
def scaffold_external_dataset_command(
    registry: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    dataset_id: Annotated[str, typer.Argument()],
    output_root: Annotated[Path, typer.Argument()],
) -> None:
    """Create a non-downloading, license-safe external dataset workspace."""
    from activemap.data.external_datasets import build_external_dataset_scaffold

    plan = build_external_dataset_scaffold(registry, dataset_id, output_root)
    console.print_json(data=plan)


@app.command("validate-external-predictions")
def validate_external_predictions_command(
    predictions: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Validate normalized third-party predictions before common evaluation."""
    from activemap.integrations.baselines.contracts import validate_prediction_jsonl

    count, errors = validate_prediction_jsonl(predictions)
    if errors:
        for error in errors:
            console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]Validated {count} external baseline predictions[/green]")


@app.command("validate-structured-map-index")
def validate_structured_map_index_command(
    index: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    allow_test: Annotated[bool, typer.Option()] = False,
    check_paths: Annotated[bool, typer.Option()] = False,
) -> None:
    """Validate an ArgoTweak/TbV-style neutral structured-map index."""
    from activemap.data.structured_map import validate_structured_map_jsonl

    count, errors = validate_structured_map_jsonl(
        index,
        allow_test=allow_test,
        check_paths=check_paths,
    )
    if errors:
        for error in errors:
            console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]Validated {count} structured map samples[/green]")


@app.command("convert-structured-map-scenes")
def convert_structured_map_scenes_command(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Argument()],
    allow_test: Annotated[bool, typer.Option()] = False,
    include_keep: Annotated[bool, typer.Option()] = False,
    check_paths: Annotated[bool, typer.Option()] = True,
) -> None:
    """Derive portable atomic edits from prior/target HD-map GeoJSON scenes."""
    from activemap.data.structured_map import convert_structured_map_scenes

    summary = convert_structured_map_scenes(
        manifest,
        output,
        allow_test=allow_test,
        include_keep=include_keep,
        check_paths=check_paths,
    )
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    console.print_json(data=summary)


@app.command("convert-av2-static-map")
def convert_av2_static_map_command(
    static_map: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Argument()],
) -> None:
    """Convert one official Argoverse 2 static-map JSON to stable GeoJSON."""
    from activemap.data.argoverse2 import convert_argoverse2_static_map

    summary = convert_argoverse2_static_map(static_map, output)
    console.print_json(data=summary)


@app.command("convert-argotweak-annotation")
def convert_argotweak_annotation_command(
    annotation: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Argument()],
) -> None:
    """Convert one ArgoTweak annotation into stale-prior/current-target maps."""
    from activemap.data.argotweak import convert_argotweak_annotation

    summary = convert_argotweak_annotation(annotation, output_dir)
    console.print_json(data=summary)


@app.command("build-argotweak-tbv-manifest")
def build_argotweak_tbv_manifest_command(
    splits: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    annotations: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Argument()],
) -> None:
    """List exact TbV camera, pose, and calibration dependencies for train/val."""
    from activemap.data.argotweak import build_argotweak_tbv_subset_manifest

    summary = build_argotweak_tbv_subset_manifest(splits, annotations, output)
    console.print_json(data=summary)


@app.command("build-argotweak-segment-scenes")
def build_argotweak_segment_scenes_command(
    segment_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    prior_map: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    target_map: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Argument()],
    split: Annotated[str, typer.Option()] = "val",
    stride: Annotated[int, typer.Option(min=1)] = 10,
) -> None:
    """Build synchronized seven-camera HD-map scenes for one TbV segment."""
    from activemap.data.argotweak import build_argotweak_segment_scenes

    summary = build_argotweak_segment_scenes(
        segment_root,
        prior_map,
        target_map,
        output,
        split=split,
        stride=stride,
    )
    console.print_json(data=summary)


@app.command("make-pilot")
def make_pilot_command(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Argument()],
    groups_per_split: Annotated[int, typer.Option(min=1)] = 1,
    min_observations: Annotated[int, typer.Option(min=1)] = 12,
    seed: Annotated[int, typer.Option()] = 20260710,
    require_udm: Annotated[bool, typer.Option()] = False,
) -> None:
    """Select complete, deterministic AOIs for a train/val/test data pilot."""
    from activemap.data.manifest import select_pilot_groups

    frame = read_manifest(manifest)
    pilot, selected = select_pilot_groups(
        frame,
        groups_per_split=groups_per_split,
        min_observations=min_observations,
        seed=seed,
        require_udm=require_udm,
    )
    write_manifest(pilot, output)
    metadata = {
        "source_manifest": str(manifest.resolve()),
        "output_manifest": str(output.resolve()),
        "seed": seed,
        "groups_per_split": groups_per_split,
        "min_observations": min_observations,
        "require_udm": require_udm,
        "selected": selected,
        "rows": len(pilot),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    console.print_json(data=metadata)


@app.command("build-updater-sn7")
def build_updater_sn7_command(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Argument()],
    image_size: Annotated[int, typer.Option(min=16)] = 128,
    context_pixels: Annotated[int, typer.Option(min=0)] = 32,
    id_column: Annotated[str | None, typer.Option()] = None,
    keep_iou_min: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.80,
    match_iou_min: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.20,
    max_centroid_distance: Annotated[float | None, typer.Option(min=0.0)] = None,
    min_area: Annotated[float, typer.Option(min=0.0)] = 0.0,
    max_per_operation: Annotated[int | None, typer.Option(min=1)] = None,
    max_invalid_fraction: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.50,
    max_month_gap: Annotated[int | None, typer.Option(min=1)] = None,
    min_change_persistence: Annotated[int, typer.Option(min=1)] = 1,
    temporal_pair_input: Annotated[
        bool,
        typer.Option(
            help="Store aligned RGB_(t-1) crops for a dedicated six-channel temporal updater."
        ),
    ] = False,
    seed: Annotated[int, typer.Option()] = 20260710,
) -> None:
    """Derive typed monthly edits and portable updater crops from SpaceNet 7."""
    from activemap.data.updater_crops import build_updater_crops

    summary = build_updater_crops(
        read_manifest(manifest),
        output_dir,
        image_size=image_size,
        context_pixels=context_pixels,
        id_column=id_column,
        keep_iou_min=keep_iou_min,
        fallback_match_iou_min=match_iou_min,
        max_centroid_distance=max_centroid_distance,
        min_area=min_area,
        max_events_per_operation=max_per_operation,
        max_invalid_fraction=max_invalid_fraction,
        max_month_gap=max_month_gap,
        min_change_persistence=min_change_persistence,
        sampling_seed=seed,
        include_prior_image=temporal_pair_input,
    )
    console.print_json(data=summary)


@app.command("scan-sn7-edits")
def scan_sn7_edits_command(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Argument()],
    aoi: Annotated[
        str | None,
        typer.Option(help="Comma-separated AOI IDs; scan all AOIs when omitted."),
    ] = None,
    id_column: Annotated[str | None, typer.Option()] = None,
    keep_iou_min: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.80,
    match_iou_min: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.20,
    max_centroid_distance: Annotated[float | None, typer.Option(min=0.0)] = None,
    min_area: Annotated[float, typer.Option(min=0.0)] = 0.0,
    max_per_operation: Annotated[int | None, typer.Option(min=1)] = None,
    max_month_gap: Annotated[int | None, typer.Option(min=1)] = None,
    min_change_persistence: Annotated[int, typer.Option(min=1)] = 1,
    seed: Annotated[int, typer.Option()] = 20260710,
) -> None:
    """Dry-run typed edit derivation and write pair-level counts."""
    from activemap.data.sn7_scan import scan_sn7_edits

    aoi_ids = {value.strip() for value in aoi.split(",") if value.strip()} if aoi else None
    summary = scan_sn7_edits(
        read_manifest(manifest),
        output,
        aoi_ids=aoi_ids,
        id_column=id_column,
        keep_iou_min=keep_iou_min,
        fallback_match_iou_min=match_iou_min,
        max_centroid_distance=max_centroid_distance,
        min_area=min_area,
        max_events_per_operation=max_per_operation,
        max_month_gap=max_month_gap,
        min_change_persistence=min_change_persistence,
        sampling_seed=seed,
    )
    console.print_json(data=summary)


@app.command("build-updater-muno21")
def build_updater_muno21_command(
    dataset_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Argument()],
    image_size: Annotated[int, typer.Option(min=64)] = 512,
    padding: Annotated[int, typer.Option(min=0)] = 128,
    max_source_crop_size: Annotated[int, typer.Option(min=256)] = 1024,
    road_width_pixels: Annotated[float, typer.Option(min=1.0)] = 6.0,
    max_source_pixels: Annotated[int, typer.Option(min=1)] = 250_000_000,
    seed: Annotated[int, typer.Option()] = 20260710,
) -> None:
    """Convert official MUNO21 graph scenarios into typed road-update samples."""
    from activemap.data.muno21 import build_muno21_updater

    summary = build_muno21_updater(
        dataset_root,
        output_dir,
        image_size=image_size,
        padding=padding,
        max_source_crop_size=max_source_crop_size,
        road_width_pixels=road_width_pixels,
        max_source_pixels=max_source_pixels,
        seed=seed,
    )
    console.print_json(data=summary)


@app.command("build-updater-inria")
def build_updater_inria_command(
    dataset_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Argument()],
    image_size: Annotated[int, typer.Option(min=64)] = 256,
    context_pixels: Annotated[int, typer.Option(min=0)] = 48,
    max_objects_per_tile: Annotated[int, typer.Option(min=1)] = 64,
    min_area_pixels: Annotated[int, typer.Option(min=1)] = 16,
    min_valid_fraction: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    validation_cities: Annotated[str, typer.Option()] = "vienna",
    test_cities: Annotated[str, typer.Option()] = "kitsap",
    seed: Annotated[int, typer.Option()] = 20260710,
) -> None:
    """Create high-resolution Inria building samples with controlled prior edits."""
    from activemap.data.inria import build_inria_updater

    summary = build_inria_updater(
        dataset_root,
        output_dir,
        image_size=image_size,
        context_pixels=context_pixels,
        max_objects_per_tile=max_objects_per_tile,
        min_area_pixels=min_area_pixels,
        min_valid_fraction=min_valid_fraction,
        validation_cities={value.strip() for value in validation_cities.split(",")},
        test_cities={value.strip() for value in test_cities.split(",")},
        seed=seed,
    )
    console.print_json(data=summary)


@app.command("build-episodes-muno21")
def build_episodes_muno21_command(
    updater_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Argument()],
    splits: Annotated[str, typer.Option()] = "train,val",
    road_width_source_pixels: Annotated[float, typer.Option(min=1.0)] = 6.0,
    frozen_test: Annotated[
        bool, typer.Option(help="Require active one-shot authorization for test-only output.")
    ] = False,
) -> None:
    """Build real multi-year MUNO21 evidence episodes without opening test."""
    from activemap.data.muno21 import build_muno21_evidence_episodes

    requested_splits = tuple(value.strip() for value in splits.split(",") if value.strip())
    summary = build_muno21_evidence_episodes(
        updater_manifest,
        output,
        splits=requested_splits,
        road_width_source_pixels=road_width_source_pixels,
        frozen_test=frozen_test,
    )
    console.print_json(data=summary)


@app.command("build-inria-segmentation")
def build_inria_segmentation_command(
    dataset_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Argument()],
    image_size: Annotated[int, typer.Option(min=64)] = 256,
    window_size: Annotated[int, typer.Option(min=64)] = 512,
    stride: Annotated[int, typer.Option(min=1)] = 512,
    min_valid_fraction: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    validation_cities: Annotated[str, typer.Option()] = "vienna",
    test_cities: Annotated[str, typer.Option()] = "kitsap",
    seed: Annotated[int, typer.Option()] = 20260721,
) -> None:
    """Create full-scene Inria semantic building segmentation crops."""
    from activemap.data.inria import build_inria_segmentation

    summary = build_inria_segmentation(
        dataset_root,
        output_dir,
        image_size=image_size,
        window_size=window_size,
        stride=stride,
        min_valid_fraction=min_valid_fraction,
        validation_cities={value.strip() for value in validation_cities.split(",")},
        test_cities={value.strip() for value in test_cities.split(",")},
        seed=seed,
    )
    console.print_json(data=summary)


@app.command("merge-updater-manifests")
def merge_updater_manifests_command(
    output: Annotated[Path, typer.Argument()],
    manifests: Annotated[
        list[Path],
        typer.Argument(exists=True, dir_okay=False, help="Two or more source manifests."),
    ],
) -> None:
    """Create one provenance-preserving manifest for staged or mixed training."""
    from activemap.data.merge_updaters import merge_updater_manifests

    summary = merge_updater_manifests(manifests, output)
    console.print_json(data=summary)


@app.command("build-episodes-sn7")
def build_episodes_sn7_command(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Argument()],
    id_column: Annotated[str | None, typer.Option()] = None,
    keep_iou_min: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.80,
    match_iou_min: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.20,
    max_centroid_distance: Annotated[float | None, typer.Option(min=0.0)] = None,
    min_area: Annotated[float, typer.Option(min=0.0)] = 0.0,
    max_per_operation: Annotated[int | None, typer.Option(min=1)] = None,
    max_month_gap: Annotated[int | None, typer.Option(min=1)] = None,
    min_change_persistence: Annotated[int, typer.Option(min=1)] = 1,
    seed: Annotated[int, typer.Option()] = 20260710,
    contexts: Annotated[str, typer.Option()] = "0,32,96",
    scales: Annotated[str, typer.Option()] = "1,2,4",
    frozen_test: Annotated[bool, typer.Option()] = False,
) -> None:
    """Create finite region x time x scale episodes from adjacent monthly maps."""
    from activemap.data.episode_builder import build_sn7_episodes

    context_values = tuple(float(value.strip()) for value in contexts.split(","))
    scale_values = tuple(int(value.strip()) for value in scales.split(","))
    frame = read_manifest(manifest)
    from activemap.frozen_test import authorize_manifest_test_access

    authorize_manifest_test_access(
        frame["split"].astype(str) if "split" in frame.columns else (), frozen_test
    )
    summary = build_sn7_episodes(
        frame,
        output,
        id_column=id_column,
        keep_iou_min=keep_iou_min,
        fallback_match_iou_min=match_iou_min,
        max_centroid_distance=max_centroid_distance,
        min_area=min_area,
        max_events_per_operation=max_per_operation,
        max_month_gap=max_month_gap,
        min_change_persistence=min_change_persistence,
        sampling_seed=seed,
        context_units=context_values,
        scales=scale_values,
    )
    console.print_json(data=summary)


@app.command("generate-selector-smoke")
def generate_selector_smoke(
    output: Annotated[Path, typer.Argument()],
    sample_count: Annotated[int, typer.Option(min=40)] = 512,
    candidate_count: Annotated[int, typer.Option(min=2)] = 8,
    seed: Annotated[int, typer.Option()] = 20260710,
) -> None:
    """Generate a deterministic end-to-end selector smoke benchmark."""
    from activemap.synthetic import generate_selector_smoke_samples, write_selector_samples

    samples = generate_selector_smoke_samples(
        sample_count=sample_count,
        candidate_count=candidate_count,
        seed=seed,
    )
    write_selector_samples(samples, output)
    console.print(f"[green]Wrote {len(samples)} selector samples to {output}[/green]")


@app.command("audit-episodes")
def audit_episodes_command(
    episodes: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Argument()],
    expected_derivation_version: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Audit episode identity, provenance paths, edit contracts, and AOI leakage."""
    from activemap.data.episode_audit import audit_episode_dataset, write_episode_audit

    summary = audit_episode_dataset(
        episodes,
        expected_derivation_version=expected_derivation_version,
    )
    write_episode_audit(summary, output)
    console.print_json(data=summary)
    if not summary["passed"]:
        raise typer.Exit(code=1)


@app.command("train-selector")
def train_selector_command(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Train one generic, edit-conditioned, or ablated evidence selector."""
    from activemap.training.selector import train_selector

    summary = train_selector(config, output_override=output)
    console.print_json(data=summary)


@app.command("train-operation-selector")
def train_operation_selector_command(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Train a typed-operation selector over frozen updater evidence."""
    from activemap.training.operation_selector import train_operation_selector

    summary = train_operation_selector(config, output_override=output)
    console.print_json(data=summary)


@app.command("evaluate-operation-selector")
def evaluate_operation_selector_command(
    checkpoint: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    feature_cache: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Argument()],
    split: Annotated[str, typer.Option()] = "val",
    device: Annotated[str, typer.Option()] = "auto",
    batch_size: Annotated[int, typer.Option(min=1)] = 64,
    num_workers: Annotated[int, typer.Option(min=0)] = 0,
    update_threshold: Annotated[float | None, typer.Option(min=0.0, max=1.0)] = None,
) -> None:
    """Evaluate an operation selector on cached train or validation evidence."""
    from activemap.evaluation.operation_selector import (
        evaluate_operation_selector_checkpoint,
    )

    summary = evaluate_operation_selector_checkpoint(
        checkpoint,
        feature_cache,
        output_dir,
        split=split,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        update_threshold=update_threshold,
    )
    console.print_json(data=summary)


@app.command("calibrate-operation-selector")
def calibrate_operation_selector_command(
    predictions: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Argument()],
    max_false_edit: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.05,
    grid_steps: Annotated[int, typer.Option(min=3, max=1001)] = 101,
) -> None:
    """Freeze a KEEP/update gate on operation-selector validation predictions."""
    from activemap.evaluation.operation_selector import calibrate_operation_predictions

    summary = calibrate_operation_predictions(
        predictions,
        output,
        max_false_edit=max_false_edit,
        grid_steps=grid_steps,
    )
    console.print_json(data=summary)


@app.command("export-updater-operation-baseline")
def export_updater_operation_baseline_command(
    feature_cache: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Argument()],
    split: Annotated[str, typer.Option()] = "val",
) -> None:
    """Export frozen updater edit probabilities under the selector protocol."""
    from activemap.evaluation.operation_selector import export_updater_operation_baseline

    summary = export_updater_operation_baseline(feature_cache, output_dir, split=split)
    console.print_json(data=summary)


@app.command("build-agent-data")
def build_agent_data_command(
    samples: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Argument()],
    split: Annotated[str, typer.Option()] = "train",
    selector_checkpoint: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    selector_checkpoints: Annotated[
        str | None,
        typer.Option(help="Comma-separated checkpoints for a score ensemble."),
    ] = None,
    device: Annotated[str, typer.Option()] = "cpu",
    top_k: Annotated[int, typer.Option(min=1)] = 8,
) -> None:
    """Build structured trajectories plus SFT and preference datasets for the agent."""
    from activemap.agent.trajectories import build_agent_datasets

    score_fn = None
    if selector_checkpoint is not None and selector_checkpoints is not None:
        raise typer.BadParameter(
            "use either --selector-checkpoint or --selector-checkpoints, not both"
        )
    if selector_checkpoints is not None:
        from activemap.inference import SelectorEnsemblePredictor

        paths = [Path(value.strip()) for value in selector_checkpoints.split(",") if value.strip()]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise typer.BadParameter(f"selector checkpoints do not exist: {missing}")
        score_fn = SelectorEnsemblePredictor(paths, device=device).action_scores
    elif selector_checkpoint is not None:
        from activemap.inference import SelectorPredictor

        score_fn = SelectorPredictor(selector_checkpoint, device=device).action_scores
    summary = build_agent_datasets(
        samples,
        output_dir,
        split=split,
        score_fn=score_fn,
        top_k=top_k,
    )
    console.print_json(data=summary)


@app.command("run-geo-tool")
def run_geo_tool_command(
    call_json: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_root: Annotated[Path, typer.Argument()],
    result_json: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Execute one validated geospatial tool call and write its auditable result."""
    from activemap.geo_tools import GeoToolCall, build_default_registry

    call = GeoToolCall.model_validate_json(call_json.read_text(encoding="utf-8"))
    result = build_default_registry(output_root).execute(call)
    if result_json is not None:
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    console.print_json(data=result.model_dump(mode="json"))
    if not result.success:
        raise typer.Exit(code=1)


@app.command("export-rsprompter-data")
def export_rsprompter_data_command(
    samples: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Argument()],
    include_delete_negatives: Annotated[bool, typer.Option()] = True,
    minimum_area: Annotated[float, typer.Option(min=0.0)] = 4.0,
    max_samples_per_split: Annotated[int | None, typer.Option(min=1)] = None,
    splits: Annotated[
        str, typer.Option(help="Comma-separated train,val,test splits")
    ] = "train,val",
) -> None:
    """Export ActiveMap updater crops as a COCO instance dataset for RSPrompter."""
    from activemap.integrations.rsprompter import export_rsprompter_dataset

    summary = export_rsprompter_dataset(
        samples,
        output_dir,
        include_delete_negatives=include_delete_negatives,
        minimum_area=minimum_area,
        max_samples_per_split=max_samples_per_split,
        splits=tuple(item.strip() for item in splits.split(",") if item.strip()),
    )
    console.print_json(data=summary)


@app.command("audit-rsprompter-data")
def audit_rsprompter_data_command(
    output_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    require_test_free: Annotated[bool, typer.Option()] = True,
) -> None:
    """Audit image integrity, COCO geometry, and split isolation."""
    from activemap.integrations.rsprompter import audit_rsprompter_dataset

    report = audit_rsprompter_dataset(
        output_dir,
        require_test_free=require_test_free,
    )
    console.print_json(data=report)


@app.command("train-rsprompter")
def train_rsprompter_command(
    repository: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    data_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    work_dir: Annotated[Path, typer.Argument()],
    python: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    resume: Annotated[bool, typer.Option()] = False,
    dry_run: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run isolated RSPrompter training without importing its dependencies."""
    from activemap.integrations.rsprompter import (
        rsprompter_train_command,
        run_rsprompter_training,
    )

    command = rsprompter_train_command(
        python=python,
        repository=repository,
        config=config,
        data_root=data_root,
        work_dir=work_dir,
        resume=resume,
    )
    if dry_run:
        console.print_json(data={"command": command, "cwd": str(repository)})
        return
    return_code = run_rsprompter_training(
        command,
        repository=repository,
        log_path=work_dir / "train.log",
    )
    if return_code:
        raise typer.Exit(code=return_code)


@app.command("rollout-agent")
def rollout_agent_command(
    samples_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Argument()],
    split: Annotated[str, typer.Option()] = "test",
    selector_checkpoint: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    device: Annotated[str, typer.Option()] = "cpu",
    top_k: Annotated[int, typer.Option(min=1)] = 8,
    max_acquisitions: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Run the closed-loop selector-updater-agent baseline and save full traces."""
    import numpy as np

    from activemap.agent.environment import MapMaintenanceEnv, rollout_agent_policy
    from activemap.agent.tools import (
        CounterfactualBeliefUpdater,
        GreedyAgentPolicy,
        StaticBeliefUpdater,
    )
    from activemap.agent.trajectories import load_selector_states

    score_fn: Any
    if selector_checkpoint is not None:
        from activemap.inference import SelectorPredictor

        score_fn = SelectorPredictor(selector_checkpoint, device=device).action_scores
    else:

        def score_fn(sample: Any) -> Any:
            return -np.asarray(sample.evidence_costs, dtype=np.float32)

    states = load_selector_states(samples_path, split=split)
    initial_states = [
        sample for sample in states if int(sample.metadata.get("oracle_step", 0)) == 0
    ]
    trajectories = []
    for sample in initial_states:
        budget = float(sample.metadata.get("budget", 1.0))
        belief_updater = (
            CounterfactualBeliefUpdater(sample)
            if isinstance(sample.metadata.get("evidence_predictions"), dict)
            else StaticBeliefUpdater(sample)
        )
        environment = MapMaintenanceEnv(
            sample,
            budget=budget,
            score_fn=score_fn,
            belief_updater=belief_updater,
            top_k=top_k,
        )
        trajectories.append(
            rollout_agent_policy(
                environment,
                GreedyAgentPolicy(),
                max_acquisitions=max_acquisitions,
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(trajectory.model_dump_json(exclude_none=True) + "\n")
    summary = {
        "trajectory_count": len(trajectories),
        "mean_reward": float(np.mean([item.total_reward for item in trajectories])),
        "mean_steps": float(np.mean([len(item.transitions) for item in trajectories])),
        "output": str(output.resolve()),
    }
    console.print_json(data=summary)


@app.command("build-selector-oracle-cache")
def build_selector_oracle_cache_command(
    episodes: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Argument()],
    image_size: Annotated[int, typer.Option(min=16)] = 512,
    image_channels: Annotated[int, typer.Option(min=1)] = 3,
    temporal_pair_input: Annotated[
        bool,
        typer.Option(help="Store RGB_(t-1)||RGB_t candidate tensors; requires --image-channels 6."),
    ] = False,
    asset_root_map: Annotated[
        str | None, typer.Option(help="Rewrite episode assets from absolute SOURCE=TARGET.")
    ] = None,
    splits: Annotated[str, typer.Option()] = "train,val",
    chunk_candidates: Annotated[int, typer.Option(min=1)] = 16,
    compression: Annotated[str, typer.Option(help="lzf, gzip, or none")] = "lzf",
    max_episodes: Annotated[int | None, typer.Option(min=1)] = None,
    frozen_test: Annotated[
        bool, typer.Option(help="Require active one-shot authorization for test-only output.")
    ] = False,
) -> None:
    """Materialize immutable candidate crops for cache-backed selector-oracle runs."""
    from activemap.oracle.updater_counterfactual import build_selector_oracle_input_cache

    asset_root_maps = ()
    if asset_root_map is not None:
        if "=" not in asset_root_map:
            raise typer.BadParameter("asset root map must use absolute SOURCE=TARGET")
        source_text, target_text = asset_root_map.split("=", 1)
        source, target = Path(source_text), Path(target_text)
        if not source.is_absolute() or not target.is_absolute():
            raise typer.BadParameter("asset root map paths must be absolute")
        asset_root_maps = ((source, target),)
    summary = build_selector_oracle_input_cache(
        episodes,
        output,
        image_size=image_size,
        image_channels=image_channels,
        temporal_pair_input=temporal_pair_input,
        asset_root_maps=asset_root_maps,
        splits=tuple(value.strip() for value in splits.split(",") if value.strip()),
        chunk_candidates=chunk_candidates,
        compression=compression,
        max_episodes=max_episodes,
        frozen_test=frozen_test,
    )
    console.print_json(data=summary)


@app.command("build-selector-oracle")
def build_selector_oracle_command(
    updater_checkpoint: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    episodes: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Argument()],
    device: Annotated[str, typer.Option()] = "auto",
    image_size: Annotated[int, typer.Option(min=16)] = 128,
    cost_weight: Annotated[float, typer.Option(min=0.0)] = 0.18,
    false_edit_weight: Annotated[float, typer.Option(min=0.0)] = 0.35,
    utility_mode: Annotated[str, typer.Option()] = "proxy",
    utility_profile: Annotated[str, typer.Option()] = "balanced",
    writeback_threshold: Annotated[float, typer.Option(min=0.01, max=0.99)] = 0.5,
    writeback_delta_margin: Annotated[float, typer.Option(min=0.0, max=0.49)] = 0.0,
    asset_root_map: Annotated[
        str | None, typer.Option(help="Rewrite episode assets from absolute SOURCE=TARGET.")
    ] = None,
    budgets: Annotated[str, typer.Option()] = "1,2,4,8",
    max_steps: Annotated[int | None, typer.Option(min=1)] = None,
    operation_selector_checkpoint: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False)
    ] = None,
    operation_update_threshold: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    initial_evidence_strategy: Annotated[
        str,
        typer.Option(
            help="updater_confidence or min_cost; use min_cost for matched updater audits."
        ),
    ] = "updater_confidence",
    prior_input_translation_pixels: Annotated[
        int,
        typer.Option(
            min=0,
            help="Translate only the rendered prior supplied to the updater.",
        ),
    ] = 0,
    prior_input_morphology: Annotated[
        str,
        typer.Option(help="Morph only the rendered prior: none, dilate, or erode."),
    ] = "none",
    prior_input_morphology_pixels: Annotated[
        int,
        typer.Option(min=0, help="Morphology radius in updater-input pixels."),
    ] = 0,
    corruption_seed: Annotated[int, typer.Option()] = 0,
    splits: Annotated[str, typer.Option()] = "train,val",
    max_episodes: Annotated[int | None, typer.Option(min=1)] = None,
    candidate_workers: Annotated[
        int,
        typer.Option(min=1, help="Parallel candidate reads per episode; preserves manifest order."),
    ] = 1,
    frozen_test: Annotated[
        bool, typer.Option(help="Require active one-shot authorization for test-only output.")
    ] = False,
    input_cache: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Read immutable candidate tensors from a matching HDF5 input cache.",
        ),
    ] = None,
) -> None:
    """Run updater counterfactuals over every episode to supervise the selector."""
    from activemap.oracle.updater_counterfactual import build_selector_oracle_samples

    asset_root_maps = ()
    if asset_root_map is not None:
        if "=" not in asset_root_map:
            raise typer.BadParameter("asset root map must use absolute SOURCE=TARGET")
        source_text, target_text = asset_root_map.split("=", 1)
        source, target = Path(source_text), Path(target_text)
        if not source.is_absolute() or not target.is_absolute():
            raise typer.BadParameter("asset root map paths must be absolute")
        asset_root_maps = ((source, target),)
    summary = build_selector_oracle_samples(
        updater_checkpoint,
        episodes,
        output,
        device=device,
        image_size=image_size,
        cost_weight=cost_weight,
        false_edit_weight=false_edit_weight,
        utility_mode=utility_mode,
        utility_profile=utility_profile,
        writeback_threshold=writeback_threshold,
        writeback_delta_margin=writeback_delta_margin,
        asset_root_maps=asset_root_maps,
        budgets=tuple(float(value.strip()) for value in budgets.split(",")),
        max_steps=max_steps,
        operation_selector_checkpoint=operation_selector_checkpoint,
        operation_update_threshold=operation_update_threshold,
        initial_evidence_strategy=initial_evidence_strategy,
        prior_input_translation_pixels=prior_input_translation_pixels,
        prior_input_morphology=prior_input_morphology,
        prior_input_morphology_pixels=prior_input_morphology_pixels,
        corruption_seed=corruption_seed,
        splits=tuple(value.strip() for value in splits.split(",") if value.strip()),
        frozen_test=frozen_test,
        input_cache=input_cache,
        max_episodes=max_episodes,
        candidate_workers=candidate_workers,
    )
    console.print_json(data=summary)


@app.command("generate-updater-smoke")
def generate_updater_smoke(
    output_dir: Annotated[Path, typer.Argument()],
    sample_count: Annotated[int, typer.Option(min=40)] = 160,
    image_size: Annotated[int, typer.Option(min=16)] = 32,
    seed: Annotated[int, typer.Option()] = 20260710,
) -> None:
    """Generate deterministic prior/image/target arrays for updater smoke tests."""
    from activemap.synthetic_updater import generate_updater_smoke_dataset

    manifest = generate_updater_smoke_dataset(
        output_dir,
        sample_count=sample_count,
        image_size=image_size,
        seed=seed,
    )
    console.print(f"[green]Wrote updater smoke dataset to {manifest}[/green]")


@app.command("render-updater-qc")
def render_updater_qc_command(
    samples: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Argument()],
    count: Annotated[int, typer.Option(min=1)] = 32,
    seed: Annotated[int, typer.Option()] = 20260710,
    sample_ids: Annotated[
        str | None,
        typer.Option(help="Comma-separated exact sample IDs; overrides random count."),
    ] = None,
    splits: Annotated[
        str,
        typer.Option(help="Comma-separated splits eligible for QC rendering."),
    ] = "train,val",
) -> None:
    """Render deterministic image/prior/target/invalid crop overlays."""
    from activemap.data.qc import render_updater_qc

    requested = (
        {value.strip() for value in sample_ids.split(",") if value.strip()} if sample_ids else None
    )
    requested_splits = {value.strip() for value in splits.split(",") if value.strip()}
    summary = render_updater_qc(
        samples,
        output_dir,
        count=count,
        seed=seed,
        sample_ids=requested,
        splits=requested_splits,
    )
    console.print_json(data=summary)


@app.command("audit-updater")
def audit_updater_command(
    samples: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Argument()],
    max_invalid_fraction: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.50,
    keep_iou_min: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.70,
    reshape_iou_max: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.95,
    max_reshape_centroid_distance: Annotated[float | None, typer.Option(min=0.0)] = 20.0,
    strict_warnings: Annotated[bool, typer.Option()] = False,
    allow_empty_keep: Annotated[bool, typer.Option()] = False,
    allow_nonlocal_polyline_reshape: Annotated[bool, typer.Option()] = False,
) -> None:
    """Audit crop shapes, ranges, edit semantics, UDM quality, and split leakage."""
    from activemap.data.updater_audit import audit_updater_dataset, write_updater_audit

    summary = audit_updater_dataset(
        samples,
        max_invalid_fraction=max_invalid_fraction,
        keep_iou_min=keep_iou_min,
        reshape_iou_max=reshape_iou_max,
        max_reshape_centroid_distance=max_reshape_centroid_distance,
        allow_empty_keep=allow_empty_keep,
        allow_nonlocal_polyline_reshape=allow_nonlocal_polyline_reshape,
    )
    write_updater_audit(summary, output)
    console.print_json(data=summary)
    if not summary["passed"] or (strict_warnings and summary["warning_count"]):
        raise typer.Exit(code=1)


@app.command("refresh-updater-masks")
def refresh_updater_masks_command(
    samples: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    report: Annotated[Path, typer.Argument()],
    write: Annotated[
        bool,
        typer.Option(help="Rewrite changed prior/target arrays; default is dry-run."),
    ] = False,
) -> None:
    """Refresh derived masks from stored geometry/crop-transform provenance."""
    from activemap.data.updater_mask_refresh import refresh_updater_masks

    summary = refresh_updater_masks(samples, report, write=write)
    console.print_json(data=summary)


@app.command("train-updater")
def train_updater_command(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Train the prior-conditioned segmentation and typed-edit updater."""
    from activemap.training.updater import train_updater

    summary = train_updater(config, output_override=output)
    console.print_json(data=summary)


@app.command("evaluate-updater")
def evaluate_updater_command(
    checkpoint: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    samples: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Argument()],
    split: Annotated[str, typer.Option()] = "test",
    device: Annotated[str, typer.Option()] = "auto",
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    num_workers: Annotated[int, typer.Option(min=0)] = 0,
    commit_threshold: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.0,
    bootstrap: Annotated[int, typer.Option(min=0)] = 1000,
    seed: Annotated[int, typer.Option()] = 20260710,
    presence_threshold: Annotated[float | None, typer.Option(min=0.0, max=1.0)] = None,
    change_threshold: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    add_threshold: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    remove_threshold: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    edit_decoding: Annotated[str, typer.Option()] = "auto",
    road_topology_tolerance: Annotated[int | None, typer.Option(min=0)] = None,
) -> None:
    """Run updater inference and emit typed-edit, IoU, calibration, and AOI CIs."""
    from activemap.evaluation.updater import evaluate_updater_checkpoint

    summary = evaluate_updater_checkpoint(
        checkpoint,
        samples,
        output_dir,
        split=split,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        commit_threshold=commit_threshold,
        bootstrap_iterations=bootstrap,
        bootstrap_seed=seed,
        presence_threshold=presence_threshold,
        change_threshold=change_threshold,
        add_threshold=add_threshold,
        remove_threshold=remove_threshold,
        edit_decoding=edit_decoding,
        road_topology_tolerance=road_topology_tolerance,
    )
    console.print_json(data=summary)


@app.command("calibrate-updater-temporal-change")
def calibrate_updater_temporal_change_command(
    checkpoint: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    samples: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Argument()],
    split: Annotated[str, typer.Option()] = "val",
    device: Annotated[str, typer.Option()] = "auto",
    batch_size: Annotated[int, typer.Option(min=1)] = 8,
    num_workers: Annotated[int, typer.Option(min=0)] = 0,
    max_stable_false_positive: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.005,
    grid_steps: Annotated[int, typer.Option(min=3, max=101)] = 37,
) -> None:
    """Freeze explicit ADD/REMOVE raster thresholds on validation data."""
    from activemap.evaluation.updater_temporal_calibration import (
        calibrate_updater_temporal_change,
    )

    summary = calibrate_updater_temporal_change(
        checkpoint,
        samples,
        output,
        split=split,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_stable_false_positive=max_stable_false_positive,
        grid_steps=grid_steps,
    )
    console.print_json(data=summary)


@app.command("calibrate-updater-hierarchy")
def calibrate_updater_hierarchy_command(
    checkpoint: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    samples: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Argument()],
    split: Annotated[str, typer.Option()] = "val",
    device: Annotated[str, typer.Option()] = "auto",
    batch_size: Annotated[int, typer.Option(min=1)] = 64,
    max_false_edit: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.05,
    grid_steps: Annotated[int, typer.Option(min=3, max=101)] = 33,
) -> None:
    """Select hierarchy thresholds on validation data under a safety constraint."""
    from activemap.evaluation.updater_calibration import calibrate_updater_hierarchy

    summary = calibrate_updater_hierarchy(
        checkpoint,
        samples,
        output,
        split=split,
        device=device,
        batch_size=batch_size,
        max_false_edit=max_false_edit,
        grid_steps=grid_steps,
    )
    console.print_json(data=summary)


@app.command("evaluate-updates")
def evaluate_updates_command(
    predictions: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Argument()],
    bootstrap: Annotated[int, typer.Option(min=0)] = 1000,
    seed: Annotated[int, typer.Option()] = 20260710,
) -> None:
    """Evaluate any method that emits the common update-prediction JSONL format."""
    from activemap.evaluation.statistics import grouped_bootstrap_intervals
    from activemap.evaluation.update import (
        evaluate_updates,
        load_update_predictions,
        save_update_evaluation,
    )

    records = load_update_predictions(predictions)
    summary = evaluate_updates(records)
    if bootstrap:
        summary["bootstrap"] = grouped_bootstrap_intervals(records, iterations=bootstrap, seed=seed)
    save_update_evaluation(summary, output)
    console.print(f"[green]Wrote update evaluation to {output}[/green]")


@app.command("compare-updates")
def compare_updates_command(
    baseline: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    challenger: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Argument()],
    metric: Annotated[str, typer.Option()] = "update_f1",
    bootstrap: Annotated[int, typer.Option(min=1)] = 1000,
    seed: Annotated[int, typer.Option()] = 20260710,
) -> None:
    """Compute an AOI-paired challenger-minus-baseline confidence interval."""
    from activemap.evaluation.statistics import paired_group_bootstrap_difference
    from activemap.evaluation.update import load_update_predictions

    result = paired_group_bootstrap_difference(
        load_update_predictions(baseline),
        load_update_predictions(challenger),
        metric=metric,
        iterations=bootstrap,
        seed=seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    console.print_json(data=result)


@app.command("calibrate-threshold")
def calibrate_threshold_command(
    validation_predictions: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Argument()],
    false_edit_weight: Annotated[float, typer.Option(min=0.0)] = 1.0,
) -> None:
    """Freeze a commit threshold on validation predictions before test evaluation."""
    from activemap.evaluation.update import load_update_predictions, select_commit_threshold

    result = select_commit_threshold(
        load_update_predictions(validation_predictions),
        false_edit_weight=false_edit_weight,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]Selected threshold {result['best_threshold']:.2f}[/green]")


@app.command("apply-updates")
def apply_updates_command(
    input_map: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    predictions: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_map: Annotated[Path, typer.Argument()],
    id_column: Annotated[str, typer.Option()] = "object_id",
    confidence_threshold: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    strict: Annotated[bool, typer.Option()] = False,
) -> None:
    """Apply committed typed predictions to GeoJSON/GPKG and write an audit log."""
    import geopandas as gpd

    from activemap.evaluation.update import load_update_predictions
    from activemap.models import EditOperation, EditRecord
    from activemap.vector_map import apply_edit

    current = gpd.read_file(input_map)
    audit: list[dict[str, Any]] = []
    for prediction in load_update_predictions(predictions):
        if not prediction.committed or prediction.confidence < confidence_threshold:
            audit.append(
                {"sample_id": prediction.sample_id, "status": "rejected", "reason": "confidence"}
            )
            continue
        if (
            prediction.predicted_edit in {EditOperation.ADD, EditOperation.RESHAPE}
            and prediction.predicted_geometry is None
        ):
            message = "vector geometry is required for ADD/RESHAPE"
            audit.append(
                {"sample_id": prediction.sample_id, "status": "rejected", "reason": message}
            )
            if strict:
                raise ValueError(f"{prediction.sample_id}: {message}")
            continue
        try:
            current = apply_edit(
                current,
                EditRecord(
                    op=prediction.predicted_edit,
                    object_id=prediction.object_id,
                    geometry=(
                        prediction.predicted_geometry
                        if prediction.predicted_edit in {EditOperation.ADD, EditOperation.RESHAPE}
                        else None
                    ),
                ),
                id_column=id_column,
            )
            audit.append({"sample_id": prediction.sample_id, "status": "applied"})
        except ValueError as exc:
            audit.append(
                {"sample_id": prediction.sample_id, "status": "rejected", "reason": str(exc)}
            )
            if strict:
                raise
    output_map.parent.mkdir(parents=True, exist_ok=True)
    current.to_file(output_map)
    audit_path = output_map.with_suffix(output_map.suffix + ".audit.jsonl")
    with audit_path.open("w", encoding="utf-8") as handle:
        for record in audit:
            handle.write(json.dumps(record) + "\n")
    applied = sum(record["status"] == "applied" for record in audit)
    console.print(f"[green]Applied {applied}/{len(audit)} edits to {output_map}[/green]")


@app.command("evaluate-selector")
def evaluate_selector_command(
    samples_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Argument()],
    split: Annotated[str, typer.Option()] = "test",
    budgets: Annotated[str, typer.Option()] = "1,2,4,8",
    checkpoint: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    seed: Annotated[int, typer.Option()] = 20260710,
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    """Evaluate all heuristic baselines and an optional learned checkpoint."""
    from activemap.evaluation.selector import (
        evaluate_score_policy,
        initial_states_for_budget,
        metrics_by_edit_type,
    )
    from activemap.policy.baselines import BaselineName, baseline_scores
    from activemap.training.data import load_selector_samples

    samples = load_selector_samples(samples_path, split=split)
    budget_values = [float(value.strip()) for value in budgets.split(",") if value.strip()]
    methods: dict[str, Any] = {
        baseline.value: (
            lambda sample, baseline=baseline: baseline_scores(sample, baseline, seed=seed)
        )
        for baseline in BaselineName
    }
    if checkpoint is not None:
        from activemap.inference import SelectorPredictor

        predictor = SelectorPredictor(checkpoint, device=device)
        methods["learned"] = predictor.action_scores

    rows: list[dict[str, Any]] = []
    details: dict[str, object] = {}
    for method, score_fn in methods.items():
        for budget in budget_values:
            budget_samples = initial_states_for_budget(samples, budget)
            metric = evaluate_score_policy(
                budget_samples,
                method=method,
                budget=budget,
                score_fn=score_fn,
            )
            rows.append(metric.as_dict())
            per_edit = metrics_by_edit_type(
                budget_samples,
                method=method,
                budget=budget,
                score_fn=score_fn,
            )
            details[f"{method}@{budget:g}"] = {
                key: value.as_dict() for key, value in per_edit.items()
            }
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "protocol": {
                    "split": split,
                    "budgets": budget_values,
                    "checkpoint": str(checkpoint) if checkpoint is not None else None,
                    "seed": seed,
                    "device": device,
                    "test_evaluation": split == "test",
                },
                "overall": rows,
                "by_edit_type": details,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    console.print(f"[green]Wrote selector evaluation to {output}[/green]")


@app.command("rollout-selector")
def rollout_selector_command(
    samples_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Argument()],
    split: Annotated[str, typer.Option()] = "test",
    budgets: Annotated[str, typer.Option()] = "1,2,4,8",
    checkpoint: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    max_steps: Annotated[int | None, typer.Option(min=1)] = None,
    seed: Annotated[int, typer.Option()] = 20260710,
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    """Execute sequential evidence policies and save every budget/action transition."""
    from activemap.evaluation.selector import initial_states_for_budget
    from activemap.policy.baselines import BaselineName, baseline_scores
    from activemap.policy.rollout import evaluate_rollouts
    from activemap.training.data import load_selector_samples

    samples = load_selector_samples(samples_path, split=split)
    budget_values = [float(value.strip()) for value in budgets.split(",") if value.strip()]
    methods: dict[str, Any] = {
        baseline.value: (
            lambda sample, baseline=baseline: baseline_scores(sample, baseline, seed=seed)
        )
        for baseline in BaselineName
    }
    if checkpoint is not None:
        from activemap.inference import SelectorPredictor

        predictor = SelectorPredictor(checkpoint, device=device)
        methods["learned"] = predictor.action_scores
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    with (output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for method, score_fn in methods.items():
            for budget in budget_values:
                budget_samples = initial_states_for_budget(samples, budget)
                summary, traces = evaluate_rollouts(
                    budget_samples,
                    method=method,
                    budget=budget,
                    score_fn=score_fn,
                    max_steps=max_steps,
                )
                summaries.append(summary)
                for trace in traces:
                    handle.write(
                        json.dumps({"method": method, "budget": budget, **trace.as_dict()}) + "\n"
                    )
    pd.DataFrame(summaries).to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "split": split,
                    "budgets": budget_values,
                    "checkpoint": str(checkpoint) if checkpoint is not None else None,
                    "seed": seed,
                    "device": device,
                    "max_steps": max_steps,
                    "test_evaluation": split == "test",
                },
                "results": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    console.print(f"[green]Wrote rollout evaluation to {output_dir}[/green]")


@app.command("export-schemas")
def export_schemas_command(
    output_dir: Annotated[Path, typer.Argument()] = Path("schemas"),
) -> None:
    """Regenerate every public JSON Schema from its canonical record model."""
    from activemap.schema_export import export_schemas

    written = export_schemas(output_dir)
    console.print(f"[green]Wrote {len(written)} schemas to {output_dir}[/green]")


@app.command("run-ablations")
def run_ablations_command(
    matrix: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Train, test, and aggregate every run registered in an ablation matrix."""
    from activemap.experiments.ablations import run_ablation_matrix

    summaries = run_ablation_matrix(matrix, output_root=output)
    console.print(f"[green]Completed {len(summaries)} ablation runs[/green]")


if __name__ == "__main__":
    app()

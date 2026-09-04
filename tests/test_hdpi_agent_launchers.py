from __future__ import annotations

from pathlib import Path

AGENT_LAUNCHERS = (
    "start_muno21_agent_sft.sh",
    "start_muno21_agent_sparse_tool_sft.sh",
    "start_muno21_agent_contextual_grpo.sh",
)
PAPER_TRAINING_LAUNCHERS = AGENT_LAUNCHERS + (
    "start_updater_training.sh",
    "start_muno21_evidence_selector.sh",
    "start_operation_selector_training.sh",
    "run_staged_updater_training.sh",
    "run_ablations.sh",
    "evaluate_promote_sparse_tool_sft.sh",
    "run_sparse_tool_sft_three_seeds.sh",
    "run_sparse_tool_sft_replicates_parallel.sh",
    "evaluate_agent_three_seeds.sh",
    "prepare_muno21_agent_pipeline.sh",
    "run_tool_belief_v4_pipeline.sh",
    "run_muno21_frozen_test_suite.sh",
)


def test_agent_launchers_use_hdpi_paths_gpu_policy_and_training_gate() -> None:
    for name in AGENT_LAUNCHERS:
        text = (Path("scripts") / name).read_text(encoding="utf-8")
        assert "/mnt/mydisk" not in text
        assert "server_hdpi_env.sh" in text
        assert "assert_training_ready.py" in text
        assert "assert_allowed_gpu.sh" in text
        assert "activemap_assert_allowed_gpu" in text
        assert "MUNO21_AGENT_GPU:-3" not in text


def test_single_gpu_policy_uses_server_specific_allowed_ids() -> None:
    policy = Path("scripts/assert_allowed_gpu.sh").read_text(encoding="utf-8")
    assert "ACTIVEMAP_GPU_IDS" in policy
    assert '",${allowed_ids}," == *",${gpu},"*' in policy
    for name in (
        "evaluate_promote_sparse_tool_sft.sh",
        "evaluate_agent_three_seeds.sh",
        "run_muno21_frozen_test_suite.sh",
    ):
        text = (Path("scripts") / name).read_text(encoding="utf-8")
        assert "assert_allowed_gpu.sh" in text
        assert "activemap_assert_allowed_gpu" in text


def test_paper_launchers_claim_the_shared_training_slot() -> None:
    for name in PAPER_TRAINING_LAUNCHERS:
        text = (Path("scripts") / name).read_text(encoding="utf-8")
        assert "acquire_training_slot.sh" in text


def test_v5_post_intake_tex_export_is_hash_bound_and_manuscript_isolated() -> None:
    text = Path("scripts/queue_sn7_v5_factorial_tex_export_hdpi.sh").read_text(
        encoding="utf-8"
    )
    assert "v5_matched_intake_audit.json" in text
    assert "three_seed_nonkeep_factorial_with_forced_summary.json" in text
    assert "audit.get(\"passed\") is not True" in text
    assert "aggregate.get(\"sha256\") != actual" in text
    assert "source_summary_sha256" in text
    assert "v5_factorial_tex_export_receipt.json" in text
    assert "iclr2027" not in text
    assert '"test_assets_read": False' in text


def test_training_slot_uses_nonblocking_inherited_file_lock() -> None:
    text = Path("scripts/acquire_training_slot.sh").read_text(encoding="utf-8")
    assert 'exec 7>"${slot_lock}"' in text
    assert "flock -n 7" in text
    assert "ACTIVEMAP_TRAINING_SLOT_HELD=1" in text


def test_sparse_tool_sft_default_matches_registry_artifact_directory() -> None:
    text = Path("scripts/start_muno21_agent_sparse_tool_sft.sh").read_text(
        encoding="utf-8"
    )
    assert "muno21_qwen3_4b_sparse_tool_sft_seed${SEED}" in text


def test_three_seed_sft_queue_promotes_validation_selected_adapters() -> None:
    queue = Path("scripts/run_sparse_tool_sft_three_seeds.sh").read_text(
        encoding="utf-8"
    )
    promotion = Path("scripts/evaluate_promote_sparse_tool_sft.sh").read_text(
        encoding="utf-8"
    )
    assert "20260821 20260822 20260823" in queue
    assert "promoted_adapter.json" in queue
    assert "select_sparse_tool_sft_checkpoint.py" in promotion
    assert "promote_sparse_tool_sft_adapter.py" in promotion
    assert "/mnt/mydisk" not in queue + promotion


def test_muno21_replicate_launcher_reuses_seed1_and_parallelizes_two_seeds() -> None:
    text = Path("scripts/run_sparse_tool_sft_replicates_parallel.sh").read_text(
        encoding="utf-8"
    )
    assert 'MUNO21_REPLICATE_SEEDS:-20260822 20260823' in text
    assert 'MUNO21_REPLICATE_GPUS:-0 1' in text
    assert "20260821" not in text
    assert "start_muno21_agent_sparse_tool_sft.sh" in text
    assert "evaluate_promote_sparse_tool_sft.sh" in text
    assert 'bash "${PROJECT_ROOT}/scripts/evaluate_promote_sparse_tool_sft.sh" &' in text
    assert "promoted_adapter.json" in text
    assert "ACTIVEMAP_SERVER_ENV" in text
    assert 'MUNO21_AGENT_RUN_FAMILY:-muno21_qwen3_4b_sparse_tool_sft' in text
    assert 'MUNO21_AGENT_START_SCRIPT:-start_muno21_agent_sparse_tool_sft.sh' in text
    assert '${RUN_FAMILY}_seed${seed}' in text
    assert 'scripts/${START_SCRIPT}' in text


def test_muno21_agent_execution_explicitly_imports_repository_scripts() -> None:
    for name in (
        "start_muno21_agent_sparse_tool_sft.sh",
        "evaluate_promote_sparse_tool_sft.sh",
        "evaluate_agent_three_seeds.sh",
    ):
        text = (Path("scripts") / name).read_text(encoding="utf-8")
        braced = "${PROJECT_ROOT}/src:${PROJECT_ROOT}"
        unbraced = "$PROJECT_ROOT/src:$PROJECT_ROOT"
        assert braced in text or unbraced in text


def test_muno21_hdpi_sync_inventories_agent_sft_inputs() -> None:
    sync = Path("tools/sync_missing_muno21_assets_to_hdpi.sh").read_text(
        encoding="utf-8"
    )
    inventory = Path("tools/inventory_canonical_data.sh").read_text(
        encoding="utf-8"
    )
    asset = "agent/agent_data_v9_natural_sparse_tools"
    assert asset in sync
    assert asset in inventory
    assert "train/sft_composed.jsonl" in inventory
    assert "val/sft_composed.jsonl" in inventory
    balanced = "agent/agent_data_v10_balanced_sparse_tools"
    assert balanced in sync
    assert balanced in inventory
    assert "balanced_prior_audit.json" in inventory


def test_paper_qc_is_rendered_into_clean_train_val_only_directories() -> None:
    renderer = Path("scripts/render_paper_dataset_qc.sh").read_text(encoding="utf-8")
    approval = Path("scripts/approve_dataset_qc.py").read_text(encoding="utf-8")
    assert 'QC_NAME="qc_train_val_v2"' in renderer
    assert "--splits train,val" in renderer
    assert '[[ ! -e "${output}" ]]' in renderer
    assert 'index["test_assets_rendered"] is False' in renderer
    assert 'QC_DIRECTORY_NAME = "qc_train_val_v2"' in approval


def test_three_seed_agent_rollouts_pair_model_and_selector_seeds() -> None:
    text = Path("scripts/evaluate_agent_three_seeds.sh").read_text(encoding="utf-8")
    assert "promoted_adapter.json" in text
    assert 'MUNO21_SELECTOR_SEEDS:-20260811 20260812 20260813' in text
    assert "muno21_evidence_conservative_v5_seed${selector_seed}/best.pt" in text
    assert "muno21_evidence_generic_v5_seed${selector_seed}/best.pt" in text
    assert "activemap-agent-selector-seed-pairing-v1" in text
    assert '"$SELECTOR_SEED_LIST" "$SPLIT"' in text
    assert "--seed \"$seed\"" in text
    assert 'MUNO21_MAX_TOOL_CALLS:-2' in text
    assert '--max-tool-calls "$MAX_TOOL_CALLS"' in text
    assert (
        'MUNO21_ASSET_ROOT_MAP:-/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}'
        in text
    )
    assert '--asset-root-map "$ASSET_ROOT_MAP"' in text
    assert "--do-sample" not in text


def test_legacy_single_seed_agent_watchers_are_disabled_by_default() -> None:
    for name in (
        "watch_evaluate_sparse_tool_agent.sh",
        "watch_start_muno21_agent_v9_after_v8.sh",
    ):
        text = (Path("scripts") / name).read_text(encoding="utf-8")
        assert "ACTIVEMAP_ALLOW_LEGACY_PIPELINE" in text
        assert "exit 64" in text


def test_sn7_three_seed_launcher_supports_bounded_hdpi_gpu_pool() -> None:
    launcher = Path("scripts/run_sn7_active_catalog_qwen.sh").read_text(
        encoding="utf-8"
    )
    hdpi = Path("scripts/server_hdpi_env.sh").read_text(encoding="utf-8")
    nts = Path("scripts/server_nts_env.sh").read_text(encoding="utf-8")
    assert 'GPU_POOL="${GPU_POOL:-${GPU_TRAIN},${GPU_SECOND}}"' in launcher
    assert "build_gpu_args" in launcher
    assert 'gpu_args+=(--gpu "${gpu}")' in launcher
    assert 'ACTIVEMAP_GPU_IDS:-0,1' in launcher
    assert 'ACTIVEMAP_GPU_IDS="${ACTIVEMAP_GPU_IDS:-0,1,2,3,4,5,6,7}"' in hdpi
    assert 'ACTIVEMAP_MAX_GPUS="${ACTIVEMAP_MAX_GPUS:-5}"' in hdpi
    assert 'ACTIVEMAP_MAX_GPUS="${ACTIVEMAP_MAX_GPUS:-2}"' in nts
    assert "/mnt/mydisk" not in launcher
    assert '${STORAGE_ROOT}/runs/updater/' in launcher


def test_sn7_seed1_watcher_only_advances_to_validation_evaluation() -> None:
    watcher = Path("scripts/watch_sn7_seed1_evaluation.sh").read_text(
        encoding="utf-8"
    )
    assert "server_hdpi_env.sh" in watcher
    assert "process_result.json" in watcher
    assert "adapter_config.json" in watcher
    assert "evaluate_seed1" in watcher
    assert "evaluate_unweighted_seed1" in watcher
    assert "three_seed" not in watcher
    assert "frozen" not in watcher.lower()
    assert '[[ ! -e "${evaluation}" ]]' in watcher
    assert 'SFT_RUN_NAME="${SFT_RUN_NAME:-qwen3vl4b_seed1_eval500}"' in watcher
    assert 'RUN_FAMILY="${RUN_FAMILY:-${SFT_RUN_NAME}}"' in watcher
    assert 'bash scripts/run_sn7_active_catalog_qwen.sh "${EVALUATION_STAGE}"' in watcher


def test_sampling_promotion_reuses_seed1_and_only_trains_two_replicates() -> None:
    launcher = Path("scripts/run_sn7_active_catalog_qwen.sh").read_text(
        encoding="utf-8"
    )
    watcher = Path("scripts/watch_active_catalog_sampling_promotion.sh").read_text(
        encoding="utf-8"
    )
    assert 'REPLICATE_SEEDS="${REPLICATE_SEEDS:-20260718,20260719}"' in watcher
    assert 'GPU_POOL="${GPU_POOL:-1,3}"' in watcher
    assert "promoted_replicates" in watcher
    assert "evaluate_promoted_replicates" in watcher
    assert "aggregate_promoted_three_seed" in watcher
    assert "paper_table_promoted" in watcher
    assert 'decision not in {"weighted", "unweighted"}' in launcher
    assert 'first_seed="${ACTIVE_CATALOG_PRIMARY_SEED:-20260717}"' in launcher
    assert "replicate seed list repeats primary seed" in launcher
    assert "active-catalog-promoted-seeds-v1" in launcher
    assert '"test_assets_read": False' in launcher


def test_step500_sampling_diagnostic_cannot_drive_promotion() -> None:
    watcher = Path("scripts/watch_active_catalog_step500_diagnostic.sh").read_text(
        encoding="utf-8"
    )
    launcher = Path("scripts/run_sn7_active_catalog_qwen.sh").read_text(
        encoding="utf-8"
    )
    comparator = Path("scripts/compare_active_catalog_sampling_ablation.py").read_text(
        encoding="utf-8"
    )
    assert 'CHECKPOINT_STEP="${CHECKPOINT_STEP:-500}"' in watcher
    assert 'WEIGHTED_GPU="${WEIGHTED_GPU:-4}"' in watcher
    assert 'UNWEIGHTED_GPU="${UNWEIGHTED_GPU:-5}"' in watcher
    assert '${weighted_seed}/checkpoints/checkpoint-${CHECKPOINT_STEP}' in watcher
    assert '${unweighted_seed}/checkpoints/checkpoint-${CHECKPOINT_STEP}' in watcher
    assert "--diagnostic-only" in watcher
    assert 'report.get("promotion_eligible") is not True' in launcher
    assert '"promotion_eligible": not args.diagnostic_only' in comparator


def test_muno21_replicates_wait_for_step500_gpu_release() -> None:
    watcher = Path(
        "scripts/watch_start_muno21_replicates_after_sn7_step500.sh"
    ).read_text(encoding="utf-8")
    assert "seed1_sampling_ablation_step500.json" in watcher
    assert 'report.get("diagnostic_only") is not True' in watcher
    assert 'report.get("promotion_eligible") is not False' in watcher
    assert 'MUNO21_REPLICATE_GPUS:-4 5' in watcher
    assert "run_sparse_tool_sft_replicates_parallel.sh" in watcher


def test_muno21_three_seed_rollout_watcher_waits_for_promotions() -> None:
    watcher = Path("scripts/watch_evaluate_muno21_agent_three_seeds.sh").read_text(
        encoding="utf-8"
    )
    assert "20260821 20260822 20260823" in watcher
    assert "promoted_adapter.json" in watcher
    assert 'MUNO21_AGENT_GPU:-4' in watcher
    assert 'MUNO21_MAX_TOOL_CALLS:-2' in watcher
    assert "evaluate_agent_three_seeds.sh" in watcher
    assert "aggregate_agent_three_seeds.py" in watcher
    assert "three_seed_bootstrap.json" in watcher
    assert "assess_agent_three_seed_promotion.py" in watcher
    assert "three_seed_promotion.json" in watcher


def test_sn7_closed_loop_waits_for_promoted_three_seed_sft() -> None:
    watcher = Path("scripts/watch_start_sn7_closed_loop_after_sft.sh").read_text(
        encoding="utf-8"
    )
    assert "qwen3vl4b_promoted_three_seed/promotion_manifest.json" in watcher
    assert 'SN7_CLOSED_LOOP_GPU:-6' in watcher
    assert "prepare_closed_loop_bundle" in watcher
    assert "closed_loop_baselines" in watcher
    assert "closed_loop_smoke" in watcher
    assert "closed_loop_seed1" in watcher
    assert "audit_episode_assets.py" in watcher
    assert "SN7_ASSET_ROOT_MAP" in watcher
    assert "compare_closed_loop_seed1" in watcher
    assert "closed_loop_writeback_qwen" in watcher
    assert "closed_loop_writeback_baseline" in watcher
    assert "assess_closed_loop_promotion" in watcher
    assert "render_closed_loop_examples" in watcher


def test_sn7_tool_belief_supports_paired_reliability_gate_ablation() -> None:
    launcher = Path("scripts/run_sn7_active_catalog_qwen.sh").read_text(
        encoding="utf-8"
    )
    assert 'TOOL_BELIEF_FAMILY="${TOOL_BELIEF_FAMILY:-joint_tool_belief}"' in launcher
    assert 'TOOL_GATE_FAMILY="${TOOL_GATE_FAMILY:-joint_tool_gate}"' in launcher
    assert (
        'TOOL_GATE_LABEL_BELIEF_FAMILY="${TOOL_GATE_LABEL_BELIEF_FAMILY:-joint_tool_belief}"'
        in launcher
    )
    assert "joint_tool_belief_gated_seed" in launcher
    assert "--reliability-gate --gate-bias -1.5" in launcher
    assert '"${tool_belief_args[@]}" "${reliability_args[@]}"' in launcher
    assert '${RUN_ROOT}/${TOOL_BELIEF_FAMILY}_${SEED_TAG}/best_promoted.pt' in launcher
    assert (
        '${RUN_ROOT}/${TOOL_GATE_LABEL_BELIEF_FAMILY}_${SEED_TAG}/best_promoted.pt'
        in launcher
    )
    assert '${RUN_ROOT}/${TOOL_GATE_FAMILY}_${SEED_TAG}/gate.joblib' in launcher
    assert 'tool_branch_suffix="_${TOOL_BELIEF_FAMILY}"' in launcher
    assert "compare_closed_loop_reliability_gate.py" in launcher
    assert "closed_loop_selective_tools_joint_tool_belief_gated_" in launcher
    assert "compare_tool_belief_reliability_gate_three_seed" in launcher
    assert "aggregate_reliability_gate_three_seed" in launcher
    assert "aggregate_reliability_gate_ablation.py" in launcher
    seed_runner = Path("scripts/run_sn7_active_catalog_tool_seed.sh").read_text(
        encoding="utf-8"
    )
    assert "joint_tool_belief_gated_seed" in seed_runner
    assert "TOOL_BELIEF_FAMILY=joint_tool_belief_gated" in seed_runner
    assert "compare_closed_loop_reliability_gate" in seed_runner
    assert "run_stage_if_missing" in seed_runner
    assert "run_stage_for_artifact_set" in seed_runner
    assert "refusing ambiguous resume" in seed_runner
    assert "closed_loop_reliability_gate_${SEED_TAG}.json" in seed_runner
    assert 'primary_seed="${ACTIVE_CATALOG_PRIMARY_SEED:-20260717}"' in seed_runner
    assert 'SEED_TAG="seed1"' in seed_runner
    assert "seed_tag_for" in launcher
    assert 'PRIMARY_SEED="${ACTIVE_CATALOG_PRIMARY_SEED:-${seed_values[0]}}"' in launcher
    assert launcher.count('tag="$(seed_tag_for "${seed}")"') >= 4


def test_sn7_reliability_watcher_runs_promoted_three_seed_pipeline() -> None:
    watcher = Path("scripts/watch_run_sn7_reliability_three_seed.sh").read_text(
        encoding="utf-8"
    )
    assert "qwen3vl4b_promoted_three_seed/promotion_manifest.json" in watcher
    assert 'SN7_RELIABILITY_GPU:-6' in watcher
    assert 'REPLICATE_SEEDS:-20260718,20260719' in watcher
    assert 'payload.get("test_assets_read") is False' in watcher
    assert 'decision in {"weighted", "unweighted"}' in watcher
    assert "run_sn7_active_catalog_tool_seed.sh" in watcher
    assert "compare_tool_belief_reliability_gate_three_seed" in watcher
    assert "aggregate_reliability_gate_three_seed" in watcher
    assert "aggregate_tool_writebacks_three_seed" in watcher
    assert '"${1:-}" == "--daemon"' in watcher


def test_step1000_sentinel_repairs_and_audits_image_assets() -> None:
    watcher = Path("scripts/watch_active_catalog_step1000_sentinel.sh").read_text(
        encoding="utf-8"
    )
    assert 'grep -q \'"image_assets"\'' in watcher
    assert "--repair-images" in watcher


def test_sn7_full_sft_uses_measured_validation_interval() -> None:
    launcher = Path("scripts/run_sn7_active_catalog_qwen.sh").read_text(
        encoding="utf-8"
    )
    assert 'SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-500}"' in launcher
    assert 'SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-500}"' in launcher
    assert 'SFT_RUN_NAME="${SFT_RUN_NAME:-qwen3vl4b_seed1_eval500}"' in launcher
    assert '--eval-steps "${SFT_EVAL_STEPS}"' in launcher
    assert '--save-steps "${SFT_SAVE_STEPS}"' in launcher
    trainer = Path("scripts/train_semantic_vlm_sft.py").read_text(encoding="utf-8")
    assert '"external_generated_action_gate"' in trainer
    assert '"load_best_model_at_end": args.teacher_loss_model_selection' in trainer
    assert "--acquire-loss-weight" in trainer


def test_muno_balanced_tool_branch_preserves_natural_validation() -> None:
    prepare = Path("scripts/prepare_muno21_agent_data_v10_balanced_tool.sh").read_text(
        encoding="utf-8"
    )
    assert 'MUNO21_USE_TOOL_REPEAT:-5' in prepare
    assert 'MUNO21_NO_TOOL_RATIO:-3.0' in prepare
    assert "--training --use-tool-repeat" in prepare
    assert "--no-tool-ratio" in prepare
    assert "--keep-all-tool-sequences" not in prepare
    assert '"validation_natural_prevalence"' in prepare
    launcher = Path("scripts/start_muno21_agent_sparse_tool_sft.sh").read_text(
        encoding="utf-8"
    )
    assert 'ACTIVEMAP_SERVER_ENV:-' in launcher
    assert 'MUNO21_AGENT_PROTOCOL:-natural-prior-sparse-grounded-tool-sft-v1' in launcher


def test_muno_v10_watcher_uses_isolated_run_family_and_positive_tool_budget() -> None:
    evaluator = Path("scripts/evaluate_agent_three_seeds.sh").read_text(
        encoding="utf-8"
    )
    watcher = Path("scripts/watch_evaluate_muno21_agent_v10.sh").read_text(
        encoding="utf-8"
    )
    assert 'MUNO21_AGENT_RUN_FAMILY:-muno21_qwen3_4b_sparse_tool_sft' in evaluator
    assert '${RUN_FAMILY}_seed${seed}' in evaluator
    assert "evaluate_promote_sparse_tool_sft.sh" in watcher
    assert "evaluate_agent_three_seeds.sh" in watcher
    assert "agent_data_v10_balanced_sparse_tools" in watcher
    assert "agent_v10_balanced_tool_val" in watcher
    assert 'MUNO21_MAX_TOOL_CALLS:-2' in watcher
    assert 'MUNO21_AGENT_SEEDS="${SEED}"' in watcher
    assert '"${1:-}" == "--daemon"' in watcher
    assert "v10_watcher.pid" in watcher
    assert 'while kill -0 "${train_pid}"' in watcher
    assert "audit_tool_call_reachability_gap.py" in watcher
    assert "--allow-single-seed-diagnostic" in watcher
    assert "assess_muno21_v10_diagnostic.py" in watcher
    assert "diagnostic_promotion.json" in watcher
    assert 'MUNO21_V10_REPLICATE_GPUS:-4 5' in watcher
    assert "run_sparse_tool_sft_replicates_parallel.sh" in watcher
    assert "agent_v10_balanced_tool_three_seed_val" in watcher
    assert "aggregate_agent_three_seeds.py" in watcher
    assert "assess_agent_three_seed_promotion.py" in watcher
    assert "test" not in watcher.lower()


def test_step1000_sentinel_is_balanced_diagnostic_and_respects_gpu_budget() -> None:
    watcher = Path("scripts/watch_active_catalog_step1000_sentinel.sh").read_text(
        encoding="utf-8"
    )
    assert "build_active_catalog_balanced_sentinel.py" in watcher
    assert "checkpoint-${CHECKPOINT_STEP}" in watcher
    assert "SENTINEL_GPU:-6" in watcher
    assert "activemap_muno21_replicates_after_sn7_step500" in watcher
    assert "--diagnostic-only" in watcher
    assert "test" not in watcher.lower()
    assert "diagnostic_adapters/checkpoint-${CHECKPOINT_STEP}" in watcher


def test_seed1_selection_uses_generated_actions_before_full_evaluation() -> None:
    watcher = Path("scripts/watch_select_evaluate_sn7_seed1.sh").read_text(
        encoding="utf-8"
    )
    assert "select_active_catalog_sentinel_checkpoint.py" in watcher
    assert "sentinel_checkpoint_selection/selection.json" in watcher
    assert "activemap_muno21_replicates_after_sn7_step500" in watcher
    assert 'SEED1_SELECTION_GPU:-6' in watcher
    assert 'SEED1_FULL_EVAL_SECOND_GPU:-7' in watcher
    assert "active_catalog_val_step1000_sentinel" not in watcher
    assert "seed1_sampling_ablation.json" in watcher
    assert "--diagnostic-only" not in watcher


def test_action_weighted_fallback_is_zero_call_and_completion_gated() -> None:
    watcher = Path("scripts/watch_start_sn7_action_weighted_fallback.sh").read_text(
        encoding="utf-8"
    )
    assert "seed1_sampling_ablation_step1000_sentinel.json" in watcher
    assert 'w == 0 and u == 0' in watcher
    assert 'get("status")=="completed"' in watcher
    assert 'ACTION_WEIGHTED_GPU:-1' in watcher
    assert "action_weighted_seed1" in watcher
    assert "action_weighted_smoke" in watcher
    launcher = Path("scripts/run_sn7_active_catalog_qwen.sh").read_text(
        encoding="utf-8"
    )
    assert "--acquire-loss-weight" in launcher
    assert "qwen3vl4b_action_weighted_seed1" in launcher


def test_paper_registry_tracks_promoted_sn7_and_selector_source_seeds() -> None:
    registry = Path("configs/experiments/paper_registry.yaml").read_text(
        encoding="utf-8"
    )
    assert "qwen3vl4b_promoted_three_seed/active_catalog_aoi_bootstrap.json" in registry
    assert "qwen3vl4b_three_seed/active_catalog_aoi_bootstrap.json" not in registry
    assert "source_training_seeds: [20260811, 20260812, 20260813]" in registry
    assert "analysis_seed_pairing: by_index_with_agent_training_seed" in registry
    assert "reliability_gate_three_seed/component_ablation.json" in registry
    assert "reliability_gate_three_seed/promotion.json" in registry
    assert "reliability_gate_ablation/three_seed_report.json" not in registry


def test_backbone_baseline_is_sampling_gated_and_validation_only() -> None:
    launcher = Path("scripts/run_active_catalog_backbone_baseline.sh").read_text(
        encoding="utf-8"
    )
    assert "server_hdpi_env.sh" in launcher
    assert "verify_sn7_active_catalog_gates.py token" in launcher
    assert "verify_sn7_active_catalog_gates.py smoke" in launcher
    assert "seed1_sampling_ablation.json" in launcher
    assert '[[ -f "${SAMPLING_REPORT}" ]] || return 0' in launcher
    assert 'require_file "${SAMPLING_REPORT}"' in launcher
    assert 'value in {"weighted","unweighted"}' in launcher
    assert "--acquire-sampling-target 0.25" in launcher
    assert "evaluate_active_catalog_selector.py" in launcher
    assert "val_evaluation_index.jsonl" in launcher
    assert "test.jsonl" not in launcher
    assert "frozen" not in launcher.lower()


def test_muno21_agent_pipeline_is_split_safe_and_hdpi_native() -> None:
    pipeline = Path("scripts/prepare_muno21_agent_pipeline.sh").read_text(
        encoding="utf-8"
    )
    composer = Path("scripts/prepare_muno21_agent_data_v9_natural.sh").read_text(
        encoding="utf-8"
    )
    assert "/mnt/mydisk" not in pipeline + composer
    assert "--splits train,val" in pipeline
    assert "episodes_train_val_v1.jsonl" in pipeline
    assert "selector_states_v1.jsonl" in pipeline
    assert "agent_rl_states_v2_grounded_tools" in pipeline
    assert "test" not in pipeline.split("case \"${STAGE}\"")[0]


def test_frozen_muno21_suite_requires_authorization_and_test_only_builders() -> None:
    text = Path("scripts/run_muno21_frozen_test_suite.sh").read_text(
        encoding="utf-8"
    )
    assert "assert_frozen_test_access" in text
    assert "--splits test --frozen-test" in text
    assert "--split test --frozen-test" in text
    assert "selector_states_test_v1.jsonl" in text
    assert "sparse_tool_labels_test_v1" in text
    assert "/mnt/mydisk" not in text


def test_v5_qualitative_export_waits_for_audited_validation_and_stays_supplement_only() -> None:
    text = Path("scripts/queue_sn7_v5_qualitative_export_hdpi.sh").read_text(
        encoding="utf-8"
    )
    assert "v5_matched_intake_audit.json" in text
    assert "three_seed_nonkeep_factorial_with_forced_summary.json" in text
    assert "aggregate hash mismatch" in text
    assert "export_sn7_v5_qualitative_evidence.py" in text
    assert "render_sn7_v5_matched_writeback.py" in text
    assert "--scope supplement" in text
    assert "--scope main" not in text
    assert '"test_assets_read": False' in text

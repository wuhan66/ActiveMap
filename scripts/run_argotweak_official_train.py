from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official ArgoTweak training compatibly.")
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--train-ann", type=Path, required=True)
    parser.add_argument("--val-ann", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--launcher", choices=("none", "pytorch"), default="none")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--autoscale-lr", action="store_true")
    parser.add_argument(
        "--freeze-prefix",
        action="append",
        default=[],
        help="parameter module prefix to freeze; repeat for multiple prefixes",
    )
    parser.add_argument(
        "--optimizer-lr",
        type=float,
        help="override the base AdamW learning rate before official auto-scaling",
    )
    parser.add_argument(
        "--auto-scale-base-batch-size",
        type=int,
        help="set the official auto-scale reference batch size explicitly",
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        help="validate every N epochs; defaults to the official config when omitted",
    )
    parser.add_argument(
        "--max-keep-checkpoints",
        type=int,
        help="retain this many checkpoints for train-only checkpoint selection",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.workers < 0:
        raise ValueError("epochs must be positive and workers must be non-negative")
    if args.optimizer_lr is not None and args.optimizer_lr <= 0:
        raise ValueError("optimizer learning rate must be positive")
    for value in (
        args.auto_scale_base_batch_size,
        args.evaluation_interval,
        args.max_keep_checkpoints,
    ):
        if value is not None and value < 1:
            raise ValueError("adapter intervals and checkpoint counts must be positive")

    official_root = args.official_root.resolve()
    sys.path.insert(0, str(official_root))
    os.chdir(official_root)

    import mmcv.utils.config as mmcv_config
    from yapf.yapflib.yapf_api import FormatCode as yapf_format_code

    def format_code_compat(text, style_config=None, verify=False):
        del verify
        return yapf_format_code(text, style_config=style_config)

    mmcv_config.FormatCode = format_code_compat
    from tools import train as official_train

    if args.freeze_prefix:
        original_train_model = official_train.train_model

        def train_model_with_frozen_modules(model, *model_args, **model_kwargs):
            frozen_parameters = []
            for name, parameter in model.named_parameters():
                if any(name == prefix or name.startswith(f"{prefix}.") for prefix in args.freeze_prefix):
                    parameter.requires_grad = False
                    frozen_parameters.append(name)
            trainable_count = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
            if not frozen_parameters:
                raise ValueError(f"freeze prefixes matched no parameters: {args.freeze_prefix}")
            if not trainable_count:
                raise ValueError("all model parameters were frozen")
            print(
                "[ActiveMap] adapter mode: "
                f"froze {len(frozen_parameters)} tensors under {args.freeze_prefix}; "
                f"trainable parameters={trainable_count}",
                flush=True,
            )
            return original_train_model(model, *model_args, **model_kwargs)

        official_train.train_model = train_model_with_frozen_modules

    cfg_options = [
        f"data.train.ann_file={args.train_ann.resolve()}",
        f"data.val.ann_file={args.val_ann.resolve()}",
        f"data.workers_per_gpu={args.workers}",
        f"runner.max_epochs={args.epochs}",
        f"total_epochs={args.epochs}",
        f"load_from={args.checkpoint.resolve()}",
    ]
    if args.optimizer_lr is not None:
        cfg_options.append(f"optimizer.lr={args.optimizer_lr}")
    if args.auto_scale_base_batch_size is not None:
        cfg_options.append(f"auto_scale_lr.base_batch_size={args.auto_scale_base_batch_size}")
    if args.evaluation_interval is not None:
        cfg_options.append(f"evaluation.interval={args.evaluation_interval}")
    if args.max_keep_checkpoints is not None:
        cfg_options.append(f"checkpoint_config.max_keep_ckpts={args.max_keep_checkpoints}")

    argv = [
        "train.py",
        "--config",
        str(args.config.resolve()),
        "--work-dir",
        str(args.work_dir.resolve()),
        "--seed",
        str(args.seed),
        "--deterministic",
        "--launcher",
        args.launcher,
        "--local_rank",
        str(args.local_rank),
        "--cfg-options",
        *cfg_options,
        "checkpoint_config.interval=1",
    ]
    if not args.autoscale_lr:
        argv.append("auto_scale_lr.enable=False")
    if args.no_validate:
        argv.append("--no-validate")
    if args.autoscale_lr:
        argv.append("--autoscale-lr")
    sys.argv = argv
    official_train.main()


if __name__ == "__main__":
    main()

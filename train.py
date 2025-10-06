from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from colorizer.config import DEFAULT_CONFIG, ExperimentConfig
from colorizer.engine import train as run_train
from colorizer.utils import set_global_seed


def _apply_overrides(
    cfg: ExperimentConfig, args: argparse.Namespace
) -> ExperimentConfig:
    # Apply only a few common overrides to keep the template minimal.
    train_cfg = replace(
        cfg.train,
        device=args.device or cfg.train.device,
        max_epochs=args.epochs or cfg.train.max_epochs,
        seed=args.seed if args.seed is not None else cfg.train.seed,
        compile=args.compile if args.compile else cfg.train.compile,
        checkpoint_interval=args.checkpoint_interval or cfg.train.checkpoint_interval,
        wandb_log=args.wandb_log if args.wandb_log else cfg.train.wandb_log,
        wandb_project=args.wandb_project or cfg.train.wandb_project,
        run_name=(args.run_name if args.run_name is not None else cfg.train.run_name),
        resume_path=(args.resume if args.resume is not None else cfg.train.resume_path),
    )
    data_cfg = replace(
        cfg.data,
        batch_size=args.batch_size or cfg.data.batch_size,
        val_batch_size=args.val_batch_size or cfg.data.val_batch_size,
        train_data_path=args.train_data_path or cfg.data.train_data_path,
        val_data_path=args.val_data_path or cfg.data.val_data_path,
        num_workers=(
            args.num_workers if args.num_workers is not None else cfg.data.num_workers
        ),
        prefetch_factor=(
            args.prefetch_factor
            if args.prefetch_factor is not None
            else cfg.data.prefetch_factor
        ),
    )
    model_cfg = replace(
        cfg.model,
        num_classes=(
            args.num_classes if args.num_classes is not None else cfg.model.num_classes
        ),
    )
    optim_cfg = replace(
        cfg.optim,
        optimizer=(args.optimizer or cfg.optim.optimizer),
        learning_rate=(
            args.learning_rate
            if args.learning_rate is not None
            else cfg.optim.learning_rate
        ),
    )

    new_cfg = replace(
        cfg,
        train=train_cfg,
        data=data_cfg,
        model=model_cfg,
        optim=optim_cfg,
        output_dir=args.output_dir or cfg.output_dir,
    )
    return new_cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training script (template)")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Compute device (default: auto)",
    )
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument(
        "--val-batch-size",
        type=int,
        help="Validation batch size (defaults to train batch size)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        help="Number of classes for classifier head (default: 365)",
    )
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for runs/checkpoints"
    )
    parser.add_argument("--seed", type=int, help="Global random seed")
    parser.add_argument("--compile", action="store_true", help="Compile the model")
    parser.add_argument("--train-data-path", type=str, help="Train data path")
    parser.add_argument("--val-data-path", type=str, help="Val data path")
    parser.add_argument(
        "--num-workers", type=int, help="Number of dataloader worker processes"
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        help="Number of batches to prefetch per worker",
    )
    parser.add_argument("--checkpoint-interval", type=int, help="Checkpoint interval")
    parser.add_argument("--wandb-log", action="store_true", help="Log to wandb")
    parser.add_argument("--wandb-project", type=str, help="Wandb project")
    parser.add_argument("--run-name", type=str, help="Optional run name")
    parser.add_argument(
        "--resume-from-checkpoint",
        dest="resume",
        type=str,
        help="Path to a checkpoint to resume from",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adadelta", "adam", "adamw", "sgd"],
        help="Optimizer to use (default: adam)",
    )
    parser.add_argument("--learning-rate", type=float, help="Optimizer learning rate")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = _apply_overrides(DEFAULT_CONFIG, args)

    # Validate data paths exist
    train_path = Path(config.data.train_data_path)
    val_path = Path(config.data.val_data_path)

    if not train_path.exists():
        print(
            f"Error: Training data path does not exist: {train_path}", file=sys.stderr
        )
        sys.exit(1)

    if not val_path.exists():
        print(
            f"Error: Validation data path does not exist: {val_path}", file=sys.stderr
        )
        sys.exit(1)

    set_global_seed(config.train.seed)

    run_train(config)


if __name__ == "__main__":
    main()

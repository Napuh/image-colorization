from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class DataConfig:
    """Configuration for dataset and dataloaders.

    Customize these fields for your own dataset.
    """

    train_data_path: str = "./data/places365_standard/train"
    val_data_path: str = "./data/places365_standard/val"
    batch_size: int = 128
    val_batch_size: int = 128
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    shuffle_train: bool = True
    drop_last: bool = True


@dataclass
class ModelConfig:
    """Configuration for the model."""

    num_classes: int = 365


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer and learning rate schedule."""

    optimizer: Literal["adadelta", "adam", "adamw", "sgd"] = "adam"
    learning_rate: float = 1e-4


@dataclass
class TrainingConfig:
    """Configuration for training/evaluation routines."""

    max_epochs: int = 10
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    compile: bool = False
    checkpoint_interval: int = 1
    eval_interval: int = 1
    seed: int = 42
    wandb_log: bool = False
    wandb_project: str = "colorizer"
    run_name: Optional[str] = None
    resume_path: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Top-level configuration aggregating sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "./runs"


DEFAULT_CONFIG = ExperimentConfig()

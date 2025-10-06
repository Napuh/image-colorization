import os
from datetime import datetime
from typing import Dict

import torch
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from colorizer.config import ExperimentConfig
from colorizer.data.dataset import MITPlacesDataset
from colorizer.models.colorizer_net import LTBCNetwork
from colorizer.utils.saving import load_checkpoint_into_training, unwrap_for_saving


class ColorizationLoss(nn.Module):
    """Combined loss for colorization and classification.

    Uses MSE loss (with sum reduction to match paper's Frobenius norm squared)
    for the colorization task and cross-entropy for the auxiliary classification.
    """

    def __init__(self, alpha: float = 1 / 300):
        """Initialize the loss.

        Args:
            alpha: Weight for the classification loss term (paper uses 1/300)
        """
        super().__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        image_output: torch.Tensor,
        classification_output: torch.Tensor,
        image_target: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            image_output: Predicted ab channels
            classification_output: Class logits
            image_target: Ground truth ab channels
            label: Ground truth class labels

        Returns:
            Combined loss value
        """
        mse_loss = self.mse_loss(image_output, image_target)
        ce_loss = self.ce_loss(classification_output, label)
        return mse_loss + (self.alpha * ce_loss)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: ExperimentConfig,
    output_dir: str,
    checkpoint_name: str,
) -> None:
    """Save checkpoint with both raw weights and full training state.

    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        config: Experiment configuration
        output_dir: Directory to save checkpoints
        checkpoint_name: Base name for the checkpoint (without extension)
    """
    raw_state = unwrap_for_saving(model).state_dict()

    # Save raw model weights for easy inference loading
    torch.save(raw_state, f"{output_dir}/{checkpoint_name}.pth")

    # Save full training state for resuming
    full_state = {
        "state_dict": raw_state,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "meta": {
            "num_classes": int(config.model.num_classes),
            "train_data_path": str(config.data.train_data_path),
            "val_data_path": str(config.data.val_data_path),
        },
    }
    torch.save(full_state, f"{output_dir}/{checkpoint_name}_full.pth")


def validate_one_epoch(
    config: ExperimentConfig,
    model: nn.Module,
    val_dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, float]:
    """Run a single validation epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    val_pbar = tqdm(val_dataloader, desc="Validating")
    with torch.inference_mode():
        for L, ab, label in val_pbar:
            L = L.to(device, non_blocking=True)
            ab = ab.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            image_output, classification_output = model(L)
            loss = criterion(image_output, classification_output, ab, label)

            # Scale validation loss to be comparable with training when using reduction="sum"
            # If train and val batch sizes differ, sum-reduced losses scale with batch size.
            # Bring val to the same scale as train: multiply by (train_batch_size / val_batch_size).
            train_bs = config.data.batch_size
            val_bs = config.data.val_batch_size if config.data.val_batch_size else 1
            scale = (train_bs / val_bs) if val_bs > 0 else 1.0
            scaled_loss_value = loss.item() * scale

            total_loss += scaled_loss_value
            num_batches += 1
            avg_loss_so_far = total_loss / num_batches
            val_pbar.set_postfix(
                {
                    "val_loss": f"{scaled_loss_value:.4f}",
                    "avg_val_loss": f"{avg_loss_so_far:.4f}",
                }
            )

    avg_loss = total_loss / max(num_batches, 1)
    return {"val_loss": avg_loss}


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        # torch.backends.mps may not exist on all builds; guard access
        mps_available = getattr(torch.backends, "mps", None)
        if (mps_available is not None) and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    return torch.device(device_arg)


def train(config: ExperimentConfig) -> None:
    """Top-level training loop.

    This function should:
      - build datasets/dataloaders
      - create model and optimizer
      - iterate epochs calling train_one_epoch/validate_one_epoch
      - log metrics and save checkpoints
    """
    device = _select_device(config.train.device)

    model = LTBCNetwork(num_classes=config.model.num_classes).to(device)

    if config.train.compile:
        print("Compiling model")
        model = torch.compile(model)

    # Build optimizer from config
    opt_name = config.optim.optimizer.lower()
    lr = config.optim.learning_rate
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif opt_name == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters())
    else:
        raise ValueError(f"Unsupported optimizer: {config.optim.optimizer}")

    # Use Frobenius norm squared (sum of squared errors) to match paper notation
    criterion = ColorizationLoss(alpha=1 / 300)

    train_dataset = MITPlacesDataset(config.data.train_data_path, is_train=True)
    val_dataset = MITPlacesDataset(config.data.val_data_path, is_train=False)

    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle_train,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        prefetch_factor=config.data.prefetch_factor,
        drop_last=config.data.drop_last,
    )
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=config.data.val_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        prefetch_factor=config.data.prefetch_factor,
        drop_last=config.data.drop_last,
    )

    print(
        f"Starting training for {config.train.max_epochs} epochs on {device.type}.\n"
        f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | "
        f"Batch size: {config.data.batch_size} | Batches/epoch: {len(train_dataloader)}"
    )

    training_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.train.run_name or training_timestamp
    output_dir = f"{config.output_dir}/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Resume from checkpoint
    start_epoch = 0

    if config.train.resume_path:
        start_epoch = load_checkpoint_into_training(
            ckpt_path=config.train.resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
            expected_num_classes=int(config.model.num_classes),
            expected_train_path=str(config.data.train_data_path),
            expected_val_path=str(config.data.val_data_path),
        )

    wandb_run = None
    if config.train.wandb_log:
        wandb_run = wandb.init(project=config.train.wandb_project, name=run_name)
        # Define custom x-axes so metrics plot against desired steps
        wandb.define_metric("global_step", hidden=True)
        wandb.define_metric("epoch", hidden=True)
        wandb.define_metric("train/batch_loss", step_metric="global_step")
        wandb.define_metric("train/avg_loss", step_metric="epoch")
        wandb.define_metric("val/avg_loss", step_metric="epoch")

    # Global step offset for logging: exact calculation based on completed epochs
    steps_per_epoch = len(train_dataloader)
    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, config.train.max_epochs):
        model.train()
        running_loss = 0.0
        num_batches = max(1, len(train_dataloader))

        batch_pbar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{config.train.max_epochs}"
        )

        for batch_idx, (L, ab, label) in enumerate(batch_pbar, start=1):
            L, ab, label = (
                L.to(device, non_blocking=True),
                ab.to(device, non_blocking=True),
                label.to(device, non_blocking=True),
            )

            optimizer.zero_grad()
            image_output, classification_output = model(L)
            loss = criterion(image_output, classification_output, ab, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            avg_loss_so_far = running_loss / batch_idx
            batch_pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "avg_loss": f"{avg_loss_so_far:.4f}"}
            )

            global_step += 1
            if wandb_run is not None:
                wandb.log(
                    {
                        "global_step": global_step,
                        "train/batch_loss": loss.item(),
                    }
                )

        epoch_avg_loss = running_loss / num_batches

        if wandb_run is not None:
            epoch_index = epoch + 1
            wandb.log(
                {
                    "epoch": epoch_index,
                    "train/avg_loss": epoch_avg_loss,
                }
            )

        if (epoch + 1) % config.train.eval_interval == 0:
            val_metrics = validate_one_epoch(
                config, model, val_dataloader, device, criterion
            )

            if wandb_run is not None:
                epoch_index = epoch + 1
                wandb.log(
                    {
                        "epoch": epoch_index,
                        "val/avg_loss": val_metrics["val_loss"],
                    }
                )

        if (epoch + 1) % config.train.checkpoint_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                config,
                output_dir,
                f"checkpoint_{epoch + 1}",
            )

    # Save final checkpoint
    save_checkpoint(
        model,
        optimizer,
        config.train.max_epochs,
        config,
        output_dir,
        "final_model",
    )

    if wandb_run is not None:
        wandb_run.finish()

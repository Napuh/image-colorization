from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def unwrap_for_saving(model: nn.Module) -> nn.Module:
    """Return the underlying nn.Module for saving a clean state_dict.

    Handles torch.compile models (exposing ._orig_mod) and DataParallel/DDP (.module).
    """
    to_save = model
    # Unwrap torch.compile wrapper
    if hasattr(to_save, "_orig_mod") and isinstance(
        getattr(to_save, "_orig_mod"), nn.Module
    ):
        to_save = getattr(to_save, "_orig_mod")
    # Unwrap DataParallel / DistributedDataParallel
    if hasattr(to_save, "module") and isinstance(getattr(to_save, "module"), nn.Module):
        to_save = getattr(to_save, "module")
    return to_save


def get_model_for_loading(model: nn.Module) -> nn.Module:
    """Return the underlying nn.Module for loading a clean state_dict.

    Mirrors unwrap_for_saving but for the loading path, ensuring we load
    weights into the real module when using torch.compile or DataParallel.
    """
    target = model
    if hasattr(target, "_orig_mod") and isinstance(
        getattr(target, "_orig_mod"), nn.Module
    ):
        target = getattr(target, "_orig_mod")
    if hasattr(target, "module") and isinstance(getattr(target, "module"), nn.Module):
        target = getattr(target, "module")
    return target


def load_checkpoint_into_training(
    ckpt_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    *,
    expected_num_classes: Optional[int] = None,
    expected_train_path: Optional[str] = None,
    expected_val_path: Optional[str] = None,
) -> int:
    """Load a full training checkpoint into `model` and `optimizer` and return start epoch.

    Enforces simple invariants if provided via the expected_* parameters.
    """
    print(f"[resume] Loading checkpoint from {ckpt_path}")
    loaded = torch.load(ckpt_path, map_location=device)

    if not (
        isinstance(loaded, dict)
        and "state_dict" in loaded
        and "optimizer" in loaded
        and "epoch" in loaded
    ):
        raise RuntimeError(
            "Expected a full checkpoint with keys: state_dict, optimizer, epoch"
        )

    state = loaded["state_dict"]
    missing, unexpected = get_model_for_loading(model).load_state_dict(
        state, strict=True
    )
    if missing or unexpected:
        print(
            f"[resume] load_state_dict mismatches -> missing: {len(missing)}, unexpected: {len(unexpected)}"
        )

    if optimizer is not None:
        optimizer.load_state_dict(loaded["optimizer"])  # type: ignore[arg-type]

    start_epoch = int(loaded.get("epoch", 0))

    meta = loaded.get("meta", {}) if isinstance(loaded, dict) else {}

    if expected_num_classes is not None:
        ckpt_nc = int(meta.get("num_classes", expected_num_classes))
        if ckpt_nc != int(expected_num_classes):
            raise RuntimeError(
                f"Checkpoint num_classes={ckpt_nc} does not match current config num_classes={expected_num_classes}"
            )

    if (expected_train_path is not None) or (expected_val_path is not None):
        train_path_ckpt = str(meta.get("train_data_path", expected_train_path))
        val_path_ckpt = str(meta.get("val_data_path", expected_val_path))
        if (
            expected_train_path is not None
            and train_path_ckpt != str(expected_train_path)
        ) or (
            expected_val_path is not None and val_path_ckpt != str(expected_val_path)
        ):
            raise RuntimeError(
                "Checkpoint dataset paths differ from current config. Refuse to resume to avoid data mismatch."
            )

    return start_epoch

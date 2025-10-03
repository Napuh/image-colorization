from __future__ import annotations

import argparse
from pathlib import Path

import kornia
import numpy as np
import torch
import torchvision.io as io
import torchvision.transforms.v2 as transforms
from PIL import Image
from torchvision.io import ImageReadMode

from colorizer.models.colorizer_net import LTBCNetwork


def normalize_lab(img_lab: torch.Tensor) -> torch.Tensor:
    """Normalize LAB image values to expected ranges (channels-first CxHxW).

    - L channel scaled to [0, 1]
    - a/b channels scaled to [0, 1] from [-128, 128]
    """
    img_lab[:1, ...] /= 100.0
    img_lab[1:, ...] = (img_lab[1:, ...] + 128.0) / 256.0
    return img_lab


def to_lab_tensor(img_tensor: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor (CxHxW uint8) to LAB tensor (CxHxW)"""
    img_tensor = img_tensor.float() / 255.0
    img_batched = img_tensor.unsqueeze(0)
    lab_batched = kornia.color.rgb_to_lab(img_batched)
    lab_tensor = lab_batched.squeeze(0)
    return lab_tensor


def _load_and_prepare_image(image_path: Path, target_size: int = 224) -> torch.Tensor:
    """Load image, resize, convert to LAB, normalize, and return L tensor.

    Uses the same pipeline as the training dataset for consistency.

    Returns:
        L_tensor: (1, 1, H, W) float32 in [0, 1]
    """
    # Load image as tensor (same as dataset.py)
    image = io.read_image(str(image_path), mode=ImageReadMode.RGB)  # CxHxW, uint8

    # Apply same transformations as dataset but deterministic for inference
    transform = transforms.Compose(
        [
            transforms.Resize(
                target_size
            ),  # Deterministic resize instead of RandomCrop
            transforms.CenterCrop(target_size),  # Ensure exact size
            to_lab_tensor,
            normalize_lab,
        ]
    )

    img_lab_tensor = transform(image)

    # Extract L channel (same as dataset.py)
    L = img_lab_tensor[:1]  # (1, H, W)

    # Add batch dimension for model input: (1, 1, 224, 224)
    L = L.unsqueeze(0)

    return L


def _postprocess_to_rgb(l_tensor: torch.Tensor, ab_pred: torch.Tensor) -> Image.Image:
    """Combine input L with predicted ab, denormalize to LAB, convert to RGB PIL.Image.

    Args:
        l_tensor: (1, 1, H, W) in [0,1]
        ab_pred: (1, 2, H, W) in [0,1] from model sigmoid
    """
    l_denorm = l_tensor.squeeze(0).cpu() * 100.0  # (1, H, W)
    ab_denorm = (ab_pred.squeeze(0).detach().cpu() * 256.0) - 128.0  # (2, H, W)

    # Create LAB image: (1, 3, H, W)
    lab_image = torch.cat([l_denorm, ab_denorm], dim=0).unsqueeze(0)

    rgb = kornia.color.lab_to_rgb(lab_image)
    rgb = rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
    rgb = (rgb * 255.0).astype(np.uint8)
    return Image.fromarray(rgb)


def colorize_image(
    weights_path: Path, image_path: Path, output_path: Path, device: torch.device
) -> None:
    model = LTBCNetwork().to(device)
    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    l_tensor = _load_and_prepare_image(image_path)
    l_tensor = l_tensor.to(device)

    with torch.inference_mode():
        ab_pred, _ = model(l_tensor)

    # Compose and save RGB
    out_img = _postprocess_to_rgb(l_tensor, ab_pred)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(str(output_path))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Colorize a grayscale image using a trained LTBC model."
    )
    p.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model weights (.pth) saved with state_dict.",
    )
    p.add_argument(
        "--image", type=str, required=True, help="Path to input image (jpg/png)."
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path. Defaults to <image>_colorized.png next to input.",
    )
    p.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Compute device (default: cpu)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    weights_path = Path(args.weights)
    image_path = Path(args.image)
    device = torch.device(args.device)
    if args.out is None:
        out_path = image_path.with_name(image_path.stem + "_colorized.png")
    else:
        out_path = Path(args.out)

    colorize_image(weights_path, image_path, out_path, device)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import kornia
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from tqdm import tqdm

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


def prepare_frame(frame: np.ndarray, target_size: int = 224) -> torch.Tensor:
    """Prepare a video frame (HxWxC numpy array) for colorization.

    Args:
        frame: RGB frame as numpy array (H, W, 3) uint8
        target_size: Target size for model input

    Returns:
        L_tensor: (1, 1, H, W) float32 in [0, 1]
    """
    # Convert numpy array to tensor (C, H, W)
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)

    # Apply same transformations as dataset
    transform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            to_lab_tensor,
            normalize_lab,
        ]
    )

    img_lab_tensor = transform(frame_tensor)

    # Extract L channel
    L = img_lab_tensor[:1]  # (1, H, W)

    # Add batch dimension for model input
    L = L.unsqueeze(0)  # (1, 1, H, W)

    return L


def postprocess_to_rgb(l_tensor: torch.Tensor, ab_pred: torch.Tensor) -> np.ndarray:
    """Combine input L with predicted ab, denormalize to LAB, convert to RGB numpy array.

    Args:
        l_tensor: (1, 1, H, W) in [0,1]
        ab_pred: (1, 2, H, W) in [0,1] from model sigmoid

    Returns:
        RGB numpy array (H, W, 3) uint8
    """
    l_denorm = l_tensor.squeeze(0).cpu() * 100.0  # (1, H, W)
    ab_denorm = (ab_pred.squeeze(0).detach().cpu() * 256.0) - 128.0  # (2, H, W)

    # Create LAB image: (1, 3, H, W)
    lab_image = torch.cat([l_denorm, ab_denorm], dim=0).unsqueeze(0)

    rgb = kornia.color.lab_to_rgb(lab_image)
    rgb = rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
    rgb = (rgb * 255.0).astype(np.uint8)
    return rgb


def extract_frames(video_path: Path, output_dir: Path) -> tuple[int, float]:
    """Extract frames from video using ffmpeg.

    Returns:
        Tuple of (total_frames, fps)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video info
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets,r_frame_rate",
        "-of",
        "csv=p=0",
        str(video_path),
    ]

    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    fps_str, frame_count = result.stdout.strip().split(",")

    # Parse fps (format: "num/denom")
    num, denom = map(int, fps_str.split("/"))
    fps = num / denom
    total_frames = int(frame_count)

    # Extract frames
    extract_cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-q:v",
        "1",  # Highest quality
        str(output_dir / "frame_%06d.png"),
    ]

    subprocess.run(extract_cmd, check=True, capture_output=True)

    return total_frames, fps


def colorize_frames(
    frames_dir: Path,
    output_dir: Path,
    model: LTBCNetwork,
    device: torch.device,
    total_frames: int,
) -> None:
    """Colorize all frames in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(frames_dir.glob("frame_*.png"))

    with torch.inference_mode():
        for frame_file in tqdm(
            frame_files, desc="Colorizing frames", total=total_frames
        ):
            # Load frame as RGB
            frame = Image.open(frame_file).convert("RGB")
            frame_np = np.array(frame)

            # Prepare for model
            l_tensor = prepare_frame(frame_np)
            l_tensor = l_tensor.to(device)

            # Colorize
            ab_pred, _ = model(l_tensor)

            # Convert back to RGB
            rgb_frame = postprocess_to_rgb(l_tensor, ab_pred)

            # Save colorized frame
            output_file = output_dir / frame_file.name
            Image.fromarray(rgb_frame).save(output_file)


def reconstruct_video(
    colorized_dir: Path, output_path: Path, fps: float, audio_path: Path | None = None
) -> None:
    """Reconstruct video from colorized frames using ffmpeg."""
    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-framerate",
        str(fps),
        "-i",
        str(colorized_dir / "frame_%06d.png"),
    ]

    # Add audio if available
    if audio_path is not None and audio_path.exists():
        cmd.extend(
            [
                "-i",
                str(audio_path),
                "-map",
                "0:v",  # Map video from first input (frames)
                "-map",
                "1:a",  # Map audio from second input (audio file)
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-b:a",
                "192k",  # Audio bitrate
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",  # High quality
                "-shortest",  # Stop at shortest stream
            ]
        )
    else:
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",  # High quality
            ]
        )

    cmd.append(str(output_path))

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise


def extract_audio(video_path: Path, output_path: Path) -> bool:
    """Extract audio from video. Returns True if audio exists."""
    try:
        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vn",  # No video
            "-acodec",
            "copy",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def colorize_video(
    weights_path: Path,
    video_path: Path,
    output_path: Path,
    device: torch.device,
    keep_frames: bool = False,
) -> None:
    """Colorize a video file using the trained model.

    Args:
        weights_path: Path to model weights
        video_path: Path to input video
        output_path: Path for output video
        device: Torch device to use
        keep_frames: If True, keep extracted and colorized frames
    """
    print(f"Loading model from {weights_path}...")
    model = LTBCNetwork().to(device)
    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        frames_dir = temp_path / "frames"
        colorized_dir = temp_path / "colorized"
        audio_path = temp_path / "audio.aac"

        # Extract frames
        print("Extracting frames from video...")
        total_frames, fps = extract_frames(video_path, frames_dir)
        print(f"Extracted {total_frames} frames at {fps:.2f} fps")

        # Extract audio
        print("Extracting audio...")
        has_audio = extract_audio(video_path, audio_path)
        if has_audio:
            print("Audio extracted successfully")
        else:
            print("No audio track found")
            audio_path = None

        # Colorize frames
        print("Colorizing frames...")
        colorize_frames(frames_dir, colorized_dir, model, device, total_frames)

        # Reconstruct video
        print("Reconstructing video...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reconstruct_video(colorized_dir, output_path, fps, audio_path)

        # Optionally save frames
        if keep_frames:
            frames_output = output_path.parent / f"{output_path.stem}_frames"
            colorized_output = output_path.parent / f"{output_path.stem}_colorized"

            import shutil

            shutil.copytree(frames_dir, frames_output)
            shutil.copytree(colorized_dir, colorized_output)
            print(f"Frames saved to {frames_output} and {colorized_output}")

    print(f"Colorized video saved to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Colorize a grayscale video using a trained LTBC model."
    )
    p.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model weights (.pth) saved with state_dict.",
    )
    p.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video (mp4, avi, mov, etc.).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output video path. Defaults to <video>_colorized.mp4 next to input.",
    )
    p.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Compute device (default: cpu)",
    )
    p.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep extracted and colorized frames after processing",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    weights_path = Path(args.weights)
    video_path = Path(args.video)
    device = torch.device(args.device)

    if args.out is None:
        out_path = video_path.with_name(video_path.stem + "_colorized.mp4")
    else:
        out_path = Path(args.out)

    colorize_video(weights_path, video_path, out_path, device, args.keep_frames)


if __name__ == "__main__":
    main()

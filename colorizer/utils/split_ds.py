import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


def is_image_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    return suffix in VALID_IMAGE_EXTENSIONS


def list_classes(split_dir: Path) -> List[str]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    classes = [p.name for p in split_dir.iterdir() if p.is_dir()]
    classes.sort()
    return classes


def list_images(class_dir: Path) -> List[Path]:
    if not class_dir.exists():
        return []
    files = [p for p in class_dir.iterdir() if p.is_file() and is_image_file(p)]
    files.sort(key=lambda p: p.name)
    return files


def safe_make_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_if_exists(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def materialize_copy(src: Path, dst: Path) -> None:
    """Copy `src` to `dst` if `dst` does not already exist."""
    if dst.exists() or dst.is_symlink():
        return
    shutil.copy2(src, dst)


def build_subset_for_split(
    src_split_dir: Path,
    dst_split_dir: Path,
    selected_classes: Iterable[str],
    per_class_limit: Optional[int],
) -> Tuple[int, int]:
    """Build a subset for a single split (train/val).

    Returns (num_classes, num_images)
    """
    safe_make_dir(dst_split_dir)

    classes_count = 0
    images_count = 0

    tasks: List[Tuple[Path, Path]] = []

    for class_name in selected_classes:
        src_class_dir = src_split_dir / class_name
        if not src_class_dir.exists():
            # Some classes may be missing in val; skip gracefully
            continue

        dst_class_dir = dst_split_dir / class_name
        safe_make_dir(dst_class_dir)

        images = list_images(src_class_dir)
        if per_class_limit is not None:
            images = images[:per_class_limit]

        for src_img in images:
            dst_img = dst_class_dir / src_img.name
            tasks.append((src_img, dst_img))

        if images:
            classes_count += 1
            images_count += len(images)

    def _do_task(pair: Tuple[Path, Path]) -> None:
        s, d = pair
        materialize_copy(s, d)

    if tasks:
        for t in tasks:
            _do_task(t)

    return classes_count, images_count


def build_places10(
    src_root: Path,
    dst_root: Path,
) -> None:
    src_train = src_root / "train"
    all_classes = list_classes(src_train)
    selected = all_classes[:10]

    for split in ("train", "val"):
        src_split = src_root / split
        dst_split = dst_root / "places10" / split
        c, n = build_subset_for_split(src_split, dst_split, selected, None)
        print(f"places10/{split}: classes={c}, images={n}")


def build_places10_small(
    src_root: Path,
    dst_root: Path,
    per_class_limit: int,
) -> None:
    src_train = src_root / "train"
    all_classes = list_classes(src_train)
    selected = all_classes[:10]

    for split in ("train", "val"):
        src_split = src_root / split
        dst_split = dst_root / "places10_small" / split
        c, n = build_subset_for_split(src_split, dst_split, selected, per_class_limit)
        print(f"places10_small/{split}: classes={c}, images={n}")


def build_places365_small(
    src_root: Path,
    dst_root: Path,
    per_class_limit: int,
) -> None:
    src_train = src_root / "train"
    all_classes = list_classes(src_train)

    for split in ("train", "val"):
        src_split = src_root / split
        dst_split = dst_root / "places365_small" / split
        c, n = build_subset_for_split(
            src_split, dst_split, all_classes, per_class_limit
        )
        print(f"places365_small/{split}: classes={c}, images={n}")


def main() -> int:
    src_root, dst_root = Path("data/places365_standard"), Path("data")
    limit = 1000

    for required in (src_root / "train", src_root / "val"):
        if not required.exists():
            print(f"ERROR: Source split missing: {required}", file=sys.stderr)
            return 2

    targets = [
        dst_root / "places10",
        dst_root / "places10_small",
        dst_root / "places365_small",
    ]

    for t in targets:
        if t.exists() or t.is_symlink():
            print(f"Removing existing dataset: {t}")
            remove_if_exists(t)

    print(f"Building subsets from {src_root}")

    build_places10(src_root, dst_root)
    build_places10_small(src_root, dst_root, limit)
    build_places365_small(src_root, dst_root, limit)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

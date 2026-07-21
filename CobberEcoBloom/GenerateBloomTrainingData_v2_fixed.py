#!/usr/bin/env python3
"""
Generate one master CobberEcoBloom dataset for three nested training sizes.

The default dataset contains:
    - 16,000 training images
    - 1,000 validation images
    - 1,000 test images

Nested training subsets are built from the same master training pool:
    - 2,000 training images
    - 8,000 training images
    - 16,000 training images

All subsets are balanced across the five classes. The 2,000-image subset is
contained inside the 8,000-image subset, which is contained inside the
16,000-image subset. Validation and test images are identical for all models.

Coverage ranges touch so the classification task contains meaningful boundary
cases:
    clear_water   0.00 <= coverage < 0.10
    mostly_clear  0.10 <= coverage < 0.25
    field_check   0.25 <= coverage < 0.75
    mostly_bloom  0.75 <= coverage < 0.90
    dense_bloom   0.90 <= coverage <= 1.00

Typical run:
    python GenerateBloomTrainingData_v2.py

Faster/smaller test run:
    python GenerateBloomTrainingData_v2.py --outdir BloomData_v2_test \
        --train-total 100 --val-total 50 --test-total 50 \
        --subset-sizes 50 100 --workers 2

Dependencies:
    pip install numpy opencv-python
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


DISPLAY_CLASS_ORDER = [
    "clear_water",
    "mostly_clear",
    "field_check",
    "mostly_bloom",
    "dense_bloom",
]

# Preserve the output-index order used by the existing no-pickle app.
MODEL_CLASS_ORDER = [
    "clear_water",
    "dense_bloom",
    "field_check",
    "mostly_bloom",
    "mostly_clear",
]

LABEL_TO_ID = {label: i for i, label in enumerate(MODEL_CLASS_ORDER)}

CATEGORIES: Dict[str, Tuple[float, float]] = {
    "clear_water": (0.00, 0.10),
    "mostly_clear": (0.10, 0.25),
    "field_check": (0.25, 0.75),
    "mostly_bloom": (0.75, 0.90),
    "dense_bloom": (0.90, 1.00),
}


@dataclass(frozen=True)
class GenerationTask:
    split: str
    label: str
    split_index: int
    train_rank: int
    sampling_mode: str
    target_boundary: float
    requested_coverage: float
    base_seed: int
    image_size: int
    edge_band_fraction: float
    outpath: str


def make_water_background(size: int, rng: np.random.Generator) -> np.ndarray:
    """Create the same synthetic RGB water background used by CobberEcoBloom."""
    y = np.linspace(0, 1, size)[:, None]
    x = np.linspace(0, 1, size)[None, :]

    base_r = 22 + 8 * y + 4 * np.sin(2 * np.pi * (x * 2.2 + y * 0.4))
    base_g = 92 + 20 * y + 8 * np.sin(2 * np.pi * (x * 1.4 - y * 0.7))
    base_b = 145 + 35 * (1 - y) + 8 * np.sin(2 * np.pi * (x * 1.8 + y * 1.1))

    img = np.dstack([base_r, base_g, base_b]).astype(np.float32)
    img += rng.normal(0, 5, (size, size, 3))

    for _ in range(int(rng.integers(8, 18))):
        yy = int(rng.integers(0, size))
        x0 = int(rng.integers(0, size // 2))
        x1 = int(rng.integers(size // 2, size))
        color = (35, 120, 170)
        thickness = int(rng.integers(1, 3))
        cv2.line(img, (x0, yy), (x1, yy + int(rng.integers(-8, 8))), color, thickness)

    return np.clip(img, 0, 255).astype(np.uint8)


def _mask_with_exact_coverage(score_field: np.ndarray, target_coverage: float) -> np.ndarray:
    """Select the highest-scoring pixels so mask coverage matches the target closely."""
    total = score_field.size
    k = int(round(float(target_coverage) * total))
    if k <= 0:
        return np.zeros(score_field.shape, dtype=np.uint8)
    if k >= total:
        return np.ones(score_field.shape, dtype=np.uint8)

    flat = score_field.reshape(-1)
    chosen = np.argpartition(flat, total - k)[total - k:]
    mask = np.zeros(total, dtype=np.uint8)
    mask[chosen] = 1
    return mask.reshape(score_field.shape)


def generate_blob_mask(size: int, target_coverage: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a patchy bloom mask with coverage matched to the requested value."""
    if target_coverage <= 0.0:
        return np.zeros((size, size), dtype=np.uint8)

    if target_coverage >= 0.98:
        score = np.ones((size, size), dtype=np.float32)
        for _ in range(int(rng.integers(5, 14))):
            center = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            axes = (int(rng.integers(8, 30)), int(rng.integers(8, 30)))
            angle = float(rng.uniform(0, 180))
            cv2.ellipse(score, center, axes, angle, 0, 360, 0.0, -1)
        score = cv2.GaussianBlur(score, (0, 0), sigmaX=5, sigmaY=5)
        return _mask_with_exact_coverage(score, target_coverage)

    field = np.zeros((size, size), dtype=np.float32)
    n_blobs = int(np.interp(target_coverage, [0.02, 0.95], [3, 42]))
    n_blobs = max(1, n_blobs)

    for _ in range(n_blobs):
        center = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        min_axis = int(np.interp(target_coverage, [0.02, 0.95], [7, 16]))
        max_axis = int(np.interp(target_coverage, [0.02, 0.95], [22, 52]))
        axes = (
            int(rng.integers(min_axis, max_axis + 1)),
            int(rng.integers(min_axis, max_axis + 1)),
        )
        angle = float(rng.uniform(0, 180))
        intensity = float(rng.uniform(0.5, 1.0))
        cv2.ellipse(field, center, axes, angle, 0, 360, intensity, -1)

    texture = rng.normal(0, 1, (size, size)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), sigmaX=7, sigmaY=7)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=5, sigmaY=5)
    field = field + 0.45 * texture

    initial = _mask_with_exact_coverage(field, target_coverage)
    softened = cv2.GaussianBlur(initial.astype(np.float32), (0, 0), sigmaX=1.1, sigmaY=1.1)
    return _mask_with_exact_coverage(softened, target_coverage)


def add_bloom_to_water(water: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Overlay green bloom patches on the water background."""
    img = water.astype(np.float32).copy()
    h, w = mask.shape

    bloom_color = np.zeros((h, w, 3), dtype=np.float32)
    bloom_color[:, :, 0] = rng.normal(65, 14, (h, w))
    bloom_color[:, :, 1] = rng.normal(150, 28, (h, w))
    bloom_color[:, :, 2] = rng.normal(55, 16, (h, w))

    streak = rng.normal(0, 1, (h, w)).astype(np.float32)
    streak = cv2.GaussianBlur(streak, (0, 0), sigmaX=8, sigmaY=2)
    bloom_color[:, :, 1] += 20 * streak
    bloom_color[:, :, 2] += 10 * streak

    alpha = (mask.astype(np.float32) * rng.uniform(0.62, 0.82))[:, :, None]
    img = (1 - alpha) * img + alpha * bloom_color

    if rng.random() < 0.35:
        side = rng.choice(["top", "bottom", "left", "right"])
        shore_color = np.array([55, 75, 38], dtype=np.float32)
        thickness = int(rng.integers(8, 24))
        if side == "top":
            img[:thickness, :, :] = 0.65 * img[:thickness, :, :] + 0.35 * shore_color
        elif side == "bottom":
            img[-thickness:, :, :] = 0.65 * img[-thickness:, :, :] + 0.35 * shore_color
        elif side == "left":
            img[:, :thickness, :] = 0.65 * img[:, :thickness, :] + 0.35 * shore_color
        else:
            img[:, -thickness:, :] = 0.65 * img[:, -thickness:, :] + 0.35 * shore_color

    if rng.random() < 0.7:
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.5)

    return np.clip(img, 0, 255).astype(np.uint8)


def make_bloom_image(size: int, requested_coverage: float, seed: int) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    water = make_water_background(size, rng)
    mask = generate_blob_mask(size, requested_coverage, rng)
    actual_coverage = float(mask.mean())
    img = add_bloom_to_water(water, mask, rng)
    return img, actual_coverage


def class_contains(label: str, coverage: float) -> bool:
    lo, hi = CATEGORIES[label]
    if label == "dense_bloom":
        return lo <= coverage <= hi
    return lo <= coverage < hi




def quantize_coverage_for_class(label: str, requested: float, image_size: int) -> float:
    """Return the nearest pixel-representable coverage that remains inside the class.

    A 128 x 128 binary mask can only represent coverage in increments of
    1 / 16,384. Values sampled extremely close to a touching class boundary
    can round into the neighboring class. Quantizing the target before image
    generation prevents that boundary-crossing failure.
    """
    total_pixels = image_size * image_size
    lo, hi = CATEGORIES[label]

    min_pixels = math.ceil(lo * total_pixels - 1e-12)
    if label == "dense_bloom":
        max_pixels = math.floor(hi * total_pixels + 1e-12)
    else:
        # Upper bounds are exclusive for every class except dense_bloom.
        max_pixels = math.ceil(hi * total_pixels - 1e-12) - 1

    if min_pixels > max_pixels:
        raise ValueError(
            f"Class {label!r} is too narrow to represent at image size {image_size}."
        )

    requested_pixels = int(round(requested * total_pixels))
    safe_pixels = min(max(requested_pixels, min_pixels), max_pixels)
    return safe_pixels / total_pixels


def internal_boundaries(label: str) -> List[float]:
    lo, hi = CATEGORIES[label]
    boundaries: List[float] = []
    if lo > 0:
        boundaries.append(lo)
    if hi < 1:
        boundaries.append(hi)
    return boundaries


def edge_band_width(label: str, edge_band_fraction: float) -> float:
    lo, hi = CATEGORIES[label]
    return max(0.01, min((hi - lo) * edge_band_fraction, 0.04))


def sample_requested_coverage(
    label: str,
    mode: str,
    rng: random.Random,
    edge_band_fraction: float,
) -> Tuple[float, float]:
    """Return requested coverage and target boundary (NaN for interior)."""
    lo, hi = CATEGORIES[label]
    band = edge_band_width(label, edge_band_fraction)
    boundaries = internal_boundaries(label)

    if mode == "edge" and boundaries:
        boundary = rng.choice(boundaries)
        if math.isclose(boundary, lo):
            value = rng.uniform(lo, min(hi, lo + band))
        else:
            value = rng.uniform(max(lo, hi - band), hi)
        return value, boundary

    interior_lo = lo + (band if lo > 0 else 0)
    interior_hi = hi - (band if hi < 1 else 0)
    if interior_hi <= interior_lo:
        interior_lo, interior_hi = lo, hi
    return rng.uniform(interior_lo, interior_hi), float("nan")


def make_mode_block(count: int, edge_fraction: float, rng: random.Random) -> List[str]:
    edge_count = int(round(count * edge_fraction))
    modes = ["edge"] * edge_count + ["interior"] * (count - edge_count)
    rng.shuffle(modes)
    return modes


def validate_counts(total: int, name: str) -> None:
    if total <= 0:
        raise ValueError(f"{name} must be positive.")
    if total % len(DISPLAY_CLASS_ORDER) != 0:
        raise ValueError(f"{name} must be divisible by {len(DISPLAY_CLASS_ORDER)}.")


def build_tasks(args, image_root: Path) -> List[GenerationTask]:
    validate_counts(args.train_total, "train-total")
    validate_counts(args.val_total, "val-total")
    validate_counts(args.test_total, "test-total")

    subset_sizes = sorted(set(args.subset_sizes))
    if subset_sizes[-1] != args.train_total:
        raise ValueError("The largest subset size must equal --train-total.")
    for size in subset_sizes:
        validate_counts(size, f"subset size {size}")

    master_rng = random.Random(args.seed)
    tasks: List[GenerationTask] = []

    train_per_class = args.train_total // len(DISPLAY_CLASS_ORDER)
    subset_per_class = [size // len(DISPLAY_CLASS_ORDER) for size in subset_sizes]
    block_ends = subset_per_class
    block_starts = [0] + block_ends[:-1]

    for label in DISPLAY_CLASS_ORDER:
        label_rng = random.Random(master_rng.randint(0, 2_000_000_000))
        train_modes: List[str] = []
        for start, end in zip(block_starts, block_ends):
            train_modes.extend(make_mode_block(end - start, args.train_edge_fraction, label_rng))
        if len(train_modes) != train_per_class:
            raise RuntimeError("Internal error while building nested training blocks.")

        for rank, mode in enumerate(train_modes, start=1):
            requested, boundary = sample_requested_coverage(label, mode, label_rng, args.edge_band_fraction)
            filename = f"train_{label}_{rank:06d}.png"
            outpath = image_root / label / filename
            tasks.append(
                GenerationTask(
                    split="train",
                    label=label,
                    split_index=rank,
                    train_rank=rank,
                    sampling_mode=mode,
                    target_boundary=boundary,
                    requested_coverage=requested,
                    base_seed=label_rng.randint(0, 2_000_000_000),
                    image_size=args.image_size,
                    edge_band_fraction=args.edge_band_fraction,
                    outpath=str(outpath),
                )
            )

        for split, total, edge_fraction in [
            ("val", args.val_total, args.eval_edge_fraction),
            ("test", args.test_total, args.eval_edge_fraction),
        ]:
            per_class = total // len(DISPLAY_CLASS_ORDER)
            modes = make_mode_block(per_class, edge_fraction, label_rng)
            for idx, mode in enumerate(modes, start=1):
                requested, boundary = sample_requested_coverage(label, mode, label_rng, args.edge_band_fraction)
                filename = f"{split}_{label}_{idx:06d}.png"
                outpath = image_root / label / filename
                tasks.append(
                    GenerationTask(
                        split=split,
                        label=label,
                        split_index=idx,
                        train_rank=0,
                        sampling_mode=mode,
                        target_boundary=boundary,
                        requested_coverage=requested,
                        base_seed=label_rng.randint(0, 2_000_000_000),
                        image_size=args.image_size,
                        edge_band_fraction=args.edge_band_fraction,
                        outpath=str(outpath),
                    )
                )

    master_rng.shuffle(tasks)
    return tasks


def worker_generate(task: GenerationTask) -> dict:
    """Generate one correctly labeled image at a pixel-safe coverage value."""
    Path(task.outpath).parent.mkdir(parents=True, exist_ok=True)

    safe_requested = quantize_coverage_for_class(
        task.label, task.requested_coverage, task.image_size
    )
    seed = task.base_seed
    img_rgb, actual = make_bloom_image(task.image_size, safe_requested, seed)

    if not class_contains(task.label, actual):
        raise RuntimeError(
            f"Internal coverage error for {task.label}: requested={safe_requested:.8f}, "
            f"actual={actual:.8f}."
        )
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(task.outpath, img_bgr)
    if not ok:
        raise RuntimeError(f"Could not write image: {task.outpath}")

    outpath = Path(task.outpath)
    relative_path = str(Path("images") / task.label / outpath.name)

    return {
        "filename": outpath.name,
        "relative_path": relative_path,
        "label": task.label,
        "label_id": LABEL_TO_ID[task.label],
        "split": task.split,
        "train_rank": task.train_rank if task.split == "train" else "",
        "sampling_mode": task.sampling_mode,
        "target_boundary": "" if math.isnan(task.target_boundary) else round(task.target_boundary, 6),
        "requested_coverage": round(float(safe_requested), 6),
        "actual_mask_coverage": round(float(actual), 6),
        "seed": int(seed),
    }


def write_csv(records: List[dict], csv_path: Path, subset_sizes: List[int]) -> None:
    fieldnames = [
        "filename",
        "relative_path",
        "label",
        "label_id",
        "split",
        "train_rank",
        "sampling_mode",
        "target_boundary",
        "requested_coverage",
        "actual_mask_coverage",
        "seed",
    ] + [f"use_{size}" for size in subset_sizes]

    per_class_limits = {size: size // len(DISPLAY_CLASS_ORDER) for size in subset_sizes}

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            for size, rank_limit in per_class_limits.items():
                row[f"use_{size}"] = int(
                    rec["split"] == "train" and int(rec["train_rank"]) <= rank_limit
                )
            writer.writerow(row)


def summarize(records: List[dict], args) -> str:
    lines: List[str] = []
    lines.append("CobberEcoBloom v2 master dataset")
    lines.append("=" * 72)
    lines.append(f"Image size: {args.image_size} x {args.image_size}")
    lines.append(f"Training images: {args.train_total}")
    lines.append(f"Validation images: {args.val_total}")
    lines.append(f"Test images: {args.test_total}")
    lines.append(f"Nested training subsets: {', '.join(map(str, args.subset_sizes))}")
    lines.append(f"Training edge-case fraction: {args.train_edge_fraction:.2f}")
    lines.append(f"Validation/test edge-case fraction: {args.eval_edge_fraction:.2f}")
    lines.append("")
    lines.append("Coverage ranges:")
    for label in DISPLAY_CLASS_ORDER:
        lo, hi = CATEGORIES[label]
        close = "]" if label == "dense_bloom" else ")"
        lines.append(f"  {label:14s} [{lo:.2f}, {hi:.2f}{close}")
    lines.append("")

    for split in ["train", "val", "test"]:
        lines.append(f"Counts for {split}:")
        for label in DISPLAY_CLASS_ORDER:
            subset = [r for r in records if r["split"] == split and r["label"] == label]
            edge_count = sum(r["sampling_mode"] == "edge" for r in subset)
            mean_cov = sum(float(r["actual_mask_coverage"]) for r in subset) / max(1, len(subset))
            lines.append(
                f"  {label:14s} total={len(subset):5d}  edge={edge_count:5d}  mean coverage={mean_cov:.4f}"
            )
        lines.append("")

    lines.append("Model output index order:")
    for idx, label in enumerate(MODEL_CLASS_ORDER):
        lines.append(f"  {idx}: {label}")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the CobberEcoBloom v2 master dataset.")
    parser.add_argument("--outdir", default="BloomData_v2", help="Output dataset directory.")
    parser.add_argument("--image-size", type=int, default=128, help="Square image size. Default: 128")
    parser.add_argument("--train-total", type=int, default=16000, help="Total master training images.")
    parser.add_argument("--val-total", type=int, default=1000, help="Total validation images.")
    parser.add_argument("--test-total", type=int, default=1000, help="Total test images.")
    parser.add_argument(
        "--subset-sizes",
        type=int,
        nargs="+",
        default=[2000, 8000, 16000],
        help="Nested training sizes. Largest must equal train-total.",
    )
    parser.add_argument("--train-edge-fraction", type=float, default=0.45)
    parser.add_argument("--eval-edge-fraction", type=float, default=0.65)
    parser.add_argument(
        "--edge-band-fraction",
        type=float,
        default=0.15,
        help="Fraction of each class width treated as near-boundary, capped at 0.04.",
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes. Default: CPU count minus one.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    for name in ["train_edge_fraction", "eval_edge_fraction", "edge_band_fraction"]:
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1.")

    args.subset_sizes = sorted(set(args.subset_sizes))
    outdir = Path(args.outdir).resolve()
    image_root = outdir / "images"
    outdir.mkdir(parents=True, exist_ok=True)
    for label in DISPLAY_CLASS_ORDER:
        (image_root / label).mkdir(parents=True, exist_ok=True)

    workers = args.workers if args.workers is not None else max(1, (mp.cpu_count() or 1) - 1)

    print("=" * 72)
    print("CobberEcoBloom v2 dataset generation")
    print("=" * 72)
    print(f"Output directory:      {outdir}")
    print(f"Image size:            {args.image_size} x {args.image_size}")
    print(f"Training images:       {args.train_total}")
    print(f"Validation images:     {args.val_total}")
    print(f"Test images:           {args.test_total}")
    print(f"Training subsets:      {args.subset_sizes}")
    print(f"Worker processes:      {workers}")
    print(f"Training edge fraction:{args.train_edge_fraction:.2f}")
    print(f"Eval edge fraction:    {args.eval_edge_fraction:.2f}")
    print("-" * 72)

    t0 = time.perf_counter()
    tasks = build_tasks(args, image_root)
    total = len(tasks)
    records: List[dict] = []

    with mp.Pool(processes=workers) as pool:
        for i, record in enumerate(pool.imap_unordered(worker_generate, tasks, chunksize=16), start=1):
            records.append(record)
            if i % 500 == 0 or i == total:
                print(f"Generated {i:6d} / {total:6d} images")

    records.sort(
        key=lambda r: (
            {"train": 0, "val": 1, "test": 2}[r["split"]],
            DISPLAY_CLASS_ORDER.index(r["label"]),
            int(r["train_rank"] or 0),
            r["filename"],
        )
    )

    csv_path = outdir / "bloom_supervisor_v2.csv"
    write_csv(records, csv_path, args.subset_sizes)

    metadata = {
        "display_class_order": DISPLAY_CLASS_ORDER,
        "model_class_order": MODEL_CLASS_ORDER,
        "coverage_ranges": CATEGORIES,
        "train_total": args.train_total,
        "val_total": args.val_total,
        "test_total": args.test_total,
        "subset_sizes": args.subset_sizes,
        "image_size": args.image_size,
        "seed": args.seed,
    }
    with open(outdir / "dataset_metadata_v2.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    summary_text = summarize(records, args)
    with open(outdir / "dataset_summary_v2.txt", "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    elapsed = time.perf_counter() - t0
    print("-" * 72)
    print(f"Supervisor CSV: {csv_path}")
    print(f"Summary:        {outdir / 'dataset_summary_v2.txt'}")
    print(f"Metadata:       {outdir / 'dataset_metadata_v2.json'}")
    print(f"Elapsed time:   {elapsed:.1f} seconds")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

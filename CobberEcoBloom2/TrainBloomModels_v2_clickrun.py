#!/usr/bin/env python3
"""
Train three CobberEcoBloom CNNs from one master v2 dataset.

Default training sizes:
    - 2,000 training images
    - 8,000 training images
    - 16,000 training images

All three models use the same validation and test images. The training subsets are
nested, balanced, and read from bloom_supervisor_v2.csv created by
GenerateBloomTrainingData_v2.py.

Typical run from PyCharm:
    Click Run. If no dataset argument is supplied, the script automatically uses
    BloomData_v2 beside this script.

Command-line run:
    python TrainBloomModels_v2_clickrun.py BloomData_v2 --mixed-precision

CPU run:
    python TrainBloomModels_v2_clickrun.py BloomData_v2 --device cpu

Dependencies:
    pip install tensorflow pandas numpy scikit-learn matplotlib
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


DISPLAY_CLASS_ORDER = [
    "clear_water",
    "mostly_clear",
    "field_check",
    "mostly_bloom",
    "dense_bloom",
]

# Preserve the model-output order already hard-coded in the no-pickle app.
MODEL_CLASS_ORDER = [
    "clear_water",
    "dense_bloom",
    "field_check",
    "mostly_bloom",
    "mostly_clear",
]

LABEL_TO_ID = {label: i for i, label in enumerate(MODEL_CLASS_ORDER)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Train the CobberEcoBloom v2 model series.")
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=None,
        help=(
            "Dataset directory created by GenerateBloomTrainingData_v2.py. "
            "If omitted, use BloomData_v2 beside this script."
        ),
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Supervisor CSV. Default: <dataset_dir>/bloom_supervisor_v2.csv",
    )
    parser.add_argument(
        "--outroot",
        default="BloomModels_v3",
        help="Root output directory. Default: BloomModels_v3",
    )
    parser.add_argument(
        "--training-sizes",
        type=int,
        nargs="+",
        default=[2000, 8000, 16000],
        help="Training sizes to fit. Default: 2000 8000 16000",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs per model.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--cache", action="store_true", help="Cache decoded images in RAM.")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def configure_tensorflow(args):
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import numpy as np
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if args.device == "gpu" and not gpus:
        print("WARNING: --device gpu was requested, but TensorFlow reports no GPU.")

    if gpus:
        print("TensorFlow-visible GPUs:")
        for gpu in gpus:
            print(f"  {gpu}")
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as exc:
                print(f"WARNING: Could not enable memory growth on {gpu}: {exc}")
    else:
        print("TensorFlow-visible GPUs: none")

    if args.mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision enabled: mixed_float16")
        except Exception as exc:
            print(f"WARNING: Could not enable mixed precision: {exc}")

    return tf, np


def load_supervisor(dataset_dir: Path, csv_path: Path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = {
        "relative_path",
        "label",
        "split",
        "train_rank",
        "actual_mask_coverage",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Supervisor CSV is missing columns: {sorted(missing)}")

    df["path"] = df["relative_path"].apply(lambda p: str(dataset_dir / str(p)))
    missing_paths = [p for p in df["path"].tolist() if not Path(p).exists()]
    if missing_paths:
        preview = "\n".join(missing_paths[:5])
        raise FileNotFoundError(
            f"{len(missing_paths)} listed images were not found. First few:\n{preview}"
        )

    unknown = sorted(set(df["label"].astype(str)) - set(MODEL_CLASS_ORDER))
    if unknown:
        raise ValueError(f"Unknown labels in supervisor CSV: {unknown}")

    df["label_id"] = df["label"].astype(str).map(LABEL_TO_ID).astype(int)
    return df


def make_tf_dataset(
    tf,
    paths: List[str],
    labels: List[int],
    image_size: int,
    batch_size: int,
    shuffle: bool,
    cache: bool,
    seed: int,
):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_png(raw, channels=3)
        img = tf.image.resize(img, [image_size, image_size], method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(paths), 10000),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(tf, image_size: int, learning_rate: float, use_augment: bool):
    layers = tf.keras.layers
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = inputs

    if use_augment:
        x = layers.RandomFlip("horizontal_and_vertical")(x)
        x = layers.RandomRotation(0.05)(x)
        x = layers.RandomZoom(0.08)(x)

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(192, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(len(MODEL_CLASS_ORDER), activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history(history, outdir: Path, size: int):
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", range(1, len(history_df) + 1))
    history_path = outdir / f"training_history_{size}.csv"
    history_df.to_csv(history_path, index=False)

    epochs = history_df["epoch"]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(epochs, history_df["loss"], label="training loss")
    if "val_loss" in history_df:
        ax.plot(epochs, history_df["val_loss"], label="validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"CobberEcoBloom loss: {size:,} training images")
    ax.legend()
    fig.tight_layout()
    loss_path = outdir / f"training_curve_{size}.png"
    fig.savefig(loss_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(epochs, history_df["accuracy"], label="training accuracy")
    if "val_accuracy" in history_df:
        ax.plot(epochs, history_df["val_accuracy"], label="validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"CobberEcoBloom accuracy: {size:,} training images")
    ax.legend()
    fig.tight_layout()
    acc_path = outdir / f"accuracy_curve_{size}.png"
    fig.savefig(acc_path, dpi=180)
    plt.close(fig)

    return history_path, loss_path, acc_path


def evaluate_and_save(model, test_ds, outdir: Path, size: int):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix

    y_true_ids: List[int] = []
    y_pred_ids: List[int] = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_pred_ids.extend(np.argmax(probs, axis=1).tolist())
        y_true_ids.extend(labels.numpy().tolist())

    y_true_labels = [ID_TO_LABEL[i] for i in y_true_ids]
    y_pred_labels = [ID_TO_LABEL[i] for i in y_pred_ids]

    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=DISPLAY_CLASS_ORDER)
    report = classification_report(
        y_true_labels,
        y_pred_labels,
        labels=DISPLAY_CLASS_ORDER,
        target_names=DISPLAY_CLASS_ORDER,
        digits=4,
        zero_division=0,
    )

    report_path = outdir / f"classification_report_{size}.txt"
    report_path.write_text(report, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"CobberEcoBloom test confusion matrix\n{size:,} training images")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_xticks(range(len(DISPLAY_CLASS_ORDER)))
    ax.set_yticks(range(len(DISPLAY_CLASS_ORDER)))
    ax.set_xticklabels([x.replace("_", " ") for x in DISPLAY_CLASS_ORDER], rotation=45, ha="right")
    ax.set_yticklabels([x.replace("_", " ") for x in DISPLAY_CLASS_ORDER])

    vmax = max(1, int(cm.max()))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > vmax * 0.55 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    cm_path = outdir / f"confusion_matrix_{size}.png"
    fig.savefig(cm_path, dpi=180)
    plt.close(fig)

    accuracy = float(sum(a == b for a, b in zip(y_true_labels, y_pred_labels)) / len(y_true_labels))
    return accuracy, cm.tolist(), report_path, cm_path


def select_training_subset(df, size: int):
    if size % len(DISPLAY_CLASS_ORDER) != 0:
        raise ValueError(f"Training size {size} is not divisible by {len(DISPLAY_CLASS_ORDER)}.")

    per_class = size // len(DISPLAY_CLASS_ORDER)
    train = df[df["split"] == "train"].copy()
    train["train_rank"] = train["train_rank"].astype(int)
    subset = train[train["train_rank"] <= per_class].copy()

    counts = subset["label"].value_counts().to_dict()
    expected = {label: per_class for label in DISPLAY_CLASS_ORDER}
    if counts != expected:
        raise ValueError(
            f"Training subset {size} is not balanced as expected. Found {counts}; expected {expected}."
        )
    return subset


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    if args.dataset_dir:
        requested_dataset = Path(args.dataset_dir).expanduser()
        dataset_dir = (requested_dataset if requested_dataset.is_absolute() else script_dir / requested_dataset).resolve()
    else:
        dataset_dir = (script_dir / "BloomData_v2").resolve()

    if not dataset_dir.exists():
        raise SystemExit(
            "Dataset directory not found:\n"
            f"  {dataset_dir}\n\n"
            "Place the BloomData_v2 folder beside this script, or supply its path "
            "as the first command-line argument."
        )

    if args.csv:
        requested_csv = Path(args.csv).expanduser()
        csv_path = (requested_csv if requested_csv.is_absolute() else script_dir / requested_csv).resolve()
    else:
        csv_path = dataset_dir / "bloom_supervisor_v2.csv"

    if not csv_path.exists():
        raise SystemExit(f"Supervisor CSV not found: {csv_path}")

    requested_outroot = Path(args.outroot).expanduser()
    outroot = (requested_outroot if requested_outroot.is_absolute() else script_dir / requested_outroot).resolve()
    outroot.mkdir(parents=True, exist_ok=True)
    args.training_sizes = sorted(set(args.training_sizes))

    tf, np = configure_tensorflow(args)
    import pandas as pd

    df = load_supervisor(dataset_dir, csv_path)
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    if val_df.empty or test_df.empty:
        raise SystemExit("Validation or test split is empty.")

    print("=" * 76)
    print("CobberEcoBloom v3 model training")
    print("=" * 76)
    print(f"Dataset directory:   {dataset_dir}")
    print(f"Supervisor CSV:      {csv_path}")
    print(f"Output root:         {outroot}")
    print(f"Training sizes:      {args.training_sizes}")
    print(f"Validation images:   {len(val_df)}")
    print(f"Test images:         {len(test_df)}")
    print(f"Image size:          {args.image_size} x {args.image_size}")
    print(f"Maximum epochs:      {args.epochs}")
    print(f"Batch size:          {args.batch_size}")
    print(f"Augmentation:        {not args.no_augment}")
    print(f"Model class order:   {MODEL_CLASS_ORDER}")
    print("-" * 76)

    val_ds = make_tf_dataset(
        tf,
        val_df["path"].tolist(),
        val_df["label_id"].tolist(),
        args.image_size,
        args.batch_size,
        shuffle=False,
        cache=args.cache,
        seed=args.seed,
    )
    test_ds = make_tf_dataset(
        tf,
        test_df["path"].tolist(),
        test_df["label_id"].tolist(),
        args.image_size,
        args.batch_size,
        shuffle=False,
        cache=args.cache,
        seed=args.seed,
    )

    comparison_rows: List[dict] = []
    overall_start = time.perf_counter()

    for size in args.training_sizes:
        print("\n" + "=" * 76)
        print(f"Training model with {size:,} training images")
        print("=" * 76)

        train_df = select_training_subset(df, size)
        outdir = outroot / f"BloomModel_{size}"
        outdir.mkdir(parents=True, exist_ok=True)

        tf.keras.backend.clear_session()
        gc.collect()
        tf.keras.utils.set_random_seed(args.seed)

        train_ds = make_tf_dataset(
            tf,
            train_df["path"].tolist(),
            train_df["label_id"].tolist(),
            args.image_size,
            args.batch_size,
            shuffle=True,
            cache=args.cache,
            seed=args.seed,
        )

        model = build_model(
            tf,
            image_size=args.image_size,
            learning_rate=args.learning_rate,
            use_augment=not args.no_augment,
        )

        best_model_path = outdir / f"cobber_bloom_model_{size}_best.keras"
        final_model_path = outdir / f"cobber_bloom_model_{size}.keras"

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(best_model_path),
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=args.patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=0.5,
                patience=max(2, args.patience // 2),
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        model.summary()
        train_start = time.perf_counter()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
            shuffle=False,
            verbose=1,
        )
        train_seconds = time.perf_counter() - train_start
        val_accuracy_history = history.history.get("val_accuracy", [])

        if val_accuracy_history:
            best_epoch = int(np.argmax(val_accuracy_history)) + 1
            best_validation_accuracy = float(val_accuracy_history[best_epoch - 1])
        else:
            best_epoch = None
            best_validation_accuracy = None

        model.save(final_model_path)
        history_path, loss_curve_path, accuracy_curve_path = save_history(history, outdir, size)

        print("Evaluating on the shared test set...")
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
        cm_accuracy, cm_values, report_path, cm_path = evaluate_and_save(model, test_ds, outdir, size)

        metadata = {
            "training_size": size,
            "train_images_per_class": size // len(DISPLAY_CLASS_ORDER),
            "validation_images": len(val_df),
            "test_images": len(test_df),
            "display_class_order": DISPLAY_CLASS_ORDER,
            "model_class_order": MODEL_CLASS_ORDER,
            "image_size": args.image_size,
            "epochs_requested": args.epochs,
            "epochs_completed": len(history.history.get("loss", [])),
            "best_epoch_by_validation_accuracy": best_epoch,
            "best_validation_accuracy": best_validation_accuracy,
            "checkpoint_monitor": "val_accuracy",
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "augmentation": not args.no_augment,
            "test_loss": float(test_loss),
            "test_accuracy_model_evaluate": float(test_accuracy),
            "test_accuracy_confusion_matrix": float(cm_accuracy),
            "training_seconds": float(train_seconds),
            "confusion_matrix_display_order": cm_values,
            "best_model": str(best_model_path),
            "final_model": str(final_model_path),
        }
        write_json(outdir / f"model_metadata_{size}.json", metadata)
        write_json(outdir / "class_order.json", {"model_class_order": MODEL_CLASS_ORDER})

        summary_lines = [
            "CobberEcoBloom v3 training summary",
            "=" * 64,
            f"Training images: {size}",
            f"Training images per class: {size // len(DISPLAY_CLASS_ORDER)}",
            f"Validation images: {len(val_df)}",
            f"Test images: {len(test_df)}",
            f"Epochs requested: {args.epochs}",
            f"Epochs completed: {len(history.history.get('loss', []))}",
            f"Best epoch by validation accuracy: {best_epoch if best_epoch is not None else 'not available'}",
            f"Best validation accuracy: {best_validation_accuracy:.6f}" if best_validation_accuracy is not None else "Best validation accuracy: not available",
            f"Checkpoint monitor: val_accuracy",
            f"Training time: {train_seconds:.3f} s",
            f"Test loss: {test_loss:.6f}",
            f"Test accuracy: {cm_accuracy:.6f}",
            f"Model output order: {', '.join(MODEL_CLASS_ORDER)}",
            "",
            f"Best model: {best_model_path}",
            f"Final model: {final_model_path}",
            f"History: {history_path}",
            f"Loss curve: {loss_curve_path}",
            f"Accuracy curve: {accuracy_curve_path}",
            f"Confusion matrix: {cm_path}",
            f"Classification report: {report_path}",
        ]
        (outdir / f"training_summary_{size}.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        comparison_rows.append(
            {
                "training_size": size,
                "epochs_completed": len(history.history.get("loss", [])),
                "best_epoch": best_epoch,
                "best_validation_accuracy": best_validation_accuracy,
                "training_seconds": round(train_seconds, 3),
                "test_loss": float(test_loss),
                "test_accuracy": float(cm_accuracy),
                "best_model": str(best_model_path),
            }
        )

        del model, train_ds
        tf.keras.backend.clear_session()
        gc.collect()

    comparison_df = pd.DataFrame(comparison_rows).sort_values("training_size")
    comparison_path = outroot / "model_comparison_v3.csv"
    comparison_df.to_csv(comparison_path, index=False)

    write_json(
        outroot / "model_series_metadata_v3.json",
        {
            "training_sizes": args.training_sizes,
            "display_class_order": DISPLAY_CLASS_ORDER,
            "model_class_order": MODEL_CLASS_ORDER,
            "model_output_class_order": MODEL_CLASS_ORDER,
            "shared_validation_images": len(val_df),
            "shared_test_images": len(test_df),
            "dataset_dir": str(dataset_dir),
            "supervisor_csv": str(csv_path),
        },
    )

    elapsed = time.perf_counter() - overall_start
    print("\n" + "=" * 76)
    print("All requested models are complete.")
    print(f"Comparison CSV: {comparison_path}")
    print(f"Total elapsed time: {elapsed:.1f} seconds")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

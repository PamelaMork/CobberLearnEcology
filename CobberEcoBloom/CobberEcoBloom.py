#!/usr/bin/env python3
"""
CobberEcoBloom

A self-contained PyQt6 application for auditing pretrained CNNs on synthetic
lake-surface images. The app generates all standard and challenge images internally and finds pretrained Keras models recursively below the application folder. The model output class order is stored directly in this file, so no pickle label-encoder files are needed.

Expected model folders may be placed anywhere below this file, for example:

    BloomModel_2000/
    BloomModel_8000/
    BloomModel_16000/
"""

from __future__ import annotations

import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPixmap, QBrush
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model


# ----------------------------
# Path helpers
# ----------------------------
def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_ROOT = app_root()

# ----------------------------
# Synthetic bloom image engine
# ----------------------------
CATEGORIES: Dict[str, Tuple[float, float]] = {
    "clear_water": (0.00, 0.10),
    "mostly_clear": (0.10, 0.25),
    "field_check": (0.25, 0.75),
    "mostly_bloom": (0.75, 0.90),
    "dense_bloom": (0.90, 1.00),
}

CATEGORY_ORDER = [
    "clear_water",
    "mostly_clear",
    "field_check",
    "mostly_bloom",
    "dense_bloom",
]

# The pretrained Keras models were trained with scikit-learn LabelEncoder,
# which sorted the five output classes alphabetically. Keep this order fixed.
MODEL_CLASS_ORDER = [
    "clear_water",
    "dense_bloom",
    "field_check",
    "mostly_bloom",
    "mostly_clear",
]

CONDITION_LABELS = {
    "standard": "Standard",
    "darker_water": "Darker water",
    "glare": "Surface glare",
    "altered_bloom_color": "Altered bloom color",
}

EXPECTED_MODEL_SIZES = ["2000", "8000", "16000"]
INFOBLUE = "#3E6990"
INFOBLUE_CMAP = LinearSegmentedColormap.from_list(
    "cobber_infoblue",
    ["#F7FBFF", "#DCE8F2", "#9EB8CF", INFOBLUE],
)


@dataclass
class SurveyEntry:
    category: str
    coverage: float
    seed: int


@dataclass
class EvalRecord:
    image_num: int
    mode: str
    condition: str
    actual: str
    student: str
    student_result: str
    predicted: str
    model_result: str
    coverage: float


def pretty_label(label: str) -> str:
    return label.replace("_", " ")


def make_water_background(size: int, rng: np.random.Generator, variant: str = "standard") -> np.ndarray:
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

    if variant == "darker_water":
        img[:, :, 2] *= 0.72  # blue
        img[:, :, 1] *= 0.86  # green
        img[:, :, 0] *= 0.82  # red
        img[:, :, 1] += 5
        img[:, :, 0] += 3

    return np.clip(img, 0, 255).astype(np.uint8)


def _mask_with_exact_coverage(score_field: np.ndarray, target_coverage: float) -> np.ndarray:
    """Select the highest-scoring pixels so mask coverage matches the target."""
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
    """Generate a patchy bloom mask with coverage matched to the target."""
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
    n_blobs = max(1, int(np.interp(target_coverage, [0.02, 0.95], [3, 42])))

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

def add_bloom_to_water(
        water: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
        variant: str = "standard",
) -> np.ndarray:
    img = water.astype(np.float32).copy()

    h, w = mask.shape
    bloom_color = np.zeros((h, w, 3), dtype=np.float32)

    if variant == "altered_bloom_color":
        bloom_color[:, :, 0] = rng.normal(82, 14, (h, w))
        bloom_color[:, :, 1] = rng.normal(128, 20, (h, w))
        bloom_color[:, :, 2] = rng.normal(52, 12, (h, w))
    else:
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


def add_surface_glare(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float32).copy()
    h, w, _ = out.shape
    overlay = np.zeros_like(out, dtype=np.float32)

    center = (float(rng.uniform(w * 0.25, w * 0.75)), float(rng.uniform(h * 0.2, h * 0.8)))
    sigma_x = float(rng.uniform(w * 0.16, w * 0.30))
    sigma_y = float(rng.uniform(h * 0.03, h * 0.08))
    angle = float(rng.uniform(-25, 25))

    yy, xx = np.mgrid[0:h, 0:w]
    x = xx - center[0]
    y = yy - center[1]
    theta = np.deg2rad(angle)
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    glare = np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
    glare += 0.45 * np.exp(-0.5 * ((xr / (sigma_x * 0.55)) ** 2 + (yr / (sigma_y * 0.45)) ** 2))
    glare = glare / glare.max()

    overlay[:, :, 0] = 250
    overlay[:, :, 1] = 248
    overlay[:, :, 2] = 235
    alpha = 0.42 * glare[:, :, None]
    out = (1 - alpha) * out + alpha * overlay

    # add a couple of narrow specular streaks
    for _ in range(3):
        y0 = int(rng.uniform(h * 0.15, h * 0.85))
        x0 = int(rng.uniform(0, w * 0.35))
        x1 = int(rng.uniform(w * 0.65, w))
        color = (245, 245, 235)
        thickness = int(rng.integers(1, 3))
        cv2.line(out, (x0, y0), (x1, y0 + int(rng.integers(-5, 6))), color, thickness)

    return np.clip(out, 0, 255).astype(np.uint8)


def make_bloom_image(size: int, coverage: float, seed: int, variant: str = "standard"):
    rng = np.random.default_rng(seed)
    water_variant = "darker_water" if variant == "darker_water" else "standard"
    water = make_water_background(size, rng, variant=water_variant)
    mask = generate_blob_mask(size, coverage, rng)
    actual_coverage = float(mask.mean())
    bloom_variant = "altered_bloom_color" if variant == "altered_bloom_color" else "standard"
    img = add_bloom_to_water(water, mask, rng, variant=bloom_variant)
    if variant == "glare":
        img = add_surface_glare(img, rng)
    return img, actual_coverage


def choose_random_actual_category() -> str:
    r = random.random()
    if r < 0.20:
        return "clear_water"
    if r < 0.40:
        return "dense_bloom"
    if r < 0.60:
        return "mostly_clear"
    if r < 0.80:
        return "mostly_bloom"
    return "field_check"


# Evaluation images are generated at one fixed size so every CNN sees the same
# underlying pixels. Images are resized only if a loaded model expects another
# input size.
EVAL_IMAGE_SIZE = 128


def _quantize_coverage_for_class(category: str, coverage: float, image_size: int) -> float:
    """Round coverage to a representable pixel count that stays inside its class."""
    total_pixels = image_size * image_size
    lo, hi = CATEGORIES[category]
    min_pixels = math.ceil(lo * total_pixels - 1e-12)
    if category == "dense_bloom":
        max_pixels = math.floor(hi * total_pixels + 1e-12)
    else:
        max_pixels = math.ceil(hi * total_pixels - 1e-12) - 1
    requested_pixels = int(round(coverage * total_pixels))
    safe_pixels = min(max(requested_pixels, min_pixels), max_pixels)
    return safe_pixels / total_pixels


def _sample_manifest_coverage(
    category: str,
    rng: random.Random,
    edge_case: bool,
    edge_band_fraction: float = 0.15,
) -> float:
    lo, hi = CATEGORIES[category]
    band = max(0.01, min((hi - lo) * edge_band_fraction, 0.04))
    boundaries = []
    if lo > 0:
        boundaries.append(lo)
    if hi < 1:
        boundaries.append(hi)

    if edge_case and boundaries:
        boundary = rng.choice(boundaries)
        if math.isclose(boundary, lo):
            value = rng.uniform(lo, min(hi, lo + band))
        else:
            value = rng.uniform(max(lo, hi - band), hi)
    else:
        interior_lo = lo + (band if lo > 0 else 0)
        interior_hi = hi - (band if hi < 1 else 0)
        if interior_hi <= interior_lo:
            interior_lo, interior_hi = lo, hi
        value = rng.uniform(interior_lo, interior_hi)

    return _quantize_coverage_for_class(category, value, EVAL_IMAGE_SIZE)


def make_fixed_manifest(
    n_per_class: int = 40,
    seed: int = 2027,
    edge_fraction: float = 0.65,
) -> List[SurveyEntry]:
    """Create a balanced fixed manifest with deliberate boundary cases."""
    rng = random.Random(seed)
    manifest: List[SurveyEntry] = []
    edge_count = int(round(n_per_class * edge_fraction))

    for category in CATEGORY_ORDER:
        modes = [True] * edge_count + [False] * (n_per_class - edge_count)
        rng.shuffle(modes)
        for edge_case in modes:
            coverage = _sample_manifest_coverage(category, rng, edge_case)
            image_seed = rng.randint(0, 2_000_000_000)
            manifest.append(SurveyEntry(category=category, coverage=coverage, seed=image_seed))

    rng.shuffle(manifest)
    return manifest


BENCHMARK_MANIFEST = make_fixed_manifest(n_per_class=40, seed=2027, edge_fraction=0.65)
MANUAL_MANIFEST = make_fixed_manifest(n_per_class=4, seed=7331, edge_fraction=0.50)
CHALLENGE_CONDITIONS = ["standard", "darker_water", "glare", "altered_bloom_color"]


# ----------------------------
# Utility helpers
# ----------------------------
def determine_error_type(actual: str, predicted: str) -> str:
    clear_types = {"clear_water", "mostly_clear"}
    bloom_types = {"dense_bloom", "mostly_bloom"}

    if actual == predicted:
        return "correct"
    if actual == "field_check" or predicted == "field_check":
        return "serious"
    if actual in clear_types and predicted in bloom_types:
        return "serious"
    if actual in bloom_types and predicted in clear_types:
        return "serious"
    return "minor"


def result_display(result: str) -> str:
    return {
        "correct": "correct",
        "minor": "minor mix-up",
        "serious": "serious error",
        "": "",
    }.get(result, result)


def numpy_rgb_to_qimage(img_rgb: np.ndarray) -> QImage:
    img_rgb = np.ascontiguousarray(img_rgb)
    height, width, channels = img_rgb.shape
    bytes_per_line = channels * width
    q_img = QImage(
        img_rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    )
    return q_img.copy()


def calculate_metrics(records: List[EvalRecord], result_field: str = "model_result") -> Dict[str, float]:
    total = len(records)
    if total == 0:
        return {"total": 0, "exact": 0, "minor": 0, "serious": 0, "accuracy": 0.0}
    exact = sum(getattr(record, result_field) == "correct" for record in records)
    minor = sum(getattr(record, result_field) == "minor" for record in records)
    serious = sum(getattr(record, result_field) == "serious" for record in records)
    return {
        "total": total,
        "exact": exact,
        "minor": minor,
        "serious": serious,
        "accuracy": 100.0 * exact / total,
    }


def model_input_size(model) -> int:
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        if shape[1] is not None:
            return int(shape[1])
    except Exception:
        pass
    return EVAL_IMAGE_SIZE


def predict_class(model, img_rgb: np.ndarray) -> str:
    input_size = model_input_size(model)
    img_resized = cv2.resize(img_rgb, (input_size, input_size))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    preds = model.predict(img_input, verbose=0)
    pred_class_idx = int(np.argmax(preds, axis=1)[0])
    if pred_class_idx < 0 or pred_class_idx >= len(MODEL_CLASS_ORDER):
        raise ValueError(f"Unexpected model output index: {pred_class_idx}")
    return MODEL_CLASS_ORDER[pred_class_idx]


def scan_model_files(base_dir: Path) -> Dict[str, Path]:
    """Find models recursively, preferring BloomModels_v3 and _best files."""
    model_candidates: Dict[str, List[Path]] = {}
    for pattern in ("cobber_bloom_model_*.keras", "cobber_bloom_model_*.h5"):
        for path in base_dir.rglob(pattern):
            match = re.fullmatch(r"cobber_bloom_model_(\d+)(?:_best)?\.(?:keras|h5)", path.name)
            if match and match.group(1) in EXPECTED_MODEL_SIZES:
                model_candidates.setdefault(match.group(1), []).append(path)

    models: Dict[str, Path] = {}
    for size, candidates in model_candidates.items():
        def rank(path: Path):
            lower_parts = [part.lower() for part in path.parts]
            if "bloommodels_v3" in lower_parts:
                version_rank = 0
            elif "bloommodels_v2" in lower_parts:
                version_rank = 1
            else:
                version_rank = 2
            return (
                version_rank,
                0 if "_best" in path.stem else 1,
                len(path.relative_to(base_dir).parts),
                str(path).lower(),
            )

        models[size] = sorted(candidates, key=rank)[0]
    return models


def bold_label(text: str) -> QLabel:
    label = QLabel(text)
    font = label.font()
    font.setBold(True)
    label.setFont(font)
    return label


def configure_table(table: QTableWidget, headers: List[str]) -> None:
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    table.setMouseTracking(False)
    table.viewport().setMouseTracking(False)
    table.setAttribute(Qt.WidgetAttribute.WA_Hover, False)
    table.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, False)
    table.setAlternatingRowColors(True)
    table.setSortingEnabled(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.horizontalHeader().setMinimumSectionSize(85)


def apply_result_color(item: QTableWidgetItem, result: str) -> None:
    if result == "correct":
        item.setBackground(QBrush(QColor("#DDF1DF")))
    elif result == "minor":
        item.setBackground(QBrush(QColor("#DDE8F5")))
    elif result == "serious":
        item.setBackground(QBrush(QColor("#F4DCDD")))


# ----------------------------
# Metric cards
# ----------------------------
class MetricCard(QFrame):
    clicked = pyqtSignal(str)

    def __init__(self, key: str, title: str, clickable: bool = False):
        super().__init__()
        self.key = key
        self.clickable = clickable
        self._active = False
        self.setCursor(Qt.CursorShape.PointingHandCursor if clickable else Qt.CursorShape.ArrowCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)

        self.title_label = QLabel(title)
        title_font = self.title_label.font()
        title_font.setBold(True)
        title_font.setPointSize(11)
        self.title_label.setFont(title_font)

        self.accuracy_label = QLabel("")
        accuracy_font = self.accuracy_label.font()
        accuracy_font.setBold(True)
        accuracy_font.setPointSize(20)
        self.accuracy_label.setFont(accuracy_font)

        self.exact_caption = QLabel("Exact accuracy")
        self.details_label = QLabel("Minor mix-ups: —\nSerious errors: —")
        self.delta_label = QLabel("")
        self.delta_label.setWordWrap(True)

        layout.addWidget(self.title_label)
        layout.addWidget(self.accuracy_label)
        layout.addWidget(self.exact_caption)
        layout.addWidget(self.details_label)
        layout.addWidget(self.delta_label)
        layout.addStretch()
        self.reset()
        self._apply_style()

    def reset(self) -> None:
        font = self.accuracy_label.font()
        font.setPointSize(11)
        font.setBold(False)
        self.accuracy_label.setFont(font)
        self.accuracy_label.setStyleSheet("color: #666666;")
        self.accuracy_label.setText("Not run")
        self.details_label.setText("Minor mix-ups: —\nSerious errors: —")
        self.delta_label.setText("")

    def set_metrics(self, metrics: Dict[str, float], delta: Optional[float] = None) -> None:
        font = self.accuracy_label.font()
        font.setPointSize(20)
        font.setBold(True)
        self.accuracy_label.setFont(font)
        self.accuracy_label.setStyleSheet("")
        self.accuracy_label.setText(f"{metrics['accuracy']:.1f}%")
        self.details_label.setText(
            f"Minor mix-ups: {int(metrics['minor'])}\nSerious errors: {int(metrics['serious'])}"
        )
        if delta is None:
            self.delta_label.setText("")
        elif abs(delta) < 0.05:
            self.delta_label.setText("Change from standard: 0.0 percentage points")
        else:
            self.delta_label.setText(f"Change from standard: {delta:+.1f} percentage points")

    def set_active(self, active: bool) -> None:
        self._active = active
        self._apply_style()

    def _apply_style(self) -> None:
        border = "3px solid #6C1D45" if self._active else "1px solid #B7B7B7"
        background = "#FBF7F9" if self._active else "#FFFFFF"
        self.setStyleSheet(
            f"QFrame {{ border: {border}; border-radius: 7px; background-color: {background}; }}"
            "QLabel { border: none; background: transparent; }"
        )

    def mousePressEvent(self, event):
        if self.clickable and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.key)
        super().mousePressEvent(event)


# ----------------------------
# Background workers
# ----------------------------
class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    record = pyqtSignal(object)
    image = pyqtSignal(QImage)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)


class BenchmarkWorker(QRunnable):
    def __init__(self, model, manifest: List[SurveyEntry], condition: str):
        super().__init__()
        self.model = model
        self.manifest = manifest
        self.condition = condition
        self.signals = WorkerSignals()
        self.is_running = True

    def run(self):
        records: List[EvalRecord] = []
        try:
            total = len(self.manifest)
            for index, entry in enumerate(self.manifest, start=1):
                if not self.is_running:
                    break
                img, actual_coverage = make_bloom_image(
                    EVAL_IMAGE_SIZE,
                    entry.coverage,
                    entry.seed,
                    variant=self.condition,
                )
                predicted = predict_class(self.model, img)
                result = determine_error_type(entry.category, predicted)
                record = EvalRecord(
                    image_num=index,
                    mode="benchmark",
                    condition=self.condition,
                    actual=entry.category,
                    student="",
                    student_result="",
                    predicted=predicted,
                    model_result=result,
                    coverage=actual_coverage,
                )
                records.append(record)
                self.signals.record.emit(record)
                self.signals.image.emit(numpy_rgb_to_qimage(img))
                self.signals.progress.emit(int(100 * index / total))
            self.signals.finished.emit(records)
        except Exception as exc:
            self.signals.error.emit(str(exc))

    def stop(self):
        self.is_running = False


class ChallengeSignals(QObject):
    progress = pyqtSignal(int)
    condition_started = pyqtSignal(str)
    condition_finished = pyqtSignal(object)
    image = pyqtSignal(object)
    finished = pyqtSignal()
    error = pyqtSignal(str)


class ChallengeWorker(QRunnable):
    def __init__(self, model, manifest: List[SurveyEntry]):
        super().__init__()
        self.model = model
        self.manifest = manifest
        self.signals = ChallengeSignals()
        self.is_running = True

    def run(self):
        try:
            total_predictions = len(self.manifest) * len(CHALLENGE_CONDITIONS)
            completed = 0
            for condition in CHALLENGE_CONDITIONS:
                if not self.is_running:
                    break
                self.signals.condition_started.emit(condition)
                records: List[EvalRecord] = []
                representative: Optional[QImage] = None
                for index, entry in enumerate(self.manifest, start=1):
                    if not self.is_running:
                        break
                    img, actual_coverage = make_bloom_image(
                        EVAL_IMAGE_SIZE,
                        entry.coverage,
                        entry.seed,
                        variant=condition,
                    )
                    predicted = predict_class(self.model, img)
                    result = determine_error_type(entry.category, predicted)
                    records.append(
                        EvalRecord(
                            image_num=index,
                            mode="challenge",
                            condition=condition,
                            actual=entry.category,
                            student="",
                            student_result="",
                            predicted=predicted,
                            model_result=result,
                            coverage=actual_coverage,
                        )
                    )
                    representative = numpy_rgb_to_qimage(img)
                    if index == 1 or index % 10 == 0:
                        self.signals.image.emit((condition, representative))
                    completed += 1
                    self.signals.progress.emit(int(100 * completed / total_predictions))
                if representative is not None:
                    self.signals.condition_finished.emit((condition, records, representative))
            self.signals.finished.emit()
        except Exception as exc:
            self.signals.error.emit(str(exc))

    def stop(self):
        self.is_running = False


# ----------------------------
# Main application
# ----------------------------
class CobberEcoBloomApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CobberEcoBloom – Four-Tab Audit")
        self.resize(1580, 900)
        self.setMinimumSize(1320, 780)
        self.setFont(QFont("Lato", 10))

        discovered_models = scan_model_files(APP_ROOT)
        self.available_models = {
            size: discovered_models[size]
            for size in EXPECTED_MODEL_SIZES
            if size in discovered_models
        }
        self.model_cache: Dict[str, object] = {}
        self.threadpool = QThreadPool()
        self.benchmark_worker: Optional[BenchmarkWorker] = None
        self.challenge_worker: Optional[ChallengeWorker] = None
        self.busy = False

        self.model_sizes = [size for size in EXPECTED_MODEL_SIZES if size in self.available_models]
        self.manual_model_size = self.model_sizes[0] if self.model_sizes else ""

        self.manual_records: List[EvalRecord] = []
        self.manual_index = -1
        self.manual_current_image: Optional[np.ndarray] = None
        self.manual_current_entry: Optional[SurveyEntry] = None
        self.manual_prediction_submitted = True

        self.standard_results: Dict[str, List[EvalRecord]] = {}
        self.challenge_results: Dict[str, Dict[str, List[EvalRecord]]] = {}
        self.challenge_images: Dict[str, Dict[str, QImage]] = {}
        self.challenge_active_condition = "standard"
        self.challenge_view_is_matrix = False

        self._build_ui()
        self._populate_model_controls()
        self._set_controls_for_models()

        if not self.available_models:
            QMessageBox.warning(
                self,
                "No models found",
                "No pretrained Keras bloom models were found below the app folder.",
            )

    # ---------- shared model loading ----------
    def get_model(self, size: str):
        if not size:
            return None
        if size in self.model_cache:
            return self.model_cache[size]
        try:
            model_path = self.available_models[size]
            model = load_model(model_path)
            self.model_cache[size] = model
            return model
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Model loading error",
                f"Could not load the {int(size):,}-image model:\n{exc}",
            )
            return None

    # ---------- top-level UI ----------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._tab_changed)
        outer.addWidget(self.tabs)

        self.manual_tab = QWidget()
        self.compare_tab = QWidget()
        self.matrices_tab = QWidget()
        self.challenge_tab = QWidget()

        self.tabs.addTab(self.manual_tab, "Manual Sort")
        self.tabs.addTab(self.compare_tab, "Compare Training Sizes")
        self.tabs.addTab(self.matrices_tab, "Compare Confusion Matrices")
        self.tabs.addTab(self.challenge_tab, "Challenge the Model")

        self._build_manual_tab()
        self._build_compare_tab()
        self._build_matrices_tab()
        self._build_challenge_tab()

    def _populate_model_controls(self) -> None:
        for size in self.model_sizes:
            label = f"{int(size):,} images"
            self.compare_model_combo.addItem(label, size)
            self.challenge_model_combo.addItem(label, size)

        if self.manual_model_size:
            self.manual_model_label.setText(
                f"CNN comparison model: {int(self.manual_model_size):,} training images"
            )
        else:
            self.manual_model_label.setText("CNN comparison model: unavailable")

    def _set_controls_for_models(self) -> None:
        available = bool(self.model_sizes)
        self.manual_show_button.setEnabled(available)
        self.compare_run_button.setEnabled(available)
        self.challenge_run_button.setEnabled(available)

    # ---------- Tab 1: Manual Sort ----------
    def _build_manual_tab(self) -> None:
        layout = QVBoxLayout(self.manual_tab)

        controls = QHBoxLayout()
        self.manual_model_label = bold_label("")
        self.manual_show_button = QPushButton("Show First Image")
        self.manual_guess_label = bold_label("Your prediction:")
        self.manual_guess_combo = QComboBox()
        for category in CATEGORY_ORDER:
            self.manual_guess_combo.addItem(pretty_label(category), category)
        self.manual_submit_button = QPushButton("Submit Prediction")
        self.manual_guess_combo.setEnabled(False)
        self.manual_submit_button.setEnabled(False)

        controls.addWidget(self.manual_model_label)
        controls.addStretch()
        controls.addWidget(self.manual_show_button)
        controls.addWidget(self.manual_guess_label)
        controls.addWidget(self.manual_guess_combo)
        controls.addWidget(self.manual_submit_button)
        layout.addLayout(controls)

        content = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()

        self.manual_image_label = self._make_image_label("Click Show First Image to begin.")
        left.addWidget(self.manual_image_label)
        self.manual_status_label = QLabel(
            "Classify each standard image before the actual class and CNN prediction are revealed."
        )
        self.manual_status_label.setWordWrap(True)
        left.addWidget(self.manual_status_label)

        self.manual_table = QTableWidget()
        configure_table(
            self.manual_table,
            [
                "Image",
                "Manual prediction",
                "Model prediction",
                "Actual class",
                "Manual result",
                "Model result",
            ],
        )
        right.addWidget(bold_label("Manual sorting results"))
        right.addWidget(self.manual_table)

        content.addLayout(left, 1)
        content.addLayout(right, 1)
        layout.addLayout(content)

        cards = QHBoxLayout()
        self.manual_student_card = MetricCard("student", "Your classifications")
        self.manual_model_card = MetricCard("model", "CNN classifications")
        cards.addWidget(self.manual_student_card)
        cards.addWidget(self.manual_model_card)
        layout.addLayout(cards)

        self.manual_show_button.clicked.connect(self._manual_show_next)
        self.manual_submit_button.clicked.connect(self._manual_submit)

    def _manual_show_next(self) -> None:
        if self.busy or not self.manual_model_size:
            return
        model = self.get_model(self.manual_model_size)
        if model is None:
            return

        if self.manual_index >= len(MANUAL_MANIFEST) - 1 and self.manual_prediction_submitted:
            self._manual_reset()

        if not self.manual_prediction_submitted:
            return

        self.manual_index += 1
        self.manual_current_entry = MANUAL_MANIFEST[self.manual_index]
        self.manual_current_image, _ = make_bloom_image(
            EVAL_IMAGE_SIZE,
            self.manual_current_entry.coverage,
            self.manual_current_entry.seed,
            variant="standard",
        )
        self._display_numpy_image(self.manual_image_label, self.manual_current_image)
        self.manual_prediction_submitted = False
        self.manual_guess_combo.setEnabled(True)
        self.manual_submit_button.setEnabled(True)
        self.manual_show_button.setEnabled(False)
        self.manual_status_label.setText(
            f"Image {self.manual_index + 1} of {len(MANUAL_MANIFEST)}. Choose a category, then submit your prediction."
        )

    def _manual_submit(self) -> None:
        if self.manual_prediction_submitted or self.manual_current_image is None or self.manual_current_entry is None:
            return
        model = self.get_model(self.manual_model_size)
        if model is None:
            return
        student_prediction = str(self.manual_guess_combo.currentData())
        model_prediction = predict_class(model, self.manual_current_image)
        student_result = determine_error_type(self.manual_current_entry.category, student_prediction)
        model_result = determine_error_type(self.manual_current_entry.category, model_prediction)

        record = EvalRecord(
            image_num=self.manual_index + 1,
            mode="manual",
            condition="standard",
            actual=self.manual_current_entry.category,
            student=student_prediction,
            student_result=student_result,
            predicted=model_prediction,
            model_result=model_result,
            coverage=self.manual_current_entry.coverage,
        )
        self.manual_records.append(record)
        self._append_manual_row(record)
        self._update_manual_cards()

        self.manual_prediction_submitted = True
        self.manual_guess_combo.setEnabled(False)
        self.manual_submit_button.setEnabled(False)
        self.manual_show_button.setEnabled(True)

        if self.manual_index == len(MANUAL_MANIFEST) - 1:
            self.manual_show_button.setText("Start Again")
            self.manual_status_label.setText(
                "Manual sort complete. Review your results and compare your error pattern with the CNN."
            )
        else:
            self.manual_show_button.setText("Show Next Image")
            self.manual_status_label.setText(
                f"Prediction submitted. The actual class and CNN prediction are now shown in row {self.manual_index + 1}."
            )

    def _manual_reset(self) -> None:
        self.manual_records = []
        self.manual_index = -1
        self.manual_current_image = None
        self.manual_current_entry = None
        self.manual_prediction_submitted = True
        self.manual_table.setRowCount(0)
        self.manual_student_card.reset()
        self.manual_model_card.reset()
        self.manual_image_label.clear()
        self.manual_image_label.setText("Click Show First Image to begin.")
        self.manual_show_button.setText("Show First Image")

    def _append_manual_row(self, record: EvalRecord) -> None:
        row = self.manual_table.rowCount()
        self.manual_table.insertRow(row)
        values = [
            str(record.image_num),
            pretty_label(record.student),
            pretty_label(record.predicted),
            pretty_label(record.actual),
            result_display(record.student_result),
            result_display(record.model_result),
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            self.manual_table.setItem(row, column, item)
        apply_result_color(self.manual_table.item(row, 4), record.student_result)
        apply_result_color(self.manual_table.item(row, 5), record.model_result)
        self.manual_table.scrollToBottom()

    def _update_manual_cards(self) -> None:
        self.manual_student_card.set_metrics(calculate_metrics(self.manual_records, "student_result"))
        self.manual_model_card.set_metrics(calculate_metrics(self.manual_records, "model_result"))

    # ---------- Tab 2: Compare training sizes ----------
    def _build_compare_tab(self) -> None:
        layout = QVBoxLayout(self.compare_tab)
        controls = QHBoxLayout()
        controls.addWidget(bold_label("Training set size:"))
        self.compare_model_combo = QComboBox()
        self.compare_run_button = QPushButton("Run Selected Model")
        controls.addWidget(self.compare_model_combo)
        controls.addWidget(self.compare_run_button)
        controls.addStretch()
        self.compare_completion_label = QLabel("")
        completion_font = self.compare_completion_label.font()
        completion_font.setBold(True)
        completion_font.setPointSize(12)
        self.compare_completion_label.setFont(completion_font)
        controls.addWidget(self.compare_completion_label)
        layout.addLayout(controls)

        content = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()
        self.compare_image_label = self._make_image_label("Run a model to begin testing.")
        left.addWidget(self.compare_image_label)
        self.compare_progress = QProgressBar()
        self.compare_progress.setVisible(False)
        left.addWidget(self.compare_progress)
        self.compare_status_label = QLabel(
            "Every model receives the same 200 standard images: 40 from each category."
        )
        self.compare_status_label.setWordWrap(True)
        left.addWidget(self.compare_status_label)

        self.compare_table = QTableWidget()
        configure_table(
            self.compare_table,
            ["Image", "Model prediction", "Actual class", "Model result"],
        )
        right.addWidget(bold_label("Current testing results"))
        right.addWidget(self.compare_table)
        content.addLayout(left, 1)
        content.addLayout(right, 1)
        layout.addLayout(content)

        card_layout = QHBoxLayout()
        self.compare_cards: Dict[str, MetricCard] = {}
        for size in EXPECTED_MODEL_SIZES:
            card = MetricCard(size, f"{int(size):,} training images")
            self.compare_cards[size] = card
            card_layout.addWidget(card)
        layout.addLayout(card_layout)

        self.compare_run_button.clicked.connect(self._run_compare_benchmark)

    def _run_compare_benchmark(self) -> None:
        if self.busy:
            return
        size = str(self.compare_model_combo.currentData() or "")
        model = self.get_model(size)
        if model is None:
            return

        self.compare_table.setRowCount(0)
        self.compare_completion_label.setText("")
        self.compare_progress.setValue(0)
        self.compare_progress.setVisible(True)
        self.compare_status_label.setText(
            f"Running the {int(size):,}-image CNN on the fixed 200-image standard benchmark."
        )
        self._set_busy(True)

        self.benchmark_worker = BenchmarkWorker(model, BENCHMARK_MANIFEST, "standard")
        self.benchmark_worker.signals.record.connect(self._append_compare_row)
        self.benchmark_worker.signals.image.connect(
            lambda qimage: self._display_qimage(self.compare_image_label, qimage)
        )
        self.benchmark_worker.signals.progress.connect(self.compare_progress.setValue)
        self.benchmark_worker.signals.finished.connect(
            lambda records, selected_size=size: self._finish_compare_benchmark(selected_size, records)
        )
        self.benchmark_worker.signals.error.connect(self._worker_error)
        self.threadpool.start(self.benchmark_worker)

    def _append_compare_row(self, record: EvalRecord) -> None:
        row = self.compare_table.rowCount()
        self.compare_table.insertRow(row)
        values = [
            str(record.image_num),
            pretty_label(record.predicted),
            pretty_label(record.actual),
            result_display(record.model_result),
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            self.compare_table.setItem(row, column, item)
        apply_result_color(self.compare_table.item(row, 3), record.model_result)
        self.compare_table.scrollToBottom()

    def _finish_compare_benchmark(self, size: str, records: List[EvalRecord]) -> None:
        self.standard_results[size] = records
        metrics = calculate_metrics(records)
        if size in self.compare_cards:
            self.compare_cards[size].set_metrics(metrics)
        self.compare_progress.setVisible(False)
        self.compare_completion_label.setText(
            f"Complete: {metrics['accuracy']:.1f}% exact accuracy"
        )
        self.compare_status_label.setText(
            "Testing complete. Results remain in the panel below so they can be compared with the other training sizes."
        )
        self._set_busy(False)
        self._plot_all_confusion_matrices()

    # ---------- Tab 3: confusion matrices ----------
    def _build_matrices_tab(self) -> None:
        layout = QVBoxLayout(self.matrices_tab)
        intro = QLabel(
            "Each matrix uses the same 200-image test set, with 40 images from each actual category."
        )
        intro.setWordWrap(True)
        intro_font = intro.font()
        intro_font.setBold(True)
        intro.setFont(intro_font)
        layout.addWidget(intro)

        self.matrices_canvas = FigureCanvas(Figure(figsize=(15, 5.8), constrained_layout=True))
        layout.addWidget(self.matrices_canvas)
        self.matrices_status_label = QLabel(
            "Run each model on Tab 2. This view updates automatically as results become available."
        )
        self.matrices_status_label.setWordWrap(True)
        layout.addWidget(self.matrices_status_label)
        self._plot_all_confusion_matrices()

    def _plot_all_confusion_matrices(self) -> None:
        if not hasattr(self, "matrices_canvas"):
            return
        fig = self.matrices_canvas.figure
        fig.clear()
        axes = fig.subplots(1, 3)

        for axis, size in zip(axes, EXPECTED_MODEL_SIZES):
            records = self.standard_results.get(size)
            if not records:
                axis.set_axis_off()
                axis.text(
                    0.5,
                    0.5,
                    f"Run the {int(size):,}-image model\non Tab 2",
                    ha="center",
                    va="center",
                    fontsize=11,
                    transform=axis.transAxes,
                )
                continue

            y_true = [record.actual for record in records]
            y_pred = [record.predicted for record in records]
            matrix = confusion_matrix(y_true, y_pred, labels=CATEGORY_ORDER)
            metrics = calculate_metrics(records)

            axis.imshow(matrix, cmap=INFOBLUE_CMAP, vmin=0, vmax=40)
            axis.set_title(
                f"{int(size):,} training images\nExact accuracy: {metrics['accuracy']:.1f}%",
                fontsize=11,
                fontweight="bold",
                pad=12,
            )
            axis.set_xticks(range(len(CATEGORY_ORDER)))
            axis.set_yticks(range(len(CATEGORY_ORDER)))
            axis.set_xticklabels(
                [pretty_label(label) for label in CATEGORY_ORDER],
                rotation=45,
                ha="right",
                fontsize=8,
            )
            axis.set_yticklabels([pretty_label(label) for label in CATEGORY_ORDER], fontsize=8)
            axis.set_xlabel("Predicted class")
            if size == EXPECTED_MODEL_SIZES[0]:
                axis.set_ylabel("Actual class")

            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    value = int(matrix[row, column])
                    text_color = "white" if value >= 24 else "#222222"
                    axis.text(
                        column,
                        row,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=text_color,
                        fontweight="bold" if value >= 24 else "normal",
                    )

        self.matrices_canvas.draw()

    # ---------- Tab 4: Challenge the model ----------
    def _build_challenge_tab(self) -> None:
        layout = QVBoxLayout(self.challenge_tab)
        controls = QHBoxLayout()
        controls.addWidget(bold_label("Model to test:"))
        self.challenge_model_combo = QComboBox()
        controls.addWidget(self.challenge_model_combo)
        self.challenge_run_button = QPushButton("Run All Four Conditions")
        self.challenge_toggle_button = QPushButton("Show Confusion Matrix")
        self.challenge_toggle_button.setEnabled(False)
        controls.addWidget(self.challenge_run_button)
        controls.addWidget(self.challenge_toggle_button)
        controls.addStretch()
        layout.addLayout(controls)

        content = QHBoxLayout()
        left = QVBoxLayout()
        right = QGridLayout()

        self.challenge_stack = QStackedWidget()
        self.challenge_image_label = self._make_image_label(
            "Choose a model and run all four conditions."
        )
        self.challenge_matrix_canvas = FigureCanvas(Figure(figsize=(6, 6)))
        self.challenge_stack.addWidget(self.challenge_image_label)
        self.challenge_stack.addWidget(self.challenge_matrix_canvas)
        left.addWidget(self.challenge_stack)

        self.challenge_progress = QProgressBar()
        self.challenge_progress.setVisible(False)
        left.addWidget(self.challenge_progress)
        self.challenge_status_label = QLabel(
            "The same 200 underlying bloom patterns are used for every condition. Only the visual appearance changes."
        )
        self.challenge_status_label.setWordWrap(True)
        left.addWidget(self.challenge_status_label)

        self.challenge_cards: Dict[str, MetricCard] = {}
        positions = {
            "standard": (0, 0),
            "darker_water": (0, 1),
            "glare": (1, 0),
            "altered_bloom_color": (1, 1),
        }
        for condition, position in positions.items():
            card = MetricCard(condition, CONDITION_LABELS[condition], clickable=True)
            card.clicked.connect(self._select_challenge_condition)
            self.challenge_cards[condition] = card
            right.addWidget(card, *position)

        content.addLayout(left, 1)
        content.addLayout(right, 1)
        layout.addLayout(content)

        self.challenge_run_button.clicked.connect(self._run_challenge_suite)
        self.challenge_toggle_button.clicked.connect(self._toggle_challenge_view)
        self.challenge_model_combo.currentIndexChanged.connect(self._challenge_model_changed)
        self._select_challenge_condition("standard", toggle_if_active=False)

    def _run_challenge_suite(self) -> None:
        if self.busy:
            return
        size = str(self.challenge_model_combo.currentData() or "")
        model = self.get_model(size)
        if model is None:
            return

        self.challenge_results[size] = {}
        self.challenge_images[size] = {}
        for card in self.challenge_cards.values():
            card.reset()
        self.challenge_progress.setValue(0)
        self.challenge_progress.setVisible(True)
        self.challenge_toggle_button.setEnabled(False)
        self.challenge_stack.setCurrentWidget(self.challenge_image_label)
        self.challenge_view_is_matrix = False
        self.challenge_toggle_button.setText("Show Confusion Matrix")
        self.challenge_status_label.setText(
            f"Running the {int(size):,}-image CNN on standard, darker-water, glare, and altered-color versions of the same benchmark."
        )
        self._set_busy(True)

        self.challenge_worker = ChallengeWorker(model, BENCHMARK_MANIFEST)
        self.challenge_worker.signals.progress.connect(self.challenge_progress.setValue)
        self.challenge_worker.signals.condition_started.connect(self._challenge_condition_started)
        self.challenge_worker.signals.image.connect(self._challenge_live_image)
        self.challenge_worker.signals.condition_finished.connect(
            lambda payload, selected_size=size: self._challenge_condition_finished(selected_size, payload)
        )
        self.challenge_worker.signals.finished.connect(self._challenge_suite_finished)
        self.challenge_worker.signals.error.connect(self._worker_error)
        self.threadpool.start(self.challenge_worker)

    def _challenge_condition_started(self, condition: str) -> None:
        self._select_challenge_condition(condition, toggle_if_active=False)
        self.challenge_status_label.setText(
            f"Testing {CONDITION_LABELS[condition].lower()} images."
        )

    def _challenge_live_image(self, payload) -> None:
        condition, qimage = payload
        size = str(self.challenge_model_combo.currentData() or "")
        self.challenge_images.setdefault(size, {})[condition] = qimage
        if condition == self.challenge_active_condition and not self.challenge_view_is_matrix:
            self._display_qimage(self.challenge_image_label, qimage)

    def _challenge_condition_finished(self, size: str, payload) -> None:
        condition, records, representative = payload
        self.challenge_results.setdefault(size, {})[condition] = records
        self.challenge_images.setdefault(size, {})[condition] = representative
        standard_records = self.challenge_results[size].get("standard")
        baseline = calculate_metrics(standard_records)["accuracy"] if standard_records else None
        metrics = calculate_metrics(records)
        delta = None if baseline is None else metrics["accuracy"] - baseline
        self.challenge_cards[condition].set_metrics(metrics, delta=delta)
        if condition == self.challenge_active_condition:
            self._refresh_challenge_display()
        active_records = self.challenge_results[size].get(self.challenge_active_condition)
        self.challenge_toggle_button.setEnabled(bool(active_records))

    def _challenge_suite_finished(self) -> None:
        self.challenge_progress.setVisible(False)
        self.challenge_status_label.setText(
            "Challenge suite complete. Select any panel to inspect its example image or confusion matrix."
        )
        self._set_busy(False)
        self._refresh_challenge_cards_for_model()
        self._select_challenge_condition("standard", toggle_if_active=False)

    def _challenge_model_changed(self) -> None:
        if not hasattr(self, "challenge_cards"):
            return
        self._refresh_challenge_cards_for_model()

    def _refresh_challenge_cards_for_model(self) -> None:
        size = str(self.challenge_model_combo.currentData() or "")
        model_results = self.challenge_results.get(size, {})
        baseline_records = model_results.get("standard")
        baseline = calculate_metrics(baseline_records)["accuracy"] if baseline_records else None
        for condition, card in self.challenge_cards.items():
            records = model_results.get(condition)
            if not records:
                card.reset()
                continue
            metrics = calculate_metrics(records)
            delta = None if baseline is None else metrics["accuracy"] - baseline
            card.set_metrics(metrics, delta=delta)
        active_records = model_results.get(self.challenge_active_condition)
        self.challenge_toggle_button.setEnabled(bool(active_records))
        self._refresh_challenge_display()

    def _select_challenge_condition(
        self,
        condition: str,
        toggle_if_active: bool = True,
    ) -> None:
        same_condition = condition == self.challenge_active_condition
        self.challenge_active_condition = condition

        for key, card in self.challenge_cards.items():
            card.set_active(key == condition)

        size = str(self.challenge_model_combo.currentData() or "")
        has_results = bool(self.challenge_results.get(size, {}).get(condition))

        # Clicking the already active panel toggles its image and matrix.
        if toggle_if_active and same_condition and has_results:
            self.challenge_view_is_matrix = not self.challenge_view_is_matrix
            self.challenge_toggle_button.setText(
                "Show Image" if self.challenge_view_is_matrix else "Show Confusion Matrix"
            )

        self.challenge_toggle_button.setEnabled(has_results)
        self._refresh_challenge_display()

    def _toggle_challenge_view(self) -> None:
        self.challenge_view_is_matrix = not self.challenge_view_is_matrix
        if self.challenge_view_is_matrix:
            self.challenge_toggle_button.setText("Show Image")
            self.challenge_stack.setCurrentWidget(self.challenge_matrix_canvas)
        else:
            self.challenge_toggle_button.setText("Show Confusion Matrix")
            self.challenge_stack.setCurrentWidget(self.challenge_image_label)
        self._refresh_challenge_display()

    def _refresh_challenge_display(self) -> None:
        if not hasattr(self, "challenge_stack"):
            return
        size = str(self.challenge_model_combo.currentData() or "")
        condition = self.challenge_active_condition
        if self.challenge_view_is_matrix:
            records = self.challenge_results.get(size, {}).get(condition)
            if records:
                self._plot_challenge_matrix(records, condition)
            return
        qimage = self.challenge_images.get(size, {}).get(condition)
        self.challenge_stack.setCurrentWidget(self.challenge_image_label)
        if qimage is not None:
            self._display_qimage(self.challenge_image_label, qimage)
        else:
            self.challenge_image_label.clear()
            self.challenge_image_label.setText(
                f"Run all four conditions to view a {CONDITION_LABELS[condition].lower()} example."
            )

    def _plot_challenge_matrix(self, records: List[EvalRecord], condition: str) -> None:
        y_true = [record.actual for record in records]
        y_pred = [record.predicted for record in records]
        matrix = confusion_matrix(y_true, y_pred, labels=CATEGORY_ORDER)
        fig = self.challenge_matrix_canvas.figure
        fig.clear()
        axis = fig.add_subplot(111)
        axis.imshow(matrix, cmap=INFOBLUE_CMAP, vmin=0, vmax=40)
        axis.set_title(CONDITION_LABELS[condition], fontsize=14, fontweight="bold")
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class")
        axis.set_xticks(range(len(CATEGORY_ORDER)))
        axis.set_yticks(range(len(CATEGORY_ORDER)))
        axis.set_xticklabels(
            [pretty_label(label) for label in CATEGORY_ORDER], rotation=45, ha="right", fontsize=8
        )
        axis.set_yticklabels([pretty_label(label) for label in CATEGORY_ORDER], fontsize=8)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = int(matrix[row, column])
                axis.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if value >= 24 else "#222222",
                    fontweight="bold" if value >= 24 else "normal",
                )
        fig.tight_layout()
        self.challenge_matrix_canvas.draw()
        self.challenge_stack.setCurrentWidget(self.challenge_matrix_canvas)

    # ---------- shared UI helpers ----------
    def _make_image_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(560, 500)
        label.setStyleSheet(
            "QLabel { border: 2px solid #B5B5B5; background-color: #F5F5F5; color: #222222; }"
        )
        return label

    def _display_numpy_image(self, label: QLabel, img_rgb: np.ndarray) -> None:
        self._display_qimage(label, numpy_rgb_to_qimage(img_rgb))

    def _display_qimage(self, label: QLabel, qimage: QImage) -> None:
        target_width = max(200, label.width() - 8)
        target_height = max(200, label.height() - 8)
        label.setPixmap(
            QPixmap.fromImage(qimage).scaled(
                target_width,
                target_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        enabled = not busy and bool(self.model_sizes)
        self.compare_run_button.setEnabled(enabled)
        self.compare_model_combo.setEnabled(not busy)
        self.challenge_run_button.setEnabled(enabled)
        self.challenge_model_combo.setEnabled(not busy)
        if busy:
            self.manual_show_button.setEnabled(False)
            self.manual_submit_button.setEnabled(False)
        else:
            if self.manual_prediction_submitted:
                self.manual_show_button.setEnabled(bool(self.model_sizes))
            else:
                self.manual_submit_button.setEnabled(True)

    def _worker_error(self, message: str) -> None:
        self.compare_progress.setVisible(False)
        self.challenge_progress.setVisible(False)
        self._set_busy(False)
        QMessageBox.critical(self, "Processing error", message)

    def _tab_changed(self, index: int) -> None:
        if index == 2:
            self._plot_all_confusion_matrices()

    def closeEvent(self, event) -> None:
        if self.benchmark_worker is not None:
            self.benchmark_worker.stop()
        if self.challenge_worker is not None:
            self.challenge_worker.stop()
        self.threadpool.waitForDone()
        event.accept()


# ----------------------------
# App styling
# ----------------------------
def apply_app_stylesheet(app: QApplication) -> None:
    app.setStyleSheet(
        """
        QWidget {
            color: #222222;
            background-color: #FFFFFF;
        }
        QMainWindow, QDialog {
            background-color: #FFFFFF;
        }
        QLabel {
            color: #222222;
            background-color: transparent;
        }
        QTabWidget::pane {
            border: 1px solid #8F8F8F;
            top: -1px;
        }
        QTabBar::tab {
            background-color: #5A5A5A;
            color: #FFFFFF;
            font-weight: normal;
            padding: 10px 24px;
            border: 1px solid #474747;
            min-width: 190px;
        }
        QTabBar::tab:selected {
            background-color: #6C1D45;
            color: #FFFFFF;
            font-weight: bold;
        }
        QTabBar::tab:hover {
            background-color: #5A5A5A;
            color: #FFFFFF;
        }
        QTabBar::tab:selected:hover {
            background-color: #6C1D45;
        }
        QPushButton {
            background-color: #6C1D45;
            color: #FFFFFF;
            font-weight: bold;
            border: 1px solid #541636;
            border-radius: 4px;
            padding: 7px 12px;
        }
        QPushButton:hover {
            background-color: #6C1D45;
            color: #FFFFFF;
        }
        QPushButton:pressed {
            background-color: #541636;
            color: #FFFFFF;
        }
        QPushButton:disabled {
            background-color: #5A5A5A;
            color: #FFFFFF;
            border-color: #4A4A4A;
        }
        QComboBox {
            background-color: #FFFFFF;
            color: #111111;
            border: 1px solid #8C8C8C;
            border-radius: 3px;
            padding: 5px 8px;
            min-width: 180px;
        }
        QTableWidget {
            background-color: #FFFFFF;
            alternate-background-color: #F7F7F7;
            color: #111111;
            border: 1px solid #A0A0A0;
            gridline-color: #D7D7D7;
        }
        QHeaderView::section {
            background-color: #E5E5E5;
            color: #111111;
            border: 1px solid #C8C8C8;
            padding: 6px;
            font-weight: bold;
        }
        QHeaderView::section:hover {
            background-color: #E5E5E5;
            color: #111111;
        }
        QProgressBar {
            border: 1px solid #8C8C8C;
            border-radius: 4px;
            text-align: center;
            background-color: #FFFFFF;
            min-height: 20px;
        }
        QProgressBar::chunk {
            background-color: #6C1D45;
        }
        """
    )


if __name__ == "__main__":
    application = QApplication(sys.argv)
    apply_app_stylesheet(application)
    window = CobberEcoBloomApp()
    window.show()
    sys.exit(application.exec())

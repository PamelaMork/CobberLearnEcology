#!/usr/bin/env python3
"""
CobberEcoLSTMApplied_v6.py

Windows-first guided application for the applied LSTM chapter in
Foundations of Machine Learning for Ecology.

Place this script beside:
    SmallFish_LSTM_TeachingDataset.xlsx

Required packages:
    pip install PyQt6 pandas numpy matplotlib openpyxl tensorflow

The first three tabs remain usable if TensorFlow is unavailable. LSTM training
is disabled with a clear message rather than crashing the program.
"""

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Conservative settings for classroom Windows laptops.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QBrush
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

APP_TITLE = "CobberEcoLSTM Applied"
DATA_FILE = "SmallFish_LSTM_TeachingDataset.xlsx"
DATA_SHEET = "FishData"

COBBER_MAROON = "#6c1d45"
INFO_BLUE = "#3e6990"
CHARCOAL = "#222222"
LIGHT_GRAY = "#f4f4f4"
MEDIUM_GRAY = "#666666"

WINDOW_LENGTH = 4
RESTORATION_DATE = pd.Timestamp("2021-01-15")


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def find_data_file() -> Optional[Path]:
    candidate = app_dir() / DATA_FILE
    return candidate if candidate.exists() else None


def fmt(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{float(value):.{digits}f}"


def table_item(value: object, bold: bool = False, maroon: bool = False) -> QTableWidgetItem:
    item = QTableWidgetItem(str(value))
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    if bold:
        font = item.font()
        font.setBold(True)
        item.setFont(font)
    if maroon:
        item.setForeground(QColor(COBBER_MAROON))
    return item


class PlotCanvas(FigureCanvas):
    def __init__(self, width: float = 6.5, height: float = 4.5):
        self.figure = Figure(figsize=(width, height), dpi=100)
        super().__init__(self.figure)

    def message(self, text: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True)
        ax.set_axis_off()
        self.figure.tight_layout()
        self.draw()


@dataclass
class WindowRecord:
    site: str
    plot: int
    site_type: str
    target_date: pd.Timestamp
    target_period: str
    input_dates: List[pd.Timestamp]
    input_values: np.ndarray
    target_value: float


@dataclass
class SequenceData:
    windows: List[WindowRecord]
    X: np.ndarray
    y: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float
    X_scaled: np.ndarray
    y_scaled: np.ndarray


@dataclass
class PredictionResult:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    records: List[WindowRecord]

    @property
    def mae(self) -> float:
        return float(np.mean(np.abs(self.y_pred - self.y_true)))

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean((self.y_pred - self.y_true) ** 2)))


@dataclass
class LSTMResult:
    history: Optional[object]
    validation_result: Optional[PredictionResult]
    test_result: Optional[PredictionResult]
    error: str = ""


def read_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=DATA_SHEET, parse_dates=["SampleDate"])
    df.columns = [str(col).strip() for col in df.columns]

    required = {
        "SampleDate",
        "Year",
        "Month",
        "Season",
        "DPM",
        "SiteType",
        "Site",
        "Plot",
        "WaterDepth_cm",
    }
    species = [f"Species{i}" for i in range(1, 21)]
    required.update(species)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"The dataset is missing expected columns: {missing}")

    df["SampleDate"] = pd.to_datetime(df["SampleDate"], errors="raise")

    # Assign the Everglades wet and dry seasons directly from the sampling month.
    # Wet season: May through October
    # Dry season: November through April
    df["Season"] = np.where(
        df["SampleDate"].dt.month.between(5, 10),
        "Wet",
        "Dry",
    )

    df["Plot"] = pd.to_numeric(df["Plot"], errors="raise").astype(int)
    df["WaterDepth_cm"] = pd.to_numeric(df["WaterDepth_cm"], errors="raise")
    for col in species:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df["TotalDensity"] = df[species].sum(axis=1).astype(float)
    df = df.sort_values(["Site", "Plot", "SampleDate"]).reset_index(drop=True)

    # Verify that each plot has unique, increasing dates.
    duplicated = df.duplicated(["Site", "Plot", "SampleDate"]).any()
    if duplicated:
        raise ValueError("At least one site and plot has duplicate sampling dates.")

    for (_, _), group in df.groupby(["Site", "Plot"], sort=False):
        if not group["SampleDate"].is_monotonic_increasing:
            raise ValueError("Sampling dates are not chronological within every plot.")

    return df


def build_sequence_data(df: pd.DataFrame, include_water_depth: bool = False) -> SequenceData:
    windows: List[WindowRecord] = []
    X_rows: List[np.ndarray] = []
    y_rows: List[float] = []

    for (site, plot), group in df.groupby(["Site", "Plot"], sort=True):
        group = group.sort_values("SampleDate").reset_index(drop=True)
        if len(group) <= WINDOW_LENGTH + 2:
            continue

        for target_pos in range(WINDOW_LENGTH, len(group)):
            history = group.iloc[target_pos - WINDOW_LENGTH: target_pos]
            target = group.iloc[target_pos]
            density = history["TotalDensity"].to_numpy(dtype=float)
            depth = history["WaterDepth_cm"].to_numpy(dtype=float)

            if include_water_depth:
                features = np.column_stack([density, depth])
            else:
                features = density.reshape(-1, 1)

            record = WindowRecord(
                site=str(site),
                plot=int(plot),
                site_type=str(target["SiteType"]),
                target_date=pd.Timestamp(target["SampleDate"]),
                target_period=str(target["DPM"]),
                input_dates=[pd.Timestamp(value) for value in history["SampleDate"].tolist()],
                input_values=density.copy(),
                target_value=float(target["TotalDensity"]),
            )
            windows.append(record)
            X_rows.append(features)
            y_rows.append(record.target_value)

    if not windows:
        raise ValueError("No four-sample sequence windows could be created.")

    X = np.stack(X_rows).astype(float)
    y = np.asarray(y_rows, dtype=float).reshape(-1, 1)

    # Time-aware split within each plot:
    # final window = untouched test; previous window = validation; all earlier = training.
    by_plot: Dict[Tuple[str, int], List[int]] = {}
    for idx, record in enumerate(windows):
        by_plot.setdefault((record.site, record.plot), []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for indices in by_plot.values():
        indices = sorted(indices, key=lambda i: windows[i].target_date)
        if len(indices) < 3:
            train_idx.extend(indices)
            continue
        train_idx.extend(indices[:-2])
        val_idx.append(indices[-2])
        test_idx.append(indices[-1])

    train_idx_arr = np.asarray(train_idx, dtype=int)
    val_idx_arr = np.asarray(val_idx, dtype=int)
    test_idx_arr = np.asarray(test_idx, dtype=int)

    if len(train_idx_arr) == 0 or len(val_idx_arr) == 0 or len(test_idx_arr) == 0:
        raise ValueError("The dataset does not provide enough windows for training, validation, and testing.")

    x_flat_train = X[train_idx_arr].reshape(-1, X.shape[-1])
    x_mean = x_flat_train.mean(axis=0)
    x_scale = x_flat_train.std(axis=0)
    x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
    X_scaled = (X - x_mean.reshape(1, 1, -1)) / x_scale.reshape(1, 1, -1)

    y_mean = float(y[train_idx_arr].mean())
    y_scale = float(y[train_idx_arr].std())
    if y_scale <= 1e-12:
        y_scale = 1.0
    y_scaled = (y - y_mean) / y_scale

    return SequenceData(
        windows=windows,
        X=X,
        y=y,
        train_idx=train_idx_arr,
        val_idx=val_idx_arr,
        test_idx=test_idx_arr,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        X_scaled=X_scaled,
        y_scaled=y_scaled,
    )


def baseline_predictions(data: SequenceData, indices: np.ndarray) -> Dict[str, PredictionResult]:
    y_true = data.y[indices, 0]
    records = [data.windows[int(i)] for i in indices]

    training_mean = float(data.y[data.train_idx, 0].mean())
    mean_pred = np.full_like(y_true, training_mean, dtype=float)
    persistence_pred = data.X[indices, -1, 0].astype(float)

    return {
        "Long-term average": PredictionResult("Long-term average", y_true, mean_pred, records),
        "Most recent sample": PredictionResult("Most recent sample", y_true, persistence_pred, records),
    }


def train_lstm_model(data: SequenceData, epochs: int) -> LSTMResult:
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception as exc:
        return LSTMResult(None, None, None, f"TensorFlow could not be imported. Details: {exc}")

    try:
        tf.random.set_seed(42)
        np.random.seed(42)
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception:
            pass

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(data.X_scaled.shape[1], data.X_scaled.shape[2])),
                keras.layers.LSTM(16),
                keras.layers.Dense(1),
            ]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.005),
            loss="mse",
            metrics=["mae"],
        )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )

        history = model.fit(
            data.X_scaled[data.train_idx],
            data.y_scaled[data.train_idx],
            validation_data=(data.X_scaled[data.val_idx], data.y_scaled[data.val_idx]),
            epochs=epochs,
            batch_size=16,
            verbose=0,
            shuffle=False,
            callbacks=[early_stopping],
        )

        def predict_for(indices: np.ndarray, name: str) -> PredictionResult:
            pred_scaled = model.predict(data.X_scaled[indices], verbose=0).reshape(-1)
            pred = pred_scaled * data.y_scale + data.y_mean
            pred = np.maximum(pred, 0.0)
            true = data.y[indices, 0]
            records = [data.windows[int(i)] for i in indices]
            return PredictionResult(name, true, pred, records)

        validation = predict_for(data.val_idx, "LSTM validation")
        test = predict_for(data.test_idx, "LSTM")
        return LSTMResult(history, validation, test, "")
    except Exception:
        return LSTMResult(None, None, None, traceback.format_exc())


class AppliedWindowWidget(QWidget):
    """Draw one four-sample input window and its next-value target."""

    def __init__(self) -> None:
        super().__init__()
        self.record: Optional[WindowRecord] = None
        self.setMinimumHeight(220)

    def set_record(self, record: Optional[WindowRecord]) -> None:
        self.record = record
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        if self.record is None:
            painter.setPen(QColor(CHARCOAL))
            painter.setFont(QFont("Lato", 12))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Click Build Four-Sample Windows to reveal one input window and its target.",
            )
            return

        values = [*self.record.input_values.tolist(), self.record.target_value]
        dates = [*self.record.input_dates, self.record.target_date]

        width = self.width()
        tile_gap = 18
        tile_width = min(125.0, (width - 110 - tile_gap * 4) / 5)
        tile_height = 64.0
        total_width = tile_width * 5 + tile_gap * 4
        start_x = (width - total_width) / 2
        tile_y = 72.0

        tile_rects = []
        for index, (value, date) in enumerate(zip(values, dates)):
            x = start_x + index * (tile_width + tile_gap)
            rect = QRectF(x, tile_y, tile_width, tile_height)
            tile_rects.append(rect)

            painter.setBrush(QBrush(QColor(LIGHT_GRAY)))
            painter.setPen(QPen(QColor("#c9c9c9"), 1))
            painter.drawRoundedRect(rect, 8, 8)

            painter.setPen(QColor("#555555"))
            painter.setFont(QFont("Lato", 9))
            painter.drawText(
                QRectF(x - 5, tile_y - 29, tile_width + 10, 20),
                Qt.AlignmentFlag.AlignCenter,
                date.strftime("%Y-%m-%d"),
            )

            value_font = QFont("Lato", 15)
            value_font.setBold(True)
            painter.setFont(value_font)
            painter.setPen(QColor(CHARCOAL))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, fmt(value, 0))

        first = tile_rects[0]
        fourth = tile_rects[3]
        input_outline = QRectF(
            first.left() - 9,
            first.top() - 9,
            fourth.right() - first.left() + 18,
            tile_height + 18,
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(COBBER_MAROON), 5))
        painter.drawRoundedRect(input_outline, 12, 12)

        target_rect = tile_rects[4]
        circle_size = max(tile_width, tile_height) + 13
        target_circle = QRectF(
            target_rect.center().x() - circle_size / 2,
            target_rect.center().y() - circle_size / 2,
            circle_size,
            circle_size,
        )
        painter.setPen(QPen(QColor(INFO_BLUE), 5))
        painter.drawEllipse(target_circle)

        label_font = QFont("Lato", 11)
        label_font.setBold(True)
        painter.setFont(label_font)
        painter.setPen(QColor(COBBER_MAROON))
        painter.drawText(
            QRectF(input_outline.left(), input_outline.bottom() + 14, input_outline.width(), 24),
            Qt.AlignmentFlag.AlignCenter,
            "four-sample input window",
        )
        painter.setPen(QColor(INFO_BLUE))
        painter.drawText(
            QRectF(target_circle.left() - 15, target_circle.bottom() + 14, target_circle.width() + 30, 24),
            Qt.AlignmentFlag.AlignCenter,
            "target",
        )


class BaseTab(QWidget):
    def __init__(self, app: "CobberEcoLSTMApp"):
        super().__init__()
        self.app = app

    def side_panel(self) -> Tuple[QFrame, QVBoxLayout]:
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setMinimumWidth(295)
        panel.setMaximumWidth(335)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(11)
        return panel, layout

    @staticmethod
    def question_box(text: str) -> QGroupBox:
        box = QGroupBox("Question")
        layout = QVBoxLayout(box)
        label = QLabel(text)
        label.setWordWrap(True)
        layout.addWidget(label)
        return box


class ExploreStudyTab(BaseTab):
    def __init__(self, app: "CobberEcoLSTMApp"):
        super().__init__(app)
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        side, side_layout = self.side_panel()
        side_layout.addWidget(self.question_box("How does one fish community change across repeated sampling visits?"))

        choose = QGroupBox("Choose one plot")
        choose_layout = QVBoxLayout(choose)
        choose_layout.addWidget(QLabel("Site:"))
        self.site_combo = QComboBox()
        self.site_combo.addItems(sorted(app.df["Site"].astype(str).unique()))
        choose_layout.addWidget(self.site_combo)
        choose_layout.addWidget(QLabel("Plot:"))
        self.plot_combo = QComboBox()
        choose_layout.addWidget(self.plot_combo)
        side_layout.addWidget(choose)

        notice = QGroupBox("Look for a pattern")
        notice_layout = QVBoxLayout(notice)

        notice_text = QLabel(
            "Follow total fish density from left to right.\n\n"
            "Do high and low values appear at similar times of year?\n\n"
            "Do neighboring samples usually resemble each other?\n\n"
            "Does the typical density or range of values change after restoration?"
        )
        notice_text.setWordWrap(True)

        notice_layout.addWidget(notice_text)
        side_layout.addWidget(notice)
        side_layout.addStretch()

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        header = QLabel("Explore the Restoration Study")
        header.setObjectName("MainHeader")
        main_layout.addWidget(header)
        intro = QLabel(
            "Dr. Rahman's team returned to each plot every two months. The dates below are read directly from the spreadsheet and sorted within each plot."
        )
        intro.setObjectName("IntroText")
        intro.setWordWrap(True)
        main_layout.addWidget(intro)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(105)
        main_layout.addWidget(self.summary)

        self.plot_canvas = PlotCanvas(width=8.5, height=4.7)
        main_layout.addWidget(self.plot_canvas, stretch=1)

        self.table = QTableWidget()
        self.table.setMaximumHeight(205)
        main_layout.addWidget(self.table)

        root.addWidget(side)
        root.addWidget(main, stretch=1)

        self.site_combo.currentTextChanged.connect(self.refresh_plots)
        self.plot_combo.currentTextChanged.connect(self.refresh)
        self.refresh_plots()

    def refresh_plots(self) -> None:
        site = self.site_combo.currentText()
        plots = sorted(self.app.df.loc[self.app.df["Site"] == site, "Plot"].unique())
        self.plot_combo.blockSignals(True)
        self.plot_combo.clear()
        self.plot_combo.addItems([str(int(p)) for p in plots])
        self.plot_combo.blockSignals(False)
        self.refresh()

    def refresh(self) -> None:
        site = self.site_combo.currentText()
        if not site or not self.plot_combo.currentText():
            return
        plot = int(self.plot_combo.currentText())
        sub = self.app.df[(self.app.df["Site"] == site) & (self.app.df["Plot"] == plot)].copy()
        sub = sub.sort_values("SampleDate")
        site_type = str(sub["SiteType"].iloc[0])
        before_n = int((sub["DPM"] == "Before").sum())
        after_n = int((sub["DPM"] == "After").sum())
        self.summary.setHtml(
            f"<b>Selected sequence:</b> {site}, Plot {plot}<br>"
            f"<b>Study role:</b> {site_type} site &nbsp;&nbsp; "
            f"<b>Sampling events:</b> {len(sub)} &nbsp;&nbsp; "
            f"<b>Before restoration:</b> {before_n} &nbsp;&nbsp; "
            f"<b>After restoration:</b> {after_n}"
        )

        self.plot_canvas.figure.clear()
        ax = self.plot_canvas.figure.add_subplot(111)
        before = sub[sub["DPM"] == "Before"]
        after = sub[sub["DPM"] == "After"]
        ax.plot(sub["SampleDate"], sub["TotalDensity"], linewidth=1.5, alpha=0.55)
        ax.scatter(before["SampleDate"], before["TotalDensity"], label="Before restoration", s=38)
        ax.scatter(after["SampleDate"], after["TotalDensity"], label="After restoration", s=38)
        ax.axvline(RESTORATION_DATE, linestyle="--", linewidth=1.2, label="Restoration begins")
        ax.set_title(f"Total fish density through time: {site}, Plot {plot}")
        ax.set_xlabel("Sampling date")
        ax.set_ylabel("Total fish density")
        ax.legend(fontsize=8)
        self.plot_canvas.figure.autofmt_xdate()
        self.plot_canvas.figure.tight_layout()
        self.plot_canvas.draw()

        preview = sub[["SampleDate", "DPM", "Season", "WaterDepth_cm", "TotalDensity"]].copy()

        self.table.clear()
        self.table.setRowCount(len(preview))
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Date", "Period", "Season", "Water depth (cm)", "Total density"]
        )

        for r, (_, row) in enumerate(preview.iterrows()):
            vals = [
                row["SampleDate"].strftime("%Y-%m-%d"),
                row["DPM"],
                row["Season"],
                fmt(row["WaterDepth_cm"], 1),
                fmt(row["TotalDensity"], 0),
            ]
            for c, value in enumerate(vals):
                self.table.setItem(r, c, table_item(value))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.scrollToTop()


class BuildWindowsTab(BaseTab):
    def __init__(self, app: "CobberEcoLSTMApp"):
        super().__init__(app)
        self.windows_revealed = False

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        side, side_layout = self.side_panel()
        side_layout.addWidget(
            self.question_box("How do four consecutive sampling events become one prediction example?")
        )

        choose = QGroupBox("Choose an example")
        choose_layout = QVBoxLayout(choose)
        choose_layout.addWidget(QLabel("Site:"))
        self.site_combo = QComboBox()
        self.site_combo.addItems(sorted(app.df["Site"].astype(str).unique()))
        choose_layout.addWidget(self.site_combo)

        choose_layout.addWidget(QLabel("Plot:"))
        self.plot_combo = QComboBox()
        choose_layout.addWidget(self.plot_combo)

        choose_layout.addWidget(QLabel("Target date:"))
        self.window_combo = QComboBox()
        choose_layout.addWidget(self.window_combo)
        side_layout.addWidget(choose)

        build_box = QGroupBox("Build windows")
        build_layout = QVBoxLayout(build_box)
        self.build_button = QPushButton("Reveal Four-Sample Windows")
        self.build_button.setObjectName("PrimaryButton")
        self.build_button.clicked.connect(self.build)
        build_layout.addWidget(self.build_button)

        nav_layout = QHBoxLayout()
        self.previous_button = QPushButton("Previous Window")
        self.next_button = QPushButton("Next Window")
        self.previous_button.setObjectName("StepButton")
        self.next_button.setObjectName("StepButton")
        self.previous_button.clicked.connect(self.previous_window)
        self.next_button.clicked.connect(self.next_window)
        nav_layout.addWidget(self.previous_button)
        nav_layout.addWidget(self.next_button)
        build_layout.addLayout(nav_layout)

        side_layout.addWidget(build_box)
        side_layout.addStretch()

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Build Four-Sample Windows")
        header.setObjectName("MainHeader")
        main_layout.addWidget(header)

        intro = QLabel(
            "Each window contains four consecutive total-density measurements from one plot. "
            "The following measurement becomes the target. Windows never cross from one plot into another."
        )
        intro.setObjectName("IntroText")
        intro.setWordWrap(True)
        main_layout.addWidget(intro)

        visual_box = QGroupBox("One four-sample window")
        visual_layout = QVBoxLayout(visual_box)

        self.window_table = QTableWidget()
        self.window_table.setFixedHeight(215)
        visual_layout.addWidget(self.window_table)

        self.window_graphic = AppliedWindowWidget()
        self.window_graphic.setMinimumHeight(330)
        visual_layout.addWidget(self.window_graphic, stretch=1)

        main_layout.addWidget(visual_box, stretch=1)

        root.addWidget(side)
        root.addWidget(main, stretch=1)

        self.site_combo.currentTextChanged.connect(self.refresh_plots)
        self.plot_combo.currentTextChanged.connect(self.refresh_windows)
        self.window_combo.currentIndexChanged.connect(self.show_window)

        self.refresh_plots()

    def refresh_plots(self) -> None:
        site = self.site_combo.currentText()
        plots = sorted(self.app.df.loc[self.app.df["Site"] == site, "Plot"].unique())
        self.plot_combo.blockSignals(True)
        self.plot_combo.clear()
        self.plot_combo.addItems([str(int(p)) for p in plots])
        self.plot_combo.blockSignals(False)
        self.refresh_windows()

    def refresh_windows(self) -> None:
        self.window_combo.blockSignals(True)
        self.window_combo.clear()

        if self.plot_combo.currentText():
            site = self.site_combo.currentText()
            plot = int(self.plot_combo.currentText())
            matches = [
                (idx, rec)
                for idx, rec in enumerate(self.app.sequence_data.windows)
                if rec.site == site and rec.plot == plot
            ]
            for idx, rec in matches:
                self.window_combo.addItem(
                    f"Predict {rec.target_date.strftime('%Y-%m-%d')}",
                    idx,
                )

        self.window_combo.blockSignals(False)
        self.show_window()

    def build(self) -> None:
        self.windows_revealed = True
        self.show_window()
        self.refresh_navigation_buttons()

    def previous_window(self) -> None:
        current = self.window_combo.currentIndex()
        if current > 0:
            self.window_combo.setCurrentIndex(current - 1)

    def next_window(self) -> None:
        current = self.window_combo.currentIndex()
        if 0 <= current < self.window_combo.count() - 1:
            self.window_combo.setCurrentIndex(current + 1)

    def refresh_navigation_buttons(self) -> None:
        current = self.window_combo.currentIndex()
        count = self.window_combo.count()
        self.previous_button.setEnabled(
            self.windows_revealed and current > 0
        )
        self.next_button.setEnabled(
            self.windows_revealed and 0 <= current < count - 1
        )

    def show_window(self) -> None:
        self.window_table.clear()
        self.window_table.setColumnCount(3)
        self.window_table.setHorizontalHeaderLabels(
            ["Position", "Date", "Total density"]
        )
        self.window_table.verticalHeader().setVisible(False)

        if self.window_combo.currentIndex() < 0:
            self.window_table.setRowCount(0)
            self.window_graphic.set_record(None)
            self.refresh_navigation_buttons()
            return

        idx = self.window_combo.currentData()
        if idx is None:
            self.window_table.setRowCount(0)
            self.window_graphic.set_record(None)
            self.refresh_navigation_buttons()
            return

        rec = self.app.sequence_data.windows[int(idx)]
        self.window_table.setRowCount(WINDOW_LENGTH + 1)

        for step in range(WINDOW_LENGTH):
            values = [
                f"Input {step + 1}",
                rec.input_dates[step].strftime("%Y-%m-%d"),
                fmt(rec.input_values[step], 0),
            ]
            for col, value in enumerate(values):
                self.window_table.setItem(
                    step,
                    col,
                    table_item(value, bold=(col == 0)),
                )

        target_values = [
            "Target",
            rec.target_date.strftime("%Y-%m-%d"),
            fmt(rec.target_value, 0),
        ]
        for col, value in enumerate(target_values):
            self.window_table.setItem(
                WINDOW_LENGTH,
                col,
                table_item(value, bold=True, maroon=True),
            )

        self.window_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        self.window_graphic.set_record(rec if self.windows_revealed else None)
        self.refresh_navigation_buttons()


class BaselinesTab(BaseTab):
    def __init__(self, app: "CobberEcoLSTMApp"):
        super().__init__(app)
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        side, side_layout = self.side_panel()
        side_layout.addWidget(self.question_box("Which simple prediction should be harder for the LSTM to beat?"))

        predict_box = QGroupBox("Make a prediction")
        predict_layout = QVBoxLayout(predict_box)
        predict_layout.addWidget(
            QLabel("Before revealing the metrics, choose the baseline you expect to perform better."))
        self.expect_combo = QComboBox()
        self.expect_combo.addItems(["Long-term average", "Most recent sample"])
        predict_layout.addWidget(self.expect_combo)
        self.lock_button = QPushButton("Lock My Prediction")
        self.lock_button.setObjectName("StepButton")
        self.lock_button.clicked.connect(self.lock_prediction)
        predict_layout.addWidget(self.lock_button)
        side_layout.addWidget(predict_box)

        run_box = QGroupBox("Test the baselines")
        run_layout = QVBoxLayout(run_box)
        self.run_button = QPushButton("Reveal Baseline Results")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.reveal)
        run_layout.addWidget(self.run_button)
        side_layout.addWidget(run_box)
        side_layout.addStretch()

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        header = QLabel("Compare the Simple Predictions")
        header.setObjectName("MainHeader")
        main_layout.addWidget(header)
        intro = QLabel(
            "The long-term average ignores the latest conditions. The most-recent-sample baseline assumes the next fish community will resemble the current one."
        )
        intro.setObjectName("IntroText")
        intro.setWordWrap(True)
        main_layout.addWidget(intro)

        self.prediction_note = QLabel(
            "Use the graph to decide which baseline you expect to perform better, then lock your prediction.")
        self.prediction_note.setObjectName("StatusLabel")
        self.prediction_note.setWordWrap(True)
        main_layout.addWidget(self.prediction_note)

        self.metrics_table = QTableWidget()
        self.metrics_table.setMaximumHeight(180)
        main_layout.addWidget(self.metrics_table)

        self.plot_canvas = PlotCanvas(width=8.0, height=4.7)
        main_layout.addWidget(self.plot_canvas, stretch=1)

        self.validation_note = QLabel(
            "The revealed metrics will use 30 validation windows, one from each plot. "
            "Lower MAE and RMSE indicate better predictions."
        )
        self.validation_note.setWordWrap(True)
        self.validation_note.setVisible(False)
        main_layout.addWidget(self.validation_note)

        root.addWidget(side)
        root.addWidget(main, stretch=1)
        self.revealed = False
        self.show_prediction_evidence()

    def reset_for_new_data(self) -> None:
        self.revealed = False
        self.run_button.setEnabled(False)
        self.lock_button.setEnabled(True)
        self.metrics_table.clear()
        self.metrics_table.setRowCount(0)
        self.validation_note.setVisible(False)
        self.prediction_note.setText(
            "Use the graph to decide which baseline you expect to perform better, then lock your prediction."
        )
        self.show_prediction_evidence()

    def show_prediction_evidence(self) -> None:
        """Show training-data evidence without revealing validation scores."""
        data = self.app.sequence_data
        train_idx = data.train_idx

        current_values = data.X[train_idx, -1, 0].astype(float)
        next_values = data.y[train_idx, 0].astype(float)
        training_mean = float(data.y[train_idx, 0].mean())

        self.plot_canvas.figure.clear()
        ax = self.plot_canvas.figure.add_subplot(111)
        ax.scatter(current_values, next_values, alpha=0.45, s=24)

        low = float(min(current_values.min(), next_values.min()))
        high = float(max(current_values.max(), next_values.max()))
        if high <= low:
            high = low + 1.0

        ax.plot(
            [low, high],
            [low, high],
            linestyle="--",
            linewidth=1.8,
            color=COBBER_MAROON,
            label="Most recent sample baseline",
        )
        ax.axhline(
            training_mean,
            linestyle=":",
            linewidth=2.0,
            color=COBBER_MAROON,
            label="Long-term average baseline",
        )
        ax.set_title("Does the next sample resemble the current sample?")
        ax.set_xlabel("Total fish density in the current sample")
        ax.set_ylabel("Total fish density in the next sample")
        ax.legend(fontsize=8)
        self.plot_canvas.figure.tight_layout()
        self.plot_canvas.draw()

    def lock_prediction(self) -> None:
        self.prediction_note.setText(f"Prediction locked: {self.expect_combo.currentText()} will have the lower error.")
        self.run_button.setEnabled(True)
        self.lock_button.setEnabled(False)

    def reveal(self) -> None:
        if not self.app.validation_baselines:
            return
        self.revealed = True
        results = self.app.validation_baselines
        self.metrics_table.clear()
        self.metrics_table.setRowCount(len(results))
        self.metrics_table.setColumnCount(3)
        self.metrics_table.setHorizontalHeaderLabels(["Model", "MAE", "RMSE"])
        for r, result in enumerate(results.values()):
            vals = [result.name, fmt(result.mae), fmt(result.rmse)]
            for c, value in enumerate(vals):
                self.metrics_table.setItem(r, c, table_item(value, bold=(c == 0)))
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.metrics_table.verticalHeader().setVisible(False)

        names = list(results.keys())
        maes = [results[name].mae for name in names]
        self.plot_canvas.figure.clear()
        ax = self.plot_canvas.figure.add_subplot(111)
        ax.bar(names, maes)
        ax.set_title("Validation MAE for the two baseline models")
        ax.set_ylabel("MAE")
        ax.tick_params(axis="x", rotation=10)
        self.plot_canvas.figure.tight_layout()
        self.plot_canvas.draw()

        winner = min(results.values(), key=lambda result: result.mae)
        prediction = self.expect_combo.currentText()
        if winner.name == prediction:
            prefix = "Your prediction matched the validation result."
        else:
            prefix = "The validation result did not match your prediction."
        self.prediction_note.setText(
            f"{prefix} {winner.name} had the lower MAE. "
            "The final test windows are still untouched."
        )
        self.validation_note.setVisible(True)


class TrainLSTMTab(BaseTab):
    def __init__(self, app: "CobberEcoLSTMApp"):
        super().__init__(app)
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        side, side_layout = self.side_panel()
        side_layout.addWidget(self.question_box("Can the LSTM use four recent samples to improve the next prediction?"))

        settings = QGroupBox("Training run")
        settings_layout = QVBoxLayout(settings)
        training_note = QLabel(
            "The model can train for up to 200 epochs. Training stops early when validation loss no longer improves."
        )


        training_note.setWordWrap(True)
        settings_layout.addWidget(training_note)
        self.train_button = QPushButton("Train the LSTM")
        self.train_button.setObjectName("PrimaryButton")
        self.train_button.clicked.connect(self.train)
        settings_layout.addWidget(self.train_button)
        side_layout.addWidget(settings)

        reminder = QGroupBox("What the model learns")
        reminder_layout = QVBoxLayout(reminder)
        reminder_text = QLabel(
            "The LSTM learns how to keep, write, and expose information while it trains. "
            "\n\nTraining loss measures error on the windows used to fit the model. "
            "Validation loss measures error on separate windows. \n\nTraining stops when validation loss no longer improves."
        )
        reminder_text.setWordWrap(True)
        reminder_layout.addWidget(reminder_text)
        side_layout.addWidget(reminder)
        side_layout.addStretch()

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        header = QLabel("Train the LSTM")
        header.setObjectName("MainHeader")
        main_layout.addWidget(header)
        intro = QLabel(
            "Training uses the training windows. One validation window from each plot checks whether performance is still improving. One test window from each plot remains separate until the final comparison."
        )
        intro.setObjectName("IntroText")
        intro.setWordWrap(True)
        main_layout.addWidget(intro)

        self.status = QLabel("Build and inspect the four-sample windows on Tab 3 before training the LSTM.")
        self.status.setObjectName("StatusLabel")
        self.status.setWordWrap(True)
        main_layout.addWidget(self.status)

        self.loss_canvas = PlotCanvas(width=8.0, height=4.0)
        self.loss_canvas.message("Training and validation loss will appear here.")
        main_layout.addWidget(self.loss_canvas, stretch=1)

        self.validation_table = QTableWidget()
        self.validation_table.setMaximumHeight(165)
        main_layout.addWidget(self.validation_table)

        root.addWidget(side)
        root.addWidget(main, stretch=1)

    def reset_for_new_data(self) -> None:
        self.train_button.setEnabled(self.app.sequence_data is not None)
        self.status.setText("The LSTM is ready to train on the four-sample windows.")
        self.loss_canvas.message("Training and validation loss will appear here.")
        self.validation_table.clear()
        self.validation_table.setRowCount(0)

    def train(self) -> None:
        self.status.setText("Training the LSTM. The window may pause briefly on this Windows computer.")
        QApplication.processEvents()
        result = train_lstm_model(self.app.sequence_data, 200)
        self.app.lstm_result = result
        if result.error:
            self.status.setText("The LSTM could not be trained. See the message for details.")
            QMessageBox.warning(self, "LSTM training problem", result.error)
            return

        history = result.history.history
        self.loss_canvas.figure.clear()
        ax = self.loss_canvas.figure.add_subplot(111)
        ax.plot(history.get("loss", []), label="Training loss")
        ax.plot(history.get("val_loss", []), label="Validation loss")
        ax.set_title("LSTM loss during training")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean squared error")
        ax.legend()
        self.loss_canvas.figure.tight_layout()
        self.loss_canvas.draw()

        validation = result.validation_result
        self.validation_table.clear()
        self.validation_table.setRowCount(1)
        self.validation_table.setColumnCount(3)
        self.validation_table.setHorizontalHeaderLabels(
            ["Model", "Validation MAE", "Validation RMSE"]
        )
        values = ["LSTM", fmt(validation.mae), fmt(validation.rmse)]
        for c, value in enumerate(values):
            self.validation_table.setItem(0, c, table_item(value, bold=(c == 0)))
        self.validation_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.validation_table.verticalHeader().setVisible(False)
        self.status.setText(
            "Training complete. The table shows validation performance only. Open Tab 5 to reveal the test windows."
        )
        self.app.compare_tab.refresh_ready_state()


class CompareTab(BaseTab):
    def __init__(self, app: "CobberEcoLSTMApp"):
        super().__init__(app)
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        side, side_layout = self.side_panel()
        side_layout.addWidget(
            self.question_box("Did several recent samples add useful information beyond the simpler predictions?"))

        reveal_box = QGroupBox("Final test")
        reveal_layout = QVBoxLayout(reveal_box)
        self.reveal_button = QPushButton("Reveal Test Results")
        self.reveal_button.setObjectName("PrimaryButton")
        self.reveal_button.clicked.connect(self.reveal)
        reveal_layout.addWidget(self.reveal_button)
        side_layout.addWidget(reveal_box)


        side_layout.addStretch()

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        header = QLabel("Compare the Three Predictions")
        header.setObjectName("MainHeader")
        main_layout.addWidget(header)
        intro = QLabel(
            "One test window from each plot was kept separate from training and validation. These 30 test windows are used only for the final comparison."
        )
        intro.setObjectName("IntroText")
        intro.setWordWrap(True)
        main_layout.addWidget(intro)

        self.interpretation = QLabel("Train the LSTM before revealing the test results.")
        self.interpretation.setObjectName("StatusLabel")
        self.interpretation.setWordWrap(True)
        main_layout.addWidget(self.interpretation)

        self.metrics_table = QTableWidget()
        self.metrics_table.setMaximumHeight(175)
        main_layout.addWidget(self.metrics_table)

        center = QHBoxLayout()
        self.plot_canvas = PlotCanvas(width=6.0, height=4.6)
        self.plot_canvas.message("The three-model comparison will appear here.")
        center.addWidget(self.plot_canvas, stretch=1)
        breakdown_panel = QWidget()
        breakdown_layout = QVBoxLayout(breakdown_panel)
        breakdown_layout.setContentsMargins(0, 0, 0, 0)

        breakdown_title = QLabel("LSTM performance by site type")
        breakdown_title.setStyleSheet(
            "font-weight: bold; color: #6c1d45;"
        )
        breakdown_layout.addWidget(breakdown_title)

        self.breakdown_table = QTableWidget()
        self.breakdown_table.setMinimumWidth(385)
        breakdown_layout.addWidget(self.breakdown_table)

        center.addWidget(breakdown_panel)
        main_layout.addLayout(center, stretch=1)

        root.addWidget(side)
        root.addWidget(main, stretch=1)
        self.revealed = False
        self.results: Dict[str, PredictionResult] = {}
        self.refresh_ready_state()

    def refresh_ready_state(self) -> None:
        ready = (
                self.app.sequence_data is not None
                and self.app.lstm_result is not None
                and self.app.lstm_result.test_result is not None
        )
        self.reveal_button.setEnabled(ready)
        if not ready:
            self.interpretation.setText("Train the LSTM before revealing the final test results.")

    def reveal(self) -> None:
        if not self.reveal_button.isEnabled():
            return
        self.results = dict(self.app.test_baselines)
        self.results["LSTM"] = self.app.lstm_result.test_result
        self.revealed = True

        self.metrics_table.clear()
        self.metrics_table.setRowCount(len(self.results))
        self.metrics_table.setColumnCount(3)
        self.metrics_table.setHorizontalHeaderLabels(
            ["Prediction", "Test MAE", "Test RMSE"]
        )
        for r, result in enumerate(self.results.values()):
            vals = [result.name, fmt(result.mae), fmt(result.rmse)]
            for c, value in enumerate(vals):
                self.metrics_table.setItem(r, c, table_item(value, bold=(c == 0)))
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.metrics_table.verticalHeader().setVisible(False)

        names = list(self.results.keys())
        maes = [self.results[name].mae for name in names]
        self.plot_canvas.figure.clear()
        ax = self.plot_canvas.figure.add_subplot(111)
        ax.bar(names, maes)
        ax.set_title("Final comparison on test windows")
        ax.set_ylabel("MAE")
        ax.tick_params(axis="x", rotation=10)
        self.plot_canvas.figure.tight_layout()
        self.plot_canvas.draw()

        winner = min(self.results.values(), key=lambda result: result.mae)
        if winner.name == "LSTM":
            text = (
                "The LSTM produced the lowest test MAE. In this dataset, several recent samples contained predictive information beyond the long-term average and the most recent sample alone."
            )
        else:
            text = (
                f"{winner.name} produced the lowest test MAE. The LSTM did not earn its added complexity for this target and four-sample window."
            )
        self.interpretation.setText(text)
        self.refresh_breakdown()

    def refresh_breakdown(self) -> None:
        self.breakdown_table.clear()
        if not self.revealed or "LSTM" not in self.results:
            self.breakdown_table.setRowCount(0)
            return
        result = self.results["LSTM"]
        rows = []
        for true, pred, rec in zip(result.y_true, result.y_pred, result.records):
            group = rec.site_type
            rows.append({"Group": group, "AbsoluteError": abs(float(pred) - float(true))})
        grouped = pd.DataFrame(rows).groupby("Group")["AbsoluteError"].agg(["mean", "count"]).reset_index()
        self.breakdown_table.setRowCount(len(grouped))
        self.breakdown_table.setColumnCount(2)
        self.breakdown_table.setHorizontalHeaderLabels(["Group", "LSTM MAE"])
        for r, row in grouped.iterrows():
            vals = [row["Group"], fmt(row["mean"])]
            for c, value in enumerate(vals):
                self.breakdown_table.setItem(r, c, table_item(value, bold=(c == 0)))
        self.breakdown_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.breakdown_table.verticalHeader().setVisible(False)


class CobberEcoLSTMApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1320, 790)
        self.setFont(QFont("Lato", 10))

        path = find_data_file()
        if path is None:
            raise FileNotFoundError(f"Place {DATA_FILE} in the same folder as this script.")
        self.data_path = path
        self.df = read_dataset(path)
        self.sequence_data: SequenceData = build_sequence_data(
            self.df,
            include_water_depth=False,
        )
        self.validation_baselines: Dict[str, PredictionResult] = baseline_predictions(
            self.sequence_data,
            self.sequence_data.val_idx,
        )
        self.test_baselines: Dict[str, PredictionResult] = baseline_predictions(
            self.sequence_data,
            self.sequence_data.test_idx,
        )
        self.lstm_result: Optional[LSTMResult] = None

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.explore_tab = ExploreStudyTab(self)
        self.baseline_tab = BaselinesTab(self)
        self.build_tab = BuildWindowsTab(self)
        self.train_tab = TrainLSTMTab(self)
        self.compare_tab = CompareTab(self)

        self.tabs.addTab(self.explore_tab, "Explore the Study")
        self.tabs.addTab(self.baseline_tab, "Compare Baselines")
        self.tabs.addTab(self.build_tab, "Build Windows")
        self.tabs.addTab(self.train_tab, "Train LSTM")
        self.tabs.addTab(self.compare_tab, "Compare Predictions")

        self.statusBar().showMessage(
            f"Loaded {self.data_path.name}: {len(self.df)} records, "
            f"{self.df.groupby(['Site', 'Plot']).ngroups} plot sequences."
        )

    def refresh_tabs_after_build(self) -> None:
        self.baseline_tab.reset_for_new_data()
        self.train_tab.reset_for_new_data()
        self.compare_tab.revealed = False
        self.compare_tab.results = {}
        self.compare_tab.metrics_table.clear()
        self.compare_tab.metrics_table.setRowCount(0)
        self.compare_tab.breakdown_table.clear()
        self.compare_tab.breakdown_table.setRowCount(0)
        self.compare_tab.plot_canvas.message("The three-model comparison will appear here.")
        self.compare_tab.refresh_ready_state()


def apply_stylesheet(app: QApplication) -> None:
    app.setStyleSheet(
        f"""
        QWidget {{
            color: {CHARCOAL};
            background-color: #ffffff;
        }}
        QTabWidget::pane {{
            border: 1px solid #cccccc;
        }}
        QTabBar::tab {{
            padding: 8px 14px;
            min-width: 150px;
            font-weight: bold;
            background: #666666;
            color: #ffffff;
        }}
        QTabBar::tab:selected {{
            background: {COBBER_MAROON};
            color: #ffffff;
        }}
        QFrame#SidePanel {{
            background-color: #fafafa;
            border: 1px solid #d6d6d6;
            border-radius: 8px;
        }}
        QLabel#MainHeader {{
            color: {COBBER_MAROON};
            font-size: 24px;
            font-weight: bold;
        }}
        QLabel#IntroText {{
            font-size: 13px;
        }}
        QLabel#StatusLabel {{
            color: {COBBER_MAROON};
            font-weight: bold;
            padding: 4px;
        }}
        QGroupBox {{
            color: {COBBER_MAROON};
            font-size: 15px;
            font-weight: bold;
            border: 1px solid #d6d6d6;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 12px;
            background-color: #fafafa;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
            background-color: #fafafa;
        }}
        QPushButton {{
            background-color: #666666;
            color: #ffffff;
            font-weight: bold;
            border: 1px solid #555555;
            border-radius: 5px;
            padding: 7px 10px;
        }}
        QPushButton:hover {{
            background-color: #777777;
        }}
        QPushButton:disabled {{
            background-color: #9a9a9a;
            color: #ffffff;
            border: 1px solid #888888;
        }}
        QPushButton#PrimaryButton,
        QPushButton#StepButton:enabled {{
            background-color: {COBBER_MAROON};
            color: #ffffff;
            border: 1px solid {COBBER_MAROON};
        }}

        QPushButton#PrimaryButton:disabled {{
            background-color: #9a9a9a;
            color: #ffffff;
            border: 1px solid #888888;
        }}

        QComboBox,
        QSpinBox {{
            background-color: #ffffff;
            color: #111111;
            border: 1px solid #9a9a9a;
            border-radius: 4px;
            padding: 4px 6px;
            min-height: 24px;
        }}
        QTableWidget,
        QTextEdit {{
            background-color: #ffffff;
            color: #111111;
            border: 1px solid #cccccc;
            selection-background-color: {COBBER_MAROON};
            selection-color: #ffffff;
        }}
        QHeaderView::section {{
            background-color: #eeeeee;
            color: #222222;
            padding: 5px;
            border: 1px solid #cccccc;
            font-weight: bold;
        }}
        """
    )


def main() -> None:
    app = QApplication(sys.argv)
    apply_stylesheet(app)
    try:
        window = CobberEcoLSTMApp()
    except Exception as exc:
        QMessageBox.critical(None, "CobberEcoLSTM could not start", str(exc))
        raise
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

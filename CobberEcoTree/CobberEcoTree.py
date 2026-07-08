#!/usr/bin/env python3
"""
CobberEcoTree.py

An urban-ecology adaptation of CobberTree for exploring decision trees.

This version mirrors the original CobberTree workflow:

    Manual Sorter:
        Students manually drag urban site cards into bins and watch the
        weighted variance in summer surface temperature change.

    Automated Tree:
        A DecisionTreeRegressor learns rules for predicting surface temperature
        from interpretable urban land-cover and vegetation features.

Expected CSV files in the same directory as this script:
    urban_heat_manual_subset.csv
    urban_heat_tree_dataset_for_tree.csv

Dependencies:
    pip install PyQt6 pandas numpy matplotlib scikit-learn

Run:
    python CobberEcoTree.py
"""

from __future__ import annotations

import sys
import re
from html import escape
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from sklearn.tree import DecisionTreeRegressor, plot_tree

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QColor, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


APP_TITLE = "CobberEcoTree"

MANUAL_FILE = "urban_heat_manual_subset.csv"
TREE_FILE = "urban_heat_tree_dataset_for_tree.csv"
TARGET = "Surface_Temp_C"

# Features used for full-model prediction/evaluation tabs.
# These are measurements students can reasonably estimate from imagery or a quick field check.
PREDICTION_FEATURES = [
    "Tree_Canopy_Pct",
    "Impervious_Pct",
    "Grass_Shrub_Pct",
    "NDVI",
    "Building_Density",
    "Road_Density",
]


def app_root() -> Path:
    """Return a sensible root directory for script or PyInstaller use."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def find_data_file(filename: str) -> Optional[Path]:
    root = app_root()
    candidates = [
        root / filename,
        Path.cwd() / filename,
        root / "data" / filename,
        root / "EcoData" / filename,
        root / "assets" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def clean_float(value, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


@dataclass
class UrbanSite:
    site_id: str
    site_name: str
    land_use_class: str
    surface_temp_c: float
    tree_canopy_pct: float
    impervious_pct: float
    grass_shrub_pct: float
    water_pct: float
    ndvi: float
    distance_to_water_m: float
    distance_to_park_m: float
    building_density: float
    road_density: float
    notes: str = ""

    @classmethod
    def from_row(cls, row: pd.Series) -> "UrbanSite":
        return cls(
            site_id=str(row.get("Site_ID", "")).strip(),
            site_name=str(row.get("Site_Name", "")).strip(),
            land_use_class=str(row.get("Land_Use_Class", "")).strip(),
            surface_temp_c=clean_float(row.get("Surface_Temp_C")),
            tree_canopy_pct=clean_float(row.get("Tree_Canopy_Pct")),
            impervious_pct=clean_float(row.get("Impervious_Pct")),
            grass_shrub_pct=clean_float(row.get("Grass_Shrub_Pct")),
            water_pct=clean_float(row.get("Water_Pct")),
            ndvi=clean_float(row.get("NDVI")),
            distance_to_water_m=clean_float(row.get("Distance_To_Water_m")),
            distance_to_park_m=clean_float(row.get("Distance_To_Park_m")),
            building_density=clean_float(row.get("Building_Density")),
            road_density=clean_float(row.get("Road_Density")),
            notes=str(row.get("Notes", "")).strip(),
        )

    @property
    def display_name(self) -> str:
        return self.site_id if self.site_id else self.site_name[:18]

    def card_text(self) -> str:
        return (
            f"{self.display_name:<5s} | Temp: {self.surface_temp_c:>4.1f} C | "
            f"Canopy: {self.tree_canopy_pct:>4.0f}% | Imp.: {self.impervious_pct:>4.0f}% | "
            f"{self.land_use_class}"
        )

    def detail_text(self) -> str:
        return (
            f"{self.display_name}: {self.site_name}\n\n"
            f"Land-use context: {self.land_use_class}\n"
            f"Summer surface temperature: {self.surface_temp_c:.1f} C\n\n"
            f"Tree canopy: {self.tree_canopy_pct:.1f}%\n"
            f"Impervious surface: {self.impervious_pct:.1f}%\n"
            f"Grass/shrub cover: {self.grass_shrub_pct:.1f}%\n"
            f"Water cover nearby: {self.water_pct:.1f}%\n"
            f"Vegetation greenness index: {self.ndvi:.3f}\n"
            f"Distance to water: {self.distance_to_water_m:.0f} m\n"
            f"Distance to park: {self.distance_to_park_m:.0f} m\n"
            f"Building density index: {self.building_density:.1f}\n"
            f"Road density index: {self.road_density:.1f}\n\n"
            f"Notes: {self.notes}\n\n"
            "Interpretation reminder:\n"
            "The decision tree is not proving that one variable caused this site to be hot. "
            "It is learning threshold rules that separate cooler and hotter urban sites in this dataset."
        )


def calculate_weighted_variance(group1: List[UrbanSite], group2: List[UrbanSite]) -> float:
    n1, n2 = len(group1), len(group2)
    n_total = n1 + n2
    if n_total == 0:
        return 0.0
    var1 = np.var([p.surface_temp_c for p in group1]) if n1 > 1 else 0.0
    var2 = np.var([p.surface_temp_c for p in group2]) if n2 > 1 else 0.0
    return float((n1 / n_total) * var1 + (n2 / n_total) * var2)


class SiteListWidget(QListWidget):
    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setAcceptDrops(True)
        self.setIconSize(QSize(80, 50))
        self.setAlternatingRowColors(True)

    def dropEvent(self, event):
        source_widget = event.source()
        if not isinstance(source_widget, SiteListWidget):
            event.ignore()
            return

        source_name = source_widget.name
        target_name = self.name

        # Same basic allowed-move structure as the original CobberTree:
        # deck -> level 1; level 1 -> deck, siblings, or level 2;
        # level 2 -> parent or sibling.
        allowed_moves = {
            "deck": ["bin1", "bin2"],
            "bin1": ["deck", "bin1_1", "bin1_2", "bin2"],
            "bin2": ["deck", "bin2_1", "bin2_2", "bin1"],
            "bin1_1": ["bin1", "bin1_2"],
            "bin1_2": ["bin1", "bin1_1"],
            "bin2_1": ["bin2", "bin2_2"],
            "bin2_2": ["bin2", "bin2_1"],
        }

        if target_name in allowed_moves.get(source_name, []):
            current = source_widget.currentItem()
            if current is None:
                event.ignore()
                return
            item = source_widget.takeItem(source_widget.row(current))
            self.addItem(item)
            event.accept()
        else:
            event.ignore()


class FirstSplitWidget(QWidget):
    """Guided first-split tab for comparing feature cutoffs.

    Students choose one urban feature, test threshold cutoffs, and watch how the
    weighted variance cost changes for the training set.
    """

    FRIENDLY_FEATURE_NAMES = {
        "Tree_Canopy_Pct": "Tree canopy (%)",
        "Impervious_Pct": "Impervious surface (%)",
        "Grass_Shrub_Pct": "Grass/shrub cover (%)",
        "Water_Pct": "Water nearby (%)",
        "NDVI": "Vegetation greenness index",
        "Distance_To_Water_m": "Distance to water (m)",
        "Distance_To_Park_m": "Distance to park (m)",
        "Building_Density": "Building density index",
        "Road_Density": "Road density index",
    }

    FEATURE_UNITS = {
        "Tree_Canopy_Pct": "%",
        "Impervious_Pct": "%",
        "Grass_Shrub_Pct": "%",
        "Water_Pct": "%",
        "NDVI": "",
        "Distance_To_Water_m": "m",
        "Distance_To_Park_m": "m",
        "Building_Density": "index",
        "Road_Density": "index",
    }

    MIN_GROUP_SIZE = 5

    def __init__(self):
        super().__init__()
        self.df: Optional[pd.DataFrame] = None
        self.training_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.features: List[str] = []
        self.current_cutoffs: List[float] = []
        self.tested_results: List[dict] = []
        self.best_results: List[dict] = []
        self.table_mode = "tested"

        main_layout = QHBoxLayout(self)
        controls_layout = QVBoxLayout()
        results_layout = QVBoxLayout()

        intro = QLabel(
            "<b>Test the first split:</b> Split the dataset, choose a feature, "
            "choose a cutoff, and score the first root-node question."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(
            "QLabel { color: #222222; background-color: #fffaf0; "
            "border: 1px solid #6C1D45; border-radius: 4px; padding: 8px; }"
        )
        controls_layout.addWidget(intro)

        self.split_button = QPushButton("Split the Dataset")
        self.test_button = QPushButton("Test This Split")
        self.best_button = QPushButton("Find Best First Split")
        self.test_button.setEnabled(False)
        self.best_button.setEnabled(False)
        controls_layout.addWidget(self.split_button)

        form = QFormLayout()
        self.feature_combo = QComboBox()
        self.cutoff_combo = QComboBox()
        self.feature_combo.setEnabled(False)
        self.cutoff_combo.setEnabled(False)
        form.addRow(QLabel("<b>Feature to test:</b>"), self.feature_combo)
        form.addRow(QLabel("<b>Cutoff value:</b>"), self.cutoff_combo)
        controls_layout.addLayout(form)

        controls_layout.addWidget(self.test_button)
        controls_layout.addWidget(self.best_button)

        self.split_explanation = QTextEdit(readOnly=True)
        self.split_explanation.setMinimumHeight(330)
        self.split_explanation.setStyleSheet(
            "QTextEdit { font-size: 13px; background-color: #ffffff; "
            "border: 1px solid #999999; padding: 6px; }"
        )
        controls_layout.addWidget(QLabel("<b>Split Details:</b>"))
        controls_layout.addWidget(self.split_explanation, 1)

        preview_label = QLabel(
            "<b>Root Node Preview</b><br>"
            "A first split is a tiny decision tree: one root question and two groups."
        )
        preview_label.setWordWrap(True)
        results_layout.addWidget(preview_label)

        self.preview_figure = Figure(figsize=(9.8, 3.4), dpi=100)
        self.preview_canvas = FigureCanvas(self.preview_figure)
        self.preview_canvas.setMinimumSize(820, 320)
        results_layout.addWidget(self.preview_canvas)

        self.results_label = QLabel(
            "<b>Cutoffs tested for this feature</b><br>"
            "Each row appears after you test one cutoff. Change the cutoff and test again to compare "
            "how the cost changes for this feature."
        )
        self.results_label.setWordWrap(True)
        results_layout.addWidget(self.results_label)

        self.summary_table = QTableWidget(0, 9)
        self.set_table_headers("tested")
        self.summary_table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self.summary_table.horizontalHeader().setMinimumHeight(78)
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.summary_table.setAlternatingRowColors(True)
        self.summary_table.setMinimumSize(820, 320)
        self.summary_table.setStyleSheet(
            "QTableWidget { background-color: #ffffff; color: #111111; gridline-color: #cccccc; }"
            "QTableWidget::item:selected { background-color: #e8d7df; color: #6C1D45; }"
            "QHeaderView::section { background-color: #6C1D45; color: #ffffff; font-weight: bold; padding: 6px; }"
        )
        results_layout.addWidget(self.summary_table, 1)

        main_layout.addLayout(controls_layout, 1)
        main_layout.addLayout(results_layout, 3)

        self.split_button.clicked.connect(self.split_dataset)
        self.feature_combo.currentIndexChanged.connect(self.update_cutoff_choices)
        self.test_button.clicked.connect(self.test_selected_split)
        self.best_button.clicked.connect(self.find_best_first_split)
        self.summary_table.itemSelectionChanged.connect(self.show_selected_best_split)

        self.load_full_dataset()
        self.draw_first_split_preview(None, "Click Split the Dataset to begin.")

    def friendly_feature(self, technical_name: str) -> str:
        return self.FRIENDLY_FEATURE_NAMES.get(technical_name, technical_name.replace("_", " "))

    def feature_unit(self, feature: str) -> str:
        return self.FEATURE_UNITS.get(feature, "")

    def set_table_headers(self, mode: str):
        self.table_mode = mode
        cutoff_label = "Cutoff\nTested" if mode == "tested" else "Best\nCutoff"
        self.summary_table.setHorizontalHeaderLabels([
            "Feature",
            cutoff_label,
            "Number ≤ cutoff\n(True box)",
            "Number > cutoff\n(False box)",
            "Avg. Surface\nTemp\nTrue box",
            "Avg. Surface\nTemp\nFalse box",
            "Variance in\nTrue box",
            "Variance in\nFalse box",
            "Cost",
        ])

    def reset_tested_cutoff_table(self):
        self.tested_results = []
        self.best_results = []
        self.set_table_headers("tested")
        self.summary_table.clearContents()
        self.summary_table.setRowCount(0)
        self.results_label.setText(
            "<b>Cutoffs tested for this feature</b><br>"
            "Each row appears after you test one cutoff. Change the cutoff and test again to compare "
            "how the cost changes for this feature."
        )

    def populate_summary_table(self, results: List[dict], highlight_first: bool = False):
        self.summary_table.clearContents()
        self.summary_table.setRowCount(len(results))

        for row_idx, result in enumerate(results):
            feature = result["feature"]
            values = [
                self.friendly_feature(feature),
                self.format_cutoff(feature, result["cutoff"]),
                str(result["n_low"]),
                str(result["n_high"]),
                f"{result['low_mean']:.1f} C",
                f"{result['high_mean']:.1f} C",
                f"{result['low_var']:.2f}",
                f"{result['high_var']:.2f}",
                f"{result['cost']:.2f}",
            ]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(value)
                if highlight_first and row_idx == 0:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    item.setBackground(QColor(232, 215, 223))
                    item.setForeground(QColor(108, 29, 69))
                self.summary_table.setItem(row_idx, col_idx, item)

    def load_full_dataset(self):
        path = find_data_file(TREE_FILE)
        if path is None:
            self.split_button.setEnabled(False)
            self.split_explanation.setPlainText(
                f"Could not find {TREE_FILE}. Put the dataset in the same folder as the app, then restart."
            )
            return

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            self.split_button.setEnabled(False)
            self.split_explanation.setPlainText(f"Could not read the dataset:\n{exc}")
            return

        if TARGET not in df.columns:
            self.split_button.setEnabled(False)
            self.split_explanation.setPlainText(f"The dataset must contain a {TARGET} column.")
            return

        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df = df.dropna(subset=[TARGET]).copy()

        self.features = self.select_features(df)
        if not self.features:
            self.split_button.setEnabled(False)
            self.split_explanation.setPlainText("No usable numeric feature columns were found.")
            return

        df = df.dropna(subset=self.features + [TARGET]).copy()
        if len(df) < 10:
            self.split_button.setEnabled(False)
            self.split_explanation.setPlainText("Too few complete records are available.")
            return

        self.df = df
        self.split_button.setEnabled(True)
        self.split_explanation.setPlainText(
            f"Loaded {len(df)} urban blocks.\n\n"
            "Before you split the dataset, make an ecological hypothesis: Which feature do you expect "
            "to create the strongest root-node split for summer surface temperature?\n\n"
            "Click Split the Dataset when you are ready."
        )

    def split_dataset(self):
        if self.df is None:
            return

        training_df = self.df.sample(frac=0.80, random_state=42).sort_index().copy()
        test_df = self.df.drop(training_df.index).sort_index().copy()
        self.training_df = training_df
        self.test_df = test_df

        self.feature_combo.blockSignals(True)
        self.feature_combo.clear()
        for feature in self.features:
            self.feature_combo.addItem(self.friendly_feature(feature), feature)
        self.feature_combo.blockSignals(False)

        self.feature_combo.setEnabled(True)
        self.cutoff_combo.setEnabled(True)
        self.test_button.setEnabled(True)
        self.best_button.setEnabled(False)
        self.reset_tested_cutoff_table()

        self.update_cutoff_choices()
        self.draw_first_split_preview(None, "Training set ready. Choose a feature and cutoff, then test the split.")
        self.split_explanation.setPlainText(
            f"Dataset split complete.\n\n"
            f"Training set: {len(training_df)} blocks\n"
            f"Hidden test set: {len(test_df)} blocks\n\n"
            "A cutoff is a dividing line. Blocks with feature values less than or equal to the cutoff "
            "go to the left group. Blocks with feature values above the cutoff go to the right group.\n\n"
            "Choose a feature and cutoff, then click Test This Split."
        )

    def select_features(self, df: pd.DataFrame) -> List[str]:
        candidates = [
            "Tree_Canopy_Pct",
            "Impervious_Pct",
            "Grass_Shrub_Pct",
            "Water_Pct",
            "NDVI",
            "Distance_To_Water_m",
            "Distance_To_Park_m",
            "Building_Density",
            "Road_Density",
        ]

        usable: List[str] = []
        for col in candidates:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if df[col].notna().sum() >= 5 and df[col].nunique(dropna=True) > 1:
                    usable.append(col)
        return usable

    def friendly_step(self, feature: str) -> float:
        if feature == "NDVI":
            return 0.10
        if feature == "Water_Pct":
            return 1.0
        if self.feature_unit(feature) == "m":
            return 100.0
        return 5.0

    def possible_cutoffs(self, df: pd.DataFrame, feature: str) -> List[float]:
        values = pd.to_numeric(df[feature], errors="coerce").dropna()
        if values.nunique() < 2:
            return []

        step = self.friendly_step(feature)
        min_value = float(values.min())
        max_value = float(values.max())
        start = np.ceil(min_value / step) * step
        stop = np.floor(max_value / step) * step
        candidates = [float(x) for x in np.arange(start, stop + step * 0.5, step)]
        candidates = [x for x in candidates if min_value < x < max_value]

        filtered: List[float] = []
        for cutoff in candidates:
            n_low = int((values <= cutoff).sum())
            n_high = int((values > cutoff).sum())
            if n_low >= self.MIN_GROUP_SIZE and n_high >= self.MIN_GROUP_SIZE:
                filtered.append(cutoff)

        if filtered:
            return filtered

        # Fallback for unusual datasets: use rounded quartiles, still avoiding empty groups.
        quantiles = values.quantile([0.25, 0.50, 0.75]).to_list()
        fallback = []
        for q in quantiles:
            if min_value < q < max_value:
                rounded = round(float(q) / step) * step
                if min_value < rounded < max_value:
                    n_low = int((values <= rounded).sum())
                    n_high = int((values > rounded).sum())
                    if n_low > 0 and n_high > 0 and rounded not in fallback:
                        fallback.append(float(rounded))
        return fallback

    def update_cutoff_choices(self):
        if self.training_df is None or self.feature_combo.count() == 0:
            return

        feature = self.feature_combo.currentData()
        if feature is None:
            return

        self.reset_tested_cutoff_table()
        self.current_cutoffs = self.possible_cutoffs(self.training_df, feature)
        self.cutoff_combo.clear()

        for cutoff in self.current_cutoffs:
            self.cutoff_combo.addItem(self.format_cutoff(feature, cutoff), cutoff)

        if self.cutoff_combo.count() > 0:
            self.cutoff_combo.setCurrentIndex(self.cutoff_combo.count() // 2)

    def score_split(self, df: pd.DataFrame, feature: str, cutoff: float) -> Optional[dict]:
        low_group = df[df[feature] <= cutoff]
        high_group = df[df[feature] > cutoff]
        n_low = len(low_group)
        n_high = len(high_group)
        n_total = n_low + n_high
        if n_low == 0 or n_high == 0 or n_total == 0:
            return None

        low_values = low_group[TARGET].to_numpy(dtype=float)
        high_values = high_group[TARGET].to_numpy(dtype=float)
        low_var = float(np.var(low_values)) if n_low > 1 else 0.0
        high_var = float(np.var(high_values)) if n_high > 1 else 0.0
        low_mean = float(np.mean(low_values))
        high_mean = float(np.mean(high_values))
        cost = float((n_low / n_total) * low_var + (n_high / n_total) * high_var)

        return {
            "feature": feature,
            "cutoff": float(cutoff),
            "n_low": n_low,
            "n_high": n_high,
            "low_mean": low_mean,
            "high_mean": high_mean,
            "low_var": low_var,
            "high_var": high_var,
            "cost": cost,
        }

    def format_cutoff(self, feature: str, cutoff: float) -> str:
        unit = self.feature_unit(feature)
        if unit == "%":
            return f"{cutoff:.0f}%"
        if unit == "m":
            return f"{cutoff:.0f} m"
        if unit == "index":
            return f"{cutoff:.0f}"
        return f"{cutoff:.2f}"

    def test_selected_split(self):
        if self.training_df is None:
            return
        feature = self.feature_combo.currentData()
        cutoff = self.cutoff_combo.currentData()
        if feature is None or cutoff is None:
            return

        result = self.score_split(self.training_df, feature, float(cutoff))
        if result is None:
            self.split_explanation.setPlainText("This cutoff does not create two usable groups.")
            return

        self.best_button.setEnabled(True)
        self.display_split_result(result, prefix="Tested split")
        self.draw_first_split_preview(result)

        # Return the table area to the current-feature cutoff comparison.
        self.set_table_headers("tested")
        self.results_label.setText(
            "<b>Cutoffs tested for this feature</b><br>"
            "Each row appears after you test one cutoff. Change the cutoff and test again to compare "
            "how the cost changes for this feature."
        )

        # Replace an older result for the same cutoff, or add this cutoff as a new row.
        replaced = False
        for idx, old_result in enumerate(self.tested_results):
            if old_result["feature"] == result["feature"] and abs(old_result["cutoff"] - result["cutoff"]) < 1e-9:
                self.tested_results[idx] = result
                replaced = True
                break
        if not replaced:
            self.tested_results.append(result)
        self.populate_summary_table(self.tested_results, highlight_first=False)

    def display_split_result(self, result: dict, prefix: str = "Split"):
        feature = result["feature"]
        feature_name = self.friendly_feature(feature)
        cutoff_text = self.format_cutoff(feature, result["cutoff"])
        low_rule = f"{feature_name} ≤ {cutoff_text}"
        high_rule = f"{feature_name} > {cutoff_text}"

        html = (
            f"<p><b>{escape(prefix)}:</b> {escape(feature_name)} at {escape(cutoff_text)}</p>"
            f"<p><b>True child node:</b> {escape(low_rule)}<br>"
            f"&nbsp;&nbsp;<b>City blocks in group:</b> {result['n_low']}<br>"
            f"&nbsp;&nbsp;<b>Average surface temp:</b> {result['low_mean']:.1f} C<br>"
            f"&nbsp;&nbsp;<b>Variance:</b> {result['low_var']:.2f}</p>"
            f"<p><b>False child node:</b> {escape(high_rule)}<br>"
            f"&nbsp;&nbsp;<b>City blocks in group:</b> {result['n_high']}<br>"
            f"&nbsp;&nbsp;<b>Average surface temp:</b> {result['high_mean']:.1f} C<br>"
            f"&nbsp;&nbsp;<b>Variance:</b> {result['high_var']:.2f}</p>"
            f"<p><b>Cost for this split:</b> {result['cost']:.2f}</p>"
        )
        self.split_explanation.setHtml(html)

    def draw_first_split_preview(self, result: Optional[dict], message: Optional[str] = None):
        self.preview_figure.clear()
        ax = self.preview_figure.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        maroon = "#6C1D45"
        grey = "#555555"
        pale = "#fffaf0"

        if result is None:
            ax.text(
                0.5, 0.55,
                message or "Choose a feature and cutoff to preview the first split.",
                ha="center", va="center", fontsize=15, color=grey,
                bbox=dict(boxstyle="round,pad=0.6", facecolor=pale, edgecolor=maroon, linewidth=1.8),
            )
            self.preview_figure.tight_layout()
            self.preview_canvas.draw()
            return

        feature = result["feature"]
        cutoff_text = self.format_cutoff(feature, result["cutoff"])
        feature_name = self.friendly_feature(feature)
        root_text = f"$\\bf{{Root\\ node}}$\n\n{feature_name} ≤ {cutoff_text}?"
        left_text = (
            f"$\\bf{{True\\ child\\ node}}$\n\n"
            f"{feature_name} ≤ {cutoff_text}\n"
            f"city blocks = {result['n_low']}\n"
            f"average surface temp = {result['low_mean']:.1f} C\n"
            f"variance = {result['low_var']:.2f}"
        )
        right_text = (
            f"$\\bf{{False\\ child\\ node}}$\n\n"
            f"{feature_name} > {cutoff_text}\n"
            f"city blocks = {result['n_high']}\n"
            f"average surface temp = {result['high_mean']:.1f} C\n"
            f"variance = {result['high_var']:.2f}"
        )

        # Draw the branches so they meet the node boxes cleanly.
        ax.plot([0.5, 0.27], [0.67, 0.39], color=grey, linewidth=2, zorder=1)
        ax.plot([0.5, 0.73], [0.67, 0.39], color=grey, linewidth=2, zorder=1)
        ax.text(0.37, 0.54, "True", ha="center", va="center", fontsize=11, color=maroon, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0), zorder=3)
        ax.text(0.63, 0.54, "False", ha="center", va="center", fontsize=11, color=maroon, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0), zorder=3)
        box = dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=maroon, linewidth=2)
        leaf_box = dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=maroon, linewidth=1.8)
        ax.text(0.5, 0.79, root_text, ha="center", va="center", fontsize=12, bbox=box, zorder=2)
        ax.text(0.25, 0.23, left_text, ha="center", va="center", fontsize=10.5, bbox=leaf_box, zorder=2)
        ax.text(0.75, 0.23, right_text, ha="center", va="center", fontsize=10.5, bbox=leaf_box, zorder=2)
        ax.text(0.5, 0.03, f"Cost = {result['cost']:.2f}", ha="center", va="bottom", fontsize=13, color=maroon, fontweight="bold")

        self.preview_figure.tight_layout()
        self.preview_canvas.draw()

    def find_best_for_feature(self, feature: str) -> Optional[dict]:
        if self.training_df is None:
            return None
        best: Optional[dict] = None
        for cutoff in self.possible_cutoffs(self.training_df, feature):
            result = self.score_split(self.training_df, feature, cutoff)
            if result is None:
                continue
            if best is None or result["cost"] < best["cost"]:
                best = result
        return best

    def find_best_first_split(self):
        if self.training_df is None:
            return

        results = []
        for feature in self.features:
            best = self.find_best_for_feature(feature)
            if best is not None:
                results.append(best)

        results.sort(key=lambda row: row["cost"])
        self.best_results = results
        self.set_table_headers("best")
        self.results_label.setText(
            "<b>Best root-node split found for each feature</b><br>"
            "Each row shows one feature. For each feature, the app tests the student-friendly "
            "cutoffs and keeps the cutoff with the lowest cost. The first row is the lowest-cost "
            "root-node split found."
        )
        self.populate_summary_table(results, highlight_first=True)

        if results:
            best = results[0]
            self.summary_table.selectRow(0)
            self.display_split_result(best, prefix="Best first split found")
            self.draw_first_split_preview(best)

    def show_selected_best_split(self):
        if self.table_mode != "best":
            return
        selected = self.summary_table.selectedItems()
        if not selected or not self.best_results:
            return
        row = selected[0].row()
        if 0 <= row < len(self.best_results):
            result = self.best_results[row]
            self.display_split_result(result, prefix="Selected ranked split")
            self.draw_first_split_preview(result)


class NextSplitWidget(QWidget):
    """Guided second-split tab.

    This tab starts with the best root-node split found from the training set.
    Students then test possible next splits inside each child node. The diagram
    grows to show the grandchild nodes created by the tested or best split.
    """

    FRIENDLY_FEATURE_NAMES = FirstSplitWidget.FRIENDLY_FEATURE_NAMES
    FEATURE_UNITS = FirstSplitWidget.FEATURE_UNITS
    MIN_GROUP_SIZE = 3

    def __init__(self):
        super().__init__()
        self.df: Optional[pd.DataFrame] = None
        self.training_df: Optional[pd.DataFrame] = None
        self.features: List[str] = []
        self.root_result: Optional[dict] = None
        self.true_df: Optional[pd.DataFrame] = None
        self.false_df: Optional[pd.DataFrame] = None
        self.true_parent_variance: float = 0.0
        self.false_parent_variance: float = 0.0
        self.true_result: Optional[dict] = None
        self.false_result: Optional[dict] = None
        self.true_best_result: Optional[dict] = None
        self.false_best_result: Optional[dict] = None
        self.true_status: str = "Choose a feature and cutoff for this child node."
        self.false_status: str = "Choose a feature and cutoff for this child node."

        main_layout = QVBoxLayout(self)

        intro = QLabel(
            "<b>Grow the next split:</b> Start with the root node and its two child nodes. "
            "Then test a new split inside each child node and watch the grandchild nodes appear."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(
            "QLabel { color: #222222; background-color: #fffaf0; "
            "border: 1px solid #6C1D45; border-radius: 4px; padding: 8px; }"
        )
        main_layout.addWidget(intro)

        self.figure = Figure(figsize=(14.5, 5.9), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(520)
        main_layout.addWidget(self.canvas, 3)

        panels = QHBoxLayout()
        self.true_panel = self.create_branch_panel("true")
        self.false_panel = self.create_branch_panel("false")
        panels.addWidget(self.true_panel["frame"])
        panels.addWidget(self.false_panel["frame"])
        main_layout.addLayout(panels, 0)

        button_row = QHBoxLayout()
        self.show_both_button = QPushButton("Show Best Splits for Both Child Nodes")
        self.show_both_button.clicked.connect(self.show_best_for_both)
        button_row.addStretch()
        button_row.addWidget(self.show_both_button)
        button_row.addStretch()
        main_layout.addLayout(button_row)

        self.load_and_prepare()

    def friendly_feature(self, technical_name: str) -> str:
        return self.FRIENDLY_FEATURE_NAMES.get(technical_name, technical_name.replace("_", " "))

    def feature_unit(self, feature: str) -> str:
        return self.FEATURE_UNITS.get(feature, "")

    def friendly_step(self, feature: str) -> float:
        if feature == "NDVI":
            return 0.10
        if feature == "Water_Pct":
            return 1.0
        if self.feature_unit(feature) == "m":
            return 100.0
        return 5.0

    def format_cutoff(self, feature: str, cutoff: float) -> str:
        unit = self.feature_unit(feature)
        if unit == "%":
            return f"{cutoff:.0f}%"
        if unit == "m":
            return f"{cutoff:.0f} m"
        if unit == "index":
            return f"{cutoff:.0f}"
        return f"{cutoff:.2f}"

    def select_features(self, df: pd.DataFrame) -> List[str]:
        candidates = [
            "Tree_Canopy_Pct",
            "Impervious_Pct",
            "Grass_Shrub_Pct",
            "Water_Pct",
            "NDVI",
            "Distance_To_Water_m",
            "Distance_To_Park_m",
            "Building_Density",
            "Road_Density",
        ]
        usable: List[str] = []
        for col in candidates:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if df[col].notna().sum() >= 5 and df[col].nunique(dropna=True) > 1:
                    usable.append(col)
        return usable

    def possible_cutoffs(self, df: pd.DataFrame, feature: str) -> List[float]:
        values = pd.to_numeric(df[feature], errors="coerce").dropna()
        if values.nunique() < 2:
            return []
        step = self.friendly_step(feature)
        min_value = float(values.min())
        max_value = float(values.max())
        start = np.ceil(min_value / step) * step
        stop = np.floor(max_value / step) * step
        candidates = [float(x) for x in np.arange(start, stop + step * 0.5, step)]
        candidates = [x for x in candidates if min_value < x < max_value]
        filtered: List[float] = []
        for cutoff in candidates:
            n_low = int((values <= cutoff).sum())
            n_high = int((values > cutoff).sum())
            if n_low >= self.MIN_GROUP_SIZE and n_high >= self.MIN_GROUP_SIZE:
                filtered.append(cutoff)
        if filtered:
            return filtered
        fallback: List[float] = []
        for q in values.quantile([0.25, 0.50, 0.75]).to_list():
            if min_value < q < max_value:
                rounded = round(float(q) / step) * step
                if min_value < rounded < max_value:
                    n_low = int((values <= rounded).sum())
                    n_high = int((values > rounded).sum())
                    if n_low > 0 and n_high > 0 and rounded not in fallback:
                        fallback.append(float(rounded))
        return fallback

    def score_split(self, df: pd.DataFrame, feature: str, cutoff: float) -> Optional[dict]:
        low_group = df[df[feature] <= cutoff]
        high_group = df[df[feature] > cutoff]
        n_low = len(low_group)
        n_high = len(high_group)
        n_total = n_low + n_high
        if n_low == 0 or n_high == 0 or n_total == 0:
            return None
        low_values = low_group[TARGET].to_numpy(dtype=float)
        high_values = high_group[TARGET].to_numpy(dtype=float)
        low_var = float(np.var(low_values)) if n_low > 1 else 0.0
        high_var = float(np.var(high_values)) if n_high > 1 else 0.0
        low_mean = float(np.mean(low_values))
        high_mean = float(np.mean(high_values))
        cost = float((n_low / n_total) * low_var + (n_high / n_total) * high_var)
        return {
            "feature": feature,
            "cutoff": float(cutoff),
            "n_low": n_low,
            "n_high": n_high,
            "low_mean": low_mean,
            "high_mean": high_mean,
            "low_var": low_var,
            "high_var": high_var,
            "cost": cost,
        }

    def find_best_for_feature(self, df: pd.DataFrame, feature: str) -> Optional[dict]:
        best: Optional[dict] = None
        for cutoff in self.possible_cutoffs(df, feature):
            result = self.score_split(df, feature, cutoff)
            if result is None:
                continue
            if best is None or result["cost"] < best["cost"]:
                best = result
        return best

    def find_best_split(self, df: pd.DataFrame) -> Optional[dict]:
        best: Optional[dict] = None
        for feature in self.features:
            result = self.find_best_for_feature(df, feature)
            if result is None:
                continue
            if best is None or result["cost"] < best["cost"]:
                best = result
        return best

    def create_branch_panel(self, branch: str) -> Dict[str, object]:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet(
            "QFrame { border: 2px solid #6C1D45; border-radius: 6px; background-color: #ffffff; }"
            "QLabel { border: none; }"
        )
        layout = QVBoxLayout(frame)
        title_text = "True child node" if branch == "true" else "False child node"
        title = QLabel(f"<h3 style='color:#6C1D45;'>{title_text}</h3>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        form = QFormLayout()
        feature_combo = QComboBox()
        cutoff_combo = QComboBox()
        form.addRow(QLabel("<b>Feature to test:</b>"), feature_combo)
        form.addRow(QLabel("<b>Cutoff value:</b>"), cutoff_combo)
        layout.addLayout(form)

        test_button = QPushButton("Test This Split")
        best_button = QPushButton("Show Best Split for This Child Node")
        layout.addWidget(test_button)
        layout.addWidget(best_button)

        status = QLabel("Choose a feature and cutoff to test a possible next split.")
        status.setWordWrap(True)
        status.setStyleSheet("QLabel { color: #444444; padding: 2px; }")
        layout.addWidget(status)

        feature_combo.currentIndexChanged.connect(lambda _=0, b=branch: self.update_branch_cutoffs(b))
        test_button.clicked.connect(lambda _=False, b=branch: self.test_branch_split(b))
        best_button.clicked.connect(lambda _=False, b=branch: self.show_best_for_branch(b))

        return {
            "frame": frame,
            "feature_combo": feature_combo,
            "cutoff_combo": cutoff_combo,
            "test_button": test_button,
            "best_button": best_button,
            "status": status,
        }

    def load_and_prepare(self):
        path = find_data_file(TREE_FILE)
        if path is None:
            self.set_both_panels_message(f"Could not find {TREE_FILE}.")
            self.draw_next_split_tree("Could not load the urban heat dataset.")
            return
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            self.set_both_panels_message(f"Could not read the dataset: {exc}")
            self.draw_next_split_tree("Could not read the urban heat dataset.")
            return
        if TARGET not in df.columns:
            self.set_both_panels_message(f"The dataset must contain a {TARGET} column.")
            self.draw_next_split_tree("Missing target column.")
            return

        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df = df.dropna(subset=[TARGET]).copy()
        self.features = self.select_features(df)
        df = df.dropna(subset=self.features + [TARGET]).copy()
        if len(df) < 10 or not self.features:
            self.set_both_panels_message("Too few complete records are available.")
            self.draw_next_split_tree("Too few records are available.")
            return

        self.df = df
        self.training_df = df.sample(frac=0.80, random_state=42).sort_index().copy()
        self.root_result = self.find_best_split(self.training_df)
        if self.root_result is None:
            self.set_both_panels_message("The app could not find a usable root-node split.")
            self.draw_next_split_tree("No usable root-node split was found.")
            return

        root_feature = self.root_result["feature"]
        root_cutoff = self.root_result["cutoff"]
        self.true_df = self.training_df[self.training_df[root_feature] <= root_cutoff].copy()
        self.false_df = self.training_df[self.training_df[root_feature] > root_cutoff].copy()
        self.true_parent_variance = float(np.var(self.true_df[TARGET].to_numpy(dtype=float))) if len(self.true_df) > 1 else 0.0
        self.false_parent_variance = float(np.var(self.false_df[TARGET].to_numpy(dtype=float))) if len(self.false_df) > 1 else 0.0

        for branch in ("true", "false"):
            panel = self.branch_panel(branch)
            combo: QComboBox = panel["feature_combo"]  # type: ignore[assignment]
            combo.blockSignals(True)
            combo.clear()
            for feature in self.features:
                combo.addItem(self.friendly_feature(feature), feature)
            combo.blockSignals(False)
            self.update_branch_cutoffs(branch)
            child_label = "True" if branch == "true" else "False"
            self.set_branch_status(branch, f"{child_label} child node ready. Choose a feature and cutoff.")
        self.draw_next_split_tree()

    def set_both_panels_message(self, message: str):
        for branch in ("true", "false"):
            self.set_branch_status(branch, message)

    def set_branch_status(self, branch: str, message: str):
        if branch == "true":
            self.true_status = message
        else:
            self.false_status = message
        panel = self.branch_panel(branch)
        status: QLabel = panel["status"]  # type: ignore[assignment]
        status.setText(message)

    def branch_panel(self, branch: str) -> Dict[str, object]:
        return self.true_panel if branch == "true" else self.false_panel

    def branch_df(self, branch: str) -> Optional[pd.DataFrame]:
        return self.true_df if branch == "true" else self.false_df

    def branch_parent_variance(self, branch: str) -> float:
        return self.true_parent_variance if branch == "true" else self.false_parent_variance

    def update_branch_cutoffs(self, branch: str):
        df = self.branch_df(branch)
        panel = self.branch_panel(branch)
        feature_combo: QComboBox = panel["feature_combo"]  # type: ignore[assignment]
        cutoff_combo: QComboBox = panel["cutoff_combo"]  # type: ignore[assignment]
        if df is None or feature_combo.count() == 0:
            return
        feature = feature_combo.currentData()
        cutoff_combo.clear()
        if feature is None:
            return
        for cutoff in self.possible_cutoffs(df, feature):
            cutoff_combo.addItem(self.format_cutoff(feature, cutoff), cutoff)
        if cutoff_combo.count() > 0:
            cutoff_combo.setCurrentIndex(cutoff_combo.count() // 2)
        else:
            self.set_branch_status(branch, "No usable cutoff is available for this feature inside this child node.")

    def test_branch_split(self, branch: str):
        df = self.branch_df(branch)
        if df is None:
            return
        panel = self.branch_panel(branch)
        feature_combo: QComboBox = panel["feature_combo"]  # type: ignore[assignment]
        cutoff_combo: QComboBox = panel["cutoff_combo"]  # type: ignore[assignment]
        feature = feature_combo.currentData()
        cutoff = cutoff_combo.currentData()
        if feature is None or cutoff is None:
            self.set_branch_status(branch, "No usable cutoff is available for this feature inside this child node.")
            return
        result = self.score_split(df, feature, float(cutoff))
        if result is None:
            self.set_branch_status(branch, "This cutoff does not create two usable groups inside this child node.")
            return
        if branch == "true":
            self.true_result = result
        else:
            self.false_result = result
        self.set_branch_status(branch, f"Tested split shown in the tree preview. Cost = {result['cost']:.2f}.")
        self.draw_next_split_tree()

    def show_best_for_branch(self, branch: str):
        df = self.branch_df(branch)
        if df is None:
            return
        result = self.find_best_split(df)
        if result is None:
            self.set_branch_status(branch, "No usable next split was found inside this child node.")
            return
        if branch == "true":
            self.true_best_result = result
            self.true_result = result
        else:
            self.false_best_result = result
            self.false_result = result
        self.set_branch_status(branch, f"Best split shown in the tree preview. Cost = {result['cost']:.2f}.")
        self.draw_next_split_tree()

    def show_best_for_both(self):
        self.show_best_for_branch("true")
        self.show_best_for_branch("false")

    def draw_next_split_tree(self, message: Optional[str] = None):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        maroon = "#6C1D45"
        grey = "#555555"
        pale = "#fffaf0"

        if message or self.root_result is None:
            ax.text(
                0.5, 0.55, message or "Preparing the root split...",
                ha="center", va="center", fontsize=15, color=grey,
                bbox=dict(boxstyle="round,pad=0.6", facecolor=pale, edgecolor=maroon, linewidth=1.8),
            )
            self.figure.tight_layout()
            self.canvas.draw()
            return

        root = self.root_result
        root_feature = root["feature"]
        root_cutoff = self.format_cutoff(root_feature, root["cutoff"])
        root_name = self.friendly_feature(root_feature)

        root_text = f"$\\bf{{Root\\ node}}$\n\n{root_name} ≤ {root_cutoff}?"
        true_text = (
            f"$\\bf{{True\\ child\\ node}}$\n\n"
            f"{root_name} ≤ {root_cutoff}\n"
            f"city blocks = {root['n_low']}\n"
            f"avg. surface temp = {root['low_mean']:.1f} C\n"
            f"variance = {root['low_var']:.2f}"
        )
        false_text = (
            f"$\\bf{{False\\ child\\ node}}$\n\n"
            f"{root_name} > {root_cutoff}\n"
            f"city blocks = {root['n_high']}\n"
            f"avg. surface temp = {root['high_mean']:.1f} C\n"
            f"variance = {root['high_var']:.2f}"
        )

        box = dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=maroon, linewidth=2)
        child_box = dict(boxstyle="round,pad=0.42", facecolor="white", edgecolor=maroon, linewidth=1.8)
        grandchild_box = dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=maroon, linewidth=1.55)

        # Root split branches.
        ax.plot([0.5, 0.25], [0.82, 0.62], color=grey, linewidth=2, zorder=1)
        ax.plot([0.5, 0.75], [0.82, 0.62], color=grey, linewidth=2, zorder=1)
        ax.text(0.36, 0.71, "True", ha="center", va="center", fontsize=11, color=maroon, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0), zorder=3)
        ax.text(0.64, 0.71, "False", ha="center", va="center", fontsize=11, color=maroon, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0), zorder=3)
        ax.text(0.5, 0.91, root_text, ha="center", va="center", fontsize=12, bbox=box, zorder=2)
        ax.text(0.25, 0.55, true_text, ha="center", va="center", fontsize=9.6, bbox=child_box, zorder=2)
        ax.text(0.75, 0.55, false_text, ha="center", va="center", fontsize=9.6, bbox=child_box, zorder=2)

        self.draw_grandchild_nodes(ax, "true", self.true_result, parent_x=0.25, left_x=0.11, right_x=0.39, maroon=maroon, grey=grey, box=grandchild_box)
        self.draw_grandchild_nodes(ax, "false", self.false_result, parent_x=0.75, left_x=0.61, right_x=0.89, maroon=maroon, grey=grey, box=grandchild_box)

        self.figure.tight_layout()
        self.canvas.draw()

    def draw_grandchild_nodes(self, ax, branch: str, result: Optional[dict], parent_x: float, left_x: float, right_x: float, maroon: str, grey: str, box: dict):
        if result is None:
            ax.text(
                parent_x, 0.24,
                "Test a possible\nnext split here.",
                ha="center", va="center", fontsize=9.5, color=grey,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#f7f7f7", edgecolor="#999999", linewidth=1.2),
            )
            return

        feature_name = self.friendly_feature(result["feature"])
        cutoff_text = self.format_cutoff(result["feature"], result["cutoff"])
        before = self.branch_parent_variance(branch)
        improvement = before - result["cost"]
        split_label = (
            f"Cost = {result['cost']:.2f}\n"
            f"Improvement = {improvement:.2f}"
        )
        ax.text(parent_x, 0.255, split_label, ha="center", va="center", fontsize=9.2, color=maroon, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none"), zorder=4)

        ax.plot([parent_x, left_x], [0.47, 0.21], color=grey, linewidth=1.65, zorder=1)
        ax.plot([parent_x, right_x], [0.47, 0.21], color=grey, linewidth=1.65, zorder=1)
        ax.text((parent_x + left_x) / 2, 0.34, "True", ha="center", va="center", fontsize=9.5, color=maroon, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8), zorder=3)
        ax.text((parent_x + right_x) / 2, 0.34, "False", ha="center", va="center", fontsize=9.5, color=maroon, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8), zorder=3)

        true_text = (
            f"$\\bf{{True\\ grandchild\\ node}}$\n\n"
            f"{feature_name} ≤ {cutoff_text}\n"
            f"city blocks = {result['n_low']}\n"
            f"avg. surface temp = {result['low_mean']:.1f} C\n"
            f"variance = {result['low_var']:.2f}"
        )
        false_text = (
            f"$\\bf{{False\\ grandchild\\ node}}$\n\n"
            f"{feature_name} > {cutoff_text}\n"
            f"city blocks = {result['n_high']}\n"
            f"avg. surface temp = {result['high_mean']:.1f} C\n"
            f"variance = {result['high_var']:.2f}"
        )
        ax.text(left_x, 0.10, true_text, ha="center", va="center", fontsize=7.8, bbox=box, zorder=2)
        ax.text(right_x, 0.10, false_text, ha="center", va="center", fontsize=7.8, bbox=box, zorder=2)



class DepthChoiceWidget(QWidget):
    """Train and test decision trees with depths 1-5.

    Students compare MAE and RMSE on hidden test data, choose the strongest
    depth, and then reveal the hidden test-block predictions.
    """

    FRIENDLY_FEATURE_NAMES = FirstSplitWidget.FRIENDLY_FEATURE_NAMES

    def __init__(self, app_state: dict):
        super().__init__()
        self.app_state = app_state
        self.df: Optional[pd.DataFrame] = None
        self.training_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.features: List[str] = []
        self.models: Dict[int, DecisionTreeRegressor] = {}
        self.depth_results: Dict[int, dict] = {}
        self.best_depth: Optional[int] = None
        self.node_artists = []

        main_layout = QHBoxLayout(self)
        controls_layout = QVBoxLayout()
        display_layout = QVBoxLayout()

        intro = QLabel(
            "<b>Choose tree depth:</b> Train decision trees with different maximum depths, "
            "then use the hidden test set to see which depth predicts new city blocks best."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(
            "QLabel { color: #222222; background-color: #fffaf0; "
            "border: 1px solid #6C1D45; border-radius: 4px; padding: 8px; }"
        )
        controls_layout.addWidget(intro)

        form = QFormLayout()
        self.depth_combo = QComboBox()
        for depth in [1, 2, 3, 4, 5]:
            self.depth_combo.addItem(f"Depth {depth}", depth)
        form.addRow(QLabel("<b>Max tree depth:</b>"), self.depth_combo)
        controls_layout.addLayout(form)

        self.train_depth_button = QPushButton("Train and Test This Depth")
        self.reveal_button = QPushButton("Reveal Hidden Test Blocks")
        self.reveal_button.setEnabled(False)
        controls_layout.addWidget(self.train_depth_button)
        controls_layout.addWidget(self.reveal_button)

        self.depth_table = QTableWidget(0, 4)
        self.depth_table.setHorizontalHeaderLabels(["Depth", "MAE", "RMSE", "Largest\nError"])
        self.depth_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.depth_table.horizontalHeader().setMinimumHeight(48)
        self.depth_table.setMinimumHeight(180)
        self.depth_table.setStyleSheet(
            "QTableWidget { background-color: #ffffff; color: #111111; gridline-color: #cccccc; }"
            "QHeaderView::section { background-color: #6C1D45; color: #ffffff; font-weight: bold; padding: 5px; }"
            "QTableWidget::item:selected { background-color: #6C1D45; color: #ffffff; }"
        )
        controls_layout.addWidget(self.depth_table)

        self.metric_box = QTextEdit(readOnly=True)
        self.metric_box.setMinimumHeight(250)
        self.metric_box.setStyleSheet(
            "QTextEdit { font-size: 13px; background-color: #ffffff; border: 1px solid #999999; padding: 6px; }"
        )
        controls_layout.addWidget(QLabel("<b>What the metrics mean:</b>"))
        controls_layout.addWidget(self.metric_box, 1)

        self.figure = Figure(figsize=(14, 7.3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(880, 560)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.mpl_connect("button_press_event", self.on_tree_click)

        self.test_title = QLabel("<b>Hidden test-block predictions</b>")
        self.test_title.setWordWrap(True)
        self.test_title.hide()
        self.test_table = QTableWidget(0, 6)
        self.test_table.setHorizontalHeaderLabels([
            "Test Block", "Site Type", "Predicted\nTemp", "Actual\nTemp", "Error", "Absolute\nError"
        ])
        self.test_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.test_table.horizontalHeader().setMinimumHeight(58)
        self.test_table.setAlternatingRowColors(True)
        self.test_table.setStyleSheet(
            "QTableWidget { background-color: #ffffff; color: #111111; gridline-color: #cccccc; }"
            "QHeaderView::section { background-color: #6C1D45; color: #ffffff; font-weight: bold; padding: 5px; }"
        )
        self.test_table.hide()

        display_layout.addWidget(self.toolbar)
        display_layout.addWidget(self.canvas, 1)
        display_layout.addWidget(self.test_title)
        display_layout.addWidget(self.test_table, 1)

        main_layout.addLayout(controls_layout, 1)
        main_layout.addLayout(display_layout, 4)

        self.train_depth_button.clicked.connect(self.train_and_test_selected_depth)
        self.reveal_button.clicked.connect(self.reveal_hidden_test_blocks)

        self.load_data()
        self.update_metric_box()
        self.draw_placeholder_tree()

    def friendly_feature(self, technical_name: str) -> str:
        return self.FRIENDLY_FEATURE_NAMES.get(technical_name, technical_name.replace("_", " "))

    def load_data(self):
        path = find_data_file(TREE_FILE)
        if path is None:
            self.metric_box.setPlainText(f"Could not find {TREE_FILE}.")
            self.train_depth_button.setEnabled(False)
            return
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            self.metric_box.setPlainText(f"Could not read the dataset:\n{exc}")
            self.train_depth_button.setEnabled(False)
            return
        if TARGET not in df.columns:
            self.metric_box.setPlainText(f"The dataset must contain {TARGET}.")
            self.train_depth_button.setEnabled(False)
            return
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        usable = []
        for feature in PREDICTION_FEATURES:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors="coerce")
                if df[feature].notna().sum() >= 5 and df[feature].nunique(dropna=True) > 1:
                    usable.append(feature)
        self.features = usable
        df = df.dropna(subset=self.features + [TARGET]).copy()
        if len(df) < 10 or not self.features:
            self.metric_box.setPlainText("Too few complete records are available for training and testing.")
            self.train_depth_button.setEnabled(False)
            return
        self.df = df
        self.training_df = df.sample(frac=0.80, random_state=42).sort_index().copy()
        self.test_df = df.drop(self.training_df.index).sort_index().copy()
        self.app_state["training_df"] = self.training_df
        self.app_state["test_df"] = self.test_df
        self.app_state["prediction_features"] = self.features

    def update_metric_box(self):
        if self.training_df is None or self.test_df is None:
            self.metric_box.setPlainText("Load the dataset to begin.")
            return
        best_html = "No best depth chosen yet."
        if self.best_depth is not None:
            best_html = f"<b>Best depth so far:</b> {self.best_depth}"
        features_html = ", ".join(self.friendly_feature(f) for f in self.features)
        self.metric_box.setHtml(
            f"Training city blocks: {len(self.training_df)}<br>"
            f"Hidden test city blocks: {len(self.test_df)}<br>"
            f"Features used: {features_html}<br><br>"
            "MAE is the average size of the prediction error.<br>"
            "RMSE also measures error, but it grows more when the model makes a few large misses.<br>"
            "Largest error shows the worst miss on one hidden test block.<br><br>"
            "Test all five depths before revealing the hidden test blocks.<br><br>"
            f"{best_html}"
        )

    def train_model(self, depth: int) -> Optional[DecisionTreeRegressor]:
        if self.training_df is None or not self.features:
            return None
        X_train = self.training_df[self.features]
        y_train = self.training_df[TARGET]
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model: DecisionTreeRegressor) -> dict:
        if self.test_df is None:
            return {}
        X_test = self.test_df[self.features]
        actual = self.test_df[TARGET].to_numpy(dtype=float)
        pred = model.predict(X_test)
        errors = pred - actual
        abs_errors = np.abs(errors)
        return {
            "mae": float(np.mean(abs_errors)),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "largest_error": float(np.max(abs_errors)),
            "predictions": pred,
            "errors": errors,
            "abs_errors": abs_errors,
        }

    def train_and_test_selected_depth(self):
        depth = int(self.depth_combo.currentData())
        model = self.train_model(depth)
        if model is None:
            return
        metrics = self.evaluate_model(model)
        self.models[depth] = model
        self.depth_results[depth] = metrics
        self.populate_depth_table()
        self.draw_tree(model, depth)
        self.canvas.show()
        self.toolbar.show()
        self.test_title.hide()
        self.test_table.hide()
        all_depths_tested = all(depth in self.depth_results for depth in [1, 2, 3, 4, 5])
        self.reveal_button.setEnabled(all_depths_tested)

        best_depth = min(self.depth_results, key=lambda d: self.depth_results[d]["mae"])
        self.best_depth = int(best_depth)
        self.app_state["best_depth"] = self.best_depth
        self.app_state["best_model"] = self.models[self.best_depth]
        self.update_metric_box()

    def populate_depth_table(self):
        depths = sorted(self.depth_results)
        self.depth_table.clearContents()
        self.depth_table.setRowCount(len(depths))
        best_depth = min(depths, key=lambda d: self.depth_results[d]["mae"]) if depths else None
        for row_idx, depth in enumerate(depths):
            m = self.depth_results[depth]
            values = [str(depth), f"{m['mae']:.2f} C", f"{m['rmse']:.2f} C", f"{m['largest_error']:.2f} C"]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(value)
                if best_depth == depth:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    item.setBackground(QColor(160, 105, 132))
                    item.setForeground(QColor(255, 255, 255))
                self.depth_table.setItem(row_idx, col_idx, item)

    def draw_placeholder_tree(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.55, "Choose a depth, then train and test the decision tree.", ha="center", va="center", fontsize=16, color="#555555",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#fffaf0", edgecolor="#6C1D45", linewidth=1.8))
        self.canvas.draw()

    def draw_tree(self, model: DecisionTreeRegressor, depth: int):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        friendly_names = [self.friendly_feature(f) for f in self.features]
        artists = plot_tree(
            model,
            feature_names=friendly_names,
            filled=False,
            rounded=True,
            fontsize=9,
            impurity=False,
            ax=ax,
        )
        ax.set_title(f"Decision Tree for Urban Surface Temperature (Max Depth = {depth})", fontsize=16)
        self.node_artists = []
        for artist in artists:
            raw = artist.get_text()
            cleaned = self.clean_node_label(raw)
            artist.set_text(cleaned)
            if raw.strip() in {"True", "False"}:
                artist.set_fontsize(10)
                artist.set_fontweight("bold")
                artist.set_color("#6C1D45")
            else:
                self.node_artists.append(artist)
        self.figure.tight_layout()
        self.canvas.draw()

    def clean_node_label(self, raw: str) -> str:
        text = raw
        if text.strip() in {"True", "False"}:
            return text.strip()
        text = text.replace("samples = ", "city blocks = ")
        text = re.sub(r"value = \[?([0-9]+(?:\.[0-9]+)?)\]?", r"predicted temp = \1 C", text)
        text = text.replace("<=", "≤")
        return text

    def explain_node_label(self, label: str) -> str:
        lines = [line.strip() for line in label.splitlines() if line.strip()]
        if not lines:
            return "This node has no text to display."
        first = lines[0]
        blocks_line = next((line for line in lines if line.startswith("city blocks =")), "")
        pred_line = next((line for line in lines if line.startswith("predicted temp =")), "")
        parts = []
        if "≤" in first:
            feature, threshold = [piece.strip() for piece in first.split("≤", 1)]
            parts.append(f"Decision question:\nIs {feature} less than or equal to {threshold}?")
            parts.append("The True branch contains city blocks that meet this rule. The False branch contains city blocks that do not.")
        else:
            parts.append("Leaf node:\nThis is the end of one decision path. The decision tree makes its prediction here.")
        if blocks_line:
            parts.append(blocks_line.replace("city blocks =", "Training city blocks in this node:"))
        if pred_line:
            parts.append(pred_line.replace("predicted temp =", "Predicted summer surface temperature:"))
        parts.append("Ecology reminder:\nThis node describes a pattern in the training data. Test-set metrics tell us whether that pattern predicts hidden city blocks well.")
        return "\n\n".join(parts)

    def on_tree_click(self, event):
        if event.inaxes is None or not self.node_artists:
            return
        renderer = self.canvas.get_renderer()
        for artist in self.node_artists:
            bbox = artist.get_window_extent(renderer=renderer).expanded(1.22, 1.35)
            if bbox.contains(event.x, event.y):
                self.show_node_popup(artist.get_text())
                return

    def show_node_popup(self, label: str):
        dialog = QDialog(self)
        dialog.setWindowTitle("Decision Tree Node")
        dialog.resize(560, 420)
        layout = QVBoxLayout(dialog)
        title = QLabel("<h2 style='color:#6C1D45;'>Decision Tree Node</h2>")
        title.setWordWrap(True)
        layout.addWidget(title)
        body = QTextEdit(readOnly=True)
        body.setPlainText(self.explain_node_label(label))
        body.setStyleSheet("QTextEdit { font-size: 15px; background-color: #fffaf0; border: 2px solid #6C1D45; padding: 10px; }")
        layout.addWidget(body, 1)
        close_button = QPushButton("Close")
        close_button.setFixedWidth(110)
        close_button.clicked.connect(dialog.accept)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(close_button)
        layout.addLayout(row)
        dialog.exec()

    def reveal_hidden_test_blocks(self):
        if self.best_depth is None:
            return
        model = self.models.get(self.best_depth)
        metrics = self.depth_results.get(self.best_depth)
        if model is None or metrics is None or self.test_df is None:
            return
        self.canvas.hide()
        self.toolbar.hide()
        self.test_title.setText(
            f"<b>Hidden test-block predictions for depth {self.best_depth}</b><br>"
            "These city blocks were held back while the decision tree was trained. "
            "Smaller error values mean the prediction landed closer to the measured surface temperature.<br>"
            f"MAE = {metrics['mae']:.2f} C &nbsp;&nbsp; RMSE = {metrics['rmse']:.2f} C &nbsp;&nbsp; Largest error = {metrics['largest_error']:.2f} C"
        )
        self.test_title.show()
        self.test_table.show()
        self.test_table.clearContents()
        self.test_table.setRowCount(len(self.test_df))
        preds = metrics["predictions"]
        errors = metrics["errors"]
        abs_errors = metrics["abs_errors"]
        for i, (_, row) in enumerate(self.test_df.iterrows()):
            site_type = str(row.get("Land_Use_Class", "Urban block"))
            values = [
                f"Test City Block {i + 1}",
                site_type,
                f"{preds[i]:.1f} C",
                f"{float(row[TARGET]):.1f} C",
                f"{errors[i]:+.1f} C",
                f"{abs_errors[i]:.1f} C",
            ]
            for col, value in enumerate(values):
                self.test_table.setItem(i, col, QTableWidgetItem(value))


class NewBlockPredictionWidget(QWidget):
    """Predict the temperature and heat-risk class for an untested city block."""

    FRIENDLY_FEATURE_NAMES = FirstSplitWidget.FRIENDLY_FEATURE_NAMES
    FEATURE_UNITS = FirstSplitWidget.FEATURE_UNITS

    def __init__(self, app_state: dict):
        super().__init__()
        self.app_state = app_state
        self.df: Optional[pd.DataFrame] = None
        self.training_df: Optional[pd.DataFrame] = None
        self.features: List[str] = []
        self.model: Optional[DecisionTreeRegressor] = None
        self.depth: int = 3
        self.feature_boxes: Dict[str, QDoubleSpinBox] = {}
        self.presets: Dict[str, Dict[str, float]] = {}
        self.baseline_temp: Optional[float] = None
        self.intervention_step: int = 0
        self.intervention_history: List[float] = []
        self.current_site_type: Optional[str] = None

        main_layout = QHBoxLayout(self)
        controls_layout = QVBoxLayout()
        output_layout = QVBoxLayout()

        intro = QLabel(
            "<b>Predict a new city block:</b> Choose a site type to load a starting scenario, "
            "adjust the measurements, and use the trained decision tree to predict summer surface temperature."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(
            "QLabel { color: #222222; background-color: #fffaf0; "
            "border: 1px solid #6C1D45; border-radius: 4px; padding: 8px; }"
        )
        controls_layout.addWidget(intro)

        self.depth_label = QLabel("Using tree depth: not trained yet")
        self.depth_label.setWordWrap(True)
        controls_layout.addWidget(self.depth_label)

        self.site_type_combo = QComboBox()
        form = QFormLayout()
        form.addRow(QLabel("<b>Site type:</b>"), self.site_type_combo)
        controls_layout.addLayout(form)

        self.feature_form = QFormLayout()
        controls_layout.addLayout(self.feature_form)

        self.predict_button = QPushButton("Predict Surface Temperature")
        self.intervention_button = QPushButton("Try More Shade and Less Pavement")
        controls_layout.addWidget(self.predict_button)
        controls_layout.addWidget(self.intervention_button)
        controls_layout.addStretch()

        self.output_box = QTextEdit(readOnly=True)
        self.output_box.setStyleSheet(
            "QTextEdit { font-size: 15px; background-color: #ffffff; border: 1px solid #999999; padding: 8px; }"
        )
        output_layout.addWidget(QLabel("<b>Prediction result:</b>"))
        output_layout.addWidget(self.output_box, 1)

        self.intervention_figure = Figure(figsize=(6.6, 2.8), dpi=100)
        self.intervention_canvas = FigureCanvas(self.intervention_figure)
        self.intervention_canvas.setMinimumHeight(260)
        output_layout.addWidget(QLabel("<b>Predicted temperature after each intervention:</b>"))
        output_layout.addWidget(self.intervention_canvas, 1)

        main_layout.addLayout(controls_layout, 1)
        main_layout.addLayout(output_layout, 2)

        self.site_type_combo.currentIndexChanged.connect(self.load_selected_preset)
        self.predict_button.clicked.connect(self.predict_new_block)
        self.intervention_button.clicked.connect(self.apply_intervention)

        self.load_data()
        self.build_feature_controls()
        self.build_or_refresh_model()
        self.output_box.setPlainText(
            "Choose a site type, adjust the measurements, and predict the surface temperature.\n\n"
            "Then try an intervention. Increasing tree canopy and grass/shrub cover while lowering impervious surface can show whether the predicted heat-risk category changes."
        )

    def friendly_feature(self, technical_name: str) -> str:
        return self.FRIENDLY_FEATURE_NAMES.get(technical_name, technical_name.replace("_", " "))

    def load_data(self):
        path = find_data_file(TREE_FILE)
        if path is None:
            self.output_box.setPlainText(f"Could not find {TREE_FILE}.")
            return
        df = pd.read_csv(path)
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        usable = []
        for feature in PREDICTION_FEATURES:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors="coerce")
                usable.append(feature)
        df = df.dropna(subset=usable + [TARGET]).copy()
        self.df = df
        self.features = usable
        self.training_df = df.sample(frac=0.80, random_state=42).sort_index().copy()
        self.build_presets()

    def build_presets(self):
        if self.df is None:
            return

        self.site_type_combo.clear()
        self.presets = {}

        # Hand-tuned starting scenarios.
        # These are not meant to be exact site medians. They are realistic teaching scenarios
        # that let students compare different kinds of city blocks.
        scenario_presets = {
            "Commercial": {
                "Tree_Canopy_Pct": 15,
                "Impervious_Pct": 80,
                "Grass_Shrub_Pct": 10,
                "NDVI": 0.25,
                "Building_Density": 70,
                "Road_Density": 60,
            },
            "Industrial": {
                "Tree_Canopy_Pct": 10,
                "Impervious_Pct": 85,
                "Grass_Shrub_Pct": 5,
                "NDVI": 0.20,
                "Building_Density": 55,
                "Road_Density": 75,
            },
            "Institutional": {
                "Tree_Canopy_Pct": 30,
                "Impervious_Pct": 55,
                "Grass_Shrub_Pct": 30,
                "NDVI": 0.45,
                "Building_Density": 45,
                "Road_Density": 40,
            },
            "Mixed Use": {
                "Tree_Canopy_Pct": 25,
                "Impervious_Pct": 65,
                "Grass_Shrub_Pct": 20,
                "NDVI": 0.35,
                "Building_Density": 60,
                "Road_Density": 55,
            },
            "Residential": {
                "Tree_Canopy_Pct": 35,
                "Impervious_Pct": 45,
                "Grass_Shrub_Pct": 35,
                "NDVI": 0.50,
                "Building_Density": 30,
                "Road_Density": 35,
            },
            "Green Space": {
                "Tree_Canopy_Pct": 45,
                "Impervious_Pct": 35,
                "Grass_Shrub_Pct": 55,
                "NDVI": 0.60,
                "Building_Density": 15,
                "Road_Density": 20,
            },
            "Urban Forest": {
                "Tree_Canopy_Pct": 80,
                "Impervious_Pct": 10,
                "Grass_Shrub_Pct": 65,
                "NDVI": 0.80,
                "Building_Density": 5,
                "Road_Density": 5,
            },
        }

        # Only keep presets for features that actually exist in this app.
        for site_type, preset in scenario_presets.items():
            cleaned = {}
            for feature in self.features:
                if feature in preset:
                    cleaned[feature] = preset[feature]
                else:
                    cleaned[feature] = float(self.df[feature].median())
            self.presets[site_type] = cleaned
            self.site_type_combo.addItem(site_type, site_type)
    def build_feature_controls(self):
        if self.df is None:
            return
        for feature in self.features:
            box = QDoubleSpinBox()
            min_val = float(self.df[feature].min())
            max_val = float(self.df[feature].max())
            box.setRange(min_val, max_val)
            if feature == "NDVI":
                box.setDecimals(2)
                box.setSingleStep(0.05)
            else:
                box.setDecimals(0)
                box.setSingleStep(5)
            self.feature_boxes[feature] = box
            self.feature_form.addRow(QLabel(f"<b>{self.friendly_feature(feature)}:</b>"), box)
        self.load_selected_preset()

    def build_or_refresh_model(self):
        best_depth = self.app_state.get("best_depth")
        if best_depth is None:
            best_depth = self.choose_best_depth_quickly()
            self.app_state["best_depth"] = best_depth
        self.depth = int(best_depth)
        if self.training_df is None or not self.features:
            return
        self.model = DecisionTreeRegressor(max_depth=self.depth, random_state=42)
        self.model.fit(self.training_df[self.features], self.training_df[TARGET])
        self.depth_label.setText(f"Using tree depth: {self.depth}. This depth comes from the hidden-test comparison, or from an automatic check if Tab 3 has not been completed yet.")

    def choose_best_depth_quickly(self) -> int:
        if self.df is None:
            return 3
        train = self.df.sample(frac=0.80, random_state=42).sort_index().copy()
        test = self.df.drop(train.index).sort_index().copy()
        best_depth = 1
        best_mae = float("inf")
        for depth in [1, 2, 3, 4, 5]:
            model = DecisionTreeRegressor(max_depth=depth, random_state=42)
            model.fit(train[self.features], train[TARGET])
            pred = model.predict(test[self.features])
            mae = float(np.mean(np.abs(pred - test[TARGET].to_numpy(dtype=float))))
            if mae < best_mae:
                best_mae = mae
                best_depth = depth
        return best_depth

    def load_selected_preset(self):
        site_type = self.site_type_combo.currentData()
        if not site_type or site_type not in self.presets:
            return

        site_changed = site_type != self.current_site_type
        self.current_site_type = site_type

        for feature, value in self.presets[site_type].items():
            if feature in self.feature_boxes:
                self.feature_boxes[feature].blockSignals(True)
                self.feature_boxes[feature].setValue(value)
                self.feature_boxes[feature].blockSignals(False)

        if site_changed:
            self.reset_intervention_history()
            self.output_box.setPlainText(
                "Choose a site type, adjust the measurements, and predict the surface temperature.\n\n"
                "Then try an intervention. Increasing tree canopy and grass/shrub cover while lowering impervious surface can show whether the predicted heat-risk category changes."
            )

    def current_input_row(self) -> pd.DataFrame:
        values = {feature: self.feature_boxes[feature].value() for feature in self.features}
        return pd.DataFrame([values], columns=self.features)

    def reset_intervention_history(self):
        self.baseline_temp = None
        self.intervention_step = 0
        self.intervention_history = []
        self.draw_intervention_graph()

    def draw_intervention_graph(self):
        self.intervention_figure.clear()
        ax = self.intervention_figure.add_subplot(111)

        if not self.intervention_history:
            ax.axis("off")
            ax.text(
                0.5, 0.5,
                "Predict a starting site, then try interventions.",
                ha="center",
                va="center",
                fontsize=12,
                color="#555555",
            )
            self.intervention_canvas.draw()
            return

        steps = list(range(len(self.intervention_history)))
        labels = ["Start"] + [str(i) for i in range(1, len(self.intervention_history))]

        ax.plot(steps, self.intervention_history, marker="o", linewidth=2)
        ax.set_xticks(steps)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Intervention step", fontweight="bold")
        ax.set_ylabel("Predicted temp (C)", fontweight="bold")
        ax.grid(True, alpha=0.25)

        if len(self.intervention_history) > 1:
            total_drop = self.intervention_history[0] - self.intervention_history[-1]
            ax.set_title(f"Predicted cooling: {total_drop:.1f} C", fontweight="bold")
        else:
            ax.set_title("Starting prediction", fontweight="bold")

        self.intervention_figure.tight_layout()
        self.intervention_canvas.draw()

    def heat_risk_class(self, temp: float) -> str:
        if temp < 30.0:
            return "Low heat risk"
        if temp < 35.0:
            return "Moderate heat risk"
        return "High heat risk"

    def predict_new_block(self):
        self.build_or_refresh_model()
        if self.model is None:
            return

        X = self.current_input_row()
        temp = float(self.model.predict(X)[0])
        risk = self.heat_risk_class(temp)
        site_type = self.site_type_combo.currentText()

        if self.baseline_temp is None:
            self.baseline_temp = temp
            self.intervention_history = [temp]
            self.intervention_step = 0
        elif not self.intervention_history:
            self.intervention_history = [self.baseline_temp]

        cooling = self.baseline_temp - temp if self.baseline_temp is not None else 0.0

        values_html = "<br>".join([
            f"&nbsp;&nbsp;{self.friendly_feature(f)}: {self.feature_boxes[f].value():.2f}" if f == "NDVI"
            else f"&nbsp;&nbsp;{self.friendly_feature(f)}: {self.feature_boxes[f].value():.0f}"
            for f in self.features
        ])

        self.output_box.setHtml(
            f"<h2 style='color:#6C1D45; margin-bottom:4px;'>Prediction result</h2>"
            f"<p style='font-size:18px; margin:4px 0;'><b>Site type:</b> {site_type}</p>"
            f"<p style='font-size:18px; margin:4px 0;'><b>Predicted summer surface temperature:</b> {temp:.1f} C</p>"
            f"<p style='font-size:18px; margin:4px 0;'><b>Heat-risk class:</b> {risk}</p>"
            f"<p style='font-size:18px; margin:4px 0;'><b>Predicted cooling from starting site:</b> {cooling:.1f} C</p>"
            "<br>"
            "<b>Measurements used by the decision tree:</b><br>"
            f"{values_html}<br><br>"
            "<b>Ecology note:</b><br>"
            "The prediction can help flag blocks for field checks or cooling interventions. It does not prove which intervention will work by itself."
        )

        self.draw_intervention_graph()

    def apply_intervention(self):
        # If the student has not made a starting prediction yet,
        # make one before applying the first intervention.
        if self.baseline_temp is None or not self.intervention_history:
            self.predict_new_block()

        for feature, delta in [
            ("Tree_Canopy_Pct", 5),
            ("Grass_Shrub_Pct", 5),
            ("Impervious_Pct", -5),
            ("Road_Density", -5),
            ("NDVI", 0.05),
        ]:
            box = self.feature_boxes.get(feature)
            if box is None:
                continue
            box.setValue(max(box.minimum(), min(box.maximum(), box.value() + delta)))

        self.intervention_step += 1

        if self.model is None:
            self.build_or_refresh_model()
        if self.model is None:
            return

        temp = float(self.model.predict(self.current_input_row())[0])

        self.intervention_history.append(temp)
        self.predict_new_block()

class CobberEcoTreeApp(QMainWindow):
    def __init__(self, manual_dataset: List[UrbanSite]):
        super().__init__()

        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")
        self.app_state: Dict[str, object] = {}

        self.setWindowTitle("CobberEcoTree")
        self.resize(1520, 880)
        self.setFont(self.lato_font)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        first_split_widget = FirstSplitWidget()
        tabs.addTab(first_split_widget, "Test First Split")

        next_split_widget = NextSplitWidget()
        tabs.addTab(next_split_widget, "Grow Next Split")

        depth_choice_widget = DepthChoiceWidget(self.app_state)
        tabs.addTab(depth_choice_widget, "Choose Tree Depth")

        new_block_widget = NewBlockPredictionWidget(self.app_state)
        tabs.addTab(new_block_widget, "Predict New Block")

    def create_manual_sorter_tab(self, dataset: List[UrbanSite]) -> QWidget:
        container = QWidget()
        self.manual_dataset = dataset
        self.site_map = {p.display_name: p for p in self.manual_dataset}

        main_layout = QVBoxLayout(container)

        top_layout = QHBoxLayout()
        title = QLabel(
            "<b>Manual sorter:</b> Drag urban site cards into bins. "
            "Try to create groups with similar summer surface temperatures."
        )
        title.setWordWrap(True)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setFixedWidth(90)
        self.reset_button.setStyleSheet("background-color: #e0e0e0; font-weight: bold;")
        self.reset_button.clicked.connect(self.reset_manual_sorting)

        top_layout.addWidget(title)
        top_layout.addStretch()
        top_layout.addWidget(self.reset_button)
        main_layout.addLayout(top_layout)

        self.deck_list_widget = SiteListWidget("deck")
        self.bin1_list_widget = SiteListWidget("bin1")
        self.bin2_list_widget = SiteListWidget("bin2")
        self.bin1_1_list_widget = SiteListWidget("bin1_1")
        self.bin1_2_list_widget = SiteListWidget("bin1_2")
        self.bin2_1_list_widget = SiteListWidget("bin2_1")
        self.bin2_2_list_widget = SiteListWidget("bin2_2")

        level0_layout = QHBoxLayout()
        deck_frame = self.create_bin_section("Unsorted Deck", self.deck_list_widget)
        level0_layout.addStretch(1)
        level0_layout.addWidget(deck_frame, 2)
        level0_layout.addStretch(1)
        main_layout.addLayout(level0_layout)

        level1_layout = QHBoxLayout()
        bin1_frame = self.create_bin_section("Bin 1", self.bin1_list_widget)
        bin2_frame = self.create_bin_section("Bin 2", self.bin2_list_widget)
        level1_layout.addWidget(bin1_frame)
        level1_layout.addWidget(bin2_frame)
        main_layout.addLayout(level1_layout)

        level2_layout = QHBoxLayout()
        bin1_1_frame = self.create_bin_section("Bin 1,1", self.bin1_1_list_widget)
        bin1_2_frame = self.create_bin_section("Bin 1,2", self.bin1_2_list_widget)
        bin2_1_frame = self.create_bin_section("Bin 2,1", self.bin2_1_list_widget)
        bin2_2_frame = self.create_bin_section("Bin 2,2", self.bin2_2_list_widget)
        level2_layout.addWidget(bin1_1_frame)
        level2_layout.addWidget(bin1_2_frame)
        level2_layout.addWidget(bin2_1_frame)
        level2_layout.addWidget(bin2_2_frame)
        main_layout.addLayout(level2_layout, stretch=1)

        bottom_layout = QHBoxLayout()

        scoreboard_layout = QHBoxLayout()
        self.score_labels: Dict[str, QLabel] = {}
        score_frame_1 = self._create_score_frame("Split 1: Deck -> Lvl 1", "split1", "Bin 1", "Bin 2")
        score_frame_2 = self._create_score_frame("Split 2: Bin 1 -> Lvl 2", "split2", "Bin 1,1", "Bin 1,2")
        score_frame_3 = self._create_score_frame("Split 3: Bin 2 -> Lvl 2", "split3", "Bin 2,1", "Bin 2,2")
        scoreboard_layout.addWidget(score_frame_1)
        scoreboard_layout.addWidget(score_frame_2)
        scoreboard_layout.addWidget(score_frame_3)

        score_container = QWidget()
        score_container.setLayout(scoreboard_layout)
        bottom_layout.addWidget(score_container, stretch=3)

        self.detail_box = QTextEdit(readOnly=True)
        self.detail_box.setMaximumHeight(165)
        self.detail_box.setText(
            "Click an urban site card to see details.\n\n"
            "Surface temperature is the target value. The decision-tree cost is the weighted variance "
            "of surface temperature in the two child bins."
        )
        bottom_layout.addWidget(self.detail_box, stretch=1)

        main_layout.addLayout(bottom_layout)

        self.populate_deck()
        self._connect_manual_signals()
        self.update_manual_calculations()
        return container

    def reset_manual_sorting(self):
        for list_widget in [
            self.deck_list_widget,
            self.bin1_list_widget,
            self.bin2_list_widget,
            self.bin1_1_list_widget,
            self.bin1_2_list_widget,
            self.bin2_1_list_widget,
            self.bin2_2_list_widget,
        ]:
            list_widget.clear()
        self.populate_deck()
        self.update_manual_calculations()
        self.detail_box.setText("Click an urban site card to see details.")

    def _connect_manual_signals(self):
        for list_widget in [
            self.deck_list_widget,
            self.bin1_list_widget,
            self.bin2_list_widget,
            self.bin1_1_list_widget,
            self.bin1_2_list_widget,
            self.bin2_1_list_widget,
            self.bin2_2_list_widget,
        ]:
            list_widget.model().rowsInserted.connect(self.update_manual_calculations)
            list_widget.model().rowsRemoved.connect(self.update_manual_calculations)
            list_widget.itemClicked.connect(self.show_site_details)

    def show_site_details(self, item: QListWidgetItem):
        key = item.data(Qt.ItemDataRole.UserRole)
        site = self.site_map.get(key)
        if site:
            self.detail_box.setText(site.detail_text())

    def _get_sites_from_list(self, list_widget: SiteListWidget) -> List[UrbanSite]:
        out = []
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            key = item.data(Qt.ItemDataRole.UserRole)
            if key in self.site_map:
                out.append(self.site_map[key])
        return out

    def update_manual_calculations(self):
        def _update_bin_stats(site_list: List[UrbanSite], key_prefix: str):
            n = len(site_list)
            self.score_labels[f"{key_prefix}_count"].setText(f"Count (n): {n}")
            if n > 0:
                mean = float(np.mean([p.surface_temp_c for p in site_list]))
                var = float(np.var([p.surface_temp_c for p in site_list])) if n > 1 else 0.0
                self.score_labels[f"{key_prefix}_mean"].setText(f"Average Temp: {mean:.1f} C")
                self.score_labels[f"{key_prefix}_var"].setText(f"Variance: {var:.2f}")
            else:
                self.score_labels[f"{key_prefix}_mean"].setText("Average Temp: N/A")
                self.score_labels[f"{key_prefix}_var"].setText("Variance: N/A")

        sites_bin1 = self._get_sites_from_list(self.bin1_list_widget)
        sites_bin2 = self._get_sites_from_list(self.bin2_list_widget)
        sites_bin1_1 = self._get_sites_from_list(self.bin1_1_list_widget)
        sites_bin1_2 = self._get_sites_from_list(self.bin1_2_list_widget)
        sites_bin2_1 = self._get_sites_from_list(self.bin2_1_list_widget)
        sites_bin2_2 = self._get_sites_from_list(self.bin2_2_list_widget)

        _update_bin_stats(sites_bin1, "split1_binA")
        _update_bin_stats(sites_bin2, "split1_binB")
        cost1 = calculate_weighted_variance(sites_bin1, sites_bin2)
        self.score_labels["split1_cost"].setText(f"<b>TOTAL COST: {cost1:.2f}</b>")

        _update_bin_stats(sites_bin1_1, "split2_binA")
        _update_bin_stats(sites_bin1_2, "split2_binB")
        cost2 = calculate_weighted_variance(sites_bin1_1, sites_bin1_2)
        self.score_labels["split2_cost"].setText(f"<b>TOTAL COST: {cost2:.2f}</b>")

        _update_bin_stats(sites_bin2_1, "split3_binA")
        _update_bin_stats(sites_bin2_2, "split3_binB")
        cost3 = calculate_weighted_variance(sites_bin2_1, sites_bin2_2)
        self.score_labels["split3_cost"].setText(f"<b>TOTAL COST: {cost3:.2f}</b>")

    def populate_deck(self):
        for site in self.manual_dataset:
            item = QListWidgetItem(site.card_text())
            item.setData(Qt.ItemDataRole.UserRole, site.display_name)
            font = QFont("Lato")
            font.setPointSize(9)
            item.setFont(font)
            self.deck_list_widget.addItem(item)

    def create_bin_section(self, title: str, list_widget: SiteListWidget) -> QFrame:
        layout = QVBoxLayout()

        label = QLabel(f"<h3>{title}</h3>")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        layout.addWidget(list_widget)

        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setLayout(layout)
        frame.setStyleSheet(
            """
            QFrame {
                border: 3px solid #6C1D45;
                border-radius: 6px;
                background-color: #FAFAFA;
            }
            QLabel {
                color: #6C1D45;
                border: none;
            }
            QListWidget {
                border: 1px solid #333333;
                border-radius: 4px;
                background-color: white;
            }
            """
        )
        return frame

    def _create_score_frame(self, title: str, key_prefix: str, binA_name: str, binB_name: str) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(frame)

        title_label = QLabel(f"<b>{title}</b>")
        layout.addWidget(title_label)

        bins_layout = QHBoxLayout()

        binA_layout = QVBoxLayout()
        binA_label = QLabel(f"<i>{binA_name}</i>")
        self.score_labels[f"{key_prefix}_binA_count"] = QLabel("Count (n): 0")
        self.score_labels[f"{key_prefix}_binA_mean"] = QLabel("Average Temp: N/A")
        self.score_labels[f"{key_prefix}_binA_var"] = QLabel("Variance: N/A")
        binA_layout.addWidget(binA_label)
        binA_layout.addWidget(self.score_labels[f"{key_prefix}_binA_count"])
        binA_layout.addWidget(self.score_labels[f"{key_prefix}_binA_mean"])
        binA_layout.addWidget(self.score_labels[f"{key_prefix}_binA_var"])

        binB_layout = QVBoxLayout()
        binB_label = QLabel(f"<i>{binB_name}</i>")
        self.score_labels[f"{key_prefix}_binB_count"] = QLabel("Count (n): 0")
        self.score_labels[f"{key_prefix}_binB_mean"] = QLabel("Average Temp: N/A")
        self.score_labels[f"{key_prefix}_binB_var"] = QLabel("Variance: N/A")
        binB_layout.addWidget(binB_label)
        binB_layout.addWidget(self.score_labels[f"{key_prefix}_binB_count"])
        binB_layout.addWidget(self.score_labels[f"{key_prefix}_binB_mean"])
        binB_layout.addWidget(self.score_labels[f"{key_prefix}_binB_var"])

        bins_layout.addLayout(binA_layout)
        bins_layout.addLayout(binB_layout)
        layout.addLayout(bins_layout)

        self.score_labels[f"{key_prefix}_cost"] = QLabel("<b>TOTAL COST: N/A</b>")
        layout.addWidget(self.score_labels[f"{key_prefix}_cost"])

        return frame


def load_manual_dataset() -> List[UrbanSite]:
    manual_path = find_data_file(MANUAL_FILE)
    tree_path = find_data_file(TREE_FILE)

    if manual_path is not None:
        df = pd.read_csv(manual_path)
    elif tree_path is not None:
        df = pd.read_csv(tree_path).head(11)
    else:
        raise FileNotFoundError(
            f"Could not find {MANUAL_FILE} or {TREE_FILE} in the script folder/current directory."
        )

    if TARGET not in df.columns:
        raise ValueError(f"Manual dataset must contain a {TARGET} column.")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()

    if len(df) == 0:
        raise ValueError(f"Manual dataset has no usable {TARGET} values.")

    # Sort cards by temperature only for a stable deck order. Students can still
    # use canopy, pavement, vegetation, or any custom feature for sorting.
    df = df.sort_values([TARGET, "Site_ID"], na_position="last").reset_index(drop=True)

    return [UrbanSite.from_row(row) for _, row in df.iterrows()]


def apply_app_stylesheet(app: QApplication) -> None:
    app.setStyleSheet(
        """
        QWidget { color: #222222; background-color: #ffffff; }
        QMainWindow, QDialog { background-color: #ffffff; }

        QTabWidget::pane {
            border: 1px solid #cccccc;
            background-color: #ffffff;
        }
        QTabBar::tab {
            background: #555555;
            color: #ffffff;
            font-weight: bold;
            padding: 7px 14px;
            border: 1px solid #444444;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background: #6C1D45;
            color: #ffffff;
            font-weight: bold;
        }
        QTabBar::tab:!selected:hover {
            background: #666666;
        }

        QGroupBox {
            color: #222222;
            font-weight: bold;
            border: 1px solid #d6d6d6;
            border-radius: 5px;
            margin-top: 8px;
            padding-top: 10px;
            background-color: #fafafa;
        }
        QLabel { color: #222222; background-color: transparent; }
        QComboBox, QSpinBox, QTextEdit, QTableWidget {
            background-color: #ffffff;
            color: #111111;
            selection-background-color: #6c1d45;
            selection-color: #ffffff;
        }
        QComboBox:disabled, QSpinBox:disabled, QTextEdit:disabled, QTableWidget:disabled {
            background-color: #eeeeee;
            color: #555555;
        }
        QListWidget {
            background-color: #ffffff;
            color: #111111;
            alternate-background-color: #f4f4f4;
            selection-background-color: #e8d7df;
            selection-color: #111111;
        }
        QListWidget::item {
            color: #111111;
            background-color: #ffffff;
            padding: 2px;
        }
        QListWidget::item:alternate {
            background-color: #f4f4f4;
        }
        QListWidget::item:selected {
            background-color: #e8d7df;
            color: #111111;
        }
        QPushButton {
            background-color: #6C1D45;
            color: #ffffff;
            font-weight: bold;
            border: 1px solid #4f1231;
            border-radius: 4px;
            padding: 6px 10px;
        }
        QPushButton:hover { background-color: #7d2753; }
        QPushButton:pressed { background-color: #4f1231; }
        QPushButton:disabled {
            background-color: #555555;
            color: #ffffff;
            font-weight: bold;
            border: 1px solid #444444;
        }
        """
    )

def main() -> int:
    app = QApplication(sys.argv)
    apply_app_stylesheet(app)

    try:
        manual_dataset = load_manual_dataset()
    except Exception as exc:
        QMessageBox.critical(None, "Could not load urban heat dataset", str(exc))
        return 1

    window = CobberEcoTreeApp(manual_dataset)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
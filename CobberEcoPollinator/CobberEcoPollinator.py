# CobberOrchidPollinator_v10.py
# Three-tab orchid-visit clustering app for a Results/interpretation chapter.
# Updated from the working CobberOrchidPollinator script.

from __future__ import annotations

import os
import random
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.path.abspath('.')

APP_VERSION = "v10"
MAX_CLASSROOM_VISITS = 2000

COBBER_MAROON = "#6c1d45"
COBBER_GOLD = "#eaaa00"
INFOBLUE = "#3e6990"
MED_DARK_GREY = "#5f6368"
LIGHT_PANEL = "#f7f7f7"
SELECTED_ROW = "#f2dce8"

FEATURES = [
    "Dwell_Time_s",
    "Entry_Depth",
    "Contact_Fit",
    "Contact_Zone_Closeness",
    "Recurrence_Score",
]

MODEL_FEATURES = [
    "Entry_Depth",
    "Contact_Fit",
    "Contact_Zone_Closeness",
]

INTERPRETATION_FEATURES = [
    "Dwell_Time_s",
    "Recurrence_Score",
]

FEATURE_LABELS = {
    "Dwell_Time_s": "Dwell time",
    "Entry_Depth": "Entry depth",
    "Contact_Fit": "Contact fit",
    "Contact_Zone_Closeness": "Contact-zone closeness",
    "Recurrence_Score": "Recurrence score",
}

REQUIRED_COLUMNS = {
    "Visit_ID",
    "Flower_ID",
    "Visitor_Group",
    "Dwell_Time_s",
    "Entry_Depth",
    "Contact_Fit",
    "Contact_Zone_Closeness",
    "Recurrence_Score",
    "Landing_X",
    "Landing_Y",
    "Seedpod_Outcome",
}

LEGACY_OUTCOME_COLUMN = "Capsule_Outcome"
OUTCOME_COLUMN = "Seedpod_Outcome"

VISITOR_COLORS = {
    "Bumblebee": "#111111",
    "Small native bee": "#6c1d45",
    "Hoverfly": "#0047ab",
    "Beetle": "#1b4332",
    "Moth": "#7b2cbf",
    "Ant": "#6f1d1b",
}

SEEDPOD_COLORS = {
    "Seedpod formed": "#2b9348",
    "No seedpod": "#9d0208",
    "Unknown": "#5f6368",
}

CONTACT_ZONE_CENTER = (0.0, 0.08)
CONTACT_ZONE_RADIUS = 0.16


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def seedpod_fraction(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(np.mean(series.astype(str).str.lower().str.contains("formed")))


def outcome_text(value: object) -> str:
    text = str(value).strip()
    lower = text.lower()
    if "formed" in lower and "no" not in lower:
        return "Seedpod formed"
    if "not" in lower or "no" in lower or "failed" in lower:
        return "No seedpod"
    if "capsule" in lower:
        return text.replace("Capsule", "Seedpod").replace("capsule", "seedpod")
    return text if text else "Unknown"


def normalize_columns(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, MinMaxScaler]:
    out = df.copy()
    scaler = MinMaxScaler()
    out[features] = scaler.fit_transform(out[features])
    return out, scaler


def normalize_all_features(df: pd.DataFrame) -> pd.DataFrame:
    out, _ = normalize_columns(df, FEATURES)
    return out


def level_01(value: float) -> str:
    if value >= 0.70:
        return "High"
    if value >= 0.40:
        return "Medium"
    return "Low"


def dwell_level(seconds: float) -> str:
    if seconds >= 9.0:
        return "Long"
    if seconds >= 5.0:
        return "Medium"
    return "Brief"


def claim_strength(means: pd.Series) -> tuple[str, str]:
    score = (
        0.35 * float(means["Contact_Fit"])
        + 0.32 * float(means["Entry_Depth"])
        + 0.33 * float(means["Contact_Zone_Closeness"])
    )
    if score >= 0.68:
        return "strong", "stronger candidate for pollen transfer"
    if score >= 0.43:
        return "mixed", "mixed evidence for pollen transfer"
    return "weak", "weaker evidence for pollen transfer"


def visit_summary_html(row: pd.Series, include_cluster: bool = True) -> str:
    cluster_line = ""
    if include_cluster and "Visit_Cluster" in row.index:
        cluster_line = f"<b>Visit cluster:</b> {int(row['Visit_Cluster'])}<br>"
    label, color = OrchidVisualizer.classify_visit(row)
    return (
        f"<b>Visit {int(row['Visit_ID'])}</b> &nbsp; Flower {int(row['Flower_ID'])}<br>"
        f"<b>Visitor:</b> {row['Visitor_Group']}<br>"
        f"{cluster_line}"
        f"<b>Dwell time:</b> {safe_float(row['Dwell_Time_s']):.2f} s<br>"
        f"<b>Entry depth:</b> {safe_float(row['Entry_Depth']):.3f}<br>"
        f"<b>Contact fit:</b> {safe_float(row['Contact_Fit']):.3f}<br>"
        f"<b>Contact-zone closeness:</b> {safe_float(row['Contact_Zone_Closeness']):.3f}<br>"
        f"<b>Recurrence score:</b> {safe_float(row['Recurrence_Score']):.3f}<br>"
        f"<b>Seedpod outcome:</b> {row[OUTCOME_COLUMN]}<br>"
        f"<span style='color:{color}; font-weight:bold;'>{label}</span>"
    )


def apply_app_style(app: QApplication):
    app.setStyleSheet(f"""
        QMainWindow {{ background-color: white; }}
        QWidget {{ font-family: Lato, Arial, sans-serif; font-size: 11pt; }}
        QGroupBox {{
            font-weight: bold;
            border: 1px solid #b8b8b8;
            border-radius: 5px;
            margin-top: 0.70em;
            padding: 8px;
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 9px; padding: 0 4px; }}
        QPushButton {{
            background-color: {COBBER_MAROON};
            color: white;
            font-weight: bold;
            border-radius: 5px;
            padding: 7px 10px;
            border: 1px solid {COBBER_MAROON};
        }}
        QPushButton:hover {{ background-color: #7d2754; }}
        QPushButton:pressed {{ background-color: #541536; }}
        QPushButton:disabled {{
            background-color: {MED_DARK_GREY};
            color: white;
            font-weight: bold;
            border: 1px solid {MED_DARK_GREY};
        }}
        QTabBar::tab {{
            background: {MED_DARK_GREY};
            color: white;
            padding: 9px 22px;
            margin-right: 2px;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
        }}
        QTabBar::tab:selected {{
            background: {COBBER_MAROON};
            color: white;
            font-weight: bold;
        }}
        QComboBox, QSlider {{ padding: 3px; }}
        QLabel#PanelLabel {{
            background-color: {LIGHT_PANEL};
            border: 1px solid #c9c9c9;
            border-radius: 4px;
            padding: 8px;
        }}
        QLabel#ClaimLabel {{
            background-color: #fbf7fa;
            border: 1px solid #c9c9c9;
            border-radius: 5px;
            padding: 10px;
            font-size: 12pt;
        }}
        QTableWidget {{
            gridline-color: #d0d0d0;
            alternate-background-color: #f6f6f6;
            selection-background-color: {SELECTED_ROW};
            selection-color: black;
        }}
        QHeaderView::section {{
            background-color: {MED_DARK_GREY};
            color: white;
            font-weight: bold;
            padding: 4px;
            border: 1px solid white;
        }}
    """)


def find_default_dataset() -> Optional[str]:
    candidates = [
        "orchid_visit_dataset.csv",
        "OrchidVisitDataset.csv",
        "CobberOrchidPollinatorData.csv",
        "CobberOrchidPollinator.csv",
    ]
    search_dirs = [SCRIPT_DIR, os.getcwd()]
    for folder in search_dirs:
        for name in candidates:
            path = os.path.join(folder, name)
            if os.path.exists(path):
                return path
    return None


def generate_orchid_dataset(n_visits: int = MAX_CLASSROOM_VISITS, seed: Optional[int] = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    visitor_profiles = {
        "Bumblebee": dict(p=0.25, dwell=(10.5, 2.2), entry=(0.80, 0.10), contact=(0.83, 0.09), zone=(0.76, 0.11), recurrence=(0.69, 0.13), landing=(0.02, 0.03, 0.16, 0.12)),
        "Small native bee": dict(p=0.22, dwell=(5.9, 1.6), entry=(0.54, 0.13), contact=(0.56, 0.14), zone=(0.55, 0.16), recurrence=(0.58, 0.15), landing=(-0.04, 0.04, 0.24, 0.18)),
        "Hoverfly": dict(p=0.18, dwell=(3.1, 1.1), entry=(0.28, 0.11), contact=(0.25, 0.12), zone=(0.31, 0.15), recurrence=(0.34, 0.14), landing=(0.20, 0.18, 0.34, 0.25)),
        "Beetle": dict(p=0.13, dwell=(8.6, 2.8), entry=(0.37, 0.15), contact=(0.34, 0.16), zone=(0.39, 0.20), recurrence=(0.42, 0.16), landing=(-0.23, -0.10, 0.34, 0.28)),
        "Moth": dict(p=0.12, dwell=(6.2, 1.9), entry=(0.48, 0.16), contact=(0.50, 0.17), zone=(0.47, 0.17), recurrence=(0.41, 0.16), landing=(0.02, -0.04, 0.29, 0.21)),
        "Ant": dict(p=0.10, dwell=(4.3, 1.3), entry=(0.17, 0.09), contact=(0.18, 0.10), zone=(0.22, 0.13), recurrence=(0.30, 0.12), landing=(-0.31, -0.36, 0.35, 0.24)),
    }
    names = list(visitor_profiles.keys())
    probs = np.array([visitor_profiles[name]["p"] for name in names], dtype=float)
    probs = probs / probs.sum()
    rows = []
    for visit_id in range(1, n_visits + 1):
        visitor = rng.choice(names, p=probs)
        profile = visitor_profiles[visitor]
        dwell = max(0.4, rng.normal(*profile["dwell"]))
        entry = float(np.clip(rng.normal(*profile["entry"]), 0, 1))
        contact = float(np.clip(rng.normal(*profile["contact"]), 0, 1))
        zone = float(np.clip(rng.normal(*profile["zone"]), 0, 1))
        recurrence = float(np.clip(rng.normal(*profile["recurrence"]), 0, 1))
        lx_mu, ly_mu, lx_sd, ly_sd = profile["landing"]
        landing_x = float(np.clip(rng.normal(lx_mu, lx_sd), -1.08, 1.08))
        landing_y = float(np.clip(rng.normal(ly_mu, ly_sd), -1.10, 1.05))
        pollination_score = 0.30 * contact + 0.27 * entry + 0.16 * zone + 0.12 * recurrence + 0.15 * min(dwell / 12.0, 1.0)
        probability = float(np.clip(0.05 + 0.82 * pollination_score, 0.02, 0.92))
        outcome = "Seedpod formed" if rng.random() < probability else "No seedpod"
        rows.append(
            {
                "Visit_ID": visit_id,
                "Flower_ID": int(rng.integers(1, 91)),
                "Visitor_Group": visitor,
                "Dwell_Time_s": round(dwell, 2),
                "Entry_Depth": round(entry, 3),
                "Contact_Fit": round(contact, 3),
                "Contact_Zone_Closeness": round(zone, 3),
                "Recurrence_Score": round(recurrence, 3),
                "Landing_X": round(landing_x, 3),
                "Landing_Y": round(landing_y, 3),
                OUTCOME_COLUMN: outcome,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Orchid drawing and visit visualization
# -----------------------------------------------------------------------------

class OrchidVisualizer:
    @staticmethod
    def draw_orchid(ax):
        """Draw a simplified pink lady's slipper flower for the landing diagram."""
        ax.set_aspect("equal", "box")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.35, 1.25)
        ax.axis("off")

        ax.plot([0, 0], [-1.25, -0.35], color="#2b9348", lw=5, solid_capstyle="round", zorder=1)
        ax.add_patch(Ellipse((-0.38, -0.95), 0.72, 0.24, angle=25, facecolor="#67b567", edgecolor="#266b35", lw=1.1, zorder=1))
        ax.add_patch(Ellipse((0.42, -0.90), 0.78, 0.25, angle=-24, facecolor="#67b567", edgecolor="#266b35", lw=1.1, zorder=1))

        ax.add_patch(Ellipse((0.0, 0.72), 0.34, 0.72, facecolor="#8b5a2b", edgecolor="#4d2d17", lw=1.2, alpha=0.95, zorder=2))
        ax.add_patch(Ellipse((-0.47, 0.34), 0.18, 0.88, angle=-58, facecolor="#9b3156", edgecolor="#5d1730", lw=1.1, alpha=0.92, zorder=2))
        ax.add_patch(Ellipse((0.47, 0.34), 0.18, 0.88, angle=58, facecolor="#9b3156", edgecolor="#5d1730", lw=1.1, alpha=0.92, zorder=2))

        verts = [
            (-0.48, 0.18), (-0.62, -0.18), (-0.36, -0.74), (0.0, -0.82),
            (0.36, -0.74), (0.62, -0.18), (0.48, 0.18), (0.20, 0.02),
            (-0.20, 0.02), (-0.48, 0.18),
        ]
        codes = [Path.MOVETO] + [Path.CURVE3] * 8 + [Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), facecolor="#e98bb8", edgecolor="#8a2f5b", lw=1.6, alpha=0.95, zorder=3))

        for x in [-0.26, -0.14, 0.0, 0.14, 0.26]:
            ax.plot([x, 0.65 * x], [-0.68, 0.02], color="#a23b72", lw=0.7, alpha=0.55, zorder=4)

        ax.add_patch(Ellipse((0.0, 0.10), 0.42, 0.22, facecolor="#f5d6e7", edgecolor="#8a2f5b", lw=1.0, zorder=5))
        zone = Circle(CONTACT_ZONE_CENTER, CONTACT_ZONE_RADIUS, facecolor="none", edgecolor="#d62828", lw=1.6, linestyle="--", zorder=7)
        ax.add_patch(zone)
        ax.text(CONTACT_ZONE_CENTER[0], CONTACT_ZONE_CENTER[1] + 0.22, "contact zone", color="#d62828", fontsize=8, ha="center", zorder=8)

    @staticmethod
    def draw_insect(ax, visitor_group: str, x: float, y: float):
        color = VISITOR_COLORS.get(visitor_group, "#495057")
        accent = COBBER_GOLD if visitor_group == "Bumblebee" else "#ff8fab"
        dx = -0.38 if x > 0 else 0.38
        dy = 0.30 if y < 0.0 else -0.25
        ax.add_patch(FancyArrowPatch((x + dx, y + dy), (x, y), arrowstyle="->", mutation_scale=13, lw=1.4, color="#495057", zorder=8))
        ax.add_patch(Ellipse((x - 0.035, y + 0.055), 0.16, 0.07, angle=32, facecolor="#dbeafe", edgecolor="#94a3b8", lw=0.7, alpha=0.65, zorder=9))
        ax.add_patch(Ellipse((x + 0.045, y - 0.050), 0.16, 0.07, angle=32, facecolor="#dbeafe", edgecolor="#94a3b8", lw=0.7, alpha=0.65, zorder=9))
        ax.add_patch(Ellipse((x, y), 0.18, 0.11, angle=25, facecolor=color, edgecolor="black", lw=1.0, zorder=10))
        ax.add_patch(Ellipse((x, y), 0.09, 0.052, angle=25, facecolor=accent, edgecolor="none", alpha=0.9, zorder=11))
        ax.plot([x], [y], marker="o", color="black", markersize=2.5, zorder=12)

    @staticmethod
    def classify_visit(row: pd.Series) -> tuple[str, str]:
        dwell_score = min(safe_float(row["Dwell_Time_s"]) / 12.0, 1.0)
        entry = safe_float(row["Entry_Depth"])
        contact = safe_float(row["Contact_Fit"])
        distance_score = safe_float(row["Contact_Zone_Closeness"])
        recurrence = safe_float(row["Recurrence_Score"])
        score = 0.30 * contact + 0.27 * entry + 0.17 * dwell_score + 0.16 * distance_score + 0.10 * recurrence
        if score >= 0.68:
            return "High-contact candidate visit", "#2b9348"
        if score >= 0.43:
            return "Moderate-contact visit", "#c77d00"
        return "Low-contact or off-target visit", "#c1121f"

    @staticmethod
    def bar_interpretation(label: str, value: float, raw_value: float) -> str:
        if label == "Dwell time":
            if raw_value >= 10:
                return "long visit"
            if raw_value >= 5:
                return "medium visit"
            return "brief visit"
        if value >= 0.70:
            return "strong evidence"
        if value >= 0.40:
            return "mixed evidence"
        return "weak evidence"

    @staticmethod
    def draw_feature_bars(ax, row: pd.Series):
        ax.clear()
        ax.set_xlim(0, 1.22)
        ax.set_ylim(-1.05, 5.85)
        ax.axis("off")
        metrics = [
            ("Dwell time", min(safe_float(row["Dwell_Time_s"]) / 16.0, 1.0), safe_float(row["Dwell_Time_s"]), f"{safe_float(row['Dwell_Time_s']):.2f} s", "#3a86ff"),
            ("Entry depth", safe_float(row["Entry_Depth"]), safe_float(row["Entry_Depth"]), f"{safe_float(row['Entry_Depth']):.3f}", "#06a77d"),
            ("Contact fit", safe_float(row["Contact_Fit"]), safe_float(row["Contact_Fit"]), f"{safe_float(row['Contact_Fit']):.3f}", "#ff006e"),
            ("Contact-zone closeness", safe_float(row["Contact_Zone_Closeness"]), safe_float(row["Contact_Zone_Closeness"]), f"{safe_float(row['Contact_Zone_Closeness']):.3f}", "#f77f00"),
            ("Recurrence score", safe_float(row["Recurrence_Score"]), safe_float(row["Recurrence_Score"]), f"{safe_float(row['Recurrence_Score']):.3f}", "#8338ec"),
        ]
        y_positions = [5.05, 3.95, 2.85, 1.75, 0.65]
        for (label, value, raw_value, text, color), y in zip(metrics, y_positions):
            ax.text(0.02, y + 0.37, label, fontsize=10, fontweight="bold", ha="left", va="center")
            ax.add_patch(Rectangle((0.02, y), 0.78, 0.22, facecolor="#e9ecef", edgecolor="#adb5bd", lw=0.8))
            ax.add_patch(Rectangle((0.02, y), 0.78 * max(0.0, min(value, 1.0)), 0.22, facecolor=color, edgecolor="none", alpha=0.95))
            ax.text(0.84, y + 0.15, text, fontsize=9.5, ha="left", va="center")
            ax.text(0.84, y - 0.09, OrchidVisualizer.bar_interpretation(label, value, raw_value), fontsize=8.5, color="#444444", ha="left", va="center")

        label, label_color = OrchidVisualizer.classify_visit(row)
        ax.text(0.02, -0.03, label, fontsize=10, fontweight="bold", color=label_color, ha="left", va="center")
        ax.text(0.02, -0.30, f"Visit {int(row['Visit_ID'])}   Flower {int(row['Flower_ID'])}", fontsize=10, ha="left", va="center")
        ax.text(0.02, -0.55, f"Visitor: {row['Visitor_Group']}", fontsize=10, ha="left", va="center")
        ax.text(0.02, -0.80, f"Seedpod outcome: {row[OUTCOME_COLUMN]}", fontsize=10, ha="left", va="center")

    @classmethod
    def plot_single_visit(cls, fig: Figure, row: pd.Series):
        fig.clear()
        gs = fig.add_gridspec(1, 2, width_ratios=[1.30, 1.0])
        ax_scene = fig.add_subplot(gs[0, 0])
        ax_bars = fig.add_subplot(gs[0, 1])
        cls.draw_orchid(ax_scene)
        cls.draw_insect(ax_scene, str(row["Visitor_Group"]), safe_float(row["Landing_X"]), safe_float(row["Landing_Y"]))
        ax_scene.set_title(f"Single visit view\nFlower {int(row['Flower_ID'])} × {row['Visitor_Group']}", fontsize=12)
        cls.draw_feature_bars(ax_bars, row)
        fig.tight_layout()


class VisitPopup(QWidget):
    def __init__(self, row: pd.Series, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Visit {int(row['Visit_ID'])} summary")
        self.setGeometry(160, 140, 900, 520)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        layout = QHBoxLayout(self)

        fig = Figure(figsize=(6.3, 4.3))
        canvas = FigureCanvas(fig)
        OrchidVisualizer.plot_single_visit(fig, row)
        layout.addWidget(canvas, 2)

        right = QVBoxLayout()
        label = QLabel(visit_summary_html(row))
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setObjectName("PanelLabel")
        label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        right.addWidget(label, 1)

        close_button = QPushButton("Close Visit Summary")
        close_button.clicked.connect(self.close)
        right.addWidget(close_button)
        layout.addLayout(right, 1)


# -----------------------------------------------------------------------------
# Data tab
# -----------------------------------------------------------------------------

class DataTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.dataset: Optional[pd.DataFrame] = None
        self.timer = QTimer(self)
        self.animation_indices: list[int] = []
        self.current_index = 0
        self.scatter_artist = None
        self.scatter_rows: list[int] = []
        self.popups: list[VisitPopup] = []
        self.current_plot_mode = "2D Scatter Plot"
        self.current_plot_features = ("Entry_Depth", "Contact_Fit", "Contact_Zone_Closeness")

        self.layout = QVBoxLayout(self)

        top = QHBoxLayout()
        self.generate_button = QPushButton("Generate New Dataset")
        self.generate_button.clicked.connect(self.generate_new_dataset)
        top.addWidget(self.generate_button)
        self.data_info_label = QLabel("Dataset will load automatically.")
        self.data_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(self.data_info_label, 1)
        self.layout.addLayout(top)

        main = QHBoxLayout()
        left = QVBoxLayout()

        plotting_box = QGroupBox("Data visualization")
        plotting_layout = QVBoxLayout(plotting_box)
        self.plot_mode_combo = QComboBox()
        self.plot_mode_combo.addItems(["2D Scatter Plot", "3D Scatter Plot"])
        self.plot_mode_combo.currentTextChanged.connect(self.update_plot_controls)
        plotting_layout.addWidget(self.plot_mode_combo)
        self.x_axis_combo = self._labeled_feature_combo("X:", plotting_layout)
        self.y_axis_combo = self._labeled_feature_combo("Y:", plotting_layout)
        self.z_axis_combo = self._labeled_feature_combo("Z:", plotting_layout)
        self.plot_button = QPushButton("Plot 2D Scatter")
        self.plot_button.clicked.connect(self.plot_data)
        plotting_layout.addWidget(self.plot_button)
        note = QLabel("Data plots use normalized feature values. The Model tab will match this 2D or 3D view after K-means runs.")
        note.setWordWrap(True)
        plotting_layout.addWidget(note)
        left.addWidget(plotting_box)

        explorer_box = QGroupBox("Single visit explorer")
        explorer_layout = QVBoxLayout(explorer_box)
        self.random_button = QPushButton("Show Single Random Visit")
        self.random_button.clicked.connect(self.show_random_visit)
        explorer_layout.addWidget(self.random_button)
        self.animate_button = QPushButton("Animate Random Visits")
        self.animate_button.clicked.connect(self.animate_visits)
        explorer_layout.addWidget(self.animate_button)
        speed = QHBoxLayout()
        speed.addWidget(QLabel("Slower"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(150, 1400)
        self.speed_slider.setValue(850)
        self.speed_slider.setInvertedAppearance(True)
        speed.addWidget(self.speed_slider)
        speed.addWidget(QLabel("Faster"))
        explorer_layout.addLayout(speed)
        left.addWidget(explorer_box)
        left.addStretch()
        main.addLayout(left, 0)

        self.figure = Figure(figsize=(10.6, 6.7))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("pick_event", self.on_pick)
        main.addWidget(self.canvas, 1)
        self.layout.addLayout(main, 1)

        self.timer.timeout.connect(self._animation_step)
        self.speed_slider.valueChanged.connect(lambda value: self.timer.setInterval(value))
        self.populate_feature_combos()

    def _labeled_feature_combo(self, text: str, parent_layout: QVBoxLayout) -> QComboBox:
        row = QHBoxLayout()
        row.addWidget(QLabel(text))
        combo = QComboBox()
        row.addWidget(combo)
        parent_layout.addLayout(row)
        return combo

    def populate_feature_combos(self):
        for combo in [self.x_axis_combo, self.y_axis_combo, self.z_axis_combo]:
            combo.clear()
            for feature in FEATURES:
                combo.addItem(FEATURE_LABELS.get(feature, feature), feature)
        self.x_axis_combo.setCurrentIndex(FEATURES.index("Entry_Depth"))
        self.y_axis_combo.setCurrentIndex(FEATURES.index("Contact_Fit"))
        self.z_axis_combo.setCurrentIndex(FEATURES.index("Contact_Zone_Closeness"))
        self.update_plot_controls(self.plot_mode_combo.currentText())

    def update_plot_controls(self, text: str):
        is_3d = text == "3D Scatter Plot"
        self.z_axis_combo.setEnabled(is_3d)
        self.plot_button.setText("Plot 3D Scatter" if is_3d else "Plot 2D Scatter")

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if OUTCOME_COLUMN not in df.columns and LEGACY_OUTCOME_COLUMN in df.columns:
            df[OUTCOME_COLUMN] = df[LEGACY_OUTCOME_COLUMN].apply(outcome_text)
        elif OUTCOME_COLUMN in df.columns:
            df[OUTCOME_COLUMN] = df[OUTCOME_COLUMN].apply(outcome_text)
        if "Contact_Zone_Closeness" not in df.columns and "Landing_Distance_From_Contact_Zone" in df.columns:
            df["Contact_Zone_Closeness"] = 1.0 - np.clip(df["Landing_Distance_From_Contact_Zone"].astype(float) / 0.80, 0.0, 1.0)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
        for feature in FEATURES + ["Landing_X", "Landing_Y"]:
            df[feature] = pd.to_numeric(df[feature], errors="coerce")
        df = df.dropna(subset=FEATURES + ["Landing_X", "Landing_Y"]).reset_index(drop=True)
        if len(df) > MAX_CLASSROOM_VISITS:
            df = df.sample(n=MAX_CLASSROOM_VISITS, random_state=42).sort_values("Visit_ID").reset_index(drop=True)
        return df

    def load_dataframe(self, df: pd.DataFrame, source: str):
        self.dataset = self.clean_dataset(df)
        if "Visit_Cluster" in self.dataset.columns:
            self.dataset = self.dataset.drop(columns=["Visit_Cluster"])
        self.data_info_label.setText(
            f"Loaded: {source}    {len(self.dataset):,} visits shown    {self.dataset['Flower_ID'].nunique():,} flowers"
        )
        self.clear_plot_area()
        self.main_window.model_tab.clear_model()
        self.main_window.results_tab.clear_results()

    def auto_load_or_generate(self):
        path = find_default_dataset()
        if path:
            try:
                self.load_dataframe(pd.read_csv(path), os.path.basename(path))
                return
            except Exception as exc:
                QMessageBox.warning(self, "Dataset Load Error", f"Could not auto-load the dataset, so a new dataset was generated.\n\n{exc}")
        self.load_dataframe(generate_orchid_dataset(seed=42), "generated orchid visit dataset")

    def generate_new_dataset(self):
        seed = random.randint(1, 999_999)
        self.load_dataframe(generate_orchid_dataset(seed=seed), f"generated orchid visit dataset (seed {seed})")

    def clear_plot_area(self):
        self.figure.clear()
        self.scatter_artist = None
        self.scatter_rows = []
        self.canvas.draw()

    def plot_data(self):
        if self.dataset is None:
            QMessageBox.warning(self, "Error", "Please generate or load a dataset first.")
            return
        df_plot = normalize_all_features(self.dataset)
        self.figure.clear()
        self.scatter_rows = list(df_plot.index)
        mode = self.plot_mode_combo.currentText()
        x = self.x_axis_combo.currentData()
        y = self.y_axis_combo.currentData()
        z = self.z_axis_combo.currentData()
        self.current_plot_mode = mode
        self.current_plot_features = (x, y, z)

        if mode == "3D Scatter Plot":
            ax = self.figure.add_subplot(111, projection="3d")
            ax.scatter(
                df_plot[x], df_plot[y], df_plot[z], alpha=0.66, s=20, color=INFOBLUE,
                edgecolors="white", linewidths=0.25
            )
            self.scatter_artist = ax.scatter(df_plot[x], df_plot[y], df_plot[z], alpha=0.0, s=55, picker=True, pickradius=5)
            ax.set_xlabel(FEATURE_LABELS.get(x, x), fontweight="bold")
            ax.set_ylabel(FEATURE_LABELS.get(y, y), fontweight="bold")
            ax.set_zlabel(FEATURE_LABELS.get(z, z), fontweight="bold")
            ax.set_title("Normalized orchid visit feature space")
            self.figure.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.03)
        else:
            ax = self.figure.add_subplot(111)
            ax.scatter(
                df_plot[x], df_plot[y], alpha=0.66, s=20, color=INFOBLUE,
                edgecolors="white", linewidths=0.25
            )
            self.scatter_artist = ax.scatter(df_plot[x], df_plot[y], alpha=0.0, s=55, picker=True, pickradius=5)
            ax.set_xlabel(FEATURE_LABELS.get(x, x), fontweight="bold")
            ax.set_ylabel(FEATURE_LABELS.get(y, y), fontweight="bold")
            ax.set_title("Normalized orchid visit feature space")
            ax.grid(True, alpha=0.28)
            self.figure.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.11)
        self.canvas.draw()

    def on_pick(self, event):
        if self.dataset is None or event.artist is not self.scatter_artist:
            return
        if not hasattr(event, "ind") or len(event.ind) == 0:
            return
        row_index = self.scatter_rows[int(event.ind[0])]
        self.show_visit_popup(self.dataset.iloc[row_index])

    def show_visit_popup(self, row: pd.Series):
        popup = VisitPopup(row, self)
        self.popups.append(popup)
        popup.show()

    def show_random_visit(self):
        if self.dataset is None:
            QMessageBox.warning(self, "Error", "Please generate or load a dataset first.")
            return
        row = self.dataset.sample(n=1).iloc[0]
        OrchidVisualizer.plot_single_visit(self.figure, row)
        self.canvas.draw()

    def animate_visits(self):
        if self.dataset is None:
            QMessageBox.warning(self, "Error", "Please generate or load a dataset first.")
            return
        self.animation_indices = random.sample(range(len(self.dataset)), min(80, len(self.dataset)))
        self.current_index = 0
        self.timer.start(self.speed_slider.value())

    def _animation_step(self):
        if self.dataset is None:
            self.timer.stop()
            return
        if self.current_index >= len(self.animation_indices):
            self.timer.stop()
            QMessageBox.information(self, "Animation Complete", "Finished showing random orchid visits.")
            return
        row = self.dataset.iloc[self.animation_indices[self.current_index]]
        OrchidVisualizer.plot_single_visit(self.figure, row)
        self.canvas.draw()
        self.current_index += 1


# -----------------------------------------------------------------------------
# Model tab
# -----------------------------------------------------------------------------

class ModelTab(QWidget):
    def __init__(self, data_tab: DataTab, main_window):
        super().__init__()
        self.data_tab = data_tab
        self.main_window = main_window
        self.trained_model: Optional[KMeans] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.scatter_artist = None
        self.scatter_rows: list[int] = []
        self.popups: list[VisitPopup] = []

        self.layout = QVBoxLayout(self)
        main = QHBoxLayout()

        left = QVBoxLayout()
        setup_box = QGroupBox("K-means clustering")
        setup_layout = QVBoxLayout(setup_box)
        row = QHBoxLayout()
        row.addWidget(QLabel("Number of visit clusters (k):"))
        self.k_combo = QComboBox()
        self.k_combo.addItems([str(i) for i in range(2, 9)])
        self.k_combo.setCurrentText("4")
        row.addWidget(self.k_combo)
        setup_layout.addLayout(row)
        self.run_button = QPushButton("Run K-Means")
        self.run_button.clicked.connect(self.run_clustering)
        setup_layout.addWidget(self.run_button)
        model_note = QLabel(
            "K-means groups visits using entry depth, contact fit, and contact-zone closeness. "
            "The map matches the 2D or 3D view selected on the Data tab."
        )
        model_note.setWordWrap(True)
        setup_layout.addWidget(model_note)
        left.addWidget(setup_box)
        left.addStretch()
        main.addLayout(left, 0)

        self.figure = Figure(figsize=(10.8, 7.0))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("pick_event", self.on_pick)
        main.addWidget(self.canvas, 1)
        self.layout.addLayout(main, 1)

    def clear_model(self):
        self.trained_model = None
        self.scaler = None
        self.scatter_artist = None
        self.scatter_rows = []
        self.figure.clear()
        self.canvas.draw()

    def run_clustering(self):
        if self.data_tab.dataset is None:
            QMessageBox.warning(self, "Error", "Please generate or load a dataset first.")
            return
        n_clusters = int(self.k_combo.currentText())
        features = self.data_tab.dataset[MODEL_FEATURES].copy()
        self.scaler = MinMaxScaler()
        features_for_model = self.scaler.fit_transform(features)
        try:
            model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
            self.data_tab.dataset["Visit_Cluster"] = model.fit_predict(features_for_model) + 1
            self.trained_model = model
            self.plot_model()
            self.main_window.results_tab.on_clustering_complete()
            self.main_window.tabs.setCurrentWidget(self)
        except Exception as exc:
            QMessageBox.critical(self, "Clustering Error", f"An error occurred:\n{exc}")

    def plot_model(self):
        if self.data_tab.dataset is None or "Visit_Cluster" not in self.data_tab.dataset.columns:
            return
        df_plot = normalize_all_features(self.data_tab.dataset)
        self.scatter_rows = list(df_plot.index)
        self.figure.clear()
        self.scatter_artist = None

        mode = self.data_tab.current_plot_mode
        x, y, z = self.data_tab.current_plot_features
        clusters = sorted(self.data_tab.dataset["Visit_Cluster"].dropna().unique())
        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0.08, 0.92, max(len(clusters), 2)))
        color_map = dict(zip(clusters, colors))

        if mode == "3D Scatter Plot":
            ax = self.figure.add_subplot(111, projection="3d")
            for cluster, group in df_plot.groupby(self.data_tab.dataset["Visit_Cluster"]):
                ax.scatter(
                    group[x], group[y], group[z],
                    alpha=0.76, s=22, color=color_map.get(cluster, INFOBLUE),
                    edgecolors="white", linewidths=0.25, label=f"Cluster {int(cluster)}"
                )
            centers = df_plot.groupby(self.data_tab.dataset["Visit_Cluster"])[[x, y, z]].mean().sort_index()
            ax.scatter(
                centers[x], centers[y], centers[z],
                marker="X", s=190, color=COBBER_MAROON,
                edgecolors="black", linewidths=0.8, label="Cluster centers"
            )
            self.scatter_artist = ax.scatter(df_plot[x], df_plot[y], df_plot[z], alpha=0.0, s=55, picker=True, pickradius=6)
            ax.set_xlabel(FEATURE_LABELS.get(x, x), fontweight="bold")
            ax.set_ylabel(FEATURE_LABELS.get(y, y), fontweight="bold")
            ax.set_zlabel(FEATURE_LABELS.get(z, z), fontweight="bold")
            ax.set_title("K-means clusters shown in the selected 3D view")
            ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
            self.figure.subplots_adjust(left=0.02, right=0.82, top=0.92, bottom=0.03)
        else:
            ax = self.figure.add_subplot(111)
            for cluster, group in df_plot.groupby(self.data_tab.dataset["Visit_Cluster"]):
                ax.scatter(
                    group[x], group[y],
                    alpha=0.76, s=24, color=color_map.get(cluster, INFOBLUE),
                    edgecolors="white", linewidths=0.25, label=f"Cluster {int(cluster)}"
                )
            centers = df_plot.groupby(self.data_tab.dataset["Visit_Cluster"])[[x, y]].mean().sort_index()
            ax.scatter(
                centers[x], centers[y],
                marker="X", s=170, color=COBBER_MAROON,
                edgecolors="black", linewidths=0.8, label="Cluster centers"
            )
            self.scatter_artist = ax.scatter(df_plot[x], df_plot[y], alpha=0.0, s=55, picker=True, pickradius=6)
            ax.set_xlabel(FEATURE_LABELS.get(x, x), fontweight="bold")
            ax.set_ylabel(FEATURE_LABELS.get(y, y), fontweight="bold")
            ax.set_title("K-means clusters shown in the selected 2D view")
            ax.grid(True, alpha=0.28)
            ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
            self.figure.subplots_adjust(left=0.08, right=0.82, top=0.92, bottom=0.11)
        self.canvas.draw()

    def on_pick(self, event):
        if self.data_tab.dataset is None or event.artist is not self.scatter_artist:
            return
        if not hasattr(event, "ind") or len(event.ind) == 0:
            return
        row_index = self.scatter_rows[int(event.ind[0])]
        popup = VisitPopup(self.data_tab.dataset.iloc[row_index], self)
        self.popups.append(popup)
        popup.show()


# -----------------------------------------------------------------------------
# Results tab
# -----------------------------------------------------------------------------

class ResultsTab(QWidget):
    def __init__(self, data_tab: DataTab, model_tab: ModelTab, main_window):
        super().__init__()
        self.data_tab = data_tab
        self.model_tab = model_tab
        self.main_window = main_window
        self.selected_cluster: Optional[int] = None

        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(6)

        top = QHBoxLayout()
        controls_box = QGroupBox("Result controls")
        controls_layout = QVBoxLayout(controls_box)
        controls_layout.addWidget(QLabel("Selected visit cluster:"))
        self.cluster_combo = QComboBox()
        self.cluster_combo.currentIndexChanged.connect(self.on_cluster_changed)
        controls_layout.addWidget(self.cluster_combo)
        controls_box.setMaximumWidth(260)
        top.addWidget(controls_box, 0, Qt.AlignmentFlag.AlignTop)

        claim_box = QGroupBox("Ecological interpretation claim")
        claim_layout = QVBoxLayout(claim_box)
        self.claim_label = QLabel("Run K-means to interpret visit clusters.")
        self.claim_label.setObjectName("ClaimLabel")
        self.claim_label.setWordWrap(True)
        self.claim_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        claim_layout.addWidget(self.claim_label)
        top.addWidget(claim_box, 1, Qt.AlignmentFlag.AlignTop)
        self.layout.addLayout(top, 0)

        self.summary_box = QGroupBox("Selected cluster summary")
        summary_layout = QGridLayout(self.summary_box)
        summary_layout.setHorizontalSpacing(18)
        summary_layout.setVerticalSpacing(4)
        self.summary_labels: dict[str, QLabel] = {}
        summary_items = [
            ("Visits", "visits"),
            ("Main visitor", "visitor"),
            ("Seedpod", "seedpod"),
            ("Entry", "entry"),
            ("Contact", "contact"),
            ("Zone", "zone"),
            ("Dwell", "dwell"),
            ("Recurrence", "recurrence"),
        ]
        for i, (title, key) in enumerate(summary_items):
            row = i // 4
            col = i % 4
            label = QLabel(f"<b>{title}:</b> —")
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            summary_layout.addWidget(label, row, col)
            self.summary_labels[key] = label
        self.layout.addWidget(self.summary_box, 0)

        plots = QHBoxLayout()
        plots.setSpacing(8)
        self.heatmap_figure = Figure(figsize=(4.6, 3.3))
        self.heatmap_canvas = FigureCanvas(self.heatmap_figure)
        plots.addWidget(self.heatmap_canvas, 1)

        self.seedpod_figure = Figure(figsize=(4.7, 3.3))
        self.seedpod_canvas = FigureCanvas(self.seedpod_figure)
        plots.addWidget(self.seedpod_canvas, 1)

        self.visitor_figure = Figure(figsize=(5.8, 3.3))
        self.visitor_canvas = FigureCanvas(self.visitor_figure)
        plots.addWidget(self.visitor_canvas, 1)
        self.layout.addLayout(plots, 1)

        table_box = QGroupBox("Cluster comparison table")
        table_layout = QVBoxLayout(table_box)
        self.comparison_table = QTableWidget()
        self.comparison_table.setAlternatingRowColors(True)
        self.comparison_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.comparison_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.comparison_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.comparison_table.cellClicked.connect(self.on_table_cell_clicked)
        table_layout.addWidget(self.comparison_table)
        self.layout.addWidget(table_box, 1)

        self.cluster_combo.setEnabled(False)
        self.clear_results()

    def clear_results(self):
        self.selected_cluster = None
        self.cluster_combo.setEnabled(False)
        self.cluster_combo.blockSignals(True)
        self.cluster_combo.clear()
        self.cluster_combo.blockSignals(False)
        self.claim_label.setText("Run K-means to interpret visit clusters.")
        for key, label in self.summary_labels.items():
            label.setText("—")
        for fig, canvas in [
            (self.heatmap_figure, self.heatmap_canvas),
            (self.seedpod_figure, self.seedpod_canvas),
            (self.visitor_figure, self.visitor_canvas),
        ]:
            fig.clear()
            canvas.draw()
        self.comparison_table.clear()
        self.comparison_table.setRowCount(0)
        self.comparison_table.setColumnCount(0)

    def on_clustering_complete(self):
        data = self.data_tab.dataset
        if data is None or "Visit_Cluster" not in data.columns:
            return
        clusters = sorted(data["Visit_Cluster"].dropna().unique())
        self.cluster_combo.blockSignals(True)
        self.cluster_combo.clear()
        for cluster in clusters:
            self.cluster_combo.addItem(f"Cluster {int(cluster)}", int(cluster))
        self.cluster_combo.blockSignals(False)
        self.selected_cluster = int(clusters[0]) if clusters else None
        self.cluster_combo.setEnabled(True)
        self.plot_all()

    def on_cluster_changed(self, index: int):
        if index < 0:
            self.selected_cluster = None
        else:
            self.selected_cluster = int(self.cluster_combo.itemData(index))
        self.plot_all()

    def on_table_cell_clicked(self, row: int, column: int):
        item = self.comparison_table.item(row, 0)
        if item is None:
            return
        try:
            cluster = int(item.text())
        except ValueError:
            return
        idx = self.cluster_combo.findData(cluster)
        if idx >= 0:
            self.cluster_combo.setCurrentIndex(idx)

    def _cluster_data(self) -> pd.DataFrame:
        data = self.data_tab.dataset
        if data is None or self.selected_cluster is None or "Visit_Cluster" not in data.columns:
            return pd.DataFrame()
        return data[data["Visit_Cluster"] == self.selected_cluster].copy()

    def plot_all(self):
        if self.data_tab.dataset is None or "Visit_Cluster" not in self.data_tab.dataset.columns:
            return
        self.update_summary_and_claim()
        self.plot_landing_heatmap()
        self.plot_seedpod_comparison()
        self.plot_visitor_distribution()
        self.update_comparison_table()

    def update_summary_and_claim(self):
        cluster_data = self._cluster_data()
        if cluster_data.empty:
            return
        means = cluster_data[FEATURES].mean()
        seedpod_pct = 100.0 * seedpod_fraction(cluster_data[OUTCOME_COLUMN])
        top_visitor = cluster_data["Visitor_Group"].value_counts().idxmax()
        size = len(cluster_data)
        strength_word, strength_phrase = claim_strength(means)

        all_seedpods = self.data_tab.dataset.groupby("Visit_Cluster")[OUTCOME_COLUMN].apply(lambda s: 100.0 * seedpod_fraction(s))
        median_seedpod = float(all_seedpods.median()) if len(all_seedpods) else seedpod_pct
        if seedpod_pct >= median_seedpod + 8:
            outcome_phrase = "Seedpod generation is higher than the middle cluster outcome."
        elif seedpod_pct <= median_seedpod - 8:
            outcome_phrase = "Seedpod generation is lower than the middle cluster outcome."
        else:
            outcome_phrase = "Seedpod generation is near the middle cluster outcome."

        claim = (
            f"Cluster {self.selected_cluster} is mostly {top_visitor.lower()} visits. "
            f"It has {level_01(means['Entry_Depth']).lower()} entry depth, "
            f"{level_01(means['Contact_Fit']).lower()} contact fit, and "
            f"{level_01(means['Contact_Zone_Closeness']).lower()} contact-zone closeness. "
            f"This is {strength_word} evidence: a {strength_phrase}. "
            f"{outcome_phrase}"
        )
        self.claim_label.setText(claim)

        self.summary_labels["visits"].setText(f"<b>Visits:</b> {size:,}")
        self.summary_labels["visitor"].setText(f"<b>Main visitor:</b> {top_visitor}")
        self.summary_labels["seedpod"].setText(f"<b>Seedpod:</b> {seedpod_pct:.1f}%")
        self.summary_labels["entry"].setText(f"<b>Entry:</b> {means['Entry_Depth']:.3f}")
        self.summary_labels["contact"].setText(f"<b>Contact:</b> {means['Contact_Fit']:.3f}")
        self.summary_labels["zone"].setText(f"<b>Zone:</b> {means['Contact_Zone_Closeness']:.3f}")
        self.summary_labels["dwell"].setText(f"<b>Dwell:</b> {means['Dwell_Time_s']:.2f} s")
        self.summary_labels["recurrence"].setText(f"<b>Recurrence:</b> {means['Recurrence_Score']:.3f}")

    def plot_landing_heatmap(self):
        cluster_data = self._cluster_data()
        self.heatmap_figure.clear()
        ax = self.heatmap_figure.add_subplot(111)
        OrchidVisualizer.draw_orchid(ax)
        if not cluster_data.empty:
            x = cluster_data["Landing_X"].to_numpy(dtype=float)
            y = cluster_data["Landing_Y"].to_numpy(dtype=float)
            heatmap, xedges, yedges = np.histogram2d(x, y, bins=45, range=[[-1.25, 1.25], [-1.35, 1.25]])
            ax.imshow(
                heatmap.T,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                origin="lower",
                cmap="hot",
                alpha=0.55,
                aspect="equal",
                zorder=20,
            )
            zone = Circle(CONTACT_ZONE_CENTER, CONTACT_ZONE_RADIUS, facecolor="none", edgecolor="#8ecae6", lw=1.8, linestyle="--", zorder=30)
            ax.add_patch(zone)
            ax.text(CONTACT_ZONE_CENTER[0], CONTACT_ZONE_CENTER[1] + 0.22, "contact zone", color="#8ecae6", fontsize=8, ha="center", zorder=31)
        ax.set_title(f"Landing positions for cluster {self.selected_cluster}", fontsize=10)
        self.heatmap_figure.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02)
        self.heatmap_canvas.draw()

    def plot_seedpod_comparison(self):
        data = self.data_tab.dataset
        self.seedpod_figure.clear()
        ax = self.seedpod_figure.add_subplot(111)
        if data is None or "Visit_Cluster" not in data.columns:
            self.seedpod_canvas.draw()
            return
        cluster_pct = data.groupby("Visit_Cluster")[OUTCOME_COLUMN].apply(lambda s: 100.0 * seedpod_fraction(s)).sort_index()
        colors = ["#8a8f94"] * len(cluster_pct)
        if self.selected_cluster in cluster_pct.index:
            idx = list(cluster_pct.index).index(self.selected_cluster)
            colors[idx] = COBBER_MAROON
        ax.bar([str(int(x)) for x in cluster_pct.index], cluster_pct.values, color=colors)
        ax.set_ylim(0, max(100, cluster_pct.max() + 10))
        ax.set_xlabel("Visit cluster")
        ax.set_ylabel("Seedpod generated (%)")
        ax.set_title("Seedpod generation by visit cluster")
        ax.grid(True, axis="y", alpha=0.25)
        self.seedpod_figure.subplots_adjust(left=0.18, right=0.96, top=0.88, bottom=0.22)
        self.seedpod_canvas.draw()

    def plot_visitor_distribution(self):
        cluster_data = self._cluster_data()
        self.visitor_figure.clear()
        ax = self.visitor_figure.add_subplot(111)
        if cluster_data.empty:
            self.visitor_canvas.draw()
            return
        counts = cluster_data["Visitor_Group"].value_counts().sort_values(ascending=True)
        ax.barh(np.arange(len(counts)), counts.values, color=INFOBLUE)
        ax.set_yticks(np.arange(len(counts)))
        ax.set_yticklabels(counts.index, fontsize=8)
        ax.set_xlabel("Visit count")
        ax.set_title(f"Visitor categories in cluster {self.selected_cluster}")
        ax.grid(True, axis="x", alpha=0.25)
        self.visitor_figure.subplots_adjust(left=0.30, right=0.97, top=0.86, bottom=0.20)
        self.visitor_canvas.draw()

    def cluster_summary_rows(self) -> list[list[str]]:
        data = self.data_tab.dataset
        if data is None or "Visit_Cluster" not in data.columns:
            return []
        rows = []
        for cluster, group in data.groupby("Visit_Cluster"):
            means = group[FEATURES].mean()
            top_visitor = group["Visitor_Group"].value_counts().idxmax()
            seedpod_pct = 100.0 * seedpod_fraction(group[OUTCOME_COLUMN])
            rows.append([
                str(int(cluster)),
                f"{len(group):,}",
                top_visitor,
                level_01(float(means["Contact_Fit"])),
                level_01(float(means["Entry_Depth"])),
                level_01(float(means["Contact_Zone_Closeness"])),
                dwell_level(float(means["Dwell_Time_s"])),
                level_01(float(means["Recurrence_Score"])),
                f"{seedpod_pct:.1f}%",
            ])
        return rows

    def update_comparison_table(self):
        rows = self.cluster_summary_rows()
        headers = ["Cluster", "Visits", "Main visitor", "Contact", "Entry", "Zone", "Dwell", "Recurrence", "Seedpod"]
        self.comparison_table.clear()
        self.comparison_table.setColumnCount(len(headers))
        self.comparison_table.setHorizontalHeaderLabels(headers)
        self.comparison_table.setRowCount(len(rows))
        for r, row_values in enumerate(rows):
            cluster_value = int(row_values[0])
            for c, value in enumerate(row_values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter if c != 2 else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                if self.selected_cluster is not None and cluster_value == int(self.selected_cluster):
                    item.setBackground(QColor(242, 220, 232))
                    if c == 0:
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)
                self.comparison_table.setItem(r, c, item)
        self.comparison_table.resizeColumnsToContents()
        header = self.comparison_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.comparison_table.verticalHeader().setVisible(False)
        self.comparison_table.setMinimumHeight(150)


class CobberOrchidPollinatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.setWindowTitle(f"CobberOrchidPollinator {APP_VERSION}")
        self.setGeometry(35, 35, 1460, 820)
        self.setFont(QFont("Lato"))

        self.tabs = QTabWidget()
        self.data_tab = DataTab(self)
        self.model_tab = ModelTab(self.data_tab, self)
        self.results_tab = ResultsTab(self.data_tab, self.model_tab, self)

        self.tabs.addTab(self.data_tab, "Data")
        self.tabs.addTab(self.model_tab, "Model")
        self.tabs.addTab(self.results_tab, "Results")
        self.setCentralWidget(self.tabs)

        QTimer.singleShot(0, self.data_tab.auto_load_or_generate)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_app_style(app)
    main_window = CobberOrchidPollinatorApp()
    main_window.show()
    sys.exit(app.exec())

# CobberEcoTracker_v8.py
# A PyQt6 exploratory movement-ecology app for Mike's large fish dataset.
#
# Goal:
#   Help students inspect individual fish movement biographies, examine
#   canal-distance transitions, engineer fish-level movement fingerprints,
#   and discover candidate movement types by clustering.
#
# Dependencies:
#   pip install PyQt6 pandas numpy matplotlib scikit-learn openpyxl
#
# Run:
#   python CobberEcoTracker_v8.py
#
# Expected data file:
#   Put LargeFish_MockDataset_BACI_Context.xlsx in the same directory as this script,
#   or use the "Load Large Fish Dataset" button.

from __future__ import annotations

import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QComboBox,
    QGroupBox,
    QTextEdit,
    QTabWidget,
    QTabBar,
    QTableWidget,
    QTableWidgetItem,
    QSplitter,
    QCheckBox,
    QHeaderView,
    QAbstractItemView,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_ROOT = app_root()


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, width: float = 7.2, height: float = 5.0, dpi: int = 100):
        self.figure = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super().__init__(self.figure)


def category_colors(categories: List[str]) -> Dict[str, str]:
    """Assign categories from the book palette in a stable order."""
    palette = [
        "#6C1D45",  # cobbermaroon
        "#3E6990",  # infoblue
        "#756D59",  # miscbrown
        "#A9823A",  # highlight
        "#3F3158",  # questionpurple
        "#184F35",  # projectgreen
        "#3D3D3D",  # charcoal
        "#D3D3D3",  # softgray
    ]
    uniq = list(dict.fromkeys([str(c) for c in categories]))
    return {cat: palette[i % len(palette)] for i, cat in enumerate(uniq)}


class ClusterColorTabBar(QTabBar):
    """Draw cluster-profile tabs with the same colors used in the PCA plot."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cluster_colors: List[QColor] = []
        self.setDrawBase(False)

    def set_cluster_colors(self, colors: List[str]):
        self._cluster_colors = [QColor(color) for color in colors]
        self.update()

    @staticmethod
    def _lighten(color: QColor, amount: float = 0.35) -> QColor:
        """Blend a color with white so inactive tabs remain visibly related."""
        red = round(color.red() + (255 - color.red()) * amount)
        green = round(color.green() + (255 - color.green()) * amount)
        blue = round(color.blue() + (255 - color.blue()) * amount)
        return QColor(red, green, blue)

    @staticmethod
    def _text_color(fill: QColor) -> QColor:
        luminance = 0.2126 * fill.red() + 0.7152 * fill.green() + 0.0722 * fill.blue()
        return QColor("black") if luminance > 155 else QColor("white")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        for index in range(self.count()):
            rect = self.tabRect(index).adjusted(1, 1, -1, -1)
            if index < len(self._cluster_colors):
                base_color = QColor(self._cluster_colors[index])
            else:
                base_color = QColor("#756D59")

            selected = index == self.currentIndex()
            fill_color = base_color if selected else self._lighten(base_color)
            border_color = QColor("#3D3D3D") if selected else QColor("#D3D3D3")

            painter.setBrush(fill_color)
            painter.setPen(QPen(border_color, 2 if selected else 1))
            painter.drawRoundedRect(rect, 3, 3)

            font = self.font()
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(self._text_color(fill_color))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.tabText(index))


def format_float(x, digits=2, missing="—"):
    try:
        if pd.isna(x):
            return missing
        return f"{float(x):.{digits}f}"
    except Exception:
        return missing


# Student-facing names for fish-level movement features. Internal dataframe
# column names remain unchanged, but underscores never appear in the interface.
FEATURE_INFO = {
    "mean_distance_from_canal": {
        "menu": "Mean distance from canal",
        "axis": "Mean distance from canal (m)",
        "table": "Mean distance\nfrom canal (m)",
        "digits": 1,
    },
    "habitat_switches": {
        "menu": "Habitat switches",
        "axis": "Habitat switches",
        "table": "Habitat\nswitches",
        "digits": 0,
    },
    "median_move_rate": {
        "menu": "Typical movement rate (median)",
        "axis": "Typical movement rate (median, m/day)",
        "table": "Typical movement rate\n(median, m/day)",
        "digits": 2,
    },
    "max_move_rate": {
        "menu": "Maximum movement rate",
        "axis": "Maximum movement rate (m/day)",
        "table": "Maximum movement\nrate (m/day)",
        "digits": 2,
    },
    "percent_canal": {
        "menu": "Canal records (%)",
        "axis": "Canal records (%)",
        "table": "Canal\nrecords (%)",
        "digits": 1,
    },
    "max_distance_from_canal": {
        "menu": "Farthest distance from canal",
        "axis": "Farthest distance from canal (m)",
        "table": "Farthest distance\nfrom canal (m)",
        "digits": 1,
    },
    "mean_abs_delta_canal": {
        "menu": "Mean absolute change in canal distance",
        "axis": "Mean absolute change in canal distance (m)",
        "table": "Mean absolute change\nin canal distance (m)",
        "digits": 1,
    },
}

FINGERPRINT_FEATURES = list(FEATURE_INFO.keys())


def feature_menu_label(column: str) -> str:
    return FEATURE_INFO.get(column, {}).get("menu", column.replace("_", " ").title())


def feature_axis_label(column: str) -> str:
    return FEATURE_INFO.get(column, {}).get("axis", feature_menu_label(column))


def feature_table_label(column: str) -> str:
    return FEATURE_INFO.get(column, {}).get("table", feature_menu_label(column))


def format_feature_value(column: str, value, include_unit: bool = True) -> str:
    if pd.isna(value):
        return "—"

    digits = int(FEATURE_INFO.get(column, {}).get("digits", 2))
    if column == "habitat_switches":
        return f"{int(round(float(value)))}"

    text = f"{float(value):,.{digits}f}"
    if not include_unit:
        return text
    if column in {"mean_distance_from_canal", "max_distance_from_canal", "mean_abs_delta_canal"}:
        return f"{text} m"
    if column in {"median_move_rate", "max_move_rate"}:
        return f"{text} m/day"
    if column == "percent_canal":
        return f"{text}%"
    return text


# ---------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------
@dataclass
class FishData:
    raw: pd.DataFrame
    clean: pd.DataFrame
    transitions: pd.DataFrame
    fingerprints: pd.DataFrame
    study_context: pd.DataFrame


class FishDataProcessor:
    REQUIRED_COLUMNS = [
        "Fish",
        "Habitat",
        "DaysBetweenRelocations",
        "DistanceMoved",
        "MovementRate",
        "DistanceFromCanal",
        "BeforeAfter",
        "ControlImpact",
        "StudySite",
    ]

    NUMERIC_COLUMNS = [
        "DaysBetweenRelocations",
        "DistanceMoved",
        "MovementRate",
        "DistanceFromCanal",
        "Depth30",
        "DepthChange30",
        "Photoperiod",
        "MaximumTemp30",
    ]

    @classmethod
    def load_excel(cls, path: Path) -> FishData:
        df = pd.read_excel(path)
        return cls.process(df)

    @classmethod
    def process(cls, df: pd.DataFrame) -> FishData:
        raw = df.copy()

        missing = [c for c in cls.REQUIRED_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        clean = raw.copy()
        clean["_row_order"] = np.arange(len(clean))

        # Normalize key categorical columns.
        clean["Fish"] = clean["Fish"].astype(int)
        clean["Habitat"] = clean["Habitat"].astype(str).str.strip().str.lower()
        clean["BeforeAfter"] = clean["BeforeAfter"].astype(str).str.strip().str.upper()
        clean["ControlImpact"] = clean["ControlImpact"].astype(str).str.strip().str.upper()
        clean["StudySite"] = clean["StudySite"].astype(str).str.strip()

        cls._validate_study_design(clean)

        # Convert numeric-looking fields to numeric. Some cells contain "." or text.
        for col in cls.NUMERIC_COLUMNS:
            if col in clean.columns:
                clean[col] = pd.to_numeric(clean[col], errors="coerce")

        # Friendly aliases for plotting and logic.
        clean["move_rate_m_per_day"] = clean["MovementRate"]
        clean["distance_from_canal"] = clean["DistanceFromCanal"]
        clean["time_between_days"] = clean["DaysBetweenRelocations"]
        clean["distance_moved_m"] = clean["DistanceMoved"]
        clean["relocation_index"] = clean.groupby("Fish").cumcount()

        transitions = cls._build_transitions(clean)
        fingerprints = cls._build_fingerprints(clean, transitions)
        study_context = cls._build_study_context(clean)

        return FishData(
            raw=raw,
            clean=clean,
            transitions=transitions,
            fingerprints=fingerprints,
            study_context=study_context,
        )

    @classmethod
    def _validate_study_design(cls, clean: pd.DataFrame) -> None:
        period_values = set(clean["BeforeAfter"].dropna().astype(str))
        location_values = set(clean["ControlImpact"].dropna().astype(str))

        unexpected_periods = period_values - {"B", "A"}
        unexpected_locations = location_values - {"C", "I"}
        if unexpected_periods:
            raise ValueError(
                "Before/After must contain only B and A. "
                f"Unexpected values: {sorted(unexpected_periods)}"
            )
        if unexpected_locations:
            raise ValueError(
                "Control/Impact must contain only C and I. "
                f"Unexpected values: {sorted(unexpected_locations)}"
            )

        fish_site_counts = clean.groupby("Fish")["StudySite"].nunique(dropna=False)
        inconsistent_fish_sites = fish_site_counts[fish_site_counts != 1].index.tolist()
        if inconsistent_fish_sites:
            raise ValueError(
                "Each fish must remain assigned to one study site. "
                f"Check fish: {inconsistent_fish_sites}"
            )

        fish_status_counts = clean.groupby("Fish")["ControlImpact"].nunique(dropna=False)
        inconsistent_fish_status = fish_status_counts[fish_status_counts != 1].index.tolist()
        if inconsistent_fish_status:
            raise ValueError(
                "Each fish must remain assigned to either control or impact. "
                f"Check fish: {inconsistent_fish_status}"
            )

        site_status_counts = clean.groupby("StudySite")["ControlImpact"].nunique(dropna=False)
        inconsistent_sites = site_status_counts[site_status_counts != 1].index.tolist()
        if inconsistent_sites:
            raise ValueError(
                "Each study site must have one fixed control or impact assignment. "
                f"Check sites: {inconsistent_sites}"
            )

        fish_periods = clean.groupby("Fish")["BeforeAfter"].agg(lambda values: set(values.astype(str)))
        missing_periods = [int(fish) for fish, values in fish_periods.items() if not {"B", "A"}.issubset(values)]
        if missing_periods:
            raise ValueError(
                "Each fish must have both before and after records. "
                f"Check fish: {missing_periods}"
            )

    @classmethod
    def _build_study_context(cls, clean: pd.DataFrame) -> pd.DataFrame:
        records = []
        for fish_id, group in clean.sort_values(["Fish", "_row_order"]).groupby("Fish"):
            records.append({
                "Fish": int(fish_id),
                "study_site": str(group["StudySite"].iloc[0]),
                "study_status": str(group["ControlImpact"].iloc[0]),
                "before_records": int((group["BeforeAfter"] == "B").sum()),
                "after_records": int((group["BeforeAfter"] == "A").sum()),
                "total_records": int(len(group)),
            })
        return pd.DataFrame.from_records(records).sort_values("Fish").reset_index(drop=True)

    @classmethod
    def _build_transitions(cls, clean: pd.DataFrame) -> pd.DataFrame:
        records = []

        for fish_id, group in clean.sort_values(["Fish", "_row_order"]).groupby("Fish"):
            g = group.reset_index(drop=True)
            if len(g) < 2:
                continue

            for i in range(len(g) - 1):
                current = g.iloc[i]
                nxt = g.iloc[i + 1]

                d0 = current.get("distance_from_canal", np.nan)
                d1 = nxt.get("distance_from_canal", np.nan)
                delta_d = d1 - d0 if pd.notna(d0) and pd.notna(d1) else np.nan

                records.append({
                    "Fish": fish_id,
                    "transition_index": i,
                    "from_row": int(current["_row_order"]),
                    "to_row": int(nxt["_row_order"]),
                    "from_habitat": current["Habitat"],
                    "to_habitat": nxt["Habitat"],
                    "habitat_transition": f"{current['Habitat']}→{nxt['Habitat']}",
                    "from_before_after": current["BeforeAfter"],
                    "to_before_after": nxt["BeforeAfter"],
                    "from_control_impact": current["ControlImpact"],
                    "to_control_impact": nxt["ControlImpact"],
                    "time_between_days": nxt.get("time_between_days", np.nan),
                    "distance_moved_m": nxt.get("distance_moved_m", np.nan),
                    "move_rate_m_per_day": nxt.get("move_rate_m_per_day", np.nan),
                    "distance_from_canal_current": d0,
                    "distance_from_canal_next": d1,
                    "delta_distance_from_canal": delta_d,
                    "abs_delta_distance_from_canal": abs(delta_d) if pd.notna(delta_d) else np.nan,
                    "depth_prev30": nxt.get("Depth30", np.nan),
                    "depth_change_prev30": nxt.get("DepthChange30", np.nan),
                    "photoperiod_minutes": nxt.get("Photoperiod", np.nan),
                    "max_temp_prev30": nxt.get("MaximumTemp30", np.nan),
                })

        return pd.DataFrame.from_records(records)

    @classmethod
    def _build_fingerprints(cls, clean: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
        records = []

        for fish_id, group in clean.sort_values(["Fish", "_row_order"]).groupby("Fish"):
            g = group.copy()
            t = transitions[transitions["Fish"] == fish_id].copy() if not transitions.empty else pd.DataFrame()

            n_obs = len(g)
            n_trans = len(t)
            habitats = g["Habitat"].dropna().astype(str)
            majority_habitat = habitats.value_counts().idxmax() if len(habitats) else "unknown"

            percent_canal = 100.0 * (habitats == "canal").mean() if len(habitats) else np.nan
            percent_marsh = 100.0 * (habitats == "marsh").mean() if len(habitats) else np.nan

            habitat_switches = 0
            if len(habitats) >= 2:
                habitat_switches = int((habitats.iloc[1:].values != habitats.iloc[:-1].values).sum())

            ba = g["BeforeAfter"].dropna().astype(str)
            percent_after = 100.0 * (ba == "A").mean() if len(ba) else np.nan

            move_rate = g["move_rate_m_per_day"]
            dist_canal = g["distance_from_canal"]
            dist_moved = g["distance_moved_m"]

            rec = {
                "Fish": fish_id,
                "n_observations": n_obs,
                "n_transitions": n_trans,
                "majority_habitat": majority_habitat,
                "percent_canal": percent_canal,
                "percent_marsh": percent_marsh,
                "habitat_switches": habitat_switches,
                "switch_rate": habitat_switches / max(1, n_trans),
                "percent_after": percent_after,
                "mean_move_rate": move_rate.mean(skipna=True),
                "median_move_rate": move_rate.median(skipna=True),
                "max_move_rate": move_rate.max(skipna=True),
                "mean_distance_moved": dist_moved.mean(skipna=True),
                "total_distance_moved": dist_moved.sum(skipna=True),
                "mean_distance_from_canal": dist_canal.mean(skipna=True),
                "median_distance_from_canal": dist_canal.median(skipna=True),
                "max_distance_from_canal": dist_canal.max(skipna=True),
                "sd_distance_from_canal": dist_canal.std(skipna=True),
                "mean_abs_delta_canal": t["abs_delta_distance_from_canal"].mean(skipna=True) if not t.empty else np.nan,
                "max_abs_delta_canal": t["abs_delta_distance_from_canal"].max(skipna=True) if not t.empty else np.nan,
                "mean_depth_prev30": g["Depth30"].mean(skipna=True) if "Depth30" in g else np.nan,
                "mean_temp_prev30": g["MaximumTemp30"].mean(skipna=True) if "MaximumTemp30" in g else np.nan,
            }
            records.append(rec)

        fp = pd.DataFrame.from_records(records).sort_values("Fish").reset_index(drop=True)
        return fp


# ---------------------------------------------------------------------
# Fish Explorer tab
# ---------------------------------------------------------------------
class FishExplorerTab(QWidget):
    def __init__(self):
        super().__init__()
        self.data: Optional[FishData] = None

        layout = QHBoxLayout(self)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left.setFixedWidth(360)

        selector_box = QGroupBox("Fish Biography")
        form = QFormLayout(selector_box)

        self.fish_combo = QComboBox()
        self.color_combo = QComboBox()
        self.color_combo.addItem("Habitat", "Habitat")
        self.color_combo.addItem("Before/After", "BeforeAfter")
        self.color_combo.addItem("Control/Impact", "ControlImpact")
        self.color_combo.addItem("Study site", "StudySite")

        fish_label = QLabel("Fish")
        fish_label.setStyleSheet("font-weight: bold;")
        color_label = QLabel("Color by")
        color_label.setStyleSheet("font-weight: bold;")

        form.addRow(fish_label, self.fish_combo)
        form.addRow(color_label, self.color_combo)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(500)
        self.summary.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.summary.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.summary.document().setDocumentMargin(8)

        summary_label = QLabel("Movement Summary")
        summary_label.setStyleSheet("color: #6C1D45; font-weight: bold;")

        left_layout.addWidget(selector_box)
        left_layout.addWidget(summary_label)
        left_layout.addWidget(self.summary)
        left_layout.addStretch(1)

        self.canvas = MplCanvas(width=8.2, height=5.8)

        layout.addWidget(left, stretch=0)
        layout.addWidget(self.canvas, stretch=1)

        self.fish_combo.currentIndexChanged.connect(self.update_view)
        self.color_combo.currentIndexChanged.connect(self.update_view)

    def set_data(self, data: FishData):
        self.data = data
        self.fish_combo.blockSignals(True)
        self.fish_combo.clear()

        for fish_id in sorted(data.clean["Fish"].unique()):
            n = len(data.clean[data.clean["Fish"] == fish_id])
            self.fish_combo.addItem(f"Fish {fish_id} ({n} tracking records)", int(fish_id))

        fish_16_index = self.fish_combo.findData(16)
        if fish_16_index >= 0:
            self.fish_combo.setCurrentIndex(fish_16_index)

        self.fish_combo.blockSignals(False)
        self.update_view()

    def select_fish(self, fish_id: int):
        index = self.fish_combo.findData(int(fish_id))
        if index >= 0:
            self.fish_combo.setCurrentIndex(index)

    def update_view(self):
        if self.data is None or self.fish_combo.count() == 0:
            self._draw_placeholder()
            return

        fish_id = self.fish_combo.currentData()
        if fish_id is None:
            return

        df = self.data.clean[self.data.clean["Fish"] == int(fish_id)].sort_values("_row_order").copy()
        fp = self.data.fingerprints[self.data.fingerprints["Fish"] == int(fish_id)].iloc[0]

        self._update_summary(fish_id, df, fp)
        self._draw_fish(fish_id, df)

    def _update_summary(self, fish_id, df, fp):
        study_site = str(df["StudySite"].iloc[0]) if "StudySite" in df.columns else "—"
        study_status_code = str(df["ControlImpact"].iloc[0]) if "ControlImpact" in df.columns else ""
        study_status = {"C": "Control", "I": "Impact"}.get(study_status_code, study_status_code or "—")
        before_records = int((df["BeforeAfter"] == "B").sum())
        after_records = int((df["BeforeAfter"] == "A").sum())

        rows = [
            ("STUDY CONTEXT", None),
            ("Study site", study_site),
            ("Study location", study_status),
            ("Before records", f"{before_records}"),
            ("After records", f"{after_records}"),
            ("TRACKING HISTORY", None),
            ("Tracking records", f"{int(fp['n_observations'])}"),
            ("Habitat switches", f"{int(fp['habitat_switches'])}"),
            ("HABITAT USE", None),
            ("Canal records", f"{format_float(fp['percent_canal'], 1)}%"),
            ("Marsh records", f"{format_float(fp['percent_marsh'], 1)}%"),
            ("MOVEMENT", None),
            ("Typical rate (median)", f"{format_float(fp['median_move_rate'])} m/day"),
            ("Maximum rate", f"{format_float(fp['max_move_rate'])} m/day"),
            ("DISTANCE FROM CANAL", None),
            ("Mean distance", f"{format_float(fp['mean_distance_from_canal'], 1)} m"),
            ("Farthest distance", f"{format_float(fp['max_distance_from_canal'], 1)} m"),
        ]

        html = [
            "<div style='font-family: Lato, Arial, sans-serif; color: #222222;'>",
            f"<div style='font-size: 15pt; font-weight: 700; margin: 0 0 8px 0;'>Fish {fish_id}</div>",
            "<table width='100%' cellspacing='0' cellpadding='3'>",
        ]

        for label, value in rows:
            if value is None:
                html.append(
                    "<tr><td colspan='2' style='padding-top: 9px; padding-bottom: 3px; "
                    "color: #6C1D45; font-weight: 700; font-size: 9.5pt;'>"
                    f"{label}</td></tr>"
                )
            else:
                html.append(
                    "<tr>"
                    f"<td style='font-size: 10pt;'>{label}</td>"
                    f"<td align='right' style='font-size: 10pt; font-weight: 600;'>{value}</td>"
                    "</tr>"
                )

        html.extend(["</table>", "</div>"])
        self.summary.setHtml("".join(html))

    @staticmethod
    def _color_configuration(color_col: str, categories: List[str]):
        present = set(str(cat) for cat in categories)

        if color_col == "Habitat":
            order = ["canal", "marsh"]
            colors = {"canal": "#6C1D45", "marsh": "#3E6990"}
            labels = {"canal": "Canal", "marsh": "Marsh"}
            legend_title = "Habitat"
            # Keep both habitat categories visible so the color key never changes.
            legend_categories = order
        elif color_col == "BeforeAfter":
            order = ["B", "A"]
            colors = {"B": "#3E6990", "A": "#6C1D45"}
            labels = {"B": "Before", "A": "After"}
            legend_title = "Study period"
            legend_categories = [cat for cat in order if cat in present]
        elif color_col == "ControlImpact":
            order = ["C", "I"]
            colors = {"C": "#3E6990", "I": "#6C1D45"}
            labels = {"C": "Control", "I": "Impact"}
            legend_title = "Study location"
            legend_categories = [cat for cat in order if cat in present]
        elif color_col == "StudySite":
            colors = category_colors(categories)
            labels = {cat: str(cat) for cat in colors}
            legend_title = "Study site"
            legend_categories = list(colors)
        else:
            colors = category_colors(categories)
            labels = {cat: str(cat) for cat in colors}
            legend_title = color_col
            legend_categories = list(colors)

        # Protect against unexpected categories without changing known colors.
        fallback_palette = ["#756D59", "#3F3158", "#184F35", "#A9823A"]
        unexpected = [cat for cat in dict.fromkeys(categories) if cat not in colors]
        for i, cat in enumerate(unexpected):
            colors[cat] = fallback_palette[i % len(fallback_palette)]
            labels[cat] = str(cat).title()
            legend_categories.append(cat)

        return colors, labels, legend_categories, legend_title

    @staticmethod
    def _plot_adjacent_valid_segments(ax, x, y):
        """Connect adjacent valid records but never bridge a missing value."""
        for i in range(len(x) - 1):
            if np.isfinite(y[i]) and np.isfinite(y[i + 1]):
                ax.plot(
                    x[i:i + 2],
                    y[i:i + 2],
                    color="#756D59",
                    linewidth=1.25,
                    alpha=0.75,
                    zorder=1,
                )

    def _draw_fish(self, fish_id, df):
        fig = self.canvas.figure
        fig.clear()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

        color_col = self.color_combo.currentData() or self.color_combo.currentText()
        categories = df[color_col].astype(str).tolist()
        colors, labels, legend_categories, legend_title = self._color_configuration(color_col, categories)

        # Display tracking records as 1, 2, 3, ... to match the chapter language.
        x = df["relocation_index"].to_numpy(dtype=int) + 1
        dist = df["distance_from_canal"].to_numpy(dtype=float)
        move = df["move_rate_m_per_day"].to_numpy(dtype=float)

        self._plot_adjacent_valid_segments(ax1, x, dist)
        self._plot_adjacent_valid_segments(ax2, x, move)

        for cat in dict.fromkeys(categories):
            mask = np.array([c == cat for c in categories])
            ax1.scatter(
                x[mask], dist[mask], s=48, color=colors[cat],
                edgecolor="white", linewidth=0.6, alpha=0.92, zorder=2,
            )
            ax2.scatter(
                x[mask], move[mask], s=48, color=colors[cat],
                edgecolor="white", linewidth=0.6, alpha=0.92, zorder=2,
            )

        legend_handles = [
            Line2D(
                [0], [0], marker="o", linestyle="None", markersize=7,
                markerfacecolor=colors[cat], markeredgecolor="white",
                label=labels[cat],
            )
            for cat in legend_categories
        ]

        legend = ax1.legend(
            handles=legend_handles,
            title=legend_title,
            loc="best",
            fontsize=8,
            frameon=True,
            facecolor="white",
            edgecolor="#D3D3D3",
            framealpha=1.0,
        )
        legend.get_title().set_fontweight("bold")

        ax1.set_title(f"Fish {fish_id}: Distance from the Canal", fontweight="bold")
        ax1.set_ylabel("Distance from canal (m)", fontweight="bold")
        ax1.grid(True, alpha=0.25)

        ax2.set_title("Movement Rate Since the Previous Tracking Record", fontweight="bold")
        ax2.set_xlabel("Tracking record number", fontweight="bold")
        ax2.set_ylabel("Movement rate (m/day)", fontweight="bold")
        ax2.grid(True, alpha=0.25)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

        if len(x):
            ax1.set_xlim(0.5, len(x) + 0.5)

        fig.tight_layout()
        self.canvas.draw()

    def _draw_placeholder(self):
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.55, "Fish Explorer", ha="center", va="center", fontsize=16, transform=ax.transAxes)
        ax.text(0.5, 0.42, "Load the large fish dataset to inspect movement biographies.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        self.canvas.draw()


# ---------------------------------------------------------------------
# Transition View tab
# ---------------------------------------------------------------------
class TransitionViewTab(QWidget):
    COLOR_OPTIONS = [
        ("Habitat transition", "habitat_transition"),
        ("Study period", "from_before_after"),
        ("Study location", "from_control_impact"),
    ]

    def __init__(self):
        super().__init__()
        self.data: Optional[FishData] = None

        layout = QHBoxLayout(self)

        left = QWidget()
        left.setFixedWidth(360)
        left_layout = QVBoxLayout(left)

        control_box = QGroupBox("Canal-Distance Transitions")
        form = QFormLayout(control_box)

        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["All fish", "Single fish"])

        self.fish_combo = QComboBox()
        self.fish_combo.setPlaceholderText("Select Single fish first")

        self.color_combo = QComboBox()
        for label, column in self.COLOR_OPTIONS:
            self.color_combo.addItem(label, column)

        self.normalize_checkbox = QCheckBox("Show change in distance per day")
        self.normalize_checkbox.setChecked(False)

        scope_label = QLabel("Scope")
        scope_label.setStyleSheet("font-weight: bold;")
        fish_label = QLabel("Fish")
        fish_label.setStyleSheet("font-weight: bold;")
        color_label = QLabel("Color by")
        color_label.setStyleSheet("font-weight: bold;")

        form.addRow(scope_label, self.scope_combo)
        form.addRow(fish_label, self.fish_combo)
        form.addRow(color_label, self.color_combo)
        form.addRow("", self.normalize_checkbox)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(430)
        self.summary.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.summary.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.summary.document().setDocumentMargin(8)

        summary_label = QLabel("Transition Summary")
        summary_label.setStyleSheet("color: #6C1D45; font-weight: bold;")

        left_layout.addWidget(control_box)
        left_layout.addWidget(summary_label)
        left_layout.addWidget(self.summary)
        left_layout.addStretch(1)

        self.canvas = MplCanvas(width=8.2, height=5.8)

        layout.addWidget(left, stretch=0)
        layout.addWidget(self.canvas, stretch=1)

        self.scope_combo.currentIndexChanged.connect(self._scope_changed)
        self.fish_combo.currentIndexChanged.connect(self.update_view)
        self.color_combo.currentIndexChanged.connect(self.update_view)
        self.normalize_checkbox.stateChanged.connect(self.update_view)
        self._refresh_controls()

    def set_data(self, data: FishData):
        self.data = data
        self.fish_combo.blockSignals(True)
        self.fish_combo.clear()
        for fish_id in sorted(data.clean["Fish"].unique()):
            n = len(data.clean[data.clean["Fish"] == fish_id])
            self.fish_combo.addItem(f"Fish {fish_id} ({n} tracking records)", int(fish_id))
        self.fish_combo.setCurrentIndex(-1)
        self.fish_combo.blockSignals(False)
        self._refresh_controls()
        self.update_view()

    def _scope_changed(self):
        single_fish = self.scope_combo.currentText() == "Single fish"

        self.fish_combo.blockSignals(True)
        if single_fish:
            if self.fish_combo.currentIndex() < 0:
                fish_16_index = self.fish_combo.findData(16)
                self.fish_combo.setCurrentIndex(fish_16_index if fish_16_index >= 0 else 0)
        else:
            self.fish_combo.setCurrentIndex(-1)
        self.fish_combo.blockSignals(False)

        self._refresh_controls()
        self.update_view()

    def _refresh_controls(self):
        single_fish = self.scope_combo.currentText() == "Single fish"
        self.fish_combo.setEnabled(single_fish)
        if not single_fish and self.fish_combo.currentIndex() != -1:
            self.fish_combo.setCurrentIndex(-1)

    def _get_transitions(self):
        if self.data is None:
            return pd.DataFrame()

        t = self.data.transitions.copy()
        if self.scope_combo.currentText() == "Single fish":
            fish_id = self.fish_combo.currentData()
            if fish_id is None:
                return pd.DataFrame()
            t = t[t["Fish"] == int(fish_id)]

        return t.dropna(
            subset=[
                "distance_from_canal_current",
                "distance_from_canal_next",
                "delta_distance_from_canal",
            ]
        )

    @staticmethod
    def _color_configuration(color_col: str, categories: List[str]):
        present = set(str(cat) for cat in categories)

        if color_col == "habitat_transition":
            order = ["canal→canal", "canal→marsh", "marsh→canal", "marsh→marsh"]
            colors = {
                "canal→canal": "#6C1D45",
                "canal→marsh": "#A9823A",
                "marsh→canal": "#756D59",
                "marsh→marsh": "#3E6990",
            }
            labels = {
                "canal→canal": "Canal to canal",
                "canal→marsh": "Canal to marsh",
                "marsh→canal": "Marsh to canal",
                "marsh→marsh": "Marsh to marsh",
            }
            title = "Habitat transition"
        elif color_col == "from_before_after":
            order = ["B", "A"]
            colors = {"B": "#3E6990", "A": "#6C1D45"}
            labels = {"B": "Before", "A": "After"}
            title = "Study period"
        elif color_col == "from_control_impact":
            order = ["C", "I"]
            colors = {"C": "#3E6990", "I": "#6C1D45"}
            labels = {"C": "Control", "I": "Impact"}
            title = "Study location"
        else:
            order = list(dict.fromkeys(categories))
            colors = category_colors(categories)
            labels = {cat: str(cat).replace("_", " ").title() for cat in order}
            title = "Color group"

        legend_categories = [cat for cat in order if cat in present]

        unexpected = [cat for cat in dict.fromkeys(categories) if cat not in colors]
        fallback = ["#6C1D45", "#3E6990", "#756D59", "#A9823A"]
        for i, cat in enumerate(unexpected):
            colors[cat] = fallback[i % len(fallback)]
            labels[cat] = str(cat).replace("_", " ").title()
            legend_categories.append(cat)

        return colors, labels, legend_categories, title

    def _display_delta(self, t: pd.DataFrame) -> Tuple[np.ndarray, str, str]:
        raw = t["delta_distance_from_canal"].to_numpy(dtype=float)
        if not self.normalize_checkbox.isChecked():
            return raw, "Change in distance from canal (m)", "Raw change"

        days = t["time_between_days"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_day = raw / days
        return per_day, "Change in distance from canal per day (m/day)", "Change per day"

    def _update_summary(self, t: pd.DataFrame, delta: np.ndarray, display_mode: str):
        finite = delta[np.isfinite(delta)]
        scope_text = self.scope_combo.currentText()
        if scope_text == "Single fish" and self.fish_combo.currentData() is not None:
            heading = f"Fish {int(self.fish_combo.currentData())}"
            note = "Summary includes the transitions recorded for this fish."
        else:
            heading = "All Fish"
            note = "Fish with more tracking records contribute more transitions to this view."

        if len(finite):
            toward_values = finite[finite < 0]
            away_values = finite[finite > 0]
            toward = int(len(toward_values))
            away = int(len(away_values))
            no_change = int(np.sum(np.isclose(finite, 0.0, atol=1e-12)))
            median = float(np.nanmedian(finite))
            largest_toward = float(np.nanmin(toward_values)) if len(toward_values) else np.nan
            largest_away = float(np.nanmax(away_values)) if len(away_values) else np.nan
        else:
            toward = away = no_change = 0
            median = largest_toward = largest_away = np.nan

        units = "m/day" if self.normalize_checkbox.isChecked() else "m"
        rows = [
            ("TRANSITIONS DISPLAYED", None),
            ("Transitions shown", f"{len(finite)}"),
            ("Toward canal", f"{toward}"),
            ("Away from canal", f"{away}"),
            ("No change", f"{no_change}"),
            ("CHANGE IN CANAL DISTANCE", None),
            ("Median change", f"{format_float(median, 2)} {units}"),
            ("Largest toward", f"{format_float(largest_toward, 2)} {units}"),
            ("Largest away", f"{format_float(largest_away, 2)} {units}"),
            ("DISPLAY MODE", None),
            ("Values shown", display_mode),
        ]

        html = [
            "<div style='font-family: Lato, Arial, sans-serif; color: #222222;'>",
            f"<div style='font-size: 15pt; font-weight: 700; margin: 0 0 8px 0;'>{heading}</div>",
            "<table width='100%' cellspacing='0' cellpadding='3'>",
        ]

        for label, value in rows:
            if value is None:
                html.append(
                    "<tr><td colspan='2' style='padding-top: 9px; padding-bottom: 3px; "
                    "color: #6C1D45; font-weight: 700; font-size: 9.5pt;'>"
                    f"{label}</td></tr>"
                )
            else:
                html.append(
                    "<tr>"
                    f"<td style='font-size: 10pt;'>{label}</td>"
                    f"<td align='right' style='font-size: 10pt; font-weight: 600;'>{value}</td>"
                    "</tr>"
                )

        html.extend([
            "</table>",
            "<div style='margin-top: 12px; color: #756D59; font-size: 9pt;'>",
            note,
            "</div></div>",
        ])
        self.summary.setHtml("".join(html))

    def update_view(self):
        if self.data is None:
            self.summary.clear()
            self._draw_placeholder()
            return

        t = self._get_transitions()
        if t.empty:
            self.summary.setHtml(
                "<div style='font-family: Lato, Arial, sans-serif; color: #756D59;'>"
                "Select <b>Single fish</b> and choose a fish to display its transitions."
                "</div>"
            )
            self._draw_empty()
            return

        fig = self.canvas.figure
        fig.clear()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        color_col = self.color_combo.currentData()
        categories = t[color_col].astype(str).tolist()
        colors, labels, legend_categories, legend_title = self._color_configuration(color_col, categories)

        x = t["distance_from_canal_current"].to_numpy(dtype=float)
        y = t["distance_from_canal_next"].to_numpy(dtype=float)

        for cat in legend_categories:
            mask = np.array([c == cat for c in categories])
            ax1.scatter(
                x[mask],
                y[mask],
                s=34,
                color=colors[cat],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.78,
                label=labels[cat],
            )

        max_val = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
        ax1.plot(
            [0, max_val],
            [0, max_val],
            linestyle="--",
            color="#6C1D45",
            linewidth=1.4,
        )
        ax1.set_xlabel("Current distance from canal (m)", fontweight="bold")
        ax1.set_ylabel("Next distance from canal (m)", fontweight="bold")
        ax1.set_title("Canal-Distance Transition Map", fontweight="bold")
        ax1.grid(True, alpha=0.25)

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor=colors[cat],
                markeredgecolor="white",
                label=labels[cat],
            )
            for cat in legend_categories
        ]
        legend = ax1.legend(
            handles=legend_handles,
            title=legend_title,
            loc="best",
            fontsize=8,
            frameon=True,
            facecolor="white",
            edgecolor="#D3D3D3",
            framealpha=1.0,
        )
        legend.get_title().set_fontweight("bold")

        delta, xlabel, display_mode = self._display_delta(t)
        finite = delta[np.isfinite(delta)]
        ax2.hist(finite, bins=28, color="#3E6990", edgecolor="white", alpha=0.90)
        ax2.axvline(0, color="#6C1D45", linestyle="--", linewidth=1.4)
        ax2.set_xlabel(xlabel, fontweight="bold")
        ax2.set_ylabel("Number of transitions", fontweight="bold")
        ax2.set_title("Toward Canal  ←  0  →  Away from Canal", fontweight="bold")
        ax2.grid(True, axis="y", alpha=0.25)

        self._update_summary(t, delta, display_mode)

        fig.tight_layout()
        self.canvas.draw()

    def _draw_placeholder(self):
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.55, "Transition View", ha="center", va="center", fontsize=16, transform=ax.transAxes)
        ax.text(0.5, 0.42, "Load the large fish dataset to inspect canal-distance transitions.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        self.canvas.draw()

    def _draw_empty(self):
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Select a fish to display its transitions.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        self.canvas.draw()


# ---------------------------------------------------------------------
# Fish Fingerprints tab
# ---------------------------------------------------------------------
class FishFingerprintsTab(QWidget):
    view_fish_history = pyqtSignal(int)
    fish_selected = pyqtSignal(int)

    COLOR_OPTIONS = [
        ("No color grouping", None),
        ("Majority habitat", "majority_habitat"),
    ]

    def __init__(self):
        super().__init__()
        self.data: Optional[FishData] = None
        self.selected_fish_id: Optional[int] = None
        self._current_fp: Optional[pd.DataFrame] = None
        self._current_ax = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        controls.setSpacing(6)

        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.color_combo = QComboBox()
        self.x_combo.setMinimumWidth(245)
        self.y_combo.setMinimumWidth(245)
        self.color_combo.setMinimumWidth(170)

        for label, column in self.COLOR_OPTIONS:
            self.color_combo.addItem(label, column)

        x_label = QLabel("X-axis")
        y_label = QLabel("Y-axis")
        color_label = QLabel("Color by")
        for label in (x_label, y_label, color_label):
            label.setStyleSheet("font-weight: bold;")

        controls.addWidget(x_label)
        controls.addWidget(self.x_combo)
        controls.addSpacing(10)
        controls.addWidget(y_label)
        controls.addWidget(self.y_combo)
        controls.addSpacing(10)
        controls.addWidget(color_label)
        controls.addWidget(self.color_combo)
        controls.addStretch(1)

        content = QHBoxLayout()
        content.setSpacing(12)

        self.canvas = MplCanvas(width=8.8, height=6.0)

        panel = QWidget()
        panel.setFixedWidth(330)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(6)

        summary_label = QLabel("Selected Fish")
        summary_label.setStyleSheet("color: #6C1D45; font-weight: bold; font-size: 12pt;")

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(500)
        self.summary.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.summary.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.summary.document().setDocumentMargin(10)

        self.view_history_button = QPushButton("View Fish History")
        self.view_history_button.setStyleSheet(
            "QPushButton { background-color: #6C1D45; color: white; font-weight: bold; "
            "padding: 7px 12px; border: none; border-radius: 4px; }"
            "QPushButton:hover { background-color: #561536; }"
            "QPushButton:disabled { background-color: #B9A7B0; color: #F5F5F5; }"
        )
        self.view_history_button.setEnabled(False)
        self.view_history_button.clicked.connect(self._request_fish_history)

        panel_layout.addWidget(summary_label)
        panel_layout.addWidget(self.summary, stretch=1)
        panel_layout.addWidget(self.view_history_button)

        content.addWidget(self.canvas, stretch=1)
        content.addWidget(panel, stretch=0)

        layout.addLayout(controls)
        layout.addLayout(content, stretch=1)

        self.x_combo.currentIndexChanged.connect(self.update_view)
        self.y_combo.currentIndexChanged.connect(self.update_view)
        self.color_combo.currentIndexChanged.connect(self.update_view)
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)

        self._show_selection_prompt()

    def set_data(self, data: FishData):
        self.data = data
        self.selected_fish_id = None

        self.x_combo.blockSignals(True)
        self.y_combo.blockSignals(True)
        self.x_combo.clear()
        self.y_combo.clear()

        for column in FINGERPRINT_FEATURES:
            label = feature_menu_label(column)
            self.x_combo.addItem(label, column)
            self.y_combo.addItem(label, column)

        self.x_combo.setCurrentIndex(self.x_combo.findData("mean_distance_from_canal"))
        self.y_combo.setCurrentIndex(self.y_combo.findData("median_move_rate"))
        self.x_combo.blockSignals(False)
        self.y_combo.blockSignals(False)

        self._show_selection_prompt()
        self.update_view()

    def update_view(self):
        if self.data is None:
            self._draw_placeholder()
            return

        fp = self.data.fingerprints.copy()
        xcol = self.x_combo.currentData()
        ycol = self.y_combo.currentData()
        ccol = self.color_combo.currentData()

        if not xcol or not ycol:
            return

        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        self._current_ax = ax
        self._current_fp = fp

        x = fp[xcol].to_numpy(dtype=float)
        y = fp[ycol].to_numpy(dtype=float)

        if ccol == "majority_habitat":
            cats = fp[ccol].astype(str).str.lower().tolist()
            colors = {"canal": "#6C1D45", "marsh": "#3E6990"}
            fallback = category_colors(cats)
            for cat in dict.fromkeys(cats):
                mask = np.array([c == cat for c in cats])
                color = colors.get(cat, fallback.get(cat, "#756D59"))
                ax.scatter(
                    x[mask], y[mask], s=120, color=color,
                    edgecolor="white", linewidth=0.8, alpha=0.92,
                    label=cat.title(), zorder=2,
                )
            legend = ax.legend(title="Majority habitat", loc="best", frameon=True)
            legend.get_title().set_fontweight("bold")
        else:
            ax.scatter(
                x, y, s=120, color="#3E6990",
                edgecolor="white", linewidth=0.8, alpha=0.92, zorder=2,
            )

        for _, row in fp.iterrows():
            if pd.notna(row[xcol]) and pd.notna(row[ycol]):
                fish_id = int(row["Fish"])
                is_selected = fish_id == self.selected_fish_id
                ax.text(
                    row[xcol], row[ycol], str(fish_id),
                    fontsize=8.0 if is_selected else 6.5,
                    fontweight="bold", color="white",
                    ha="center", va="center",
                    zorder=5 if is_selected else 3,
                    clip_on=True,
                )

        if self.selected_fish_id is not None:
            selected = fp[fp["Fish"] == self.selected_fish_id]
            if not selected.empty:
                row = selected.iloc[0]
                if pd.notna(row[xcol]) and pd.notna(row[ycol]):
                    ax.scatter(
                        [row[xcol]], [row[ycol]], s=220,
                        facecolors="none", edgecolors="#A9823A",
                        linewidths=2.6, zorder=4,
                    )

        ax.set_xlabel(feature_axis_label(xcol), fontweight="bold")
        ax.set_ylabel(feature_axis_label(ycol), fontweight="bold")
        ax.set_title("Fish Movement Fingerprints", fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.margins(0.08)
        fig.tight_layout()
        self.canvas.draw()

        if self.selected_fish_id is not None:
            self._update_summary(self.selected_fish_id)

    def _on_plot_click(self, event):
        if (
            self.data is None
            or self._current_fp is None
            or self._current_ax is None
            or event.inaxes != self._current_ax
            or event.x is None
            or event.y is None
        ):
            return

        xcol = self.x_combo.currentData()
        ycol = self.y_combo.currentData()
        valid = self._current_fp.dropna(subset=[xcol, ycol]).copy()
        if valid.empty:
            return

        xy = valid[[xcol, ycol]].to_numpy(dtype=float)
        display_xy = self._current_ax.transData.transform(xy)
        distances = np.sqrt((display_xy[:, 0] - event.x) ** 2 + (display_xy[:, 1] - event.y) ** 2)
        nearest = int(np.argmin(distances))

        if distances[nearest] <= 20:
            self.selected_fish_id = int(valid.iloc[nearest]["Fish"])
            self._update_summary(self.selected_fish_id)
            self.fish_selected.emit(self.selected_fish_id)
            self.update_view()

    def _show_selection_prompt(self):
        self.view_history_button.setEnabled(False)
        self.summary.setHtml(
            "<div style='font-family: Lato, Arial, sans-serif; color: #222222;'>"
            "<p style='font-size: 11pt;'><b>Click a fish on the graph</b> to examine its fingerprint.</p>"
            "<p>The panel will show the selected axis values and several details that help you interpret the point.</p>"
            "<p>Use <b>View Fish History</b> to return from the fingerprint to the fish's sequence of tracking records.</p>"
            "</div>"
        )

    def _update_summary(self, fish_id: int):
        if self.data is None:
            return

        match = self.data.fingerprints[self.data.fingerprints["Fish"] == int(fish_id)]
        if match.empty:
            return

        row = match.iloc[0]
        xcol = self.x_combo.currentData()
        ycol = self.y_combo.currentData()

        majority = str(row["majority_habitat"]).title()
        html = [
            "<div style='font-family: Lato, Arial, sans-serif; color: #222222;'>",
            f"<div style='font-size: 16pt; font-weight: 700; margin-bottom: 10px;'>Fish {fish_id}</div>",
            "<div style='color: #6C1D45; font-weight: 700; margin-top: 5px;'>PLOT VALUES</div>",
            "<table width='100%' cellspacing='0' cellpadding='4'>",
            f"<tr><td>{feature_menu_label(xcol)}</td><td align='right'><b>{format_feature_value(xcol, row[xcol])}</b></td></tr>",
            f"<tr><td>{feature_menu_label(ycol)}</td><td align='right'><b>{format_feature_value(ycol, row[ycol])}</b></td></tr>",
            "</table>",
            "<div style='color: #6C1D45; font-weight: 700; margin-top: 12px;'>FINGERPRINT CONTEXT</div>",
            "<table width='100%' cellspacing='0' cellpadding='4'>",
            f"<tr><td>Tracking records</td><td align='right'><b>{int(row['n_observations'])}</b></td></tr>",
            f"<tr><td>Majority habitat</td><td align='right'><b>{majority}</b></td></tr>",
            f"<tr><td>Canal records</td><td align='right'><b>{format_float(row['percent_canal'], 1)}%</b></td></tr>",
            f"<tr><td>Habitat switches</td><td align='right'><b>{int(row['habitat_switches'])}</b></td></tr>",
            "</table>",
            "<p style='margin-top: 14px;'>A point preserves only the selected features. Return to the fish history before drawing a biological conclusion.</p>",
            "</div>",
        ]
        self.summary.setHtml("".join(html))
        self.view_history_button.setEnabled(True)

    def _request_fish_history(self):
        if self.selected_fish_id is not None:
            self.view_fish_history.emit(int(self.selected_fish_id))

    def _draw_placeholder(self):
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.55, "Fish Fingerprints", ha="center", va="center", fontsize=16, transform=ax.transAxes)
        ax.text(0.5, 0.42, "Load the large fish dataset to compute fish-level movement features.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        self.canvas.draw()


# ---------------------------------------------------------------------
# Fingerprint Data tab
# ---------------------------------------------------------------------
class FingerprintDataTab(QWidget):
    view_fish_history = pyqtSignal(int)

    TABLE_COLUMNS = [
        "Fish",
        "n_observations",
        "majority_habitat",
        "mean_distance_from_canal",
        "habitat_switches",
        "median_move_rate",
        "max_move_rate",
        "percent_canal",
        "max_distance_from_canal",
        "mean_abs_delta_canal",
    ]

    COLUMN_LABELS = {
        "Fish": "Fish",
        "n_observations": "Tracking\nrecords",
        "majority_habitat": "Majority\nhabitat",
    }

    def __init__(self):
        super().__init__()
        self.data: Optional[FishData] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        note = QLabel(
            "This fish-level dataset contains one row per fish. Use it to inspect exact fingerprint values. "
            "Double-click a row to open that fish in the Fish Explorer."
        )
        note.setWordWrap(True)
        note.setStyleSheet("font-size: 10.5pt;")

        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.cellDoubleClicked.connect(self._open_selected_fish)

        header = self.table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setMinimumHeight(66)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        self.table.setStyleSheet(
            """
            QTableWidget {
                gridline-color: #D3D3D3;
                alternate-background-color: #F5F5F5;
                font-size: 11pt;
                selection-background-color: rgba(108, 29, 69, 42);
                selection-color: #111111;
            }
            QTableWidget::item { padding: 4px; }
            QTableWidget::item:hover {
                background-color: transparent;
                color: #111111;
            }
            QTableWidget::item:selected {
                background-color: rgba(108, 29, 69, 42);
                color: #111111;
            }
            QHeaderView::section {
                background-color: #6C1D45;
                color: white;
                font-weight: bold;
                border: 1px solid white;
                padding: 6px;
            }
            """
        )
        self.table.setMouseTracking(False)
        self.table.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, False)

        layout.addWidget(note)
        layout.addWidget(self.table, stretch=1)

    def set_data(self, data: FishData):
        self.data = data
        self._populate_table()

    def _populate_table(self):
        if self.data is None:
            return

        fp = self.data.fingerprints.copy()
        cols = self.TABLE_COLUMNS
        labels = [
            self.COLUMN_LABELS.get(col, feature_table_label(col))
            for col in cols
        ]

        self.table.clear()
        self.table.setRowCount(len(fp))
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(labels)

        for r, (_, row) in enumerate(fp[cols].iterrows()):
            for c, col in enumerate(cols):
                val = row[col]
                if col == "Fish":
                    text = f"Fish {int(val)}"
                elif col == "n_observations":
                    text = str(int(val))
                elif col == "majority_habitat":
                    text = str(val).title()
                else:
                    text = format_feature_value(col, val, include_unit=False)

                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == "Fish":
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.table.setItem(r, c, item)

        widths = [86, 94, 110, 145, 105, 155, 145, 105, 150, 175]
        for index, width in enumerate(widths):
            self.table.setColumnWidth(index, width)
        self.table.resizeRowsToContents()

    def select_fish(self, fish_id: int):
        """Select and center the table row for the requested fish."""
        target = int(fish_id)

        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is None:
                continue

            text = item.text().replace("Fish", "").strip()
            try:
                row_fish_id = int(text)
            except ValueError:
                continue

            if row_fish_id == target:
                self.table.setCurrentCell(row, 0)
                self.table.selectRow(row)
                self.table.scrollToItem(
                    item,
                    QAbstractItemView.ScrollHint.PositionAtCenter,
                )
                return

    def _open_selected_fish(self, row: int, _column: int):
        item = self.table.item(row, 0)
        if item is None:
            return
        text = item.text().replace("Fish", "").strip()
        try:
            self.view_fish_history.emit(int(text))
        except ValueError:
            return


# ---------------------------------------------------------------------
# Movement Types tab
# ---------------------------------------------------------------------
class MovementTypesTab(QWidget):
    clusters_updated = pyqtSignal(object)

    # These five features match the movement fingerprint developed in the chapter.
    CLUSTER_FEATURES = [
        "mean_distance_from_canal",
        "habitat_switches",
        "median_move_rate",
        "max_move_rate",
        "percent_canal",
    ]

    LOG_TRANSFORM_FEATURES = [
        "median_move_rate",
        "max_move_rate",
    ]

    def __init__(self):
        super().__init__()
        self.data: Optional[FishData] = None
        self.clustered: Optional[pd.DataFrame] = None
        self.pca: Optional[PCA] = None
        self.loadings: Optional[pd.DataFrame] = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        left = QWidget()
        left.setFixedWidth(455)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(7)

        control_box = QGroupBox("Candidate Movement Types")
        form = QFormLayout(control_box)

        self.k_combo = QComboBox()
        self.k_combo.addItems(["2", "3", "4", "5", "6", "7", "8"])
        self.k_combo.setCurrentText("4")

        number_label = QLabel("Number of clusters")
        number_label.setStyleSheet("font-weight: bold;")

        self.run_button = QPushButton("Run Clustering")
        self.run_button.setStyleSheet(
            "QPushButton { background-color: #6C1D45; color: white; font-weight: bold; "
            "padding: 7px 12px; border: none; border-radius: 4px; }"
            "QPushButton:hover { background-color: #561536; }"
        )
        self.run_button.clicked.connect(self.run_clustering)

        form.addRow(number_label, self.k_combo)
        form.addRow("", self.run_button)

        cluster_label = QLabel("Cluster Profiles")
        cluster_label.setStyleSheet("color: #6C1D45; font-weight: bold; font-size: 11.5pt;")

        self.cluster_tabs = QTabWidget()
        self.cluster_tab_bar = ClusterColorTabBar()
        self.cluster_tabs.setTabBar(self.cluster_tab_bar)
        self.cluster_tabs.setMinimumHeight(225)

        loadings_label = QLabel("PCA Feature Loadings")
        loadings_label.setStyleSheet("color: #6C1D45; font-weight: bold; font-size: 11.5pt;")

        self.loadings_canvas = MplCanvas(width=4.4, height=2.2)
        self.loadings_canvas.setMinimumHeight(180)

        loadings_note = QLabel(
            "Longer bars show stronger contributions to a PCA axis. The sign shows direction within the plot, "
            "not whether a feature is good or bad. Movement rates are log-transformed, and all five features "
            "are standardized before clustering and PCA."
        )
        loadings_note.setWordWrap(True)
        loadings_note.setStyleSheet("font-size: 9pt; color: #3D3D3D;")

        left_layout.addWidget(control_box)
        left_layout.addWidget(cluster_label)
        left_layout.addWidget(self.cluster_tabs, stretch=1)
        left_layout.addWidget(loadings_label)
        left_layout.addWidget(self.loadings_canvas)
        left_layout.addWidget(loadings_note)

        self.canvas = MplCanvas(width=8.0, height=5.8)

        layout.addWidget(left, stretch=0)
        layout.addWidget(self.canvas, stretch=1)

        self._show_cluster_prompt()
        self._draw_loadings_placeholder()

    def set_data(self, data: FishData):
        self.data = data
        self.run_clustering()

    def run_clustering(self):
        if self.data is None:
            self._draw_placeholder()
            self._draw_loadings_placeholder()
            self._show_cluster_prompt()
            return

        if not SKLEARN_AVAILABLE:
            QMessageBox.warning(
                self,
                "Missing scikit-learn",
                "scikit-learn is required for clustering. Install with: pip install scikit-learn",
            )
            return

        fp = self.data.fingerprints.copy()
        features = [feature for feature in self.CLUSTER_FEATURES if feature in fp.columns]
        if len(features) < 2:
            QMessageBox.warning(self, "Missing features", "Not enough fingerprint features are available for clustering.")
            return

        X = fp[features].copy()

        # Compress extreme movement-rate values before standardizing.
        for column in self.LOG_TRANSFORM_FEATURES:
            if column in X.columns:
                X[column] = np.log1p(X[column])

        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True))

        n_fish = len(fp)
        k = min(int(self.k_combo.currentText()), max(2, n_fish - 1))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = KMeans(n_clusters=k, random_state=17, n_init=20)
        clusters = model.fit_predict(X_scaled)

        pca = PCA(n_components=2, random_state=17)
        coordinates = pca.fit_transform(X_scaled)

        clustered = fp.copy()
        clustered["cluster"] = clusters
        clustered["PC1"] = coordinates[:, 0]
        clustered["PC2"] = coordinates[:, 1]

        loadings = pd.DataFrame(
            pca.components_.T,
            index=features,
            columns=["PC1", "PC2"],
        )

        self.clustered = clustered
        self.pca = pca
        self.loadings = loadings

        self._draw_clusters(clustered, pca)
        self._populate_cluster_tabs(clustered)
        self._draw_loadings(loadings)

        cluster_context = clustered[["Fish", "cluster"]].copy()
        cluster_context["cluster"] = cluster_context["cluster"].astype(int) + 1
        self.clusters_updated.emit(cluster_context)

    @staticmethod
    def _text_color_for_fill(hex_color: str) -> str:
        value = hex_color.lstrip("#")
        red, green, blue = (int(value[i:i + 2], 16) for i in (0, 2, 4))
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        return "black" if luminance > 150 else "white"

    def _draw_clusters(self, clustered: pd.DataFrame, pca: PCA):
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        clusters = sorted(clustered["cluster"].unique())
        display_labels = [str(int(cluster) + 1) for cluster in clusters]
        colors = category_colors(display_labels)

        for cluster in clusters:
            display_number = int(cluster) + 1
            color = colors[str(display_number)]
            subset = clustered[clustered["cluster"] == cluster]
            ax.scatter(
                subset["PC1"], subset["PC2"],
                s=120, alpha=0.92,
                label=f"Cluster {display_number}",
                color=color, edgecolor="white", linewidth=0.8,
                zorder=2,
            )
            text_color = self._text_color_for_fill(color)
            for _, row in subset.iterrows():
                ax.text(
                    row["PC1"], row["PC2"], str(int(row["Fish"])),
                    fontsize=6.5, fontweight="bold", color=text_color,
                    ha="center", va="center", zorder=3,
                )

        ax.axhline(0, color="0.75", linewidth=0.8)
        ax.axvline(0, color="0.75", linewidth=0.8)
        ax.set_xlabel(
            f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var.)",
            fontweight="bold",
        )
        ax.set_ylabel(
            f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var.)",
            fontweight="bold",
        )
        ax.set_title("Candidate Movement Types from Fish Fingerprints", fontweight="bold")
        ax.grid(True, alpha=0.25)
        legend = ax.legend(loc="best", frameon=True)
        for item in legend.get_texts():
            item.set_fontsize(9)
        fig.tight_layout()
        self.canvas.draw()

    def _populate_cluster_tabs(self, clustered: pd.DataFrame):
        self.cluster_tabs.clear()

        clusters = sorted(clustered["cluster"].unique())
        display_labels = [str(int(cluster) + 1) for cluster in clusters]
        colors = category_colors(display_labels)
        tab_colors = []

        for cluster in clusters:
            display_number = int(cluster) + 1
            subset = clustered[clustered["cluster"] == cluster].copy()
            fish_ids = ", ".join(str(int(value)) for value in subset["Fish"].sort_values().tolist())

            habitat_counts = subset["majority_habitat"].astype(str).str.title().value_counts()
            habitat_text = ", ".join(f"{name}: {count}" for name, count in habitat_counts.items())

            rows = [
                ("Mean distance from canal", format_feature_value(
                    "mean_distance_from_canal", subset["mean_distance_from_canal"].median()
                )),
                ("Habitat switches", format_feature_value(
                    "habitat_switches", subset["habitat_switches"].median()
                )),
                ("Typical movement rate", format_feature_value(
                    "median_move_rate", subset["median_move_rate"].median()
                )),
                ("Maximum movement rate", format_feature_value(
                    "max_move_rate", subset["max_move_rate"].median()
                )),
                ("Canal records", format_feature_value(
                    "percent_canal", subset["percent_canal"].median()
                )),
            ]

            html_rows = "".join(
                f"<tr><td style='padding: 3px 8px 3px 0;'>{label}</td>"
                f"<td align='right' style='padding: 3px 0;'><b>{value}</b></td></tr>"
                for label, value in rows
            )

            html = (
                "<div style='font-family: Lato, Arial, sans-serif; color: #222222;'>"
                "<p style='margin-top: 2px;'><b>Provisional label:</b> Not assigned</p>"
                f"<p><b>Fish:</b> {fish_ids}</p>"
                f"<p><b>Number of fish:</b> {len(subset)}</p>"
                f"<p><b>Majority habitat:</b> {habitat_text}</p>"
                "<div style='color: #6C1D45; font-weight: 700; margin-top: 10px;'>"
                "CLUSTER MEDIANS</div>"
                "<table width='100%' cellspacing='0' cellpadding='0'>"
                f"{html_rows}</table>"
                "<p style='margin-top: 10px; font-size: 9.5pt;'>These medians summarize the cluster. "
                "Inspect individual fish before assigning an ecological name.</p>"
                "</div>"
            )

            text = QTextEdit()
            text.setReadOnly(True)
            text.setHtml(html)
            text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            text.document().setDocumentMargin(9)

            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            tab_layout.setContentsMargins(4, 4, 4, 4)
            tab_layout.addWidget(text)

            self.cluster_tabs.addTab(tab, str(display_number))
            tab_colors.append(colors[str(display_number)])

        self.cluster_tab_bar.set_cluster_colors(tab_colors)

    def _draw_loadings(self, loadings: pd.DataFrame):
        fig = self.loadings_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        features = list(loadings.index)
        positions = np.arange(len(features))
        height = 0.34

        ax.barh(
            positions - height / 2,
            loadings["PC1"].to_numpy(),
            height=height,
            color="#6C1D45",
            label="PC1",
        )
        ax.barh(
            positions + height / 2,
            loadings["PC2"].to_numpy(),
            height=height,
            color="#3E6990",
            label="PC2",
        )

        ax.set_yticks(positions)
        ax.set_yticklabels([feature_menu_label(feature) for feature in features], fontsize=7.0)
        ax.invert_yaxis()
        ax.axvline(0, color="#3D3D3D", linewidth=0.8)
        ax.set_xlabel("Loading", fontweight="bold", fontsize=8.0)
        ax.set_title("Features Shaping PC1 and PC2", fontweight="bold", fontsize=9.5)
        ax.tick_params(axis="x", labelsize=7.0)
        ax.grid(True, axis="x", alpha=0.22)
        ax.legend(loc="best", frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        self.loadings_canvas.draw()

    def _show_cluster_prompt(self):
        self.cluster_tabs.clear()
        tab = QWidget()
        layout = QVBoxLayout(tab)
        label = QLabel(
            "Run clustering to create one profile tab for each candidate movement type."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addStretch(1)
        self.cluster_tabs.addTab(tab, "—")
        self.cluster_tab_bar.set_cluster_colors(["#756D59"])

    def _draw_loadings_placeholder(self):
        fig = self.loadings_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(
            0.5, 0.5,
            "Run clustering to view the PCA loadings.",
            ha="center", va="center", fontsize=9,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        self.loadings_canvas.draw()

    def _draw_placeholder(self):
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.55, "Movement Types", ha="center", va="center", fontsize=16, transform=ax.transAxes)
        ax.text(0.5, 0.42, "Load the large fish dataset to cluster movement fingerprints.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        self.canvas.draw()


# ---------------------------------------------------------------------
# BACI Context tab
# ---------------------------------------------------------------------
class BaciContextTab(QWidget):
    view_fish_history = pyqtSignal(int)

    STUDY_GROUPS = [
        ("Before-control", "B", "C", "#3E6990"),
        ("After-control", "A", "C", "#3F3158"),
        ("Before-impact", "B", "I", "#A9823A"),
        ("After-impact", "A", "I", "#6C1D45"),
    ]

    def __init__(self):
        super().__init__()
        self.data: Optional[FishData] = None
        self.cluster_assignments = pd.DataFrame(columns=["Fish", "cluster"])

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        left = QWidget()
        left.setFixedWidth(430)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        controls = QGroupBox("BACI Coverage and Cluster Context")
        form = QFormLayout(controls)
        self.scope_combo = QComboBox()
        self.scope_combo.addItem("All fish", None)
        scope_label = QLabel("Scope")
        scope_label.setStyleSheet("font-weight: bold;")
        form.addRow(scope_label, self.scope_combo)

        coverage_label = QLabel("Coverage Summary")
        coverage_label.setStyleSheet("color: #6C1D45; font-weight: bold; font-size: 11.5pt;")

        self.coverage_table = QTableWidget(4, 4)
        self.coverage_table.setHorizontalHeaderLabels(["Study group", "Records", "Fish", "Sites"])
        self.coverage_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.coverage_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.coverage_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.coverage_table.verticalHeader().setVisible(False)
        self.coverage_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 4):
            self.coverage_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.coverage_table.setMinimumHeight(205)
        self.coverage_table.setMaximumHeight(225)

        caution = QLabel(
            "These summaries show how before, after, control, and impact records are distributed across fish "
            "and clusters. They do not estimate a BACI effect or show that the management change caused a "
            "movement pattern."
        )
        caution.setWordWrap(True)
        caution.setStyleSheet(
            "background-color: #F3EEF1; border: 1px solid #6C1D45; border-radius: 4px; "
            "padding: 9px; color: #3D3D3D; font-size: 9.5pt;"
        )

        guidance = QLabel(
            "Compare record counts with the number of fish and sites. Repeated relocations from one fish are "
            "not additional biological replicates."
        )
        guidance.setWordWrap(True)
        guidance.setStyleSheet("font-size: 9.5pt; color: #3D3D3D;")

        left_layout.addWidget(controls)
        left_layout.addWidget(coverage_label)
        left_layout.addWidget(self.coverage_table)
        left_layout.addWidget(guidance)
        left_layout.addWidget(caution)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(7)

        self.canvas = MplCanvas(width=8.2, height=4.0)
        self.canvas.setMinimumHeight(330)

        fish_label = QLabel("Fish Represented in the Selected Scope")
        fish_label.setStyleSheet("color: #6C1D45; font-weight: bold; font-size: 11.5pt;")

        self.fish_table = QTableWidget()
        self.fish_table.setColumnCount(6)
        self.fish_table.setHorizontalHeaderLabels(
            ["Fish", "Cluster", "Study site", "Location", "Before", "After"]
        )
        self.fish_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.fish_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.fish_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.fish_table.verticalHeader().setVisible(False)
        header = self.fish_table.horizontalHeader()

        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)

        for column in range(2, 6):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        self.fish_table.setMinimumHeight(205)

        self.view_button = QPushButton("View Fish History")
        self.view_button.setStyleSheet(
            "QPushButton { background-color: #6C1D45; color: white; font-weight: bold; "
            "padding: 7px 12px; border: none; border-radius: 4px; }"
            "QPushButton:hover { background-color: #561536; }"
        )

        right_layout.addWidget(self.canvas, stretch=1)
        right_layout.addWidget(fish_label)
        right_layout.addWidget(self.fish_table)
        left_layout.insertWidget(3, self.view_button)

        layout.addWidget(left, stretch=0)
        layout.addWidget(right, stretch=1)

        self.scope_combo.currentIndexChanged.connect(self.update_view)
        self.view_button.clicked.connect(self._open_selected_fish)
        self.fish_table.cellDoubleClicked.connect(self._open_double_clicked_fish)

        self._draw_placeholder()
        self._clear_tables()

    def set_data(self, data: FishData):
        self.data = data
        self.update_view()

    def set_clusters(self, cluster_assignments: pd.DataFrame):
        if cluster_assignments is None or cluster_assignments.empty:
            self.cluster_assignments = pd.DataFrame(columns=["Fish", "cluster"])
        else:
            assignments = cluster_assignments[["Fish", "cluster"]].copy()
            assignments["Fish"] = assignments["Fish"].astype(int)
            assignments["cluster"] = assignments["cluster"].astype(int)
            self.cluster_assignments = assignments.sort_values("Fish").reset_index(drop=True)

        current_cluster = self.scope_combo.currentData()
        self.scope_combo.blockSignals(True)
        self.scope_combo.clear()
        self.scope_combo.addItem("All fish", None)
        for cluster in sorted(self.cluster_assignments["cluster"].unique()):
            self.scope_combo.addItem(f"Cluster {int(cluster)}", int(cluster))
        if current_cluster is not None:
            index = self.scope_combo.findData(int(current_cluster))
            if index >= 0:
                self.scope_combo.setCurrentIndex(index)
        self.scope_combo.blockSignals(False)
        self.update_view()

    def _selected_fish_ids(self) -> Optional[List[int]]:
        cluster = self.scope_combo.currentData()
        if cluster is None or self.cluster_assignments.empty:
            return None
        subset = self.cluster_assignments[self.cluster_assignments["cluster"] == int(cluster)]
        return subset["Fish"].astype(int).tolist()

    def _filtered_records(self) -> pd.DataFrame:
        if self.data is None:
            return pd.DataFrame()
        fish_ids = self._selected_fish_ids()
        if fish_ids is None:
            return self.data.clean.copy()
        return self.data.clean[self.data.clean["Fish"].isin(fish_ids)].copy()

    def update_view(self):
        if self.data is None:
            self._draw_placeholder()
            self._clear_tables()
            return

        records = self._filtered_records()
        coverage = self._coverage_summary(records)
        self._populate_coverage_table(coverage)
        self._draw_coverage(coverage)
        self._populate_fish_table(records)

    def _coverage_summary(self, records: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for label, period, location, color in self.STUDY_GROUPS:
            subset = records[
                (records["BeforeAfter"] == period)
                & (records["ControlImpact"] == location)
            ]
            rows.append({
                "study_group": label,
                "records": int(len(subset)),
                "fish": int(subset["Fish"].nunique()),
                "sites": int(subset["StudySite"].nunique()),
                "color": color,
            })
        return pd.DataFrame(rows)

    def _populate_coverage_table(self, coverage: pd.DataFrame):
        self.coverage_table.setRowCount(len(coverage))
        for row_index, row in coverage.reset_index(drop=True).iterrows():
            values = [row["study_group"], int(row["records"]), int(row["fish"]), int(row["sites"])]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if column == 0:
                    item.setForeground(QColor(row["color"]))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.coverage_table.setItem(row_index, column, item)

    def _draw_coverage(self, coverage: pd.DataFrame):
        fig = self.canvas.figure
        fig.clear()
        ax_records = fig.add_subplot(2, 1, 1)
        ax_fish = fig.add_subplot(2, 1, 2)

        labels = coverage["study_group"].tolist()
        positions = np.arange(len(labels))
        colors = coverage["color"].tolist()

        record_bars = ax_records.bar(positions, coverage["records"], color=colors)
        ax_records.set_title("Relocation Records by BACI Group", fontweight="bold")
        ax_records.set_ylabel("Records", fontweight="bold")
        ax_records.set_xticks(positions)
        ax_records.set_xticklabels([])
        ax_records.grid(True, axis="y", alpha=0.25)
        ax_records.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax_records.bar_label(record_bars, padding=2, fontsize=8)

        fish_bars = ax_fish.bar(positions, coverage["fish"], color=colors)
        ax_fish.set_title("Fish Represented by BACI Group", fontweight="bold")
        ax_fish.set_ylabel("Fish", fontweight="bold")
        ax_fish.set_xticks(positions)
        ax_fish.set_xticklabels(labels, rotation=0, fontsize=8.5)
        ax_fish.grid(True, axis="y", alpha=0.25)
        ax_fish.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax_fish.bar_label(fish_bars, padding=2, fontsize=8)

        scope = self.scope_combo.currentText() if self.scope_combo.count() else "All fish"
        fig.suptitle(f"BACI Coverage: {scope}", fontweight="bold", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        self.canvas.draw()

    def _populate_fish_table(self, records: pd.DataFrame):
        if self.data is None:
            self.fish_table.setRowCount(0)
            return

        fish_ids = sorted(records["Fish"].astype(int).unique().tolist())
        context = self.data.study_context[self.data.study_context["Fish"].isin(fish_ids)].copy()
        if not self.cluster_assignments.empty:
            context = context.merge(self.cluster_assignments, on="Fish", how="left")
        else:
            context["cluster"] = np.nan
        context = context.sort_values("Fish").reset_index(drop=True)

        self.fish_table.setRowCount(len(context))
        for row_index, row in context.iterrows():
            cluster_text = "—" if pd.isna(row["cluster"]) else str(int(row["cluster"]))
            location_text = {"C": "Control", "I": "Impact"}.get(str(row["study_status"]), str(row["study_status"]))
            values = [
                f"Fish {int(row['Fish'])}",
                cluster_text,
                str(row["study_site"]),
                location_text,
                int(row["before_records"]),
                int(row["after_records"]),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column in {0, 1, 3, 4, 5}:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.fish_table.setItem(row_index, column, item)

        if self.fish_table.rowCount() > 0:
            self.fish_table.selectRow(0)

    def _open_selected_fish(self):
        row = self.fish_table.currentRow()
        if row < 0:
            return
        item = self.fish_table.item(row, 0)
        if item is None:
            return
        try:
            fish_id = int(item.text().replace("Fish", "").strip())
        except ValueError:
            return
        self.view_fish_history.emit(fish_id)

    def _open_double_clicked_fish(self, row: int, _column: int):
        self.fish_table.selectRow(row)
        self._open_selected_fish()

    def _clear_tables(self):
        self.coverage_table.clearContents()
        self.fish_table.setRowCount(0)

    def _draw_placeholder(self):
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.57, "BACI Context", ha="center", va="center", fontsize=16, transform=ax.transAxes)
        ax.text(
            0.5, 0.43,
            "Load the revised dataset to compare before, after, control, and impact coverage.",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        self.canvas.draw()


# ---------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------
class CobberEcoTrackerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CobberEcoTracker v8")
        self.setGeometry(100, 100, 1380, 790)
        self.setFont(QFont("Lato"))

        self.data: Optional[FishData] = None
        self.current_file: Optional[Path] = None

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        header = QHBoxLayout()
        self.load_button = QPushButton("Load Large Fish Dataset")
        self.load_button.clicked.connect(self.load_file_dialog)

        self.file_label = QLabel("No dataset loaded.")
        self.file_label.setWordWrap(True)

        header.addWidget(self.load_button)
        header.addWidget(self.file_label, stretch=1)

        self.tabs = QTabWidget()
        self.fish_explorer = FishExplorerTab()
        self.transition_view = TransitionViewTab()
        self.fingerprints = FishFingerprintsTab()
        self.fingerprint_data = FingerprintDataTab()
        self.movement_types = MovementTypesTab()
        self.baci_context = BaciContextTab()

        self.tabs.addTab(self.fish_explorer, "Fish Explorer")
        self.tabs.addTab(self.transition_view, "Transition View")
        self.tabs.addTab(self.fingerprints, "Fish Fingerprints")
        self.tabs.addTab(self.fingerprint_data, "Fingerprint Data")
        self.tabs.addTab(self.movement_types, "Movement Types")
        self.tabs.addTab(self.baci_context, "BACI Context")

        self.fingerprints.view_fish_history.connect(self._show_fish_history)
        self.fingerprints.fish_selected.connect(self.fingerprint_data.select_fish)
        self.fingerprint_data.view_fish_history.connect(self._show_fish_history)
        self.baci_context.view_fish_history.connect(self._show_fish_history)
        self.movement_types.clusters_updated.connect(self.baci_context.set_clusters)

        root_layout.addLayout(header)
        root_layout.addWidget(self.tabs)

        self._try_auto_load()

    def _show_fish_history(self, fish_id: int):
        self.fish_explorer.select_fish(int(fish_id))
        self.tabs.setCurrentWidget(self.fish_explorer)

    def _try_auto_load(self):
        candidates = [
            APP_ROOT / "LargeFish_MockDataset_BACI_Context.xlsx",
        ]

        for path in candidates:
            if path.exists():
                try:
                    self.load_dataset(path)
                    return
                except Exception as exc:
                    friendly_name = f"{path.stem.replace('_', ' ')}{path.suffix}"
                    self.file_label.setText(f"Found {friendly_name}, but could not load it: {exc}")
                    return

        self.file_label.setText(
            "No dataset loaded. Put the Large Fish Dataset workbook beside the program, "
            "or click 'Load Large Fish Dataset'."
        )

    def load_file_dialog(self):
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select Large Fish Dataset",
            str(APP_ROOT),
            "Excel files (*.xlsx *.xls);;All files (*)",
        )
        if path_str:
            self.load_dataset(Path(path_str))

    def load_dataset(self, path: Path):
        try:
            data = FishDataProcessor.load_excel(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", f"Could not load dataset:\n{exc}")
            return

        self.data = data
        self.current_file = path

        n_rows = len(data.clean)
        n_fish = data.clean["Fish"].nunique()
        n_sites = data.clean["StudySite"].nunique()
        n_trans = len(data.transitions)

        friendly_name = f"{path.stem.replace('_', ' ')}{path.suffix}"
        self.file_label.setText(
            f"Loaded: {friendly_name}  |  rows={n_rows}, fish={n_fish}, sites={n_sites}, transitions={n_trans}"
        )

        self.fish_explorer.set_data(data)
        self.transition_view.set_data(data)
        self.fingerprints.set_data(data)
        self.fingerprint_data.set_data(data)
        self.baci_context.set_data(data)
        self.movement_types.set_data(data)


def apply_app_stylesheet(app: QApplication) -> None:
    app.setStyleSheet(
        """
        QWidget { color: #222222; background-color: #ffffff; }
        QMainWindow, QDialog { background-color: #ffffff; }
        QTabWidget::pane { border: 1px solid #cccccc; }
        QTabBar::tab {
            background: #5F5F5F;
            color: #ffffff;
            padding: 6px 12px;
            border-right: 1px solid #ffffff;
        }
        QTabBar::tab:selected {
            background: #6C1D45;
            color: #ffffff;
            font-weight: bold;
        }
        QTabBar::tab:hover:!selected { background: #3D3D3D; }
        QLabel { color: #222222; background-color: transparent; }
        QGroupBox {
            color: #222222;
            font-weight: bold;
            border: 1px solid #d6d6d6;
            border-radius: 5px;
            margin-top: 8px;
            padding-top: 10px;
            background-color: #fafafa;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px 0 4px;
            color: #6c1d45;
            background-color: #fafafa;
        }
        QComboBox, QTextEdit, QTableWidget {
            background-color: #ffffff;
            color: #111111;
            border: 1px solid #a0a0a0;
            border-radius: 3px;
            padding: 3px 6px;
            selection-background-color: #6c1d45;
            selection-color: #ffffff;
        }
        QPushButton {
            background-color: #f7f7f7;
            color: #111111;
            border: 1px solid #9a9a9a;
            border-radius: 4px;
            padding: 6px 10px;
        }
        QPushButton:hover { background-color: #eeeeee; }
        QPushButton:pressed { background-color: #dddddd; }
        """
    )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_app_stylesheet(app)
    window = CobberEcoTrackerApp()
    window.show()
    sys.exit(app.exec())
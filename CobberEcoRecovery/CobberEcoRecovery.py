# CobberEcoRecovery_v10.py
# A PyQt6 application for systematic regression model selection on Megan Hartley's
# fictional post-fire vegetation recovery dataset.
#
# Teaching spine:
#   1. Train each model with several recovery-clue sets.
#   2. Choose one clue set for each model.
#   3. Test those selected setups on hidden survey plots.
#   4. Choose one strongest setup.
#   5. Use only that setup for the unsurveyed-site restoration plan.
#
# Expected CSV files in the same folder:
#   recovery_surveyed_sites.csv
#   recovery_unsurveyed_sites.csv
#
# Dependencies:
#   pip install PyQt6 pandas numpy matplotlib scikit-learn
#
# Run:
#   python CobberEcoRecovery_v10.py

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ----------------------------------------------------------------------
# Files and constants
# ----------------------------------------------------------------------

def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_ROOT = app_root()
SURVEYED_FILE = "recovery_surveyed_sites.csv"
UNSURVEYED_FILE = "recovery_unsurveyed_sites.csv"
TARGET_COLUMN = "vegetation_recovery_index"
ROLE_COLUMN = "data_role"
TRAINING_ROLE = "training"
HIDDEN_ROLE = "hidden_test"

COBBER_MAROON = "#6c1d45"
DARK_GRAY = "#555555"
LIGHT_GRAY = "#f5f5f5"

FEATURE_SETS: dict[str, list[str]] = {
    "Burn severity only": ["burn_severity"],
    "Burn severity + greenness": ["burn_severity", "greenness_index"],
    "Quick field survey": [
        "burn_severity",
        "greenness_index",
        "soil_moisture",
        "canopy_cover_pct",
    ],
    "Full recovery survey": [
        "burn_severity",
        "greenness_index",
        "soil_moisture",
        "canopy_cover_pct",
        "slope_degrees",
        "regrowth_height_cm",
    ],
}

MODEL_ORDER = [
    "Linear Regression",
    "Decision Tree",
    "Random Forest",
    "k-Nearest Neighbors",
]

FRIENDLY_NAMES = {
    "site_id": "Site",
    "data_role": "Data role",
    "training": "Training site",
    "hidden_test": "Hidden survey plot",
    "landscape_position": "Landscape position",
    "ridge": "Ridge",
    "open_slope": "Open slope",
    "drainage": "Drainage",
    "burn_severity": "Burn severity",
    "greenness_index": "Greenness index",
    "soil_moisture": "Soil moisture",
    "canopy_cover_pct": "Canopy cover (%)",
    "slope_degrees": "Slope (degrees)",
    "regrowth_height_cm": "Regrowth height (cm)",
    "vegetation_recovery_index": "Vegetation recovery index",
    "planning_context": "Field note",
}

MODEL_NOTES = {
    "Linear Regression": (
        "Linear Regression fits one broad trend."
    ),
    "Decision Tree": (
        "A Decision Tree splits the training sites into regions."
    ),
    "Random Forest": (
        "A Random Forest averages many tree-based predictions."
    ),
    "k-Nearest Neighbors": (
        "k-Nearest Neighbors predicts from similar training sites."
    ),
}

COMMON_PLOT_OBSERVATIONS = [
    "Choose an observation...",
    "Often overpredicts low-recovery sites",
    "Often underpredicts high-recovery sites",
    "Has its largest errors in the moderate range",
    "Errors are fairly even across the recovery range",
]

MODEL_SPECIFIC_OBSERVATIONS = {
    "Linear Regression": "Errors bend in a pattern instead of scattering randomly",
    "Decision Tree": "Predictions fall into several horizontal bands",
    "Random Forest": "Predictions are pulled toward the middle",
    "k-Nearest Neighbors": "Predictions form small local groups rather than one smooth pattern",
}


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------

@dataclass
class Metrics:
    mae: float
    rmse: float
    r2: float


@dataclass
class SetupResult:
    model_name: str
    feature_set_name: str
    train_metrics: Metrics | None = None
    hidden_metrics: Metrics | None = None
    plot_observation: str | None = None


# ----------------------------------------------------------------------
# Styling and helpers
# ----------------------------------------------------------------------

ACTIVE_BUTTON_STYLE = f"""
QPushButton {{
    background-color: {COBBER_MAROON};
    color: white;
    font-weight: bold;
    border: 1px solid {COBBER_MAROON};
    border-radius: 4px;
    padding: 7px 10px;
}}
QPushButton:hover {{ background-color: #7d2652; }}
"""

INACTIVE_BUTTON_STYLE = f"""
QPushButton {{
    background-color: {DARK_GRAY};
    color: white;
    font-weight: bold;
    border: 1px solid {DARK_GRAY};
    border-radius: 4px;
    padding: 7px 10px;
}}
QPushButton:hover {{ background-color: #666666; }}
"""

DISABLED_BUTTON_STYLE = """
QPushButton {
    background-color: #c9c9c9;
    color: #555555;
    font-weight: bold;
    border: 1px solid #b0b0b0;
    border-radius: 4px;
    padding: 7px 10px;
}
"""

CARD_STYLE = """
QFrame#featureCard {
    border: 1px solid #c7c7c7;
    border-radius: 7px;
    background-color: #ffffff;
}
"""

SELECTED_CARD_STYLE = f"""
QFrame#featureCard {{
    border: 2px solid {COBBER_MAROON};
    border-radius: 7px;
    background-color: #fbf7fa;
}}
"""

INFO_BOX_STYLE = f"background-color: {LIGHT_GRAY}; border: 1px solid #cccccc; padding: 8px;"


def set_button_enabled(button: QPushButton, enabled: bool) -> None:
    button.setEnabled(enabled)
    button.setStyleSheet(INACTIVE_BUTTON_STYLE if enabled else DISABLED_BUTTON_STYLE)


def set_button_active(button: QPushButton) -> None:
    button.setEnabled(True)
    button.setStyleSheet(ACTIVE_BUTTON_STYLE)


def format_metric(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.3f}"


def friendly(value: str) -> str:
    return FRIENDLY_NAMES.get(value, value.replace("_", " ").title())


def feature_list_text(feature_set_name: str) -> str:
    return ", ".join(friendly(c) for c in FEATURE_SETS[feature_set_name])


def metrics_from(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> Metrics:
    return Metrics(
        mae=float(mean_absolute_error(actual, predicted)),
        rmse=float(np.sqrt(mean_squared_error(actual, predicted))),
        r2=float(r2_score(actual, predicted)),
    )


# ----------------------------------------------------------------------
# Plot helper
# ----------------------------------------------------------------------

class PlotCanvas(FigureCanvas):
    def __init__(self, width: float = 5.0, height: float = 2.65, dpi: int = 100):
        self.figure = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)

    def clear_message(self, title: str, message: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_axis_off()
        self.draw()


def draw_prediction_plot(canvas: PlotCanvas, actual, predicted, title: str) -> None:
    canvas.figure.clear()
    ax = canvas.figure.add_subplot(111)
    ax.scatter(actual, predicted, alpha=0.78, s=24)
    max_val = max(
        float(np.max(actual)),
        float(np.max(predicted)),
    )

    pad = max(0.25, max_val * 0.07)
    lims = [0, max_val + pad]
    ax.plot(lims, lims, linestyle="--", color=COBBER_MAROON, linewidth=2.0, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Measured recovery index")
    ax.set_ylabel("Predicted recovery index")
    ax.set_title(title)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best")
    canvas.draw()


def draw_residual_plot(canvas: PlotCanvas, actual, predicted, title: str) -> None:
    residuals = np.asarray(predicted) - np.asarray(actual)
    canvas.figure.clear()
    ax = canvas.figure.add_subplot(111)
    ax.scatter(actual, residuals, alpha=0.78, s=24)
    ax.axhline(0, linestyle="--", color=COBBER_MAROON, linewidth=2.0, label="Zero error")
    ax.set_xlabel("Measured recovery index")
    ax.set_ylabel("Residual (predicted - measured)")
    ax.set_title(title)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best")
    canvas.draw()


def draw_restoration_plan_plot(canvas: PlotCanvas, df: pd.DataFrame) -> None:
    """Show unsurveyed sites sorted by predicted recovery."""
    plot_df = df.sort_values(
        "predicted_recovery_index",
        ascending=True,
    ).reset_index(drop=True)

    category_colors = {
        "Monitor recovery": "#8fbc8f",
        "Possible restoration help": "#e6b84a",
        "Field check before action": "#d97979",
        "Compare with field notes": "#7fa6c9",
    }

    canvas.figure.clear()
    ax = canvas.figure.add_subplot(111)

    y_positions = np.arange(len(plot_df))
    predictions = plot_df["predicted_recovery_index"].to_numpy()

    # Light guide lines make it easier to connect each site label to its point.
    ax.hlines(
        y=y_positions,
        xmin=0,
        xmax=predictions,
        color="#d9d9d9",
        linewidth=1.0,
        zorder=1,
    )

    for category, color in category_colors.items():
        mask = plot_df["planning_note"] == category
        if mask.any():
            ax.scatter(
                predictions[mask],
                y_positions[mask],
                s=48,
                color=color,
                edgecolor="#555555",
                linewidth=0.5,
                label=category,
                zorder=2,
            )

    site_labels = []
    for raw_site in plot_df["site_id"].astype(str):
        site_labels.append(
            f"Site {raw_site}" if raw_site.startswith("U") else raw_site
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(site_labels, fontsize=8.5)
    ax.set_xlim(0, 10)
    ax.set_xlabel("Predicted recovery index")
    ax.set_title("Predicted recovery for unsurveyed sites")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(
        loc="lower right",
        fontsize=8,
        frameon=True,
    )

    canvas.figure.tight_layout()
    canvas.draw()


# ----------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------

class CobberEcoRecoveryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CobberEcoRecovery v10")
        self.setGeometry(60, 60, 1460, 850)
        self.setFont(QFont("Lato"))

        self.surveyed_df: pd.DataFrame | None = None
        self.unsurveyed_df: pd.DataFrame | None = None
        self.training_df: pd.DataFrame | None = None
        self.hidden_df: pd.DataFrame | None = None

        self.training_results: dict[str, dict[str, Metrics]] = {m: {} for m in MODEL_ORDER}
        self.plot_observations: dict[str, dict[str, str]] = {m: {} for m in MODEL_ORDER}
        self.selected_feature_by_model: dict[str, str] = {}
        self.hidden_results: dict[str, SetupResult] = {}
        self.hidden_candidate_model: str | None = None
        self.final_setup: SetupResult | None = None

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.load_data()
        if self.surveyed_df is None or self.unsurveyed_df is None:
            self.build_error_tab()
            return

        self.model_tabs: dict[str, ModelTrainingTab] = {}
        for model_name in MODEL_ORDER:
            tab = ModelTrainingTab(self, model_name)
            self.model_tabs[model_name] = tab
            self.tabs.addTab(tab, model_name)

        self.hidden_tab = HiddenSurveyTestTab(self)
        self.tabs.addTab(self.hidden_tab, "Hidden Test")

        self.plan_tab = RestorationPlanTab(self)
        self.tabs.addTab(self.plan_tab, "Restoration Plan")

        self.tabs.currentChanged.connect(self.handle_tab_changed)

    def handle_tab_changed(self, index: int) -> None:
        widget = self.tabs.widget(index)
        if hasattr(widget, "refresh"):
            widget.refresh()

    def load_data(self) -> None:
        try:
            surveyed_path = APP_ROOT / SURVEYED_FILE
            unsurveyed_path = APP_ROOT / UNSURVEYED_FILE
            if not surveyed_path.exists():
                raise FileNotFoundError(f"Missing {SURVEYED_FILE}")
            if not unsurveyed_path.exists():
                raise FileNotFoundError(f"Missing {UNSURVEYED_FILE}")

            surveyed = pd.read_csv(surveyed_path)
            unsurveyed = pd.read_csv(unsurveyed_path)
            self.validate_surveyed(surveyed)
            self.validate_unsurveyed(unsurveyed)

            self.surveyed_df = surveyed.copy()
            self.unsurveyed_df = unsurveyed.copy()
            self.training_df = surveyed[surveyed[ROLE_COLUMN] == TRAINING_ROLE].copy()
            self.hidden_df = surveyed[surveyed[ROLE_COLUMN] == HIDDEN_ROLE].copy()
        except Exception as exc:
            self.surveyed_df = None
            self.unsurveyed_df = None
            self.training_df = None
            self.hidden_df = None
            QMessageBox.critical(self, "Data Error", str(exc))

    def validate_surveyed(self, df: pd.DataFrame) -> None:
        required = set(["site_id", ROLE_COLUMN, "landscape_position", TARGET_COLUMN] + FEATURE_SETS["Full recovery survey"])
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{SURVEYED_FILE} is missing columns: {', '.join(sorted(missing))}")
        roles = set(df[ROLE_COLUMN].dropna().unique())
        if TRAINING_ROLE not in roles or HIDDEN_ROLE not in roles:
            raise ValueError(f"{SURVEYED_FILE} must include training and hidden_test data roles.")

    def validate_unsurveyed(self, df: pd.DataFrame) -> None:
        required = set(["site_id", "landscape_position"] + FEATURE_SETS["Full recovery survey"])
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{UNSURVEYED_FILE} is missing columns: {', '.join(sorted(missing))}")

    def counts_text(self) -> str:
        train_n = len(self.training_df) if self.training_df is not None else 0
        hidden_n = len(self.hidden_df) if self.hidden_df is not None else 0
        unsurveyed_n = len(self.unsurveyed_df) if self.unsurveyed_df is not None else 0
        return (
            f"<b>Surveyed sites loaded:</b> &nbsp; "
            f"Training: <b>{train_n}</b> &nbsp;&nbsp; "
            f"Hidden survey plots: <b>{hidden_n}</b> &nbsp;&nbsp; "
            f"Unsurveyed: <b>{unsurveyed_n}</b>"
        )


    def make_model(self, model_name: str):
        if model_name == "Linear Regression":
            return LinearRegression()
        if model_name == "Decision Tree":
            return DecisionTreeRegressor(random_state=42, max_depth=3, min_samples_leaf=20)
        if model_name == "Random Forest":
            return RandomForestRegressor(
                random_state=42,
                n_estimators=100,
                min_samples_leaf=12,
                max_features=0.65,
            )
        if model_name == "k-Nearest Neighbors":
            return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=17, weights="uniform"))
        raise ValueError(f"Unknown model: {model_name}")

    def fit_model(self, model_name: str, feature_set_name: str, use_all_surveyed: bool = False):
        df = self.surveyed_df if use_all_surveyed else self.training_df
        if df is None:
            raise ValueError("Survey data are not loaded.")
        cols = FEATURE_SETS[feature_set_name]
        model = self.make_model(model_name)
        model.fit(df[cols], df[TARGET_COLUMN])
        return model

    def training_predictions(self, model_name: str, feature_set_name: str) -> tuple[np.ndarray, np.ndarray, Metrics]:
        model = self.fit_model(model_name, feature_set_name, use_all_surveyed=False)
        cols = FEATURE_SETS[feature_set_name]
        actual = self.training_df[TARGET_COLUMN]
        pred = model.predict(self.training_df[cols])
        return actual.to_numpy(), pred, metrics_from(actual, pred)

    def hidden_predictions(self, model_name: str, feature_set_name: str) -> tuple[np.ndarray, np.ndarray, Metrics]:
        model = self.fit_model(model_name, feature_set_name, use_all_surveyed=False)
        cols = FEATURE_SETS[feature_set_name]
        actual = self.hidden_df[TARGET_COLUMN]
        pred = model.predict(self.hidden_df[cols])
        return actual.to_numpy(), pred, metrics_from(actual, pred)

    def select_feature_for_model(self, model_name: str, feature_set_name: str) -> None:
        self.selected_feature_by_model[model_name] = feature_set_name
        self.hidden_results.pop(model_name, None)
        self.hidden_candidate_model = None
        self.final_setup = None
        if hasattr(self, "hidden_tab"):
            self.hidden_tab.refresh()
        if hasattr(self, "plan_tab"):
            self.plan_tab.refresh()

    def build_error_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        msg = QLabel(
            "Could not load the recovery CSV files. Put recovery_surveyed_sites.csv "
            "and recovery_unsurveyed_sites.csv in the same folder as this script."
        )
        msg.setWordWrap(True)
        layout.addWidget(msg)
        self.tabs.addTab(tab, "Data Error")


# ----------------------------------------------------------------------
# Model training tabs
# ----------------------------------------------------------------------

class FeatureSummaryCard(QWidget):
    def __init__(
        self,
        app: CobberEcoRecoveryApp,
        model_name: str,
        feature_set_name: str,
        radio_group: QButtonGroup,
    ):
        super().__init__()

        self.app = app
        self.model_name = model_name
        self.feature_set_name = feature_set_name

        self.setMinimumHeight(205)
        self.setMaximumHeight(225)
        # The outer widget allows the title to overlap the QFrame border.
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 7, 0, 0)
        outer_layout.setSpacing(0)

        self.card_frame = QFrame()
        self.card_frame.setObjectName("featureCard")
        self.card_frame.setStyleSheet(CARD_STYLE)
        outer_layout.addWidget(self.card_frame)

        # This title sits over the frame border rather than inside the card.
        self.title_label = QLabel(f"<b>{feature_set_name}</b>", self)
        self.title_label.setStyleSheet(
            """
            QLabel {
                background-color: #ffffff;
                border: none;
                padding: 0 5px;
            }
            """
        )
        self.title_label.adjustSize()
        self.title_label.move(9, 0)
        self.title_label.raise_()

        frame_layout = QVBoxLayout(self.card_frame)
        frame_layout.setContentsMargins(11, 12, 11, 8)
        frame_layout.setSpacing(2)
        frame_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Keep the radio button in the upper-right corner.
        radio_row = QHBoxLayout()
        radio_row.setContentsMargins(0, 0, 0, 0)
        radio_row.setSpacing(0)
        radio_row.addStretch(1)

        self.radio = QRadioButton("")
        self.radio.setToolTip(
            "Select this trained clue set to carry forward."
        )
        self.radio.setEnabled(False)
        self.radio.clicked.connect(self.handle_selected)
        radio_group.addButton(self.radio)

        radio_row.addWidget(
            self.radio,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
        )
        frame_layout.addLayout(radio_row)

        self.features_label = QLabel(
            f"<b>Features:</b><br>{feature_list_text(feature_set_name)}"
        )
        self.features_label.setWordWrap(True)
        self.features_label.setMinimumHeight(30)
        frame_layout.addWidget(
            self.features_label,
            0,
            Qt.AlignmentFlag.AlignTop,
        )

        frame_layout.addSpacing(5)

        self.metrics_label = QLabel(
            "<b>Training metrics</b><br>"
            "<b>MAE:</b> --<br>"
            "<b>RMSE:</b> --<br>"
            "<b>R²:</b> --"
        )
        self.metrics_label.setWordWrap(True)

        frame_layout.addWidget(
            self.metrics_label,
            0,
            Qt.AlignmentFlag.AlignTop,
        )

        frame_layout.addSpacing(5)

        self.observation_label = QLabel(
            "<b>Observation:</b> Not recorded"
        )
        self.observation_label.setWordWrap(True)
        self.observation_label.setStyleSheet(
            "color: #444444; font-size: 9.5pt;"
        )
        frame_layout.addWidget(
            self.observation_label,
            0,
            Qt.AlignmentFlag.AlignTop,
        )
        frame_layout.addStretch(1)

    def resizeEvent(self, event) -> None:
        """Keep the overlapping title correctly positioned."""
        super().resizeEvent(event)
        self.title_label.adjustSize()
        self.title_label.move(9, 0)
        self.title_label.raise_()

    def update_metrics(self, metrics: Metrics) -> None:
        self.metrics_label.setText(
            "<b>Training metrics</b><br>"
            f"<b>MAE:</b> {format_metric(metrics.mae)}<br>"
            f"<b>RMSE:</b> {format_metric(metrics.rmse)}<br>"
            f"<b>R²:</b> {format_metric(metrics.r2)}"
        )
        self.radio.setEnabled(True)

    def update_observation(self, observation: str | None) -> None:
        text = observation or "Not recorded"
        self.observation_label.setText(
            f"<b>Observation:</b> {text}"
        )

    def set_selected_style(self, selected: bool) -> None:
        self.card_frame.setStyleSheet(
            SELECTED_CARD_STYLE if selected else CARD_STYLE
        )

        # Keep the title background consistent with the selected card.
        title_background = "#fbf7fa" if selected else "#ffffff"
        self.title_label.setStyleSheet(
            f"""
            QLabel {{
                background-color: {title_background};
                border: none;
                padding: 0 5px;
            }}
            """
        )

        self.radio.blockSignals(True)
        self.radio.setChecked(selected)
        self.radio.blockSignals(False)

    def handle_selected(self) -> None:
        if (
            self.feature_set_name
            not in self.app.training_results[self.model_name]
        ):
            return

        self.app.select_feature_for_model(
            self.model_name,
            self.feature_set_name,
        )

        parent = self.parentWidget()
        while (
            parent is not None
            and not isinstance(parent, ModelTrainingTab)
        ):
            parent = parent.parentWidget()

        if isinstance(parent, ModelTrainingTab):
            parent.refresh_card_selection()

class ModelTrainingTab(QWidget):
    def __init__(self, app: CobberEcoRecoveryApp, model_name: str):
        super().__init__()
        self.app = app
        self.model_name = model_name
        self.cards: dict[str, FeatureSummaryCard] = {}
        self.radio_group = QButtonGroup(self)
        self.radio_group.setExclusive(True)
        self.build_ui()

    def build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        clue_label = QLabel("<b>Recovery clues:</b>")
        self.feature_combo = QComboBox()
        self.feature_combo.addItems(list(FEATURE_SETS.keys()))
        self.feature_combo.setMinimumWidth(240)
        self.feature_combo.currentTextChanged.connect(
            self.refresh_observation_combo
        )
        self.train_button = QPushButton("Train selected clue set")
        self.train_button.clicked.connect(self.train_selected_feature_set)
        set_button_active(self.train_button)

        top_row.addWidget(clue_label)
        top_row.addWidget(self.feature_combo)
        top_row.addWidget(self.train_button)
        top_row.addSpacing(22)

        counts = QLabel(self.app.counts_text())
        counts.setWordWrap(False)
        top_row.addWidget(counts)

        top_row.addStretch(1)
        layout.addLayout(top_row)

        title = QLabel(f"<b>{self.model_name}: train with different recovery clues</b>")
        title.setStyleSheet("font-size: 13pt;")
        layout.addWidget(title)

        note = QLabel(MODEL_NOTES[self.model_name])
        note.setWordWrap(True)
        layout.addWidget(note)

        plots = QHBoxLayout()
        plots.setContentsMargins(0, 0, 0, 0)
        plots.setSpacing(8)

        self.pred_plot = PlotCanvas(width=5.4, height=2.30)
        self.resid_plot = PlotCanvas(width=5.4, height=2.30)

        self.pred_plot.setFixedHeight(365)
        self.resid_plot.setFixedHeight(365)

        plots.addWidget(
            self.pred_plot,
            1,
            Qt.AlignmentFlag.AlignTop,
        )
        plots.addWidget(
            self.resid_plot,
            1,
            Qt.AlignmentFlag.AlignTop,
        )

        layout.addLayout(plots, 0)

        observation_row = QHBoxLayout()
        observation_row.setContentsMargins(0, 0, 0, 0)
        observation_row.setSpacing(8)

        observation_label = QLabel("<b>What pattern do you notice in this plot?</b>")
        self.observation_combo = QComboBox()
        self.observation_combo.addItems(
            COMMON_PLOT_OBSERVATIONS
            + [MODEL_SPECIFIC_OBSERVATIONS[self.model_name]]
        )
        self.observation_combo.setMinimumWidth(430)
        self.observation_combo.setEnabled(False)
        self.observation_combo.currentTextChanged.connect(
            self.save_plot_observation
        )

        observation_row.addWidget(observation_label)
        observation_row.addWidget(self.observation_combo)
        observation_row.addStretch(1)
        layout.addLayout(observation_row)

        cards_row = QHBoxLayout()
        cards_row.setSpacing(8)
        for feature_set_name in FEATURE_SETS.keys():
            card = FeatureSummaryCard(
                self.app,
                self.model_name,
                feature_set_name,
                self.radio_group,
            )
            self.cards[feature_set_name] = card
            cards_row.addWidget(card, 1)

        layout.addLayout(cards_row, 0)

        # Absorb unused vertical space below the content rather than
        # distributing it between the rows above.
        layout.addStretch(1)

    def train_selected_feature_set(self) -> None:
        feature_set_name = self.feature_combo.currentText()
        try:
            actual, pred, metrics = self.app.training_predictions(self.model_name, feature_set_name)
            self.app.training_results[self.model_name][feature_set_name] = metrics
            self.cards[feature_set_name].update_metrics(metrics)
            draw_prediction_plot(
                self.pred_plot,
                actual,
                pred,
                f"{self.model_name} with {feature_set_name}"
            )
            draw_residual_plot(self.resid_plot, actual, pred, f"{feature_set_name}: training residuals")

            saved_observation = self.app.plot_observations[self.model_name].get(
                feature_set_name,
                "Choose an observation...",
            )
            self.observation_combo.blockSignals(True)
            self.observation_combo.setCurrentText(saved_observation)
            self.observation_combo.blockSignals(False)
            self.observation_combo.setEnabled(True)
            self.cards[feature_set_name].update_observation(
                None if saved_observation == "Choose an observation..."
                else saved_observation
            )
        except Exception as exc:
            QMessageBox.critical(self, "Training Error", str(exc))

    def save_plot_observation(self, observation: str) -> None:
        feature_set_name = self.feature_combo.currentText()
        if feature_set_name not in self.app.training_results[self.model_name]:
            return

        if observation == "Choose an observation...":
            self.app.plot_observations[self.model_name].pop(
                feature_set_name,
                None,
            )
            saved_observation = None
        else:
            self.app.plot_observations[self.model_name][
                feature_set_name
            ] = observation
            saved_observation = observation

        self.cards[feature_set_name].update_observation(
            saved_observation
        )

        if hasattr(self.app, "hidden_tab"):
            self.app.hidden_tab.refresh()

    def refresh_observation_combo(self) -> None:
        feature_set_name = self.feature_combo.currentText()
        trained = feature_set_name in self.app.training_results[self.model_name]

        saved_observation = self.app.plot_observations[self.model_name].get(
            feature_set_name,
            "Choose an observation...",
        )
        self.observation_combo.blockSignals(True)
        self.observation_combo.setCurrentText(saved_observation)
        self.observation_combo.blockSignals(False)
        self.observation_combo.setEnabled(trained)

    def refresh_card_selection(self) -> None:
        selected = self.app.selected_feature_by_model.get(self.model_name)
        for feature_set_name, card in self.cards.items():
            card.set_selected_style(feature_set_name == selected)
            card.update_observation(
                self.app.plot_observations[self.model_name].get(
                    feature_set_name
                )
            )

    def refresh(self) -> None:
        self.refresh_card_selection()
        self.refresh_observation_combo()


# ----------------------------------------------------------------------
# Hidden survey plot testing tab
# ----------------------------------------------------------------------

class HiddenSurveyTestTab(QWidget):
    def __init__(self, app: CobberEcoRecoveryApp):
        super().__init__()
        self.app = app
        self.row_radios: dict[str, QRadioButton] = {}
        self.final_radio_group = QButtonGroup(self)
        self.final_radio_group.setExclusive(True)
        self.build_ui()
        self.refresh()

    def build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        title = QLabel("<b>Hidden Test: test the selected setups</b>")
        title.setStyleSheet("font-size: 13pt;")
        top_row.addWidget(title)
        top_row.addStretch(1)
        top_row.addWidget(QLabel(self.app.counts_text()))
        layout.addLayout(top_row)

        self.note = QLabel(
            "The table carries forward one trained setup from each model tab. "
            "Select one row, test that setup on the hidden survey plots, and repeat for the other rows. "
            "After all four setups have been tested, compare the results and select the strongest setup."
        )
        self.note.setWordWrap(True)
        layout.addWidget(self.note)

        self.table = QTableWidget()
        self.table.setWordWrap(True)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 2)

        button_row = QHBoxLayout()

        self.test_button = QPushButton("Test selected setup on hidden survey plots")
        self.test_button.clicked.connect(self.test_selected_setup)
        set_button_enabled(self.test_button, False)
        button_row.addWidget(self.test_button)

        self.use_button = QPushButton("Use this model and features for restoration plan")
        self.use_button.clicked.connect(self.use_selected_setup)
        set_button_enabled(self.use_button, False)
        button_row.addWidget(self.use_button)

        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.status_label = QLabel(
            "Select one setup in the table to begin the hidden test."
        )
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        plots = QHBoxLayout()
        self.pred_plot = PlotCanvas(width=5.4, height=2.8)
        self.resid_plot = PlotCanvas(width=5.4, height=2.8)
        plots.addWidget(self.pred_plot)
        plots.addWidget(self.resid_plot)
        layout.addLayout(plots, 3)



    def selected_setup_for_model(self, model_name: str) -> SetupResult:
        feature_set_name = self.app.selected_feature_by_model.get(model_name)
        train_metrics = None
        if feature_set_name is not None:
            train_metrics = self.app.training_results[model_name].get(
                feature_set_name
            )

        hidden_metrics = None
        result = self.app.hidden_results.get(model_name)
        if (
            result is not None
            and result.feature_set_name == feature_set_name
        ):
            hidden_metrics = result.hidden_metrics

        plot_observation = None
        if feature_set_name is not None:
            plot_observation = self.app.plot_observations[model_name].get(
                feature_set_name
            )

        return SetupResult(
            model_name=model_name,
            feature_set_name=feature_set_name or "Choose on model tab",
            train_metrics=train_metrics,
            hidden_metrics=hidden_metrics,
            plot_observation=plot_observation,
        )

    def refresh(self) -> None:
        headers = [
            "Select",
            "Model",
            "Chosen clues",
            "Plot observation",
            "Train MAE",
            "Train RMSE",
            "Train R²",
            "Hidden MAE",
            "Hidden RMSE",
            "Hidden R²",
        ]
        self.table.clear()
        self.table.setRowCount(len(MODEL_ORDER))
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

        self.row_radios = {}
        self.final_radio_group = QButtonGroup(self)
        self.final_radio_group.setExclusive(True)

        for r, model_name in enumerate(MODEL_ORDER):
            setup = self.selected_setup_for_model(model_name)

            radio = QRadioButton("")
            radio.setStyleSheet(
                f"""
                QRadioButton {{
                    background-color: transparent;
                }}
                QRadioButton::indicator {{
                    width: 14px;
                    height: 14px;
                    border: 1px solid #777777;
                    border-radius: 7px;
                    background-color: white;
                }}
                QRadioButton::indicator:checked {{
                    border: 2px solid {COBBER_MAROON};
                    background-color: {COBBER_MAROON};
                }}
                QRadioButton::indicator:disabled {{
                    border: 1px solid #cccccc;
                    background-color: #f5f5f5;
                }}
                """
            )

            ready = (
                setup.feature_set_name != "Choose on model tab"
                and setup.train_metrics is not None
            )
            radio.setEnabled(ready)
            radio.clicked.connect(
                lambda checked, m=model_name: self.select_setup(m)
            )
            self.final_radio_group.addButton(radio)

            if self.app.hidden_candidate_model == model_name:
                radio.setChecked(True)

            self.row_radios[model_name] = radio

            radio_holder = QWidget()
            radio_layout = QHBoxLayout(radio_holder)
            radio_layout.setContentsMargins(0, 0, 0, 0)
            radio_layout.addStretch(1)
            radio_layout.addWidget(radio)
            radio_layout.addStretch(1)
            self.table.setCellWidget(r, 0, radio_holder)

            hidden_mae = (
                format_metric(setup.hidden_metrics.mae)
                if setup.hidden_metrics else "Not tested"
            )
            hidden_rmse = (
                format_metric(setup.hidden_metrics.rmse)
                if setup.hidden_metrics else "Not tested"
            )
            hidden_r2 = (
                format_metric(setup.hidden_metrics.r2)
                if setup.hidden_metrics else "Not tested"
            )

            values = [
                model_name,
                setup.feature_set_name,
                setup.plot_observation or "Not recorded",
                format_metric(
                    setup.train_metrics.mae if setup.train_metrics else None
                ),
                format_metric(
                    setup.train_metrics.rmse if setup.train_metrics else None
                ),
                format_metric(
                    setup.train_metrics.r2 if setup.train_metrics else None
                ),
                hidden_mae,
                hidden_rmse,
                hidden_r2,
            ]

            for c, value in enumerate(values, start=1):
                item = QTableWidgetItem(value)
                align = (
                    Qt.AlignmentFlag.AlignCenter
                    if c >= 4
                    else Qt.AlignmentFlag.AlignLeft
                    | Qt.AlignmentFlag.AlignVCenter
                )
                item.setTextAlignment(align)
                self.table.setItem(r, c, item)

            self.table.setRowHeight(r, 46)

        selected = self.app.hidden_candidate_model
        selected_ready = (
            selected is not None
            and selected in self.app.selected_feature_by_model
            and self.app.training_results[selected].get(
                self.app.selected_feature_by_model[selected]
            ) is not None
        )
        set_button_enabled(self.test_button, selected_ready)
        if selected_ready:
            set_button_active(self.test_button)

        all_tested = all(
            model_name in self.app.hidden_results
            and self.app.hidden_results[model_name].feature_set_name
            == self.app.selected_feature_by_model.get(model_name)
            for model_name in MODEL_ORDER
        )
        selected_tested = (
            selected is not None
            and selected in self.app.hidden_results
            and self.app.hidden_results[selected].feature_set_name
            == self.app.selected_feature_by_model.get(selected)
        )

        enable_use = all_tested and selected_tested
        set_button_enabled(self.use_button, enable_use)
        if enable_use:
            set_button_active(self.use_button)

        if all_tested:
            self.note.setText(
                "All four selected setups have been tested on the hidden survey plots. "
                "Compare the hidden metrics and plots, then select the strongest setup for the restoration plan."
            )
        else:
            tested_count = sum(
                model_name in self.app.hidden_results
                and self.app.hidden_results[model_name].feature_set_name
                == self.app.selected_feature_by_model.get(model_name)
                for model_name in MODEL_ORDER
            )
            self.note.setText(
                "The table carries forward one trained setup from each model tab. "
                f"{tested_count} of 4 setups have been tested. "
                "Select one row, test that setup on the hidden survey plots, and repeat."
            )

    def select_setup(self, model_name: str) -> None:
        setup = self.selected_setup_for_model(model_name)
        if (
            setup.feature_set_name == "Choose on model tab"
            or setup.train_metrics is None
        ):
            return

        self.app.hidden_candidate_model = model_name

        if setup.hidden_metrics is not None:
            self.update_plots_for_model(model_name)
            self.status_label.setText(
                f"{model_name} with {setup.feature_set_name} has already been tested. "
                "The plots show that hidden-test result."
            )
        else:
            self.pred_plot.figure.clear()
            self.pred_plot.draw()

            self.resid_plot.figure.clear()
            self.resid_plot.draw()

            self.status_label.setText(
                f"Selected: {model_name} with {setup.feature_set_name}. "
                "Click the test button to evaluate this setup on the hidden survey plots."
            )

        self.refresh()

    def test_selected_setup(self) -> None:
        model_name = self.app.hidden_candidate_model
        if model_name is None:
            QMessageBox.warning(
                self,
                "No Setup Selected",
                "Select one setup in the table first.",
            )
            return

        feature_set_name = self.app.selected_feature_by_model.get(model_name)
        if feature_set_name is None:
            QMessageBox.warning(
                self,
                "No Clue Set Selected",
                f"Choose one trained clue set on the {model_name} tab first.",
            )
            return

        train_metrics = self.app.training_results[model_name].get(
            feature_set_name
        )
        if train_metrics is None:
            QMessageBox.warning(
                self,
                "Setup Not Trained",
                f"Train {model_name} with {feature_set_name} first.",
            )
            return

        try:
            actual, pred, hidden_metrics = self.app.hidden_predictions(
                model_name,
                feature_set_name,
            )

            self.app.hidden_results[model_name] = SetupResult(
                model_name=model_name,
                feature_set_name=feature_set_name,
                train_metrics=train_metrics,
                hidden_metrics=hidden_metrics,
                plot_observation=self.app.plot_observations[model_name].get(
                    feature_set_name
                ),
            )
            self.app.final_setup = None

            draw_prediction_plot(
                self.pred_plot,
                actual,
                pred,
                f"{model_name}: hidden survey plots",
            )
            draw_residual_plot(
                self.resid_plot,
                actual,
                pred,
                f"{feature_set_name}: hidden residuals",
            )

            tested_count = sum(
                name in self.app.hidden_results
                and self.app.hidden_results[name].feature_set_name
                == self.app.selected_feature_by_model.get(name)
                for name in MODEL_ORDER
            )

            self.status_label.setText(
                f"Hidden test complete for {model_name} with {feature_set_name}. "
                f"{tested_count} of 4 setups have now been tested."
            )
            self.refresh()

        except Exception as exc:
            QMessageBox.critical(self, "Hidden Test Error", str(exc))

    def update_plots_for_model(self, model_name: str) -> None:
        result = self.app.hidden_results.get(model_name)
        if result is None:
            return

        if (
            result.feature_set_name
            != self.app.selected_feature_by_model.get(model_name)
        ):
            return

        actual, pred, _ = self.app.hidden_predictions(
            result.model_name,
            result.feature_set_name,
        )
        draw_prediction_plot(
            self.pred_plot,
            actual,
            pred,
            f"{result.model_name}: hidden survey plots",
        )
        draw_residual_plot(
            self.resid_plot,
            actual,
            pred,
            f"{result.feature_set_name}: hidden residuals",
        )

    def use_selected_setup(self) -> None:
        chosen = self.app.hidden_candidate_model
        if chosen is None or chosen not in self.app.hidden_results:
            QMessageBox.warning(
                self,
                "No Setup Selected",
                "Select one hidden-tested setup first.",
            )
            return

        all_tested = all(
            model_name in self.app.hidden_results
            and self.app.hidden_results[model_name].feature_set_name
            == self.app.selected_feature_by_model.get(model_name)
            for model_name in MODEL_ORDER
        )
        if not all_tested:
            QMessageBox.warning(
                self,
                "Complete the Hidden Tests",
                "Test all four selected setups before choosing the strongest one.",
            )
            return

        self.app.final_setup = self.app.hidden_results[chosen]
        self.status_label.setText(
            f"Final setup selected: {self.app.final_setup.model_name} "
            f"with {self.app.final_setup.feature_set_name}. "
            "Open the Restoration Plan tab to predict unsurveyed sites."
        )
        self.app.plan_tab.refresh()


# ----------------------------------------------------------------------
# Restoration plan tab
# ----------------------------------------------------------------------

class RestorationPlanTab(QWidget):
    def __init__(self, app: CobberEcoRecoveryApp):
        super().__init__()
        self.app = app
        self.build_ui()
        self.refresh()

    def build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        title = QLabel("<b>Restoration Plan: predict unsurveyed sites</b>")
        title.setStyleSheet("font-size: 13pt;")
        top_row.addWidget(title)
        top_row.addStretch(1)
        top_row.addWidget(QLabel(self.app.counts_text()))
        layout.addLayout(top_row)

        self.setup_label = QLabel("Choose one hidden-tested setup on the Hidden Test tab before predicting unsurveyed sites.")
        self.setup_label.setWordWrap(True)
        self.setup_label.setStyleSheet(INFO_BOX_STYLE)
        layout.addWidget(self.setup_label)

        button_row = QHBoxLayout()
        self.predict_button = QPushButton("Predict unsurveyed sites with final setup")
        self.predict_button.clicked.connect(self.predict_unsurveyed)
        set_button_enabled(self.predict_button, False)
        button_row.addWidget(self.predict_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        guide_frame = QFrame()
        guide_frame.setStyleSheet(
            """
            QFrame {
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: white;
            }
            QLabel {
                border: none;
                background-color: transparent;
            }
            """
        )

        guide_layout = QVBoxLayout(guide_frame)
        guide_layout.setContentsMargins(10, 7, 10, 7)
        guide_layout.setSpacing(5)

        guide_title = QLabel("<b>Planning-note guide:</b>")
        guide_layout.addWidget(guide_title)

        guide_grid = QGridLayout()
        guide_grid.setContentsMargins(0, 0, 0, 0)
        guide_grid.setHorizontalSpacing(28)
        guide_grid.setVerticalSpacing(5)

        monitor_label = QLabel(
            "<span style='background-color:#dff0d8;'>"
            "<b>Monitor recovery</b></span> "
            "Recovery appears strong enough for monitoring."
        )
        monitor_label.setWordWrap(True)

        help_label = QLabel(
            "<span style='background-color:#fce8b2;'>"
            "<b>Possible restoration help</b></span> "
            "Conditions suggest that intervention may be useful."
        )
        help_label.setWordWrap(True)

        field_label = QLabel(
            "<span style='background-color:#f4cccc;'>"
            "<b>Field check before action</b></span> "
            "Visit the site before making a decision."
        )
        field_label.setWordWrap(True)

        review_label = QLabel(
            "<span style='background-color:#ddebf7;'>"
            "<b>Compare with field notes</b></span> "
            "Compare field notes to the prediction before action."
        )
        review_label.setWordWrap(True)

        guide_grid.addWidget(monitor_label, 0, 0)
        guide_grid.addWidget(help_label, 0, 1)
        guide_grid.addWidget(field_label, 1, 0)
        guide_grid.addWidget(review_label, 1, 1)

        guide_grid.setColumnStretch(0, 1)
        guide_grid.setColumnStretch(1, 1)

        guide_layout.addLayout(guide_grid)
        layout.addWidget(guide_frame)

        plan_row = QHBoxLayout()
        plan_row.setContentsMargins(0, 0, 0, 0)
        plan_row.setSpacing(10)

        self.plan_table = QTableWidget()
        self.plan_table.setWordWrap(True)
        header = self.plan_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.plan_table.verticalHeader().setVisible(False)
        self.plan_table.setSelectionMode(
            QTableWidget.SelectionMode.NoSelection
        )

        self.plan_plot = PlotCanvas(width=5.4, height=4.95)
        self.plan_plot.setMinimumWidth(465)
        self.plan_plot.setFixedHeight(495)
        self.plan_plot.figure.clear()
        self.plan_plot.draw()

        plan_row.addWidget(self.plan_table, 11)
        plan_row.addWidget(
            self.plan_plot,
            7,
            Qt.AlignmentFlag.AlignTop,
        )

        layout.addLayout(plan_row, 1)

    def refresh(self) -> None:
        setup = self.app.final_setup
        if setup is None:
            self.setup_label.setText("Choose one hidden-tested setup on the Hidden Test tab before predicting unsurveyed sites.")
            set_button_enabled(self.predict_button, False)
            return
        hidden_rmse = setup.hidden_metrics.rmse if setup.hidden_metrics else None
        self.setup_label.setText(
            f"<b>Final setup:</b> {setup.model_name} with {setup.feature_set_name}. "
            f"Hidden RMSE: {format_metric(hidden_rmse)}. "
            "This setup will be refit with all measured survey plots before predicting unsurveyed sites."
        )
        set_button_active(self.predict_button)

    def predict_unsurveyed(self) -> None:
        setup = self.app.final_setup
        if setup is None:
            QMessageBox.warning(self, "No Final Setup", "Choose one final setup on the Hidden Test tab first.")
            return
        try:
            model = self.app.fit_model(setup.model_name, setup.feature_set_name, use_all_surveyed=True)
            cols = FEATURE_SETS[setup.feature_set_name]
            out = self.app.unsurveyed_df.copy()
            out["predicted_recovery_index"] = np.round(
                model.predict(out[cols]),
                2,
            )
            out["planning_note"] = [
                self.planning_note(row)
                for _, row in out.iterrows()
            ]
            self.populate_plan_table(out)
            draw_restoration_plan_plot(self.plan_plot, out)
        except Exception as exc:
            QMessageBox.critical(self, "Prediction Error", str(exc))

    def planning_note(self, row: pd.Series) -> str:
        pred = float(row["predicted_recovery_index"])
        burn = float(row["burn_severity"])
        moisture = float(row["soil_moisture"])
        green = float(row["greenness_index"])
        slope = float(row["slope_degrees"])

        if 1.8 <= burn <= 3.3:
            return "Field check before action"
        if pred >= 7.0 and green >= 0.50 and moisture >= 0.36:
            return "Monitor recovery"
        if pred < 4.2 or (burn > 3.4 and moisture < 0.38) or slope > 28:
            return "Possible restoration help"
        return "Compare with field notes"

    def populate_plan_table(self, df: pd.DataFrame) -> None:
        headers = ["Site", "Position", "Burn", "Green", "Moisture", "Slope", "Predicted recovery", "Planning note"]
        cols = [
            "site_id",
            "landscape_position",
            "burn_severity",
            "greenness_index",
            "soil_moisture",
            "slope_degrees",
            "predicted_recovery_index",
            "planning_note",
        ]
        self.plan_table.clear()
        self.plan_table.setRowCount(len(df))
        self.plan_table.setColumnCount(len(headers))
        self.plan_table.setHorizontalHeaderLabels(headers)
        for r, (_, row) in enumerate(df.iterrows()):
            for c, col in enumerate(cols):
                value = row[col]
                if col == "site_id":
                    raw = str(value)
                    text = f"Site {raw}" if raw.startswith("U") else raw
                elif col == "landscape_position":
                    text = friendly(str(value))
                elif isinstance(value, (float, np.floating)):
                    text = f"{value:.2f}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter
                    if c not in [1, 7]
                    else Qt.AlignmentFlag.AlignLeft
                    | Qt.AlignmentFlag.AlignVCenter
                )

                planning_note = str(row["planning_note"])
                note_colors = {
                    "Monitor recovery": QColor("#dff0d8"),
                    "Possible restoration help": QColor("#fce8b2"),
                    "Field check before action": QColor("#f4cccc"),
                    "Compare with field notes": QColor("#ddebf7"),
                }

                if c == 7:
                    item.setBackground(note_colors[planning_note])

                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)

                self.plan_table.setItem(r, c, item)
            self.plan_table.setRowHeight(r, 38)


# ----------------------------------------------------------------------
# App stylesheet
# ----------------------------------------------------------------------

def apply_app_stylesheet(app: QApplication) -> None:
    app.setStyleSheet(
        f"""
        QWidget {{
            background-color: #ffffff;
            color: #222222;
            font-family: Lato, Arial, sans-serif;
            font-size: 10.5pt;
        }}
        QLabel {{ background-color: transparent; }}
        QComboBox {{
            background-color: #ffffff;
            color: #111111;
            border: 1px solid #999999;
            border-radius: 4px;
            padding: 4px 8px;
        }}
        QTabWidget::pane {{ border: 1px solid #cccccc; }}
        QTabBar::tab {{
            background: {DARK_GRAY};
            color: white;
            padding: 8px 14px;
            min-width: 128px;
        }}
        QTabBar::tab:selected {{
            background: {COBBER_MAROON};
            color: white;
            font-weight: bold;
        }}
        QTableWidget {{
            background-color: white;
            gridline-color: #dddddd;
            border: 1px solid #cccccc;
        }}
        QTableWidget::item:selected {{
            background-color: white;
            color: #222222;
        }}
        QHeaderView::section {{
            background-color: {COBBER_MAROON};
            color: white;
            font-weight: bold;
            padding: 5px;
            border: 1px solid {COBBER_MAROON};
        }}
        QRadioButton {{ background-color: transparent; }}
        """
    )


if __name__ == "__main__":
    qapp = QApplication(sys.argv)
    qapp.setFont(QFont("Lato"))
    apply_app_stylesheet(qapp)
    window = CobberEcoRecoveryApp()
    window.show()
    sys.exit(qapp.exec())

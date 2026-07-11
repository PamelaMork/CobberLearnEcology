# CobberEcoRecovery_v5.py
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
#   python CobberEcoRecovery_v5.py

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QGroupBox,
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
        "Linear Regression fits one broad trend. Train it with each recovery-clue set and watch whether extra clues help the straight-line baseline."
    ),
    "Decision Tree": (
        "A Decision Tree splits the training sites into regions. Train it with each clue set and watch whether the tree finds useful thresholds."
    ),
    "Random Forest": (
        "A Random Forest averages many tree-based predictions. Train it with each clue set and watch whether it gains from more information."
    ),
    "k-Nearest Neighbors": (
        "k-Nearest Neighbors predicts from similar training sites. Train it with each clue set and watch whether similarity becomes more useful."
    ),
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
QGroupBox {
    border: 1px solid #c7c7c7;
    border-radius: 7px;
    margin-top: 12px;
    padding: 9px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
"""

SELECTED_CARD_STYLE = f"""
QGroupBox {{
    border: 2px solid {COBBER_MAROON};
    border-radius: 7px;
    margin-top: 12px;
    padding: 8px;
    font-weight: bold;
    background-color: #fbf7fa;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {COBBER_MAROON};
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
    min_val = min(float(np.min(actual)), float(np.min(predicted)))
    max_val = max(float(np.max(actual)), float(np.max(predicted)))
    pad = max(0.25, (max_val - min_val) * 0.07)
    lims = [min_val - pad, max_val + pad]
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


# ----------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------

class CobberEcoRecoveryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CobberEcoRecovery")
        self.setGeometry(60, 60, 1460, 850)
        self.setFont(QFont("Lato"))

        self.surveyed_df: pd.DataFrame | None = None
        self.unsurveyed_df: pd.DataFrame | None = None
        self.training_df: pd.DataFrame | None = None
        self.hidden_df: pd.DataFrame | None = None

        self.training_results: dict[str, dict[str, Metrics]] = {m: {} for m in MODEL_ORDER}
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

    def split_text(self) -> str:
        train_n = len(self.training_df) if self.training_df is not None else 0
        hidden_n = len(self.hidden_df) if self.hidden_df is not None else 0
        return f"<b>Split:</b> {train_n} training / {hidden_n} hidden survey plots"

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

class FeatureSummaryCard(QGroupBox):
    def __init__(self, app: CobberEcoRecoveryApp, model_name: str, feature_set_name: str, radio_group: QButtonGroup):
        super().__init__(feature_set_name)
        self.app = app
        self.model_name = model_name
        self.feature_set_name = feature_set_name
        self.setStyleSheet(CARD_STYLE)

        layout = QVBoxLayout(self)
        layout.setSpacing(5)

        radio_row = QHBoxLayout()
        radio_row.addStretch(1)
        self.radio = QRadioButton("")
        self.radio.setToolTip("Select this trained clue set to carry forward.")
        self.radio.setEnabled(False)
        self.radio.clicked.connect(self.handle_selected)
        radio_group.addButton(self.radio)
        radio_row.addWidget(self.radio, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        layout.addLayout(radio_row)

        self.features_label = QLabel(feature_list_text(feature_set_name))
        self.features_label.setWordWrap(True)
        self.features_label.setMinimumHeight(46)
        layout.addWidget(self.features_label)

        self.metrics_label = QLabel("<b>MAE:</b> --<br><b>RMSE:</b> --<br><b>R²:</b> --")
        self.metrics_label.setWordWrap(True)
        layout.addWidget(self.metrics_label)

    def update_metrics(self, metrics: Metrics) -> None:
        self.metrics_label.setText(
            f"<b>MAE:</b> {format_metric(metrics.mae)}<br>"
            f"<b>RMSE:</b> {format_metric(metrics.rmse)}<br>"
            f"<b>R²:</b> {format_metric(metrics.r2)}"
        )
        self.radio.setEnabled(True)

    def set_selected_style(self, selected: bool) -> None:
        self.setStyleSheet(SELECTED_CARD_STYLE if selected else CARD_STYLE)
        self.radio.blockSignals(True)
        self.radio.setChecked(selected)
        self.radio.blockSignals(False)

    def handle_selected(self) -> None:
        if self.feature_set_name not in self.app.training_results[self.model_name]:
            return
        self.app.select_feature_for_model(self.model_name, self.feature_set_name)
        parent = self.parentWidget()
        while parent is not None and not isinstance(parent, ModelTrainingTab):
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
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(7)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        clue_label = QLabel("<b>Recovery clues:</b>")
        self.feature_combo = QComboBox()
        self.feature_combo.addItems(list(FEATURE_SETS.keys()))
        self.feature_combo.setMinimumWidth(240)
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
        top_row.addSpacing(18)

        split = QLabel(self.app.split_text())
        split.setWordWrap(False)
        top_row.addWidget(split)
        top_row.addStretch(1)
        layout.addLayout(top_row)

        title = QLabel(f"<b>{self.model_name}: train with different recovery clues</b>")
        title.setStyleSheet("font-size: 13pt;")
        layout.addWidget(title)

        note = QLabel(
            f"{MODEL_NOTES[self.model_name]} "
            "Train the four clue sets, compare the training metrics, then use the radio button on one panel to carry that setup forward."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        plots = QHBoxLayout()
        self.pred_plot = PlotCanvas(width=5.4, height=2.55)
        self.resid_plot = PlotCanvas(width=5.4, height=2.55)
        plots.addWidget(self.pred_plot)
        plots.addWidget(self.resid_plot)
        layout.addLayout(plots, 3)
        self.pred_plot.clear_message("Training sites: measured vs. predicted", "Train a clue set to see the plot.")
        self.resid_plot.clear_message("Training sites: residuals", "Train a clue set to see residuals.")

        cards_row = QHBoxLayout()
        cards_row.setSpacing(8)
        for feature_set_name in FEATURE_SETS.keys():
            card = FeatureSummaryCard(self.app, self.model_name, feature_set_name, self.radio_group)
            self.cards[feature_set_name] = card
            cards_row.addWidget(card, 1)
        layout.addLayout(cards_row, 2)

    def train_selected_feature_set(self) -> None:
        feature_set_name = self.feature_combo.currentText()
        try:
            actual, pred, metrics = self.app.training_predictions(self.model_name, feature_set_name)
            self.app.training_results[self.model_name][feature_set_name] = metrics
            self.cards[feature_set_name].update_metrics(metrics)
            draw_prediction_plot(self.pred_plot, actual, pred, f"{self.model_name}: training sites")
            draw_residual_plot(self.resid_plot, actual, pred, f"{feature_set_name}: training residuals")
        except Exception as exc:
            QMessageBox.critical(self, "Training Error", str(exc))

    def refresh_card_selection(self) -> None:
        selected = self.app.selected_feature_by_model.get(self.model_name)
        for feature_set_name, card in self.cards.items():
            card.set_selected_style(feature_set_name == selected)

    def refresh(self) -> None:
        self.refresh_card_selection()


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

        note = QLabel(
            "This tab tests only the clue set you selected on each model tab. "
            "Use the hidden survey plots to compare the four selected setups, then choose one strongest setup for the restoration plan."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.table = QTableWidget()
        self.table.setWordWrap(True)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 2)

        button_row = QHBoxLayout()
        self.test_button = QPushButton("Test selected setups on hidden survey plots")
        self.test_button.clicked.connect(self.run_hidden_tests)
        set_button_active(self.test_button)
        button_row.addWidget(self.test_button)

        self.use_button = QPushButton("Use selected setup for restoration plan")
        self.use_button.clicked.connect(self.use_selected_setup)
        set_button_enabled(self.use_button, False)
        button_row.addWidget(self.use_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        plots = QHBoxLayout()
        self.pred_plot = PlotCanvas(width=5.4, height=2.8)
        self.resid_plot = PlotCanvas(width=5.4, height=2.8)
        plots.addWidget(self.pred_plot)
        plots.addWidget(self.resid_plot)
        layout.addLayout(plots, 3)
        self.pred_plot.clear_message("Hidden survey plots: measured vs. predicted", "Run the hidden test to see the plot.")
        self.resid_plot.clear_message("Hidden survey plots: residuals", "Run the hidden test to see residuals.")

    def selected_setup_for_model(self, model_name: str) -> SetupResult:
        feature_set_name = self.app.selected_feature_by_model.get(model_name)
        train_metrics = None
        if feature_set_name is not None:
            train_metrics = self.app.training_results[model_name].get(feature_set_name)
        hidden_metrics = None
        if model_name in self.app.hidden_results:
            hidden_metrics = self.app.hidden_results[model_name].hidden_metrics
        return SetupResult(
            model_name=model_name,
            feature_set_name=feature_set_name or "Choose on model tab",
            train_metrics=train_metrics,
            hidden_metrics=hidden_metrics,
        )

    def refresh(self) -> None:
        headers = ["Use", "Model", "Chosen clues", "Train MAE", "Train RMSE", "Train R²", "Hidden MAE", "Hidden RMSE", "Hidden R²"]
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
            radio.setEnabled(setup.hidden_metrics is not None)
            radio.clicked.connect(lambda checked, m=model_name: self.choose_candidate(m))
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

            values = [
                model_name,
                setup.feature_set_name,
                format_metric(setup.train_metrics.mae if setup.train_metrics else None),
                format_metric(setup.train_metrics.rmse if setup.train_metrics else None),
                format_metric(setup.train_metrics.r2 if setup.train_metrics else None),
                format_metric(setup.hidden_metrics.mae if setup.hidden_metrics else None),
                format_metric(setup.hidden_metrics.rmse if setup.hidden_metrics else None),
                format_metric(setup.hidden_metrics.r2 if setup.hidden_metrics else None),
            ]
            for c, value in enumerate(values, start=1):
                item = QTableWidgetItem(value)
                align = Qt.AlignmentFlag.AlignCenter if c >= 3 else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                item.setTextAlignment(align)
                self.table.setItem(r, c, item)
            self.table.setRowHeight(r, 38)

        enable_use = self.app.hidden_candidate_model in self.app.hidden_results
        set_button_enabled(self.use_button, enable_use)
        if enable_use:
            set_button_active(self.use_button)

    def run_hidden_tests(self) -> None:
        try:
            missing = [m for m in MODEL_ORDER if m not in self.app.selected_feature_by_model]
            if missing:
                raise ValueError("Choose one trained clue set on each model tab before running the hidden test.")
            for model_name in MODEL_ORDER:
                feature_set_name = self.app.selected_feature_by_model[model_name]
                train_metrics = self.app.training_results[model_name].get(feature_set_name)
                if train_metrics is None:
                    raise ValueError(f"Train {model_name} with {feature_set_name} before testing it.")
                _, _, hidden_metrics = self.app.hidden_predictions(model_name, feature_set_name)
                self.app.hidden_results[model_name] = SetupResult(
                    model_name=model_name,
                    feature_set_name=feature_set_name,
                    train_metrics=train_metrics,
                    hidden_metrics=hidden_metrics,
                )
            self.app.final_setup = None
            self.app.hidden_candidate_model = None
            self.status_label.setText("Hidden survey plot test complete. Select the strongest setup, then send it to the restoration plan.")
            self.refresh()
            self.show_best_hidden_plot()
        except Exception as exc:
            QMessageBox.critical(self, "Hidden Test Error", str(exc))

    def show_best_hidden_plot(self) -> None:
        if not self.app.hidden_results:
            return
        best_model = min(self.app.hidden_results, key=lambda m: self.app.hidden_results[m].hidden_metrics.rmse)
        self.update_plots_for_model(best_model)

    def choose_candidate(self, model_name: str) -> None:
        if model_name not in self.app.hidden_results:
            return
        self.app.hidden_candidate_model = model_name
        self.update_plots_for_model(model_name)
        set_button_active(self.use_button)

    def update_plots_for_model(self, model_name: str) -> None:
        if model_name not in self.app.hidden_results:
            return
        setup = self.app.hidden_results[model_name]
        actual, pred, _ = self.app.hidden_predictions(setup.model_name, setup.feature_set_name)
        draw_prediction_plot(self.pred_plot, actual, pred, f"{setup.model_name}: hidden survey plots")
        draw_residual_plot(self.resid_plot, actual, pred, f"{setup.feature_set_name}: hidden residuals")

    def use_selected_setup(self) -> None:
        chosen = self.app.hidden_candidate_model
        if chosen is None or chosen not in self.app.hidden_results:
            QMessageBox.warning(self, "No Setup Selected", "Select one hidden-tested setup first.")
            return
        self.app.final_setup = self.app.hidden_results[chosen]
        self.status_label.setText(
            f"Final setup selected: {self.app.final_setup.model_name} with {self.app.final_setup.feature_set_name}. "
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

        self.plan_table = QTableWidget()
        self.plan_table.setWordWrap(True)
        self.plan_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.plan_table.verticalHeader().setVisible(False)
        self.plan_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        layout.addWidget(self.plan_table, 1)

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
            out["predicted_recovery_index"] = np.round(model.predict(out[cols]), 2)
            out["planning_note"] = [self.planning_note(row) for _, row in out.iterrows()]
            self.populate_plan_table(out)
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
        return "Review with field notes"

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
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter if c not in [1, 7] else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
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

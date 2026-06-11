# CobberEcoResidue.py
# A teaching app for exploring prediction error in ecological models.
# Built for the ML for Ecology error-analysis chapter:
# "Judging the Fit: A Practical Guide to Error Analysis in Ecology"

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QFileDialog, QLabel, QTextEdit,
    QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ---------------------------------------------------------------------
# Dataset list
# ---------------------------------------------------------------------

DATASET_OPTIONS = {
    "Good overall fit": "good_overall_fit.csv",
    "High scatter": "high_scatter.csv",
    "Consistent overprediction": "consistent_overprediction.csv",
    "Consistent underprediction": "consistent_underprediction.csv",
    "Severe blooms underpredicted": "severe_blooms_underpredicted.csv",
    "Developed lakes underpredicted": "developed_lakes_underpredicted.csv",
    "August blooms underpredicted": "august_blooms_underpredicted.csv",
    "Post-storm blooms missed": "post_storm_blooms_missed.csv",
}

GROUPING_COLUMNS = {
    "Bloom severity": "BloomSeverity",
    "Month": "Month",
    "Lake setting": "LakeSetting",
    "Storm status": "StormStatus",
}

DEFAULT_GROUPING_BY_DATASET = {
    "Good overall fit": "Bloom severity",
    "High scatter": "Bloom severity",
    "Consistent overprediction": "Bloom severity",
    "Consistent underprediction": "Bloom severity",
    "Severe blooms underpredicted": "Bloom severity",
    "Developed lakes underpredicted": "Lake setting",
    "August blooms underpredicted": "Month",
    "Post-storm blooms missed": "Storm status",
}

DATASET_STORIES = {
    "Good overall fit": (
        "Predictions are close across most lake samples. "
        "Use this as a baseline for what a reasonably balanced model can look like."
    ),
    "High scatter": (
        "Predictions are noisy. The model is not strongly high or low overall, "
        "but many individual errors are large."
    ),
    "Consistent overprediction": (
        "The model predicts too high across most lake samples."
    ),
    "Consistent underprediction": (
        "The model predicts too low across most lake samples."
    ),
    "Severe blooms underpredicted": (
        "Severe blooms are often underpredicted, even when the overall fit "
        "looks somewhat acceptable."
    ),
    "Developed lakes underpredicted": (
        "Developed shoreline lakes are underpredicted more strongly than "
        "isolated shoreline lakes."
    ),
    "August blooms underpredicted": (
        "August bloom conditions are underpredicted more strongly than "
        "other months."
    ),
    "Post-storm blooms missed": (
        "After-storm bloom samples are underpredicted more strongly than "
        "before-storm samples."
    ),
}
# ---------------------------------------------------------------------
# Book-ish colors
# ---------------------------------------------------------------------

COBBER_MAROON = "#7A0019"
INFO_BLUE = "#2E6B8E"
PROJECT_GREEN = "#3B7A57"
ETHICS_LAVENDER = "#6F5AA7"
CHARCOAL = "#2F2F2F"
LIGHT_GRAY = "#F5F5F5"


# ---------------------------------------------------------------------
# Matplotlib canvas
# ---------------------------------------------------------------------

class PlotCanvas(FigureCanvas):
    def __init__(self, width=7, height=5, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()


# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------

class CobberEcoResidueApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CobberEcoResidue: Judging Model Fit")
        self.resize(1350, 850)

        self.df = None
        self.current_dataset_name = None
        self.current_file_path = None

        self._build_ui()
        self._apply_styles()

        # Try to load the default teaching dataset.
        self.load_selected_dataset()

    # -----------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------

    def _build_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        title = QLabel("CobberEcoResidue")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setObjectName("AppTitle")

        subtitle = QLabel("Judging Model Fit in Ecological Predictions")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setObjectName("AppSubtitle")

        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        # Shared top controls
        control_layout = QHBoxLayout()

        dataset_label = QLabel("Choose dataset:")
        dataset_label.setObjectName("ControlLabel")




        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(DATASET_OPTIONS.keys())
        self.dataset_combo.currentIndexChanged.connect(self.load_selected_dataset)
        self.dataset_combo.setFixedWidth(320)




        control_layout.addWidget(dataset_label)
        control_layout.addWidget(self.dataset_combo)
        control_layout.addStretch()



        main_layout.addLayout(control_layout)

        self.dataset_story_label = QLabel("")
        self.dataset_story_label.setWordWrap(True)
        self.dataset_story_label.setObjectName("DatasetStory")
        main_layout.addWidget(self.dataset_story_label)

        # Tabs
        self.tabs = QTabWidget()
        self.overall_tab = QWidget()
        self.group_tab = QWidget()

        self.tabs.addTab(self.overall_tab, "Overall Fit")
        self.tabs.addTab(self.group_tab, "Where Errors Happen")

        self._build_overall_tab()
        self._build_group_tab()

        main_layout.addWidget(self.tabs)
        self.setCentralWidget(main_widget)

    def _build_overall_tab(self):
        layout = QGridLayout(self.overall_tab)

        self.overall_canvas = PlotCanvas(width=9, height=6)
        layout.addWidget(self.overall_canvas, 0, 0, 2, 2)

        self.metric_box = QTextEdit()
        self.metric_box.setReadOnly(True)
        self.metric_box.setObjectName("MetricBox")
        layout.addWidget(self.metric_box, 0, 2)

        self.overall_interpretation = QTextEdit()
        self.overall_interpretation.setReadOnly(True)
        self.overall_interpretation.setObjectName("InterpretationBox")
        layout.addWidget(self.overall_interpretation, 1, 2)

        layout.setColumnStretch(0, 3)
        layout.setColumnStretch(1, 3)
        layout.setColumnStretch(2, 2)
        layout.setRowStretch(0, 3)
        layout.setRowStretch(1, 2)

    def _build_group_tab(self):
        layout = QVBoxLayout(self.group_tab)

        top_layout = QHBoxLayout()

        group_label = QLabel("Group errors by:")
        group_label.setObjectName("ControlLabel")

        self.group_combo = QComboBox()
        self.group_combo.currentIndexChanged.connect(self.update_group_view)

        top_layout.addWidget(group_label)
        top_layout.addWidget(self.group_combo)
        top_layout.addStretch()

        layout.addLayout(top_layout)




        content_layout = QGridLayout()

        self.group_canvas = PlotCanvas(width=8, height=5)
        content_layout.addWidget(self.group_canvas, 0, 0)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.group_table = QTableWidget()
        self.group_table.setObjectName("GroupTable")
        self.group_table.setMinimumHeight(120)
        self.group_table.setMaximumHeight(270)
        self.group_table.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed
        )

        self.group_interpretation = QTextEdit()
        self.group_interpretation.setReadOnly(True)
        self.group_interpretation.setObjectName("InterpretationBox")
        self.group_interpretation.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        right_layout.addWidget(self.group_table)
        right_layout.addWidget(self.group_interpretation, stretch=1)

        content_layout.addWidget(right_panel, 0, 1)

        content_layout.setColumnStretch(0, 3)
        content_layout.setColumnStretch(1, 2)

        layout.addLayout(content_layout)




    def _apply_styles(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: white;
            }}

            QLabel#AppTitle {{
                font-size: 26px;
                font-weight: bold;
                color: {COBBER_MAROON};
                padding-top: 8px;
            }}

            QLabel#AppSubtitle {{
                font-size: 15px;
                color: {CHARCOAL};
                padding-bottom: 8px;
            }}

            QLabel#ControlLabel {{
                font-size: 14px;
                font-weight: bold;
                color: {CHARCOAL};
            }}

            QLabel#DatasetStory {{
                background-color: #F5F5F5;
                color: {CHARCOAL};
                border: 1px solid #DDDDDD;
                border-radius: 6px;
                padding: 7px 10px;
                font-size: 13px;
            }}
            
            QPushButton {{
                background-color: {COBBER_MAROON};
                color: white;
                font-weight: bold;
                border-radius: 5px;
                padding: 7px 12px;
            }}

            QPushButton:hover {{
                background-color: #9B1C31;
            }}

            QComboBox {{
                padding: 5px;
                font-size: 13px;
            }}

            QTabWidget::pane {{
                border: 1px solid #DDDDDD;
                background-color: white;
            }}

            QTabBar::tab {{
                background-color: #6B6B6B;
                color: white;
                font-weight: bold;
                padding: 7px 14px;
                border: 1px solid #555555;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }}

            QTabBar::tab:selected {{
                background-color: {COBBER_MAROON};
                color: white;
                font-weight: bold;
            }}

            QTabBar::tab:!selected:hover {{
                background-color: #555555;
            }}
            
            
            QTextEdit#MetricBox {{
                background-color: {LIGHT_GRAY};
                border: 1px solid #CCCCCC;
                border-radius: 6px;
                padding: 8px;
                font-size: 13px;
            }}

            QTextEdit#InterpretationBox {{
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                border-radius: 6px;
                padding: 8px;
                font-size: 13px;
            }}

            QTableWidget#GroupTable {{
                background-color: white;
                border: 1px solid #CCCCCC;
                gridline-color: #DDDDDD;
                font-size: 12px;
            }}

            QHeaderView::section {{
                background-color: {COBBER_MAROON};
                color: white;
                font-weight: bold;
                padding: 5px;
                border: 1px solid white;
            }}
        """)

    # -----------------------------------------------------------------
    # Dataset loading
    # -----------------------------------------------------------------

    def find_dataset_file(self, filename):
        """
        Search likely folder locations for the selected teaching dataset.
        """

        base_dir = Path(__file__).resolve().parent

        candidate_dirs = [
            base_dir,
            base_dir / "data",
            base_dir / "data" / "Error",
            base_dir / "datasets",
            base_dir / "cobber_ecoresidue_algal_bloom_datasets",
            Path.cwd(),
            Path.cwd() / "data",
            Path.cwd() / "data" / "Error",
            Path.cwd() / "datasets",
            Path.cwd() / "cobber_ecoresidue_algal_bloom_datasets",
        ]

        for folder in candidate_dirs:
            path = folder / filename
            if path.exists():
                return path

        return None

    def load_selected_dataset(self):
        dataset_name = self.dataset_combo.currentText()
        filename = DATASET_OPTIONS.get(dataset_name)

        if filename is None:
            return

        path = self.find_dataset_file(filename)

        if path is None:
            self.clear_display()
            self.dataset_story_label.setText(
                "<b>Dataset story:</b> No dataset is loaded yet."
            )
            self.metric_box.setText(
                "Dataset not found.\n\n"
                f"I looked for:\n{filename}\n\n"
                "Put the CSV files in one of these places:\n"
                "• the same folder as CobberEcoResidue.py\n"
                "• a folder named data\n"
                "• a folder named data/Error\n"
                "• a folder named datasets\n"
                "• a folder named cobber_ecoresidue_algal_bloom_datasets"
            )
            self.overall_interpretation.setText(
                "No dataset is loaded yet. Once the CSV files are in the right folder, "
                "choose a dataset from the dropdown or click Reload Dataset."
            )
            return

        try:
            df = pd.read_csv(path)
            self.set_dataframe(df, dataset_name=dataset_name, file_path=path)
        except Exception as error:
            QMessageBox.critical(
                self,
                "Could not load dataset",
                f"Could not load:\n{path}\n\nError:\n{error}"
            )

    def load_custom_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Custom CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)
            self.set_dataframe(
                df,
                dataset_name="Custom loaded dataset",
                file_path=Path(file_path)
            )
        except Exception as error:
            QMessageBox.critical(
                self,
                "Could not load custom CSV",
                f"Could not load:\n{file_path}\n\nError:\n{error}"
            )

    def reload_current_dataset(self):
        if self.current_file_path and Path(self.current_file_path).exists():
            try:
                df = pd.read_csv(self.current_file_path)
                self.set_dataframe(
                    df,
                    dataset_name=self.current_dataset_name,
                    file_path=self.current_file_path
                )
            except Exception as error:
                QMessageBox.critical(
                    self,
                    "Could not reload dataset",
                    f"Could not reload:\n{self.current_file_path}\n\nError:\n{error}"
                )
        else:
            self.load_selected_dataset()

    def set_dataframe(self, df, dataset_name, file_path):
        df = self.clean_columns(df)

        required = ["Actual", "Predicted"]
        missing = [col for col in required if col not in df.columns]

        if missing:
            QMessageBox.warning(
                self,
                "Missing columns",
                "This CSV must contain columns named Actual and Predicted.\n\n"
                f"Missing: {', '.join(missing)}"
            )
            return

        df = df.copy()
        df["Actual"] = pd.to_numeric(df["Actual"], errors="coerce")
        df["Predicted"] = pd.to_numeric(df["Predicted"], errors="coerce")
        df = df.dropna(subset=["Actual", "Predicted"])

        if df.empty:
            QMessageBox.warning(
                self,
                "No usable data",
                "After reading Actual and Predicted values, no usable rows remained."
            )
            return

        df["Residual"] = df["Predicted"] - df["Actual"]

        # Make grouping columns readable if present.
        for col in ["Lake", "Month", "LakeSetting", "StormStatus", "BloomSeverity"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

        self.df = df
        self.current_dataset_name = dataset_name
        self.current_file_path = file_path

        story = DATASET_STORIES.get(
            dataset_name,
            "Custom dataset loaded. Use the plots and metrics to judge where the model misses."
        )
        self.dataset_story_label.setText(f"<b>Dataset story:</b> {story}")

        self.update_overall_view()
        self.update_grouping_options()
        self.update_group_view()

    def clean_columns(self, df):
        """
        Accept exact columns, but also rescue common case variants.
        """

        df = df.copy()
        rename_map = {}

        normalized = {col.strip().lower(): col for col in df.columns}

        expected = {
            "actual": "Actual",
            "predicted": "Predicted",
            "lake": "Lake",
            "month": "Month",
            "lakesetting": "LakeSetting",
            "lake_setting": "LakeSetting",
            "stormstatus": "StormStatus",
            "storm_status": "StormStatus",
            "bloomseverity": "BloomSeverity",
            "bloom_severity": "BloomSeverity",
        }

        for low_name, proper_name in expected.items():
            if low_name in normalized:
                original = normalized[low_name]
                rename_map[original] = proper_name

        return df.rename(columns=rename_map)

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------

    def calculate_metrics(self, data):
        actual = data["Actual"].to_numpy(dtype=float)
        predicted = data["Predicted"].to_numpy(dtype=float)
        residuals = predicted - actual

        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))
        bias = np.mean(residuals)

        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)

        if ss_tot == 0:
            r2 = np.nan
        else:
            r2 = 1 - (ss_res / ss_tot)

        return {
            "count": len(data),
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "r2": r2,
        }

    # -----------------------------------------------------------------
    # Overall Fit tab
    # -----------------------------------------------------------------

    def update_overall_view(self):
        if self.df is None:
            self.clear_display()
            return

        metrics = self.calculate_metrics(self.df)

        self.metric_box.setHtml(self.make_metric_text(metrics))
        self.overall_interpretation.setHtml(self.make_overall_interpretation(metrics))
        self.draw_overall_plots()

    def make_metric_text(self, metrics):
        r2_text = "not defined" if np.isnan(metrics["r2"]) else f"{metrics['r2']:.3f}"

        return (
            "<b>Model Evaluation Metrics</b><br>"
            "────────────────────────<br><br>"
            f"Dataset: {self.current_dataset_name}<br>"
            f"Rows: {metrics['count']}<br><br>"
            f"<b>MAE:</b> {metrics['mae']:.2f} µg/L<br>"
            "Typical prediction error.<br><br>"
            f"<b>RMSE:</b> {metrics['rmse']:.2f} µg/L<br>"
            "Highlights larger misses.<br><br>"
            f"<b>Bias:</b> {metrics['bias']:.2f} µg/L<br>"
            f"{self.bias_sentence(metrics['bias'])}<br><br>"
            f"<b>R²:</b> {r2_text}<br>"
            f"{self.r2_sentence(metrics['r2'])}<br>"
        )

    def bias_sentence(self, bias):
        if bias > 3:
            return "The model overpredicts on average."
        elif bias < -3:
            return "The model underpredicts on average."
        else:
            return "The average bias is near zero."

    def r2_sentence(self, r2):
        if np.isnan(r2):
            return "R² cannot be calculated for this dataset."
        elif r2 >= 0.80:
            return "The model follows the overall pattern fairly well."
        elif r2 >= 0.50:
            return "The model follows part of the overall pattern."
        else:
            return "The model does not follow the overall pattern strongly."

    def make_overall_interpretation(self, metrics):
        lines = []
        lines.append("<b>Overall Interpretation</b>")
        lines.append("──────────────────────")


        if metrics["rmse"] > metrics["mae"] * 1.35:
            lines.append(
                "RMSE is larger than MAE, so some larger errors may matter."
            )
        else:
            lines.append(
                "MAE and RMSE are fairly close, so the errors are more even in size."
            )





        if metrics["bias"] < -3:
            lines.append(
                "\nBias is negative, so the model underpredicts on average."
            )
        elif metrics["bias"] > 3:
            lines.append(
                "\nBias is positive, so the model overpredicts on average."
            )
        else:
            lines.append(
                "\nBias is near zero, so overpredictions and underpredictions balance out overall."
            )




        if not np.isnan(metrics["r2"]) and metrics["r2"] >= 0.75:
            lines.append(
                "\nR² is fairly strong, so the model follows the overall pattern."
            )
        elif not np.isnan(metrics["r2"]) and metrics["r2"] >= 0.50:
            lines.append(
                "\nR² is moderate, so the model follows part of the overall pattern."
            )
        elif not np.isnan(metrics["r2"]):
            lines.append(
                "\nR² is low, so the model does not follow the overall pattern well."
            )

        lines.append(
            "<b>Next step:</b> Use the residual plot and the Where Errors Happen tab to see where the model misses."
        )

        header = "<br>".join(lines[:2])
        body = "<br><br>".join(lines[2:])
        return header + "<br><br>" + body

    def draw_overall_plots(self):
        self.overall_canvas.figure.clear()

        ax1 = self.overall_canvas.figure.add_subplot(1, 2, 1)
        ax2 = self.overall_canvas.figure.add_subplot(1, 2, 2)

        actual = self.df["Actual"]
        predicted = self.df["Predicted"]
        residuals = self.df["Residual"]

        # Predicted vs actual
        ax1.scatter(
            actual,
            predicted,
            s=42,
            color=INFO_BLUE,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
            label="Lake samples"
        )



        max_value = max(actual.max(), predicted.max())
        plot_upper = max_value * 1.08

        ax1.plot(
            [0, plot_upper],
            [0, plot_upper],
            linestyle="--",
            color=COBBER_MAROON,
            linewidth=2,
            label="Ideal fit"
        )

        ax1.set_xlim(0, plot_upper)
        ax1.set_ylim(0, plot_upper)




        ax1.set_title("Predicted vs. Actual Chlorophyll a", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Actual chlorophyll a (µg/L)")
        ax1.set_ylabel("Predicted chlorophyll a (µg/L)")
        ax1.legend(loc="best", fontsize=8)
        ax1.grid(True, alpha=0.25)

        # Residual plot
        ax2.scatter(
            actual,
            residuals,
            s=42,
            color=PROJECT_GREEN,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
            label="Lake samples"
        )

        ax2.axhline(
            0,
            linestyle="--",
            color=COBBER_MAROON,
            linewidth=2,
            label="Zero error"
        )

        residual_x_upper = actual.max() * 1.08
        ax2.set_xlim(0, residual_x_upper)

        ax2.set_title("Residuals vs. Actual Chlorophyll a", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Actual chlorophyll a (µg/L)")
        ax2.set_ylabel("Residual (µg/L)", labelpad=-8)
        ax2.legend(loc="best", fontsize=8)
        ax2.grid(True, alpha=0.25)

        self.overall_canvas.figure.tight_layout()
        self.overall_canvas.draw()

    # -----------------------------------------------------------------
    # Group tab
    # -----------------------------------------------------------------

    def update_grouping_options(self):
        self.group_combo.blockSignals(True)
        self.group_combo.clear()

        if self.df is None:
            self.group_combo.blockSignals(False)
            return

        for label, column in GROUPING_COLUMNS.items():
            if column in self.df.columns:
                self.group_combo.addItem(label, column)

        default_group = DEFAULT_GROUPING_BY_DATASET.get(self.current_dataset_name)

        if default_group:
            index = self.group_combo.findText(default_group)
            if index >= 0:
                self.group_combo.setCurrentIndex(index)

        self.group_combo.blockSignals(False)





    def update_group_view(self):
        if self.df is None:
            return

        if self.group_combo.count() == 0:
            self.group_canvas.figure.clear()
            self.group_canvas.draw()
            self.group_table.clear()
            self.group_interpretation.setText(
                "No grouping columns were found in this dataset.\n\n"
                "To use this tab, the CSV needs one or more of these columns:\n"
                "BloomSeverity, Month, LakeSetting, StormStatus."
            )
            return

        group_column = self.group_combo.currentData()

        if group_column is None or group_column not in self.df.columns:
            return

        grouped_metrics = self.calculate_group_metrics(group_column)

        self.draw_group_residual_plot(group_column)
        self.populate_group_table(grouped_metrics)
        self.group_interpretation.setHtml(
            self.make_group_interpretation(group_column, grouped_metrics)
        )

    def calculate_group_metrics(self, group_column):
        rows = []

        for group_name, group_df in self.df.groupby(group_column):
            metrics = self.calculate_metrics(group_df)
            rows.append({
                "Group": str(group_name),
                "Count": metrics["count"],
                "MAE": metrics["mae"],
                "RMSE": metrics["rmse"],
                "Bias": metrics["bias"],
            })

        result = pd.DataFrame(rows)

        # Sort months in calendar order if needed.
        if group_column == "Month":
            month_order = {
                "May": 1,
                "June": 2,
                "July": 3,
                "August": 4,
                "September": 5,
            }
            result["SortOrder"] = result["Group"].map(month_order).fillna(99)
            result = result.sort_values("SortOrder").drop(columns=["SortOrder"])
        else:
            result = result.sort_values("Group")

        return result.reset_index(drop=True)

    def populate_group_table(self, grouped_metrics):
        self.group_table.clear()
        self.group_table.setRowCount(len(grouped_metrics))
        self.group_table.setColumnCount(5)
        self.group_table.setHorizontalHeaderLabels(["Group", "Count", "MAE", "RMSE", "Bias"])

        for row_index, row in grouped_metrics.iterrows():
            values = [
                row["Group"],
                f"{int(row['Count'])}",
                f"{row['MAE']:.2f}",
                f"{row['RMSE']:.2f}",
                f"{row['Bias']:.2f}",
            ]

            for col_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.group_table.setItem(row_index, col_index, item)

        self.group_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.group_table.verticalHeader().setVisible(False)

        row_height = 31
        header_height = 38
        extra_space = 14

        table_height = header_height + extra_space + row_height * len(grouped_metrics)
        table_height = min(270, max(120, table_height))

        self.group_table.setFixedHeight(table_height)

        row_height = 31
        header_height = 38
        extra_space = 12
        table_height = header_height + extra_space + row_height * len(grouped_metrics)
        table_height = min(260, max(120, table_height))
        self.group_table.setFixedHeight(table_height)

    def draw_group_residual_plot(self, group_column):
        self.group_canvas.figure.clear()
        ax = self.group_canvas.figure.add_subplot(1, 1, 1)

        unique_groups = list(self.df[group_column].dropna().unique())

        palette = [
            INFO_BLUE,
            PROJECT_GREEN,
            COBBER_MAROON,
            ETHICS_LAVENDER,
            "#C27D38",
            "#4C7C9C",
            "#8C6D31",
            "#5B8C5A",
        ]

        for index, group in enumerate(unique_groups):
            group_df = self.df[self.df[group_column] == group]
            color = palette[index % len(palette)]

            ax.scatter(
                group_df["Actual"],
                group_df["Residual"],
                s=46,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.88,
                label=str(group)
            )

        ax.axhline(
            0,
            linestyle="--",
            color=COBBER_MAROON,
            linewidth=2,
            label="Zero error"
        )

        friendly_label = self.group_combo.currentText()

        ax.set_title(
            f"Residuals by {friendly_label.title()}",
            fontsize=12,
            fontweight="bold"
        )
        ax.set_xlabel("Actual chlorophyll a (µg/L)")
        ax.set_ylabel("Residual (µg/L)", labelpad=-8)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

        self.group_canvas.figure.tight_layout()
        self.group_canvas.draw()




    def make_group_interpretation(self, group_column, grouped_metrics):
        lines = []
        friendly = self.group_combo.currentText()

        def format_interpretation(lines_to_format):
            header = "<br>".join(lines_to_format[:2])
            body = "<br><br>".join(lines_to_format[2:])
            return header + "<br><br>" + body

        lines.append("<b>Where Errors Happen</b>")
        lines.append("───────────────────")
        lines.append(f"Grouping variable: {friendly}")

        if grouped_metrics.empty:
            lines.append("No groups were available for this dataset.")
            return format_interpretation(lines)

        worst_mae_row = grouped_metrics.loc[grouped_metrics["MAE"].idxmax()]
        most_negative_bias_row = grouped_metrics.loc[grouped_metrics["Bias"].idxmin()]
        most_positive_bias_row = grouped_metrics.loc[grouped_metrics["Bias"].idxmax()]

        strong_bias_cutoff = 5.0
        strong_error_cutoff = 8.0

        all_positive_bias = (grouped_metrics["Bias"] > strong_bias_cutoff).all()
        all_negative_bias = (grouped_metrics["Bias"] < -strong_bias_cutoff).all()
        no_strong_bias = (grouped_metrics["Bias"].abs() < strong_bias_cutoff).all()
        no_large_mae = (grouped_metrics["MAE"] < strong_error_cutoff).all()

        # Pattern 1: Balanced errors
        if no_strong_bias and no_large_mae:
            lines.append("The errors are fairly small across these groups.")
            lines.append("No group shows a strong positive or negative bias.")
            lines.append(
                "This is what a reasonably balanced model can look like. "
                "The residual plot helps check whether that balance holds across ecological groups."
            )
            return format_interpretation(lines)

        # Pattern 2: Broad overprediction
        if all_positive_bias:
            lines.append("All groups show positive bias.")
            lines.append(
                "This suggests broad overprediction across the dataset, not a failure "
                "limited to one ecological group."
            )
            lines.append(
                f"The strongest overprediction is in: {most_positive_bias_row['Group']} "
                f"(Bias = {most_positive_bias_row['Bias']:.2f} µg/L)."
            )
            lines.append(
                "Use the residual plot to check whether most points sit above the zero-error line."
            )
            return format_interpretation(lines)

        # Pattern 3: Broad underprediction
        if all_negative_bias:
            lines.append("All groups show negative bias.")
            lines.append(
                "This suggests broad underprediction across the dataset, not a failure "
                "limited to one ecological group."
            )
            lines.append(
                f"The strongest underprediction is in: {most_negative_bias_row['Group']} "
                f"(Bias = {most_negative_bias_row['Bias']:.2f} µg/L)."
            )
            lines.append(
                "Use the residual plot to check whether most points sit below the zero-error line."
            )
            return format_interpretation(lines)

        # Pattern 4: Group-specific problem
        lines.append(
            f"The largest typical error is in: {worst_mae_row['Group']} "
            f"(MAE = {worst_mae_row['MAE']:.2f} µg/L)."
        )

        if most_negative_bias_row["Bias"] < -strong_bias_cutoff:
            lines.append(
                f"The strongest underprediction is in: {most_negative_bias_row['Group']} "
                f"(Bias = {most_negative_bias_row['Bias']:.2f} µg/L)."
            )

        if most_positive_bias_row["Bias"] > strong_bias_cutoff:
            lines.append(
                f"The strongest overprediction is in: {most_positive_bias_row['Group']} "
                f"(Bias = {most_positive_bias_row['Bias']:.2f} µg/L)."
            )

        if group_column == "BloomSeverity":
            lines.append(
                "Ecological question: Does the model miss mild, moderate, and severe blooms "
                "in the same way?"
            )

            severe = grouped_metrics[grouped_metrics["Group"].str.lower() == "severe"]
            if not severe.empty:
                severe_bias = severe.iloc[0]["Bias"]
                if severe_bias < -strong_bias_cutoff:
                    lines.append(
                        "Severe blooms have a negative bias. The model often "
                        "underpredicts high chlorophyll a values."
                    )
                    lines.append(
                        "That pattern matters because severe blooms are the cases where "
                        "missed warnings may be most costly."
                    )

        elif group_column == "Month":
            lines.append(
                "Ecological question: Does model error change across the bloom season?"
            )

            august = grouped_metrics[grouped_metrics["Group"].str.lower() == "august"]
            if not august.empty:
                august_bias = august.iloc[0]["Bias"]
                if august_bias < -strong_bias_cutoff:
                    lines.append(
                        "August has a negative bias. The model may be missing late-summer "
                        "bloom conditions."
                    )
                    lines.append(
                        "Warm water and accumulated nutrients can make late-summer errors "
                        "especially important."
                    )

        elif group_column == "LakeSetting":
            lines.append(
                "Ecological question: Does the model work equally well for different lake settings?"
            )

            developed = grouped_metrics[
                grouped_metrics["Group"].str.lower().str.contains("developed")
            ]
            if not developed.empty:
                developed_bias = developed.iloc[0]["Bias"]
                if developed_bias < -strong_bias_cutoff:
                    lines.append(
                        "Developed shoreline lakes have a negative bias. The model may be "
                        "underpredicting lakes affected by runoff, docks, lawns, or heavier use."
                    )

        elif group_column == "StormStatus":
            lines.append(
                "Ecological question: Does the model miss bloom risk after disturbance events?"
            )

            after = grouped_metrics[
                grouped_metrics["Group"].str.lower().str.contains("after")
            ]
            if not after.empty:
                after_bias = after.iloc[0]["Bias"]
                if after_bias < -strong_bias_cutoff:
                    lines.append(
                        "After-storm samples have a negative bias. The model may be missing "
                        "blooms that follow runoff events."
                    )

        lines.append(
            "Use this table with the residual plot. A model can look acceptable overall "
            "while failing in one important part of the data."
        )

        return format_interpretation(lines)



    # -----------------------------------------------------------------
    # Clear display
    # -----------------------------------------------------------------

    def clear_display(self):
        self.overall_canvas.figure.clear()
        self.overall_canvas.draw()

        self.group_canvas.figure.clear()
        self.group_canvas.draw()

        self.group_table.clear()
        self.metric_box.clear()
        self.overall_interpretation.clear()
        self.group_interpretation.clear()


# ---------------------------------------------------------------------
# Run app
# ---------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    window = CobberEcoResidueApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

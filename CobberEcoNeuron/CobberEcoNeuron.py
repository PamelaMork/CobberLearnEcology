# CobberEcoNeuron.py
# A PyQt6 application for exploring single neurons and full neural networks.
#
# Ecology version of CobberNeuron.
#
# Teaching story:
#   A simplified blueberry pollination response curve relates air temperature
#   to the number of bee visits counted during a ten-minute observation window.
#
#   The teaching goal is to help students see why one neuron can only make
#   one flexible piece, while a hidden layer can combine several pieces into
#   a more flexible ecological response curve.
#
# Dependencies:
#   pip install PyQt6 numpy matplotlib scikit-learn
#
# Run:
#   python CobberEcoNeuron.py

from __future__ import annotations

import sys
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QTabWidget, QPushButton, QSlider, QFormLayout,
    QSpinBox, QLineEdit, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Teaching dataset
# ---------------------------------------------------------------------
# Simplified blueberry pollination response data.
#
# Input:
#   Air temperature, degrees Celsius
#
# Output:
#   Bee visits during a ten-minute observation window
#
# These values represent the chapter story: bee visits are low when
# temperatures are cool, increase as the field warms, peak under favorable
# warm conditions, and decline when the afternoon becomes too hot.
BLUEBERRY_BEE_VISIT_DATA = {
    12: 2,
    14: 4,
    16: 7,
    18: 10,
    20: 16,
    22: 24,
    24: 31,
    26: 36,
    28: 40,
    30: 42,
    32: 34,
    33: 25,
    34: 22,
    36: 19,
    38: 12,
}


# ---------------------------------------------------------------------
# Core model helpers
# ---------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def calculate_neuron_output(x_input: np.ndarray, weight: float, bias: float) -> np.ndarray:
    return sigmoid(weight * x_input + bias)


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def draw_mid_arrow(ax, x0, y0, x1, y1, frac=0.18, color="grey", lw=1, alpha=0.8, head=12):
    """Draw a line with a small arrow centered on the segment."""
    ax.plot([x0, x1], [y0, y1], color=color, alpha=alpha, linewidth=lw)

    dx, dy = (x1 - x0), (y1 - y0)
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length == 0:
        return

    ux, uy = dx / length, dy / length
    midx, midy = (x0 + x1) / 2, (y0 + y1) / 2
    half = (frac * length) / 2
    xa, ya = midx - ux * half, midy - uy * half
    xb, yb = midx + ux * half, midy + uy * half

    arrow = FancyArrowPatch(
        (xa, ya), (xb, yb),
        arrowstyle="-|>",
        mutation_scale=head,
        linewidth=max(1, lw - 1),
        color=color,
        alpha=alpha,
    )
    ax.add_patch(arrow)


def draw_mid_arrow_on_line(ax, x0, x1, y, frac=0.18, color="grey", lw=1, alpha=0.8, head=12):
    """Draw a short centered arrow on a horizontal line."""
    ax.plot([x0, x1], [y, y], color=color, alpha=alpha, linewidth=lw)

    mid = 0.5 * (x0 + x1)
    half = 0.5 * frac * (x1 - x0)
    xa, xb = mid - half, mid + half

    arrow = FancyArrowPatch(
        (xa, y), (xb, y),
        arrowstyle="-|>",
        mutation_scale=head,
        linewidth=lw,
        color=color,
        alpha=alpha,
    )
    ax.add_patch(arrow)


def y_nudge_pixels(ax, y, pixels):
    """Return a y value nudged by a given number of screen pixels."""
    _, y_disp = ax.transData.transform((0, y))
    y2 = ax.transData.inverted().transform((0, y_disp + pixels))[1]
    return y2


# ---------------------------------------------------------------------
# Matplotlib canvas
# ---------------------------------------------------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


# ---------------------------------------------------------------------
# Tab 1: Single neuron
# ---------------------------------------------------------------------
class SingleNeuronTab(QWidget):
    def __init__(self, air_temps_c: np.ndarray, bee_visits: np.ndarray):
        super().__init__()

        self.air_temps_c = air_temps_c
        self.bee_visits = bee_visits

        self.scaled_temps, self.temp_min, self.temp_max = self.normalize(self.air_temps_c)
        self.scaled_visits, self.visits_min, self.visits_max = self.normalize(self.bee_visits)

        layout = QHBoxLayout(self)

        controls_panel = QFrame()
        controls_panel.setFrameShape(QFrame.Shape.StyledPanel)
        controls_layout = QVBoxLayout(controls_panel)

        schematic_label = QLabel("<h3>The Single Neuron Model</h3>")
        self.schematic_canvas = MplCanvas(self, width=3, height=2, dpi=70)

        schematic_text = QLabel(
            "<p>Adjust the weight and bias. \nWatch where the neuron follows the bee-visit data and where it misses the pattern. \nA single neuron can make one curved response, but the blueberry field may need more than one piece.</p>"
        )
        schematic_text.setWordWrap(True)

        form_layout = QFormLayout()

        self.weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.bias_slider = QSlider(Qt.Orientation.Horizontal)

        self.rmse_label = QLabel(
            '<span style="color:#6C1D45; font-weight:bold;">RMSE Score:</span> N/A bee visits'
        )
        self.rmse_label.setStyleSheet("font-size: 16px;")

        self.weight_slider.setRange(-100, 100)
        self.weight_slider.setValue(10)

        self.bias_slider.setRange(-50, 50)
        self.bias_slider.setValue(0)

        form_layout.addRow("Weight (w):", self.weight_slider)
        form_layout.addRow("Bias (b):", self.bias_slider)

        self.resid_canvas = MplCanvas(self, width=30, height=2.6, dpi=90)

        controls_layout.addWidget(schematic_label)
        controls_layout.addWidget(self.schematic_canvas)
        controls_layout.addWidget(schematic_text)

        controls_layout.addSpacing(8)
        controls_layout.addLayout(form_layout)
        controls_layout.addWidget(self.rmse_label)

        controls_layout.addSpacing(10)
        controls_layout.addWidget(self.resid_canvas)

        controls_layout.addStretch(1)

        plot_panel = QFrame()
        plot_panel.setFrameShape(QFrame.Shape.StyledPanel)
        plot_layout = QVBoxLayout(plot_panel)

        self.canvas = MplCanvas(self, width=7, height=6, dpi=100)
        plot_layout.addWidget(self.canvas)

        layout.addWidget(controls_panel, 1)
        layout.addWidget(plot_panel, 2)

        self.weight_slider.valueChanged.connect(self.update_prediction)
        self.bias_slider.valueChanged.connect(self.update_prediction)

        self.update_prediction()

    def normalize(self, data):
        min_val, max_val = np.min(data), np.max(data)
        if max_val == min_val:
            return data, min_val, max_val
        return (data - min_val) / (max_val - min_val), min_val, max_val

    def denormalize(self, scaled_data, min_val, max_val):
        return scaled_data * (max_val - min_val) + min_val

    def update_prediction(self):
        weight = self.weight_slider.value() / 10.0
        bias = self.bias_slider.value() / 10.0

        self.draw_schematic(weight, bias)

        predicted_scaled = calculate_neuron_output(self.scaled_temps, weight, bias)

        predicted_denormalized = self.denormalize(
            predicted_scaled,
            self.visits_min,
            self.visits_max
        )

        mse = calculate_mse(self.bee_visits, predicted_denormalized)
        rmse = np.sqrt(mse)
        self.rmse_label.setText(
            f'<span style="color:#6C1D45; font-weight:bold;">RMSE Score:</span> {rmse:.2f} bee visits'
        )

        ax = self.canvas.axes
        ax.clear()

        ax.scatter(
            self.air_temps_c,
            self.bee_visits,
            label="Blueberry Field Data"
        )
        ax.plot(
            self.air_temps_c,
            predicted_denormalized,
            "r--",
            label=f"Neuron Prediction (w={weight:.1f}, b={bias:.1f})",
        )

        ax.set_title("Bee Visits vs. Air Temperature")
        ax.set_xlabel("Air Temperature (°C)")
        ax.set_ylabel("Bee Visits in 10 Minutes")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower right")

        # Residuals use predicted minus actual, matching the Error chapter.
        residuals = predicted_denormalized - self.bee_visits

        resid_ax = self.resid_canvas.axes
        resid_ax.clear()

        resid_ax.axhline(0.0, linestyle="--", alpha=0.7)
        resid_ax.vlines(self.air_temps_c, 0.0, residuals, alpha=0.6)
        resid_ax.scatter(self.air_temps_c, residuals, s=18)

        resid_ax.set_xlim(self.air_temps_c.min() - 1, self.air_temps_c.max() + 1)

        resid_ax.set_title("Residuals", fontsize=12)
        resid_ax.set_xlabel("Air Temp (°C)", fontsize=10)
        resid_ax.set_ylabel("Predicted − Actual", fontsize=10)
        resid_ax.tick_params(labelsize=9)
        resid_ax.grid(True, linestyle=":", alpha=0.4)

        self.resid_canvas.draw()

        self.canvas.draw()

    def draw_schematic(self, weight, bias):
        ax = self.schematic_canvas.axes
        ax.clear()

        input_y, hidden_y, output_y = 0.5, 0.5, 0.5

        draw_mid_arrow(ax, 0.94, input_y, 1.02, hidden_y, color="grey", lw=2, alpha=0.8, head=14)
        draw_mid_arrow(ax, -0.03, input_y, 0.07, hidden_y, color="grey", lw=2, alpha=0.8, head=14)
        draw_mid_arrow(ax, 0.1, input_y, 0.5, hidden_y, color="grey", lw=2, alpha=0.8, head=14)
        draw_mid_arrow(ax, 0.5, hidden_y, 0.9, output_y, color="grey", lw=2, alpha=0.8, head=14)

        ax.text(
            0.37,
            0.62,
            f"w={weight:.1f}",
            ha="center",
            va="center",
            fontsize=16,
            bbox=dict(facecolor="white", alpha=0.0, edgecolor="none"),
        )
        ax.text(
            0.53,
            0.25,
            f"b={bias:.1f}",
            ha="center",
            va="center",
            fontsize=16,
            bbox=dict(facecolor="white", alpha=0.0, edgecolor="none"),
        )

        ax.scatter([0.1], [input_y], s=1000, c="#d3d3d3", marker="s", zorder=5)
        ax.text(-0.08, output_y, "Air\nTemp", ha="center", va="center", fontsize=14)
        ax.text(0.1, input_y - 0.4, "Input\nNeuron", ha="center", va="center", fontsize=16)

        ax.scatter([0.5], [hidden_y], s=1000, c="#6C1D45", marker="s", zorder=5)

        ax.scatter([0.9], [output_y], s=1000, c="#3D3D3D", marker="s", zorder=5)
        ax.text(0.9, output_y - 0.4, "Output\nNeuron", ha="center", va="center", fontsize=16)
        ax.text(1.07, output_y, "Bee\nvisits", ha="center", va="center", fontsize=14)

        ax.set_xlim(-0.12, 1.12)
        ax.set_ylim(0, 1)
        ax.axis("off")
        self.schematic_canvas.draw()


# ---------------------------------------------------------------------
# Tab 2: Neural network
# ---------------------------------------------------------------------
class NeuralNetworkTab(QWidget):
    def __init__(self, air_temps_c: np.ndarray, bee_visits: np.ndarray, main_window):
        super().__init__()

        self.main_window = main_window
        self.air_temps_c = air_temps_c
        self.bee_visits = bee_visits

        self.model = None
        self.scaler_X = None
        self.scaler_y = None

        layout = QHBoxLayout(self)

        controls_panel = QFrame()
        controls_panel.setFrameShape(QFrame.Shape.StyledPanel)
        controls_layout = QVBoxLayout(controls_panel)

        form_layout = QFormLayout()

        self.neurons_spinner = QSpinBox()
        self.neurons_spinner.setRange(2, 16)
        self.neurons_spinner.setValue(8)

        self.learning_rate_input = QLineEdit("0.005")
        self.epochs_input = QLineEdit("1000")
        self.train_button = QPushButton("Train Network")

        self.train_button.setStyleSheet("""
            QPushButton {
                background-color: #6C1D45;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 8px 14px;
                border-radius: 4px;
            }

            QPushButton:hover {
                background-color: #7A2854;
            }

            QPushButton:pressed {
                background-color: #551737;
            }

            QPushButton:disabled {
                background-color: #9A8F96;
                color: white;
                font-weight: bold;
            }
        """)

        neurons_label = QLabel("Neurons in Hidden Layer:")
        neurons_label.setStyleSheet("font-weight: bold;")

        learning_rate_label = QLabel("Learning Rate:")
        learning_rate_label.setStyleSheet("font-weight: bold;")

        epochs_label = QLabel("Training Cycles (Epochs):")
        epochs_label.setStyleSheet("font-weight: bold;")

        form_layout.addRow(neurons_label, self.neurons_spinner)
        form_layout.addRow(learning_rate_label, self.learning_rate_input)
        form_layout.addRow(epochs_label, self.epochs_input)

        controls_layout.addWidget(QLabel("<h3>Network Configuration</h3>"))
        controls_layout.addLayout(form_layout)
        controls_layout.addWidget(self.train_button)
        controls_layout.addStretch()

        schematic_panel = QFrame()
        schematic_panel.setFrameShape(QFrame.Shape.StyledPanel)
        schematic_layout = QVBoxLayout(schematic_panel)

        schematic_layout.addWidget(QLabel("<h3>Network Architecture</h3>"))
        self.schematic_canvas = MplCanvas(self, width=7, height=7)
        schematic_layout.addWidget(self.schematic_canvas)

        results_panel = QTabWidget()

        self.loss_canvas = MplCanvas(self)
        results_panel.addTab(self.loss_canvas, "Training Progress")

        self.fit_canvas = MplCanvas(self)
        results_panel.addTab(self.fit_canvas, "Final Model Fit")

        self.resid_tab = QWidget()
        resid_layout = QVBoxLayout(self.resid_tab)
        self.resid_canvas = MplCanvas(self)

        self.rmse_label = QLabel(
            '<span style="color:#6C1D45; font-weight:bold;">RMSE:</span> '
            '<span style="color:#3D3D3D;">-</span>'
        )
        self.rmse_label.setStyleSheet("font-size: 18px;")

        labels_layout = QHBoxLayout()
        labels_layout.addStretch(1)
        labels_layout.addWidget(self.rmse_label)
        labels_layout.addStretch(1)
        resid_layout.setSpacing(6)
        resid_layout.setContentsMargins(6, 6, 6, 6)
        resid_layout.addWidget(self.resid_canvas)
        resid_layout.addLayout(labels_layout)

        results_panel.addTab(self.resid_tab, "Residuals")

        layout.addWidget(controls_panel, 3)
        layout.addWidget(schematic_panel, 7)
        layout.addWidget(results_panel, 7)

        self.train_button.clicked.connect(self.train_network)
        self.neurons_spinner.valueChanged.connect(self.update_schematic)

        self.setup_initial_plots()
        self.update_schematic()

    def train_network(self):
        try:
            hidden_neurons = self.neurons_spinner.value()
            learning_rate = float(self.learning_rate_input.text())
            epochs = int(self.epochs_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please ensure all parameters are valid numbers.")
            return

        X = self.air_temps_c.reshape(-1, 1)
        y = self.bee_visits

        self.train_button.setText("Training...")
        self.train_button.setEnabled(False)
        QApplication.processEvents()

        scaler_X = StandardScaler().fit(X)
        scaler_y = StandardScaler().fit(y.reshape(-1, 1))

        X_scaled = scaler_X.transform(X)
        y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()

        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden_neurons,),
            activation="relu",
            solver="adam",
            learning_rate_init=learning_rate,
            max_iter=1,
            warm_start=True,
            random_state=42
        )

        self.loss_canvas.axes.clear()
        self.loss_canvas.axes.set_title("Training Loss vs. Epoch")
        self.loss_canvas.axes.set_xlabel("Epoch")
        self.loss_canvas.axes.set_ylabel("Scaled Training Loss")
        self.loss_canvas.axes.grid(True, alpha=0.3)
        self.loss_canvas.draw()

        losses = []
        update_every = 25

        for epoch in range(epochs):
            self.model.fit(X_scaled, y_scaled)

            if hasattr(self.model, "loss_"):
                losses.append(self.model.loss_)

            if epoch % update_every == 0 or epoch == epochs - 1:
                self.loss_canvas.axes.clear()
                self.loss_canvas.axes.plot(range(1, len(losses) + 1), losses)
                self.loss_canvas.axes.set_title("Training Loss vs. Epoch")
                self.loss_canvas.axes.set_xlabel("Epoch")
                self.loss_canvas.axes.set_ylabel("Scaled Training Loss")
                self.loss_canvas.axes.grid(True, alpha=0.3)
                self.loss_canvas.draw()

                self.train_button.setText(f"Training... epoch {epoch + 1} of {epochs}")
                QApplication.processEvents()

        self.train_button.setText("Train Network")
        self.train_button.setEnabled(True)

        self.update_schematic()

        x_smooth = np.linspace(self.air_temps_c.min(), self.air_temps_c.max(), 240).reshape(-1, 1)
        x_smooth_scaled = scaler_X.transform(x_smooth)
        y_pred_scaled = self.model.predict(x_smooth_scaled)
        y_pred_denormalized = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        self.fit_canvas.axes.clear()
        self.fit_canvas.axes.scatter(
            self.air_temps_c,
            self.bee_visits,
            label="Blueberry Field Data"
        )
        self.fit_canvas.axes.plot(
            x_smooth.ravel(),
            y_pred_denormalized,
            "r-",
            label="Neural Network Fit",
            linewidth=2,
        )
        self.fit_canvas.axes.set_title("Blueberry Pollination Response & Neural Network Fit")
        self.fit_canvas.axes.set_xlabel("Air Temperature (°C)")
        self.fit_canvas.axes.set_ylabel("Bee Visits in 10 Minutes")
        self.fit_canvas.axes.legend()
        self.fit_canvas.axes.grid(True)
        self.fit_canvas.draw()

        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

        y_train_pred_scaled = self.model.predict(X_scaled)
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()

        # Residuals use predicted minus actual, matching the Error chapter.
        residuals = y_train_pred - self.bee_visits
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        axr = self.resid_canvas.axes
        axr.clear()

        axr.axhline(0.0, linestyle="--", alpha=0.7)
        axr.vlines(self.air_temps_c, 0.0, residuals, alpha=0.6)
        axr.scatter(self.air_temps_c, residuals, s=25)

        axr.set_title("Residuals Across Temperature")
        axr.set_xlabel("Air Temperature (°C)")
        axr.set_ylabel("Predicted − Actual")
        axr.grid(True, linestyle=":", alpha=0.5)
        self.resid_canvas.draw()

        self.rmse_label.setText(
            f'<span style="color:#6C1D45; font-weight:bold;">RMSE:</span> '
            f'<span style="color:#3D3D3D;">{rmse:.2f} bee visits</span>'
        )
    def update_schematic(self):
        ax = self.schematic_canvas.axes
        ax.clear()

        num_neurons = self.neurons_spinner.value()

        input_y = [0.5]
        hidden_y = np.linspace(0.1, 0.9, num_neurons)
        output_y = [0.5]

        for hy in hidden_y:
            draw_mid_arrow(ax, 0.1, input_y[0], 0.5, hy, color="grey", lw=2, alpha=0.8, head=14)
            draw_mid_arrow(ax, 0.5, hy, 0.9, output_y[0], color="grey", lw=2, alpha=0.8, head=14)

        yin = y_nudge_pixels(ax, input_y[0], +1)
        yout = y_nudge_pixels(ax, output_y[0], +1)

        draw_mid_arrow_on_line(ax, -0.02, 0.08, yin, color="grey", lw=2, alpha=0.8, head=14)
        draw_mid_arrow_on_line(ax, 0.95, 1.02, yout, color="grey", lw=2, alpha=0.8, head=14)

        ax.scatter([0.1], input_y, s=1000, c="#d3d3d3", marker="s", zorder=5)
        ax.text(0.075, input_y[0] - 0.09, "Input\nNeuron", ha="center", va="center", fontsize=13)

        ax.scatter([0.5] * num_neurons, hidden_y, s=400, c="#6C1D45", marker="s", zorder=5)
        ax.text(0.5, input_y[0] + 0.475, "Hidden Layer", ha="center", va="center", fontsize=12)

        ax.scatter([0.9], output_y, s=1000, c="#3D3D3D", marker="s", zorder=5)
        ax.text(0.925, output_y[0] - 0.09, "Output\nNeuron", ha="center", va="center", fontsize=13)

        ax.text(-0.08, output_y[0], "Air\nTemp", ha="center", va="center", fontsize=14)
        ax.text(1.075, output_y[0], "Bee\nvisits", ha="center", va="center", fontsize=14)

        ax.set_xlim(-0.12, 1.12)
        ax.set_ylim(0, 1)
        ax.axis("off")
        self.schematic_canvas.draw()
    def setup_initial_plots(self):
        self.loss_canvas.axes.set_title("Training Loss vs. Epoch")
        self.loss_canvas.axes.set_xlabel("Epoch")
        self.loss_canvas.axes.set_ylabel("Scaled Training Loss")
        self.loss_canvas.axes.grid(True)
        self.loss_canvas.draw()

        self.fit_canvas.axes.clear()
        self.fit_canvas.axes.scatter(
            self.air_temps_c,
            self.bee_visits,
            label="Blueberry Field Data"
        )
        self.fit_canvas.axes.set_title("Blueberry Pollination Response & Model Fit")
        self.fit_canvas.axes.set_xlabel("Air Temperature (°C)")
        self.fit_canvas.axes.set_ylabel("Bee Visits in 10 Minutes")
        self.fit_canvas.axes.legend()
        self.fit_canvas.axes.grid(True)
        self.fit_canvas.draw()

        self.resid_canvas.axes.clear()
        self.resid_canvas.axes.axhline(0.0, linestyle="--", alpha=0.7)
        self.resid_canvas.axes.set_title("Residuals (Predicted − Actual)")
        self.resid_canvas.axes.set_xlabel("Air Temperature (°C)")
        self.resid_canvas.axes.set_ylabel("Residual (bee visits)")
        self.resid_canvas.axes.text(
            0.5,
            0.5,
            "Train the network to see residuals.",
            ha="center",
            va="center",
            transform=self.resid_canvas.axes.transAxes,
            alpha=0.7,
        )
        self.resid_canvas.draw()


# ---------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------
class CobberEcoNeuronApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")

        self.setWindowTitle("CobberEcoNeuron")
        self.setGeometry(100, 100, 1400, 700)
        self.setFont(self.lato_font)

        self.air_temps_c = np.array(list(BLUEBERRY_BEE_VISIT_DATA.keys()), dtype=float)
        self.bee_visits = np.array(list(BLUEBERRY_BEE_VISIT_DATA.values()), dtype=float)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #d0d0d0;
                background: white;
            }

            QTabBar::tab {
                background: #4A4A4A;
                color: white;
                font-weight: bold;
                padding: 8px 18px;
                border: 1px solid #d0d0d0;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }

            QTabBar::tab:selected {
                background: #6C1D45;
                color: white;
                font-weight: bold;
            }

            QTabBar::tab:selected:hover {
                background: #6C1D45;
                color: white;
                font-weight: bold;
            }

            QTabBar::tab:!selected {
                background: #4A4A4A;
                color: white;
                font-weight: bold;
            }

            QTabBar::tab:!selected:hover {
                background: #5A5A5A;
                color: white;
                font-weight: bold;
            }
        """)

        tabs.addTab(SingleNeuronTab(self.air_temps_c, self.bee_visits), "The Single Neuron")
        tabs.addTab(NeuralNetworkTab(self.air_temps_c, self.bee_visits, self), "The Neural Network")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CobberEcoNeuronApp()
    window.show()
    sys.exit(app.exec())
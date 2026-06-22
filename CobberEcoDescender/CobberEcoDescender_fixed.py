# CobberEcoDescender.py
# An application for exploring gradient descent with red squirrel territory data.
# Refactored for the CobberLearnEco launcher.

import sys
import numpy as np
from typing import List, Tuple

# --- Matplotlib and PyQt6 Integration ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QPushButton, QFormLayout, QMessageBox, QLineEdit, QTextEdit,
    QGroupBox
)
from PyQt6.QtGui import QFont, QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch

# --- Core Engine ---

Dataset = List[Tuple[float, float]]


def generate_linear_data(
    num_points: int = 50,
    m_range=(0.35, 0.65),
    b_range=(1.2, 2.5),
    noise_level: float = 1.0
) -> Tuple[Dataset, float, float]:
    true_slope = round(np.random.uniform(*m_range), 2)
    true_intercept = round(np.random.uniform(*b_range), 2)
    x_data = np.linspace(0, 10, num_points)
    y_data = true_slope * x_data + true_intercept
    noise = np.random.normal(0, noise_level, num_points)
    y_data_noisy = y_data + noise
    y_data_noisy = np.clip(y_data_noisy, 0, 10)
    return list(zip(x_data, y_data_noisy)), true_slope, true_intercept


def calculate_mse(data: Dataset, m: float, b: float) -> float:
    n = len(data)
    if n == 0:
        return 0.0
    return sum((y_i - (m * x_i + b)) ** 2 for x_i, y_i in data) / n


def calculate_gradient(data: Dataset, m: float, b: float) -> Tuple[float, float]:
    n = len(data)
    if n == 0:
        return 0.0, 0.0

    dm, db = 0.0, 0.0
    for x_i, y_i in data:
        error_term = y_i - (m * x_i + b)
        dm += -2 * x_i * error_term
        db += -2 * error_term

    return dm / n, db / n


# --- GUI Development ---

class MplCanvas(FigureCanvas):
    """A custom matplotlib canvas."""

    def __init__(self, parent=None, width=5.5, height=5.5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)

    def resizeEvent(self, event):
        super(MplCanvas, self).resizeEvent(event)
        try:
            self.fig.tight_layout()
        except ValueError:
            pass


# --- Main Application Class ---

class CobberEcoDescenderApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- BRANDING ---
        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")

        self.setWindowTitle("CobberEcoDescender")
        self.setGeometry(75, 75, 1250, 700)
        self.setFont(self.lato_font)

        self.dataset = []
        self.true_m, self.true_b = 0, 0
        self.path_points = []
        self.current_pos = None
        self.mse_grid = None
        self.best_fit_pos = None
        self.landscape_revealed = False

        main_layout = QHBoxLayout()

        controls_panel = QFrame()
        controls_panel.setFrameShape(QFrame.Shape.StyledPanel)
        controls_layout = QVBoxLayout(controls_panel)

        self.generate_data_button = QPushButton("Generate Squirrel Data")

        self.random_start_button = QPushButton("Choose Random Starting Model")
        self.random_start_button.setEnabled(False)

        self.locate_line_button = QPushButton("Locate Line in Parameter Space")
        self.locate_line_button.setEnabled(False)

        self.show_contours_button = QPushButton("Choose Starting Model First")
        self.show_contours_button.setEnabled(False)

        self.style_button(self.generate_data_button)
        self.style_button(self.random_start_button)
        self.style_button(self.locate_line_button)
        self.style_button(self.show_contours_button)

        self.step_size_input = QLineEdit("0.5")
        self.noise_input = QLineEdit("1.0")

        self.gradient_label = QLabel("Direction check: N/A")
        self.gradient_label.setStyleSheet("font-size: 18px; font-weight: bold;")

        # --- Panel 1: Data Generation ---
        data_panel, data_layout = self.make_panel("Data Generation")
        data_form = QFormLayout()
        data_form.addRow("Ecological Scatter:", self.noise_input)
        data_layout.addLayout(data_form)
        data_layout.addWidget(self.generate_data_button)
        controls_layout.addWidget(data_panel)

        # --- Panel 2: Plan Your Steps ---
        plan_panel, plan_layout = self.make_panel("Plan Your Steps")
        plan_form = QFormLayout()
        plan_form.addRow("Step Size:", self.step_size_input)
        plan_layout.addLayout(plan_form)
        plan_layout.addWidget(self.random_start_button)
        plan_layout.addWidget(self.locate_line_button)
        controls_layout.addWidget(plan_panel)

        # --- Panel 3: Take Your Steps ---
        walk_panel, walk_layout = self.make_panel("Take Your Steps")
        walk_layout.addWidget(self.gradient_label)

        self.status_box = QTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMinimumHeight(130)
        self.status_box.setText("Generate red squirrel and cone data to begin.")
        walk_layout.addWidget(self.status_box)

        controls_layout.addWidget(walk_panel)

        # --- Reveal Button ---
        controls_layout.addStretch()
        controls_layout.addWidget(self.show_contours_button)

        # --- Plot Panels ---
        scatter_panel = QFrame()
        scatter_panel.setFrameShape(QFrame.Shape.StyledPanel)
        scatter_layout = QVBoxLayout(scatter_panel)
        self.scatter_canvas = MplCanvas(self)
        scatter_layout.addWidget(self.scatter_canvas)

        contour_panel = QFrame()
        contour_panel.setFrameShape(QFrame.Shape.StyledPanel)
        contour_layout = QVBoxLayout(contour_panel)
        self.contour_canvas = MplCanvas(self)
        contour_layout.addWidget(self.contour_canvas)

        main_layout.addWidget(controls_panel, 2)
        main_layout.addWidget(scatter_panel, 6)
        main_layout.addWidget(contour_panel, 6)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.generate_data_button.clicked.connect(self.run_data_generation)
        self.random_start_button.clicked.connect(self.choose_random_starting_model)
        self.locate_line_button.clicked.connect(self.locate_line_in_parameter_space)
        self.show_contours_button.clicked.connect(self.run_show_contours)

        self.contour_canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.contour_canvas.mpl_connect('motion_notify_event', self.on_canvas_motion)
        self.contour_canvas.mpl_connect('axes_leave_event', self.on_canvas_leave)

        self.reset_plots()

    def style_button(self, button):
        button.setMinimumHeight(32)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.cobber_maroon.name()};
                color: white;
                font-weight: bold;
                padding: 6px;
                border-radius: 5px;
            }}
            QPushButton:disabled {{
                background-color: #555555;
                color: white;
                font-weight: bold;
            }}
        """)

    def make_panel(self, title):
        panel = QGroupBox(title)
        panel.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 6px;
                margin-top: 10px;
                padding: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        layout = QVBoxLayout(panel)
        return panel, layout

    def run_data_generation(self):
        try:
            noise = float(self.noise_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Ecological scatter must be a number.")
            return

        self.dataset, self.true_m, self.true_b = generate_linear_data(noise_level=noise)
        self.mse_grid = None
        self.path_points = []
        self.current_pos = None
        self.best_fit_pos = None
        self.landscape_revealed = False

        self.gradient_label.setText("Direction check: N/A")
        self.gradient_label.setStyleSheet("font-size: 16px; font-weight: bold; color: black;")

        self.status_box.setText(
            "Red squirrel territory data generated.\n\n"
            "Now choose a random starting survival prediction line."
        )

        self.random_start_button.setEnabled(True)
        self.locate_line_button.setEnabled(False)

        self.show_contours_button.setText("Choose Starting Model First")
        self.show_contours_button.setEnabled(False)

        self.reset_plots()
        self.draw_scatter_plot()

    def choose_random_starting_model(self):
        if not self.dataset:
            self.status_box.setText(
                "Generate red squirrel territory data before choosing a starting model."
            )
            return

        random_m = float(np.random.uniform(0.05, 0.9))
        random_b = float(np.random.uniform(0.5, 5.0))

        self.current_pos = (random_m, random_b)
        self.path_points = []
        self.landscape_revealed = False

        current_mse = calculate_mse(self.dataset, random_m, random_b)

        self.status_box.setText(
            f"Random starting model selected:\n"
            f"m = {random_m:.2f}\n"
            f"b = {random_b:.2f}\n"
            f"MSE = {current_mse:.2f}\n\n"
            f"Now locate this line in parameter space."
        )

        self.gradient_label.setText("Direction check: N/A")
        self.gradient_label.setStyleSheet("font-size: 16px; font-weight: bold; color: black;")

        self.locate_line_button.setEnabled(True)

        self.show_contours_button.setText("Locate Line First")
        self.show_contours_button.setEnabled(False)

        self.draw_contour_plot()
        self.draw_scatter_plot()

    def locate_line_in_parameter_space(self):
        if self.current_pos is None:
            self.status_box.setText(
                "Choose a random starting model before locating it in parameter space."
            )
            return

        self.path_points = [self.current_pos]
        self.landscape_revealed = False

        current_mse = calculate_mse(self.dataset, self.current_pos[0], self.current_pos[1])

        self.status_box.setHtml(
            f"<b>Current model:</b><br>"
            f"m = {self.current_pos[0]:.2f}<br>"
            f"b = {self.current_pos[1]:.2f}<br>"
            f"MSE = {current_mse:.2f}<br><br>"
            f"The red point marks this prediction line in parameter space. "
            f"Now hover to test directions and click downhill to improve the model."
        )

        self.locate_line_button.setEnabled(False)

        self.show_contours_button.setText("Reveal MSE Landscape")
        self.show_contours_button.setEnabled(True)

        self.draw_contour_plot()
        self.draw_scatter_plot()

    def on_canvas_click(self, event):
        if not self.dataset:
            QMessageBox.warning(self, "No Data", "Please generate red squirrel data first.")
            return
        if event.inaxes != self.contour_canvas.axes:
            return
        if self.current_pos is not None and not self.path_points:
            return

        click_pos = (event.xdata, event.ydata)

        if self.current_pos is None:
            self.current_pos = click_pos
        else:
            try:
                step_size = float(self.step_size_input.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Step size must be a number.")
                return

            dx = click_pos[0] - self.current_pos[0]
            dy = click_pos[1] - self.current_pos[1]
            norm = np.sqrt(dx ** 2 + dy ** 2)

            if norm == 0:
                return

            unit_vector = (dx / norm, dy / norm)
            new_m = self.current_pos[0] + step_size * unit_vector[0]
            new_b = self.current_pos[1] + step_size * unit_vector[1]

            # Keep the model inside the visible, biologically plausible parameter space.
            new_m = float(np.clip(new_m, 0, 1))
            new_b = float(np.clip(new_b, 0, 8))

            self.current_pos = (new_m, new_b)

        self.path_points.append(self.current_pos)
        self.show_contours_button.setEnabled(True)

        current_mse = calculate_mse(self.dataset, self.current_pos[0], self.current_pos[1])

        if len(self.path_points) == 1:
            status_text = (
                f"<b>Current model:</b><br>"
                f"m = {self.current_pos[0]:.2f}<br>"
                f"b = {self.current_pos[1]:.2f}<br>"
                f"MSE = {current_mse:.2f}<br><br>"
                f"Now hover in different directions. Green means that direction lowers MSE."
            )
        else:
            previous_pos = self.path_points[-2]
            previous_mse = calculate_mse(self.dataset, previous_pos[0], previous_pos[1])
            mse_change = current_mse - previous_mse

            if mse_change < 0:
                result_line = f"MSE decreased by {abs(mse_change):.2f}."
                meaning_line = "The survival line fits the squirrel data better."
            elif mse_change > 0:
                result_line = f"MSE increased by {mse_change:.2f}."
                meaning_line = "The survival line fits the squirrel data worse."
            else:
                result_line = "MSE did not change."
                meaning_line = "The survival line stayed about the same."

            status_text = (
                f"<b>Current model:</b><br>"
                f"m = {self.current_pos[0]:.2f}<br>"
                f"b = {self.current_pos[1]:.2f}<br>"
                f"MSE = {current_mse:.2f}<br><br>"
                f"<b>Last move:</b><br>"
                f"{result_line}<br>"
                f"{meaning_line}"
            )

        self.status_box.setHtml(status_text)

        self.draw_contour_plot()
        self.draw_scatter_plot(show_best_fit=self.landscape_revealed)

    def on_canvas_motion(self, event):
        if self.current_pos is None or not self.path_points or event.inaxes != self.contour_canvas.axes:
            return

        mouse_pos = (event.xdata, event.ydata)
        dx = mouse_pos[0] - self.current_pos[0]
        dy = mouse_pos[1] - self.current_pos[1]
        norm = np.sqrt(dx ** 2 + dy ** 2)

        if norm == 0:
            return

        unit_vector = (dx / norm, dy / norm)
        grad_m, grad_b = calculate_gradient(self.dataset, self.current_pos[0], self.current_pos[1])
        directional_derivative = grad_m * unit_vector[0] + grad_b * unit_vector[1]

        self.gradient_label.setText(f"Direction check: {directional_derivative:.2f}")
        color = "green" if directional_derivative < 0 else "red"
        self.gradient_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color};")

        arrow_end_pos = (
            self.current_pos[0] + 0.5 * unit_vector[0],
            self.current_pos[1] + 0.5 * unit_vector[1]
        )

        self.draw_contour_plot(arrow_end=arrow_end_pos)

    def on_canvas_leave(self, event):
        if self.current_pos:
            self.draw_contour_plot()

    def run_show_contours(self):
        if not self.dataset:
            return

        if not self.landscape_revealed:
            if self.mse_grid is None:
                m_vals = np.linspace(0, 1, 50)
                b_vals = np.linspace(0, 8, 50)
                self.mse_grid = np.array([[calculate_mse(self.dataset, m, b) for m in m_vals] for b in b_vals])
                min_b_idx, min_m_idx = np.unravel_index(np.argmin(self.mse_grid), self.mse_grid.shape)
                self.best_fit_pos = (m_vals[min_m_idx], b_vals[min_b_idx])

            self.landscape_revealed = True
            self.show_contours_button.setText("Hide MSE Landscape")
        else:
            self.landscape_revealed = False
            self.show_contours_button.setText("Reveal MSE Landscape")

        self.draw_contour_plot()
        self.draw_scatter_plot(show_best_fit=self.landscape_revealed)

    def reset_plots(self):
        for canvas in [self.contour_canvas, self.scatter_canvas]:
            canvas.axes.clear()

        self.contour_canvas.axes.set_xlim(0, 1)
        self.contour_canvas.axes.set_ylim(0, 8)
        self.contour_canvas.axes.set_xlabel("Slope of survival line (m)")
        self.contour_canvas.axes.set_ylabel("Intercept of survival line (b)")
        self.contour_canvas.axes.set_title("Model Parameter Space")
        self.contour_canvas.axes.grid(True, linestyle='--', alpha=0.5)

        self.scatter_canvas.axes.set_xlim(-1, 11)
        self.scatter_canvas.axes.set_ylim(0, 10)
        self.scatter_canvas.axes.set_xlabel("Cone availability index")
        self.scatter_canvas.axes.set_ylabel("Juvenile survival index")
        self.scatter_canvas.axes.set_title("Red Squirrel Territory Data")
        self.scatter_canvas.axes.grid(True, linestyle='--', alpha=0.5)

        self.contour_canvas.draw()
        self.scatter_canvas.draw()

    def draw_scatter_plot(self, show_best_fit=False):
        ax = self.scatter_canvas.axes
        ax.clear()

        if not self.dataset:
            self.reset_plots()
            return

        x_data, y_data = zip(*self.dataset)

        ax.scatter(x_data, y_data, label='Red squirrel territories', alpha=0.7, zorder=2)

        if self.current_pos:
            m, b = self.current_pos
            x_line = np.array([-1, 11])
            y_line = m * x_line + b
            ax.plot(
                x_line,
                y_line,
                color='red',
                linestyle='--',
                label=f'Current survival line (m={m:.2f}, b={b:.2f})',
                zorder=3
            )

        if show_best_fit:
            x_line = np.array([-1, 11])
            y_line = self.true_m * x_line + self.true_b
            ax.plot(
                x_line,
                y_line,
                color='green',
                linewidth=3,
                label=f'Hidden cone-survival line (m={self.true_m:.2f}, b={self.true_b:.2f})',
                zorder=4
            )

        ax.set_xlim(-1, 11)
        ax.set_ylim(0, 10)
        ax.set_xlabel("Cone availability index")
        ax.set_ylabel("Juvenile survival index")
        ax.set_title("Prediction Line for Squirrel Survival")
        ax.grid(True)
        ax.legend(loc='upper left')

        self.scatter_canvas.draw()

    def draw_contour_plot(self, show_contours=False, show_minimum=False, show_best_fit=False, arrow_end=None):
        if self.landscape_revealed:
            show_contours = True
            show_minimum = True
            show_best_fit = True

        ax = self.contour_canvas.axes
        ax.clear()

        if show_contours and self.mse_grid is not None:
            m_vals = np.linspace(0, 1, 50)
            b_vals = np.linspace(0, 8, 50)
            levels = np.geomspace(self.mse_grid.min() + 0.01, self.mse_grid.max(), 20)
            ax.contourf(m_vals, b_vals, self.mse_grid, levels=levels, cmap='viridis_r', zorder=1)

        if self.path_points:
            path_m, path_b = zip(*self.path_points)
            ax.plot(path_m, path_b, 'r-o', label='Downhill model path', zorder=5)

        if arrow_end and self.current_pos:
            ax.add_patch(
                FancyArrowPatch(
                    self.current_pos,
                    arrow_end,
                    color='darkorange',
                    mutation_scale=20,
                    arrowstyle='->',
                    zorder=10
                )
            )

        if show_minimum:
            ax.plot(
                self.true_m,
                self.true_b,
                'g*',
                markersize=20,
                label='Hidden cone-survival line',
                zorder=10,
                markeredgecolor='black'
            )

        if show_best_fit and self.best_fit_pos:
            ax.plot(
                self.best_fit_pos[0],
                self.best_fit_pos[1],
                'bX',
                markersize=15,
                label='Lowest MSE found',
                zorder=9,
                markeredgecolor='white'
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 8)
        ax.set_xlabel("Slope of survival line (m)")
        ax.set_ylabel("Intercept of survival line (b)")

        if show_contours:
            ax.set_title("MSE Loss Landscape")
        else:
            ax.set_title("Model Parameter Space")

        ax.grid(True)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper left')

        self.contour_canvas.draw()


# --- Standalone Execution Guard ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CobberEcoDescenderApp()
    window.show()
    sys.exit(app.exec())

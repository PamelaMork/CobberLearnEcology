#!/usr/bin/env python3
"""
CobberEcoLSTMguided.py

Guided rebuild of CobberEcoLSTMbasic.

Current rebuild stage:
    Tab 1: Build Windows
    Tab 2: Watch Memory
    Tab 3: Expose Memory
    Tab 4: Check Prediction

This version keeps Darin's calculation ideas but wraps them in a
student-facing learning path for the ML for Ecology LSTM chapter.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Dict, List

from PyQt6.QtCore import Qt, QRectF, QTimer
from PyQt6.QtGui import QFont, QPainter, QPen, QColor, QBrush
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
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


APP_TITLE = "CobberEcoLSTMguided"

COBBER_MAROON = "#6c1d45"
INFO_BLUE = "#2f6690"
CHARCOAL = "#222222"
LIGHT_GRAY = "#f4f4f4"


# ============================================================
# Calculator core
# ============================================================

PRESETS: Dict[str, List[int]] = {
    "Chapter sequence": [1, 1, 0, -1, -1, 0, 1, 1, 0, -1],
    "Alternating trend": [1, -1, 1, -1, 1, -1, 0, 1, -1, 0],
    "Stable then rising": [0, 0, 0, 1, 1, 1, 0, -1, 0, 1],
    "Long decrease": [1, 0, -1, -1, -1, 0, 0, 1, 0, -1],
}


@dataclass
class WindowExample:
    window_number: int
    start_index: int
    inputs: List[int]
    target: int


@dataclass
class MemoryStep:
    step: int
    x: float
    c_prev: float
    kept_old_memory: float
    written_new_input: float
    c_new: float


@dataclass
class ExposureStep:
    step: int
    x: float
    c_value: float
    tanh_c: float
    o_value: float
    h_value: float


@dataclass
class PredictionRow:
    window_number: int
    start_index: int
    inputs: List[int]
    target: int
    final_h: float
    yhat: float
    error: float
    loss: float


class TinyMemoryCell:
    @staticmethod
    def build_windows(sequence: List[int], window_len: int) -> List[WindowExample]:
        windows: List[WindowExample] = []

        for start in range(0, len(sequence) - window_len):
            inputs = sequence[start : start + window_len]
            target = sequence[start + window_len]
            windows.append(
                WindowExample(
                    window_number=len(windows) + 1,
                    start_index=start,
                    inputs=inputs,
                    target=target,
                )
            )

        return windows

    @staticmethod
    def memory_steps(inputs: List[int], f_value: float, i_value: float) -> List[MemoryStep]:
        steps: List[MemoryStep] = []
        c_prev = 0.0

        for step_number, x_value in enumerate(inputs, start=1):
            kept_old_memory = f_value * c_prev
            written_new_input = i_value * float(x_value)
            c_new = kept_old_memory + written_new_input

            steps.append(
                MemoryStep(
                    step=step_number,
                    x=float(x_value),
                    c_prev=c_prev,
                    kept_old_memory=kept_old_memory,
                    written_new_input=written_new_input,
                    c_new=c_new,
                )
            )

            c_prev = c_new

        return steps

    @staticmethod
    def exposure_steps(inputs: List[int], f_value: float, i_value: float, o_value: float) -> List[ExposureStep]:
        exposure_steps: List[ExposureStep] = []
        memory_steps = TinyMemoryCell.memory_steps(inputs, f_value, i_value)

        for memory_step in memory_steps:
            tanh_c = math.tanh(memory_step.c_new)
            h_value = o_value * tanh_c
            exposure_steps.append(
                ExposureStep(
                    step=memory_step.step,
                    x=memory_step.x,
                    c_value=memory_step.c_new,
                    tanh_c=tanh_c,
                    o_value=o_value,
                    h_value=h_value,
                )
            )

        return exposure_steps

    @staticmethod
    def prediction_rows(
        sequence: List[int],
        window_len: int,
        f_value: float,
        i_value: float,
        o_value: float,
        w_value: float,
        b_value: float,
    ) -> List[PredictionRow]:
        rows: List[PredictionRow] = []

        for window in TinyMemoryCell.build_windows(sequence, window_len):
            exposure_steps = TinyMemoryCell.exposure_steps(
                window.inputs,
                f_value,
                i_value,
                o_value,
            )
            final_h = exposure_steps[-1].h_value if exposure_steps else 0.0
            yhat = b_value + w_value * final_h
            error = yhat - float(window.target)
            loss = error * error

            rows.append(
                PredictionRow(
                    window_number=window.window_number,
                    start_index=window.start_index,
                    inputs=window.inputs,
                    target=window.target,
                    final_h=final_h,
                    yhat=yhat,
                    error=error,
                    loss=loss,
                )
            )

        return rows


# ============================================================
# Small helper functions
# ============================================================

def trend_text(value: int | float) -> str:
    value_int = int(value)
    if value_int > 0:
        return "+1"
    if value_int < 0:
        return "-1"
    return "0"


def format_window_values(values: List[int]) -> str:
    return ", ".join(trend_text(v) for v in values)


def fmt_decimal(value: float) -> str:
    return f"{value:.3f}"


def make_table_item(text: object) -> QTableWidgetItem:
    item = QTableWidgetItem(str(text))
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    return item


def make_maroon_bold_table_item(text: object) -> QTableWidgetItem:
    item = make_table_item(text)
    font = item.font()
    font.setBold(True)
    item.setFont(font)
    item.setForeground(QColor(COBBER_MAROON))
    return item


def make_input_cell(text: str, is_current: bool) -> QLabel:
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setMargin(4)

    if is_current:
        label.setStyleSheet(
            """
            QLabel {
                color: #222222;
                background-color: #ffffff;
                border: 2px solid #6c1d45;
                border-radius: 5px;
                font-weight: bold;
                padding: 3px;
            }
            """
        )
    else:
        label.setStyleSheet(
            """
            QLabel {
                color: #222222;
                background-color: #ffffff;
                border: 1px solid transparent;
                padding: 3px;
            }
            """
        )

    return label


def make_target_item(text: str, is_current: bool) -> QTableWidgetItem:
    item = make_table_item(text)

    if is_current:
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        item.setForeground(QColor(INFO_BLUE))

    return item


# ============================================================
# Custom visual widget: sliding-window sequence strip
# ============================================================

class SequenceStripWidget(QWidget):
    """
    Draws one ecological trend sequence as a row of tiles.

    When windows are built, the selected input window is outlined
    with a maroon rounded rectangle and the target value is circled
    in blue.
    """

    def __init__(self) -> None:
        super().__init__()

        self.sequence: List[int] = []
        self.window_length: int = 4
        self.selected_window_index: int = 0
        self.windows_built: bool = False
        self.show_title: bool = True
        self.show_counter: bool = True
        self.ready_message: str = "Click Build First Window to box the input window and circle the target."

        self.setMinimumHeight(230)

    def set_state(
        self,
        sequence: List[int],
        window_length: int,
        selected_window_index: int,
        windows_built: bool,
    ) -> None:
        self.sequence = sequence
        self.window_length = window_length
        self.selected_window_index = selected_window_index
        self.windows_built = windows_built
        self.update()

    def set_display_options(
        self,
        show_title: bool = True,
        show_counter: bool = True,
        ready_message: str | None = None,
    ) -> None:
        self.show_title = show_title
        self.show_counter = show_counter
        if ready_message is not None:
            self.ready_message = ready_message
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        maroon = QColor(COBBER_MAROON)
        blue = QColor(INFO_BLUE)
        charcoal = QColor(CHARCOAL)
        light_gray = QColor(LIGHT_GRAY)
        mid_gray = QColor("#d6d6d6")
        white = QColor("#ffffff")

        painter.fillRect(self.rect(), white)

        if not self.sequence:
            painter.setPen(charcoal)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No sequence loaded.")
            return

        n = len(self.sequence)
        left_margin = 70
        right_margin = 70
        available_width = max(200, width - left_margin - right_margin)
        tile_gap = 10
        tile_width = min(70, (available_width - tile_gap * (n - 1)) / n)
        tile_height = 54
        total_strip_width = n * tile_width + (n - 1) * tile_gap
        start_x = (width - total_strip_width) / 2
        tile_y = 88 if self.show_title else 62

        if self.show_title:
            title_font = QFont("Lato", 15)
            title_font.setBold(True)
            painter.setFont(title_font)
            painter.setPen(charcoal)
            title = "Current sliding window" if self.windows_built else "Sequence"
            painter.drawText(QRectF(0, 14, width, 28), Qt.AlignmentFlag.AlignCenter, title)

        tile_rects: List[QRectF] = []
        for index, value in enumerate(self.sequence):
            x = start_x + index * (tile_width + tile_gap)
            tile_rect = QRectF(x, tile_y, tile_width, tile_height)
            tile_rects.append(tile_rect)

            painter.setFont(QFont("Lato", 9))
            painter.setPen(QColor("#555555"))
            painter.drawText(QRectF(x, tile_y - 28, tile_width, 20), Qt.AlignmentFlag.AlignCenter, f"t{index + 1}")

            painter.setPen(QPen(mid_gray, 1))
            painter.setBrush(QBrush(light_gray))
            painter.drawRoundedRect(tile_rect, 8, 8)

            value_font = QFont("Lato", 14)
            value_font.setBold(True)
            painter.setFont(value_font)
            painter.setPen(charcoal)
            painter.drawText(tile_rect, Qt.AlignmentFlag.AlignCenter, trend_text(value))

        if not self.windows_built:
            painter.setFont(QFont("Lato", 12))
            painter.setPen(maroon)
            painter.drawText(
                QRectF(0, tile_y + tile_height + 28, width, 40),
                Qt.AlignmentFlag.AlignCenter,
                self.ready_message,
            )
            return

        input_start = self.selected_window_index
        input_end = input_start + self.window_length - 1
        target_index = input_start + self.window_length

        if input_start < 0 or target_index >= len(tile_rects):
            return

        first_rect = tile_rects[input_start]
        last_rect = tile_rects[input_end]
        input_outline = QRectF(
            first_rect.left() - 8,
            first_rect.top() - 8,
            last_rect.right() - first_rect.left() + 16,
            tile_height + 16,
        )

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(maroon, 5))
        painter.drawRoundedRect(input_outline, 12, 12)

        target_rect = tile_rects[target_index]
        circle_size = max(tile_width, tile_height) + 8
        circle_rect = QRectF(
            target_rect.center().x() - circle_size / 2,
            target_rect.center().y() - circle_size / 2,
            circle_size,
            circle_size,
        )
        painter.setPen(QPen(blue, 5))
        painter.drawEllipse(circle_rect)

        label_font = QFont("Lato", 11)
        label_font.setBold(True)
        painter.setFont(label_font)
        painter.setPen(maroon)
        painter.drawText(QRectF(input_outline.left(), input_outline.bottom() + 14, input_outline.width(), 24), Qt.AlignmentFlag.AlignCenter, "input window")
        painter.setPen(blue)
        painter.drawText(QRectF(circle_rect.left() - 20, circle_rect.bottom() + 14, circle_rect.width() + 40, 24), Qt.AlignmentFlag.AlignCenter, "target")

        if self.show_counter:
            counter_font = QFont("Lato", 12)
            counter_font.setBold(True)
            painter.setFont(counter_font)
            painter.setPen(charcoal)
            total_windows = len(self.sequence) - self.window_length
            painter.drawText(QRectF(0, height - 34, width, 24), Qt.AlignmentFlag.AlignCenter, f"Window {self.selected_window_index + 1} of {total_windows}")


# ============================================================
# Custom visual widget: input-window strip for Tabs 2 and 3
# ============================================================

class MemoryInputStripWidget(QWidget):
    """
    Draws the full sequence.

    The selected input window is boxed in maroon. During a pass,
    an arrow points to the current new input inside that box.
    """

    def __init__(self) -> None:
        super().__init__()

        self.sequence: List[int] = []
        self.window_start_index: int = 0
        self.window_length: int = 4
        self.current_step_index: int = 0
        self.pass_started: bool = False
        self.show_current_input: bool = False

        self.setMinimumHeight(155)

    def set_state(
        self,
        sequence: List[int],
        window_start_index: int,
        window_length: int,
        current_step_index: int,
        pass_started: bool,
        show_current_input: bool,
    ) -> None:
        self.sequence = sequence
        self.window_start_index = window_start_index
        self.window_length = window_length
        self.current_step_index = current_step_index
        self.pass_started = pass_started
        self.show_current_input = show_current_input
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        maroon = QColor(COBBER_MAROON)
        charcoal = QColor(CHARCOAL)
        light_gray = QColor(LIGHT_GRAY)
        mid_gray = QColor("#d6d6d6")
        white = QColor("#ffffff")

        painter.fillRect(self.rect(), white)

        if not self.sequence:
            painter.setPen(charcoal)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Choose a sequence to begin.")
            return

        n = len(self.sequence)
        left_margin = 16
        right_margin = 16
        available_width = max(200, width - left_margin - right_margin)
        tile_gap = 7
        tile_width = min(58, (available_width - tile_gap * (n - 1)) / n)
        tile_height = 44
        total_width = n * tile_width + (n - 1) * tile_gap
        start_x = (width - total_width) / 2
        tile_y = 42

        tile_rects: List[QRectF] = []
        for index, value in enumerate(self.sequence):
            x = start_x + index * (tile_width + tile_gap)
            tile_rect = QRectF(x, tile_y, tile_width, tile_height)
            tile_rects.append(tile_rect)

            painter.setBrush(QBrush(light_gray))
            painter.setPen(QPen(mid_gray, 1))
            painter.drawRoundedRect(tile_rect, 8, 8)

            value_font = QFont("Lato", 12)
            value_font.setBold(True)
            painter.setFont(value_font)
            painter.setPen(charcoal)
            painter.drawText(tile_rect, Qt.AlignmentFlag.AlignCenter, trend_text(value))

            painter.setFont(QFont("Lato", 8))
            painter.setPen(QColor("#555555"))
            painter.drawText(QRectF(x, tile_y - 22, tile_width, 16), Qt.AlignmentFlag.AlignCenter, f"t{index + 1}")

        input_start = self.window_start_index
        input_end = self.window_start_index + self.window_length - 1
        if 0 <= input_start < len(tile_rects) and 0 <= input_end < len(tile_rects):
            first_rect = tile_rects[input_start]
            last_rect = tile_rects[input_end]
            input_outline = QRectF(
                first_rect.left() - 7,
                first_rect.top() - 7,
                last_rect.right() - first_rect.left() + 14,
                tile_height + 14,
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(maroon, 4))
            painter.drawRoundedRect(input_outline, 12, 12)

        if self.pass_started and self.show_current_input:
            current_full_index = self.window_start_index + self.current_step_index
            if 0 <= current_full_index < len(tile_rects):
                current_rect = tile_rects[current_full_index]
                arrow_x = int(current_rect.center().x())
                arrow_tip_y = int(current_rect.bottom() + 6)
                arrow_start_y = int(current_rect.bottom() + 34)

                painter.setPen(QPen(maroon, 3))
                painter.drawLine(arrow_x, arrow_start_y, arrow_x, arrow_tip_y)
                painter.drawLine(arrow_x, arrow_tip_y, arrow_x - 7, arrow_tip_y + 9)
                painter.drawLine(arrow_x, arrow_tip_y, arrow_x + 7, arrow_tip_y + 9)

                label_font = QFont("Lato", 10)
                label_font.setBold(True)
                painter.setFont(label_font)
                painter.setPen(maroon)
                painter.drawText(
                    QRectF(current_rect.left() - 10, current_rect.bottom() + 34, tile_width + 20, 34),
                    Qt.AlignmentFlag.AlignCenter,
                    "current\ninput",
                )

# ============================================================
# Tab 1: Build Windows
# ============================================================

class BuildWindowsTab(QWidget):
    """
    Tab 1 student story:

    One ordered sequence can become many training examples.
    Each training example pairs recent history with the next value.
    """

    def __init__(self) -> None:
        super().__init__()

        self.sequence: List[int] = PRESETS["Chapter sequence"].copy()
        self.window_length: int = 4
        self.windows: List[WindowExample] = []
        self.selected_window_index: int = 0
        self.highest_revealed_window_index: int = 0
        self.windows_built: bool = False

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(12)

        self.side_panel = self._build_side_panel()
        self.main_panel = self._build_main_panel()

        root_layout.addWidget(self.side_panel, stretch=0)
        root_layout.addWidget(self.main_panel, stretch=1)

        self._refresh_everything()

    def _build_side_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setMinimumWidth(295)
        panel.setMaximumWidth(335)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        question_box = QGroupBox("Question")
        question_layout = QVBoxLayout(question_box)
        question = QLabel("How does one sequence become many training examples?")
        question.setWordWrap(True)
        question_layout.addWidget(question)
        layout.addWidget(question_box)

        directions_box = QGroupBox("What to do")
        directions_layout = QVBoxLayout(directions_box)
        directions = QLabel(
            "1. Choose a sequence.\n"
            "2. Choose a window length.\n"
            "3. Click Build First Window.\n"
            "4. Use Next Window to watch the recent-history window move."
        )
        directions.setWordWrap(True)
        directions_layout.addWidget(directions)
        layout.addWidget(directions_box)

        sequence_box = QGroupBox("Sequence")
        sequence_layout = QVBoxLayout(sequence_box)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        sequence_layout.addWidget(QLabel("Choose a sequence:"))
        sequence_layout.addWidget(self.preset_combo)
        sequence_note = QLabel("+1 = increasing damage\n0 = stable damage\n-1 = decreasing damage")
        sequence_note.setObjectName("SmallNote")
        sequence_layout.addWidget(sequence_note)
        layout.addWidget(sequence_box)

        controls_box = QGroupBox("Build windows")
        controls_layout = QVBoxLayout(controls_box)
        self.window_combo = QComboBox()
        self.window_combo.addItems(["2", "3", "4", "5", "6"])
        self.window_combo.setCurrentText(str(self.window_length))
        self.window_combo.setFixedWidth(90)
        self.window_combo.currentTextChanged.connect(self._window_length_changed)
        window_length_layout = QHBoxLayout()
        window_length_layout.addWidget(QLabel("Window length:"))
        window_length_layout.addWidget(self.window_combo)
        window_length_layout.addStretch()
        controls_layout.addLayout(window_length_layout)

        self.build_button = QPushButton("Build First Window")
        self.build_button.setObjectName("PrimaryButton")
        self.build_button.clicked.connect(self._build_windows)
        controls_layout.addWidget(self.build_button)

        nav_layout = QHBoxLayout()
        self.previous_button = QPushButton("Previous Window")
        self.next_button = QPushButton("Next Window")
        self.previous_button.setObjectName("StepButton")
        self.next_button.setObjectName("StepButton")
        self.previous_button.clicked.connect(self._previous_window)
        self.next_button.clicked.connect(self._next_window)
        nav_layout.addWidget(self.previous_button)
        nav_layout.addWidget(self.next_button)
        controls_layout.addLayout(nav_layout)

        self.window_status_label = QLabel("")
        self.window_status_label.setObjectName("StatusLabel")
        controls_layout.addWidget(self.window_status_label)
        layout.addWidget(controls_box)
        layout.addStretch()
        return panel

    def _build_main_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        header = QLabel("Build Training Windows")
        header.setObjectName("MainHeader")
        layout.addWidget(header)

        intro = QLabel(
            "A sliding window turns one ordered sequence into several training examples. "
            "Each example uses a short recent history to predict the next value."
        )
        intro.setWordWrap(True)
        intro.setObjectName("IntroText")
        layout.addWidget(intro)

        visual_box = QGroupBox("Sliding-window view")
        visual_layout = QVBoxLayout(visual_box)
        self.sequence_strip = SequenceStripWidget()
        visual_layout.addWidget(self.sequence_strip)
        self.window_summary = QTextEdit()
        self.window_summary.setReadOnly(True)
        self.window_summary.setMaximumHeight(120)
        visual_layout.addWidget(self.window_summary)
        layout.addWidget(visual_box)

        table_box = QGroupBox("Training examples created from this sequence")
        table_layout = QVBoxLayout(table_box)
        self.window_table = QTableWidget()
        table_layout.addWidget(self.window_table)
        layout.addWidget(table_box, stretch=2)
        return panel

    def _apply_preset(self, preset_name: str) -> None:
        if preset_name not in PRESETS:
            return
        self.sequence = PRESETS[preset_name].copy()
        self.selected_window_index = 0
        self.highest_revealed_window_index = 0
        self.windows_built = False
        self.windows = []
        self._refresh_everything()

    def _window_length_changed(self, value: str) -> None:
        self.window_length = int(value)
        self.selected_window_index = 0
        self.highest_revealed_window_index = 0
        self.windows_built = False
        self.windows = []
        self._refresh_everything()

    def _build_windows(self) -> None:
        self.windows = TinyMemoryCell.build_windows(self.sequence, self.window_length)
        if not self.windows:
            QMessageBox.warning(
                self,
                "No windows",
                "The window length is too long for this sequence. Choose a shorter window length or use a longer sequence.",
            )
            self.windows_built = False
            self.selected_window_index = 0
            self.highest_revealed_window_index = 0
            self._refresh_everything()
            return
        self.windows_built = True
        self.selected_window_index = 0
        self.highest_revealed_window_index = 0
        self._refresh_everything()

    def _previous_window(self) -> None:
        if not self.windows_built or not self.windows:
            return
        self.selected_window_index = max(0, self.selected_window_index - 1)
        self._refresh_everything()

    def _next_window(self) -> None:
        if not self.windows_built or not self.windows:
            return
        self.selected_window_index = min(len(self.windows) - 1, self.selected_window_index + 1)
        self.highest_revealed_window_index = max(self.highest_revealed_window_index, self.selected_window_index)
        self._refresh_everything()

    def _refresh_everything(self) -> None:
        self._refresh_buttons()
        self._refresh_sequence_display()
        self._refresh_window_summary()
        self._refresh_window_table()

    def _refresh_buttons(self) -> None:
        has_windows = self.windows_built and bool(self.windows)
        self.previous_button.setEnabled(has_windows and self.selected_window_index > 0)
        self.next_button.setEnabled(has_windows and self.selected_window_index < len(self.windows) - 1)
        if has_windows:
            self.window_status_label.setVisible(True)
            self.window_status_label.setText(f"Window {self.selected_window_index + 1} of {len(self.windows)}")
        else:
            self.window_status_label.setText("")
            self.window_status_label.setVisible(False)

    def _refresh_sequence_display(self) -> None:
        self.sequence_strip.set_state(
            sequence=self.sequence,
            window_length=self.window_length,
            selected_window_index=self.selected_window_index,
            windows_built=self.windows_built and bool(self.windows),
        )

    def _refresh_window_summary(self) -> None:
        if not self.windows_built or not self.windows:
            self.window_summary.setHtml(
                "<h3>Ready to build windows</h3>"
                "<p>Choose a window length and click <b>Build First Window</b>.</p>"
                "<p>The app will show how a short recent history is paired with the next value.</p>"
            )
            return
        current_window = self.windows[self.selected_window_index]
        inputs = format_window_values(current_window.inputs)
        target = trend_text(current_window.target)
        self.window_summary.setHtml(
            f"<h3>Window {current_window.window_number}</h3>"
            f"<p><b>Input window:</b> {inputs}</p>"
            f"<p><b>Target value:</b> {target}</p>"
            "<p>The model gets the input window as recent history. The target is the next value it tries to predict.</p>"
        )

    def _refresh_window_table(self) -> None:
        self.window_table.clear()
        self.window_table.setColumnCount(3)
        self.window_table.setHorizontalHeaderLabels(["Training example", "Input window", "Target"])
        self.window_table.verticalHeader().setVisible(False)
        if not self.windows_built or not self.windows:
            self.window_table.setRowCount(0)
            self.window_table.setColumnWidth(0, 150)
            self.window_table.setColumnWidth(1, 150)
            self.window_table.setColumnWidth(2, 90)
            return

        last_visible_index = max(self.highest_revealed_window_index, self.selected_window_index)
        visible_windows = self.windows[: last_visible_index + 1]
        self.window_table.setRowCount(len(visible_windows))
        for row_index, window in enumerate(visible_windows):
            is_current = row_index == self.selected_window_index
            example_item = make_table_item(window.window_number)
            if is_current:
                font = example_item.font()
                font.setBold(True)
                example_item.setFont(font)
            self.window_table.setItem(row_index, 0, example_item)
            self.window_table.setCellWidget(row_index, 1, make_input_cell(format_window_values(window.inputs), is_current=is_current))
            self.window_table.setItem(row_index, 2, make_target_item(trend_text(window.target), is_current=is_current))
            if is_current:
                self.window_table.selectRow(row_index)
        self.window_table.setColumnWidth(0, 150)
        self.window_table.setColumnWidth(1, 150)
        self.window_table.setColumnWidth(2, 90)
        self.window_table.resizeRowsToContents()


# ============================================================
# Shared base helpers for Tabs 2 and 3
# ============================================================

class WindowSelectionMixin:
    sequence: List[int]
    window_length: int
    windows: List[WindowExample]
    selected_window_index: int
    window_combo: QComboBox

    def _populate_window_combo(self) -> None:
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        for window in self.windows:
            self.window_combo.addItem(f"Start at t{window.start_index + 1}")
        self.window_combo.blockSignals(False)

    def _current_window(self) -> WindowExample | None:
        if not self.windows:
            return None
        self.selected_window_index = max(0, min(self.selected_window_index, len(self.windows) - 1))
        return self.windows[self.selected_window_index]


# ============================================================
# Tab 2: Watch Memory
# ============================================================

class WatchMemoryTab(QWidget, WindowSelectionMixin):
    """
    Tab 2 student story:

    The model reads one input window one value at a time.
    At each step, it keeps part of the old memory, writes part
    of the current input, and carries the new memory forward.
    """

    def __init__(self) -> None:
        super().__init__()

        self.sequence: List[int] = PRESETS["Chapter sequence"].copy()
        self.window_length: int = 4
        self.windows: List[WindowExample] = TinyMemoryCell.build_windows(self.sequence, self.window_length)
        self.selected_window_index: int = 0
        self.f_value: float = 0.60
        self.i_value: float = 0.40
        self.steps: List[MemoryStep] = []
        self.selected_step_index: int = 0
        self.highest_completed_step_index: int = -1
        self.pass_started: bool = False
        self.calc_stage: int = 0

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(12)
        self.side_panel = self._build_side_panel()
        self.main_panel = self._build_main_panel()
        root_layout.addWidget(self.side_panel, stretch=0)
        root_layout.addWidget(self.main_panel, stretch=1)
        self._refresh_everything()

    def _build_side_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setMinimumWidth(295)
        panel.setMaximumWidth(335)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        question_box = QGroupBox("Question")
        question_layout = QVBoxLayout(question_box)
        question = QLabel("What happens as the model reads one input window from a full sequence?")
        question.setWordWrap(True)
        question_layout.addWidget(question)
        layout.addWidget(question_box)

        window_box = QGroupBox("Choose sequence and window")
        window_layout = QVBoxLayout(window_box)
        self.sequence_combo = QComboBox()
        self.sequence_combo.addItems(list(PRESETS.keys()))
        self.sequence_combo.setCurrentText("Chapter sequence")
        self.sequence_combo.currentTextChanged.connect(self._sequence_changed)
        self.window_combo = QComboBox()
        self._populate_window_combo()
        self.window_combo.currentIndexChanged.connect(self._window_changed)
        window_layout.addWidget(QLabel("Sequence:"))
        window_layout.addWidget(self.sequence_combo)
        window_layout.addWidget(QLabel("Input window starts at:"))
        window_layout.addWidget(self.window_combo)
        layout.addWidget(window_box)

        controls_box = QGroupBox("Memory controls")
        controls_layout = QVBoxLayout(controls_box)
        self.f_label = QLabel("")
        self.f_slider = QSlider(Qt.Orientation.Horizontal)
        self.f_slider.setRange(0, 100)
        self.f_slider.setValue(60)
        self.f_slider.valueChanged.connect(self._f_slider_changed)
        controls_layout.addWidget(self.f_label)
        controls_layout.addWidget(self.f_slider)
        self.i_label = QLabel("")
        self.i_slider = QSlider(Qt.Orientation.Horizontal)
        self.i_slider.setRange(0, 100)
        self.i_slider.setValue(40)
        self.i_slider.valueChanged.connect(self._i_slider_changed)
        controls_layout.addWidget(self.i_label)
        controls_layout.addWidget(self.i_slider)
        layout.addWidget(controls_box)

        step_box = QGroupBox("Read the input window")
        step_layout = QVBoxLayout(step_box)
        self.show_old_button = QPushButton("Show Old Memory")
        self.calculate_kept_button = QPushButton("Calculate Kept Memory")
        self.show_input_button = QPushButton("Show New Input")
        self.calculate_written_button = QPushButton("Calculate Written Input")
        self.calculate_new_memory_button = QPushButton("Calculate New Memory")
        self.next_step_button = QPushButton("Next Input")
        for button in [
            self.show_old_button,
            self.calculate_kept_button,
            self.show_input_button,
            self.calculate_written_button,
            self.calculate_new_memory_button,
            self.next_step_button,
        ]:
            button.setObjectName("StepButton")
            step_layout.addWidget(button)
        self.show_old_button.clicked.connect(self._show_old_memory)
        self.calculate_kept_button.clicked.connect(self._calculate_kept_memory)
        self.show_input_button.clicked.connect(self._show_new_input)
        self.calculate_written_button.clicked.connect(self._calculate_written_input)
        self.calculate_new_memory_button.clicked.connect(self._calculate_new_memory)
        self.next_step_button.clicked.connect(self._next_step)
        self.step_status_label = QLabel("")
        self.step_status_label.setObjectName("StatusLabel")
        step_layout.addWidget(self.step_status_label)
        layout.addWidget(step_box)
        layout.addStretch()
        return panel

    def _build_main_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        header = QLabel("Watch Memory Change")
        header.setObjectName("MainHeader")
        layout.addWidget(header)

        main_row = QHBoxLayout()
        main_row.setSpacing(10)
        left_column = QVBoxLayout()
        left_column.setSpacing(8)

        strip_box = QGroupBox("Full sequence with input window")
        strip_layout = QVBoxLayout(strip_box)
        strip_layout.setContentsMargins(8, 8, 8, 8)
        self.input_strip = MemoryInputStripWidget()
        self.input_strip.setFixedHeight(155)
        strip_box.setFixedHeight(205)
        strip_layout.addWidget(self.input_strip)
        left_column.addWidget(strip_box, stretch=0)

        table_box = QGroupBox("Memory steps")
        table_layout = QVBoxLayout(table_box)
        table_layout.setContentsMargins(8, 8, 8, 8)
        self.memory_table = QTableWidget()
        self.memory_table.setMinimumHeight(260)
        table_layout.addWidget(self.memory_table)
        left_column.addWidget(table_box, stretch=1)
        left_panel = QWidget()
        left_panel.setLayout(left_column)
        left_panel.setMinimumWidth(640)
        left_panel.setMaximumWidth(700)
        main_row.addWidget(left_panel)

        cards_box = QGroupBox("Current step calculation")
        cards_box.setFixedWidth(410)
        cards_layout = QVBoxLayout(cards_box)
        cards_layout.setContentsMargins(8, 8, 8, 8)
        cards_layout.setSpacing(8)
        self.memory_rule_card = QTextEdit()
        self.memory_rule_card.setReadOnly(True)
        self.memory_rule_card.setMinimumHeight(86)
        self.memory_rule_card.setMaximumHeight(96)
        self.memory_rule_card.setFont(QFont("Lato", 11))
        self.memory_rule_card.setObjectName("RuleCard")
        self.memory_rule_card.setHtml(
            "<h3>Memory update rule</h3>"
            "<div style='font-size: 14px; font-weight: bold;'>"
            "new memory = <span style='color:#6c1d45;'>kept old memory</span> + "
            "<span style='color:#6c1d45;'>written new input</span><br>"
            "c<sub>t</sub> = <span style='color:#6c1d45;'>f</span> c<sub>t-1</sub> + "
            "<span style='color:#6c1d45;'>i</span> x<sub>t</sub>"
            "</div>"
        )
        self.old_memory_card = self._make_calc_card("Old memory coming in")
        self.kept_memory_card = self._make_calc_card("Kept old memory")
        self.written_input_card = self._make_calc_card("Written new input")
        self.new_memory_card = self._make_calc_card("New memory")
        cards_layout.addWidget(self.memory_rule_card)
        cards_layout.addWidget(self.old_memory_card)
        cards_layout.addWidget(self.kept_memory_card)
        cards_layout.addWidget(self.written_input_card)
        cards_layout.addWidget(self.new_memory_card)
        cards_layout.addStretch()
        main_row.addWidget(cards_box, stretch=2)
        layout.addLayout(main_row, stretch=1)
        return panel

    def _make_calc_card(self, title: str) -> QTextEdit:
        card = QTextEdit()
        card.setReadOnly(True)
        card.setMinimumHeight(112)
        card.setMaximumHeight(135)
        card.setFont(QFont("Lato", 11))
        card.setObjectName("CalcCard")
        card.setHtml(f"<h3>{title}</h3><p>Use the buttons to reveal this part of the calculation.</p>")
        return card

    def _sequence_changed(self, preset_name: str) -> None:
        if preset_name not in PRESETS:
            return
        self.sequence = PRESETS[preset_name].copy()
        self.windows = TinyMemoryCell.build_windows(self.sequence, self.window_length)
        self.selected_window_index = 0
        self._populate_window_combo()
        self._reset_memory_pass()
        self._refresh_everything()

    def _window_changed(self, index: int) -> None:
        if index < 0:
            return
        self.selected_window_index = index
        self._reset_memory_pass()
        self._refresh_everything()

    def _f_slider_changed(self, value: int) -> None:
        self.f_value = value / 100.0
        self._reset_memory_pass()
        self._refresh_everything()

    def _i_slider_changed(self, value: int) -> None:
        self.i_value = value / 100.0
        self._reset_memory_pass()
        self._refresh_everything()

    def _reset_memory_pass(self) -> None:
        self.steps = []
        self.selected_step_index = 0
        self.highest_completed_step_index = -1
        self.pass_started = False
        self.calc_stage = 0

    def _recalculate_steps(self) -> None:
        current_window = self._current_window()
        if current_window is None:
            self.steps = []
            return
        self.steps = TinyMemoryCell.memory_steps(current_window.inputs, self.f_value, self.i_value)
        if self.steps:
            self.selected_step_index = max(0, min(self.selected_step_index, len(self.steps) - 1))

    def _ensure_memory_pass_started(self) -> bool:
        if not self.pass_started:
            self._recalculate_steps()
            if not self.steps:
                return False
            self.pass_started = True
            self.selected_step_index = 0
            self.calc_stage = 0
            self.highest_completed_step_index = -1
        return bool(self.steps)

    def _show_old_memory(self) -> None:
        if not self._ensure_memory_pass_started():
            return
        self.calc_stage = 1
        self._refresh_everything()

    def _calculate_kept_memory(self) -> None:
        if self.pass_started and self.calc_stage == 1:
            self.calc_stage = 2
            self._refresh_everything()

    def _show_new_input(self) -> None:
        if self.pass_started and self.calc_stage == 2:
            self.calc_stage = 3
            self._refresh_everything()

    def _calculate_written_input(self) -> None:
        if self.pass_started and self.calc_stage == 3:
            self.calc_stage = 4
            self._refresh_everything()

    def _calculate_new_memory(self) -> None:
        if self.pass_started and self.calc_stage == 4:
            self.calc_stage = 5
            self.highest_completed_step_index = max(self.highest_completed_step_index, self.selected_step_index)
            self._refresh_everything()

    def _next_step(self) -> None:
        if not self.pass_started or not self.steps or self.calc_stage < 5:
            return
        if self.selected_step_index >= len(self.steps) - 1:
            return
        self.selected_step_index += 1
        self.calc_stage = 0
        self._refresh_everything()

    def _refresh_everything(self) -> None:
        self._refresh_sliders()
        self._refresh_buttons()
        self._refresh_input_strip()
        self._refresh_calc_cards()
        self._refresh_memory_table()

    def _refresh_sliders(self) -> None:
        self.f_label.setText(f"Keep old memory, f = {self.f_value:.2f}")
        self.i_label.setText(f"Write new input, i = {self.i_value:.2f}")

    def _refresh_buttons(self) -> None:
        has_started = self.pass_started and bool(self.steps)
        has_more_steps = has_started and self.selected_step_index < len(self.steps) - 1
        self.show_old_button.setEnabled((not has_started) or self.calc_stage == 0)
        self.calculate_kept_button.setEnabled(has_started and self.calc_stage == 1)
        self.show_input_button.setEnabled(has_started and self.calc_stage == 2)
        self.calculate_written_button.setEnabled(has_started and self.calc_stage == 3)
        self.calculate_new_memory_button.setEnabled(has_started and self.calc_stage == 4)
        self.next_step_button.setEnabled(has_more_steps and self.calc_stage == 5)
        if has_started:
            stage_text = {
                0: "Ready for old memory",
                1: "Old memory shown",
                2: "Kept memory calculated",
                3: "New input shown",
                4: "Written input calculated",
                5: "New memory calculated",
            }.get(self.calc_stage, "")
            self.step_status_label.setVisible(True)
            self.step_status_label.setText(f"Reading input {self.selected_step_index + 1} of {len(self.steps)}: {stage_text}")
        else:
            self.step_status_label.setText("")
            self.step_status_label.setVisible(False)

    def _refresh_input_strip(self) -> None:
        current_window = self._current_window()
        if current_window is None:
            self.input_strip.set_state([], 0, self.window_length, 0, False, False)
            return
        self.input_strip.set_state(
            sequence=self.sequence,
            window_start_index=current_window.start_index,
            window_length=self.window_length,
            current_step_index=self.selected_step_index,
            pass_started=self.pass_started and bool(self.steps),
            show_current_input=self.pass_started and bool(self.steps) and self.calc_stage >= 3,
        )

    def _card_placeholder(self, title: str, message: str) -> str:
        return f"<h3>{title}</h3><div style='font-size: 13px; color:#444444;'>{message}</div>"

    def _refresh_calc_cards(self) -> None:
        if not self.pass_started or not self.steps:
            self.old_memory_card.setHtml(self._card_placeholder("Old memory coming in", "Click <b>Show Old Memory</b> to begin this step."))
            self.kept_memory_card.setHtml(self._card_placeholder("Kept old memory", "This part appears after old memory is shown."))
            self.written_input_card.setHtml(self._card_placeholder("Written new input", "This part appears after the current input is shown."))
            self.new_memory_card.setHtml(self._card_placeholder("New memory", "This part appears after the kept memory and written input are calculated."))
            return

        step = self.steps[self.selected_step_index]
        step_sub = step.step
        prev_sub = step.step - 1

        if self.calc_stage >= 1:
            if step_sub == 1:
                self.old_memory_card.setHtml("<h3>Old memory coming in</h3><div style='font-size: 14px;'>c<sub>0</sub> = <b>0.000</b></div>")
            else:
                self.old_memory_card.setHtml(
                    f"<h3>Old memory carried from read step {prev_sub}</h3>"
                    "<div style='font-size: 14px;'>"
                    f"The new memory from read step {prev_sub} becomes the old memory for this read step.<br>"
                    f"<span style='color:{COBBER_MAROON}; font-weight:bold;'>c<sub>{prev_sub}</sub> = {fmt_decimal(step.c_prev)}</span>"
                    "</div>"
                )
        else:
            self.old_memory_card.setHtml(self._card_placeholder("Old memory coming in", "Click <b>Show Old Memory</b>."))

        if self.calc_stage >= 2:
            self.kept_memory_card.setHtml(
                "<h3>Kept old memory</h3>"
                "<div style='font-size: 14px;'>"
                f"<span style='color:{COBBER_MAROON}; font-weight:bold;'>f</span> &times; c<sub>{prev_sub}</sub><br>"
                f"{self.f_value:.2f} &times; {fmt_decimal(step.c_prev)} = <b>{fmt_decimal(step.kept_old_memory)}</b>"
                "</div>"
            )
        else:
            self.kept_memory_card.setHtml(self._card_placeholder("Kept old memory", "Click <b>Calculate Kept Memory</b>."))

        if self.calc_stage == 3:
            self.written_input_card.setHtml(
                "<h3>Written new input</h3>"
                "<div style='font-size: 14px;'>"
                f"Current input: x<sub>{step_sub}</sub> = <b>{trend_text(step.x)}</b><br>"
                "Now calculate how much of it is written."
                "</div>"
            )
        elif self.calc_stage >= 4:
            self.written_input_card.setHtml(
                "<h3>Written new input</h3>"
                "<div style='font-size: 14px;'>"
                f"<span style='color:{COBBER_MAROON}; font-weight:bold;'>i</span> &times; x<sub>{step_sub}</sub><br>"
                f"{self.i_value:.2f} &times; {trend_text(step.x)} = <b>{fmt_decimal(step.written_new_input)}</b>"
                "</div>"
            )
        else:
            self.written_input_card.setHtml(self._card_placeholder("Written new input", "Click <b>Show New Input</b>."))

        if self.calc_stage >= 5:
            self.new_memory_card.setHtml(
                "<h3>New memory</h3>"
                "<div style='font-size: 14px;'>"
                f"c<sub>{step_sub}</sub> = kept old memory + written input<br>"
                f"{fmt_decimal(step.kept_old_memory)} + {fmt_decimal(step.written_new_input)} = "
                f"<span style='color:{COBBER_MAROON}; font-weight:bold;'>c<sub>{step_sub}</sub> = {fmt_decimal(step.c_new)}</span>"
                "</div>"
            )
        else:
            self.new_memory_card.setHtml(self._card_placeholder("New memory", "Click <b>Calculate New Memory</b>."))

    def _refresh_memory_table(self) -> None:
        self.memory_table.clear()
        self.memory_table.setColumnCount(6)
        self.memory_table.setHorizontalHeaderLabels(["Read step", "Old c", "Kept old", "x_t", "Written", "New c_t"])
        self.memory_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.memory_table.verticalHeader().setVisible(False)
        if not self.pass_started or not self.steps:
            self.memory_table.setRowCount(0)
            return

        last_visible_index = max(self.highest_completed_step_index, self.selected_step_index)
        if last_visible_index < 0:
            self.memory_table.setRowCount(0)
            return
        visible_steps = self.steps[: last_visible_index + 1]
        self.memory_table.setRowCount(len(visible_steps))
        for row_index, step in enumerate(visible_steps):
            is_current_row = row_index == self.selected_step_index
            row_is_completed = row_index <= self.highest_completed_step_index
            if row_is_completed and not is_current_row:
                row_stage = 5
            elif row_is_completed and is_current_row:
                row_stage = max(self.calc_stage, 5)
            elif is_current_row:
                row_stage = self.calc_stage
            else:
                row_stage = 0
            self.memory_table.setItem(row_index, 0, make_table_item(step.step))
            self.memory_table.setItem(row_index, 1, make_table_item(fmt_decimal(step.c_prev)) if row_stage >= 1 else make_table_item(""))
            self.memory_table.setItem(row_index, 2, make_table_item(fmt_decimal(step.kept_old_memory)) if row_stage >= 2 else make_table_item(""))
            self.memory_table.setItem(row_index, 3, make_table_item(trend_text(step.x)) if row_stage >= 3 else make_table_item(""))
            self.memory_table.setItem(row_index, 4, make_table_item(fmt_decimal(step.written_new_input)) if row_stage >= 4 else make_table_item(""))
            self.memory_table.setItem(row_index, 5, make_maroon_bold_table_item(fmt_decimal(step.c_new)) if row_stage >= 5 else make_table_item(""))
        self.memory_table.resizeRowsToContents()


# ============================================================
# Tab 3: Expose Memory
# ============================================================

class ExposeMemoryTab(QWidget, WindowSelectionMixin):
    """
    Tab 3 student story:

    Stored memory c_t is not used directly. The model first squashes it
    with tanh, then exposes part of it as the hidden state h_t.
    """

    FIXED_F_VALUE = 0.60
    FIXED_I_VALUE = 0.40

    def __init__(self) -> None:
        super().__init__()

        self.sequence: List[int] = PRESETS["Chapter sequence"].copy()
        self.window_length: int = 4
        self.windows: List[WindowExample] = TinyMemoryCell.build_windows(self.sequence, self.window_length)
        self.selected_window_index: int = 0
        self.o_value: float = 0.90
        self.steps: List[ExposureStep] = []
        self.selected_step_index: int = 0
        self.highest_completed_step_index: int = -1
        self.pass_started: bool = False
        self.calc_stage: int = 0

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(12)
        self.side_panel = self._build_side_panel()
        self.main_panel = self._build_main_panel()
        root_layout.addWidget(self.side_panel, stretch=0)
        root_layout.addWidget(self.main_panel, stretch=1)
        self._refresh_everything()

    def _build_side_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setMinimumWidth(295)
        panel.setMaximumWidth(335)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        question_box = QGroupBox("Question")
        question_layout = QVBoxLayout(question_box)
        question = QLabel("How does stored memory become exposed memory?")
        question.setWordWrap(True)
        question_layout.addWidget(question)
        layout.addWidget(question_box)

        window_box = QGroupBox("Choose sequence and window")
        window_layout = QVBoxLayout(window_box)
        self.sequence_combo = QComboBox()
        self.sequence_combo.addItems(list(PRESETS.keys()))
        self.sequence_combo.setCurrentText("Chapter sequence")
        self.sequence_combo.currentTextChanged.connect(self._sequence_changed)
        self.window_combo = QComboBox()
        self._populate_window_combo()
        self.window_combo.currentIndexChanged.connect(self._window_changed)
        window_layout.addWidget(QLabel("Sequence:"))
        window_layout.addWidget(self.sequence_combo)
        window_layout.addWidget(QLabel("Input window starts at:"))
        window_layout.addWidget(self.window_combo)
        layout.addWidget(window_box)

        fixed_box = QGroupBox("Memory settings")
        fixed_layout = QVBoxLayout(fixed_box)
        fixed_note = QLabel("This tab keeps f = 0.60 and i = 0.40 so we can focus on exposing memory.")
        fixed_note.setWordWrap(True)
        fixed_note.setObjectName("SmallNote")
        fixed_layout.addWidget(fixed_note)
        layout.addWidget(fixed_box)

        expose_box = QGroupBox("Expose control")
        expose_layout = QVBoxLayout(expose_box)
        self.o_label = QLabel("")
        self.o_slider = QSlider(Qt.Orientation.Horizontal)
        self.o_slider.setRange(0, 100)
        self.o_slider.setValue(90)
        self.o_slider.valueChanged.connect(self._o_slider_changed)
        expose_layout.addWidget(self.o_label)
        expose_layout.addWidget(self.o_slider)
        layout.addWidget(expose_box)

        step_box = QGroupBox("Expose memory")
        step_layout = QVBoxLayout(step_box)
        self.show_cell_button = QPushButton("Show Cell Memory")
        self.squash_button = QPushButton("Squash Memory")
        self.expose_button = QPushButton("Expose Hidden State")
        self.next_step_button = QPushButton("Next Input")
        for button in [self.show_cell_button, self.squash_button, self.expose_button, self.next_step_button]:
            button.setObjectName("StepButton")
            step_layout.addWidget(button)
        self.show_cell_button.clicked.connect(self._show_cell_memory)
        self.squash_button.clicked.connect(self._squash_memory)
        self.expose_button.clicked.connect(self._expose_hidden_state)
        self.next_step_button.clicked.connect(self._next_step)
        self.step_status_label = QLabel("")
        self.step_status_label.setObjectName("StatusLabel")
        step_layout.addWidget(self.step_status_label)
        layout.addWidget(step_box)
        layout.addStretch()
        return panel

    def _build_main_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        header = QLabel("Expose Memory")
        header.setObjectName("MainHeader")
        layout.addWidget(header)

        intro = QLabel(
            "Cell memory, c_t, is stored memory. Hidden state, h_t, is exposed memory that the next part of the model can use."
        )
        intro.setWordWrap(True)
        intro.setObjectName("IntroText")
        layout.addWidget(intro)

        main_row = QHBoxLayout()
        main_row.setSpacing(10)
        left_column = QVBoxLayout()
        left_column.setSpacing(8)

        strip_box = QGroupBox("Full sequence with input window")
        strip_layout = QVBoxLayout(strip_box)
        strip_layout.setContentsMargins(8, 8, 8, 8)
        self.input_strip = MemoryInputStripWidget()
        self.input_strip.setFixedHeight(155)
        strip_box.setFixedHeight(205)
        strip_layout.addWidget(self.input_strip)
        left_column.addWidget(strip_box, stretch=0)

        table_box = QGroupBox("Hidden state table")
        table_layout = QVBoxLayout(table_box)
        table_layout.setContentsMargins(8, 8, 8, 8)
        self.exposure_table = QTableWidget()
        self.exposure_table.setMinimumHeight(260)
        table_layout.addWidget(self.exposure_table)
        left_column.addWidget(table_box, stretch=1)
        left_panel = QWidget()
        left_panel.setLayout(left_column)
        left_panel.setMinimumWidth(640)
        left_panel.setMaximumWidth(700)
        main_row.addWidget(left_panel)

        cards_box = QGroupBox("Current exposure calculation")
        cards_box.setFixedWidth(410)
        cards_layout = QVBoxLayout(cards_box)
        cards_layout.setContentsMargins(8, 8, 8, 8)
        cards_layout.setSpacing(8)
        self.exposure_rule_card = QTextEdit()
        self.exposure_rule_card.setReadOnly(True)
        self.exposure_rule_card.setMinimumHeight(100)
        self.exposure_rule_card.setMaximumHeight(120)
        self.exposure_rule_card.setFont(QFont("Lato", 11))
        self.exposure_rule_card.setObjectName("RuleCard")
        self.exposure_rule_card.setHtml(
            "<h3>Memory exposure rule</h3>"
            "<div style='font-size: 14px; font-weight: bold;'>"
            "hidden memory = expose control &middot; squashed memory<br>"
            "h<sub>t</sub> = o &middot; tanh(c<sub>t</sub>)"
            "</div>"
            "<p>tanh keeps the direction of memory but limits its size.</p>"
        )
        self.cell_memory_card = self._make_calc_card("Cell memory")
        self.squashed_memory_card = self._make_calc_card("Squashed memory")
        self.hidden_state_card = self._make_calc_card("Hidden state")
        cards_layout.addWidget(self.exposure_rule_card)
        cards_layout.addWidget(self.cell_memory_card)
        cards_layout.addWidget(self.squashed_memory_card)
        cards_layout.addWidget(self.hidden_state_card)
        cards_layout.addStretch()
        main_row.addWidget(cards_box, stretch=2)
        layout.addLayout(main_row, stretch=1)
        return panel

    def _make_calc_card(self, title: str) -> QTextEdit:
        card = QTextEdit()
        card.setReadOnly(True)
        card.setMinimumHeight(120)
        card.setMaximumHeight(145)
        card.setFont(QFont("Lato", 11))
        card.setObjectName("CalcCard")
        card.setHtml(f"<h3>{title}</h3><p>Use the buttons to reveal this part of the calculation.</p>")
        return card

    def _sequence_changed(self, preset_name: str) -> None:
        if preset_name not in PRESETS:
            return
        self.sequence = PRESETS[preset_name].copy()
        self.windows = TinyMemoryCell.build_windows(self.sequence, self.window_length)
        self.selected_window_index = 0
        self._populate_window_combo()
        self._reset_exposure_pass()
        self._refresh_everything()

    def _window_changed(self, index: int) -> None:
        if index < 0:
            return
        self.selected_window_index = index
        self._reset_exposure_pass()
        self._refresh_everything()

    def _o_slider_changed(self, value: int) -> None:
        self.o_value = value / 100.0
        self._reset_exposure_pass()
        self._refresh_everything()

    def _reset_exposure_pass(self) -> None:
        self.steps = []
        self.selected_step_index = 0
        self.highest_completed_step_index = -1
        self.pass_started = False
        self.calc_stage = 0

    def _recalculate_steps(self) -> None:
        current_window = self._current_window()
        if current_window is None:
            self.steps = []
            return
        self.steps = TinyMemoryCell.exposure_steps(
            current_window.inputs,
            self.FIXED_F_VALUE,
            self.FIXED_I_VALUE,
            self.o_value,
        )
        if self.steps:
            self.selected_step_index = max(0, min(self.selected_step_index, len(self.steps) - 1))

    def _ensure_exposure_pass_started(self) -> bool:
        if not self.pass_started:
            self._recalculate_steps()
            if not self.steps:
                return False
            self.pass_started = True
            self.selected_step_index = 0
            self.calc_stage = 0
            self.highest_completed_step_index = -1
        return bool(self.steps)

    def _show_cell_memory(self) -> None:
        if not self._ensure_exposure_pass_started():
            return
        self.calc_stage = 1
        self._refresh_everything()

    def _squash_memory(self) -> None:
        if self.pass_started and self.calc_stage == 1:
            self.calc_stage = 2
            self._refresh_everything()

    def _expose_hidden_state(self) -> None:
        if self.pass_started and self.calc_stage == 2:
            self.calc_stage = 3
            self.highest_completed_step_index = max(self.highest_completed_step_index, self.selected_step_index)
            self._refresh_everything()

    def _next_step(self) -> None:
        if not self.pass_started or not self.steps or self.calc_stage < 3:
            return
        if self.selected_step_index >= len(self.steps) - 1:
            return
        self.selected_step_index += 1
        self.calc_stage = 0
        self._refresh_everything()

    def _refresh_everything(self) -> None:
        self._refresh_slider()
        self._refresh_buttons()
        self._refresh_input_strip()
        self._refresh_calc_cards()
        self._refresh_exposure_table()

    def _refresh_slider(self) -> None:
        self.o_label.setText(f"Expose memory, o = {self.o_value:.2f}")

    def _refresh_buttons(self) -> None:
        has_started = self.pass_started and bool(self.steps)
        has_more_steps = has_started and self.selected_step_index < len(self.steps) - 1
        self.show_cell_button.setEnabled((not has_started) or self.calc_stage == 0)
        self.squash_button.setEnabled(has_started and self.calc_stage == 1)
        self.expose_button.setEnabled(has_started and self.calc_stage == 2)
        self.next_step_button.setEnabled(has_more_steps and self.calc_stage == 3)
        if has_started:
            stage_text = {
                0: "Ready for cell memory",
                1: "Cell memory shown",
                2: "Squashed memory shown",
                3: "Hidden state exposed",
            }.get(self.calc_stage, "")
            self.step_status_label.setVisible(True)
            self.step_status_label.setText(f"Reading input {self.selected_step_index + 1} of {len(self.steps)}: {stage_text}")
        else:
            self.step_status_label.setText("")
            self.step_status_label.setVisible(False)

    def _refresh_input_strip(self) -> None:
        current_window = self._current_window()
        if current_window is None:
            self.input_strip.set_state([], 0, self.window_length, 0, False, False)
            return
        self.input_strip.set_state(
            sequence=self.sequence,
            window_start_index=current_window.start_index,
            window_length=self.window_length,
            current_step_index=self.selected_step_index,
            pass_started=self.pass_started and bool(self.steps),
            show_current_input=self.pass_started and bool(self.steps),
        )

    def _card_placeholder(self, title: str, message: str) -> str:
        return f"<h3>{title}</h3><div style='font-size: 13px; color:#444444;'>{message}</div>"

    def _refresh_calc_cards(self) -> None:
        if not self.pass_started or not self.steps:
            self.cell_memory_card.setHtml(self._card_placeholder("Cell memory", "Click <b>Show Cell Memory</b> to begin."))
            self.squashed_memory_card.setHtml(self._card_placeholder("Squashed memory", "This part appears after cell memory is shown."))
            self.hidden_state_card.setHtml(self._card_placeholder("Hidden state", "This part appears after memory is squashed."))
            return

        step = self.steps[self.selected_step_index]
        step_sub = step.step
        if self.calc_stage >= 1:
            self.cell_memory_card.setHtml(
                "<h3>Cell memory</h3>"
                "<div style='font-size: 14px;'>"
                "The cell has stored memory from reading this input.<br>"
                f"<span style='color:{COBBER_MAROON}; font-weight:bold;'>c<sub>{step_sub}</sub> = {fmt_decimal(step.c_value)}</span>"
                "</div>"
            )
        else:
            self.cell_memory_card.setHtml(self._card_placeholder("Cell memory", "Click <b>Show Cell Memory</b>."))

        if self.calc_stage >= 2:
            self.squashed_memory_card.setHtml(
                "<h3>Squashed memory</h3>"
                "<div style='font-size: 14px;'>"
                "tanh keeps the direction but limits the size.<br>"
                f"tanh({fmt_decimal(step.c_value)}) = <b>{fmt_decimal(step.tanh_c)}</b>"
                "</div>"
            )
        else:
            self.squashed_memory_card.setHtml(self._card_placeholder("Squashed memory", "Click <b>Squash Memory</b>."))

        if self.calc_stage >= 3:
            self.hidden_state_card.setHtml(
                "<h3>Hidden state</h3>"
                "<div style='font-size: 14px;'>"
                f"h<sub>{step_sub}</sub> = o &times; tanh(c<sub>{step_sub}</sub>)<br>"
                f"{self.o_value:.2f} &times; {fmt_decimal(step.tanh_c)} = "
                f"<span style='color:{COBBER_MAROON}; font-weight:bold;'>h<sub>{step_sub}</sub> = {fmt_decimal(step.h_value)}</span>"
                "</div>"
            )
        else:
            self.hidden_state_card.setHtml(self._card_placeholder("Hidden state", "Click <b>Expose Hidden State</b>."))

    def _refresh_exposure_table(self) -> None:
        self.exposure_table.clear()
        self.exposure_table.setColumnCount(5)
        self.exposure_table.setHorizontalHeaderLabels(["Read step", "c_t", "tanh(c_t)", "o", "h_t"])
        self.exposure_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.exposure_table.verticalHeader().setVisible(False)
        if not self.pass_started or not self.steps:
            self.exposure_table.setRowCount(0)
            return

        last_visible_index = max(self.highest_completed_step_index, self.selected_step_index)
        if last_visible_index < 0:
            self.exposure_table.setRowCount(0)
            return
        visible_steps = self.steps[: last_visible_index + 1]
        self.exposure_table.setRowCount(len(visible_steps))
        for row_index, step in enumerate(visible_steps):
            is_current_row = row_index == self.selected_step_index
            row_is_completed = row_index <= self.highest_completed_step_index
            if row_is_completed and not is_current_row:
                row_stage = 3
            elif row_is_completed and is_current_row:
                row_stage = max(self.calc_stage, 3)
            elif is_current_row:
                row_stage = self.calc_stage
            else:
                row_stage = 0
            self.exposure_table.setItem(row_index, 0, make_table_item(step.step))
            self.exposure_table.setItem(row_index, 1, make_table_item(fmt_decimal(step.c_value)) if row_stage >= 1 else make_table_item(""))
            self.exposure_table.setItem(row_index, 2, make_table_item(fmt_decimal(step.tanh_c)) if row_stage >= 2 else make_table_item(""))
            self.exposure_table.setItem(row_index, 3, make_table_item(f"{step.o_value:.2f}") if row_stage >= 3 else make_table_item(""))
            self.exposure_table.setItem(row_index, 4, make_maroon_bold_table_item(fmt_decimal(step.h_value)) if row_stage >= 3 else make_table_item(""))
        self.exposure_table.resizeRowsToContents()




# ============================================================
# Tab 4: Make Prediction
# ============================================================

class CheckPredictionTab(QWidget):
    """
    Tab 4 student story:

    Students complete the first two prediction checks by pressing
    the buttons for each conceptual step. After two guided rows,
    Complete Table finishes the remaining rows automatically.
    """

    FIXED_F_VALUE = 0.60
    FIXED_I_VALUE = 0.40
    FIXED_O_VALUE = 0.90
    OUTPUT_W_VALUE = 0.80
    OUTPUT_B_VALUE = 0.10

    CARD_TIME_MS = 900
    FILL_TIME_MS = 250
    GUIDED_ROW_COUNT = 2

    PHASES = ["window", "hidden", "prediction", "error", "loss"]

    def __init__(self) -> None:
        super().__init__()

        self.sequence: List[int] = PRESETS["Chapter sequence"].copy()
        self.window_length: int = 4
        self.windows: List[WindowExample] = TinyMemoryCell.build_windows(
            self.sequence,
            self.window_length,
        )
        self.rows: List[PredictionRow] = TinyMemoryCell.prediction_rows(
            self.sequence,
            self.window_length,
            self.FIXED_F_VALUE,
            self.FIXED_I_VALUE,
            self.FIXED_O_VALUE,
            self.OUTPUT_W_VALUE,
            self.OUTPUT_B_VALUE,
        )

        self.completed_row_count: int = 0
        self.active_row_index: int = 0
        self.manual_revealed_phase_index: int = -1
        self.selected_visual_window_index: int = 0

        self.auto_running: bool = False
        self.auto_phase_index: int = 0
        self.auto_phase_filled: bool = False

        self.last_card_row_index: int | None = None
        self.last_card_phase_index: int | None = None
        self.last_card_is_fill: bool = False

        self.action_timer = QTimer(self)
        self.action_timer.setInterval(self.CARD_TIME_MS)
        self.action_timer.timeout.connect(self._advance_auto_completion)

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(12)

        self.side_panel = self._build_side_panel()
        self.main_panel = self._build_main_panel()

        root_layout.addWidget(self.side_panel, stretch=0)
        root_layout.addWidget(self.main_panel, stretch=1)

        self._refresh_everything()

    def _build_side_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setMinimumWidth(295)
        panel.setMaximumWidth(335)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(9)

        question_box = QGroupBox("Question")
        question_layout = QVBoxLayout(question_box)
        question = QLabel("How does the model make and check a prediction for every training window?")
        question.setWordWrap(True)
        question_layout.addWidget(question)
        layout.addWidget(question_box)

        sequence_box = QGroupBox("Sequence")
        sequence_layout = QVBoxLayout(sequence_box)
        sequence_layout.setSpacing(6)
        sequence_layout.addWidget(QLabel("Choose a sequence:"))

        self.sequence_combo = QComboBox()
        self.sequence_combo.addItems(list(PRESETS.keys()))
        self.sequence_combo.setCurrentText("Chapter sequence")
        self.sequence_combo.currentTextChanged.connect(self._sequence_changed)
        sequence_layout.addWidget(self.sequence_combo)
        layout.addWidget(sequence_box)

        settings_box = QGroupBox("Fixed model settings")
        settings_layout = QVBoxLayout(settings_box)
        settings_layout.setSpacing(4)
        for text in [
            "Keep old memory: f = 0.60",
            "Write new input: i = 0.40",
            "Expose memory: o = 0.90",
            "Output weight: w = 0.80",
            "Output bias: b = 0.10",
        ]:
            label = QLabel(text)
            label.setObjectName("SmallNote")
            settings_layout.addWidget(label)
        layout.addWidget(settings_box)

        check_box = QGroupBox("Make prediction")
        check_layout = QVBoxLayout(check_box)
        check_layout.setSpacing(7)

        self.show_windows_button = QPushButton("Show All Windows")
        self.show_hidden_button = QPushButton("Show Final Hidden States")
        self.make_predictions_button = QPushButton("Make Predictions")
        self.calculate_error_button = QPushButton("Calculate Error")
        self.calculate_loss_button = QPushButton("Calculate Loss")
        self.complete_table_button = QPushButton("Complete Table")

        phase_buttons = [
            self.show_windows_button,
            self.show_hidden_button,
            self.make_predictions_button,
            self.calculate_error_button,
            self.calculate_loss_button,
            self.complete_table_button,
        ]

        for button in phase_buttons:
            button.setObjectName("StepButton")
            button.setMinimumHeight(36)
            check_layout.addWidget(button)

        self.show_windows_button.clicked.connect(lambda: self._manual_reveal_phase("window"))
        self.show_hidden_button.clicked.connect(lambda: self._manual_reveal_phase("hidden"))
        self.make_predictions_button.clicked.connect(lambda: self._manual_reveal_phase("prediction"))
        self.calculate_error_button.clicked.connect(lambda: self._manual_reveal_phase("error"))
        self.calculate_loss_button.clicked.connect(lambda: self._manual_reveal_phase("loss"))
        self.complete_table_button.clicked.connect(self._start_auto_completion)

        layout.addWidget(check_box)
        layout.addStretch()
        return panel

    def _build_main_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QLabel("Make Prediction")
        header.setObjectName("MainHeader")
        layout.addWidget(header)

        intro = QLabel(
            "Each input window is read all the way through. The model uses the final hidden memory from that window to predict the target."
        )
        intro.setWordWrap(True)
        intro.setObjectName("IntroText")
        layout.addWidget(intro)

        visual_box = QGroupBox("All windows in the sequence")
        visual_layout = QVBoxLayout(visual_box)
        visual_layout.setContentsMargins(8, 8, 8, 8)
        self.sequence_strip = SequenceStripWidget()
        self.sequence_strip.setMinimumHeight(155)
        self.sequence_strip.setFixedHeight(165)
        self.sequence_strip.set_display_options(
            show_title=False,
            show_counter=False,
            ready_message="Click Show All Windows to begin.",
        )
        visual_layout.addWidget(self.sequence_strip)
        visual_box.setFixedHeight(205)
        layout.addWidget(visual_box, stretch=0)

        table_box = QGroupBox("Prediction check table")
        table_layout = QVBoxLayout(table_box)
        table_layout.setContentsMargins(8, 8, 8, 8)
        self.prediction_table = QTableWidget()
        self.prediction_table.setMinimumHeight(225)
        table_layout.addWidget(self.prediction_table)
        layout.addWidget(table_box, stretch=1)

        card_box = QGroupBox("Current calculation")
        card_layout = QVBoxLayout(card_box)
        card_layout.setContentsMargins(8, 8, 8, 8)
        self.calculation_card = QTextEdit()
        self.calculation_card.setReadOnly(True)
        self.calculation_card.setFont(QFont("Lato", 11))
        self.calculation_card.setObjectName("CalcCard")
        self.calculation_card.setMinimumHeight(118)
        self.calculation_card.setMaximumHeight(140)
        card_layout.addWidget(self.calculation_card)
        layout.addWidget(card_box, stretch=0)

        return panel

    def _sequence_changed(self, preset_name: str) -> None:
        if preset_name not in PRESETS:
            return
        self.sequence = PRESETS[preset_name].copy()
        self.windows = TinyMemoryCell.build_windows(self.sequence, self.window_length)
        self.rows = TinyMemoryCell.prediction_rows(
            self.sequence,
            self.window_length,
            self.FIXED_F_VALUE,
            self.FIXED_I_VALUE,
            self.FIXED_O_VALUE,
            self.OUTPUT_W_VALUE,
            self.OUTPUT_B_VALUE,
        )
        self._clear_prediction_check()

    def _clear_prediction_check(self) -> None:
        if self.action_timer.isActive():
            self.action_timer.stop()
        self.completed_row_count = 0
        self.active_row_index = 0
        self.manual_revealed_phase_index = -1
        self.selected_visual_window_index = 0
        self.auto_running = False
        self.auto_phase_index = 0
        self.auto_phase_filled = False
        self.last_card_row_index = None
        self.last_card_phase_index = None
        self.last_card_is_fill = False
        self._refresh_everything()

    def _manual_reveal_phase(self, phase: str) -> None:
        if self.auto_running:
            return
        if self.active_row_index >= len(self.rows):
            return
        if self.completed_row_count >= self.GUIDED_ROW_COUNT:
            return

        phase_index = self.PHASES.index(phase)
        expected_phase_index = self.manual_revealed_phase_index + 1
        if phase_index != expected_phase_index:
            return

        self.manual_revealed_phase_index = phase_index
        self.selected_visual_window_index = self.active_row_index
        self.last_card_row_index = self.active_row_index
        self.last_card_phase_index = phase_index
        self.last_card_is_fill = True

        if phase == "loss":
            self.completed_row_count = max(self.completed_row_count, self.active_row_index + 1)
            self.active_row_index += 1
            self.manual_revealed_phase_index = -1
            if self.active_row_index < len(self.rows):
                self.selected_visual_window_index = self.active_row_index

        self._refresh_everything()

    def _start_auto_completion(self) -> None:
        if self.auto_running:
            return
        if self.completed_row_count < self.GUIDED_ROW_COUNT:
            return
        if self.completed_row_count >= len(self.rows):
            return

        self.active_row_index = self.completed_row_count
        self.selected_visual_window_index = self.active_row_index
        self.manual_revealed_phase_index = -1
        self.auto_running = True
        self.auto_phase_index = 0
        self.auto_phase_filled = False
        self.last_card_row_index = self.active_row_index
        self.last_card_phase_index = self.auto_phase_index
        self.last_card_is_fill = False
        self.action_timer.setInterval(self.CARD_TIME_MS)
        self.action_timer.start()
        self._refresh_everything()

    def _advance_auto_completion(self) -> None:
        if not self.auto_running:
            self.action_timer.stop()
            return

        if self.active_row_index >= len(self.rows):
            self._finish_auto_completion()
            return

        if not self.auto_phase_filled:
            self.auto_phase_filled = True
            self.last_card_row_index = self.active_row_index
            self.last_card_phase_index = self.auto_phase_index
            self.last_card_is_fill = True
            self.action_timer.setInterval(self.FILL_TIME_MS)
            self._refresh_everything()
            return

        self.auto_phase_filled = False
        self.auto_phase_index += 1

        if self.auto_phase_index >= len(self.PHASES):
            self.completed_row_count = max(self.completed_row_count, self.active_row_index + 1)
            self.active_row_index += 1
            self.auto_phase_index = 0

            if self.active_row_index >= len(self.rows):
                self._finish_auto_completion()
                return

            self.selected_visual_window_index = self.active_row_index

        self.last_card_row_index = self.active_row_index
        self.last_card_phase_index = self.auto_phase_index
        self.last_card_is_fill = False
        self.action_timer.setInterval(self.CARD_TIME_MS)
        self._refresh_everything()

    def _finish_auto_completion(self) -> None:
        self.action_timer.stop()
        self.completed_row_count = len(self.rows)
        self.auto_running = False
        self.auto_phase_index = 0
        self.auto_phase_filled = False
        self.active_row_index = len(self.rows)
        self.manual_revealed_phase_index = -1
        if self.rows:
            self.selected_visual_window_index = len(self.rows) - 1
        self._refresh_everything()

    def _refresh_everything(self) -> None:
        self._refresh_buttons()
        self._refresh_sequence_display()
        self._refresh_prediction_table()
        self._refresh_calculation_card()

    def _refresh_buttons(self) -> None:
        for button in [
            self.show_windows_button,
            self.show_hidden_button,
            self.make_predictions_button,
            self.calculate_error_button,
            self.calculate_loss_button,
            self.complete_table_button,
        ]:
            button.setEnabled(False)

        if self.auto_running or not self.rows:
            return

        if self.completed_row_count < self.GUIDED_ROW_COUNT and self.active_row_index < len(self.rows):
            expected_phase_index = self.manual_revealed_phase_index + 1
            button_by_phase = {
                0: self.show_windows_button,
                1: self.show_hidden_button,
                2: self.make_predictions_button,
                3: self.calculate_error_button,
                4: self.calculate_loss_button,
            }
            button = button_by_phase.get(expected_phase_index)
            if button is not None:
                button.setEnabled(True)
            return

        if self.completed_row_count >= self.GUIDED_ROW_COUNT and self.completed_row_count < len(self.rows):
            self.complete_table_button.setEnabled(True)

    def _refresh_sequence_display(self) -> None:
        windows_visible = (
            self.completed_row_count > 0
            or self.manual_revealed_phase_index >= 0
            or self.auto_running
        )

        if not self.windows:
            self.sequence_strip.set_state(
                sequence=self.sequence,
                window_length=self.window_length,
                selected_window_index=0,
                windows_built=False,
            )
            return

        self.selected_visual_window_index = max(
            0,
            min(self.selected_visual_window_index, len(self.windows) - 1),
        )

        self.sequence_strip.set_state(
            sequence=self.sequence,
            window_length=self.window_length,
            selected_window_index=self.selected_visual_window_index,
            windows_built=windows_visible,
        )

    def _visible_row_count(self) -> int:
        visible_count = self.completed_row_count
        if self.auto_running and self.active_row_index < len(self.rows):
            visible_count = max(visible_count, self.active_row_index + 1)
        elif self.manual_revealed_phase_index >= 0 and self.active_row_index < len(self.rows):
            visible_count = max(visible_count, self.active_row_index + 1)
        return visible_count

    def _phase_is_visible_for_row(self, row_index: int, phase: str) -> bool:
        phase_index = self.PHASES.index(phase)

        if row_index < self.completed_row_count:
            return True

        if self.auto_running and row_index == self.active_row_index:
            if phase_index < self.auto_phase_index:
                return True
            if phase_index == self.auto_phase_index and self.auto_phase_filled:
                return True
            return False

        if row_index == self.active_row_index:
            return phase_index <= self.manual_revealed_phase_index

        return False

    def _refresh_prediction_table(self) -> None:
        self.prediction_table.clear()
        self.prediction_table.setColumnCount(7)
        self.prediction_table.setHorizontalHeaderLabels(
            [
                "Window",
                "Input window",
                "Target y",
                "Final h_4",
                "Prediction ŷ",
                "Error e",
                "Loss e²",
            ]
        )
        self.prediction_table.verticalHeader().setVisible(False)
        self.prediction_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        column_widths = [80, 190, 90, 120, 120, 100, 100]
        for col_index, width in enumerate(column_widths):
            self.prediction_table.setColumnWidth(col_index, width)

        visible_count = self._visible_row_count()
        self.prediction_table.setRowCount(visible_count)

        if visible_count == 0:
            return

        for row_index, row in enumerate(self.rows[:visible_count]):
            is_current = row_index == self.selected_visual_window_index and (
                self.auto_running or row_index == self.active_row_index
            )

            if self._phase_is_visible_for_row(row_index, "window"):
                self.prediction_table.setItem(row_index, 0, make_table_item(row.window_number))
                self.prediction_table.setCellWidget(
                    row_index,
                    1,
                    make_input_cell(format_window_values(row.inputs), is_current=is_current),
                )
                self.prediction_table.setItem(
                    row_index,
                    2,
                    make_target_item(trend_text(row.target), is_current=is_current),
                )
            else:
                self.prediction_table.setItem(row_index, 0, make_table_item(""))
                self.prediction_table.setItem(row_index, 1, make_table_item(""))
                self.prediction_table.setItem(row_index, 2, make_table_item(""))

            if self._phase_is_visible_for_row(row_index, "hidden"):
                self.prediction_table.setItem(row_index, 3, make_maroon_bold_table_item(fmt_decimal(row.final_h)))
            else:
                self.prediction_table.setItem(row_index, 3, make_table_item(""))

            if self._phase_is_visible_for_row(row_index, "prediction"):
                self.prediction_table.setItem(row_index, 4, make_table_item(fmt_decimal(row.yhat)))
            else:
                self.prediction_table.setItem(row_index, 4, make_table_item(""))

            if self._phase_is_visible_for_row(row_index, "error"):
                self.prediction_table.setItem(row_index, 5, make_table_item(fmt_decimal(row.error)))
            else:
                self.prediction_table.setItem(row_index, 5, make_table_item(""))

            if self._phase_is_visible_for_row(row_index, "loss"):
                self.prediction_table.setItem(row_index, 6, make_table_item(fmt_decimal(row.loss)))
            else:
                self.prediction_table.setItem(row_index, 6, make_table_item(""))

        self.prediction_table.resizeRowsToContents()

    def _refresh_calculation_card(self) -> None:
        if self.last_card_row_index is not None and self.last_card_phase_index is not None:
            if 0 <= self.last_card_row_index < len(self.rows):
                row = self.rows[self.last_card_row_index]
                phase = self.PHASES[self.last_card_phase_index]
                self.calculation_card.setHtml(
                    self._card_html(row, phase, self.last_card_is_fill)
                )
                return

        if self.completed_row_count == 0:
            self.calculation_card.setHtml(
                "<h3>Ready</h3>"
                "<p>Use the buttons to complete the first two rows. Then click <b>Complete Table</b> to finish the remaining windows.</p>"
            )
        elif self.completed_row_count < self.GUIDED_ROW_COUNT:
            self.calculation_card.setHtml(
                "<h3>Next window</h3>"
                "<p>Use the buttons again to complete the second guided row.</p>"
            )
        elif self.completed_row_count < len(self.rows):
            self.calculation_card.setHtml(
                "<h3>Ready to complete the table</h3>"
                "<p>Click <b>Complete Table</b> to let the app finish the remaining windows.</p>"
            )
        else:
            self.calculation_card.setHtml(
                "<h3>Prediction check complete</h3>"
                "<p>Every training window now has a target, final hidden memory, prediction, error, and loss.</p>"
                "<p>Tab 5 will use these losses to adjust the output layer.</p>"
            )

    def _card_html(self, row: PredictionRow, phase: str, is_fill: bool) -> str:
        placement_note = "<p><b>Now placing this result in the table.</b></p>" if is_fill else ""

        if phase == "window":
            input_start = row.start_index + 1
            input_end = row.start_index + self.window_length
            target_pos = row.start_index + self.window_length + 1
            return (
                f"<h3>Window {row.window_number}</h3>"
                "<div style='font-size: 14px;'>"
                f"The model reads <b>t{input_start} through t{input_end}</b>.<br>"
                f"The target is the next value, <b>t{target_pos}</b>.<br>"
                f"Input window: <b>{format_window_values(row.inputs)}</b><br>"
                f"Target y = <b>{trend_text(row.target)}</b>"
                "</div>"
                f"{placement_note}"
            )

        if phase == "hidden":
            return (
                f"<h3>Final hidden memory for Window {row.window_number}</h3>"
                "<div style='font-size: 14px;'>"
                "The model reads all four inputs in this window.<br>"
                "Prediction uses the final hidden memory from the fourth read step.<br>"
                f"<span style='color:{COBBER_MAROON}; font-weight:bold;'>"
                f"h<sub>4</sub> = {fmt_decimal(row.final_h)}</span>"
                "</div>"
                f"{placement_note}"
            )

        if phase == "prediction":
            weighted = self.OUTPUT_W_VALUE * row.final_h
            return (
                f"<h3>Prediction for Window {row.window_number}</h3>"
                "<div style='font-size: 14px;'>"
                "prediction = bias + output weight &middot; final hidden memory<br>"
                "ŷ = b + w &middot; h<sub>4</sub><br>"
                f"ŷ = {self.OUTPUT_B_VALUE:.2f} + {self.OUTPUT_W_VALUE:.2f} &middot; {fmt_decimal(row.final_h)}<br>"
                f"ŷ = {self.OUTPUT_B_VALUE:.2f} + {fmt_decimal(weighted)} = "
                f"<span style='color:{COBBER_MAROON}; font-weight:bold;'>{fmt_decimal(row.yhat)}</span>"
                "</div>"
                f"{placement_note}"
            )

        if phase == "error":
            return (
                f"<h3>Error for Window {row.window_number}</h3>"
                "<div style='font-size: 14px;'>"
                "error = prediction - target<br>"
                "e = ŷ - y<br>"
                f"e = {fmt_decimal(row.yhat)} - ({trend_text(row.target)})<br>"
                f"e = <span style='color:{COBBER_MAROON}; font-weight:bold;'>{fmt_decimal(row.error)}</span>"
                "</div>"
                f"{placement_note}"
            )

        if phase == "loss":
            return (
                f"<h3>Loss for Window {row.window_number}</h3>"
                "<div style='font-size: 14px;'>"
                "loss = error<sup>2</sup><br>"
                f"loss = ({fmt_decimal(row.error)})<sup>2</sup><br>"
                f"loss = <span style='color:{COBBER_MAROON}; font-weight:bold;'>{fmt_decimal(row.loss)}</span>"
                "</div>"
                f"{placement_note}"
            )

        return "<h3>Ready</h3><p>Use the buttons to check the model.</p>"




# ============================================================
# Tab 5: One Training Step
# ============================================================

class OneTrainingStepTab(QWidget):
    """
    Tab 5 student story:

    The model already has final hidden memories, predictions,
    errors, and losses from Tab 4. This tab uses those errors to
    calculate update signals, averages the signals across all
    training windows, and uses the learning rate eta to update
    only the output weight and output bias.
    """

    FIXED_F_VALUE = 0.60
    FIXED_I_VALUE = 0.40
    FIXED_O_VALUE = 0.90
    START_W_VALUE = 0.80
    START_B_VALUE = 0.10
    ETA_VALUE = 0.10

    CARD_TIME_MS = 900
    GUIDED_ROW_COUNT = 2

    def __init__(self) -> None:
        super().__init__()

        self.sequence: List[int] = PRESETS["Chapter sequence"].copy()
        self.window_length: int = 4
        self.rows: List[PredictionRow] = []

        self.before_visible_count: int = 0
        self.signal_visible: List[List[bool]] = []
        self.after_visible: List[List[bool]] = []

        self.average_w_signal: float = 0.0
        self.average_b_signal: float = 0.0
        self.new_w_value: float = self.START_W_VALUE
        self.new_b_value: float = self.START_B_VALUE
        self.averages_calculated: bool = False
        self.parameters_updated: bool = False

        self.timer_mode: str = ""
        self.timer_row_index: int = 0
        self.timer_field_index: int = 0

        self.action_timer = QTimer(self)
        self.action_timer.setInterval(self.CARD_TIME_MS)
        self.action_timer.timeout.connect(self._advance_timer_action)

        self._rebuild_training_rows()

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(6)

        header = QLabel("One Training Step")
        header.setObjectName("MainHeader")
        root_layout.addWidget(header)

        intro = QLabel(
            "This tab freezes the memory controls and updates only the output weight and output bias. "
            "The update uses errors from all training windows."
        )
        intro.setWordWrap(True)
        intro.setObjectName("IntroText")
        root_layout.addWidget(intro)

        settings_box = QGroupBox("Training settings")
        settings_layout = QVBoxLayout(settings_box)
        settings_layout.setContentsMargins(10, 8, 10, 8)
        settings_layout.setSpacing(3)
        self.settings_label = QLabel("")
        self.settings_label.setObjectName("SmallNote")
        self.settings_label.setWordWrap(True)
        self.summary_label = QLabel("")
        self.summary_label.setObjectName("StatusLabel")
        self.summary_label.setWordWrap(True)
        settings_layout.addWidget(self.settings_label)
        settings_layout.addWidget(self.summary_label)
        root_layout.addWidget(settings_box, stretch=0)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        top_row.addWidget(self._build_before_button_card(), stretch=0)
        top_row.addWidget(self._build_before_section(), stretch=1)
        top_row.addWidget(self._build_before_card_section(), stretch=0)
        root_layout.addLayout(top_row, stretch=1)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)
        bottom_row.addWidget(self._build_update_button_card(), stretch=0)
        bottom_row.addWidget(self._build_update_section(), stretch=1)
        bottom_row.addWidget(self._build_update_card_section(), stretch=0)
        root_layout.addLayout(bottom_row, stretch=1)

        self._refresh_everything()

    def _build_before_button_card(self) -> QWidget:
        box = QGroupBox("Earlier calculations")
        box.setFixedWidth(245)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        note = QLabel("Bring in the values from Tab 4 one row at a time.")
        note.setWordWrap(True)
        note.setObjectName("SmallNote")
        layout.addWidget(note)

        self.add_before_button = QPushButton("Add Earlier Calculations")
        self.add_before_button.setObjectName("StepButton")
        self.add_before_button.setMinimumHeight(40)
        layout.addWidget(self.add_before_button)

        layout.addStretch()
        self.add_before_button.clicked.connect(self._add_before_row)
        return box

    def _build_before_section(self) -> QWidget:
        box = QGroupBox("Earlier calculations from Tab 4")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.before_table = QTableWidget()
        self.before_table.setMinimumHeight(165)
        self.before_table.setMaximumHeight(210)
        layout.addWidget(self.before_table, stretch=1)
        return box

    def _build_before_card_section(self) -> QWidget:
        box = QGroupBox("Training bridge card")
        box.setFixedWidth(390)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        self.before_card = QTextEdit()
        self.before_card.setReadOnly(True)
        self.before_card.setFont(QFont("Lato", 11))
        self.before_card.setObjectName("CalcCard")
        self.before_card.setMinimumHeight(190)
        layout.addWidget(self.before_card)
        return box

    def _build_update_button_card(self) -> QWidget:
        box = QGroupBox("Training actions")
        box.setFixedWidth(245)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        note = QLabel("Calculate signals, average them, then use η to update w and b.")
        note.setWordWrap(True)
        note.setObjectName("SmallNote")
        layout.addWidget(note)

        self.calculate_signals_button = QPushButton("Calculate Update Signals")
        self.complete_signals_button = QPushButton("Complete Signals")
        self.average_signals_button = QPushButton("Average Signals")
        self.update_parameters_button = QPushButton("Use η to Update w and b")
        self.check_after_button = QPushButton("Check After Update")
        self.complete_after_button = QPushButton("Complete After Table")

        for button in [
            self.calculate_signals_button,
            self.complete_signals_button,
            self.average_signals_button,
            self.update_parameters_button,
            self.check_after_button,
            self.complete_after_button,
        ]:
            button.setObjectName("StepButton")
            button.setMinimumHeight(38)
            layout.addWidget(button)

        layout.addStretch()

        self.calculate_signals_button.clicked.connect(self._start_guided_signals)
        self.complete_signals_button.clicked.connect(self._start_complete_signals)
        self.average_signals_button.clicked.connect(self._average_signals)
        self.update_parameters_button.clicked.connect(self._update_w_and_b)
        self.check_after_button.clicked.connect(self._start_guided_after_update)
        self.complete_after_button.clicked.connect(self._start_complete_after_table)
        return box

    def _build_update_section(self) -> QWidget:
        box = QGroupBox("One output-layer update")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.update_table = QTableWidget()
        self.update_table.setMinimumHeight(205)
        layout.addWidget(self.update_table, stretch=1)
        return box

    def _build_update_card_section(self) -> QWidget:
        box = QGroupBox("Training calculation card")
        box.setFixedWidth(390)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        self.update_card = QTextEdit()
        self.update_card.setReadOnly(True)
        self.update_card.setFont(QFont("Lato", 11))
        self.update_card.setObjectName("CalcCard")
        self.update_card.setMinimumHeight(250)
        layout.addWidget(self.update_card)
        return box

    def _rebuild_training_rows(self) -> None:
        self.rows = TinyMemoryCell.prediction_rows(
            self.sequence,
            self.window_length,
            self.FIXED_F_VALUE,
            self.FIXED_I_VALUE,
            self.FIXED_O_VALUE,
            self.START_W_VALUE,
            self.START_B_VALUE,
        )
        self.signal_visible = [[False, False] for _ in self.rows]
        self.after_visible = [[False, False, False] for _ in self.rows]

    def _all_signals_complete(self) -> bool:
        return bool(self.signal_visible) and all(all(values) for values in self.signal_visible)

    def _guided_signals_complete(self) -> bool:
        if len(self.signal_visible) < self.GUIDED_ROW_COUNT:
            return False
        return all(all(self.signal_visible[row_index]) for row_index in range(self.GUIDED_ROW_COUNT))

    def _all_after_complete(self) -> bool:
        return bool(self.after_visible) and all(all(values) for values in self.after_visible)

    def _guided_after_complete(self) -> bool:
        if len(self.after_visible) < self.GUIDED_ROW_COUNT:
            return False
        return all(all(self.after_visible[row_index]) for row_index in range(self.GUIDED_ROW_COUNT))

    def _w_signal(self, row: PredictionRow) -> float:
        return 2.0 * row.error * row.final_h

    def _b_signal(self, row: PredictionRow) -> float:
        return 2.0 * row.error

    def _after_values(self, row: PredictionRow) -> tuple[float, float, float]:
        yhat_after = self.new_b_value + self.new_w_value * row.final_h
        error_after = yhat_after - float(row.target)
        loss_after = error_after * error_after
        return yhat_after, error_after, loss_after

    def _add_before_row(self) -> None:
        if self.action_timer.isActive():
            return
        if self.before_visible_count >= len(self.rows):
            return
        self.timer_mode = "before"
        self.timer_row_index = self.before_visible_count
        self._show_before_timer_card()
        self.action_timer.start()

    def _start_guided_signals(self) -> None:
        if self.action_timer.isActive() or self.before_visible_count < min(self.GUIDED_ROW_COUNT, len(self.rows)):
            return
        if self._guided_signals_complete():
            return
        self.timer_mode = "signals_guided"
        self.timer_row_index = 0
        self.timer_field_index = 0
        self._show_signal_timer_card()
        self.action_timer.start()

    def _start_complete_signals(self) -> None:
        if self.action_timer.isActive() or not self._guided_signals_complete():
            return
        if self._all_signals_complete():
            return
        self.timer_mode = "signals_complete"
        self.timer_row_index = self.GUIDED_ROW_COUNT
        self.timer_field_index = 0
        self._show_signal_timer_card()
        self.action_timer.start()

    def _average_signals(self) -> None:
        if self.action_timer.isActive() or not self._all_signals_complete():
            return
        w_signals = [self._w_signal(row) for row in self.rows]
        b_signals = [self._b_signal(row) for row in self.rows]
        self.average_w_signal = sum(w_signals) / len(w_signals)
        self.average_b_signal = sum(b_signals) / len(b_signals)
        self.averages_calculated = True
        self.update_card.setHtml(self._average_card_html(w_signals, b_signals))
        self._refresh_everything()

    def _update_w_and_b(self) -> None:
        if self.action_timer.isActive() or not self.averages_calculated:
            return
        self.new_w_value = self.START_W_VALUE - self.ETA_VALUE * self.average_w_signal
        self.new_b_value = self.START_B_VALUE - self.ETA_VALUE * self.average_b_signal
        self.parameters_updated = True
        self.update_card.setHtml(self._eta_update_card_html())
        self._refresh_everything()

    def _start_guided_after_update(self) -> None:
        if self.action_timer.isActive() or not self.parameters_updated:
            return
        if self._guided_after_complete():
            return
        self.timer_mode = "after_guided"
        self.timer_row_index = 0
        self.timer_field_index = 0
        self._show_after_timer_card()
        self.action_timer.start()

    def _start_complete_after_table(self) -> None:
        if self.action_timer.isActive() or not self._guided_after_complete():
            return
        if self._all_after_complete():
            return
        self.timer_mode = "after_complete"
        self.timer_row_index = self.GUIDED_ROW_COUNT
        self.timer_field_index = 0
        self._show_after_timer_card()
        self.action_timer.start()

    def _advance_timer_action(self) -> None:
        if self.timer_mode == "before":
            self.before_visible_count = min(len(self.rows), self.before_visible_count + 1)
            self._refresh_everything()
            self.timer_row_index = self.before_visible_count
            if self.before_visible_count >= len(self.rows):
                self.action_timer.stop()
                self.timer_mode = ""
                self._refresh_buttons()
                self.before_card.setHtml(
                    "<h3>Earlier calculations complete</h3>"
                    "<p>The table now shows the h<sub>4</sub>, prediction, error, and loss values that will feed the training step.</p>"
                    "<p>Next, calculate the w and b update signals in the lower table.</p>"
                )
            else:
                self._show_before_timer_card()
            return

        if self.timer_mode in ["signals_guided", "signals_complete"]:
            self._fill_current_signal_field()
            self._refresh_everything()
            self._advance_signal_pointer()
            if self._signal_timer_finished():
                self.action_timer.stop()
                self.timer_mode = ""
                self._refresh_buttons()
                self.update_card.setHtml(
                    "<h3>Update signals complete</h3>"
                    "<p>The w-signal and b-signal columns are ready to be averaged.</p>"
                    if self._all_signals_complete()
                    else "<h3>First two signal rows complete</h3><p>Click <b>Complete Signals</b> to finish rows 3 through 6.</p>"
                )
            else:
                self._show_signal_timer_card()
            return

        if self.timer_mode in ["after_guided", "after_complete"]:
            self._fill_current_after_field()
            self._refresh_everything()
            self._advance_after_pointer()
            if self._after_timer_finished():
                self.action_timer.stop()
                self.timer_mode = ""
                self._refresh_buttons()
                self.update_card.setHtml(
                    "<h3>After-update check complete</h3>"
                    "<p>The table now shows how one training step changed the predictions, errors, and losses.</p>"
                    if self._all_after_complete()
                    else "<h3>First two after-update rows complete</h3><p>Click <b>Complete After Table</b> to finish rows 3 through 6.</p>"
                )
            else:
                self._show_after_timer_card()
            return

    def _show_before_timer_card(self) -> None:
        if 0 <= self.timer_row_index < len(self.rows):
            self.before_card.setHtml(self._before_card_html(self.rows[self.timer_row_index], is_fill=False))

    def _show_signal_timer_card(self) -> None:
        if 0 <= self.timer_row_index < len(self.rows):
            field = "w" if self.timer_field_index == 0 else "b"
            self.update_card.setHtml(self._signal_card_html(self.rows[self.timer_row_index], field))

    def _show_after_timer_card(self) -> None:
        if 0 <= self.timer_row_index < len(self.rows):
            fields = ["prediction", "error", "loss"]
            field = fields[self.timer_field_index]
            self.update_card.setHtml(self._after_card_html(self.rows[self.timer_row_index], field))

    def _fill_current_signal_field(self) -> None:
        if 0 <= self.timer_row_index < len(self.signal_visible):
            self.signal_visible[self.timer_row_index][self.timer_field_index] = True

    def _advance_signal_pointer(self) -> None:
        if self.timer_field_index == 0:
            self.timer_field_index = 1
        else:
            self.timer_field_index = 0
            self.timer_row_index += 1

    def _signal_timer_finished(self) -> bool:
        end_row = self.GUIDED_ROW_COUNT if self.timer_mode == "signals_guided" else len(self.rows)
        return self.timer_row_index >= end_row

    def _fill_current_after_field(self) -> None:
        if 0 <= self.timer_row_index < len(self.after_visible):
            self.after_visible[self.timer_row_index][self.timer_field_index] = True

    def _advance_after_pointer(self) -> None:
        if self.timer_field_index < 2:
            self.timer_field_index += 1
        else:
            self.timer_field_index = 0
            self.timer_row_index += 1

    def _after_timer_finished(self) -> bool:
        end_row = self.GUIDED_ROW_COUNT if self.timer_mode == "after_guided" else len(self.rows)
        return self.timer_row_index >= end_row

    def _refresh_everything(self) -> None:
        self._refresh_settings_strip()
        self._refresh_buttons()
        self._refresh_before_table()
        self._refresh_update_table()
        self._refresh_default_cards_if_needed()

    def _refresh_settings_strip(self) -> None:
        self.settings_label.setText(
            "Fixed memory controls: f = 0.60, i = 0.40, o = 0.90    |    "
            "Starting output layer: w = 0.80, b = 0.10    |    "
            "Learning rate: η = 0.10"
        )

        if self.parameters_updated:
            self.summary_label.setText(
                f"Averages: w signal = {fmt_decimal(self.average_w_signal)}, "
                f"b signal = {fmt_decimal(self.average_b_signal)}    |    "
                f"Updated output layer: w = {fmt_decimal(self.new_w_value)}, "
                f"b = {fmt_decimal(self.new_b_value)}"
            )
        elif self.averages_calculated:
            self.summary_label.setText(
                f"Averages: w signal = {fmt_decimal(self.average_w_signal)}, "
                f"b signal = {fmt_decimal(self.average_b_signal)}    |    "
                "Next: use η to update w and b."
            )
        else:
            self.summary_label.setText("Averages and updated parameters will appear after the signal columns are complete.")

    def _refresh_buttons(self) -> None:
        busy = self.action_timer.isActive()
        before_done = self.before_visible_count >= len(self.rows)

        self.add_before_button.setEnabled(not busy and not before_done)

        self.calculate_signals_button.setEnabled(not busy and before_done and not self._guided_signals_complete())
        self.complete_signals_button.setEnabled(not busy and before_done and self._guided_signals_complete() and not self._all_signals_complete())
        self.average_signals_button.setEnabled(not busy and before_done and self._all_signals_complete() and not self.averages_calculated)
        self.update_parameters_button.setEnabled(not busy and before_done and self.averages_calculated and not self.parameters_updated)
        self.check_after_button.setEnabled(not busy and before_done and self.parameters_updated and not self._guided_after_complete())
        self.complete_after_button.setEnabled(not busy and before_done and self._guided_after_complete() and not self._all_after_complete())

    def _refresh_before_table(self) -> None:
        self.before_table.clear()
        self.before_table.setColumnCount(6)
        self.before_table.setHorizontalHeaderLabels(
            ["Window", "h_4", "y", "ŷ before", "e before", "loss before"]
        )
        self.before_table.verticalHeader().setVisible(False)
        self.before_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for col_index, width in enumerate([70, 80, 70, 105, 95, 110]):
            self.before_table.setColumnWidth(col_index, width)
        self.before_table.setRowCount(self.before_visible_count)

        for row_index, row in enumerate(self.rows[: self.before_visible_count]):
            self.before_table.setItem(row_index, 0, make_table_item(row.window_number))
            self.before_table.setItem(row_index, 1, make_maroon_bold_table_item(fmt_decimal(row.final_h)))
            self.before_table.setItem(row_index, 2, make_target_item(trend_text(row.target), is_current=False))
            self.before_table.setItem(row_index, 3, make_table_item(fmt_decimal(row.yhat)))
            self.before_table.setItem(row_index, 4, make_table_item(fmt_decimal(row.error)))
            self.before_table.setItem(row_index, 5, make_table_item(fmt_decimal(row.loss)))
        self.before_table.resizeRowsToContents()

    def _refresh_update_table(self) -> None:
        self.update_table.clear()
        self.update_table.setColumnCount(6)
        self.update_table.setHorizontalHeaderLabels(
            ["Window", "w signal", "b signal", "ŷ after", "e after", "loss after"]
        )
        self.update_table.verticalHeader().setVisible(False)
        self.update_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for col_index, width in enumerate([70, 105, 105, 105, 95, 105]):
            self.update_table.setColumnWidth(col_index, width)
        self.update_table.setRowCount(len(self.rows))

        for row_index, row in enumerate(self.rows):
            self.update_table.setItem(row_index, 0, make_table_item(row.window_number))
            if self.signal_visible[row_index][0]:
                self.update_table.setItem(row_index, 1, make_table_item(fmt_decimal(self._w_signal(row))))
            else:
                self.update_table.setItem(row_index, 1, make_table_item(""))
            if self.signal_visible[row_index][1]:
                self.update_table.setItem(row_index, 2, make_table_item(fmt_decimal(self._b_signal(row))))
            else:
                self.update_table.setItem(row_index, 2, make_table_item(""))

            yhat_after, error_after, loss_after = self._after_values(row)
            if self.after_visible[row_index][0]:
                self.update_table.setItem(row_index, 3, make_table_item(fmt_decimal(yhat_after)))
            else:
                self.update_table.setItem(row_index, 3, make_table_item(""))
            if self.after_visible[row_index][1]:
                self.update_table.setItem(row_index, 4, make_table_item(fmt_decimal(error_after)))
            else:
                self.update_table.setItem(row_index, 4, make_table_item(""))
            if self.after_visible[row_index][2]:
                self.update_table.setItem(row_index, 5, make_table_item(fmt_decimal(loss_after)))
            else:
                self.update_table.setItem(row_index, 5, make_table_item(""))
        self.update_table.resizeRowsToContents()

    def _refresh_default_cards_if_needed(self) -> None:
        if not self.before_card.toPlainText().strip():
            self.before_card.setHtml(
                "<h3>Ready</h3>"
                "<p>Click <b>Add Earlier Calculations</b>. The table will copy the Tab 4 values one row at a time.</p>"
                "<p>The next table uses <b>h<sub>4</sub></b> and <b>e before</b> to calculate the w and b update signals.</p>"
            )
        if not self.update_card.toPlainText().strip():
            self.update_card.setHtml(
                "<h3>Ready</h3>"
                "<p>After the earlier calculations are loaded, calculate the update signals for w and b.</p>"
            )

    def _before_card_html(self, row: PredictionRow, is_fill: bool) -> str:
        note = "<p><b>Now placing this row in the earlier-calculations table.</b></p>" if is_fill else ""
        return (
            f"<h3>Copying Window {row.window_number}</h3>"
            "<div style='font-size: 14px;'>"
            "This table brings forward the values already calculated in Tab 4.<br><br>"
            f"The next table will use <b>h<sub>4</sub> = {fmt_decimal(row.final_h)}</b> and "
            f"<b>e before = {fmt_decimal(row.error)}</b> to build update signals.<br><br>"
            "w signal = 2 &middot; e &middot; h<sub>4</sub><br>"
            "b signal = 2 &middot; e"
            "</div>"
            f"{note}"
        )

    def _signal_card_html(self, row: PredictionRow, field: str) -> str:
        if field == "w":
            return (
                f"<h3>w signal for Window {row.window_number}</h3>"
                "<div style='font-size: 14px;'>"
                "w signal = 2 &middot; e &middot; h<sub>4</sub><br>"
                f"w signal = 2 &middot; {fmt_decimal(row.error)} &middot; {fmt_decimal(row.final_h)}<br>"
                f"w signal = <span style='color:{COBBER_MAROON}; font-weight:bold;'>"
                f"{fmt_decimal(self._w_signal(row))}</span>"
                "</div>"
            )
        return (
            f"<h3>b signal for Window {row.window_number}</h3>"
            "<div style='font-size: 14px;'>"
            "b signal = 2 &middot; e<br>"
            f"b signal = 2 &middot; {fmt_decimal(row.error)}<br>"
            f"b signal = <span style='color:{COBBER_MAROON}; font-weight:bold;'>"
            f"{fmt_decimal(self._b_signal(row))}</span>"
            "</div>"
        )

    def _average_card_html(self, w_signals: List[float], b_signals: List[float]) -> str:
        w_list = " + ".join(fmt_decimal(value) for value in w_signals)
        b_list = " + ".join(fmt_decimal(value) for value in b_signals)
        n = len(w_signals)
        return (
            "<h3>Average update signals</h3>"
            "<div style='font-size: 14px;'>"
            "The book used one row, so no average was needed. Here the app uses all six training windows.<br><br>"
            f"average w signal = ({w_list}) / {n}<br>"
            f"average w signal = <b>{fmt_decimal(self.average_w_signal)}</b><br><br>"
            f"average b signal = ({b_list}) / {n}<br>"
            f"average b signal = <b>{fmt_decimal(self.average_b_signal)}</b>"
            "</div>"
        )

    def _eta_update_card_html(self) -> str:
        return (
            "<h3>Use η to update w and b</h3>"
            "<div style='font-size: 14px;'>"
            f"η = {self.ETA_VALUE:.2f}<br><br>"
            "new w = old w - η &middot; average w signal<br>"
            f"new w = {self.START_W_VALUE:.3f} - {self.ETA_VALUE:.3f} &middot; {fmt_decimal(self.average_w_signal)}<br>"
            f"new w = <span style='color:{COBBER_MAROON}; font-weight:bold;'>"
            f"{fmt_decimal(self.new_w_value)}</span><br><br>"
            "new b = old b - η &middot; average b signal<br>"
            f"new b = {self.START_B_VALUE:.3f} - {self.ETA_VALUE:.3f} &middot; {fmt_decimal(self.average_b_signal)}<br>"
            f"new b = <span style='color:{COBBER_MAROON}; font-weight:bold;'>"
            f"{fmt_decimal(self.new_b_value)}</span>"
            "</div>"
        )

    def _after_card_html(self, row: PredictionRow, field: str) -> str:
        yhat_after, error_after, loss_after = self._after_values(row)
        if field == "prediction":
            weighted = self.new_w_value * row.final_h
            return (
                f"<h3>After-update prediction for Window {row.window_number}</h3>"
                "<div style='font-size: 14px;'>"
                "ŷ after = new b + new w &middot; h<sub>4</sub><br>"
                f"ŷ after = {fmt_decimal(self.new_b_value)} + {fmt_decimal(self.new_w_value)} &middot; {fmt_decimal(row.final_h)}<br>"
                f"ŷ after = {fmt_decimal(self.new_b_value)} + {fmt_decimal(weighted)} = "
                f"<span style='color:{COBBER_MAROON}; font-weight:bold;'>{fmt_decimal(yhat_after)}</span>"
                "</div>"
            )
        if field == "error":
            return (
                f"<h3>After-update error for Window {row.window_number}</h3>"
                "<div style='font-size: 14px;'>"
                "e after = ŷ after - y<br>"
                f"e after = {fmt_decimal(yhat_after)} - ({trend_text(row.target)})<br>"
                f"e after = <span style='color:{COBBER_MAROON}; font-weight:bold;'>"
                f"{fmt_decimal(error_after)}</span>"
                "</div>"
            )
        return (
            f"<h3>After-update loss for Window {row.window_number}</h3>"
            "<div style='font-size: 14px;'>"
            "loss after = (e after)<sup>2</sup><br>"
            f"loss after = ({fmt_decimal(error_after)})<sup>2</sup><br>"
            f"loss after = <span style='color:{COBBER_MAROON}; font-weight:bold;'>"
            f"{fmt_decimal(loss_after)}</span>"
            "</div>"
        )

# ============================================================
# Main app window
# ============================================================

class CobberEcoLSTMguidedApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1250, 760)
        self.setFont(QFont("Lato", 10))

        tabs = QTabWidget()
        tabs.addTab(BuildWindowsTab(), "1. Build Windows")
        tabs.addTab(WatchMemoryTab(), "2. Watch Memory")
        tabs.addTab(ExposeMemoryTab(), "3. Expose Memory")

        tabs.addTab(CheckPredictionTab(), "4. Make Prediction")

        tabs.addTab(OneTrainingStepTab(), "5. One Training Step")

        placeholder_6 = QLabel("Tab 6 will show repeated training and a loss graph.")
        placeholder_6.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tabs.addTab(placeholder_6, "6. Continue Training")

        self.setCentralWidget(tabs)


# ============================================================
# Styles
# ============================================================

def apply_app_stylesheet(app: QApplication) -> None:
    app.setStyleSheet(
        """
        QWidget {
            color: #222222;
            background-color: #ffffff;
        }

        QTabWidget::pane {
            border: 1px solid #cccccc;
        }

        QTabBar::tab {
            padding: 8px 16px;
            min-width: 130px;
            font-weight: bold;
            background: #e6e6e6;
            color: #222222;
        }

        QTabBar::tab:selected {
            background: #6c1d45;
            color: #ffffff;
        }

        QFrame#SidePanel {
            background-color: #fafafa;
            border: 1px solid #d6d6d6;
            border-radius: 8px;
        }

        QLabel#MainHeader {
            color: #6c1d45;
            font-size: 24px;
            font-weight: bold;
        }

        QLabel#IntroText {
            font-size: 13px;
        }

        QLabel#SmallNote {
            color: #444444;
            font-size: 12px;
            padding-top: 4px;
        }

        QLabel#StatusLabel {
            color: #6c1d45;
            font-weight: bold;
            padding-top: 4px;
        }

        QGroupBox {
            color: #6c1d45;
            font-size: 15px;
            font-weight: bold;
            border: 1px solid #d6d6d6;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 12px;
            background-color: #fafafa;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px 0 4px;
            background-color: #fafafa;
        }

        QPushButton {
            background-color: #f7f7f7;
            color: #111111;
            border: 1px solid #9a9a9a;
            border-radius: 5px;
            padding: 7px 10px;
        }

        QPushButton:hover {
            background-color: #eeeeee;
        }

        QPushButton:pressed {
            background-color: #dddddd;
        }

        QPushButton:disabled {
            color: #888888;
            background-color: #f2f2f2;
        }

        QPushButton#PrimaryButton {
            background-color: #6c1d45;
            color: #ffffff;
            font-weight: bold;
            border: 1px solid #6c1d45;
        }

        QPushButton#PrimaryButton:hover {
            background-color: #7f2855;
        }

        QPushButton#StepButton {
            background-color: #666666;
            color: #ffffff;
            font-weight: bold;
            border: 1px solid #555555;
        }

        QPushButton#StepButton:enabled {
            background-color: #6c1d45;
            color: #ffffff;
            font-weight: bold;
            border: 1px solid #6c1d45;
        }

        QPushButton#StepButton:disabled {
            background-color: #9a9a9a;
            color: #ffffff;
            font-weight: bold;
            border: 1px solid #888888;
        }

        QComboBox {
            background-color: #ffffff;
            color: #111111;
            border: 1px solid #9a9a9a;
            border-radius: 4px;
            padding: 4px 6px;
            min-height: 24px;
        }

        QTextEdit,
        QTableWidget {
            background-color: #ffffff;
            color: #111111;
            border: 1px solid #cccccc;
        }

        QTextEdit#CalcCard,
        QTextEdit#RuleCard {
            border: 1px solid #d6d6d6;
            border-radius: 6px;
            background-color: #ffffff;
        }

        QHeaderView::section {
            background-color: #eeeeee;
            color: #222222;
            padding: 5px;
            border: 1px solid #cccccc;
            font-weight: bold;
        }

        QSlider::groove:horizontal {
            height: 8px;
            background: #d6d6d6;
            border-radius: 4px;
        }

        QSlider::handle:horizontal {
            background: #6c1d45;
            border: 1px solid #6c1d45;
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }

        QSlider::sub-page:horizontal {
            background: #6c1d45;
            border-radius: 4px;
        }
        """
    )


# ============================================================
# Run app
# ============================================================

def main() -> None:
    app = QApplication(sys.argv)
    apply_app_stylesheet(app)
    window = CobberEcoLSTMguidedApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

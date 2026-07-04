# CobberEcoRL_v9.py
# A PyQt6 application for exploring reinforcement learning in a simplified
# ecological management landscape.
#
# Adapted from CobberTarPit for the Ecology machine learning book.
#
# Core idea:
#   Tab 1 shows how a Q-table learns in a fixed or randomized wetland landscape.
#   Tab 2 gives students a separate trained Q-table and asks them to read the policy.
#
# Dependencies:
#   pip install PyQt6 numpy
#
# Run:
#   python CobberEcoRL_v9.py

import sys
import random
from pathlib import Path

import numpy as np

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QDialog,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


# -----------------------------------------------------------------------------
# Brand colors
# -----------------------------------------------------------------------------
COBBER_MAROON = "#6c1d45"
COBBER_MAROON_DARK = "#4c1230"
COBBER_MAROON_LIGHT = "#e5c6d5"
INFOBLUE = "#3e6990"
INFOBLUE_LIGHT = "#d7e8f4"
PROJECTGREEN_LIGHT = "#d9f5d8"
CHARCOAL = "#333333"
MEDIUM_DARK_GREY = "#5f6368"
LIGHT_GREY = "#eeeeee"
BORDER_GREY = "#555555"
WARNING_LIGHT = "#f5e9ef"


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_TO_INDEX = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
INDEX_TO_ACTION = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
ACTION_ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}


def next_state_from_action(state: int, action: int) -> int:
    """
    Grid is 4 x 4, states numbered:

        S0  S1  S2  S3
        S4  S5  S6  S7
        S8  S9  S10 S11
        S12 S13 S14 S15

    Actions:
        0 = UP
        1 = DOWN
        2 = LEFT
        3 = RIGHT
    """
    if action == 0:
        return state - 4 if state > 3 else state
    if action == 1:
        return state + 4 if state < 12 else state
    if action == 2:
        return state - 1 if state % 4 != 0 else state
    return state + 1 if state % 4 != 3 else state


def q_value_color(q_val: float) -> tuple[str, str]:
    """Return gentle Q-table colors. Keep numbers black for readability."""
    if q_val > 0.01:
        strength = min(0.75, q_val / 140.0)
        # Blend white toward light infoblue.
        r0, g0, b0 = 255, 255, 255
        r1, g1, b1 = 150, 181, 204
        r = int(r0 + (r1 - r0) * strength)
        g = int(g0 + (g1 - g0) * strength)
        b = int(b0 + (b1 - b0) * strength)
        return f"#{r:02x}{g:02x}{b:02x}", "#111111"

    if q_val < -0.01:
        strength = min(0.75, abs(q_val) / 140.0)
        # Blend white toward light cobbermaroon.
        r0, g0, b0 = 255, 255, 255
        r1, g1, b1 = 218, 180, 199
        r = int(r0 + (r1 - r0) * strength)
        g = int(g0 + (g1 - g0) * strength)
        b = int(b0 + (b1 - b0) * strength)
        return f"#{r:02x}{g:02x}{b:02x}", "#111111"

    return "#ffffff", "#111111"


class AutoTrainerWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    q_table_updated = pyqtSignal(np.ndarray)
    animation_step = pyqtSignal(int)
    episode_finished = pyqtSignal(bool)
    update_reward = pyqtSignal(float)

    def __init__(
        self,
        q_table: np.ndarray,
        rewards: np.ndarray,
        gamma: float,
        start_pos: int,
        goal_pos: int,
        severe_invasion_pos: int,
        max_steps: int = 100,
        epsilon: float = 0.1,
        num_episodes: int = 100,
    ):
        super().__init__()
        self.q_table = q_table.copy()
        self.rewards = rewards
        self.gamma = gamma
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.severe_invasion_pos = severe_invasion_pos
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.num_actions = 4
        self._is_running = True

    def run(self):
        episode_count = 0

        while self._is_running and episode_count < self.num_episodes:
            episode_count += 1
            state = self.start_pos

            episode_reward = 0.0
            self.update_reward.emit(episode_reward)
            self.animation_step.emit(state)
            QThread.msleep(60)

            for _ in range(self.max_steps):
                if not self._is_running:
                    break

                # Epsilon-greedy action selection:
                # sometimes explore randomly, otherwise choose the best known action.
                if random.uniform(0, 1) < self.epsilon:
                    action = random.randint(0, self.num_actions - 1)
                else:
                    action = int(np.argmax(self.q_table[state, :]))

                s_prime = next_state_from_action(state, action)
                reward = self.rewards[s_prime]
                episode_reward += reward
                self.update_reward.emit(episode_reward)

                self.animation_step.emit(s_prime)
                QThread.msleep(60)

                next_max = np.max(self.q_table[s_prime, :])

                # Simplified Q-learning update used for this introductory activity.
                new_value = reward + self.gamma * next_max
                self.q_table[state, action] = new_value

                state = s_prime

                if state == self.goal_pos or state == self.severe_invasion_pos:
                    break

            if not self._is_running:
                break

            self.progress.emit(episode_count)
            self.q_table_updated.emit(self.q_table.copy())
            self.episode_finished.emit(state == self.goal_pos)
            QThread.msleep(180)

        self.finished.emit()

    def stop(self):
        self._is_running = False


class CobberEcoRLApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CobberEcoRL")
        self.setGeometry(40, 60, 1260, 720)
        self.setMinimumSize(1180, 680)
        self.setFont(QFont("Lato", 10))

        self.num_states = 16
        self.num_actions = 4
        self.gamma = 0.9

        # Tab 1 fixed learning landscape.
        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.cumulative_reward = 0.0
        self.grid_is_hidden = True

        self.start_pos = 12
        self.goal_pos = 3
        self.severe_invasion_pos = 7
        self.agent_pos = self.start_pos

        self.rewards = np.full(self.num_states, -1.0)
        self.rewards[self.goal_pos] = 100.0
        self.rewards[self.severe_invasion_pos] = -100.0

        self.last_state = -1
        self.last_action = -1
        self.s_prime = -1
        self.reward = 0.0
        self.last_future_value = 0.0
        self.policy_visible = False

        # Tab 1 widgets.
        self.grid_labels: list[QLabel] = []
        self.q_value_labels: dict[int, list[QLabel]] = {}
        self.q_state_headers: dict[int, QLabel] = {}

        # Animation worker.
        self.worker: AutoTrainerWorker | None = None
        self.thread: QThread | None = None
        self.animation_is_stopping = False

        # Tab 2 unknown wetland state.
        self.unknown_start_pos = 12
        self.unknown_goal_pos = 3
        self.unknown_invasion_pos = 7
        self.unknown_agent_pos = self.unknown_start_pos
        self.unknown_revealed = False
        self.unknown_episode_done = False
        self.unknown_total_reward = 0.0
        self.unknown_moves = 0
        self.unknown_visited: set[int] = set()
        self.unknown_grid_labels: list[QLabel] = []

        # Tab 2 policy-reading widgets and state.
        self.policy_grid_labels: list[QLabel] = []
        self.policy_q_value_labels: dict[int, list[QLabel]] = {}
        self.policy_q_state_headers: dict[int, QLabel] = {}
        self.policy_agent_pos = self.start_pos
        self.policy_q_table = self.q_table.copy()
        self.policy_steps = 0
        self.policy_missteps = 0
        self.policy_start_pos = 12
        self.policy_goal_pos = 3
        self.policy_invasion_pos = 7
        self.policy_rewards = np.full(self.num_states, -1.0)
        self.policy_revealed = False
        self.policy_done = False

        self._setup_ui()
        self.full_reset()
        self.create_policy_challenge()

    # -------------------------------------------------------------------------
    # UI setup
    # -------------------------------------------------------------------------
    def _setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        outer_layout = QVBoxLayout(main_widget)
        outer_layout.setContentsMargins(10, 8, 10, 10)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_learning_tab(), "How the Q-table Learns")
        self.tabs.addTab(self._build_policy_tab(), "Follow the Policy")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        outer_layout.addWidget(self.tabs)

    def _build_learning_tab(self) -> QWidget:
        tab = QWidget()
        main_layout = QHBoxLayout(tab)
        main_layout.setContentsMargins(8, 10, 8, 8)
        main_layout.setSpacing(12)

        controls_column = QVBoxLayout()
        controls_column.addWidget(self._build_controls_panel(), 4)
        controls_column.addWidget(self._build_update_panel(), 2)

        grid_groupbox = self._build_grid_panel()
        q_table_groupbox = self._build_q_table_panel()

        main_layout.addLayout(controls_column, 1)
        main_layout.addWidget(grid_groupbox, 1)
        main_layout.addWidget(q_table_groupbox, 1)

        return tab

    def _build_controls_panel(self) -> QGroupBox:
        controls_groupbox = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_groupbox)
        controls_layout.setSpacing(8)

        # Hidden/internal status label. The visible feedback now lives in the
        # Current Update panel, which avoids duplicating messages.
        self.status_label = QLabel("")
        self.status_label.hide()

        reward_layout = QHBoxLayout()
        reward_text = QLabel("Current Episode Reward:")
        reward_text.setFont(QFont("Lato", 10, QFont.Weight.Bold))
        self.reward_label = QLabel("0.00")
        self.reward_label.setObjectName("RewardLabel")
        self.reward_label.setFont(QFont("Lato", 18, QFont.Weight.Bold))
        reward_layout.addWidget(reward_text)
        reward_layout.addStretch()
        reward_layout.addWidget(self.reward_label)
        controls_layout.addLayout(reward_layout)

        instruction_label = QLabel(
            "Click a neighboring patch in the landscape to move the field crew. "
            "Then update the Q-value for that move."
        )
        instruction_label.setObjectName("TeachingNote")
        instruction_label.setWordWrap(True)
        controls_layout.addWidget(instruction_label)
        # Reuse this compact note as the visible feedback area. This avoids the
        # large redundant status box but keeps error/training messages visible.
        self.status_label = instruction_label

        # Hidden/internal widgets kept so older methods remain safe.
        self.action_combo = QComboBox()
        self.action_combo.addItems(ACTIONS)
        self.action_combo.hide()
        self.execute_button = QPushButton("Execute Action")
        self.execute_button.hide()

        self.update_q_button = QPushButton("Update Q-value")
        self.reset_button = QPushButton("Start a Fresh Episode")
        self.reveal_button = QPushButton("Reveal Management Landscape")
        self.random_landscape_button = QPushButton("New Reward Landscape")
        self.random_landscape_button.setEnabled(False)

        self.train_agent_button = QPushButton("Train Agent for 100 Episodes")
        self.stop_training_button = QPushButton("Stop Training")
        self.show_policy_button = QPushButton("Show Policy")

        # Internal/non-visible controls retained for compatibility.
        self.full_reset_button = QPushButton("Reset Learning")
        self.full_reset_button.hide()
        self.chapter_landscape_button = QPushButton("Use Chapter Landscape")
        self.chapter_landscape_button.hide()
        self.landscape_setup_label = QLabel("")
        self.landscape_setup_label.hide()

        self.execute_button.clicked.connect(self.execute_action)
        self.update_q_button.clicked.connect(self.update_q_value)
        self.reveal_button.clicked.connect(self.reveal_grid)
        self.reset_button.clicked.connect(self.reset_episode)
        self.full_reset_button.clicked.connect(self.full_reset)
        self.train_agent_button.clicked.connect(self.start_animation)
        self.stop_training_button.clicked.connect(self.stop_animation)
        self.show_policy_button.clicked.connect(self.toggle_policy)
        self.chapter_landscape_button.clicked.connect(self.use_chapter_landscape)
        self.random_landscape_button.clicked.connect(self.new_reward_landscape)

        controls_layout.addWidget(self.update_q_button)
        controls_layout.addWidget(self.reset_button)
        controls_layout.addWidget(self.reveal_button)
        controls_layout.addWidget(self.random_landscape_button)
        controls_layout.addStretch()

        return controls_groupbox

    def _build_update_panel(self) -> QGroupBox:
        update_groupbox = QGroupBox("Current Update")
        update_layout = QVBoxLayout(update_groupbox)
        update_layout.setSpacing(6)

        self.update_state_label = QLabel("State: —")
        self.update_action_label = QLabel("Action: —")
        self.update_new_state_label = QLabel("New state: —")
        self.update_reward_label = QLabel("Reward/Penalty: —")
        self.update_future_label = QLabel("Best future Q-value: —")
        self.update_formula_label = QLabel("Update: —")
        self.update_formula_label.setWordWrap(True)
        self.update_formula_label.setObjectName("FormulaLabel")

        for label in [
            self.update_state_label,
            self.update_action_label,
            self.update_new_state_label,
            self.update_reward_label,
            self.update_future_label,
            self.update_formula_label,
        ]:
            label.setWordWrap(True)
            update_layout.addWidget(label)

        return update_groupbox

    def _build_grid_panel(self) -> QGroupBox:
        grid_groupbox = QGroupBox("Management Landscape")
        grid_layout = QGridLayout(grid_groupbox)
        grid_layout.setSpacing(8)
        grid_layout.setContentsMargins(15, 20, 15, 15)

        for i in range(16):
            label = QLabel()
            label.setFixedSize(86, 86)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setFont(QFont("Lato", 16, QFont.Weight.Bold))
            label.setCursor(Qt.CursorShape.PointingHandCursor)
            label.mousePressEvent = lambda event, state=i: self.handle_grid_click(state)
            self.grid_labels.append(label)
            grid_layout.addWidget(label, i // 4, i % 4)

        return grid_groupbox

    def _build_q_table_panel(self) -> QGroupBox:
        q_table_groupbox = QGroupBox("The Manager's Brain (Q-Table)")
        outer_layout = QVBoxLayout(q_table_groupbox)

        q_table_layout = QGridLayout()
        q_table_layout.setHorizontalSpacing(2)
        q_table_layout.setVerticalSpacing(2)

        q_table_layout.addWidget(QLabel(""), 0, 0)

        for c, action_name in enumerate(ACTIONS):
            header = QLabel(action_name)
            header.setFont(QFont("Lato", 10, QFont.Weight.Bold))
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            q_table_layout.addWidget(header, 0, c + 1)

        for r in range(16):
            state_header = QLabel(f"S{r}")
            state_header.setFont(QFont("Lato", 10, QFont.Weight.Bold))
            state_header.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            state_header.setFixedWidth(34)
            self.q_state_headers[r] = state_header
            q_table_layout.addWidget(state_header, r + 1, 0)

            self.q_value_labels[r] = []
            for c in range(4):
                q_label = QLabel("0.00")
                q_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                q_label.setFont(QFont("Lato", 9))
                q_label.setStyleSheet("border:1px solid #cccccc;background-color:white;color:#111111;")
                q_label.setFixedSize(66, 21)
                self.q_value_labels[r].append(q_label)
                q_table_layout.addWidget(q_label, r + 1, c + 1)

        q_table_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignCenter)
        outer_layout.addLayout(q_table_layout)

        training_controls = QGroupBox("Learning and Policy")
        training_layout = QVBoxLayout(training_controls)
        train_buttons = QHBoxLayout()
        train_buttons.addWidget(self.train_agent_button)
        train_buttons.addWidget(self.stop_training_button)
        training_layout.addLayout(train_buttons)
        training_layout.addWidget(self.show_policy_button)
        outer_layout.addWidget(training_controls)
        outer_layout.addStretch()

        return q_table_groupbox

    def _build_policy_tab(self) -> QWidget:
        tab = QWidget()
        main_layout = QHBoxLayout(tab)
        main_layout.setContentsMargins(8, 10, 8, 8)
        main_layout.setSpacing(12)

        main_layout.addWidget(self._build_policy_controls_panel(), 1)
        main_layout.addWidget(self._build_policy_grid_panel(), 1)
        main_layout.addWidget(self._build_policy_q_table_panel(), 1)
        return tab

    def _build_policy_controls_panel(self) -> QGroupBox:
        controls = QGroupBox("Policy Reading")
        layout = QVBoxLayout(controls)
        layout.setSpacing(8)

        self.policy_status_label = QLabel(
            "Read the highlighted row in the Q-table. Click the neighboring patch "
            "that matches the largest Q-value. The native and loosestrife patches "
            "are hidden until the path ends."
        )
        self.policy_status_label.setObjectName("StatusLabel")
        self.policy_status_label.setWordWrap(True)
        self.policy_status_label.setMinimumHeight(125)
        self.policy_status_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.policy_status_label)

        self.policy_steps_label = QLabel("Correct policy moves: 0")
        self.policy_missteps_label = QLabel("Missteps: 0")
        self.policy_current_state_label = QLabel("Current state: —")
        for label in [self.policy_steps_label, self.policy_missteps_label, self.policy_current_state_label]:
            label.setWordWrap(True)
            layout.addWidget(label)

        note = QLabel(
            "This challenge uses a separate reward landscape and a trained Q-table. "
            "Use the table, not the map, to decide where to move."
        )
        note.setObjectName("TeachingNote")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()
        return controls

    def _build_policy_grid_panel(self) -> QGroupBox:
        grid_groupbox = QGroupBox("Policy Landscape")
        grid_layout = QGridLayout(grid_groupbox)
        grid_layout.setSpacing(8)
        grid_layout.setContentsMargins(15, 20, 15, 15)

        for i in range(16):
            label = QLabel()
            label.setFixedSize(86, 86)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setFont(QFont("Lato", 16, QFont.Weight.Bold))
            label.setCursor(Qt.CursorShape.PointingHandCursor)
            label.mousePressEvent = lambda event, state=i: self.handle_policy_grid_click(state)
            self.policy_grid_labels.append(label)
            grid_layout.addWidget(label, i // 4, i % 4)
        return grid_groupbox

    def _build_policy_q_table_panel(self) -> QGroupBox:
        q_groupbox = QGroupBox("Read the Q-Table")
        outer_layout = QVBoxLayout(q_groupbox)
        q_layout = QGridLayout()
        q_layout.setHorizontalSpacing(2)
        q_layout.setVerticalSpacing(2)
        q_layout.addWidget(QLabel(""), 0, 0)

        for c, action_name in enumerate(ACTIONS):
            header = QLabel(action_name)
            header.setFont(QFont("Lato", 10, QFont.Weight.Bold))
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            q_layout.addWidget(header, 0, c + 1)

        for r in range(16):
            state_header = QLabel(f"S{r}")
            state_header.setFont(QFont("Lato", 10, QFont.Weight.Bold))
            state_header.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            state_header.setFixedWidth(34)
            self.policy_q_state_headers[r] = state_header
            q_layout.addWidget(state_header, r + 1, 0)
            self.policy_q_value_labels[r] = []
            for c in range(4):
                q_label = QLabel("0.00")
                q_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                q_label.setFont(QFont("Lato", 9))
                q_label.setFixedSize(66, 21)
                self.policy_q_value_labels[r].append(q_label)
                q_layout.addWidget(q_label, r + 1, c + 1)

        q_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignCenter)
        outer_layout.addLayout(q_layout)
        outer_layout.addStretch()
        return q_groupbox

    def _build_unknown_tab(self) -> QWidget:
        tab = QWidget()
        main_layout = QHBoxLayout(tab)
        main_layout.setContentsMargins(8, 10, 8, 8)
        main_layout.setSpacing(12)

        main_layout.addWidget(self._build_unknown_controls_panel(), 1)
        main_layout.addWidget(self._build_unknown_grid_panel(), 1)
        main_layout.addWidget(self._build_unknown_summary_panel(), 1)

        return tab

    def _build_unknown_controls_panel(self) -> QGroupBox:
        controls = QGroupBox("Manual Search Controls")
        layout = QVBoxLayout(controls)
        layout.setSpacing(8)

        self.unknown_status_label = QLabel("Explore the hidden wetland.")
        self.unknown_status_label.setObjectName("StatusLabel")
        self.unknown_status_label.setWordWrap(True)
        self.unknown_status_label.setMinimumHeight(110)
        self.unknown_status_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.unknown_status_label)

        move_label = QLabel("Move the field crew:")
        move_label.setFont(QFont("Lato", 10, QFont.Weight.Bold))
        layout.addWidget(move_label)

        move_grid = QGridLayout()
        self.unknown_up_button = QPushButton("Move Up")
        self.unknown_down_button = QPushButton("Move Down")
        self.unknown_left_button = QPushButton("Move Left")
        self.unknown_right_button = QPushButton("Move Right")

        self.unknown_up_button.clicked.connect(lambda: self.unknown_move(0))
        self.unknown_down_button.clicked.connect(lambda: self.unknown_move(1))
        self.unknown_left_button.clicked.connect(lambda: self.unknown_move(2))
        self.unknown_right_button.clicked.connect(lambda: self.unknown_move(3))

        move_grid.addWidget(self.unknown_up_button, 0, 1)
        move_grid.addWidget(self.unknown_left_button, 1, 0)
        move_grid.addWidget(self.unknown_right_button, 1, 2)
        move_grid.addWidget(self.unknown_down_button, 2, 1)
        layout.addLayout(move_grid)

        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator1)

        self.unknown_reset_attempt_button = QPushButton("Reset Attempt")
        self.unknown_reveal_button = QPushButton("Reveal Landscape")
        self.unknown_new_button = QPushButton("New Unknown Wetland")

        self.unknown_reset_attempt_button.clicked.connect(self.reset_unknown_attempt)
        self.unknown_reveal_button.clicked.connect(self.reveal_unknown_landscape)
        self.unknown_new_button.clicked.connect(self.new_unknown_wetland)

        layout.addWidget(self.unknown_reset_attempt_button)
        layout.addWidget(self.unknown_reveal_button)
        layout.addWidget(self.unknown_new_button)

        teaching_note = QLabel(
            "Tab 2 hides the native patch and severe loosestrife source. "
            "There is no Q-table here. Your job is to feel the exploration problem."
        )
        teaching_note.setObjectName("TeachingNote")
        teaching_note.setWordWrap(True)
        layout.addWidget(teaching_note)
        layout.addStretch()

        return controls

    def _build_unknown_grid_panel(self) -> QGroupBox:
        grid_groupbox = QGroupBox("Unknown Wetland")
        grid_layout = QGridLayout(grid_groupbox)
        grid_layout.setSpacing(8)
        grid_layout.setContentsMargins(15, 20, 15, 15)

        for i in range(16):
            label = QLabel()
            label.setFixedSize(86, 86)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setFont(QFont("Lato", 16, QFont.Weight.Bold))
            self.unknown_grid_labels.append(label)
            grid_layout.addWidget(label, i // 4, i % 4)

        return grid_groupbox

    def _build_unknown_summary_panel(self) -> QGroupBox:
        summary = QGroupBox("Attempt Summary")
        layout = QVBoxLayout(summary)
        layout.setSpacing(8)

        self.unknown_reward_label = QLabel("Total reward: 0.00")
        self.unknown_reward_label.setObjectName("RewardLabel")
        self.unknown_reward_label.setFont(QFont("Lato", 18, QFont.Weight.Bold))

        self.unknown_moves_label = QLabel("Moves taken: 0")
        self.unknown_outcome_label = QLabel("Outcome: searching")
        self.unknown_current_state_label = QLabel("Current state: S12")

        for label in [
            self.unknown_reward_label,
            self.unknown_moves_label,
            self.unknown_current_state_label,
            self.unknown_outcome_label,
        ]:
            label.setWordWrap(True)
            layout.addWidget(label)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        self.unknown_notes_label = QLabel(
            "Every ordinary move costs -1. Reaching the hidden native patch gives +100. "
            "Entering the hidden dense loosestrife source gives -100. "
            "After the episode ends, reveal the landscape and compare your route with the hidden layout."
        )
        self.unknown_notes_label.setObjectName("TeachingNote")
        self.unknown_notes_label.setWordWrap(True)
        layout.addWidget(self.unknown_notes_label)
        layout.addStretch()

        return summary

    # -------------------------------------------------------------------------
    # Tab 1: Q-learning landscape
    # -------------------------------------------------------------------------
    def configure_landscape(self, start: int, goal: int, invasion: int, status: str):
        self.stop_animation()
        self.start_pos = start
        self.goal_pos = goal
        self.severe_invasion_pos = invasion
        self.agent_pos = self.start_pos
        self.rewards = np.full(self.num_states, -1.0)
        self.rewards[self.goal_pos] = 100.0
        self.rewards[self.severe_invasion_pos] = -100.0
        self.q_table.fill(0)
        self.policy_visible = False
        self.show_policy_button.setText("Show Policy")
        if hasattr(self, "landscape_setup_label"):
            self.landscape_setup_label.setText("")
        self.update_q_table_ui()
        self.clear_update_panel()
        self.reset_episode()
        self.status_label.setText(status)
        self.update_random_landscape_button_state()

    def use_chapter_landscape(self):
        self.configure_landscape(
            start=12,
            goal=3,
            invasion=7,
            status=(
                "Chapter landscape restored. The Q-table has been cleared so the "
                "agent can learn this landscape from scratch."
            ),
        )

    def grid_distance(self, a: int, b: int) -> int:
        return abs(a // 4 - b // 4) + abs(a % 4 - b % 4)

    def new_reward_landscape(self):
        states = list(range(self.num_states))
        for _ in range(500):
            start = random.choice(states)
            goal = random.choice([s for s in states if s != start])
            invasion = random.choice([s for s in states if s not in {start, goal}])
            if self.grid_distance(start, goal) >= 3 and self.grid_distance(start, invasion) >= 2:
                self.configure_landscape(
                    start=start,
                    goal=goal,
                    invasion=invasion,
                    status=(
                        f"New reward landscape created. Start is S{start}, native patch is S{goal}, "
                        f"and loosestrife source is S{invasion}. The Q-table was cleared."
                    ),
                )
                return

        self.configure_landscape(
            start=12, goal=3, invasion=7,
            status="Could not create a suitable random landscape, so the chapter landscape was restored.",
        )

    def full_reset(self):
        self.stop_animation()
        self.q_table.fill(0)
        self.policy_visible = False
        self.show_policy_button.setText("Show Policy")
        self.update_q_table_ui()
        self.clear_update_panel()
        self.reset_episode()
        self.status_label.setText(
            "New learning session started. The manager's brain has been cleared. "
            "Click a neighboring patch to begin exploring."
        )
        self.update_random_landscape_button_state()

    def reset_episode(self):
        self.grid_is_hidden = True
        self.agent_pos = self.start_pos
        self.cumulative_reward = 0.0

        self.last_state = -1
        self.last_action = -1
        self.s_prime = -1
        self.reward = 0.0
        self.last_future_value = 0.0

        self.update_grid_ui()
        self.update_q_table_ui()
        self.reward_label.setText(f"{self.cumulative_reward:.2f}")
        self.clear_update_panel()

        self.set_controls_enabled(True)
        self.update_q_button.setEnabled(False)
        self.reveal_button.setEnabled(False)
        self.stop_training_button.setEnabled(False)
        self.update_random_landscape_button_state()

        self.status_label.setText(
            f"New episode started. The landscape is hidden. "
            f"The crew begins at S{self.start_pos}. Click a neighboring patch to move."
        )

    def handle_grid_click(self, clicked_state: int):
        if self.thread is not None and self.thread.isRunning():
            self.status_label.setText("Stop training before moving manually.")
            return

        if self.update_q_button.isEnabled() and self.last_state >= 0:
            self.status_label.setText("Update the Q-value for the current move before choosing another patch.")
            return

        if self.agent_pos in {self.goal_pos, self.severe_invasion_pos}:
            self.status_label.setText("This episode has ended. Reset the episode to start again.")
            return

        action = self.action_from_neighbor(self.agent_pos, clicked_state)
        if action is None:
            self.status_label.setText(
                f"S{clicked_state} is not an allowed move from S{self.agent_pos}. "
                "Click a neighboring patch: up, down, left, or right."
            )
            return

        self.execute_action(action)

    def action_from_neighbor(self, state: int, clicked_state: int) -> int | None:
        for action in range(self.num_actions):
            if next_state_from_action(state, action) == clicked_state and clicked_state != state:
                return action
        return None

    def execute_action(self, action: int | None = None):
        state = self.agent_pos
        if action is None:
            action_text = self.action_combo.currentText()
            action = ACTION_TO_INDEX[action_text]
        else:
            action_text = INDEX_TO_ACTION[action]

        s_prime = next_state_from_action(state, action)

        self.agent_pos = s_prime

        reward = self.rewards[s_prime]
        self.cumulative_reward += reward
        self.reward_label.setText(f"{self.cumulative_reward:.2f}")

        self.last_state = state
        self.last_action = action
        self.s_prime = s_prime
        self.reward = reward
        self.last_future_value = float(np.max(self.q_table[s_prime, :]))

        self.update_grid_ui()
        self.update_q_table_ui()
        self.set_update_panel(
            state=state,
            action_text=action_text,
            s_prime=s_prime,
            reward=reward,
            future_value=self.last_future_value,
            new_value=None,
        )

        self.status_label.setText(
            f"Moved from S{state} to S{s_prime}.\n"
            f"Now update the Q-table for Q(S{state}, {action_text})."
        )

        self.update_q_button.setEnabled(True)

    def update_q_value(self):
        state = self.last_state
        action = self.last_action
        s_prime = self.s_prime
        reward = self.reward

        if state < 0 or action < 0 or s_prime < 0:
            self.status_label.setText("Execute an action before updating the Q-value.")
            return

        future_value = float(np.max(self.q_table[s_prime, :]))
        new_q_value = reward + self.gamma * future_value
        self.q_table[state, action] = new_q_value

        self.update_q_table_ui()
        self.update_grid_ui()

        action_text = INDEX_TO_ACTION[action]
        self.set_update_panel(
            state=state,
            action_text=action_text,
            s_prime=s_prime,
            reward=reward,
            future_value=future_value,
            new_value=new_q_value,
        )

        self.status_label.setText(
            f"Q(S{state}, {action_text}) was updated to {new_q_value:.2f}.\n"
            f"The table now records what happened after this move.\n"
            f"Choose the next action, or reset the episode."
        )

        is_terminal = s_prime == self.goal_pos or s_prime == self.severe_invasion_pos

        if is_terminal:
            self.update_q_button.setEnabled(False)
            self.reveal_button.setEnabled(True)

            if s_prime == self.goal_pos:
                self.show_popup(
                    "Native Patch Reached",
                    "The field crew reached the high-priority native wetland patch.",
                    is_success=True,
                )
            else:
                self.show_popup(
                    "Severe Invasion Source",
                    "The crew entered the dense purple loosestrife source.",
                    is_success=False,
                )
        else:
            self.update_q_button.setEnabled(False)

    def set_update_panel(
        self,
        state: int,
        action_text: str,
        s_prime: int,
        reward: float,
        future_value: float,
        new_value: float | None,
    ):
        self.update_state_label.setText(f"State: S{state}")
        self.update_action_label.setText(f"Action: {action_text}")
        self.update_new_state_label.setText(f"New state: S{s_prime}")
        self.update_reward_label.setText(f"Reward/Penalty: {reward:.2f}")
        self.update_future_label.setText(f"Best future Q-value from S{s_prime}: {future_value:.2f}")

        formula = (
            f"Update: Q(S{state}, {action_text}) = "
            f"{reward:.2f} + {self.gamma:.1f} × {future_value:.2f}"
        )
        if new_value is not None:
            formula += f" = {new_value:.2f}"
        self.update_formula_label.setText(formula)

    def clear_update_panel(self):
        self.update_state_label.setText("State: —")
        self.update_action_label.setText("Action: —")
        self.update_new_state_label.setText("New state: —")
        self.update_reward_label.setText("Reward/Penalty: —")
        self.update_future_label.setText("Best future Q-value: —")
        self.update_formula_label.setText("Update: —")

    def start_animation(self):
        # Do not start a second worker while the first one is still shutting down.
        if self.thread is not None and self.thread.isRunning():
            self.status_label.setText("Training is already running.")
            return

        self.animation_is_stopping = False
        self.policy_visible = False
        self.show_policy_button.setText("Show Policy")
        self.set_controls_enabled(False)
        self.stop_training_button.setEnabled(True)
        self.train_agent_button.setEnabled(False)
        self.random_landscape_button.setEnabled(False)
        self.status_label.setText("Training agent. Watch the Q-table fill in.")

        thread = QThread(self)
        worker = AutoTrainerWorker(
            q_table=self.q_table,
            rewards=self.rewards,
            gamma=self.gamma,
            start_pos=self.start_pos,
            goal_pos=self.goal_pos,
            severe_invasion_pos=self.severe_invasion_pos,
            num_episodes=100,
        )

        self.thread = thread
        self.worker = worker
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        worker.finished.connect(self._on_animation_finished)
        worker.progress.connect(self._on_auto_train_progress)
        worker.q_table_updated.connect(self._on_q_table_updated)
        worker.animation_step.connect(self._on_animation_step)
        worker.episode_finished.connect(self._on_episode_finished)
        worker.update_reward.connect(self._on_reward_updated)

        thread.start()

    def stop_animation(self):
        # If a worker is running, ask it to stop and wait for its finished signal.
        # Do not delete or overwrite self.worker here; the worker still lives in
        # its thread until run() exits. Clearing it early can crash PyQt.
        if self.worker is not None and self.thread is not None and self.thread.isRunning():
            self.animation_is_stopping = True
            self.worker.stop()
            self.status_label.setText("Stopping training...")
            self.stop_training_button.setEnabled(False)
            self.train_agent_button.setEnabled(False)
            return

        # Nothing is actively running. This path is used during startup/reset.
        if hasattr(self, "status_label"):
            self.status_label.setText("Training stopped.")
            self.set_controls_enabled(True)
            self.update_q_button.setEnabled(False)
            self.stop_training_button.setEnabled(False)

    def _on_reward_updated(self, reward_value: float):
        self.reward_label.setText(f"{reward_value:.2f}")

    def _on_auto_train_progress(self, episode_num: int):
        self.status_label.setText(
            f"Training agent.\n"
            f"Episodes completed: {episode_num}\n"
            f"Positive values point toward the native patch. "
            f"Negative values point toward costly paths."
        )

    def _on_q_table_updated(self, new_q_table: np.ndarray):
        self.q_table = new_q_table
        self.update_q_table_ui()
        self.update_grid_ui()

    def _on_animation_step(self, position: int):
        self.agent_pos = position
        self.update_grid_ui()
        self.update_q_table_ui()

    def _on_episode_finished(self, is_success: bool):
        # Training now runs for a fixed number of episodes. Suppress popups during
        # training so students can watch the Q-table fill without interruption.
        return

    def _on_animation_finished(self):
        if self.animation_is_stopping:
            self.status_label.setText("Training stopped.")
        else:
            self.status_label.setText("Training finished.")

        self.animation_is_stopping = False
        self.set_controls_enabled(True)
        self.update_q_button.setEnabled(False)
        self.stop_training_button.setEnabled(False)
        self.reward_label.setText("0.00")
        self.thread = None
        self.worker = None
        self.update_random_landscape_button_state()

    def set_controls_enabled(self, enabled: bool):
        self.execute_button.setEnabled(enabled)
        self.update_q_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.reveal_button.setEnabled(enabled)
        self.train_agent_button.setEnabled(enabled)
        self.stop_training_button.setEnabled(not enabled)
        self.show_policy_button.setEnabled(enabled)
        self.action_combo.setEnabled(enabled)
        self.update_random_landscape_button_state()

    def reveal_grid(self):
        self.grid_is_hidden = False
        self.update_grid_ui()
        self.reveal_button.setEnabled(False)
        self.status_label.setText(
            f"The landscape is revealed. Start is S{self.start_pos}, the native patch is S{self.goal_pos}, "
            f"and the dense loosestrife source is S{self.severe_invasion_pos}."
        )
        self.update_random_landscape_button_state()

    def toggle_policy(self):
        self.policy_visible = not self.policy_visible
        self.show_policy_button.setText("Hide Policy" if self.policy_visible else "Show Policy")
        self.grid_is_hidden = False
        self.update_grid_ui()
        self.status_label.setText(
            "Policy shown. Each arrow points to the action with the largest Q-value from that state."
            if self.policy_visible
            else "Policy hidden. The management landscape view is restored."
        )
        self.update_random_landscape_button_state()


    def get_popup_image_path(self, is_success: bool) -> str | None:
        """Return a popup image path if one is available beside this script.

        The first names let you use simple app-file names. The later names match
        the draft/generated images we have been testing in this chat.
        """
        base_dir = Path(__file__).resolve().parent

        if is_success:
            candidates = [
                base_dir / "win1.png",
                base_dir / "win2.png",
                base_dir / "success1.png",
                base_dir / "success2.png",
                base_dir / "field_researcher_in_vibrant_wetland_meadow.png",
                base_dir / "summer_meadow_and_stream_oasis.png",
                base_dir / "summer_wildflower_meadow_by_the_pond.png",
            ]
        else:
            candidates = [
                base_dir / "loss1.png",
                base_dir / "loss2.png",
                base_dir / "failure1.png",
                base_dir / "failure2.png",
                base_dir / "wildflower_marsh_with_heron_stand.png",
                base_dir / "contemplative_stroll_through_a_wetland.png",
            ]

        existing = [p for p in candidates if p.exists()]
        if not existing:
            return None
        return str(random.choice(existing))

    def show_popup(self, title: str, message: str, is_success: bool) -> None:
        """Show a safe custom image popup.

        QMessageBox icon pixmaps were brittle in this app and could crash when a
        terminal state was reached. This dialog uses ordinary QLabel widgets for
        the image and text, which is more stable across platforms.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)
        dialog.setMinimumWidth(420)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFont(QFont("Lato", 16, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {COBBER_MAROON}; background-color: transparent;")
        layout.addWidget(title_label)

        image_path = self.get_popup_image_path(is_success)
        if image_path:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                image_label = QLabel()
                image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                image_label.setPixmap(
                    pixmap.scaled(
                        320,
                        320,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                layout.addWidget(image_label)

        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message_label.setFont(QFont("Lato", 11))
        message_label.setStyleSheet("color: #222222; background-color: transparent;")
        layout.addWidget(message_label)

        button = QPushButton("Continue")
        button.clicked.connect(dialog.accept)
        layout.addWidget(button)

        dialog.setStyleSheet(
            f"""
            QDialog {{
                background-color: #ffffff;
                font-family: Lato;
            }}
            QPushButton {{
                background-color: {COBBER_MAROON};
                color: #ffffff;
                font-weight: bold;
                border: 1px solid {COBBER_MAROON_DARK};
                border-radius: 5px;
                padding: 8px 14px;
                min-height: 24px;
            }}
            QPushButton:hover {{ background-color: {COBBER_MAROON_DARK}; }}
            """
        )
        dialog.exec()

    def update_random_landscape_button_state(self):
        can_randomize = (
            hasattr(self, "random_landscape_button")
            and self.policy_visible
            and not self.grid_is_hidden
            and not (self.thread is not None and self.thread.isRunning())
        )
        if hasattr(self, "random_landscape_button"):
            self.random_landscape_button.setEnabled(bool(can_randomize))

    def best_action_for_state(self, state: int) -> int | None:
        row = self.q_table[state, :]
        if np.all(np.abs(row) < 0.01):
            return None
        return int(np.argmax(row))

    def update_grid_ui(self):
        for i, label in enumerate(self.grid_labels):
            label.setText(f"S{i}")
            label.setFont(QFont("Lato", 16, QFont.Weight.Bold))

            border_color = BORDER_GREY
            border_width = 2
            background = LIGHT_GREY
            text_color = "#888888"

            if self.grid_is_hidden:
                background = "#dddddd"
                text_color = "#888888"
            else:
                if i == self.start_pos:
                    background = PROJECTGREEN_LIGHT
                    text_color = "#063b16"
                    label.setText("Start")
                elif i == self.goal_pos:
                    background = INFOBLUE_LIGHT
                    text_color = INFOBLUE
                    label.setText("Native")
                elif i == self.severe_invasion_pos:
                    background = COBBER_MAROON
                    text_color = "#ffffff"
                    label.setText("Loosestrife")
                    label.setFont(QFont("Lato", 11, QFont.Weight.Bold))
                else:
                    background = "#f7f7f7"
                    text_color = "#555555"
                    label.setText("")

                if self.policy_visible and i not in {self.goal_pos, self.severe_invasion_pos}:
                    best_action = self.best_action_for_state(i)
                    if best_action is not None:
                        arrow = ACTION_ARROWS[best_action]
                        if i == self.start_pos:
                            label.setText(f"Start\n{arrow}")
                            label.setFont(QFont("Lato", 28, QFont.Weight.Bold))
                        else:
                            label.setText(arrow)
                            label.setFont(QFont("Lato", 46, QFont.Weight.Bold))
                            background = "#ffffff"
                            text_color = COBBER_MAROON

            if i == self.agent_pos:
                border_color = COBBER_MAROON
                border_width = 4

            label.setStyleSheet(
                f"""
                background-color: {background};
                color: {text_color};
                border: {border_width}px solid {border_color};
                border-radius: 6px;
                """
            )

    def update_q_table_ui(self):
        active_row = self.agent_pos
        for r in range(self.num_states):
            state_header = self.q_state_headers[r]
            if r == active_row:
                state_header.setStyleSheet(
                    f"color: {COBBER_MAROON}; font-weight: bold; background-color: transparent;"
                )
            else:
                state_header.setStyleSheet("color: #222222; font-weight: bold; background-color: transparent;")

            for c in range(self.num_actions):
                q_val = self.q_table[r, c]
                label = self.q_value_labels[r][c]
                label.setText(f"{q_val:.2f}")

                background, foreground = q_value_color(q_val)
                font_weight = "bold" if abs(q_val) > 0.01 else "normal"

                if r == active_row:
                    border = f"2px solid {COBBER_MAROON}"
                else:
                    border = "1px solid #cccccc"

                label.setStyleSheet(
                    f"""
                    background-color: {background};
                    color: {foreground};
                    border: {border};
                    font-weight: {font_weight};
                    """
                )


    # -------------------------------------------------------------------------
    # Tab 2: policy reading
    # -------------------------------------------------------------------------
    def on_tab_changed(self, index: int):
        if self.tabs.tabText(index) == "Follow the Policy" and np.all(np.abs(self.policy_q_table) < 0.01):
            self.create_policy_challenge()

    def choose_policy_challenge_landscape(self) -> tuple[int, int, int]:
        states = list(range(self.num_states))
        for _ in range(1000):
            start = random.choice(states)
            goal = random.choice([s for s in states if s != start])
            invasion = random.choice([s for s in states if s not in {start, goal}])
            if (
                self.grid_distance(start, goal) >= 3
                and self.grid_distance(start, invasion) >= 2
                and self.grid_distance(goal, invasion) >= 2
            ):
                return start, goal, invasion
        return 12, 3, 7

    def value_iteration_q_table(self, rewards: np.ndarray, goal: int, invasion: int, iterations: int = 200) -> np.ndarray:
        q_table = np.zeros((self.num_states, self.num_actions))
        for _ in range(iterations):
            new_q = q_table.copy()
            for state in range(self.num_states):
                if state in {goal, invasion}:
                    new_q[state, :] = 0.0
                    continue
                for action in range(self.num_actions):
                    s_prime = next_state_from_action(state, action)
                    reward = rewards[s_prime]
                    new_q[state, action] = reward + self.gamma * np.max(q_table[s_prime, :])
            q_table = new_q
        return q_table

    def make_policy_reading_q_table(self) -> np.ndarray:
        """Return a hand-shaped Q-table for the Tab 2 policy-reading challenge.

        The values are intentionally less uniform than a fully converged table.
        This makes students read the highlighted row instead of assuming every
        nearby move is equally good. The intended policy path is:

            S15 -> S14 -> S10 -> S6 -> S5 -> S4

        with S4 as the native patch and S11 as the loosestrife source.
        """
        return np.array([
            [54.95, 42.61, -2.71, 48.46],   # S0
            [62.17, 54.95, 48.46, 42.61],   # S1
            [70.19, 62.17, 54.95, -1.90],   # S2
            [62.17, -2.71, 48.46, -1.90],   # S3
            [0.00, 0.00, 0.00, 0.00],       # S4 Native terminal
            [54.95, 48.46, 100.00, 70.19],  # S5 -> LEFT reaches native
            [62.17, 42.61, 89.00, -100.00], # S6 -> LEFT points to S5; RIGHT is bad
            [48.46, -2.71, 62.17, -1.90],   # S7
            [42.61, -1.90, -2.71, 79.10],   # S8
            [54.95, -2.71, 62.17, 70.19],   # S9
            [80.00, 35.00, 20.00, -100.00], # S10 -> UP points to S6; RIGHT is bad
            [0.00, 0.00, 0.00, 0.00],       # S11 Loosestrife terminal
            [-2.71, -1.90, -2.71, 54.95],   # S12
            [35.00, -1.90, 42.61, 62.17],   # S13
            [70.00, -2.71, 52.00, -40.00],  # S14 -> UP points to S10
            [-20.00, -30.00, 60.00, -10.00],# S15 -> LEFT points to S14
        ], dtype=float)

    def create_policy_challenge(self):
        # Fixed policy-reading challenge. This is intentionally different
        # from the chapter landscape used in Tab 1.
        start = 15
        goal = 4
        invasion = 11
        self.policy_start_pos = start
        self.policy_goal_pos = goal
        self.policy_invasion_pos = invasion
        self.policy_agent_pos = start
        self.policy_steps = 0
        self.policy_missteps = 0
        self.policy_revealed = False
        self.policy_done = False
        self.policy_rewards = np.full(self.num_states, -1.0)
        self.policy_rewards[goal] = 100.0
        self.policy_rewards[invasion] = -100.0
        self.policy_q_table = self.make_policy_reading_q_table()
        self.update_policy_grid_ui()
        self.update_policy_q_table_ui()
        self.update_policy_labels()
        self.policy_status_label.setText(
            f"A trained Q-table is ready. Start at S{self.policy_start_pos}. "
            "Read the highlighted row and click the neighboring patch for the largest Q-value."
        )

    def update_policy_labels(self):
        self.policy_steps_label.setText(f"Correct policy moves: {self.policy_steps}")
        self.policy_missteps_label.setText(f"Missteps: {self.policy_missteps}")
        self.policy_current_state_label.setText(f"Current state: S{self.policy_agent_pos}")

    def handle_policy_grid_click(self, clicked_state: int):
        if self.policy_done:
            self.policy_status_label.setText("This policy path has ended. The terminal patches are now revealed.")
            return

        action = self.action_from_neighbor(self.policy_agent_pos, clicked_state)
        if action is None:
            self.policy_status_label.setText(
                f"S{clicked_state} is not a neighboring patch from S{self.policy_agent_pos}."
            )
            return

        best_actions = self.best_policy_actions_for_state(self.policy_agent_pos)
        if not best_actions:
            self.policy_status_label.setText(
                f"The S{self.policy_agent_pos} row has no learned values."
            )
            return

        if action not in best_actions:
            self.policy_missteps += 1
            self.update_policy_labels()
            best_action_text = " or ".join(INDEX_TO_ACTION[a] for a in best_actions)
            self.policy_status_label.setText(
                f"Not quite. From S{self.policy_agent_pos}, the largest Q-value is under "
                f"{best_action_text}. Try that neighboring patch."
            )
            return

        old_state = self.policy_agent_pos
        self.policy_agent_pos = clicked_state
        self.policy_steps += 1

        if self.policy_agent_pos in {self.policy_goal_pos, self.policy_invasion_pos}:
            self.policy_revealed = True
            self.policy_done = True

        self.update_policy_grid_ui()
        self.update_policy_q_table_ui()
        self.update_policy_labels()

        if self.policy_agent_pos == self.policy_goal_pos:
            self.policy_status_label.setText(
                f"Correct. From S{old_state}, {INDEX_TO_ACTION[action]} followed the largest Q-value. "
                "The native patch was reached by reading the policy."
            )
            self.show_popup(
                "Native Patch Reached",
                "You followed the learned policy to the high-priority native wetland patch.",
                is_success=True,
            )
        elif self.policy_agent_pos == self.policy_invasion_pos:
            self.policy_status_label.setText(
                "This policy path entered the dense loosestrife source. Check the highlighted row and Q-values."
            )
            self.show_popup(
                "Severe Invasion Source",
                "The path entered the dense purple loosestrife source.",
                is_success=False,
            )
        else:
            self.policy_status_label.setText(
                f"Correct. From S{old_state}, {INDEX_TO_ACTION[action]} had the largest Q-value. "
                f"Now read the S{self.policy_agent_pos} row."
            )

    def best_policy_actions_for_state(self, state: int) -> list[int]:
        row = self.policy_q_table[state, :]
        if np.all(np.abs(row) < 0.01):
            return []
        best_value = float(np.max(row))
        return [a for a, value in enumerate(row) if abs(float(value) - best_value) < 0.01]

    def best_policy_action_for_state(self, state: int) -> int | None:
        best_actions = self.best_policy_actions_for_state(state)
        return best_actions[0] if best_actions else None

    def update_policy_grid_ui(self):
        for i, label in enumerate(self.policy_grid_labels):
            label.setFont(QFont("Lato", 16, QFont.Weight.Bold))
            border_color = BORDER_GREY
            border_width = 2
            background = "#f7f7f7"
            text_color = "#555555"
            text = f"S{i}"

            if self.policy_revealed:
                if i == self.policy_start_pos:
                    background = PROJECTGREEN_LIGHT
                    text_color = "#063b16"
                    text = "Start"
                elif i == self.policy_goal_pos:
                    background = INFOBLUE_LIGHT
                    text_color = INFOBLUE
                    text = "Native"
                elif i == self.policy_invasion_pos:
                    background = COBBER_MAROON
                    text_color = "#ffffff"
                    text = "Loosestrife"
                    label.setFont(QFont("Lato", 11, QFont.Weight.Bold))

            if i == self.policy_agent_pos:
                border_color = COBBER_MAROON
                border_width = 4
                if not self.policy_revealed:
                    background = PROJECTGREEN_LIGHT
                    text_color = "#063b16"
                    text = f"S{i}"

            label.setText(text)
            label.setStyleSheet(
                f"""
                background-color: {background};
                color: {text_color};
                border: {border_width}px solid {border_color};
                border-radius: 6px;
                """
            )

    def update_policy_q_table_ui(self):
        active_row = self.policy_agent_pos
        for r in range(self.num_states):
            state_header = self.policy_q_state_headers.get(r)
            if state_header is None:
                continue

            if r == active_row:
                state_header.setStyleSheet(
                    f"""
                    color: {COBBER_MAROON};
                    font-weight: bold;
                    background-color: {COBBER_MAROON_LIGHT};
                    border: 3px solid {COBBER_MAROON};
                    """
                )
            else:
                state_header.setStyleSheet(
                    "color: #222222; font-weight: bold; background-color: transparent; border: none;"
                )

            for c in range(self.num_actions):
                q_val = self.policy_q_table[r, c]
                label = self.policy_q_value_labels[r][c]
                label.setText(f"{q_val:.2f}")
                background, foreground = q_value_color(q_val)

                if r == active_row:
                    # Highlight the whole row, not only the best-action cell.
                    border = f"3px solid {COBBER_MAROON}"
                    font_weight = "bold"
                else:
                    border = "1px solid #cccccc"
                    font_weight = "normal"

                label.setStyleSheet(
                    f"""
                    background-color: {background};
                    color: {foreground};
                    border: {border};
                    font-weight: {font_weight};
                    """
                )


def apply_app_stylesheet(app: QApplication) -> None:
    app.setStyleSheet(
        f"""
        QWidget {{
            color: #222222;
            background-color: #ffffff;
            font-family: Lato;
            font-size: 10pt;
        }}
        QMainWindow, QDialog {{
            background-color: #ffffff;
        }}
        QTabWidget::pane {{
            border: 1px solid #d6d6d6;
            background-color: #ffffff;
            top: -1px;
        }}
        QTabBar::tab {{
            background-color: {MEDIUM_DARK_GREY};
            color: #ffffff;
            font-weight: bold;
            padding: 8px 16px;
            border: 1px solid {MEDIUM_DARK_GREY};
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            margin-right: 2px;
        }}
        QTabBar::tab:selected {{
            background-color: {COBBER_MAROON};
            color: #ffffff;
            border: 1px solid {COBBER_MAROON_DARK};
        }}
        QTabBar::tab:!selected {{
            background-color: {MEDIUM_DARK_GREY};
            color: #ffffff;
        }}
        QGroupBox {{
            color: #222222;
            font-weight: bold;
            border: 1px solid #d6d6d6;
            border-radius: 6px;
            margin-top: 9px;
            padding-top: 12px;
            background-color: #fafafa;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 5px 0 5px;
            color: {COBBER_MAROON};
            background-color: #fafafa;
            font-weight: bold;
        }}
        QLabel {{
            color: #222222;
            background-color: transparent;
        }}
        QLabel#StatusLabel {{
            color: #333333;
            background-color: #ffffff;
            border: 1px solid #d8d8d8;
            border-radius: 5px;
            padding: 8px;
            font-style: italic;
        }}
        QLabel#RewardLabel {{
            color: {COBBER_MAROON};
        }}
        QLabel#TeachingNote {{
            color: #333333;
            background-color: #ffffff;
            border: 1px solid #d8d8d8;
            border-radius: 5px;
            padding: 8px;
        }}
        QLabel#FormulaLabel {{
            color: {COBBER_MAROON};
            background-color: #ffffff;
            border: 1px solid #d8d8d8;
            border-radius: 5px;
            padding: 8px;
            font-weight: bold;
        }}
        QComboBox {{
            background-color: #ffffff;
            color: #111111;
            border: 1px solid #9a9a9a;
            border-radius: 4px;
            padding: 4px 8px;
            min-height: 26px;
            selection-background-color: {COBBER_MAROON};
            selection-color: #ffffff;
        }}
        QComboBox:disabled {{
            background-color: {MEDIUM_DARK_GREY};
            color: #ffffff;
            border: 1px solid {MEDIUM_DARK_GREY};
        }}
        QComboBox QAbstractItemView {{
            background-color: #ffffff;
            color: #111111;
            selection-background-color: {COBBER_MAROON};
            selection-color: #ffffff;
        }}
        QPushButton {{
            background-color: {COBBER_MAROON};
            color: #ffffff;
            font-weight: bold;
            border: 1px solid {COBBER_MAROON_DARK};
            border-radius: 5px;
            padding: 7px 10px;
            min-height: 24px;
        }}
        QPushButton:hover {{
            background-color: {COBBER_MAROON_DARK};
        }}
        QPushButton:pressed {{
            background-color: #2e0a1d;
        }}
        QPushButton:disabled {{
            background-color: {MEDIUM_DARK_GREY};
            color: #ffffff;
            border: 1px solid {MEDIUM_DARK_GREY};
            font-weight: bold;
        }}
        """
    )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_app_stylesheet(app)
    window = CobberEcoRLApp()
    window.show()
    sys.exit(app.exec())

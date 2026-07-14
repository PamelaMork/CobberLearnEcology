# CobberEcoHydro_NorthMarsh_v8.py
# A PyQt6 application for training, testing, and inspecting a Deep Q-Network
# that manages water entering a simplified managed marsh.
#
# Ecological framing:
#   North Marsh is below its seasonal target depth. A learning agent adjusts
#   a water-control gate by closing it slightly, holding its current setting,
#   or opening it slightly. The agent must bring the marsh into an acceptable
#   depth range and reduce inflow before the water rises too far. Overshoot is
#   treated as especially costly because low black tern nests may be vulnerable
#   to rising water.
#
# State received by the agent:
#   [depth estimate received by the agent, inflow rate, target water depth]
#
# The environment privately tracks the simulated marsh depth. Depth-measurement
# uncertainty is applied only to the estimate received by the agent. The target
# depth is selected by the management team and is therefore known exactly.
#
# Optional images in the same directory as this script:
#   crusty_happy.png
#   crusty_angry.png
#
# Dependencies:
#   pip install PyQt6 numpy matplotlib tensorflow
#
# Run:
#   python CobberEcoHydro_NorthMarsh_v8.py

from __future__ import annotations

import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


# Book/app palette
COBBER_MAROON = "#6C1D45"
INFO_BLUE = "#3E6990"
PROJECT_GREEN = "#184F35"
MISC_BROWN = "#756D59"
CHARCOAL = "#3D3D3D"
SOFT_GRAY = "#D3D3D3"
HIGHLIGHT_GOLD = "#A9823A"


OUTCOME_TABLE_COLORS = {
    "Stable target": PROJECT_GREEN,
    "Below target": "#8A6B2F",
    "Overshoot": COBBER_MAROON,
}

ACTION_COLORS = {
    0: COBBER_MAROON,
    1: HIGHLIGHT_GOLD,
    2: INFO_BLUE,
}


def style_outcome_cell(item: QTableWidgetItem, outcome: str) -> None:
    """Apply the same high-contrast outcome styling on both tables."""
    item.setBackground(QColor(OUTCOME_TABLE_COLORS.get(outcome, CHARCOAL)))
    item.setForeground(QColor("#FFFFFF"))
    font = item.font()
    font.setBold(True)
    item.setFont(font)


def configure_read_only_table(table: QTableWidget) -> None:
    """Prevent selection and hover states from obscuring outcome colors."""
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    table.setMouseTracking(False)
    table.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, False)


def app_root() -> Path:
    """Return the directory containing the script or packaged executable."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_ROOT = app_root()


def load_pixmap(filename: str) -> QPixmap:
    """Load an optional app image from a few likely folders."""
    candidates = [
        APP_ROOT / filename,
        APP_ROOT / "images" / filename,
        APP_ROOT / "figures" / "Robot" / filename,
    ]
    for path in candidates:
        if path.exists():
            return QPixmap(str(path))
    return QPixmap()


def scaled_pixmap(pixmap: QPixmap, width: int, height: int) -> QPixmap:
    if pixmap.isNull():
        return pixmap
    return pixmap.scaled(
        width,
        height,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class WetlandControlEnv:
    """Simplified managed-marsh water-level environment.

    The DQN receives normalized versions of three ecological state variables:

        [depth estimate received by the agent, inflow rate, target water depth]

    Actions:
        0 -> close the gate slightly / reduce inflow
        1 -> hold the current gate setting
        2 -> open the gate slightly / increase inflow
    """

    ACTION_LABELS = {
        0: "Close slightly",
        1: "Hold setting",
        2: "Open slightly",
    }

    def __init__(
        self,
        depth_measurement_uncertainty: float = 0.5,
        seed: Optional[int] = None,
    ):
        # The app displays decision steps rather than elapsed time. The internal
        # step length is retained so inflow can use a meaningful rate unit.
        self.dt_hours = 0.25

        # Inflow is expressed as centimeters of marsh-level rise per hour.
        self.min_inflow_rate = 0.0
        self.max_inflow_rate = 1.50
        self.inflow_change = 0.25

        # The management target is the center of an acceptable range.
        self.depth_tolerance = 0.50
        self.stable_inflow_threshold = 0.05
        self.stable_steps_required = 3
        self.max_steps = 120
        self.no_progress_limit = 40

        self.training_target_min = 12.0
        self.training_target_max = 24.0
        self.starting_gap_min = 4.0
        self.starting_gap_max = 8.0

        self.state_size = 3
        self.action_size = 3
        self.depth_measurement_uncertainty = float(depth_measurement_uncertainty)
        self.fixed_target_depth: Optional[float] = None
        self.rng = np.random.default_rng(seed)

        # Depth values are scaled before entering the neural network.
        self.depth_scale = 30.0

        self.reset()

    @property
    def lower_limit(self) -> float:
        return self.target_depth - self.depth_tolerance

    @property
    def upper_limit(self) -> float:
        return self.target_depth + self.depth_tolerance

    def set_target_depth(
        self,
        target_depth: Optional[float],
        depth_measurement_uncertainty: Optional[float] = None,
    ) -> None:
        """Set a fixed target for evaluation, or None for random training targets."""
        self.fixed_target_depth = None if target_depth is None else float(target_depth)
        if depth_measurement_uncertainty is not None:
            self.depth_measurement_uncertainty = float(
                depth_measurement_uncertainty
            )

    def reset(self, starting_depth: Optional[float] = None) -> np.ndarray:
        """Start a new episode below the selected target depth."""
        if self.fixed_target_depth is None:
            self.target_depth = float(
                self.rng.uniform(self.training_target_min, self.training_target_max)
            )
        else:
            self.target_depth = float(self.fixed_target_depth)

        if starting_depth is None:
            gap = float(self.rng.uniform(self.starting_gap_min, self.starting_gap_max))
            self.simulated_marsh_depth = max(0.0, self.target_depth - gap)
        else:
            self.simulated_marsh_depth = max(0.0, float(starting_depth))

        self.starting_depth = self.simulated_marsh_depth
        self.inflow_rate = 0.0
        self.step_count = 0
        self.stable_step_count = 0
        self.no_progress_steps = 0
        self.done = False
        self.outcome = "In progress"
        self.last_action = 1
        self.depth_estimate = self._estimate_depth()
        return self._get_state()

    def _estimate_depth(self) -> float:
        error = self.rng.uniform(
            -self.depth_measurement_uncertainty,
            self.depth_measurement_uncertainty,
        )
        return float(max(0.0, self.simulated_marsh_depth + error))

    def _get_state(self) -> np.ndarray:
        """Return normalized state values for neural-network training."""
        return np.array(
            [
                self.depth_estimate / self.depth_scale,
                self.inflow_rate / self.max_inflow_rate,
                self.target_depth / self.depth_scale,
            ],
            dtype=np.float32,
        )

    def _distance_to_acceptable_range(self, depth: float) -> float:
        if depth < self.lower_limit:
            return self.lower_limit - depth
        if depth > self.upper_limit:
            return depth - self.upper_limit
        return 0.0

    def _braking_distance(self, inflow_rate: Optional[float] = None) -> float:
        """Estimate additional rise if the gate is closed at every later step."""
        rate = self.inflow_rate if inflow_rate is None else float(inflow_rate)
        additional_rise = 0.0
        while rate > self.min_inflow_rate:
            rate = max(self.min_inflow_rate, rate - self.inflow_change)
            additional_rise += rate * self.dt_hours
        return float(additional_rise)

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Cannot call step() after the episode has ended.")

        action = int(action)
        if action not in (0, 1, 2):
            raise ValueError(f"Invalid action: {action}")

        previous_depth = self.simulated_marsh_depth
        previous_distance = self._distance_to_acceptable_range(previous_depth)
        previous_projected_excess = max(
            0.0,
            previous_depth
            + self._braking_distance(self.inflow_rate)
            - self.upper_limit,
        )
        self.last_action = action

        if action == 0:
            self.inflow_rate = max(
                self.min_inflow_rate,
                self.inflow_rate - self.inflow_change,
            )
        elif action == 2:
            self.inflow_rate = min(
                self.max_inflow_rate,
                self.inflow_rate + self.inflow_change,
            )

        # Water already entering through the channel changes the marsh depth.
        self.simulated_marsh_depth += self.inflow_rate * self.dt_hours
        self.step_count += 1
        self.depth_estimate = self._estimate_depth()

        new_distance = self._distance_to_acceptable_range(
            self.simulated_marsh_depth
        )

        # Reward design:
        #   1. Reward progress toward the acceptable range.
        #   2. Penalize sitting below the target with little or no inflow.
        #   3. Make high inflow increasingly costly near the acceptable range.
        #   4. Reward braking when it reduces a projected overshoot.
        #   5. Keep terminal rewards moderate so DQN training remains stable.
        reward = 2.5 * (previous_distance - new_distance)
        reward -= 0.03

        remaining_to_range = max(
            0.0,
            self.lower_limit - self.simulated_marsh_depth,
        )
        normalized_inflow = self.inflow_rate / self.max_inflow_rate

        # Closing too early should not become a comfortable local optimum.
        if remaining_to_range > 0.0:
            reward -= 0.12 * remaining_to_range
            if self.inflow_rate < self.inflow_change:
                reward -= 0.35

        proximity = float(
            np.clip((2.5 - remaining_to_range) / 2.5, 0.0, 1.0)
        )
        reward -= 1.4 * proximity * (normalized_inflow**2)

        projected_stopping_depth = (
            self.simulated_marsh_depth + self._braking_distance()
        )
        projected_excess = max(
            0.0,
            projected_stopping_depth - self.upper_limit,
        )
        reward -= 3.0 * projected_excess

        # Closing becomes valuable when it measurably lowers overshoot risk.
        braking_improvement = max(
            0.0,
            previous_projected_excess - projected_excess,
        )
        if action == 0 and braking_improvement > 0.0:
            reward += 2.0 * braking_improvement

        if (
            self.simulated_marsh_depth < self.lower_limit
            and self.inflow_rate <= self.stable_inflow_threshold
        ):
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0

        # Crossing the upper management limit is an immediate overshoot.
        if self.simulated_marsh_depth > self.upper_limit:
            self.done = True
            self.outcome = "Overshoot"
            overshoot_amount = self.simulated_marsh_depth - self.upper_limit
            reward -= 28.0 + 12.0 * overshoot_amount

        else:
            in_depth_range = (
                self.lower_limit
                <= self.simulated_marsh_depth
                <= self.upper_limit
            )
            inflow_stable = self.inflow_rate <= self.stable_inflow_threshold

            if in_depth_range and inflow_stable:
                self.stable_step_count += 1
                reward += 2.0
            else:
                self.stable_step_count = 0

            if self.stable_step_count >= self.stable_steps_required:
                self.done = True
                self.outcome = "Stable target"
                reward += 35.0

            elif self.no_progress_steps >= self.no_progress_limit:
                self.done = True
                self.outcome = "Below target"
                remaining = self.lower_limit - self.simulated_marsh_depth
                reward -= 24.0 + 4.0 * remaining

            elif self.step_count >= self.max_steps:
                self.done = True
                if self.simulated_marsh_depth < self.lower_limit:
                    self.outcome = "Below target"
                    remaining = self.lower_limit - self.simulated_marsh_depth
                    reward -= 24.0 + 4.0 * remaining
                else:
                    self.outcome = "Overshoot"
                    reward -= 28.0

        info = {
            "simulated_depth": float(self.simulated_marsh_depth),
            "depth_estimate": float(self.depth_estimate),
            "inflow_rate": float(self.inflow_rate),
            "target_depth": float(self.target_depth),
            "lower_limit": float(self.lower_limit),
            "upper_limit": float(self.upper_limit),
            "outcome": self.outcome,
            "action_label": self.ACTION_LABELS[action],
            "decision_step": int(self.step_count),
        }
        return self._get_state(), float(reward), self.done, info


class DQN(tf.keras.Model):
    """Small neural network that estimates one Q-value for each gate action."""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(64, activation="relu")
        self.output_layer = layers.Dense(action_size)
        self._state_size = state_size

    def call(self, x, training: bool = False):
        del training  # The network contains no training-dependent layers.
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)


class ReplayMemory:
    """Fixed-size memory used for experience replay."""

    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class TrainingThread(QThread):
    update_checkpoint = pyqtSignal(object)
    update_episode = pyqtSignal(object)
    update_progress = pyqtSignal(int)
    status_message = pyqtSignal(str)
    training_finished = pyqtSignal(bool)

    def __init__(self, num_episodes: int, depth_measurement_uncertainty: float):
        super().__init__()
        self.num_episodes = int(num_episodes)
        self.env = WetlandControlEnv(depth_measurement_uncertainty)
        self.model: Optional[DQN] = None
        self.target_model: Optional[DQN] = None
        self.memory = ReplayMemory(20000)

        self.gamma = 0.98
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 1200.0
        self.batch_size = 64
        self.target_update_episodes = 10
        self.steps_done = 0

        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _epsilon(self) -> float:
        return float(
            self.epsilon_end
            + (self.epsilon_start - self.epsilon_end)
            * np.exp(-self.steps_done / self.epsilon_decay)
        )

    def select_action(self, state: np.ndarray) -> int:
        epsilon = self._epsilon()
        self.steps_done += 1
        if random.random() < epsilon:
            return random.randrange(self.env.action_size)

        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = self.model(state_tensor, training=False)
        return int(tf.argmax(q_values[0]).numpy())

    def _greedy_action(self, state: np.ndarray) -> int:
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = self.model(state_tensor, training=False)
        return int(tf.argmax(q_values[0]).numpy())

    def _train_batch(self, optimizer, loss_function) -> None:
        transitions = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = map(
            np.array,
            zip(*transitions),
        )

        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones_tensor = tf.convert_to_tensor(
            dones.astype(np.float32),
            dtype=tf.float32,
        )
        actions_tensor = tf.convert_to_tensor(
            actions.astype(np.int32),
            dtype=tf.int32,
        )

        next_q_values = self.target_model(next_states_tensor, training=False)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        target_q_values = rewards_tensor + (
            self.gamma * max_next_q_values * (1.0 - dones_tensor)
        )

        with tf.GradientTape() as tape:
            all_q_values = self.model(states_tensor, training=True)
            action_masks = tf.one_hot(actions_tensor, self.env.action_size)
            chosen_q_values = tf.reduce_sum(all_q_values * action_masks, axis=1)
            loss = loss_function(target_q_values, chosen_q_values)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradient_pairs = [
            (gradient, variable)
            for gradient, variable in zip(
                gradients,
                self.model.trainable_variables,
            )
            if gradient is not None
        ]
        if not gradient_pairs:
            return

        gradient_values, variables = zip(*gradient_pairs)
        clipped_gradients, _ = tf.clip_by_global_norm(gradient_values, 5.0)
        optimizer.apply_gradients(zip(clipped_gradients, variables))

    def _run_checkpoint(self, training_episode: int) -> dict:
        """Evaluate the current policy on the same scenario without exploration."""
        checkpoint_env = WetlandControlEnv(
            self.env.depth_measurement_uncertainty,
            seed=2027,
        )
        checkpoint_env.set_target_depth(
            18.0,
            self.env.depth_measurement_uncertainty,
        )
        state = checkpoint_env.reset(starting_depth=12.0)

        simulated_depths = [checkpoint_env.simulated_marsh_depth]
        depth_estimates = [checkpoint_env.depth_estimate]
        inflow_rates = [checkpoint_env.inflow_rate]
        actions: list[int] = []
        total_reward = 0.0

        for _ in range(checkpoint_env.max_steps):
            action = self._greedy_action(state)
            next_state, reward, done, _ = checkpoint_env.step(action)
            state = next_state
            total_reward += reward
            simulated_depths.append(checkpoint_env.simulated_marsh_depth)
            depth_estimates.append(checkpoint_env.depth_estimate)
            inflow_rates.append(checkpoint_env.inflow_rate)
            actions.append(action)
            if done:
                break

        return {
            "training_episode": training_episode,
            "simulated_depths": simulated_depths,
            "depth_estimates": depth_estimates,
            "inflow_rates": inflow_rates,
            "actions": actions,
            "target_depth": checkpoint_env.target_depth,
            "lower_limit": checkpoint_env.lower_limit,
            "upper_limit": checkpoint_env.upper_limit,
            "starting_depth": checkpoint_env.starting_depth,
            "final_depth": checkpoint_env.simulated_marsh_depth,
            "final_inflow": checkpoint_env.inflow_rate,
            "total_reward": total_reward,
            "outcome": checkpoint_env.outcome,
            "random_action_chance": self._epsilon(),
            "max_inflow_rate": checkpoint_env.max_inflow_rate,
            "steps": checkpoint_env.step_count,
        }

    def run(self) -> None:
        stopped_early = False
        try:
            self.model = DQN(self.env.state_size, self.env.action_size)
            self.target_model = DQN(self.env.state_size, self.env.action_size)

            dummy_input = tf.zeros((1, self.env.state_size), dtype=tf.float32)
            self.model(dummy_input)
            self.target_model(dummy_input)
            self.target_model.set_weights(self.model.get_weights())

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
            loss_function = tf.keras.losses.Huber()

            for episode in range(1, self.num_episodes + 1):
                if self._stop_requested:
                    stopped_early = True
                    break

                state = self.env.reset()
                total_reward = 0.0

                for _ in range(self.env.max_steps):
                    if self._stop_requested:
                        stopped_early = True
                        break

                    action = self.select_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    self.memory.push(state, action, reward, next_state, done)

                    state = next_state
                    total_reward += reward

                    if len(self.memory) >= self.batch_size:
                        self._train_batch(optimizer, loss_function)

                    if done:
                        break

                if stopped_early:
                    break

                if episode % self.target_update_episodes == 0:
                    self.target_model.set_weights(self.model.get_weights())

                result = {
                    "episode": episode,
                    "starting_depth": self.env.starting_depth,
                    "target_depth": self.env.target_depth,
                    "lower_limit": self.env.lower_limit,
                    "upper_limit": self.env.upper_limit,
                    "final_depth": self.env.simulated_marsh_depth,
                    "final_inflow": self.env.inflow_rate,
                    "total_reward": float(total_reward),
                    "outcome": self.env.outcome,
                    "random_action_chance": self._epsilon(),
                    "steps": self.env.step_count,
                }
                self.update_episode.emit(result)

                progress_percent = int(round(100 * episode / self.num_episodes))
                self.update_progress.emit(progress_percent)

                if episode == 1 or episode % 5 == 0 or episode == self.num_episodes:
                    self.update_checkpoint.emit(self._run_checkpoint(episode))

            if stopped_early:
                self.status_message.emit(
                    "Training stopped. The partially trained model may still be saved."
                )
            else:
                self.status_message.emit(
                    "Training complete. Review Training Progress, then save the model if desired."
                )

        except Exception as exc:  # pragma: no cover - GUI error reporting
            self.status_message.emit(f"Training error: {exc}")
        finally:
            self.training_finished.emit(stopped_early)


class EvaluationThread(QThread):
    status_message = pyqtSignal(str)
    update_run = pyqtSignal(object)
    update_summary = pyqtSignal(object)
    evaluation_finished = pyqtSignal()

    def __init__(
        self,
        model: DQN,
        target_depth: float,
        depth_measurement_uncertainty: float,
        num_runs: int,
    ):
        super().__init__()
        self.model = model
        self.target_depth = float(target_depth)
        self.depth_measurement_uncertainty = float(
            depth_measurement_uncertainty
        )
        self.num_runs = int(num_runs)

    def _run_single_trial(self, env: WetlandControlEnv, run_number: int) -> dict:
        state = env.reset()
        simulated_depths = [env.simulated_marsh_depth]
        depth_estimates = [env.depth_estimate]
        inflow_rates = [env.inflow_rate]
        actions: list[int] = []
        total_reward = 0.0

        for _ in range(env.max_steps):
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            q_values = self.model(state_tensor, training=False)
            action = int(tf.argmax(q_values[0]).numpy())

            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            simulated_depths.append(env.simulated_marsh_depth)
            depth_estimates.append(env.depth_estimate)
            inflow_rates.append(env.inflow_rate)
            actions.append(action)

            if done:
                break

        final_error = env.simulated_marsh_depth - env.target_depth
        return {
            "run": run_number,
            "simulated_depths": simulated_depths,
            "depth_estimates": depth_estimates,
            "inflow_rates": inflow_rates,
            "actions": actions,
            "target_depth": env.target_depth,
            "lower_limit": env.lower_limit,
            "upper_limit": env.upper_limit,
            "starting_depth": env.starting_depth,
            "final_depth": env.simulated_marsh_depth,
            "final_inflow": env.inflow_rate,
            "error": final_error,
            "absolute_error": abs(final_error),
            "outcome": env.outcome,
            "reward": total_reward,
            "steps": env.step_count,
            "max_inflow_rate": env.max_inflow_rate,
        }

    def run(self) -> None:
        try:
            env = WetlandControlEnv(self.depth_measurement_uncertainty)
            env.set_target_depth(
                self.target_depth,
                self.depth_measurement_uncertainty,
            )

            self.status_message.emit(
                f"Testing at a {self.target_depth:.2f} cm target with "
                f"depth-measurement uncertainty ±{self.depth_measurement_uncertainty:.2f} cm."
            )

            results: list[dict] = []
            for run_number in range(1, self.num_runs + 1):
                result = self._run_single_trial(env, run_number)
                results.append(result)
                self.update_run.emit(result)

            successes = sum(
                result["outcome"] == "Stable target" for result in results
            )
            underfills = sum(
                result["outcome"] == "Below target" for result in results
            )
            overshoots = sum(
                result["outcome"] == "Overshoot" for result in results
            )

            summary = {
                "num_runs": self.num_runs,
                "successes": successes,
                "success_rate": 100.0 * successes / self.num_runs,
                "average_final_depth": float(
                    np.mean([result["final_depth"] for result in results])
                ),
                "average_error": float(
                    np.mean([result["error"] for result in results])
                ),
                "mean_absolute_error": float(
                    np.mean([result["absolute_error"] for result in results])
                ),
                "average_final_inflow": float(
                    np.mean([result["final_inflow"] for result in results])
                ),
                "underfills": underfills,
                "overshoots": overshoots,
            }
            self.update_summary.emit(summary)
            self.status_message.emit("Testing complete.")

        except Exception as exc:  # pragma: no cover - GUI error reporting
            self.status_message.emit(f"Testing error: {exc}")
        finally:
            self.evaluation_finished.emit()


def apply_app_stylesheet(app: QApplication) -> None:
    app.setStyleSheet(
        f"""
        QWidget {{
            color: {CHARCOAL};
            background-color: #ffffff;
            font-family: Lato;
            font-size: 10.5pt;
        }}
        QMainWindow, QDialog {{ background-color: #ffffff; }}

        QTabWidget::pane {{
            border: 1px solid #b8b8b8;
            top: -1px;
        }}
        QTabBar::tab {{
            background: #666666;
            color: #ffffff;
            padding: 8px 18px;
            margin-right: 2px;
            font-weight: 600;
        }}
        QTabBar::tab:selected {{
            background: {COBBER_MAROON};
            color: #ffffff;
            font-weight: 700;
        }}

        QLabel {{ background-color: transparent; }}

        QGroupBox {{
            border: 1px solid #c7c7c7;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 8px;
            font-weight: 700;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: {COBBER_MAROON};
        }}

        QComboBox, QTableWidget {{
            background-color: #ffffff;
            color: #111111;
            border: 1px solid #a0a0a0;
            border-radius: 3px;
            padding: 3px 6px;
            selection-background-color: {COBBER_MAROON};
            selection-color: #ffffff;
        }}

        QPushButton {{
            border-radius: 4px;
            padding: 7px 14px;
            font-weight: 700;
        }}
        QPushButton#primaryButton {{
            background-color: {COBBER_MAROON};
            color: #ffffff;
            border: 1px solid {COBBER_MAROON};
        }}
        QPushButton#primaryButton:hover {{ background-color: #7d2953; }}
        QPushButton#primaryButton:disabled {{
            background-color: #b9a2af;
            border-color: #b9a2af;
        }}
        QPushButton#secondaryButton {{
            background-color: #666666;
            color: #ffffff;
            border: 1px solid #666666;
        }}
        QPushButton#secondaryButton:hover {{ background-color: #555555; }}
        QPushButton#secondaryButton:disabled {{
            background-color: #bdbdbd;
            border-color: #bdbdbd;
        }}

        QProgressBar {{
            border: 1px solid #9a9a9a;
            border-radius: 3px;
            text-align: center;
            background-color: #f5f5f5;
        }}
        QProgressBar::chunk {{ background-color: {PROJECT_GREEN}; }}

        QHeaderView::section {{
            background-color: {CHARCOAL};
            color: #ffffff;
            padding: 5px;
            border: 0;
            font-weight: 700;
        }}
        """
    )


def make_outcome_card() -> QFrame:
    card = QFrame()
    card.setObjectName("outcomeCard")
    card.setFrameShape(QFrame.Shape.StyledPanel)
    card.setStyleSheet(
        "QFrame#outcomeCard {"
        "background-color: #f5f5f5;"
        "border: 1px solid #c7c7c7;"
        "border-radius: 5px;"
        "padding: 8px;"
        "}"
    )
    return card


def outcome_style(outcome: str) -> str:
    if outcome == "Stable target":
        background = "#E8F1EC"
        border = PROJECT_GREEN
    elif outcome == "Overshoot":
        background = "#F7E9EE"
        border = COBBER_MAROON
    elif outcome == "Below target":
        background = "#F3EFE5"
        border = HIGHLIGHT_GOLD
    else:
        background = "#F5F5F5"
        border = SOFT_GRAY

    return (
        "QFrame#outcomeCard {"
        f"background-color: {background};"
        f"border: 2px solid {border};"
        "border-radius: 5px;"
        "padding: 8px;"
        "}"
    )


def plot_episode(
    figure: Figure,
    simulated_depths,
    depth_estimates,
    inflow_rates,
    target_depth: float,
    lower_limit: float,
    upper_limit: float,
    max_inflow_rate: float,
    title: str,
) -> None:
    figure.clear()
    depth_ax, inflow_ax = figure.subplots(2, 1, sharex=True)

    simulated_depths = np.asarray(simulated_depths, dtype=float)
    depth_estimates = np.asarray(depth_estimates, dtype=float)
    inflow_rates = np.asarray(inflow_rates, dtype=float)
    decision_steps = np.arange(len(simulated_depths), dtype=int)

    depth_ax.axhspan(
        lower_limit,
        upper_limit,
        color=HIGHLIGHT_GOLD,
        alpha=0.18,
        label="Acceptable range",
    )
    depth_ax.axhline(
        target_depth,
        color=COBBER_MAROON,
        linestyle="--",
        linewidth=1.6,
        label=f"Target ({target_depth:.1f} cm)",
    )
    depth_ax.plot(
        decision_steps,
        simulated_depths,
        color=INFO_BLUE,
        linewidth=2.0,
        label="Simulated marsh depth",
    )
    depth_ax.scatter(
        decision_steps,
        depth_estimates,
        color=CHARCOAL,
        s=12,
        alpha=0.55,
        label="Depth estimate received by agent",
    )
    depth_ax.set_ylabel("Water depth (cm)")
    depth_ax.set_ylim(
        max(0.0, float(np.min(simulated_depths)) - 1.5),
        max(upper_limit + 1.5, float(np.max(simulated_depths)) + 1.0),
    )
    depth_ax.set_title(title, fontsize=11, fontweight="bold")
    depth_ax.legend(loc="best", fontsize=8)
    depth_ax.grid(alpha=0.20)

    inflow_ax.step(
        decision_steps,
        inflow_rates,
        where="post",
        color=PROJECT_GREEN,
        linewidth=2.0,
    )
    inflow_ax.set_xlabel("Decision step")
    inflow_ax.set_ylabel("Inflow rate (cm/hr)")
    inflow_ax.set_ylim(0, max_inflow_rate * 1.10)
    inflow_ax.grid(alpha=0.20)

    figure.tight_layout()


def plot_reward_history(figure: Figure, rewards: list[float]) -> None:
    figure.clear()
    ax = figure.add_subplot(111)

    if rewards:
        episodes = np.arange(1, len(rewards) + 1)
        ax.plot(
            episodes,
            rewards,
            color=SOFT_GRAY,
            linewidth=1.2,
            label="Individual episode",
        )

        window = min(10, len(rewards))
        if window >= 2:
            kernel = np.ones(window) / window
            rolling = np.convolve(np.asarray(rewards), kernel, mode="valid")
            rolling_episodes = np.arange(window, len(rewards) + 1)
            ax.plot(
                rolling_episodes,
                rolling,
                color=COBBER_MAROON,
                linewidth=2.2,
                label=f"{window}-episode average",
            )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.grid(alpha=0.20)
    if rewards:
        ax.legend(loc="best", fontsize=8)
    figure.tight_layout()


def plot_stable_rate(figure: Figure, outcomes: list[str]) -> None:
    figure.clear()
    ax = figure.add_subplot(111)

    if outcomes:
        episodes = np.arange(1, len(outcomes) + 1)
        stable = np.array(
            [100.0 if outcome == "Stable target" else 0.0 for outcome in outcomes]
        )
        ax.scatter(
            episodes,
            stable,
            color=SOFT_GRAY,
            s=10,
            alpha=0.65,
            label="Episode outcome",
        )

        window = min(20, len(outcomes))
        if window >= 2:
            kernel = np.ones(window) / window
            rolling = np.convolve(stable, kernel, mode="valid")
            rolling_episodes = np.arange(window, len(outcomes) + 1)
            ax.plot(
                rolling_episodes,
                rolling,
                color=PROJECT_GREEN,
                linewidth=2.2,
                label=f"{window}-episode stable-target rate",
            )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Stable-target rate (%)")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.20)
    if outcomes:
        ax.legend(loc="best", fontsize=8)
    figure.tight_layout()


def set_manager_feedback(
    label: QLabel,
    outcome: str,
    happy_pixmap: QPixmap,
    angry_pixmap: QPixmap,
) -> None:
    pixmap = happy_pixmap if outcome == "Stable target" else angry_pixmap
    if pixmap.isNull():
        fallback = "Crusty manager approves." if outcome == "Stable target" else (
            "Crusty manager is not impressed."
        )
        label.setText(fallback)
        return
    label.setText("")
    label.setPixmap(scaled_pixmap(pixmap, 230, 125))


def ecological_interpretation(outcome: str) -> str:
    if outcome == "Stable target":
        return (
            "The simulated marsh depth entered the acceptable range, and inflow "
            "slowed before the checkpoint ended."
        )
    if outcome == "Below target":
        return (
            "The marsh remained below the acceptable range. Shallow pools may stay "
            "disconnected."
        )
    if outcome == "Overshoot":
        return (
            "The water rose above the acceptable range. Rising water may threaten "
            "low black tern nests."
        )
    return "No outcome is available yet."


class TrainingProgressTab(QWidget):
    """Across-episode evidence: rewards, outcomes, and a proper results table."""

    def __init__(self):
        super().__init__()
        self.rewards: list[float] = []
        self.outcomes: list[str] = []

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 10, 12, 10)
        root_layout.setSpacing(7)

        intro = QLabel(
            "Use this tab to decide whether training is improving across many attempts. "
            "These rows record exploratory training episodes, so some actions are random."
        )
        intro.setWordWrap(True)
        root_layout.addWidget(intro)

        summary_group = QGroupBox("Training Outcomes")
        summary_layout = QGridLayout(summary_group)
        self.episodes_value = QLabel("0")
        self.stable_value = QLabel("0")
        self.below_value = QLabel("0")
        self.overshoot_value = QLabel("0")

        summary_items = [
            ("Episodes", self.episodes_value),
            ("Stable targets", self.stable_value),
            ("Below target", self.below_value),
            ("Overshoots", self.overshoot_value),
        ]
        for column, (label_text, value_label) in enumerate(summary_items):
            label = QLabel(label_text)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label.setStyleSheet(
                f"font-size: 16pt; font-weight: 700; color: {COBBER_MAROON};"
            )
            summary_layout.addWidget(value_label, 0, column)
            summary_layout.addWidget(label, 1, column)
        root_layout.addWidget(summary_group)

        plots_splitter = QSplitter(Qt.Orientation.Horizontal)

        reward_group = QGroupBox("Reward Across Episodes")
        reward_layout = QVBoxLayout(reward_group)
        self.reward_figure = Figure(figsize=(5.6, 3.0))
        self.reward_canvas = FigureCanvas(self.reward_figure)
        reward_layout.addWidget(self.reward_canvas)
        plots_splitter.addWidget(reward_group)

        rate_group = QGroupBox("Stable-Target Rate")
        rate_layout = QVBoxLayout(rate_group)
        self.rate_figure = Figure(figsize=(5.6, 3.0))
        self.rate_canvas = FigureCanvas(self.rate_figure)
        rate_layout.addWidget(self.rate_canvas)
        plots_splitter.addWidget(rate_group)

        plots_splitter.setStretchFactor(0, 1)
        plots_splitter.setStretchFactor(1, 1)
        root_layout.addWidget(plots_splitter, 1)

        table_group = QGroupBox("Training Episode Table")
        table_layout = QVBoxLayout(table_group)
        self.results_table = QTableWidget(0, 8)
        self.results_table.setHorizontalHeaderLabels(
            [
                "Episode",
                "Start depth",
                "Target range",
                "Final depth",
                "Final inflow",
                "Reward",
                "Random action (%)",
                "Outcome",
            ]
        )
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setAlternatingRowColors(True)
        configure_read_only_table(self.results_table)
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        table_layout.addWidget(self.results_table)
        root_layout.addWidget(table_group, 2)

        self.status_label = QLabel("No training session has started.")
        self.status_label.setStyleSheet(f"color: {MISC_BROWN}; font-weight: 700;")
        root_layout.addWidget(self.status_label)

        plot_reward_history(self.reward_figure, [])
        plot_stable_rate(self.rate_figure, [])
        self.reward_canvas.draw()
        self.rate_canvas.draw()

    def reset_session(self) -> None:
        self.rewards.clear()
        self.outcomes.clear()
        self.results_table.setRowCount(0)
        self._update_summary()
        plot_reward_history(self.reward_figure, [])
        plot_stable_rate(self.rate_figure, [])
        self.reward_canvas.draw()
        self.rate_canvas.draw()
        self.status_label.setText("Training is in progress.")

    def add_episode_result(self, result: dict) -> None:
        self.rewards.append(float(result["total_reward"]))
        self.outcomes.append(result["outcome"])

        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        values = [
            str(result["episode"]),
            f"{result['starting_depth']:.2f} cm",
            f"{result['lower_limit']:.2f}–{result['upper_limit']:.2f} cm",
            f"{result['final_depth']:.2f} cm",
            f"{result['final_inflow']:.2f} cm/hr",
            f"{result['total_reward']:+.1f}",
            f"{100.0 * result['random_action_chance']:.0f}%",
            result["outcome"],
        ]

        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if column == 7:
                style_outcome_cell(item, result["outcome"])
            self.results_table.setItem(row, column, item)

        self.results_table.scrollToBottom()
        self._update_summary()

        episode = int(result["episode"])
        if episode == 1 or episode % 5 == 0:
            plot_reward_history(self.reward_figure, self.rewards)
            plot_stable_rate(self.rate_figure, self.outcomes)
            self.reward_canvas.draw()
            self.rate_canvas.draw()

    def _update_summary(self) -> None:
        self.episodes_value.setText(str(len(self.outcomes)))
        self.stable_value.setText(str(self.outcomes.count("Stable target")))
        self.below_value.setText(str(self.outcomes.count("Below target")))
        self.overshoot_value.setText(str(self.outcomes.count("Overshoot")))

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)
        plot_reward_history(self.reward_figure, self.rewards)
        plot_stable_rate(self.rate_figure, self.outcomes)
        self.reward_canvas.draw()
        self.rate_canvas.draw()


class TrainingTab(QWidget):
    def __init__(self, progress_tab: TrainingProgressTab):
        super().__init__()
        self.progress_tab = progress_tab
        self.training_thread: Optional[TrainingThread] = None
        self.happy_pixmap = load_pixmap("crusty_happy.png")
        self.angry_pixmap = load_pixmap("crusty_angry.png")

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 10, 12, 10)
        root_layout.setSpacing(7)

        intro = QLabel(
            "Train a DQN to bring North Marsh into its acceptable depth range "
            "and reduce inflow before the water rises too far."
        )
        intro.setWordWrap(True)
        root_layout.addWidget(intro)

        controls_group = QGroupBox("Training Setup")
        controls_layout = QHBoxLayout(controls_group)

        self.episodes_input = QComboBox()
        for episode_count in (50, 100, 150, 200):
            self.episodes_input.addItem(
                f"{episode_count} episodes",
                episode_count,
            )

        self.uncertainty_input = QComboBox()
        for uncertainty in (0.10, 0.50, 1.00):
            self.uncertainty_input.addItem(
                f"{uncertainty:.2f} cm",
                uncertainty,
            )
        self.uncertainty_input.setCurrentIndex(1)

        self.start_button = QPushButton("Start Training")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self.start_training)

        self.stop_button = QPushButton("Stop Training")
        self.stop_button.setObjectName("secondaryButton")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)

        self.save_button = QPushButton("Save Model")
        self.save_button.setObjectName("secondaryButton")
        self.save_button.clicked.connect(self.save_model)
        self.save_button.setEnabled(False)

        controls_layout.addWidget(QLabel("Number of episodes:"))
        controls_layout.addWidget(self.episodes_input)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(QLabel("Depth-measurement uncertainty (±):"))
        controls_layout.addWidget(self.uncertainty_input)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.save_button)
        root_layout.addWidget(controls_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        root_layout.addWidget(self.progress_bar)

        checkpoint_note = QLabel(
            "The checkpoint uses the current policy with random actions turned off. "
            "It repeats the same target and starting depth so you can compare progress."
        )
        checkpoint_note.setWordWrap(True)
        checkpoint_note.setStyleSheet(f"color: {MISC_BROWN};")
        root_layout.addWidget(checkpoint_note)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        episode_group = QGroupBox("Current Policy Checkpoint")
        episode_layout = QVBoxLayout(episode_group)
        self.episode_figure = Figure(figsize=(8.0, 5.0))
        self.episode_canvas = FigureCanvas(self.episode_figure)
        episode_layout.addWidget(self.episode_canvas)
        splitter.addWidget(episode_group)

        self.outcome_card = make_outcome_card()
        outcome_layout = QVBoxLayout(self.outcome_card)

        self.manager_label = QLabel()
        self.manager_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.manager_label.setMinimumSize(260, 180)
        outcome_layout.addWidget(self.manager_label)

        self.outcome_title = QLabel("No checkpoint displayed yet")
        self.outcome_title.setStyleSheet(
            f"font-size: 13pt; font-weight: 700; color: {COBBER_MAROON};"
        )
        self.outcome_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outcome_layout.addWidget(self.outcome_title)

        self.outcome_details = QLabel(
            "Start training to see how the current policy manages a fixed North Marsh scenario."
        )
        self.outcome_details.setWordWrap(True)
        outcome_layout.addWidget(self.outcome_details)
        outcome_layout.addStretch(1)

        splitter.addWidget(self.outcome_card)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        root_layout.addWidget(splitter, 1)

        self.status_label = QLabel("Ready to train.")
        self.status_label.setStyleSheet(f"color: {MISC_BROWN}; font-weight: 700;")
        root_layout.addWidget(self.status_label)

    def start_training(self) -> None:
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.episodes_input.setEnabled(False)
        self.uncertainty_input.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Training is in progress.")
        self.progress_tab.reset_session()

        self.training_thread = TrainingThread(
            int(self.episodes_input.currentData()),
            float(self.uncertainty_input.currentData()),
        )
        self.training_thread.update_checkpoint.connect(self.update_checkpoint)
        self.training_thread.update_episode.connect(
            self.progress_tab.add_episode_result
        )
        self.training_thread.update_progress.connect(self.progress_bar.setValue)
        self.training_thread.status_message.connect(self.update_status)
        self.training_thread.training_finished.connect(self.on_training_finished)
        self.training_thread.start()

    def stop_training(self) -> None:
        if self.training_thread is not None and self.training_thread.isRunning():
            self.stop_button.setEnabled(False)
            self.status_label.setText(
                "Stop requested. The current training step will finish first."
            )
            self.training_thread.request_stop()

    def save_model(self) -> None:
        if self.training_thread is None or self.training_thread.model is None:
            QMessageBox.warning(self, "Save Model", "No trained model is available.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Trained Wetland Model",
            str(APP_ROOT / "north_marsh_dqn_v9.weights.h5"),
            "Keras Weights (*.weights.h5)",
        )
        if not filepath:
            return

        if not filepath.endswith(".weights.h5"):
            filepath += ".weights.h5"

        try:
            self.training_thread.model.save_weights(filepath)
            self.status_label.setText(
                f"Model saved as {os.path.basename(filepath)}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save Model Error", str(exc))

    def update_checkpoint(self, data: dict) -> None:
        plot_episode(
            self.episode_figure,
            data["simulated_depths"],
            data["depth_estimates"],
            data["inflow_rates"],
            data["target_depth"],
            data["lower_limit"],
            data["upper_limit"],
            data["max_inflow_rate"],
            f"Checkpoint after Episode {data['training_episode']}: {data['outcome']}",
        )
        self.episode_canvas.draw()

        self.outcome_card.setStyleSheet(outcome_style(data["outcome"]))
        self.outcome_title.setText(data["outcome"])
        set_manager_feedback(
            self.manager_label,
            data["outcome"],
            self.happy_pixmap,
            self.angry_pixmap,
        )
        self.outcome_details.setText(
            f"Starting depth: <b>{data['starting_depth']:.2f} cm</b><br>"
            f"Target range: <b>{data['lower_limit']:.2f} to "
            f"{data['upper_limit']:.2f} cm</b><br>"
            f"Final depth: <b>{data['final_depth']:.2f} cm</b><br>"
            f"Final inflow: <b>{data['final_inflow']:.2f} cm/hr</b><br>"
            f"Checkpoint reward: <b>{data['total_reward']:+.1f}</b><br>"
            f"Training random-action chance: "
            f"<b>{100.0 * data['random_action_chance']:.0f}%</b><br><br>"
            f"{ecological_interpretation(data['outcome'])}"
        )

    def update_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.progress_tab.set_status(message)

    def on_training_finished(self, stopped_early: bool) -> None:
        del stopped_early
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.episodes_input.setEnabled(True)
        self.uncertainty_input.setEnabled(True)
        self.save_button.setEnabled(
            self.training_thread is not None
            and self.training_thread.model is not None
        )


class EvaluationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.model: Optional[DQN] = None
        self.evaluation_thread: Optional[EvaluationThread] = None
        self.happy_pixmap = load_pixmap("crusty_happy.png")
        self.angry_pixmap = load_pixmap("crusty_angry.png")

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 10, 12, 10)
        root_layout.setSpacing(7)

        intro = QLabel(
            "Load a saved model, choose a North Marsh target, and test how reliably "
            "the policy reaches a stable water level with random actions turned off."
        )
        intro.setWordWrap(True)
        root_layout.addWidget(intro)

        controls_group = QGroupBox("Testing Setup")
        controls_layout = QHBoxLayout(controls_group)

        self.target_input = QComboBox()
        for target_depth in range(12, 25):
            self.target_input.addItem(
                f"{target_depth:.2f} cm",
                float(target_depth),
            )
        self.target_input.setCurrentIndex(self.target_input.findData(20.0))

        self.uncertainty_input = QComboBox()
        for uncertainty in (0.10, 0.50, 1.00):
            self.uncertainty_input.addItem(
                f"{uncertainty:.2f} cm",
                uncertainty,
            )
        self.uncertainty_input.setCurrentIndex(1)

        self.runs_input = QComboBox()
        for run_count in (5, 10, 15, 20):
            self.runs_input.addItem(
                f"{run_count} runs",
                run_count,
            )

        self.load_button = QPushButton("Load Model")
        self.load_button.setObjectName("secondaryButton")
        self.load_button.clicked.connect(self.load_model)

        self.run_button = QPushButton("Run Tests")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(self.run_tests)
        self.run_button.setEnabled(False)

        controls_layout.addWidget(QLabel("Target depth:"))
        controls_layout.addWidget(self.target_input)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(QLabel("Depth-measurement uncertainty (±):"))
        controls_layout.addWidget(self.uncertainty_input)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(QLabel("Number of runs:"))
        controls_layout.addWidget(self.runs_input)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.load_button)
        controls_layout.addWidget(self.run_button)
        root_layout.addWidget(controls_group)

        robustness_note = QLabel(
            "Changing the test uncertainty changes the measurement pattern received by "
            "the agent. A policy trained under one uncertainty level may respond "
            "differently at another level."
        )
        robustness_note.setWordWrap(True)
        robustness_note.setStyleSheet(f"color: {MISC_BROWN};")
        root_layout.addWidget(robustness_note)

        self.model_status = QLabel("No model loaded.")
        self.model_status.setStyleSheet(f"color: {MISC_BROWN}; font-weight: 700;")
        root_layout.addWidget(self.model_status)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setFixedHeight(340)

        run_group = QGroupBox("Most Recent Test Run")
        run_group.setFixedHeight(340)
        run_layout = QVBoxLayout(run_group)
        self.run_figure = Figure(figsize=(8.0, 4.7))
        self.run_canvas = FigureCanvas(self.run_figure)
        run_layout.addWidget(self.run_canvas)
        splitter.addWidget(run_group)

        self.summary_card = make_outcome_card()
        self.summary_card.setFixedSize(290, 340)
        summary_layout = QVBoxLayout(self.summary_card)

        self.manager_label = QLabel()
        self.manager_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.manager_label.setFixedSize(240, 128)
        summary_layout.addWidget(
            self.manager_label,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )

        self.summary_title = QLabel("No test results yet")
        self.summary_title.setFixedHeight(28)
        self.summary_title.setStyleSheet(
            f"font-size: 13pt; font-weight: 700; color: {COBBER_MAROON};"
        )
        self.summary_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        summary_layout.addWidget(self.summary_title)

        self.summary_details = QLabel(
            "Run several trials to evaluate consistency, underfilling, and overshoot risk."
        )
        self.summary_details.setWordWrap(True)
        self.summary_details.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.summary_details.setStyleSheet("font-size: 10.5pt;")
        summary_layout.addWidget(self.summary_details, 1)

        splitter.addWidget(self.summary_card)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, 290])
        root_layout.addWidget(splitter)

        results_group = QGroupBox("Run Results")
        results_group.setFixedHeight(205)
        results_layout = QVBoxLayout(results_group)
        self.results_table = QTableWidget(0, 6)
        self.results_table.setHorizontalHeaderLabels(
            ["Run", "Start depth", "Final depth", "Final inflow", "Error", "Outcome"]
        )
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setAlternatingRowColors(True)
        configure_read_only_table(self.results_table)
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.results_table.setFixedHeight(170)
        results_layout.addWidget(self.results_table)
        root_layout.addWidget(results_group)

        self.status_label = QLabel("Load a saved model to begin.")
        self.status_label.setStyleSheet(f"color: {MISC_BROWN}; font-weight: 700;")
        root_layout.addWidget(self.status_label)

    def load_model(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Trained Wetland Model",
            str(APP_ROOT),
            "Keras Weights (*.weights.h5)",
        )
        if not filepath:
            return

        try:
            model = DQN(3, 3)
            model(tf.zeros((1, 3), dtype=tf.float32))
            model.load_weights(filepath)
            self.model = model
            self.model_status.setText(
                f"Loaded model: {os.path.basename(filepath)}"
            )
            self.model_status.setStyleSheet(
                f"color: {PROJECT_GREEN}; font-weight: 700;"
            )
            self.run_button.setEnabled(True)
            self.status_label.setText("Model loaded. Choose the test settings.")
        except Exception as exc:
            self.model = None
            self.run_button.setEnabled(False)
            QMessageBox.critical(
                self,
                "Load Model Error",
                "The model could not be loaded. Models saved by the older four-input "
                f"app are not compatible with this version.\n\n{exc}",
            )

    def run_tests(self) -> None:
        if self.model is None:
            QMessageBox.warning(self, "Testing Setup", "Load a trained model first.")
            return

        self.run_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.results_table.setRowCount(0)
        self.summary_card.setStyleSheet(outcome_style("In progress"))
        self.summary_title.setText("Testing in progress")
        self.summary_details.setText("The model is completing the requested runs.")
        self.status_label.setText("Testing is in progress.")

        self.evaluation_thread = EvaluationThread(
            self.model,
            float(self.target_input.currentData()),
            float(self.uncertainty_input.currentData()),
            int(self.runs_input.currentData()),
        )
        self.evaluation_thread.status_message.connect(self.status_label.setText)
        self.evaluation_thread.update_run.connect(self.add_run_result)
        self.evaluation_thread.update_summary.connect(self.show_summary)
        self.evaluation_thread.evaluation_finished.connect(
            self.on_testing_finished
        )
        self.evaluation_thread.start()

    def add_run_result(self, result: dict) -> None:
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        values = [
            str(result["run"]),
            f"{result['starting_depth']:.2f} cm",
            f"{result['final_depth']:.2f} cm",
            f"{result['final_inflow']:.2f} cm/hr",
            f"{result['error']:+.2f} cm",
            result["outcome"],
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if column == 5:
                style_outcome_cell(item, result["outcome"])
            self.results_table.setItem(row, column, item)

        plot_episode(
            self.run_figure,
            result["simulated_depths"],
            result["depth_estimates"],
            result["inflow_rates"],
            result["target_depth"],
            result["lower_limit"],
            result["upper_limit"],
            result["max_inflow_rate"],
            f"Run {result['run']}: {result['outcome']}",
        )
        self.run_canvas.draw()

        self.summary_card.setStyleSheet(outcome_style(result["outcome"]))
        self.summary_title.setText(f"Run {result['run']}: {result['outcome']}")
        set_manager_feedback(
            self.manager_label,
            result["outcome"],
            self.happy_pixmap,
            self.angry_pixmap,
        )
        self.summary_details.setText(
            f"Starting depth: <b>{result['starting_depth']:.2f} cm</b><br>"
            f"Target range: <b>{result['lower_limit']:.2f} to "
            f"{result['upper_limit']:.2f} cm</b><br>"
            f"Final depth: <b>{result['final_depth']:.2f} cm</b><br>"
            f"Final inflow: <b>{result['final_inflow']:.2f} cm/hr</b>"
        )

    def show_summary(self, summary: dict) -> None:
        if summary["success_rate"] >= 80.0:
            summary_outcome = "Stable target"
            title = "Strong test performance"
        elif summary["overshoots"] > 0:
            summary_outcome = "Overshoot"
            title = "Overshoot risk remains"
        elif summary["underfills"] > 0:
            summary_outcome = "Below target"
            title = "Underfilling remains common"
        else:
            summary_outcome = "Below target"
            title = "The policy is not yet reliable"

        self.summary_card.setStyleSheet(outcome_style(summary_outcome))
        self.summary_title.setText(title)
        set_manager_feedback(
            self.manager_label,
            summary_outcome,
            self.happy_pixmap,
            self.angry_pixmap,
        )
        self.summary_details.setText(
            f"Successful runs: <b>{summary['successes']} of {summary['num_runs']}</b> "
            f"({summary['success_rate']:.0f}%)<br>"
            f"Average final depth: <b>{summary['average_final_depth']:.2f} cm</b><br>"
            f"Average error: <b>{summary['average_error']:+.2f} cm</b><br>"
            f"Average final inflow: <b>{summary['average_final_inflow']:.2f} cm/hr</b><br>"
            f"Below target: <b>{summary['underfills']}</b><br>"
            f"Overshoots: <b>{summary['overshoots']}</b>"
        )

    def on_testing_finished(self) -> None:
        self.run_button.setEnabled(self.model is not None)
        self.load_button.setEnabled(True)


class PolicyInspectionTab(QWidget):
    """Visualize the action with the largest Q-value across wetland states."""

    def __init__(self):
        super().__init__()
        self.model: Optional[DQN] = None
        self.loaded_filename = ""
        self.depth_values: Optional[np.ndarray] = None
        self.inflow_values: Optional[np.ndarray] = None
        self.action_grid: Optional[np.ndarray] = None
        self.q_value_grid: Optional[np.ndarray] = None
        self.map_axes = None
        self.selected_marker = None
        self.map_target_depth: Optional[float] = None

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 10, 12, 10)
        root_layout.setSpacing(7)

        intro = QLabel(
            "What did the DQN actually learn? Load a saved policy and examine which "
            "gate action has the highest estimated long-term value across combinations "
            "of depth estimate and inflow rate."
        )
        intro.setWordWrap(True)
        root_layout.addWidget(intro)

        controls_group = QGroupBox("Policy Map Setup")
        controls_layout = QHBoxLayout(controls_group)

        self.target_input = QComboBox()
        for target_depth in range(12, 25):
            self.target_input.addItem(
                f"{target_depth:.2f} cm",
                float(target_depth),
            )
        self.target_input.setCurrentIndex(self.target_input.findData(20.0))
        self.target_input.currentIndexChanged.connect(self.target_changed)

        self.load_button = QPushButton("Load Model")
        self.load_button.setObjectName("secondaryButton")
        self.load_button.clicked.connect(self.load_model)

        self.map_button = QPushButton("Build Policy Map")
        self.map_button.setObjectName("primaryButton")
        self.map_button.clicked.connect(self.build_policy_map)
        self.map_button.setEnabled(False)

        controls_layout.addWidget(QLabel("Target depth:"))
        controls_layout.addWidget(self.target_input)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.load_button)
        controls_layout.addWidget(self.map_button)
        root_layout.addWidget(controls_group)

        explanation = QLabel(
            "The neural network first estimates one Q-value for each action. The policy "
            "then chooses the action with the largest Q-value. The colored map shows that "
            "second step. Click the map to inspect all three Q-values for one state."
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet(f"color: {MISC_BROWN};")
        root_layout.addWidget(explanation)

        self.model_status = QLabel("No model loaded.")
        self.model_status.setStyleSheet(f"color: {MISC_BROWN}; font-weight: 700;")
        root_layout.addWidget(self.model_status)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        map_group = QGroupBox("Preferred Gate Action Across States")
        map_layout = QVBoxLayout(map_group)
        self.map_figure = Figure(figsize=(8.4, 5.5))
        self.map_canvas = FigureCanvas(self.map_figure)
        self.map_canvas.mpl_connect("button_press_event", self.map_clicked)
        map_layout.addWidget(self.map_canvas)
        splitter.addWidget(map_group)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(7)

        state_group = QGroupBox("Selected State")
        state_layout = QVBoxLayout(state_group)
        self.selected_title = QLabel("No state selected")
        self.selected_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.selected_title.setStyleSheet(
            f"font-size: 13pt; font-weight: 700; color: {COBBER_MAROON};"
        )
        state_layout.addWidget(self.selected_title)

        self.state_details = QLabel(
            "Load a model and build the map. Then click any colored region."
        )
        self.state_details.setWordWrap(True)
        self.state_details.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        state_layout.addWidget(self.state_details)
        right_layout.addWidget(state_group)

        q_group = QGroupBox("Q-Values for the Selected State")
        q_layout = QVBoxLayout(q_group)
        self.q_figure = Figure(figsize=(4.8, 4.2))
        self.q_canvas = FigureCanvas(self.q_figure)
        q_layout.addWidget(self.q_canvas)
        right_layout.addWidget(q_group, 1)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([820, 500])
        root_layout.addWidget(splitter, 1)

        self.status_label = QLabel("Load a saved model to inspect its policy.")
        self.status_label.setStyleSheet(f"color: {MISC_BROWN}; font-weight: 700;")
        root_layout.addWidget(self.status_label)

        self.show_map_message("Load a saved model to build the policy map.")
        self.show_q_message("Select a state to compare its three Q-values.")

    def show_map_message(self, message: str) -> None:
        self.map_figure.clear()
        ax = self.map_figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            wrap=True,
            color=MISC_BROWN,
        )
        ax.set_axis_off()
        self.map_figure.tight_layout()
        self.map_canvas.draw()
        self.map_axes = None
        self.selected_marker = None

    def show_q_message(self, message: str) -> None:
        self.q_figure.clear()
        ax = self.q_figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            wrap=True,
            color=MISC_BROWN,
        )
        ax.set_axis_off()
        self.q_figure.tight_layout()
        self.q_canvas.draw()

    def load_model(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Trained Wetland Model",
            str(APP_ROOT),
            "Keras Weights (*.weights.h5)",
        )
        if not filepath:
            return

        try:
            model = DQN(3, 3)
            model(tf.zeros((1, 3), dtype=tf.float32))
            model.load_weights(filepath)
            self.model = model
            self.loaded_filename = os.path.basename(filepath)
            self.model_status.setText(f"Loaded model: {self.loaded_filename}")
            self.model_status.setStyleSheet(
                f"color: {PROJECT_GREEN}; font-weight: 700;"
            )
            self.map_button.setEnabled(True)
            self.status_label.setText(
                "Model loaded. Building the policy map for the selected target."
            )
            self.build_policy_map()
        except Exception as exc:
            self.model = None
            self.loaded_filename = ""
            self.map_button.setEnabled(False)
            self.show_map_message("The selected model could not be loaded.")
            self.show_q_message("No Q-values are available.")
            QMessageBox.critical(
                self,
                "Load Model Error",
                "The model could not be loaded. Models saved by the older four-input "
                f"app are not compatible with this version.\n\n{exc}",
            )

    def target_changed(self) -> None:
        if self.model is None:
            return

        selected_target = float(self.target_input.currentData())
        if self.map_target_depth is None:
            self.status_label.setText(
                f"Target set to {selected_target:.2f} cm. Click Build Policy Map."
            )
            return

        if np.isclose(selected_target, self.map_target_depth):
            self.status_label.setText(
                f"The displayed map already uses a {self.map_target_depth:.2f} cm target."
            )
        else:
            self.status_label.setText(
                f"Target changed to {selected_target:.2f} cm. The displayed map still uses "
                f"{self.map_target_depth:.2f} cm. Click Build Policy Map to update it."
            )

    def build_policy_map(self) -> None:
        if self.model is None:
            QMessageBox.warning(
                self,
                "Policy Map Setup",
                "Load a trained model first.",
            )
            return

        target_depth = float(self.target_input.currentData())
        self.map_target_depth = target_depth
        env = WetlandControlEnv(depth_measurement_uncertainty=0.0)

        depth_min = max(0.0, target_depth - 8.0)
        depth_max = target_depth + 1.5
        self.depth_values = np.linspace(depth_min, depth_max, 96)
        self.inflow_values = np.linspace(
            env.min_inflow_rate,
            env.max_inflow_rate,
            61,
        )

        depth_grid, inflow_grid = np.meshgrid(
            self.depth_values,
            self.inflow_values,
        )
        states = np.column_stack(
            [
                depth_grid.ravel() / env.depth_scale,
                inflow_grid.ravel() / env.max_inflow_rate,
                np.full(depth_grid.size, target_depth / env.depth_scale),
            ]
        ).astype(np.float32)

        q_values = self.model(
            tf.convert_to_tensor(states, dtype=tf.float32),
            training=False,
        ).numpy()
        actions = np.argmax(q_values, axis=1)

        self.action_grid = actions.reshape(depth_grid.shape)
        self.q_value_grid = q_values.reshape(
            depth_grid.shape[0],
            depth_grid.shape[1],
            3,
        )

        self.map_figure.clear()
        ax = self.map_figure.add_subplot(111)
        self.map_axes = ax
        self.selected_marker = None

        cmap = ListedColormap(
            [ACTION_COLORS[0], ACTION_COLORS[1], ACTION_COLORS[2]]
        )
        ax.imshow(
            self.action_grid,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[
                float(self.depth_values[0]),
                float(self.depth_values[-1]),
                float(self.inflow_values[0]),
                float(self.inflow_values[-1]),
            ],
            cmap=cmap,
            vmin=-0.5,
            vmax=2.5,
        )

        lower_limit = target_depth - env.depth_tolerance
        upper_limit = target_depth + env.depth_tolerance
        ax.axvspan(
            lower_limit,
            upper_limit,
            color=HIGHLIGHT_GOLD,
            alpha=0.14,
            hatch="//",
            label="Acceptable depth range",
        )
        ax.axvline(
            target_depth,
            color=CHARCOAL,
            linestyle="--",
            linewidth=1.5,
        )

        legend_handles = [
            Patch(facecolor=ACTION_COLORS[0], label="Close slightly"),
            Patch(facecolor=ACTION_COLORS[1], label="Hold setting"),
            Patch(facecolor=ACTION_COLORS[2], label="Open slightly"),
            Patch(
                facecolor="white",
                edgecolor=HIGHLIGHT_GOLD,
                hatch="//",
                label="Acceptable depth range",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            fontsize=8,
            framealpha=0.95,
        )
        ax.set_xlabel("Depth estimate received by the DQN (cm)")
        ax.set_ylabel("Current inflow rate (cm/hr)")
        ax.set_title(
            f"Highest-Q Action at a {target_depth:.0f} cm Target",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(alpha=0.12)
        self.map_figure.tight_layout()
        self.map_canvas.draw()

        counts = np.bincount(actions, minlength=3)
        total_states = int(actions.size)
        self.status_label.setText(
            f"Policy map built from {total_states:,} states. "
            f"Close: {100.0 * counts[0] / total_states:.0f}% | "
            f"Hold: {100.0 * counts[1] / total_states:.0f}% | "
            f"Open: {100.0 * counts[2] / total_states:.0f}%. "
            "Click the map to inspect one state."
        )

        # Begin with a representative state near the target while water is moving.
        self.select_state(target_depth - 1.0, 0.75)

    def map_clicked(self, event) -> None:
        if (
            self.model is None
            or self.map_axes is None
            or event.inaxes is not self.map_axes
            or event.xdata is None
            or event.ydata is None
        ):
            return
        self.select_state(float(event.xdata), float(event.ydata))

    def select_state(self, depth_estimate: float, inflow_rate: float) -> None:
        if (
            self.model is None
            or self.depth_values is None
            or self.inflow_values is None
            or self.q_value_grid is None
            or self.map_axes is None
        ):
            return

        depth_index = int(
            np.argmin(np.abs(self.depth_values - float(depth_estimate)))
        )
        inflow_index = int(
            np.argmin(np.abs(self.inflow_values - float(inflow_rate)))
        )

        selected_depth = float(self.depth_values[depth_index])
        selected_inflow = float(self.inflow_values[inflow_index])
        q_values = np.asarray(
            self.q_value_grid[inflow_index, depth_index, :],
            dtype=float,
        )
        best_action = int(np.argmax(q_values))
        sorted_q = np.sort(q_values)
        action_gap = float(sorted_q[-1] - sorted_q[-2])
        if self.map_target_depth is None:
            return
        target_depth = float(self.map_target_depth)

        if self.selected_marker is not None:
            try:
                self.selected_marker.remove()
            except ValueError:
                pass
        self.selected_marker = self.map_axes.plot(
            selected_depth,
            selected_inflow,
            marker="o",
            markersize=11,
            markerfacecolor="none",
            markeredgecolor="#000000",
            markeredgewidth=2.0,
            zorder=10,
        )[0]
        self.map_canvas.draw_idle()

        lower_limit = target_depth - 0.50
        upper_limit = target_depth + 0.50
        if selected_depth < lower_limit:
            depth_position = "below the acceptable range"
        elif selected_depth > upper_limit:
            depth_position = "above the acceptable range"
        else:
            depth_position = "inside the acceptable range"

        action_label = WetlandControlEnv.ACTION_LABELS[best_action]
        self.selected_title.setText(action_label)
        self.state_details.setText(
            f"Depth estimate: <b>{selected_depth:.2f} cm</b><br>"
            f"Current inflow: <b>{selected_inflow:.2f} cm/hr</b><br>"
            f"Target depth: <b>{target_depth:.2f} cm</b><br>"
            f"Location: <b>{depth_position}</b><br>"
            f"Gap between the two highest Q-values: <b>{action_gap:.3f}</b>"
        )

        self.q_figure.clear()
        ax = self.q_figure.add_subplot(111)
        action_names = ["Close", "Hold", "Open"]
        bars = ax.barh(
            action_names,
            q_values,
            color=[ACTION_COLORS[0], ACTION_COLORS[1], ACTION_COLORS[2]],
        )
        ax.axvline(0.0, color=CHARCOAL, linewidth=0.8)
        ax.set_xlabel("Estimated Q-value")
        ax.set_title(
            f"Largest value: {action_names[best_action]}",
            fontsize=10,
            fontweight="bold",
            pad=8,
        )
        ax.grid(axis="x", alpha=0.20)

        q_min = float(np.min(q_values))
        q_max = float(np.max(q_values))
        q_span = max(q_max - q_min, 0.25)
        axis_min = min(0.0, q_min) - 0.20 * q_span
        axis_max = max(0.0, q_max) + 0.20 * q_span
        ax.set_xlim(axis_min, axis_max)

        label_offset = 0.025 * (axis_max - axis_min)
        for index, bar in enumerate(bars):
            value = float(q_values[index])
            if value >= 0:
                x_position = value + label_offset
                horizontal_alignment = "left"
            else:
                x_position = value - label_offset
                horizontal_alignment = "right"
            ax.text(
                x_position,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha=horizontal_alignment,
                fontsize=9,
                fontweight="bold" if index == best_action else "normal",
                clip_on=False,
            )

        self.q_figure.subplots_adjust(
            left=0.16,
            right=0.95,
            bottom=0.20,
            top=0.82,
        )
        self.q_canvas.draw()


class CobberEcoHydroApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CobberEcoHydro: North Marsh Water-Level Control")
        self.resize(1360, 760)
        self.setMinimumSize(1040, 680)
        self.setFont(QFont("Lato", 10))

        self.tabs = QTabWidget()
        self.progress_tab = TrainingProgressTab()
        self.training_tab = TrainingTab(self.progress_tab)
        self.testing_tab = EvaluationTab()
        self.policy_tab = PolicyInspectionTab()

        self.tabs.addTab(self.training_tab, "Model Training")
        self.tabs.addTab(self.progress_tab, "Training Progress")
        self.tabs.addTab(self.testing_tab, "Model Testing")
        self.tabs.addTab(self.policy_tab, "Policy Map")
        self.setCentralWidget(self.tabs)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_app_stylesheet(app)
    window = CobberEcoHydroApp()
    window.show()
    sys.exit(app.exec())

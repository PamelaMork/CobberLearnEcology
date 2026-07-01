# CobberEcoCNN_Rebuild_Tabs1_2_3_4.py
# Rebuilt CobberEcoCNN app, Tabs 1, 2, 3, and 4.
#
# Tab 1: Image as Numbers
#   Track image -> pixel grid -> number grid
#
# Tab 2: Filter Scan
#   Choose an edge filter.
#   Watch the 3x3 filter scan the image grid.
#   Show the product matrix after multiplication.
#   Show the convolutional score.
#   After a short delay, write the convolutional score into the raw feature map.
#   Completed raw feature maps are autosaved for later tabs.
#
# Tab 3: ReLU Activation
#   Choose an autosaved raw feature map.
#   Apply ReLU cell by cell.
#   Show Before ReLU and After ReLU.
#   Toggle heatmap display for the activated feature map.
#   Completed ReLU feature maps are autosaved for later tabs.
#
# Tab 4: Feature Stack
#   Silently precomputes all four activated feature maps for all three tracks.
#   Shows the four-filter feature stack, a combined max response map,
#   side-by-side combined maps for all tracks, and a feature signature table.

import sys
from typing import Dict

import numpy as np

from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------
# Simplified animal-track glyphs
# ---------------------------------------------------------------------
# These 11x11 binary images are teaching glyphs.
# black pixel = 1
# white pixel = 0

IMAGES: Dict[str, np.ndarray] = {
    "Hoof Track": np.zeros((11, 11), dtype=float),
    "Bird Track": np.zeros((11, 11), dtype=float),
    "Paw Track": np.zeros((11, 11), dtype=float),
}

# Hoof Track: two elongated hoof marks with slight separation.
hoof = IMAGES["Hoof Track"]
hoof[2:9, 3] = 1
hoof[2:9, 4] = 1
hoof[2:9, 6] = 1
hoof[2:9, 7] = 1
hoof[2, 4] = 0
hoof[2, 6] = 0
hoof[8, 3] = 0
hoof[8, 7] = 0

# Bird Track: three forward toes plus a small rear toe.
bird = IMAGES["Bird Track"]
bird[6, 5] = 1
bird[5, 5] = 1
bird[2:6, 5] = 1
for y, x in [(5, 4), (4, 3), (3, 2)]:
    bird[y, x] = 1
for y, x in [(5, 6), (4, 7), (3, 8)]:
    bird[y, x] = 1
bird[7:10, 5] = 1

# Paw Track: central pad plus toe marks.
paw = IMAGES["Paw Track"]
paw[6:9, 4:7] = 1
paw[5, 5] = 1
paw[8, 4] = 0
paw[8, 6] = 0
paw[2:4, 2:4] = 1
paw[1:3, 4:6] = 1
paw[2:4, 7:9] = 1
paw[4:6, 1:3] = 1
paw[4:6, 8:10] = 1


# ---------------------------------------------------------------------
# Edge filters
# ---------------------------------------------------------------------
# These are fixed teaching filters.
# In a trained CNN, these entries would be learned weights.

FILTERS: Dict[str, np.ndarray] = {
    "Vertical Edge": np.array(
        [[1, 0, -1],
         [2, 0, -2],
         [1, 0, -1]],
        dtype=float,
    ),
    "Horizontal Edge": np.array(
        [[1, 2, 1],
         [0, 0, 0],
         [-1, -2, -1]],
        dtype=float,
    ),
    "Junction / Sharp Change": np.array(
        [[-1, -1, -1],
         [-1, 8, -1],
         [-1, -1, -1]],
        dtype=float,
    ),
    "Diagonal Edge": np.array(
        [[-1, -1, 2],
         [-1, 2, -1],
         [2, -1, -1]],
        dtype=float,
    ),
}

FILTER_DESCRIPTIONS = {
    "Vertical Edge": "This filter compares the left side of a patch with the right side.",
    "Horizontal Edge": "This filter compares the top of a patch with the bottom.",
    "Junction / Sharp Change": "This filter responds when the center differs strongly from its neighbors.",
    "Diagonal Edge": "This filter responds to diagonal contrast inside the patch.",
}


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()


class MplGridCanvas(FigureCanvas):
    def __init__(self, parent=None, nrows=1, ncols=1, width=6, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.subplots(nrows, ncols)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()


class CobberEcoCNNApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.cobber_maroon = "#6C1D45"
        self.cobber_gold = "#EAAA00"
        self.infoblue = "#2E6B8E"
        self.medium_gray = "#B0B0B0"
        self.light_grid_gray = "#C8C8C8"
        self.inactive_tab_gray = "#666666"

        self.setWindowTitle("CobberEcoCNN")
        self.setGeometry(100, 100, 1450, 800)
        self.setFont(QFont("Lato", 10))

        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.cobber_maroon};
                color: white;
                font-weight: bold;
                border: 1px solid {self.cobber_maroon};
                border-radius: 4px;
                padding: 6px 10px;
            }}

            QPushButton:hover {{
                background-color: #7A2550;
                border: 1px solid #7A2550;
            }}

            QPushButton:pressed {{
                background-color: #541637;
                border: 1px solid #541637;
            }}

            QPushButton:disabled {{
                background-color: #666666;
                color: white;
                font-weight: bold;
                border: 1px solid #666666;
            }}
            """
        )

        # Shared autosave storage for all completed feature maps.
        self.saved_feature_maps = {}
        self.latest_saved_raw_map_name = None
        self.latest_saved_relu_map_name = None

        # Tab 1 state
        self.current_image_name = "Hoof Track"
        self.current_image = IMAGES[self.current_image_name]
        self.mapping_has_run = False
        self.current_view_mode = "Track image"
        self.animation_stage = 0
        self.number_index = 0

        self.map_timer = QTimer(self)
        self.map_timer.timeout.connect(self.mapping_animation_step)

        # Tab 2 state
        self.scan_image_name = "Hoof Track"
        self.scan_image = IMAGES[self.scan_image_name]
        self.scan_filter_name = "Vertical Edge"
        self.scan_filter = FILTERS[self.scan_filter_name]

        self.scan_timer = QTimer(self)
        self.scan_timer.timeout.connect(self.filter_scan_step)

        self.scan_y = 0
        self.scan_x = 0
        self.scan_phase = "show_products"
        self.scan_paused = False

        self.raw_feature_map = np.zeros((9, 9), dtype=float)
        self.raw_feature_filled = np.zeros((9, 9), dtype=bool)
        self.scan_has_run = False
        self.current_products = np.zeros((3, 3), dtype=float)
        self.current_raw_score = 0.0

        self.pending_y = 0
        self.pending_x = 0
        self.pending_raw_score = 0.0

        # The patch currently shown in the Patch × Filter display.
        # This keeps Pause/Resume aligned with the right panel.
        self.current_patch_y = 0
        self.current_patch_x = 0
        self.raw_feature_highlight_pos = None

        # Tab 3 state
        self.selected_raw_map_name = None
        self.selected_raw_map_data = np.zeros((9, 9), dtype=float)
        self.relu_feature_map = np.zeros((9, 9), dtype=float)
        self.relu_feature_filled = np.zeros((9, 9), dtype=bool)
        self.relu_index = 0
        self.relu_has_run = False

        self.relu_positive_kept = 0
        self.relu_negative_zeroed = 0
        self.relu_zeros_kept = 0

        self.relu_timer = QTimer(self)
        self.relu_timer.timeout.connect(self.relu_animation_step)

        # Tab 4 state.
        # Precompute once at startup for reliability. Tab 4 does not depend on
        # students manually scanning all track/filter combinations.
        self.feature_stack_track_name = None
        self.feature_signature_revealed = False
        self.feature_cache = {}
        self.build_feature_cache()

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            f"""
            QTabWidget::pane {{
                border: none;
                top: -1px;
            }}

            QTabBar::tab {{
                background: {self.inactive_tab_gray};
                color: white;
                padding: 8px 18px;
                border: 1px solid {self.inactive_tab_gray};
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                margin-right: 2px;
                font-weight: bold;
            }}

            QTabBar::tab:selected {{
                background: {self.cobber_maroon};
                color: white;
                border: 1px solid {self.cobber_maroon};
                border-bottom: none;
                font-weight: bold;
            }}

            QTabBar::tab:!selected {{
                background: {self.inactive_tab_gray};
                color: white;
                border: 1px solid {self.inactive_tab_gray};
                border-bottom: none;
            }}

            QTabBar::tab:!selected:hover {{
                background: #777777;
                border: 1px solid #777777;
                border-bottom: none;
            }}
            """
        )

        self.setCentralWidget(self.tabs)

        self.image_grid_tab = QWidget()
        self.filter_scan_tab = QWidget()
        self.relu_tab = QWidget()
        self.feature_stack_tab = QWidget()

        self.tabs.addTab(self.image_grid_tab, "Image Grid")
        self.tabs.addTab(self.filter_scan_tab, "Filter Scan")
        self.tabs.addTab(self.relu_tab, "ReLU Activation")
        self.tabs.addTab(self.feature_stack_tab, "Feature Stack")

        self.build_image_grid_tab()
        self.build_filter_scan_tab()
        self.build_relu_tab()
        self.build_feature_stack_tab()

        self.draw_image_grid()
        self.draw_selected_filter()
        self.draw_scan_image_grid()
        self.draw_product_matrix(empty=True)
        self.draw_raw_feature_map()
        self.refresh_relu_source_selector()
        self.draw_relu_raw_map(empty=True)
        self.draw_relu_output_map(empty=True)
        self.draw_blank_feature_stack()
        self.draw_blank_feature_summary()

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def color_for_value(self, value):
        if value > 0:
            return self.cobber_maroon
        if value < 0:
            return self.infoblue
        return self.medium_gray

    def set_placeholder_font(self, label, size=12):
        label.setFont(QFont("Lato", size))
        label.setStyleSheet("color: #777777;")

    def set_raw_score_label(self, value=None):
        if value is None:
            self.raw_score_label.setText("Convolutional score =")
            return

        color = self.color_for_value(value)
        self.raw_score_label.setText(
            f"<span style='font-weight:bold;'>Convolutional score = </span>"
            f"<span style='font-weight:bold; color:{color};'>{value:.0f}</span>"
        )

    def raw_map_name(self, track_type, filter_name):
        return f"{track_type} + {filter_name} + Raw"

    def relu_map_name(self, track_type, filter_name):
        return f"{track_type} + {filter_name} + ReLU"

    def autosave_completed_raw_feature_map(self):
        map_name = self.raw_map_name(self.scan_image_name, self.scan_filter_name)

        self.saved_feature_maps[map_name] = {
            "track_type": self.scan_image_name,
            "filter_name": self.scan_filter_name,
            "map_stage": "Raw",
            "data": self.raw_feature_map.copy(),
        }

        self.latest_saved_raw_map_name = map_name

        self.saved_map_label.setText(
            f"Saved raw feature map: "
            f"<span style='font-weight:bold; color:{self.cobber_maroon};'>{map_name}</span>"
        )

        self.refresh_relu_source_selector(select_map_name=map_name)

    def autosave_completed_relu_feature_map(self):
        if self.selected_raw_map_name is None:
            return

        source_record = self.saved_feature_maps.get(self.selected_raw_map_name)
        if source_record is None:
            return

        track_type = source_record["track_type"]
        filter_name = source_record["filter_name"]
        map_name = self.relu_map_name(track_type, filter_name)

        self.saved_feature_maps[map_name] = {
            "track_type": track_type,
            "filter_name": filter_name,
            "map_stage": "ReLU",
            "source_map_name": self.selected_raw_map_name,
            "data": self.relu_feature_map.copy(),
        }

        self.latest_saved_relu_map_name = map_name

    def get_saved_raw_map_names(self):
        return [
            name for name, record in self.saved_feature_maps.items()
            if record.get("map_stage") == "Raw"
        ]

    def compute_raw_feature_map(self, image, filter_matrix):
        output_height = image.shape[0] - 2
        output_width = image.shape[1] - 2
        raw_map = np.zeros((output_height, output_width), dtype=float)

        for y in range(output_height):
            for x in range(output_width):
                patch = image[y:y + 3, x:x + 3]
                raw_map[y, x] = np.sum(patch * filter_matrix)

        return raw_map

    def apply_relu_array(self, raw_map):
        return np.maximum(0, raw_map)

    def build_feature_cache(self):
        self.feature_cache = {}

        for track_name, image in IMAGES.items():
            track_record = {
                "track_image": image.copy(),
                "filters": {},
                "combined_max": None,
            }

            relu_maps = []

            for filter_name, filter_matrix in FILTERS.items():
                raw_map = self.compute_raw_feature_map(image, filter_matrix)
                relu_map = self.apply_relu_array(raw_map)

                track_record["filters"][filter_name] = {
                    "raw": raw_map,
                    "relu": relu_map,
                    "signature": int(np.sum(relu_map)),
                }

                relu_maps.append(relu_map)

            track_record["combined_max"] = np.maximum.reduce(relu_maps)
            self.feature_cache[track_name] = track_record

    # ------------------------------------------------------------------
    # Tab 1 layout
    # ------------------------------------------------------------------

    def build_image_grid_tab(self):
        main_layout = QHBoxLayout(self.image_grid_tab)

        controls_panel = QFrame()
        controls_panel.setFrameShape(QFrame.Shape.StyledPanel)
        controls_panel.setMinimumWidth(330)
        controls_layout = QVBoxLayout(controls_panel)

        display_panel = QFrame()
        display_panel.setFrameShape(QFrame.Shape.StyledPanel)
        display_layout = QVBoxLayout(display_panel)

        main_layout.addWidget(controls_panel, 1)
        main_layout.addWidget(display_panel, 3)

        title = QLabel("<h2>Image as Numbers</h2>")
        title.setWordWrap(True)
        controls_layout.addWidget(title)

        intro = QLabel(
            "A CNN does not begin with an animal track the way you see it. "
            "It begins with pixel values arranged in a grid."
        )
        intro.setWordWrap(True)
        controls_layout.addWidget(intro)

        controls_layout.addSpacing(12)

        controls_layout.addWidget(QLabel("<h3>Choose a Track Image</h3>"))

        self.track_selector = QComboBox()
        self.track_selector.addItems(IMAGES.keys())
        self.track_selector.currentTextChanged.connect(self.change_track_image)
        controls_layout.addWidget(self.track_selector)

        controls_layout.addSpacing(12)

        controls_layout.addWidget(QLabel("<h3>Map the Image</h3>"))

        self.map_button = QPushButton("Map Image to Numbers")
        self.map_button.clicked.connect(self.start_mapping_animation)
        controls_layout.addWidget(self.map_button)

        self.reset_button = QPushButton("Reset Mapping")
        self.reset_button.clicked.connect(self.reset_mapping)
        self.reset_button.setEnabled(False)
        controls_layout.addWidget(self.reset_button)

        controls_layout.addSpacing(20)

        self.view_title = QLabel("<h3>View After Mapping</h3>")
        self.view_title.setWordWrap(True)
        controls_layout.addWidget(self.view_title)

        self.view_group_box = QGroupBox()
        view_layout = QVBoxLayout(self.view_group_box)

        self.view_button_group = QButtonGroup(self)

        self.track_only_radio = QRadioButton("Track image")
        self.image_grid_radio = QRadioButton("Image + grid")
        self.number_grid_radio = QRadioButton("Number grid")
        self.both_radio = QRadioButton("Image + grid + numbers")

        self.track_only_radio.setChecked(True)

        for button in [
            self.track_only_radio,
            self.image_grid_radio,
            self.number_grid_radio,
            self.both_radio,
        ]:
            self.view_button_group.addButton(button)
            view_layout.addWidget(button)
            button.setEnabled(False)
            button.toggled.connect(self.update_view_mode)

        controls_layout.addWidget(self.view_group_box)

        controls_layout.addSpacing(44)

        key_box = QGroupBox("Pixel values")
        key_layout = QVBoxLayout(key_box)
        key_layout.addWidget(QLabel("black pixel = 1"))
        key_layout.addWidget(QLabel("white pixel = 0"))
        controls_layout.addWidget(key_box)

        controls_layout.addStretch()

        self.display_title = QLabel("<h2>Track image</h2>")
        self.display_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        display_layout.addWidget(self.display_title)

        self.canvas = MplCanvas(self, width=7, height=7, dpi=100)
        display_layout.addWidget(self.canvas)

        self.caption = QLabel(
            "Choose a track image. Then click Map Image to Numbers."
        )
        self.caption.setWordWrap(True)
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        display_layout.addWidget(self.caption)

    # ------------------------------------------------------------------
    # Tab 2 layout
    # ------------------------------------------------------------------

    def build_filter_scan_tab(self):
        main_layout = QHBoxLayout(self.filter_scan_tab)

        controls_panel = QFrame()
        controls_panel.setFrameShape(QFrame.Shape.StyledPanel)
        controls_panel.setMinimumWidth(330)
        controls_layout = QVBoxLayout(controls_panel)

        center_panel = QFrame()
        center_panel.setFrameShape(QFrame.Shape.StyledPanel)
        center_layout = QVBoxLayout(center_panel)

        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.Shape.StyledPanel)
        right_layout = QVBoxLayout(right_panel)

        main_layout.addWidget(controls_panel, 1)
        main_layout.addWidget(center_panel, 2)
        main_layout.addWidget(right_panel, 2)

        title = QLabel("<h2>Filter Scan</h2>")
        title.setWordWrap(True)
        controls_layout.addWidget(title)

        intro = QLabel(
            "A filter is a small grid of weights. It scans the image one "
            "patch at a time and writes one convolutional score into the feature map."
        )
        intro.setWordWrap(True)
        controls_layout.addWidget(intro)

        controls_layout.addSpacing(8)

        controls_layout.addWidget(QLabel("<h3>Choose a Track Image</h3>"))

        self.scan_track_selector = QComboBox()
        self.scan_track_selector.addItems(IMAGES.keys())
        self.scan_track_selector.currentTextChanged.connect(self.change_scan_track_image)
        controls_layout.addWidget(self.scan_track_selector)

        controls_layout.addSpacing(8)

        controls_layout.addWidget(QLabel("<h3>Choose an Edge Filter</h3>"))

        self.filter_selector = QComboBox()
        self.filter_selector.addItems(FILTERS.keys())
        self.filter_selector.currentTextChanged.connect(self.change_scan_filter)
        controls_layout.addWidget(self.filter_selector)

        self.filter_details_frame = QFrame()
        self.filter_details_frame.setFrameShape(QFrame.Shape.StyledPanel)
        filter_details_layout = QVBoxLayout(self.filter_details_frame)

        filter_details_layout.addWidget(QLabel("<h3>Selected Filter</h3>"))

        self.filter_matrix_canvas = MplCanvas(self, width=2.1, height=1.9, dpi=100)
        filter_details_layout.addWidget(self.filter_matrix_canvas)

        self.filter_description_label = QLabel(FILTER_DESCRIPTIONS[self.scan_filter_name])
        self.filter_description_label.setWordWrap(True)
        filter_details_layout.addWidget(self.filter_description_label)

        weight_key_box = QGroupBox("Filter weight key")
        weight_key_layout = QVBoxLayout(weight_key_box)

        positive_label = QLabel(
            f"<span style='color:{self.cobber_maroon}; font-weight:bold;'>positive weights</span>"
            " = one side of the edge"
        )
        negative_label = QLabel(
            f"<span style='color:{self.infoblue}; font-weight:bold;'>negative weights</span>"
            " = the other side of the edge"
        )
        zero_label = QLabel(
            f"<span style='color:{self.medium_gray}; font-weight:bold;'>zero weights</span>"
            " = ignored"
        )

        weight_key_layout.addWidget(positive_label)
        weight_key_layout.addWidget(negative_label)
        weight_key_layout.addWidget(zero_label)

        filter_details_layout.addWidget(weight_key_box)

        controls_layout.addWidget(self.filter_details_frame)

        controls_layout.addSpacing(8)

        controls_layout.addWidget(QLabel("<h3>Run the Scan</h3>"))

        self.start_scan_button = QPushButton("Start Filter Scan")
        self.start_scan_button.clicked.connect(self.start_filter_scan)
        controls_layout.addWidget(self.start_scan_button)

        pause_reset_layout = QHBoxLayout()

        self.pause_scan_button = QPushButton("Pause Scan")
        self.pause_scan_button.clicked.connect(self.toggle_scan_pause)
        self.pause_scan_button.setEnabled(False)
        pause_reset_layout.addWidget(self.pause_scan_button)

        self.reset_scan_button = QPushButton("Reset Scan")
        self.reset_scan_button.clicked.connect(self.reset_filter_scan)
        self.reset_scan_button.setEnabled(False)
        pause_reset_layout.addWidget(self.reset_scan_button)

        controls_layout.addLayout(pause_reset_layout)

        controls_layout.addStretch()

        self.scan_display_title = QLabel("<h2>Image Grid with Moving Filter</h2>")
        self.scan_display_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(self.scan_display_title)

        self.scan_image_canvas = MplCanvas(self, width=5.2, height=5.2, dpi=100)
        center_layout.addWidget(self.scan_image_canvas)

        self.scan_caption = QLabel(
            "Click Start Filter Scan to move the 3 by 3 filter across the image."
        )
        self.scan_caption.setWordWrap(True)
        self.scan_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(self.scan_caption)

        self.product_title = QLabel("<h2>Current Products</h2>")
        self.product_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.product_title)

        self.product_canvas = MplCanvas(self, width=4.4, height=2.9, dpi=100)
        right_layout.addWidget(self.product_canvas)

        self.raw_score_label = QLabel("Convolutional score =")
        self.raw_score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.raw_score_label.setFont(QFont("Lato", 13, QFont.Weight.Bold))
        right_layout.addWidget(self.raw_score_label)

        self.feature_map_title = QLabel("<h2>Raw Feature Map</h2>")
        self.feature_map_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.feature_map_title)

        self.raw_feature_canvas = MplCanvas(self, width=4.8, height=4.8, dpi=100)
        right_layout.addWidget(self.raw_feature_canvas)

        self.feature_caption = QLabel(
            "Each convolutional score becomes one cell in the raw feature map."
        )
        self.feature_caption.setWordWrap(True)
        self.feature_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.feature_caption)

        self.saved_map_label = QLabel("No raw feature map saved yet.")
        self.saved_map_label.setWordWrap(True)
        self.saved_map_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.saved_map_label)

    # ------------------------------------------------------------------
    # Tab 3 layout
    # ------------------------------------------------------------------

    def build_relu_tab(self):
        main_layout = QHBoxLayout(self.relu_tab)

        controls_panel = QFrame()
        controls_panel.setFrameShape(QFrame.Shape.StyledPanel)
        controls_panel.setMinimumWidth(340)
        controls_layout = QVBoxLayout(controls_panel)

        raw_panel = QFrame()
        raw_panel.setFrameShape(QFrame.Shape.StyledPanel)
        raw_layout = QVBoxLayout(raw_panel)

        activated_panel = QFrame()
        activated_panel.setFrameShape(QFrame.Shape.StyledPanel)
        activated_layout = QVBoxLayout(activated_panel)

        main_layout.addWidget(controls_panel, 1)
        main_layout.addWidget(raw_panel, 2)
        main_layout.addWidget(activated_panel, 2)

        title = QLabel("<h2>ReLU Activation</h2>")
        title.setWordWrap(True)
        controls_layout.addWidget(title)

        intro = QLabel(
            "ReLU does not scan the original image again. It works on the "
            "raw feature map from the filter scan."
        )
        intro.setWordWrap(True)
        controls_layout.addWidget(intro)

        controls_layout.addSpacing(10)

        controls_layout.addWidget(QLabel("<h3>Choose a Saved Raw Feature Map</h3>"))

        self.relu_source_selector = QComboBox()
        self.relu_source_selector.currentTextChanged.connect(self.change_relu_source_map)
        controls_layout.addWidget(self.relu_source_selector)

        controls_layout.addSpacing(10)

        controls_layout.addWidget(QLabel("<h3>Apply Activation</h3>"))

        self.apply_relu_button = QPushButton("Apply ReLU")
        self.apply_relu_button.clicked.connect(self.start_relu_animation)
        self.apply_relu_button.setEnabled(False)
        controls_layout.addWidget(self.apply_relu_button)

        self.reset_relu_button = QPushButton("Reset Activation")
        self.reset_relu_button.clicked.connect(self.reset_relu_activation)
        self.reset_relu_button.setEnabled(False)
        controls_layout.addWidget(self.reset_relu_button)

        self.heatmap_checkbox = QCheckBox("Show Heatmap")
        self.heatmap_checkbox.setEnabled(False)
        self.heatmap_checkbox.toggled.connect(self.toggle_relu_heatmap)
        controls_layout.addWidget(self.heatmap_checkbox)

        heatmap_note = QLabel(
            "The heatmap does not change the feature map. It only uses color "
            "to show stronger responses."
        )
        heatmap_note.setWordWrap(True)
        controls_layout.addWidget(heatmap_note)

        controls_layout.addSpacing(12)

        rule_box = QGroupBox("ReLU rule")
        rule_layout = QVBoxLayout(rule_box)

        rule_layout.addWidget(QLabel(
            f"<span style='color:{self.cobber_maroon}; font-weight:bold;'>positive score</span>"
            " → keep it"
        ))
        rule_layout.addWidget(QLabel(
            f"<span style='color:{self.medium_gray}; font-weight:bold;'>zero score</span>"
            " → keep 0"
        ))
        rule_layout.addWidget(QLabel(
            f"<span style='color:{self.infoblue}; font-weight:bold;'>negative score</span>"
            " → change to 0"
        ))

        controls_layout.addWidget(rule_box)

        controls_layout.addSpacing(12)

        decision_box = QGroupBox("Current decision")
        decision_layout = QVBoxLayout(decision_box)

        self.relu_raw_decision_label = QLabel("Convolutional score =")
        self.relu_output_decision_label = QLabel("ReLU output =")

        self.relu_raw_decision_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.relu_output_decision_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        decision_layout.addWidget(self.relu_raw_decision_label)
        decision_layout.addWidget(self.relu_output_decision_label)

        controls_layout.addWidget(decision_box)

        controls_layout.addSpacing(12)

        summary_box = QGroupBox("ReLU summary")
        summary_layout = QVBoxLayout(summary_box)

        self.relu_positive_summary_label = QLabel("Positive scores kept: —")
        self.relu_negative_summary_label = QLabel("Negative scores changed to 0: —")
        self.relu_zero_summary_label = QLabel("Zeros kept: —")

        summary_layout.addWidget(self.relu_positive_summary_label)
        summary_layout.addWidget(self.relu_negative_summary_label)
        summary_layout.addWidget(self.relu_zero_summary_label)

        controls_layout.addWidget(summary_box)

        controls_layout.addStretch()



        self.relu_raw_title = QLabel("<h2>Before ReLU: Raw Feature Map</h2>")
        self.relu_raw_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        raw_layout.addWidget(self.relu_raw_title)

        self.relu_raw_canvas = MplCanvas(self, width=5.4, height=5.4, dpi=100)
        raw_layout.addWidget(self.relu_raw_canvas)

        self.relu_raw_caption = QLabel(
            "Choose a saved raw feature map from Tab 2."
        )
        self.relu_raw_caption.setWordWrap(True)
        self.relu_raw_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        raw_layout.addWidget(self.relu_raw_caption)

        self.relu_output_title = QLabel("<h2>After ReLU: Activated Feature Map</h2>")
        self.relu_output_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        activated_layout.addWidget(self.relu_output_title)

        self.relu_output_canvas = MplCanvas(self, width=5.4, height=5.4, dpi=100)
        activated_layout.addWidget(self.relu_output_canvas)

        self.relu_output_caption = QLabel(
            "Click Apply ReLU to build the activated feature map cell by cell."
        )
        self.relu_output_caption.setWordWrap(True)
        self.relu_output_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        activated_layout.addWidget(self.relu_output_caption)

    # ------------------------------------------------------------------
    # Tab 4 layout
    # ------------------------------------------------------------------

    def build_feature_stack_tab(self):
        main_layout = QHBoxLayout(self.feature_stack_tab)

        controls_panel = QFrame()
        controls_panel.setFrameShape(QFrame.Shape.StyledPanel)
        controls_panel.setMinimumWidth(340)
        controls_layout = QVBoxLayout(controls_panel)

        stack_panel = QFrame()
        stack_panel.setFrameShape(QFrame.Shape.StyledPanel)
        stack_layout = QVBoxLayout(stack_panel)

        summary_panel = QFrame()
        summary_panel.setFrameShape(QFrame.Shape.StyledPanel)
        summary_layout = QVBoxLayout(summary_panel)

        main_layout.addWidget(controls_panel, 1)
        main_layout.addWidget(stack_panel, 2)
        main_layout.addWidget(summary_panel, 2)

        title = QLabel("<h2>Feature Stack</h2>")
        title.setWordWrap(True)
        controls_layout.addWidget(title)

        intro = QLabel(
            "This tab has already applied all four filters to each track. "
            "Choose a track to inspect its feature stack. Then reveal the "
            "feature signature."
        )
        intro.setWordWrap(True)
        controls_layout.addWidget(intro)

        controls_layout.addSpacing(10)

        controls_layout.addWidget(QLabel("<h3>Choose a Track</h3>"))

        self.feature_stack_track_selector = QComboBox()
        self.feature_stack_track_selector.addItem("Choose a track...")
        self.feature_stack_track_selector.addItems(IMAGES.keys())
        self.feature_stack_track_selector.currentTextChanged.connect(
            self.change_feature_stack_track
        )
        controls_layout.addWidget(self.feature_stack_track_selector)

        controls_layout.addSpacing(8)

        self.reveal_signature_button = QPushButton("Reveal Feature Signature")
        self.reveal_signature_button.clicked.connect(self.reveal_feature_signature)
        self.reveal_signature_button.setEnabled(False)
        controls_layout.addWidget(self.reveal_signature_button)

        controls_layout.addSpacing(12)

        note_box = QGroupBox("What this tab shows")
        note_layout = QVBoxLayout(note_box)

        note_text = QLabel(
            "A single heatmap may not look like a hoof, bird, or paw to our eyes. "
            "The computer can still use the pattern of responses across filters "
            "as a feature signature."
        )
        note_text.setWordWrap(True)
        note_layout.addWidget(note_text)

        note_text_2 = QLabel(
            "The combined map keeps the strongest activated response at each "
            "location. This is a teaching summary of the stack."
        )
        note_text_2.setWordWrap(True)
        note_layout.addWidget(note_text_2)

        controls_layout.addWidget(note_box)

        controls_layout.addSpacing(12)

        scale_note = QLabel(
            "Heatmap scale: purple shows low response; yellow shows high response."
        )
        scale_note.setWordWrap(True)
        controls_layout.addWidget(scale_note)

        controls_layout.addStretch()

        # Middle panel: blank prompt first, then the selected track's stack.
        self.feature_stack_title = QLabel("<h2>Four Activated Feature Maps</h2>")
        self.feature_stack_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stack_layout.addWidget(self.feature_stack_title)

        self.feature_stack_placeholder = QLabel(
            "<h3>Choose a track to build its feature stack.</h3>"
            "<p>The original image and four activated feature maps will appear here.</p>"
        )
        self.feature_stack_placeholder.setWordWrap(True)
        self.feature_stack_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feature_stack_placeholder.setStyleSheet(
            "QLabel { color: #777777; background: #F7F7F7; "
            "border: 1px solid #DDDDDD; border-radius: 8px; padding: 28px; }"
        )
        stack_layout.addWidget(self.feature_stack_placeholder, 1)

        self.feature_stack_visual_frame = QFrame()
        self.feature_stack_visual_frame.setFrameShape(QFrame.Shape.NoFrame)
        feature_visual_layout = QVBoxLayout(self.feature_stack_visual_frame)
        feature_visual_layout.setContentsMargins(0, 0, 0, 0)
        feature_visual_layout.setSpacing(4)

        self.feature_original_canvas = MplCanvas(self, width=2.7, height=2.15, dpi=100)
        feature_visual_layout.addWidget(self.feature_original_canvas)

        self.feature_stack_canvas = MplGridCanvas(
            self, nrows=2, ncols=2, width=5.9, height=4.7, dpi=100
        )
        feature_visual_layout.addWidget(self.feature_stack_canvas)

        self.feature_stack_caption = QLabel(
            "Each heatmap is one activated feature map from the same track image."
        )
        self.feature_stack_caption.setWordWrap(True)
        self.feature_stack_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feature_stack_caption.setFont(QFont("Lato", 11))
        feature_visual_layout.addWidget(self.feature_stack_caption)

        stack_layout.addWidget(self.feature_stack_visual_frame, 5)
        self.feature_stack_visual_frame.setVisible(False)

        # Right panel: blank prompt first, then combined maps and signature table.
        self.combined_title = QLabel("<h2>Feature Signature Reveal</h2>")
        self.combined_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        summary_layout.addWidget(self.combined_title)

        self.feature_summary_placeholder = QLabel(
            "<h3>Reveal the feature signature when you are ready.</h3>"
            "<p>The combined response map, all-track comparison, and signature table will appear here.</p>"
        )
        self.feature_summary_placeholder.setWordWrap(True)
        self.feature_summary_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feature_summary_placeholder.setStyleSheet(
            "QLabel { color: #777777; background: #F7F7F7; "
            "border: 1px solid #DDDDDD; border-radius: 8px; padding: 28px; }"
        )
        summary_layout.addWidget(self.feature_summary_placeholder, 1)

        self.feature_summary_visual_frame = QFrame()
        self.feature_summary_visual_frame.setFrameShape(QFrame.Shape.NoFrame)
        summary_visual_layout = QVBoxLayout(self.feature_summary_visual_frame)
        summary_visual_layout.setContentsMargins(0, 0, 0, 0)
        summary_visual_layout.setSpacing(4)

        self.combined_canvas = MplCanvas(self, width=4.3, height=3.15, dpi=100)
        summary_visual_layout.addWidget(self.combined_canvas)

        self.comparison_title = QLabel("<h3>All Track Comparison</h3>")
        self.comparison_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        summary_visual_layout.addWidget(self.comparison_title)

        self.all_combined_canvas = MplGridCanvas(
            self, nrows=1, ncols=3, width=5.9, height=1.75, dpi=100
        )
        summary_visual_layout.addWidget(self.all_combined_canvas)

        self.signature_title = QLabel("<h3>Feature Signature Summary</h3>")
        self.signature_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        summary_visual_layout.addWidget(self.signature_title)

        self.signature_table_label = QLabel()
        self.signature_table_label.setWordWrap(True)
        self.signature_table_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        summary_visual_layout.addWidget(self.signature_table_label)

        self.feature_summary_caption = QLabel(
            "Each table value is the total activated response for one filter."
        )
        self.feature_summary_caption.setWordWrap(True)
        self.feature_summary_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feature_summary_caption.setFont(QFont("Lato", 11))
        summary_visual_layout.addWidget(self.feature_summary_caption)

        summary_layout.addWidget(self.feature_summary_visual_frame, 5)
        self.feature_summary_visual_frame.setVisible(False)

    # ------------------------------------------------------------------
    # Tab 1 interaction methods
    # ------------------------------------------------------------------

    def change_track_image(self, image_name):
        self.current_image_name = image_name
        self.current_image = IMAGES[image_name]
        self.reset_mapping()

    def start_mapping_animation(self):
        self.mapping_has_run = False
        self.animation_stage = 1
        self.number_index = 0

        self.map_button.setEnabled(False)
        self.reset_button.setEnabled(False)

        for button in self.view_button_group.buttons():
            button.setEnabled(False)

        self.current_view_mode = "Track image"
        self.track_only_radio.setChecked(True)

        self.caption.setText("The image is made from small square pixels.")
        self.draw_image_grid(show_image=True, show_grid=False, show_numbers=False)

        self.map_timer.start(350)

    def mapping_animation_step(self):
        if self.animation_stage == 1:
            self.caption.setText("The grid shows the pixel positions in the image.")
            self.draw_image_grid(show_image=True, show_grid=True, show_numbers=False)
            self.animation_stage = 2
            return

        if self.animation_stage == 2:
            self.caption.setText(
                "Each black pixel is stored as 1. Each white pixel is stored as 0."
            )

            self.number_index += 8
            total_cells = self.current_image.size

            self.draw_image_grid(
                show_image=True,
                show_grid=True,
                show_numbers=True,
                numbers_to_show=min(self.number_index, total_cells),
            )

            if self.number_index >= total_cells:
                self.animation_stage = 3

            return

        if self.animation_stage == 3:
            self.map_timer.stop()

            self.mapping_has_run = True
            self.current_view_mode = "Image + grid + numbers"
            self.both_radio.setChecked(True)

            self.map_button.setText("Replay Mapping")
            self.map_button.setEnabled(True)
            self.reset_button.setEnabled(True)

            for button in self.view_button_group.buttons():
                button.setEnabled(True)

            self.caption.setText(
                "The picture and the number grid show the same information "
                "in two forms."
            )

            self.draw_image_grid(show_image=True, show_grid=True, show_numbers=True)
            return

    def reset_mapping(self):
        self.map_timer.stop()

        self.mapping_has_run = False
        self.animation_stage = 0
        self.number_index = 0
        self.current_view_mode = "Track image"

        self.map_button.setText("Map Image to Numbers")
        self.map_button.setEnabled(True)
        self.reset_button.setEnabled(False)

        self.track_only_radio.setChecked(True)

        for button in self.view_button_group.buttons():
            button.setEnabled(False)

        self.caption.setText(
            "Choose a track image. Then click Map Image to Numbers."
        )

        self.draw_image_grid(show_image=True, show_grid=False, show_numbers=False)

    def update_view_mode(self):
        if not self.mapping_has_run:
            return

        selected_button = self.view_button_group.checkedButton()
        if selected_button is None:
            return

        self.current_view_mode = selected_button.text()

        if self.current_view_mode == "Track image":
            self.caption.setText("This is the track image as a person sees it.")
            self.draw_image_grid(show_image=True, show_grid=False, show_numbers=False)

        elif self.current_view_mode == "Image + grid":
            self.caption.setText("The grid shows the pixel positions in the image.")
            self.draw_image_grid(show_image=True, show_grid=True, show_numbers=False)

        elif self.current_view_mode == "Number grid":
            self.caption.setText(
                "The CNN receives the image as a grid of numbers."
            )
            self.draw_image_grid(show_image=False, show_grid=True, show_numbers=True)

        elif self.current_view_mode == "Image + grid + numbers":
            self.caption.setText(
                "The picture and the number grid show the same information "
                "in two forms."
            )
            self.draw_image_grid(show_image=True, show_grid=True, show_numbers=True)

    # ------------------------------------------------------------------
    # Tab 2 interaction methods
    # ------------------------------------------------------------------

    def change_scan_track_image(self, image_name):
        self.scan_image_name = image_name
        self.scan_image = IMAGES[image_name]
        self.reset_filter_scan()

    def change_scan_filter(self, filter_name):
        self.scan_filter_name = filter_name
        self.scan_filter = FILTERS[filter_name]
        self.filter_description_label.setText(FILTER_DESCRIPTIONS[filter_name])
        self.draw_selected_filter()
        self.reset_filter_scan()

    def start_filter_scan(self):
        self.scan_timer.stop()

        self.scan_y = 0
        self.scan_x = 0
        self.scan_phase = "show_products"
        self.scan_paused = False

        self.raw_feature_map = np.zeros((9, 9), dtype=float)
        self.raw_feature_filled = np.zeros((9, 9), dtype=bool)
        self.current_products = np.zeros((3, 3), dtype=float)
        self.current_raw_score = 0.0

        self.pending_y = self.scan_y
        self.pending_x = self.scan_x
        self.pending_raw_score = 0.0

        self.current_patch_y = self.scan_y
        self.current_patch_x = self.scan_x
        self.raw_feature_highlight_pos = None

        self.scan_has_run = False

        self.start_scan_button.setEnabled(False)
        self.pause_scan_button.setText("Pause Scan")
        self.pause_scan_button.setEnabled(True)
        self.reset_scan_button.setEnabled(True)
        self.scan_track_selector.setEnabled(False)
        self.filter_selector.setEnabled(False)

        self.set_raw_score_label(None)

        self.draw_scan_image_grid(scan_box_pos=(self.scan_x, self.scan_y))
        self.draw_product_matrix(empty=True)
        self.draw_raw_feature_map()

        self.scan_timer.start(80)

    def toggle_scan_pause(self):
        if self.scan_has_run:
            return

        if self.scan_paused:
            self.scan_paused = False
            self.pause_scan_button.setText("Pause Scan")
            self.scan_caption.setText(
                "The filter is scanning one 3 by 3 image patch at a time."
            )

            self.draw_scan_image_grid(
                scan_box_pos=(self.current_patch_x, self.current_patch_y)
            )
            self.draw_raw_feature_map(
                highlight_pos=(self.current_patch_x, self.current_patch_y)
            )

            self.scan_timer.start(80)

        else:
            self.scan_paused = True
            self.pause_scan_button.setText("Resume Scan")
            self.scan_timer.stop()

            self.draw_scan_image_grid(
                scan_box_pos=(self.current_patch_x, self.current_patch_y)
            )
            self.draw_raw_feature_map(
                highlight_pos=(self.current_patch_x, self.current_patch_y)
            )

            self.scan_caption.setText(
                "Scan paused. The highlighted image patch matches the Patch × Filter "
                "display, convolutional score, and highlighted raw feature-map cell."
            )

    def filter_scan_step(self):
        if self.scan_phase == "show_products":
            patch = self.scan_image[
                self.scan_y:self.scan_y + 3,
                self.scan_x:self.scan_x + 3,
            ]

            products = patch * self.scan_filter
            raw_score = np.sum(products)

            self.current_products = products
            self.current_raw_score = raw_score
            self.current_patch_y = self.scan_y
            self.current_patch_x = self.scan_x

            self.pending_y = self.scan_y
            self.pending_x = self.scan_x
            self.pending_raw_score = raw_score
            self.raw_feature_highlight_pos = (self.pending_x, self.pending_y)

            self.draw_scan_image_grid(
                scan_box_pos=(self.current_patch_x, self.current_patch_y)
            )
            self.draw_product_matrix(empty=False)
            self.set_raw_score_label(raw_score)
            self.draw_raw_feature_map(highlight_pos=self.raw_feature_highlight_pos)

            self.scan_caption.setText(
                "The filter multiplies with the selected patch. "
                "The convolutional score is the sum of the product values."
            )

            self.scan_phase = "write_score"
            return

        if self.scan_phase == "write_score":
            self.raw_feature_map[self.pending_y, self.pending_x] = self.pending_raw_score
            self.raw_feature_filled[self.pending_y, self.pending_x] = True
            self.raw_feature_highlight_pos = (self.pending_x, self.pending_y)

            self.draw_scan_image_grid(
                scan_box_pos=(self.current_patch_x, self.current_patch_y)
            )
            self.draw_raw_feature_map(highlight_pos=self.raw_feature_highlight_pos)



            self.scan_x += 1

            if self.scan_x >= self.raw_feature_map.shape[1]:
                self.scan_x = 0
                self.scan_y += 1

            if self.scan_y >= self.raw_feature_map.shape[0]:
                self.scan_timer.stop()
                self.finish_filter_scan()
                return

            self.scan_phase = "show_products"
            return

    def finish_filter_scan(self):
        self.scan_has_run = True
        self.scan_paused = False

        self.autosave_completed_raw_feature_map()

        self.start_scan_button.setText("Replay Scan")
        self.start_scan_button.setEnabled(True)
        self.pause_scan_button.setText("Pause Scan")
        self.pause_scan_button.setEnabled(False)
        self.reset_scan_button.setEnabled(True)
        self.scan_track_selector.setEnabled(True)
        self.filter_selector.setEnabled(True)

        self.scan_caption.setText(
            "The scan is complete. The raw feature map has been saved for the next tab."
        )

        self.draw_scan_image_grid(scan_box_pos=None)

    def reset_filter_scan(self):
        self.scan_timer.stop()

        self.scan_y = 0
        self.scan_x = 0
        self.scan_phase = "show_products"
        self.scan_paused = False

        self.raw_feature_map = np.zeros((9, 9), dtype=float)
        self.raw_feature_filled = np.zeros((9, 9), dtype=bool)
        self.current_products = np.zeros((3, 3), dtype=float)
        self.current_raw_score = 0.0
        self.pending_y = 0
        self.pending_x = 0
        self.pending_raw_score = 0.0

        self.current_patch_y = 0
        self.current_patch_x = 0
        self.raw_feature_highlight_pos = None

        self.scan_has_run = False

        self.start_scan_button.setText("Start Filter Scan")
        self.start_scan_button.setEnabled(True)
        self.pause_scan_button.setText("Pause Scan")
        self.pause_scan_button.setEnabled(False)
        self.reset_scan_button.setEnabled(False)
        self.scan_track_selector.setEnabled(True)
        self.filter_selector.setEnabled(True)

        self.scan_caption.setText(
            "Click Start Filter Scan to move the 3 by 3 filter across the image."
        )
        self.set_raw_score_label(None)

        self.draw_selected_filter()
        self.draw_scan_image_grid(scan_box_pos=None)
        self.draw_product_matrix(empty=True)
        self.draw_raw_feature_map()

    # ------------------------------------------------------------------
    # Tab 3 interaction methods
    # ------------------------------------------------------------------

    def refresh_relu_source_selector(self, select_map_name=None):
        if not hasattr(self, "relu_source_selector"):
            return

        current = select_map_name or self.selected_raw_map_name

        self.relu_source_selector.blockSignals(True)
        self.relu_source_selector.clear()

        raw_names = self.get_saved_raw_map_names()

        if not raw_names:
            self.relu_source_selector.addItem("No saved raw feature maps yet")
            self.relu_source_selector.setEnabled(False)
            self.apply_relu_button.setEnabled(False)
            self.selected_raw_map_name = None
        else:
            self.relu_source_selector.setEnabled(True)
            for name in raw_names:
                self.relu_source_selector.addItem(name)

            if current in raw_names:
                self.relu_source_selector.setCurrentText(current)
                self.selected_raw_map_name = current
            else:
                self.selected_raw_map_name = raw_names[-1]
                self.relu_source_selector.setCurrentText(self.selected_raw_map_name)

        self.relu_source_selector.blockSignals(False)

        self.load_selected_relu_source()

    def change_relu_source_map(self, map_name):
        if map_name == "No saved raw feature maps yet":
            return

        self.selected_raw_map_name = map_name
        self.load_selected_relu_source()

    def load_selected_relu_source(self):
        self.relu_timer.stop()

        if self.selected_raw_map_name is None:
            self.selected_raw_map_data = np.zeros((9, 9), dtype=float)
            self.relu_feature_map = np.zeros((9, 9), dtype=float)
            self.relu_feature_filled = np.zeros((9, 9), dtype=bool)
            self.relu_has_run = False

            self.apply_relu_button.setEnabled(False)
            self.reset_relu_button.setEnabled(False)
            self.heatmap_checkbox.setEnabled(False)
            self.heatmap_checkbox.setChecked(False)

            self.draw_relu_raw_map(empty=True)
            self.draw_relu_output_map(empty=True)
            return

        record = self.saved_feature_maps.get(self.selected_raw_map_name)
        if record is None:
            return

        self.selected_raw_map_data = record["data"].copy()

        self.relu_feature_map = np.zeros_like(self.selected_raw_map_data)
        self.relu_feature_filled = np.zeros_like(self.selected_raw_map_data, dtype=bool)
        self.relu_index = 0
        self.relu_has_run = False

        self.apply_relu_button.setText("Apply ReLU")
        self.apply_relu_button.setEnabled(True)
        self.reset_relu_button.setEnabled(False)
        self.heatmap_checkbox.setEnabled(False)
        self.heatmap_checkbox.setChecked(False)

        self.relu_positive_kept = 0
        self.relu_negative_zeroed = 0
        self.relu_zeros_kept = 0

        self.relu_raw_decision_label.setText("Convolutional score =")
        self.relu_output_decision_label.setText("ReLU output =")
        self.relu_positive_summary_label.setText("Positive scores kept: —")
        self.relu_negative_summary_label.setText("Negative scores changed to 0: —")
        self.relu_zero_summary_label.setText("Zeros kept: —")


        self.relu_raw_caption.setText(
            f"Selected raw feature map: {self.selected_raw_map_name}"
        )
        self.relu_output_caption.setText(
            "Click Apply ReLU to build the activated feature map cell by cell."
        )

        self.draw_relu_raw_map(empty=False)
        self.draw_relu_output_map(empty=True)

    def start_relu_animation(self):
        if self.selected_raw_map_name is None:
            return

        self.relu_timer.stop()

        self.relu_feature_map = np.zeros_like(self.selected_raw_map_data)
        self.relu_feature_filled = np.zeros_like(self.selected_raw_map_data, dtype=bool)
        self.relu_index = 0
        self.relu_has_run = False

        self.relu_positive_kept = 0
        self.relu_negative_zeroed = 0
        self.relu_zeros_kept = 0

        self.apply_relu_button.setEnabled(False)
        self.reset_relu_button.setEnabled(False)
        self.relu_source_selector.setEnabled(False)
        self.heatmap_checkbox.setEnabled(False)
        self.heatmap_checkbox.setChecked(False)


        self.relu_positive_summary_label.setText("Positive scores kept: —")
        self.relu_negative_summary_label.setText("Negative scores changed to 0: —")
        self.relu_zero_summary_label.setText("Zeros kept: —")

        self.draw_relu_raw_map(empty=False, highlight_index=0)
        self.draw_relu_output_map(empty=True)

        self.relu_output_caption.setText(
            "ReLU is changing the raw feature map one cell at a time."
        )

        self.relu_timer.start(80)

    def relu_animation_step(self):
        h, w = self.selected_raw_map_data.shape

        if self.relu_index >= h * w:
            self.relu_timer.stop()
            self.finish_relu_activation()
            return

        y = self.relu_index // w
        x = self.relu_index % w

        raw_value = self.selected_raw_map_data[y, x]
        relu_value = max(0, raw_value)

        self.relu_feature_map[y, x] = relu_value
        self.relu_feature_filled[y, x] = True

        if raw_value > 0:
            self.relu_positive_kept += 1
        elif raw_value < 0:
            self.relu_negative_zeroed += 1
        else:
            self.relu_zeros_kept += 1

        raw_color = self.color_for_value(raw_value)
        output_color = self.color_for_value(relu_value)

        self.relu_raw_decision_label.setText(
            f"<span style='font-weight:bold;'>Convolutional score = </span>"
            f"<span style='font-weight:bold; color:{raw_color};'>{raw_value:.0f}</span>"
        )
        self.relu_output_decision_label.setText(
            f"<span style='font-weight:bold;'>ReLU output = </span>"
            f"<span style='font-weight:bold; color:{output_color};'>{relu_value:.0f}</span>"
        )

        self.draw_relu_raw_map(empty=False, highlight_index=self.relu_index)
        self.draw_relu_output_map(empty=False, highlight_index=self.relu_index)

        self.relu_index += 1

    def finish_relu_activation(self):
        self.relu_has_run = True

        self.apply_relu_button.setText("Replay ReLU")
        self.apply_relu_button.setEnabled(True)
        self.reset_relu_button.setEnabled(True)
        self.relu_source_selector.setEnabled(True)
        self.heatmap_checkbox.setEnabled(True)

        self.relu_positive_summary_label.setText(
            f"Positive scores kept: {self.relu_positive_kept}"
        )
        self.relu_negative_summary_label.setText(
            f"Negative scores changed to 0: {self.relu_negative_zeroed}"
        )
        self.relu_zero_summary_label.setText(
            f"Zeros kept: {self.relu_zeros_kept}"
        )

        self.relu_output_caption.setText(
            "ReLU is complete. Negative values are now 0, and positive values remain."
        )

        self.autosave_completed_relu_feature_map()

        self.draw_relu_raw_map(empty=False)
        self.draw_relu_output_map(empty=False)

    def reset_relu_activation(self):
        self.relu_timer.stop()
        self.load_selected_relu_source()

    def toggle_relu_heatmap(self, checked):
        if not self.relu_has_run:
            return

        self.draw_relu_output_map(empty=False, heatmap=checked)

        if checked:
            self.relu_output_caption.setText(
                "Heatmap view uses color to show stronger activated responses. \nYellow areas show the highest responses. \nPurple areas show the lowest responses. "
                "\nThe feature map values have not changed."
            )
        else:
            self.relu_output_caption.setText(
                "Number view shows the activated feature map after ReLU."
            )

    # ------------------------------------------------------------------
    # Tab 4 interaction methods
    # ------------------------------------------------------------------

    def change_feature_stack_track(self, track_name):
        if track_name == "Choose a track...":
            self.feature_stack_track_name = None
            self.feature_signature_revealed = False
            self.reveal_signature_button.setEnabled(False)
            self.draw_blank_feature_stack()
            self.draw_blank_feature_summary()
            return

        self.feature_stack_track_name = track_name
        self.feature_signature_revealed = False
        self.reveal_signature_button.setEnabled(True)
        self.draw_selected_feature_stack()
        self.draw_blank_feature_summary()

    def reveal_feature_signature(self):
        if self.feature_stack_track_name is None:
            return

        self.feature_signature_revealed = True
        self.draw_selected_combined_map()
        self.draw_all_combined_maps()
        self.update_signature_table()
        self.feature_summary_caption.setText(
            "The three combined maps and the table show different feature signatures "
            "for the three track types."
        )

    # ------------------------------------------------------------------
    # Tab 1 drawing method
    # ------------------------------------------------------------------

    def draw_image_grid(
        self,
        show_image=True,
        show_grid=False,
        show_numbers=False,
        numbers_to_show=None,
    ):
        data = self.current_image
        h, w = data.shape

        self.canvas.axes.clear()

        if show_image:
            self.canvas.axes.imshow(data, cmap="gray_r", vmin=0, vmax=1)
        else:
            self.canvas.axes.imshow(
                np.ones_like(data),
                cmap="gray",
                vmin=0,
                vmax=1,
            )

        if show_grid or show_numbers:
            for y in range(h):
                for x in range(w):
                    pixel_value = data[y, x]

                    if show_image:
                        facecolor = "none"
                        grid_edge_color = "black"
                        grid_linewidth = 0.8
                    else:
                        facecolor = "white"
                        grid_edge_color = self.light_grid_gray
                        grid_linewidth = 0.7

                    self.canvas.axes.add_patch(
                        Rectangle(
                            (x - 0.5, y - 0.5),
                            1,
                            1,
                            facecolor=facecolor,
                            edgecolor=grid_edge_color,
                            linewidth=grid_linewidth,
                        )
                    )

                    if show_numbers:
                        if numbers_to_show is not None:
                            cell_number = y * w + x
                            if cell_number >= numbers_to_show:
                                continue

                        value = int(pixel_value)

                        if not show_image:
                            if value == 1:
                                text_color = self.cobber_maroon
                                font_size = 18
                                font_weight = "bold"
                            else:
                                text_color = self.medium_gray
                                font_size = 13
                                font_weight = "normal"
                        else:
                            text_color = "white" if value == 1 else "black"
                            font_size = 15
                            font_weight = "bold"

                        self.canvas.axes.text(
                            x,
                            y,
                            str(value),
                            ha="center",
                            va="center",
                            fontsize=font_size,
                            fontweight=font_weight,
                            color=text_color,
                        )

        self.display_title.setText(f"<h2>{self.current_image_name}</h2>")

        self.canvas.axes.set_xticks([])
        self.canvas.axes.set_yticks([])
        self.canvas.axes.set_xlim(-0.5, w - 0.5)
        self.canvas.axes.set_ylim(h - 0.5, -0.5)
        self.canvas.axes.set_aspect("equal")

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Tab 2 drawing methods
    # ------------------------------------------------------------------

    def draw_selected_filter(self):
        kernel = self.scan_filter
        ax = self.filter_matrix_canvas.axes
        ax.clear()

        ax.imshow(np.ones_like(kernel), cmap="gray", vmin=0, vmax=1)

        h, w = kernel.shape

        for y in range(h):
            for x in range(w):
                value = int(kernel[y, x])

                ax.add_patch(
                    Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        facecolor="white",
                        edgecolor=self.light_grid_gray,
                        linewidth=0.8,
                    )
                )

                if value > 0:
                    text_color = self.cobber_maroon
                    font_weight = "bold"
                elif value < 0:
                    text_color = self.infoblue
                    font_weight = "bold"
                else:
                    text_color = self.medium_gray
                    font_weight = "normal"

                ax.text(
                    x,
                    y,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight=font_weight,
                    color=text_color,
                )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect("equal")
        self.filter_matrix_canvas.fig.tight_layout()
        self.filter_matrix_canvas.draw()

    def draw_scan_image_grid(self, scan_box_pos=None):
        data = self.scan_image
        h, w = data.shape

        ax = self.scan_image_canvas.axes
        ax.clear()

        ax.imshow(data, cmap="gray_r", vmin=0, vmax=1)

        for y in range(h):
            for x in range(w):
                value = int(data[y, x])

                ax.add_patch(
                    Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        facecolor="none",
                        edgecolor="black",
                        linewidth=0.6,
                    )
                )

                text_color = "white" if value == 1 else self.medium_gray
                font_size = 9 if value == 0 else 11
                font_weight = "normal" if value == 0 else "bold"

                ax.text(
                    x,
                    y,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    fontweight=font_weight,
                    color=text_color,
                )

        if scan_box_pos is not None:
            x, y = scan_box_pos

            ax.add_patch(
                Rectangle(
                    (x - 0.5, y - 0.5),
                    3,
                    3,
                    facecolor=self.cobber_gold,
                    edgecolor=self.cobber_maroon,
                    linewidth=2.5,
                    alpha=0.25,
                )
            )

            ax.add_patch(
                Rectangle(
                    (x - 0.5, y - 0.5),
                    3,
                    3,
                    facecolor="none",
                    edgecolor=self.cobber_maroon,
                    linewidth=2.5,
                )
            )

        self.scan_display_title.setText(f"<h2>{self.scan_image_name}</h2>")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect("equal")

        self.scan_image_canvas.fig.tight_layout()
        self.scan_image_canvas.draw()

    def draw_product_matrix(self, empty=False):
        ax = self.product_canvas.axes
        ax.clear()

        if empty:
            products = np.zeros((3, 3), dtype=float)
        else:
            products = self.current_products

        h, w = products.shape

        ax.imshow(np.ones_like(products), cmap="gray", vmin=0, vmax=1)

        for y in range(h):
            for x in range(w):
                value = products[y, x]

                ax.add_patch(
                    Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        facecolor="white",
                        edgecolor=self.light_grid_gray,
                        linewidth=0.9,
                    )
                )

                if empty:
                    text = ""
                    text_color = self.medium_gray
                    font_weight = "normal"
                else:
                    text = f"{value:.0f}"

                    if value > 0:
                        text_color = self.cobber_maroon
                        font_weight = "bold"
                    elif value < 0:
                        text_color = self.infoblue
                        font_weight = "bold"
                    else:
                        text_color = self.medium_gray
                        font_weight = "normal"

                ax.text(
                    x,
                    y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=22,
                    fontweight=font_weight,
                    color=text_color,
                )

        ax.set_title("Patch × Filter", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect("equal")

        self.product_canvas.fig.tight_layout()
        self.product_canvas.draw()

    def draw_raw_feature_map(self, highlight_pos=None):
        data = self.raw_feature_map
        filled = self.raw_feature_filled
        h, w = data.shape

        ax = self.raw_feature_canvas.axes
        ax.clear()

        ax.imshow(np.ones_like(data), cmap="gray", vmin=0, vmax=1)

        for y in range(h):
            for x in range(w):
                ax.add_patch(
                    Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        facecolor="white",
                        edgecolor=self.light_grid_gray,
                        linewidth=0.8,
                    )
                )

                if not filled[y, x]:
                    continue

                value = data[y, x]

                if value > 0:
                    text_color = self.cobber_maroon
                    font_weight = "bold"
                elif value < 0:
                    text_color = self.infoblue
                    font_weight = "bold"
                else:
                    text_color = self.medium_gray
                    font_weight = "normal"

                ax.text(
                    x,
                    y,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight=font_weight,
                    color=text_color,
                )

        if highlight_pos is not None:
            highlight_x, highlight_y = highlight_pos

            ax.add_patch(
                Rectangle(
                    (highlight_x - 0.5, highlight_y - 0.5),
                    1,
                    1,
                    facecolor="none",
                    edgecolor=self.cobber_gold,
                    linewidth=3,
                )
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect("equal")

        self.raw_feature_canvas.fig.tight_layout()
        self.raw_feature_canvas.draw()

    # ------------------------------------------------------------------
    # Tab 3 drawing methods
    # ------------------------------------------------------------------

    def draw_relu_raw_map(self, empty=False, highlight_index=None):
        ax = self.relu_raw_canvas.axes
        ax.clear()

        if empty:
            data = np.zeros((9, 9), dtype=float)
        else:
            data = self.selected_raw_map_data

        self.draw_value_grid_on_axes(
            ax=ax,
            data=data,
            filled=None if not empty else np.zeros_like(data, dtype=bool),
            highlight_index=highlight_index,
            show_negative=True,
            title=None,
            heatmap=False,
        )

        self.relu_raw_canvas.fig.tight_layout()
        self.relu_raw_canvas.draw()

    def draw_relu_output_map(self, empty=False, highlight_index=None, heatmap=False):
        ax = self.relu_output_canvas.axes
        ax.clear()

        if empty:
            data = np.zeros((9, 9), dtype=float)
            filled = np.zeros((9, 9), dtype=bool)
        else:
            data = self.relu_feature_map
            filled = self.relu_feature_filled

        self.draw_value_grid_on_axes(
            ax=ax,
            data=data,
            filled=filled,
            highlight_index=highlight_index,
            show_negative=False,
            title=None,
            heatmap=heatmap,
        )

        self.relu_output_canvas.fig.tight_layout()
        self.relu_output_canvas.draw()

    def draw_value_grid_on_axes(
        self,
        ax,
        data,
        filled=None,
        highlight_index=None,
        show_negative=True,
        title=None,
        heatmap=False,
    ):
        h, w = data.shape

        ax.clear()

        if heatmap:
            vmax = max(1.0, float(np.max(data)))
            ax.imshow(data, cmap="viridis", vmin=0, vmax=vmax)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-0.5, w - 0.5)
            ax.set_ylim(h - 0.5, -0.5)
            ax.set_aspect("equal")
            return

        ax.imshow(np.ones_like(data), cmap="gray", vmin=0, vmax=1)

        for y in range(h):
            for x in range(w):
                ax.add_patch(
                    Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        facecolor="white",
                        edgecolor=self.light_grid_gray,
                        linewidth=0.8,
                    )
                )

                if filled is not None and not filled[y, x]:
                    continue

                value = data[y, x]

                if value > 0:
                    text_color = self.cobber_maroon
                    font_weight = "bold"
                elif value < 0:
                    text_color = self.infoblue if show_negative else self.medium_gray
                    font_weight = "bold" if show_negative else "normal"
                else:
                    text_color = self.medium_gray
                    font_weight = "normal"

                ax.text(
                    x,
                    y,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight=font_weight,
                    color=text_color,
                )

        if highlight_index is not None:
            y = highlight_index // w
            x = highlight_index % w

            ax.add_patch(
                Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    facecolor="none",
                    edgecolor=self.cobber_gold,
                    linewidth=3,
                )
            )

        if title:
            ax.set_title(title, fontsize=11)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect("equal")


    # ------------------------------------------------------------------
    # Tab 4 drawing methods
    # ------------------------------------------------------------------

    def draw_track_image_for_stack(self, ax, image, title):
        ax.clear()
        ax.imshow(image, cmap="gray_r", vmin=0, vmax=1)

        h, w = image.shape

        for y in range(h):
            for x in range(w):
                ax.add_patch(
                    Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        facecolor="none",
                        edgecolor="black",
                        linewidth=0.5,
                    )
                )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect("equal")

    def draw_heatmap_on_axes(self, ax, data, title, vmax=None):
        ax.clear()

        if vmax is None:
            vmax = max(1.0, float(np.max(data)))

        ax.imshow(data, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, data.shape[1] - 0.5)
        ax.set_ylim(data.shape[0] - 0.5, -0.5)
        ax.set_aspect("equal")

    def draw_blank_feature_stack(self):
        self.feature_stack_title.setText("<h2>Four Activated Feature Maps</h2>")
        self.feature_stack_placeholder.setVisible(True)
        self.feature_stack_visual_frame.setVisible(False)
        self.feature_stack_caption.setText(
            "Choose a track to see its four activated feature maps."
        )

    def draw_blank_feature_summary(self):
        self.combined_title.setText("<h2>Feature Signature Reveal</h2>")
        self.feature_summary_placeholder.setVisible(True)
        self.feature_summary_visual_frame.setVisible(False)
        self.feature_summary_caption.setText(
            "After you inspect the feature stack, reveal the combined response map and feature signature."
        )

    def draw_selected_feature_stack(self):
        if self.feature_stack_track_name is None:
            self.draw_blank_feature_stack()
            return

        track_name = self.feature_stack_track_name
        track_record = self.feature_cache[track_name]

        self.feature_stack_placeholder.setVisible(False)
        self.feature_stack_visual_frame.setVisible(True)

        self.feature_stack_title.setText(
            f"<h2>{track_name}: Four Activated Feature Maps</h2>"
        )

        self.feature_original_canvas.axes.clear()
        self.draw_track_image_for_stack(
            self.feature_original_canvas.axes,
            track_record["track_image"],
            f"Original {track_name}",
        )
        self.feature_original_canvas.fig.tight_layout()
        self.feature_original_canvas.draw()

        relu_maps = [
            track_record["filters"][filter_name]["relu"]
            for filter_name in FILTERS.keys()
        ]
        stack_vmax = max(1.0, max(float(np.max(map_data)) for map_data in relu_maps))

        axes = np.array(self.feature_stack_canvas.axes).reshape(2, 2)
        filter_names = list(FILTERS.keys())

        for ax, filter_name in zip(axes.flatten(), filter_names):
            relu_map = track_record["filters"][filter_name]["relu"]
            self.draw_heatmap_on_axes(ax, relu_map, filter_name, vmax=stack_vmax)

        self.feature_stack_canvas.fig.tight_layout()
        self.feature_stack_canvas.draw()

        self.feature_stack_caption.setText(
            "Each heatmap shows one filter response after ReLU. The maps do not "
            "need to look like the track to carry useful information."
        )

    def draw_selected_combined_map(self):
        if self.feature_stack_track_name is None:
            return

        track_name = self.feature_stack_track_name
        track_record = self.feature_cache[track_name]

        self.feature_summary_placeholder.setVisible(False)
        self.feature_summary_visual_frame.setVisible(True)

        self.combined_title.setText(
            f"<h2>{track_name}: Combined Max Response Map</h2>"
        )

        combined = track_record["combined_max"]
        combined_vmax = max(1.0, float(np.max(combined)))

        self.draw_heatmap_on_axes(
            self.combined_canvas.axes,
            combined,
            "Strongest response at each location",
            vmax=combined_vmax,
        )
        self.combined_canvas.fig.tight_layout()
        self.combined_canvas.draw()

    def draw_all_combined_maps(self):
        axes = np.array(self.all_combined_canvas.axes).flatten()

        combined_maps = [
            self.feature_cache[track_name]["combined_max"]
            for track_name in IMAGES.keys()
        ]
        common_vmax = max(1.0, max(float(np.max(map_data)) for map_data in combined_maps))

        for ax, track_name in zip(axes, IMAGES.keys()):
            combined = self.feature_cache[track_name]["combined_max"]
            self.draw_heatmap_on_axes(ax, combined, track_name, vmax=common_vmax)

        self.all_combined_canvas.fig.tight_layout()
        self.all_combined_canvas.draw()

    def update_signature_table(self):
        filter_names = list(FILTERS.keys())

        short_names = {
            "Vertical Edge": "Vertical",
            "Horizontal Edge": "Horizontal",
            "Junction / Sharp Change": "Junction",
            "Diagonal Edge": "Diagonal",
        }

        cell_style = "padding:4px 7px; text-align:center; font-size:11pt;"
        header_style = "padding:4px 7px; text-align:center; font-size:11pt; font-weight:bold;"
        track_style = "padding:4px 7px; text-align:center; font-size:11pt; font-weight:bold;"

        header_cells = "".join(
            f"<th style='{header_style}'>{short_names[name]}</th>"
            for name in filter_names
        )

        rows = []
        for track_name in IMAGES.keys():
            row_cells = []
            for filter_name in filter_names:
                value = self.feature_cache[track_name]["filters"][filter_name]["signature"]
                row_cells.append(
                    f"<td style='{cell_style}'>{value}</td>"
                )

            rows.append(
                "<tr>"
                f"<td style='{track_style}'>{track_name.replace(' Track', '')}</td>"
                + "".join(row_cells)
                + "</tr>"
            )

        table_html = (
            "<table style='margin-left:auto; margin-right:auto; border-collapse:collapse;'>"
            "<tr>"
            f"<th style='{header_style}'>Track</th>"
            f"{header_cells}"
            "</tr>"
            + "".join(rows)
            + "</table>"
        )

        self.signature_table_label.setText(table_html)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CobberEcoCNNApp()
    window.show()
    sys.exit(app.exec())
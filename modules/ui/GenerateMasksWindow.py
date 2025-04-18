

# Sub window for the CaptionUI tools.
# Allows for creation of image masks

import os
from PySide6.QtWidgets import (
    QDialog, QLabel, QLineEdit, QComboBox, QCheckBox, QProgressBar,
    QPushButton, QFileDialog, QGridLayout, QVBoxLayout
)
from PySide6.QtCore import Qt, Signal
# Use the same LongTaskButton as in the captions window.
from modules.util.ui.LongTaskButton import LongTaskButton

class GenerateMasksWindow(QDialog):
    """
    Window for generating masks for a folder of images.
    """
    # Signal for progress updates (current, max)
    progress_signal = Signal(int, int)

    def __init__(self, parent, path, parent_include_subdirectories, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.parent = parent  # Reference to the parent object/window.
        self.setWindowTitle("Batch generate masks")
        self.resize(360, 430)

        # Default path
        if path is None:
            path = ""

        # ---------------------------------------------------------------------
        # Variables / state
        # ---------------------------------------------------------------------
        self.modes = ["Replace all masks", "Create if absent", "Add to existing",
                      "Subtract from existing", "Blend with existing"]
        self.models = ["ClipSeg", "Rembg", "Rembg-Human", "Hex Color"]

        # ---------------------------------------------------------------------
        # Main layout
        # ---------------------------------------------------------------------
        layout = QVBoxLayout()
        self.setLayout(layout)
        grid = QGridLayout()
        layout.addLayout(grid)

        # Model label and combo box
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.models)
        self.model_combo.setCurrentIndex(self.models.index("ClipSeg"))
        grid.addWidget(model_label, 0, 0)
        grid.addWidget(self.model_combo, 0, 1)

        # Path label, line edit, and browse button
        path_label = QLabel("Folder:")
        self.path_edit = QLineEdit(path)
        path_button = QPushButton("...")
        path_button.clicked.connect(self.browse_for_path)

        grid.addWidget(path_label, 1, 0)
        grid.addWidget(self.path_edit, 1, 1)
        grid.addWidget(path_button, 1, 2)

        # Prompt label + entry
        prompt_label = QLabel("Prompt:")
        self.prompt_edit = QLineEdit()
        grid.addWidget(prompt_label, 2, 0)
        grid.addWidget(self.prompt_edit, 2, 1, 1, 2)

        # Mode label and combo box
        mode_label = QLabel("Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.modes)
        self.mode_combo.setCurrentIndex(self.modes.index("Create if absent"))
        grid.addWidget(mode_label, 3, 0)
        grid.addWidget(self.mode_combo, 3, 1, 1, 2)

        # Threshold label + entry
        threshold_label = QLabel("Threshold:")
        self.threshold_edit = QLineEdit()
        self.threshold_edit.setPlaceholderText("0.0 - 1.0")
        self.threshold_edit.setText("0.3")
        grid.addWidget(threshold_label, 4, 0)
        grid.addWidget(self.threshold_edit, 4, 1, 1, 2)

        # Smooth label + entry
        smooth_label = QLabel("Smooth:")
        self.smooth_edit = QLineEdit()
        self.smooth_edit.setPlaceholderText("5")
        self.smooth_edit.setText("5")
        grid.addWidget(smooth_label, 5, 0)
        grid.addWidget(self.smooth_edit, 5, 1, 1, 2)

        # Expand label + entry
        expand_label = QLabel("Expand:")
        self.expand_edit = QLineEdit()
        self.expand_edit.setPlaceholderText("10")
        self.expand_edit.setText("10")
        grid.addWidget(expand_label, 6, 0)
        grid.addWidget(self.expand_edit, 6, 1, 1, 2)

        # Alpha label + entry
        alpha_label = QLabel("Alpha:")
        self.alpha_edit = QLineEdit()
        self.alpha_edit.setPlaceholderText("1")
        self.alpha_edit.setText("1")
        grid.addWidget(alpha_label, 7, 0)
        grid.addWidget(self.alpha_edit, 7, 1, 1, 2)

        # Include subfolders
        subfolders_label = QLabel("Include subfolders:")
        self.include_sub_check = QCheckBox()
        self.include_sub_check.setChecked(bool(parent_include_subdirectories))
        grid.addWidget(subfolders_label, 8, 0)
        grid.addWidget(self.include_sub_check, 8, 1)

        # Progress label + bar
        self.progress_label = QLabel("Progress: 0/0")
        self.progress_bar = QProgressBar()
        grid.addWidget(self.progress_label, 9, 0)
        grid.addWidget(self.progress_bar, 9, 1, 1, 2)

        # Create masks button: use LongTaskButton for threading.
        self.create_button = LongTaskButton(
            "Create Masks",
            "Task Running - Click to Cancel",
            self.create_masks  # Callback accepts a stop_event argument.
        )
        grid.addWidget(self.create_button, 10, 0, 1, 3)

        layout.addStretch(1)

        self.progress_signal.connect(self.set_progress)

    def browse_for_path(self):
        """
        Open a directory dialog and update the path_edit field.
        """
        chosen_dir = QFileDialog.getExistingDirectory(self, "Select Directory", self.path_edit.text())
        if chosen_dir:
            self.path_edit.setText(chosen_dir)

    def set_progress(self, value, max_value):
        """
        Update progress bar and label.
        """
        percentage = int((value / max_value) * 100) if max_value else 0
        self.progress_bar.setValue(percentage)
        self.progress_label.setText(f"Progress: {value}/{max_value}")

    def create_masks(self, stop_event):
        """
        Callback for create_button.
        Gathers UI values, resets the stop_event, and runs the mask generation process.
        """
        # Ask parent to load the chosen model
        model_name = self.model_combo.currentText()
        self.parent.load_masking_model(model_name)

        # Map selected string to your internal strings
        mode_map = {
            "Replace all masks": "replace",
            "Create if absent": "fill",
            "Add to existing": "add",
            "Subtract from existing": "subtract",
            "Blend with existing": "blend",
        }
        selected_mode = mode_map.get(self.mode_combo.currentText(), "fill")

        # Convert text fields
        alpha = float(self.alpha_edit.text() or "1")
        threshold = float(self.threshold_edit.text() or "0.3")
        smooth_pixels = int(self.smooth_edit.text() or "5")
        expand_pixels = int(self.expand_edit.text() or "10")

        # Call the parent's masking model in the worker thread,
        # using a lambda func to emit progress updates.
        self.parent.masking_model.mask_folder(
            sample_dir=self.path_edit.text(),
            prompts=[self.prompt_edit.text()],
            mode=selected_mode,
            alpha=alpha,
            threshold=threshold,
            smooth_pixels=smooth_pixels,
            expand_pixels=expand_pixels,
            progress_callback=lambda i, m: self.progress_signal.emit(i, m),
            include_subdirectories=self.include_sub_check.isChecked(),
            stop_event=stop_event  # If the masking model supports stopping via an event.
        )

        self.post_worker()

    def post_worker(self):
        """
        Final updates after mask generation (for example, refreshing the parent's image).
        """
        if hasattr(self.parent, 'load_image'):
            self.parent.load_image()

    def closeEvent(self, event):
        """
        Ensure that any running task is signaled to stop when the window is closed.
        """
        if self.create_button:
            self.create_button.stop_task()
        super().closeEvent(event)

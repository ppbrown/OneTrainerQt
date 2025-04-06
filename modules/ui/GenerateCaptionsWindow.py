"""
Generates a pop-up window related to the CaptionsUI/data tools window,
specific to generating captions for a folder of images.
Note: we use Signal() passing in some places instead of direct calls,
for Thread-safety with the LongTaskButton worker thread.

"""



import os
import threading
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QLineEdit, QComboBox,
    QCheckBox, QProgressBar,
    QFileDialog, QGridLayout, QVBoxLayout, QPushButton
)
from PySide6.QtCore import Qt, Signal
import torch

# Updated import path for LongTaskButton.
from modules.util.ui.LongTaskButton import LongTaskButton

# These imports trigger BaseImageCaptionModel
from modules.module.WDModel import WDModel
from modules.module.BlipModel import BlipModel
from modules.module.Blip2Model import Blip2Model
from modules.module.Moondream2Model import Moondream2Model

from modules.module.BaseImageCaptionModel import BaseImageCaptionModel
from modules.util.torch_util import default_device

class GenerateCaptionsWindow(QMainWindow):
    """
    Window for generating captions for a folder of images.
    """
    # Signal for progress updates (current, max)
    progress_signal = Signal(int, int)
    
    def __init__(self, parent, path, parent_include_subdirectories, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.parent = parent  # Reference to the parent if needed.
        self.setWindowTitle("Batch generate captions")
        self.resize(360, 360)

        self.progress_signal.connect(self.set_progress)

        # Default path.
        if path is None:
            path = ""

        # ---------------------------------------------------------------------
        # Variables / state
        # ---------------------------------------------------------------------
        self.caption_model_list = BaseImageCaptionModel.get_all_model_choices()
        self.caption_modelname_list = list(self.caption_model_list.keys())

        self.modes = ["Replace all captions", "Create if absent", "Add as new line"]

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # We’ll use a QGridLayout for row/column alignment
        grid = QGridLayout()
        layout.addLayout(grid)

        # Model label and combo
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.caption_modelname_list)
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

        # Initial caption
        caption_label = QLabel("Initial Caption:")
        self.caption_entry = QLineEdit()
        grid.addWidget(caption_label, 2, 0)
        grid.addWidget(self.caption_entry, 2, 1, 1, 2)

        # Caption prefix
        prefix_label = QLabel("Caption Prefix:")
        self.prefix_entry = QLineEdit()
        grid.addWidget(prefix_label, 3, 0)
        grid.addWidget(self.prefix_entry, 3, 1, 1, 2)

        # Caption postfix
        postfix_label = QLabel("Caption Postfix:")
        self.postfix_entry = QLineEdit()
        grid.addWidget(postfix_label, 4, 0)
        grid.addWidget(self.postfix_entry, 4, 1, 1, 2)

        # Mode label and combo
        mode_label = QLabel("Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.modes)
        self.mode_combo.setCurrentIndex(self.modes.index("Create if absent"))  # default
        grid.addWidget(mode_label, 5, 0)
        grid.addWidget(self.mode_combo, 5, 1, 1, 2)

        # Include subfolders
        subfolders_label = QLabel("Include subfolders:")
        self.include_sub_check = QCheckBox()
        self.include_sub_check.setChecked(bool(parent_include_subdirectories))
        grid.addWidget(subfolders_label, 6, 0)
        grid.addWidget(self.include_sub_check, 6, 1)

        # Progress label and bar
        self.progress_label = QLabel("Progress: 0/0")
        self.progress_bar = QProgressBar()
        grid.addWidget(self.progress_label, 7, 0)
        grid.addWidget(self.progress_bar, 7, 1, 1, 2)

        # Create captions button: use LongTaskButton.
        # The LongTaskButton creates its own stop_event.
        self.create_button = LongTaskButton(
            "Create Captions",
            "Task Running - Click to Cancel",
            self.create_captions  # Callback accepts a stop_event argument.
        )
        grid.addWidget(self.create_button, 8, 0, 1, 3)

        # Stretch to fill
        layout.addStretch(1)

        # Must be done after create_button is initialized, since it uses the stop_event.
        self.set_caption_model(self.caption_modelname_list[0])

    def set_caption_model(self, modelname: str):
        if modelname not in self.caption_model_list:
            print(f"INTERNAL ERROR: {modelname} not in caption_model_list")
            return
        
        self.caption_modelname = modelname
        stop_event = self.create_button.stop_event
        self.caption_model = self.caption_model_list[modelname](
            default_device, torch.float16, modelname, stop_event
        )

    def browse_for_path(self):
        """
        Open a directory dialog, and update the path_edit line.
        """
        chosen_dir = QFileDialog.getExistingDirectory(self, "Select Directory", self.path_edit.text())
        if chosen_dir:
            self.path_edit.setText(chosen_dir)

    def set_progress(self, value, max_value):
        """
        Update the progress bar and progress label.
        """
        percentage = int((value / max_value) * 100) if max_value else 0
        self.progress_bar.setValue(percentage)
        self.progress_label.setText(f"Progress: {value}/{max_value}")

    def create_captions(self, stop_event):
        """
        Callback for create_button.
        Gathers current UI values, clears the stop_event, and runs caption_folder.
        Also updates the model's stop_event to use the one from the button.
        """
        modelname = self.model_combo.currentText()
        if modelname != self.caption_modelname:
            self.set_caption_model(modelname)


        mode_map = {
            "Replace all captions": "replace",
            "Create if absent": "fill",
            "Add as new line": "add",
        }
        self.selected_mode = mode_map.get(self.mode_combo.currentText(), "fill")


        self.caption_model.caption_folder(
            sample_dir=self.path_edit.text(),
            initial_caption=self.caption_entry.text(),
            caption_prefix=self.prefix_entry.text(),
            caption_postfix=self.postfix_entry.text(),
            mode=self.selected_mode,
            progress_callback=lambda i, m: self.progress_signal.emit(i, m),
            include_subdirectories=self.include_sub_check.isChecked()
        )

        self.post_worker()

    def closeEvent(self, event):
        """
        Ensure that any running task is signaled to stop if the window is closed.
        """
        if self.create_button:
            self.create_button.stop_task()
        super().closeEvent(event)

    def post_worker(self):
        """
        Final updates after caption generation (e.g., refreshing the parent's image display).
        """
        if hasattr(self.parent, 'load_image'):
            self.parent.load_image()

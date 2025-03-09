# generate_captions_window.py

""" Conversion warning:
  This file has been converted from a customtkinter-based UI to a PySide6-based QDialog.
  Adjust layout or widget sizing as you prefer. In Qt, using layout-based sizing is usually recommended.
Replace references to self.parent with signals/slots or direct method calls as needed 
(since in Qt, “parent” typically refers to the QWidget parent, which might not contain 
 all the same methods as your old parent window).
"""

import os
from PySide6.QtWidgets import (
    QMainWindow, QLabel, QLineEdit, QComboBox,
    QPushButton, QCheckBox, QProgressBar,
    QFileDialog, QGridLayout, QVBoxLayout
)
from PySide6.QtCore import Qt

class GenerateCaptionsWindow(QMainWindow):
    """
    Window for generating captions for a folder of images.
    Replaces the customtkinter-based class with a QDialog using PySide6.
    """

    def __init__(self, parent, path, parent_include_subdirectories, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.parent = parent  # reference to the parent, if you need to call parent methods
        self.setWindowTitle("Batch generate captions")
        self.resize(360, 360)  # or set a fixed size if you prefer: self.setFixedSize(360, 360)

        # Default path
        if path is None:
            path = ""

        # ---------------------------------------------------------------------
        # Variables / state
        # ---------------------------------------------------------------------
        self.models = ["Blip", "Blip2", "WD14 VIT v2"]
        self.modes = ["Replace all captions", "Create if absent", "Add as new line"]

        # ---------------------------------------------------------------------
        # Main layout
        # ---------------------------------------------------------------------
        layout = QVBoxLayout()
        self.setLayout(layout)

        # We’ll use a QGridLayout for row/column alignment
        grid = QGridLayout()
        layout.addLayout(grid)

        # Model label and combo
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.models)
        self.model_combo.setCurrentIndex(self.models.index("Blip"))  # default to "Blip"
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

        # Create captions button
        create_button = QPushButton("Create Captions")
        create_button.clicked.connect(self.create_captions)
        grid.addWidget(create_button, 8, 0, 1, 3)

        # stretch to fill
        layout.addStretch(1)

        # Modal-like behavior (if you want)
        # self.setModal(True)  # If you want a truly modal dialog

    def browse_for_path(self):
        """
        Open a directory dialog, set the result to the path_edit line.
        """
        chosen_dir = QFileDialog.getExistingDirectory(self, "Select Directory", self.path_edit.text())
        if chosen_dir:
            self.path_edit.setText(chosen_dir)

    def set_progress(self, value, max_value):
        """
        Update the progress bar and progress label.
        """
        if max_value == 0:
            percentage = 0
        else:
            percentage = int((value / max_value) * 100)

        self.progress_bar.setValue(percentage)
        self.progress_label.setText(f"Progress: {value}/{max_value}")
        # If you want to ensure an immediate GUI refresh:
        # self.progress_bar.repaint()
        # self.progress_label.repaint()
        # QApplication.processEvents()

    def create_captions(self):
        """
        Called when "Create Captions" is clicked.  
        Loads the model, calls the parent's captioning_model, etc.
        """
        if not self.parent:
            return  # or raise an exception

        # Ask parent to load the chosen model
        model_name = self.model_combo.currentText()
        self.parent.load_captioning_model(model_name)

        # Convert selected mode to your internal strings
        mode_map = {
            "Replace all captions": "replace",
            "Create if absent": "fill",
            "Add as new line": "add",
        }
        selected_mode = mode_map.get(self.mode_combo.currentText(), "fill")

        # Call parent's model to caption folder
        self.parent.captioning_model.caption_folder(
            sample_dir=self.path_edit.text(),
            initial_caption=self.caption_entry.text(),
            caption_prefix=self.prefix_entry.text(),
            caption_postfix=self.postfix_entry.text(),
            mode=selected_mode,
            progress_callback=self.set_progress,
            include_subdirectories=self.include_sub_check.isChecked(),
        )

        # Reload the parent’s image display or do any final updates
        self.parent.load_image()

        # Optionally close the dialog automatically
        # self.accept()  # or self.close()

# concept_tab.py

import os
import pathlib

from PySide6.QtWidgets import (
    QDialog, QFrame, QLabel, QPushButton, QCheckBox, QVBoxLayout, QHBoxLayout,
    QGridLayout, QWidget
)
from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QPixmap, QImage

from PIL import Image

from modules.ui.ConceptWindow import ConceptWindow  # presumably also converted to PySide
from modules.ui.ConfigList import ConfigList  # your PySide version of "ConfigList"
from modules.util import path_util
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui.UIState import UIState


# pyside6 conversion warning:
# Current typing is a hackjob, because of the original code's design.
# The init routine creates and manipulates things outside itself, violating OOP.
# This class needs to be redesigned to be 
# a proper QWidget with a layout and proper signal/slot connections.
class ConceptsTab(ConfigList):

    def __init__(self, parent: QWidget, train_config: TrainConfig, ui_state: UIState):
        super().__init__(
            master=parent,
            train_config=train_config,
            ui_state=ui_state,
            from_external_file=True,
            attr_name="concept_file_name",
            config_dir="training_concepts",
            default_config_name="concepts.json",
            add_button_text="add concept",
            is_full_width=False,
        )

    def create_widget(self, parent_widget: QWidget, element, i, open_command, remove_command, clone_command, save_command):
        """
        Return a widget instance that represents one concept in the list.
        """
        return ConceptWidget(parent_widget, element, i, open_command, remove_command, clone_command, save_command)

    def create_new_element(self) -> dict:
        """
        Return a new concept config dict (the default).
        """
        return ConceptConfig.default_values()

    def open_element_window(self, i, ui_state) -> QDialog:
        """
        The original code returned ctk.CTkToplevel. We'll return a QDialog instead.
        We'll assume ConceptWindow is also converted to PySide6.
        """
        # ui_state is a tuple: (self.ui_state, self.image_ui_state, self.text_ui_state)
        return ConceptWindow(self, self.current_config[i], ui_state[0], ui_state[1], ui_state[2])


class ConceptWidget(QFrame):
    """
    Displays:
      - A 150x150 preview image
      - A name label
      - A close button
      - A clone button
      - An "enabled" switch
    and calls commands passed from the parent.
    """
    def __init__(self, parent: QWidget, concept, i, open_command, remove_command, clone_command, save_command):
        super().__init__()
        self.concept = concept
        self.ui_state = UIState(self, concept)
        self.image_ui_state = UIState(self, concept.image)
        self.text_ui_state = UIState(self, concept.text)
        self.i = i

        # We'll use a QGridLayout to replicate the 2-row approach: 
        #   row0 = image, row1 = name label 
        # plus absolute positioning for close/clone/switch if we want to replicate .place(...).
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedSize(150, 170)
        self.grid_layout = QGridLayout(self)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setHorizontalSpacing(0)
        self.grid_layout.setVerticalSpacing(0)
        self.setLayout(self.grid_layout)

        # row=0 col=0 => image
        # row=1 col=0 => name
        # We'll also do absolute positioning for the buttons/switch on top
        # We'll create a sub-container for absolute.
        container = QWidget(self)
        container.setGeometry(0, 0, 150, 170)

        # We'll replicate the image label
        self.image_label = QLabel(self)
        self.image_label.setGeometry(0, 0, 150, 150)  # matches the original code
        # set the pixmap
        self._pixmap = self._get_preview_pixmap()
        self.image_label.setPixmap(self._pixmap)
        self.image_label.setScaledContents(True)

        # We'll connect a mousePressEvent or use eventFilter. 
        # For the original "bind <Button-1>", let's do a mousePressEvent override.

        # Name label in row=1
        self.name_label = QLabel(self)
        self.name_label.setGeometry(5, 150, 140, 20)
        self.name_label.setText(self._get_display_name())
        self.name_label.setWordWrap(True)

        # Close button (x=0, y=0)
        self.close_button = QPushButton("X", self)
        self.close_button.setStyleSheet("background-color: #C00000; color: white; border-radius:2px;")
        self.close_button.setGeometry(0, 0, 20, 20)
        self.close_button.clicked.connect(lambda: remove_command(self.i))

        # Clone button (x=25, y=0)
        self.clone_button = QPushButton("+", self)
        self.clone_button.setStyleSheet("background-color: #00C000; color: white; border-radius:2px;")
        self.clone_button.setGeometry(25, 0, 20, 20)
        self.clone_button.clicked.connect(lambda: clone_command(self.i, self._randomize_seed))

        # "enabled" switch => QCheckBox 
        self.enabled_switch = QCheckBox(self)
        self.enabled_switch.setText("")  # no text
        self.enabled_switch.setGeometry(110, 0, 40, 20)
        # If your concept "enabled" is a bool, we can do:
        self.enabled_switch.setChecked(bool(self.concept.enabled))
        self.enabled_switch.stateChanged.connect(lambda _: save_command())

    def mousePressEvent(self, event):
        """
        If the user clicks on the image area, call the open_command.
        We'll check if the click is within the image's rectangle.
        """
        print("DEBUG: ConceptWidget mousePressEvent")

        # If within the 150x150 area, we interpret it as a click on the image
        # Original code used <Button-1> on the label. We'll replicate that logic:
        pos = event.pos()
        if 0 <= pos.x() < 150 and 0 <= pos.y() < 150:
            # open the concept
            # we do not have direct references to open_command, but we can store it as a property
            # or we can connect a signal. For simplicity, let's store it:
            # The original code: open_command(self.i, (self.ui_state, self.image_ui_state, self.text_ui_state))
            if event.button() == Qt.LeftButton and hasattr(self, 'open_command'):
                self.open_command(self.i, (self.ui_state, self.image_ui_state, self.text_ui_state))


    def set_open_command(self, func):
        """
        So we can store open_command if needed.
        """
        self.open_command = func

    def _randomize_seed(self, concept: ConceptConfig):
        concept.seed = ConceptConfig.default_values().seed
        return concept

    def _get_display_name(self):
        if self.concept.name:
            return self.concept.name
        elif self.concept.path:
            return os.path.basename(self.concept.path)
        else:
            return ""

    def configure_element(self):
        """
        Refresh the display name and preview image after an update.
        Called by the parent if something changed in the concept.
        """
        self.name_label.setText(self._get_display_name())
        self._pixmap = self._get_preview_pixmap()
        self.image_label.setPixmap(self._pixmap)

    def _get_preview_pixmap(self) -> QPixmap:
        """
        Load the first image in the directory, or fallback to 'resources/icons/icon.png',
        then convert to a 150x150 QPixmap.
        """
        preview_path = "resources/icons/icon.png"
        glob_pattern = "**/*.*" if self.concept.include_subdirectories else "*.*"

        if os.path.isdir(self.concept.path):
            for path in pathlib.Path(self.concept.path).glob(glob_pattern):
                extension = os.path.splitext(path)[1]
                if (path.is_file() and 
                        path_util.is_supported_image_extension(extension) and 
                        not path.name.endswith("-masklabel.png")):
                    preview_path = path_util.canonical_join(self.concept.path, path)
                    break

        # open with PIL
        pil_img = Image.open(preview_path)
        size = min(pil_img.width, pil_img.height)
        left = (pil_img.width - size) // 2
        top = (pil_img.height - size) // 2
        pil_img = pil_img.crop((left, top, left + size, top + size))
        pil_img = pil_img.resize((150, 150), Image.Resampling.LANCZOS)

        # convert to QPixmap
        qimg = QImage(
            pil_img.tobytes("raw", "RGB"),
            pil_img.width,
            pil_img.height,
            3 * pil_img.width,
            QImage.Format_RGB888
        )
        return QPixmap.fromImage(qimg)

    def place_in_list(self):
        """
        The original code had: x = self.i % 6, y = self.i // 6
        self.grid(row=y, column=x, ...)
        In Qt, you'd typically have the parent do the layout in a grid.
        For example, if the parent is QGridLayout, it might call:
           parent_layout.addWidget(self, y, x)
        We'll keep this method as a stub or call it from the parent.
        """
        x = self.i % 6
        y = self.i // 6
        # If we rely on the parent's layout:
        if self.parentWidget() and isinstance(self.parentWidget().layout(), QGridLayout):
            self.parentWidget().layout().addWidget(self, y, x)

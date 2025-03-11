# sampling_tab.py

from PySide6.QtWidgets import (
    QWidget, QFrame, QLabel, QPushButton, QLineEdit, QCheckBox, QGridLayout
)
from PySide6.QtCore import Qt

from modules.ui.ConfigList import ConfigList  # your PySide6-based class
from modules.ui.SampleParamsWindow import SampleParamsWindow
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui.UIState import UIState


# This inherits from ConfigList, which is a special abstract class that should be redesigned.
# It breaks good object oriented design, by being forced to init things OUTSIDE of itself, via
# its init method. This is a bad design pattern, and should be refactored.
class SamplingTab(ConfigList):
    def __init__(self, parent, train_config, ui_state):
        
        super().__init__(
            master=parent,
            train_config=train_config,
            ui_state=ui_state,
            from_external_file=True,
            attr_name="sample_definition_file_name",
            config_dir="training_samples",
            default_config_name="samples.json",
            add_button_text="add sample",
            is_full_width=True,
        )
        

    def create_widget(self, parent_widget, element, i, open_command, remove_command, clone_command, save_command):

        return SampleWidget(element, i, open_command, remove_command, clone_command, save_command)

    def create_new_element(self) -> dict:

        return SampleConfig.default_values()

    def open_element_window(self, i, ui_state):

        # ui_state is presumably a single UIState or some structure
        return SampleParamsWindow(self, self.current_config[i], ui_state)


class SampleWidget(QFrame):
    """
    Lays out controls for a single sample in a row: 
      [X] [ + ] [switch] [width] [height] [seed] [prompt] [...]
    """
    def __init__(self, element, i, open_command, remove_command, clone_command, save_command):
        super().__init__()

        self.element = element
        self.ui_state = UIState(self, element)
        self.i = i
        self.save_command = save_command
        self.open_command = open_command

        # QFrame config
        self.setFrameShape(QFrame.StyledPanel)
        # We can replicate the background, corners, etc. if you like:
        # self.setStyleSheet("background-color: transparent; border-radius: 10px;")

        # We'll use a QGridLayout to replicate .grid(...) with columns.
        self.layout_grid = QGridLayout(self)
        self.layout_grid.setContentsMargins(5, 5, 5, 5)
        self.layout_grid.setSpacing(5)

        # 0) remove ("X") button
        self.close_button = QPushButton("X", self)
        self.close_button.setStyleSheet("background-color: #C00000; color: white; border-radius:2px;")
        self.close_button.setFixedSize(20, 20)
        self.close_button.clicked.connect(lambda: remove_command(self.i))
        self.layout_grid.addWidget(self.close_button, 0, 0)

        # 1) clone ("+") button
        self.clone_button = QPushButton("+", self)
        self.clone_button.setStyleSheet("background-color: #00C000; color: white; border-radius:2px;")
        self.clone_button.setFixedSize(20, 20)
        self.clone_button.clicked.connect(lambda: clone_command(self.i))
        self.layout_grid.addWidget(self.clone_button, 0, 1)

        # 2) enabled switch => QCheckBox
        self.enabled_switch = QCheckBox(self)
        self.enabled_switch.setText("")
        self.enabled_switch.setFixedWidth(40)
        # If your sample config has 'enabled' as a bool, you can do:
        self.enabled_switch.setChecked(bool(self.element.enabled))
        self.enabled_switch.stateChanged.connect(self.__switch_enabled)
        self.layout_grid.addWidget(self.enabled_switch, 0, 2)

        # 3) width
        lbl_width = QLabel("width:")
        self.layout_grid.addWidget(lbl_width, 0, 3)
        self.width_entry = QLineEdit()
        self.width_entry.setFixedWidth(50)
        self.width_entry.setText(str(self.element.width))  # or from UIState
        self.width_entry.editingFinished.connect(lambda: save_command())
        self.layout_grid.addWidget(self.width_entry, 0, 4)

        # 4) height
        lbl_height = QLabel("height:")
        self.layout_grid.addWidget(lbl_height, 0, 5)
        self.height_entry = QLineEdit()
        self.height_entry.setFixedWidth(50)
        self.height_entry.setText(str(self.element.height))
        self.height_entry.editingFinished.connect(lambda: save_command())
        self.layout_grid.addWidget(self.height_entry, 0, 6)

        # 5) seed
        lbl_seed = QLabel("seed:")
        self.layout_grid.addWidget(lbl_seed, 0, 7)
        self.seed_entry = QLineEdit()
        self.seed_entry.setFixedWidth(80)
        self.seed_entry.setText(str(self.element.seed))
        self.seed_entry.editingFinished.connect(lambda: save_command())
        self.layout_grid.addWidget(self.seed_entry, 0, 8)

        # 6) prompt
        lbl_prompt = QLabel("prompt:")
        self.layout_grid.addWidget(lbl_prompt, 0, 9)
        self.prompt_entry = QLineEdit()
        self.prompt_entry.setText(self.element.prompt if self.element.prompt else "")
        self.prompt_entry.editingFinished.connect(lambda: save_command())
        self.layout_grid.addWidget(self.prompt_entry, 0, 10)

        # 7) "..." advanced settings button
        self.adv_button = QPushButton("...")
        self.adv_button.setFixedWidth(40)
        self.adv_button.clicked.connect(lambda: open_command(self.i, self.ui_state))
        self.layout_grid.addWidget(self.adv_button, 0, 11)

        self.__set_enabled()

    def __switch_enabled(self, state):
        """
        Called when the user toggles the QCheckBox.
        state is NOT a bool, but 0,1,2 for unchecked, partial, checked.
        """
        self.element.enabled = True if state == 2 else False
        self.save_command()
        self.__set_enabled()

    def __set_enabled(self):
        """
        Enable/disable other fields based on self.element.enabled
        """
        enabled = bool(self.element.enabled)
        self.width_entry.setEnabled(enabled)
        self.height_entry.setEnabled(enabled)
        self.prompt_entry.setEnabled(enabled)
        self.seed_entry.setEnabled(enabled)
        self.adv_button.setEnabled(enabled)

    def configure_element(self):
        """
        Called if something changes externally and we want to refresh the UI.
        """
        # Refresh the fields if needed
        self.width_entry.setText(str(self.element.width))
        self.height_entry.setText(str(self.element.height))
        self.prompt_entry.setText(self.element.prompt or "")
        self.seed_entry.setText(str(self.element.seed))
        self.enabled_switch.setChecked(bool(self.element.enabled))
        self.__set_enabled()

    def place_in_list(self):
        """
        The original code used .grid(row=self.i, column=0, ...)
        In Qt, typically the parent would do layout.addWidget(self, row, col).
        We'll keep the method as a stub or call from the parent.
        """
        if self.parentWidget() and hasattr(self.parentWidget(), 'layout'):
            parent_layout = self.parentWidget().layout()
            if isinstance(parent_layout, QGridLayout):
                parent_layout.addWidget(self, self.i, 0, 1, 1)

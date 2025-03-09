# scheduler_params_window.py

from PySide6.QtWidgets import (
    QDialog, QGridLayout, QFrame, QPushButton
)
from PySide6.QtCore import Qt

from modules.ui.ConfigList import ConfigList  # PySide6 version
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class KvParams(ConfigList):
    """
    PySide6 translation of your ctk-based KvParams.
    Inherits from ConfigList, referencing train_config.scheduler_params.
    """
    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super().__init__(
            master,
            train_config,
            ui_state,
            attr_name="scheduler_params",
            from_external_file=False,
            add_button_text="add parameter",
            is_full_width=True
        )

    def refresh_ui(self):
        self._create_element_list()

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return KvWidget(master, element, i, open_command, remove_command, clone_command, save_command)

    def create_new_element(self) -> dict[str, str]:
        return {"key": "", "value": ""}

    def open_element_window(self, i, ui_state):
        pass


class KvWidget(QFrame):
    """
    PySide6 version of your ctk.CTkFrame-based KvWidget.
    """
    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command):
        super().__init__(master)
        self.element = element
        self.ui_state = UIState(self, element)
        self.i = i
        self.save_command = save_command

        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(5)
        self.setLayout(self.layout)

        # "X" close button
        close_button = components.button(self, 0, 0, "X",
                                         command=lambda: remove_command(self.i),
                                         tooltip="Remove this parameter")
        # Key
        tooltip_key = "Key name for an argument in your scheduler"
        self.key_entry = components.entry(
            self, 0, 1,
            self.ui_state, "key",
            tooltip=tooltip_key,
            wide_tooltip=True
        )
        self.key_entry.bind("<FocusOut>", lambda _: self.save_command())
        self.key_entry.configure(width=50)

        # Value
        tooltip_val = (
            "Value for an argument in your scheduler. Some special values can be used, wrapped in percent signs: "
            "LR, EPOCHS, STEPS_PER_EPOCH, TOTAL_STEPS, SCHEDULER_STEPS."
        )
        self.value_entry = components.entry(
            self, 0, 2,
            self.ui_state, "value",
            tooltip=tooltip_val,
            wide_tooltip=True
        )
        self.value_entry.bind("<FocusOut>", lambda _: self.save_command())
        self.value_entry.configure(width=50)

    def place_in_list(self):
        self.layout.setRowMinimumHeight(self.i, 30)
        self.layout.setRowStretch(self.i, 0)
        self.layout.setColumnStretch(2, 1)
        self.grid_layout_parent = self.parentWidget().layout()
        if self.grid_layout_parent:
            self.grid_layout_parent.addWidget(self, self.i, 0, 1, 1)


class SchedulerParamsWindow(QDialog):
    """
    PySide6 translation of your ctk.CTkToplevel-based SchedulerParamsWindow.
    """
    def __init__(self, parent, train_config: TrainConfig, ui_state, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.parent = parent
        self.train_config = train_config
        self.ui_state = ui_state

        self.setWindowTitle("Learning Rate Scheduler Settings")
        self.resize(800, 400)
        self.setModal(False)

        main_layout = QGridLayout(self)
        main_layout.setContentsMargins(5,5,5,5)
        main_layout.setSpacing(5)
        self.setLayout(main_layout)

        self.frame = QFrame(self)
        self.frame_layout = QGridLayout(self.frame)
        self.frame_layout.setContentsMargins(5,5,5,5)
        self.frame_layout.setSpacing(5)
        self.frame.setLayout(self.frame_layout)
        main_layout.addWidget(self.frame, 0, 0, 1, 1)

        # "ok" button
        self.ok_button = components.button(self, 1, 0, "ok", self.on_window_close)
        main_layout.addWidget(self.ok_button, 1, 0, alignment=Qt.AlignRight)

        # columns
        self.frame_layout.setColumnStretch(0, 0)
        self.frame_layout.setColumnStretch(1, 1)

        self.expand_frame = QFrame(self.frame)
        self.expand_frame_layout = QGridLayout(self.expand_frame)
        self.expand_frame_layout.setContentsMargins(0,0,0,0)
        self.expand_frame_layout.setSpacing(5)
        self.expand_frame.setLayout(self.expand_frame_layout)
        self.frame_layout.addWidget(self.expand_frame, 1, 0, 1, 2)

        self.main_frame(self.frame)

    def main_frame(self, master: QFrame):
        if self.train_config.learning_rate_scheduler == LearningRateScheduler.CUSTOM:
            components.label(master, 0, 0, "Class Name",
                             tooltip="Python class module and name for the custom scheduler class.")
            components.entry(master, 0, 1, self.ui_state, "custom_learning_rate_scheduler")

        # key-value params
        self.params = KvParams(self.expand_frame, self.train_config, self.ui_state)

    def on_window_close(self):
        self.close()

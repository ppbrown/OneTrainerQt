
import os
import json
import copy
import contextlib
from abc import ABC, abstractmethod

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
    QScrollArea, QPushButton, QComboBox, QLayout, QInputDialog
)
from PySide6.QtCore import Qt

from modules.util import path_util
from modules.util.config.BaseConfig import BaseConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui.UIState import UIState


class ConfigList(ABC):
    """
    ABbstract class that manages a list of config elements (of type BaseConfig)
    Child classes must implement create_widget() and create_new_element().
    In additiona to handling a list of elements, this class may also handle loading and saving them
    to an external file. In which case, it organized named "configs" of elements.
    The configs can be saved to external .json files.
    """

    def __init__(
        self,
        master: QWidget,
        train_config: TrainConfig,
        ui_state: UIState,
        from_external_file: bool,
        attr_name: str = "",
        config_dir: str = "",
        default_config_name: str = "",
        add_button_text: str = "",
        is_full_width: bool = False
    ):
        """
        :param master: parent widget (a QWidget or QFrame).
        :param train_config: the main TrainConfig object.
        :param ui_state: a UIState or similar data-binding object.
        :param from_external_file: if True, we read/write a separate .json file for the elements.
        :param attr_name: the attribute name in train_config for storing these elements (e.g. "additional_embeddings").
        :param config_dir: directory for storing external .json config files, if from_external_file == True.
        :param default_config_name: name for a default config file if none exist.
        :param add_button_text: text on the “add element” button (e.g. "add embedding").
        :param is_full_width: if True, the scroll area uses full-width layout for each item.
        """
        super().__init__()

        self.master = master
        self.train_config = train_config
        self.ui_state = ui_state
        self.from_external_file = from_external_file
        self.attr_name = attr_name
        self.config_dir = config_dir
        self.default_config_name = default_config_name
        self.is_full_width = is_full_width

        if not master.layout():
            self.master_layout = QVBoxLayout(master)
            self.master_layout.setContentsMargins(0, 0, 0, 0)
            master.setLayout(self.master_layout)
            self.master_layout.setSizeConstraint(QLayout.SetMinimumSize)
        else:
            self.master_layout = master.layout()

        # A top "frame" (QFrame) for controls
        self.top_frame = QFrame(master)
        self.top_frame_layout = QHBoxLayout(self.top_frame)
        self.top_frame_layout.setContentsMargins(0, 0, 0, 0)
        self.top_frame_layout.setSpacing(5)
        self.master_layout.addWidget(self.top_frame)

        # Area for added items.
        # The "element_list" area is typically scrollable
        self.scroll_area = QScrollArea(master)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setSizeConstraint(QLayout.SetMinimumSize)
        self.scroll_area.setWidget(self.scroll_content)
        
        self.master_layout.addWidget(self.scroll_area)

        # This is where we store the actual config list elements
        if from_external_file:
            # We hold a dropdown for selecting config files, plus an “add config” button, etc.
            self.configs_dropdown = None
            self.configs = []
            self.__load_available_config_names()

            # The train_config has an attribute for the current config file path, e.g. "train_config.my_attr".
            self.current_config_path = getattr(self.train_config, self.attr_name)
            self.current_config = []

            self.__load_current_config(self.current_config_path)

            self.__create_configs_dropdown()

            # Add a “add config” button
            self.add_config_button = QPushButton("add config")
            self.add_config_button.clicked.connect(self.__add_config)
            self.top_frame_layout.addWidget(self.add_config_button)

            # Add an “add element” button
            self.add_element_button = QPushButton(add_button_text)
            self.add_element_button.clicked.connect(self.__add_element)
            self.top_frame_layout.addWidget(self.add_element_button)

        else:
            # No external file. We directly have a list in train_config
            self.current_config = getattr(self.train_config, self.attr_name, [])
            # add an “add element” button
            self.add_element_button = QPushButton(add_button_text)
            self.add_element_button.clicked.connect(self.__add_element)
            self.top_frame_layout.addWidget(self.add_element_button)

            # Build the UI
            self._create_element_list()

        self.top_frame_layout.addStretch()

    # -----------------------------------------------------------------------
    # Abstract methods that child classes must implement
    # -----------------------------------------------------------------------
    @abstractmethod
    def create_widget(self, parent, element, i, open_command, remove_command, clone_command, save_command):
        """
        Return a QWidget for displaying a single element in the list.
        """

    @abstractmethod
    def create_new_element(self):
        """
        Return a new element object (e.g., a config BaseConfig).
        """

    @abstractmethod
    def open_element_window(self, i, ui_state):
        """
        Open a window/dialog for editing the i-th element. Return the dialog.
        """

    # -----------------------------------------------------------------------
    # UI creation for external-file mode
    # -----------------------------------------------------------------------
    def __create_configs_dropdown(self):
        """
        Creates or rebuilds a QComboBox for listing config files in self.configs.
        """
        if self.configs_dropdown is not None:
            self.top_frame_layout.removeWidget(self.configs_dropdown)
            self.configs_dropdown.deleteLater()
            self.configs_dropdown = None

        self.configs_dropdown = QComboBox()
        for (display_name, fullpath) in self.configs:
            self.configs_dropdown.addItem(display_name, fullpath)

        # If we have a known path (self.current_config_path), set the combo index
        for idx in range(self.configs_dropdown.count()):
            data = self.configs_dropdown.itemData(idx)
            if data == self.current_config_path:
                self.configs_dropdown.setCurrentIndex(idx)
                break

        self.configs_dropdown.currentIndexChanged.connect(self.__on_config_selected)
        self.top_frame_layout.addWidget(self.configs_dropdown)

    def __on_config_selected(self, index):
        """
        Called when user picks a different config file from the dropdown.
        We load that config.
        """
        path = self.configs_dropdown.itemData(index)
        setattr(self.train_config, self.attr_name, path)  # e.g. train_config.my_attr = path
        self.current_config_path = path
        self.__load_current_config(path)

    # -----------------------------------------------------------------------
    # Element list creation
    # -----------------------------------------------------------------------
    def _create_element_list(self):
        """
        Clears and rebuilds the "element list" scroll area from self.current_config.
        """
        # Clear existing
        for i in reversed(range(self.scroll_layout.count())):
            item = self.scroll_layout.takeAt(i)
            if item.widget():
                item.widget().deleteLater()

        # Add a widget for each element
        for i, element in enumerate(self.current_config):
            w = self.create_widget(
                parent=self.scroll_content,
                element=element,
                i=i,
                open_command=self.__open_element_window,
                remove_command=self.__remove_element,
                clone_command=self.__clone_element,
                save_command=self.__save_current_config
            )
            self.scroll_layout.addWidget(w)

    # -----------------------------------------------------------------------
    # External-file config loading
    # -----------------------------------------------------------------------
    def __load_available_config_names(self):
        # Build self.configs from .json files in self.config_dir
        if os.path.isdir(self.config_dir):
            for filename in os.listdir(self.config_dir):
                fullpath = os.path.join(self.config_dir, filename)
                if filename.endswith(".json") and os.path.isfile(fullpath):
                    name, _ = os.path.splitext(filename)
                    self.configs.append((name, fullpath))

        if len(self.configs) == 0:
            # Create a default config if none exist
            name = self.default_config_name.removesuffix(".json")
            path = self.__create_config(name)
            # Also set train_config's attribute to this path
            setattr(self.train_config, self.attr_name, path)
            self.__save_current_config()

    def __create_config(self, name: str):
        """
        Create a new (display_name, path) entry in self.configs.
        """
        safe_name = name.replace(" ", "_")  # or however you want to sanitize
        path = os.path.join(self.config_dir, f"{safe_name}.json")
        self.configs.append((safe_name, path))
        return path

    # -----------------------------------------------------------------------
    # Buttons
    # -----------------------------------------------------------------------
    def __add_config(self):
        """
        In your code, you used dialogs.StringInputDialog(...).
        We'll replicate with QInputDialog.getText(...) as a quick solution.
        """
        text, ok = QInputDialog.getText(self.master, "Name", "Enter new config name:")
        if ok and text.strip():
            path = self.__create_config(text.strip())
            self.__create_configs_dropdown()
            # Switch to that new config
            if self.configs_dropdown:
                for idx in range(self.configs_dropdown.count()):
                    data = self.configs_dropdown.itemData(idx)
                    if data == path:
                        self.configs_dropdown.setCurrentIndex(idx)
                        break

    def __add_element(self):
        i = len(self.current_config)
        new_element = self.create_new_element()
        self.current_config.append(new_element)

        w = self.create_widget(
            parent=self.scroll_content,
            element=new_element,
            i=i,
            open_command=self.__open_element_window,
            remove_command=self.__remove_element,
            clone_command=self.__clone_element,
            save_command=self.__save_current_config
        )
        self.scroll_layout.addWidget(w)

        self.__save_current_config()

    def __clone_element(self, clone_i, modify_element_fun=None):
        i = len(self.current_config)
        # deep copy
        new_element = copy.deepcopy(self.current_config[clone_i])

        if modify_element_fun is not None:
            new_element = modify_element_fun(new_element)

        self.current_config.append(new_element)

        w = self.create_widget(
            parent=self.scroll_content,
            element=new_element,
            i=i,
            open_command=self.__open_element_window,
            remove_command=self.__remove_element,
            clone_command=self.__clone_element,
            save_command=self.__save_current_config
        )
        self.scroll_layout.addWidget(w)

        self.__save_current_config()

    def __remove_element(self, remove_i):
        if 0 <= remove_i < len(self.current_config):
            self.current_config.pop(remove_i)
            # Rebuild the list from scratch to reindex
            self._create_element_list()
            self.__save_current_config()

    # -----------------------------------------------------------------------
    # Loading / Saving
    # -----------------------------------------------------------------------
    def __load_current_config(self, filename):
        """
        Load from external file into self.current_config, then rebuild the UI.
        """
        self.current_config.clear()
        if not filename or not os.path.isfile(filename):
            self._create_element_list()
            return

        try:
            with open(filename, "r", encoding="utf-8") as f:
                loaded_config_json = json.load(f)
                for element_json in loaded_config_json:
                    # Child classes define create_new_element().from_dict(...).
                    # We'll call it here if we want to parse each element.
                    e = self.create_new_element()
                    e = e.from_dict(element_json)  # e.g. if your BaseConfig supports from_dict
                    self.current_config.append(e)
        except Exception as e:
            print(f"Error loading config from {filename}: {e}")
            self.current_config = []

        self._create_element_list()

    def __save_current_config(self):
        if self.from_external_file:
            path = getattr(self.train_config, self.attr_name, None)
            if not path:
                return

            # Ensure directory
            dir_ = os.path.dirname(path)
            if not os.path.exists(dir_):
                with contextlib.suppress(Exception):
                    os.makedirs(dir_, exist_ok=True)

            try:
                with open(path, "w", encoding="utf-8") as f:
                    # Each element => .to_dict()
                    data = [elem.to_dict() for elem in self.current_config]
                    json.dump(data, f, indent=4)
            except Exception as e:
                print(f"Error saving config to {path}: {e}")

    # -----------------------------------------------------------------------
    # Opening element windows
    # -----------------------------------------------------------------------
    def __open_element_window(self, i, ui_state):
        """
        The original code calls self.open_element_window(...) then does:
          self.master.wait_window(window)
        In Qt, we typically do window.exec().
        Then we call the widget's configure_element() and save.
        """
        w = self.open_element_window(i, ui_state)
        if w:
            w.exec()  # or w.show() if you want modeless
        # Then reconfigure
        # If we keep references to the widget in a list, we can call configure_element() if needed
        self.__save_current_config()

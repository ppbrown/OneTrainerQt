# additional_embeddings_tab.py

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QFrame, QLabel, QPushButton, QLineEdit, QCheckBox,
    QGridLayout
)
from PySide6.QtCore import Qt

from modules.ui.ConfigList import ConfigList     # PySide6 version of ConfigList
from modules.util.config.TrainConfig import TrainConfig
from modules.util.config.TrainConfig import TrainEmbeddingConfig
from modules.util.ui.UIState import UIState


class AdditionalEmbeddingsTab(ConfigList):

    def __init__(self, parent, train_config, ui_state):

        super().__init__(
            master=parent,
            train_config=train_config,
            ui_state=ui_state,
            attr_name="additional_embeddings",
            from_external_file=False,
            add_button_text="add embedding",
            is_full_width=True
        )

        self.parent = parent
        self.train_config = train_config
        self.ui_state = ui_state

    def refresh_ui(self):
        """
        Overridden method that re-creates the widget list from the config.
        """
        self._create_element_list()

    def create_widget(self, parent_widget, element, i, open_command, remove_command, clone_command, save_command):
        """
        Creates an EmbeddingWidget for each embedding.
        """
        return EmbeddingWidget(parent_widget, element, i, open_command, remove_command, clone_command, save_command)

    def create_new_element(self) -> dict:
        """
        Returns a default dictionary for a new embedding.
        """
        return TrainEmbeddingConfig.default_values()

    def open_element_window(self, i, ui_state):
        """
        Stub method: your code returns a ctk.CTkToplevel, but you have no
        actual code here, so we simply pass or create a stub.
        """
        pass


class EmbeddingWidget(QFrame):
    """
    Displays row for each embedding:
      top_frame: close [+], base embedding, placeholder, token_count
      bottom_frame: train, output embedding, stop training after, etc.
    """
    def __init__(self, parent, element, i, open_command, remove_command, clone_command, save_command):
        super().__init__()

        self.element = element
        self.ui_state = UIState(self, element)
        self.i = i
        self.save_command = save_command

        # QFrame config
        self.setFrameShape(QFrame.StyledPanel)
        # If you want a "rounded" background, you'd do QSS.
        # self.setStyleSheet("background-color: transparent; border-radius: 10px;")

        # We'll use a top-level QGridLayout with two rows:
        #   row=0 => top_frame
        #   row=1 => bottom_frame
        self.layout_grid = QGridLayout(self)
        self.layout_grid.setContentsMargins(5, 5, 5, 5)
        self.layout_grid.setSpacing(5)
        self.setLayout(self.layout_grid)

        # top_frame
        self.top_frame = QFrame(self)
        self.top_frame_layout = QGridLayout(self.top_frame)
        self.top_frame_layout.setContentsMargins(0, 0, 0, 0)
        self.top_frame_layout.setSpacing(5)
        self.layout_grid.addWidget(self.top_frame, 0, 0)

        # bottom_frame
        self.bottom_frame = QFrame(self)
        self.bottom_frame_layout = QGridLayout(self.bottom_frame)
        self.bottom_frame_layout.setContentsMargins(0, 0, 0, 0)
        self.bottom_frame_layout.setSpacing(5)
        self.layout_grid.addWidget(self.bottom_frame, 1, 0)

        # Top row
        # 1) Close button (X)
        self.close_button = QPushButton("X", self.top_frame)
        self.close_button.setStyleSheet("background-color: #C00000; color: white; border-radius:2px;")
        self.close_button.setFixedSize(20, 20)
        self.close_button.clicked.connect(lambda: remove_command(self.i))
        self.top_frame_layout.addWidget(self.close_button, 0, 0)

        # 2) Clone button (+)
        self.clone_button = QPushButton("+", self.top_frame)
        self.clone_button.setStyleSheet("background-color: #00C000; color: white; border-radius:2px;")
        self.clone_button.setFixedSize(20, 20)
        self.clone_button.clicked.connect(lambda: clone_command(self.i, self.__randomize_uuid))
        self.top_frame_layout.addWidget(self.clone_button, 0, 1)

        # 3) base embedding label
        lbl_base_embed = QLabel("base embedding:")
        lbl_base_embed.setToolTip("The base embedding to train on. Leave empty to create a new embedding.")
        self.top_frame_layout.addWidget(lbl_base_embed, 0, 2)

        # 3a) base embedding entry (like file_entry logic)
        # We'll do a QLineEdit with a "browse" button if you want. For simplicity, just the line:
        self.base_embed_edit = QLineEdit(self.top_frame)
        # If you want path_modifier logic: you can handle signals or do it in code
        self.base_embed_edit.setText(str(self.element.model_name or ""))
        # Connect if you want to handle text changes
        # self.base_embed_edit.editingFinished.connect(self.save_command)
        self.top_frame_layout.addWidget(self.base_embed_edit, 0, 3)

        # 4) placeholder
        lbl_placeholder = QLabel("placeholder:")
        lbl_placeholder.setToolTip("The placeholder used when using the embedding in a prompt.")
        self.top_frame_layout.addWidget(lbl_placeholder, 0, 4)

        self.placeholder_edit = QLineEdit(self.top_frame)
        self.placeholder_edit.setText(str(self.element.placeholder or ""))
        self.top_frame_layout.addWidget(self.placeholder_edit, 0, 5)

        # 5) token count
        lbl_token_count = QLabel("token count:")
        lbl_token_count.setToolTip("Used when creating a new embedding. Leave empty to auto detect.")
        self.top_frame_layout.addWidget(lbl_token_count, 0, 6)

        self.token_count_edit = QLineEdit(self.top_frame)
        self.token_count_edit.setFixedWidth(40)
        if self.element.token_count is not None:
            self.token_count_edit.setText(str(self.element.token_count))
        self.top_frame_layout.addWidget(self.token_count_edit, 0, 7)

        # Bottom row
        # 1) train
        lbl_train = QLabel("train:")
        self.bottom_frame_layout.addWidget(lbl_train, 0, 0)
        self.train_switch = QCheckBox(self.bottom_frame)
        self.train_switch.setText("")
        self.train_switch.setChecked(bool(self.element.train))
        self.bottom_frame_layout.addWidget(self.train_switch, 0, 1)

        # 2) output embedding
        lbl_out_embed = QLabel("output embedding:")
        lbl_out_embed.setToolTip("If true, the embedding is at the output of the text encoder, not the input.")
        self.bottom_frame_layout.addWidget(lbl_out_embed, 0, 2)
        self.output_embedding_switch = QCheckBox(self.bottom_frame)
        self.output_embedding_switch.setText("")
        self.output_embedding_switch.setChecked(bool(self.element.is_output_embedding))
        self.bottom_frame_layout.addWidget(self.output_embedding_switch, 0, 3)

        # 3) stop training after (time_entry)
        lbl_stop = QLabel("stop training after:")
        lbl_stop.setToolTip("When to stop training the embedding.")
        self.bottom_frame_layout.addWidget(lbl_stop, 0, 4)
        # We'll do a line edit (or 2 line edits if you handle "stop_training_after" + "stop_training_after_unit")
        self.stop_time_edit = QLineEdit(self.bottom_frame)
        if self.element.stop_training_after is not None:
            self.stop_time_edit.setText(str(self.element.stop_training_after))
        self.bottom_frame_layout.addWidget(self.stop_time_edit, 0, 5)

        # 4) initial embedding text
        lbl_init_text = QLabel("initial embedding text:")
        lbl_init_text.setToolTip("Initial text used when creating a new embedding.")
        self.bottom_frame_layout.addWidget(lbl_init_text, 0, 6)

        self.init_text_edit = QLineEdit(self.bottom_frame)
        if self.element.initial_embedding_text:
            self.init_text_edit.setText(self.element.initial_embedding_text)
        self.bottom_frame_layout.addWidget(self.init_text_edit, 0, 7)

    def __randomize_uuid(self, embedding_config):
        """
        Cloning logic: randomize the 'uuid' field
        """
        embedding_config.uuid = type(embedding_config).default_values().uuid
        return embedding_config

    def configure_element(self):
        """
        Called if the element changes externally. Refresh the fields.
        """
        self.base_embed_edit.setText(str(self.element.model_name or ""))
        self.placeholder_edit.setText(str(self.element.placeholder or ""))
        if self.element.token_count is not None:
            self.token_count_edit.setText(str(self.element.token_count))
        else:
            self.token_count_edit.clear()

        self.train_switch.setChecked(bool(self.element.train))
        self.output_embedding_switch.setChecked(bool(self.element.is_output_embedding))
        if self.element.stop_training_after is not None:
            self.stop_time_edit.setText(str(self.element.stop_training_after))
        else:
            self.stop_time_edit.clear()

        self.init_text_edit.setText(str(self.element.initial_embedding_text or ""))

    def place_in_list(self):
        """
        Replicates the .grid(...) usage. 
        In Qt, typically the parent layout does `layout.addWidget(self, row, col)`.
        We'll just keep a stub.
        """
        if self.parentWidget() and hasattr(self.parentWidget(), 'layout'):
            parent_layout = self.parentWidget().layout()
            if isinstance(parent_layout, QGridLayout):
                parent_layout.addWidget(self, self.i, 0, 1, 1)

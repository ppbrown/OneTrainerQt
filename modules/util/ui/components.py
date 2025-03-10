"""
    
    It provides a set of functions to create and manage UI components such as labels, buttons, text fields, and more.
"""

import os
from typing import Any, Callable

from PySide6.QtWidgets import (
    QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox, QProgressBar,
    QScrollArea, QVBoxLayout, QGridLayout,
    QFileDialog, QFrame, QWidget
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap

from PIL import Image

from modules.util.enum.TimeUnit import TimeUnit
from modules.util.path_util import supported_image_extensions
from modules.util.ui.UIState import UIState

PAD = 10

def add_tooltip(widget: QWidget, text: str = "widget info"):
        widget.setToolTip(text)

def app_title(master: QWidget, row: int, column: int):
    """
    Creates a small frame with an icon + "OneTrainer" label, placed at (row, column).
    """
    # Create the frame with a horizontal layout (simpler for one row of widgets)
    frame = QFrame(master)
    layout = QHBoxLayout(frame)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(PAD)

    # Add the frame to master's grid layout if it exists
    grid = master.layout()
    if grid is not None and isinstance(grid, QGridLayout):
        grid.addWidget(frame, row, column)

    # Load and process the icon image
    try:
        pil_img = Image.open("resources/icons/icon.png").resize((40, 40), Image.Resampling.LANCZOS)
        pixmap = QPixmap.fromImage(pil_img.toqimage())
    except Exception:
        pixmap = QPixmap()

    label_icon = QLabel()
    label_icon.setPixmap(pixmap)
    label_icon.setFixedSize(40, 40)
    layout.addWidget(label_icon)

    label_text = QLabel("OneTrainer")
    label_text.setStyleSheet("font-size: 20px; font-weight: bold;")
    layout.addWidget(label_text)

def create_gridlayout(scroll_area: QScrollArea):
    """
    Given a QScrollArea widget, will pre-populate it with a widget tree that allows use of a gridlayout
    that will compact itself around the added widgets, rather than expanding to fill all available space.
    Parameters:
        scroll_area (QScrollArea): The scroll area where the container widget and grid layout are embedded.
        
    Returns:
        QWidget: A container with a grid layout for adding further widgets. Use container.layout() to get the grid layout.
    """
    scroll_container = QFrame()

    vbox = QVBoxLayout(scroll_container)
    vbox.setContentsMargins(0, 0, 0, 0)
    vbox.setSpacing(10)
    scroll_area.setWidget(scroll_container)
    scroll_container.setMinimumWidth(300)
    scroll_container.setMinimumHeight(200)
    scroll_container.setLayout(vbox)

    container = QWidget()
    grid_layout = QGridLayout(container)
    container.setLayout(grid_layout)
    vbox.addWidget(container, alignment=Qt.AlignTop | Qt.AlignLeft)

    # Add a stretch to push the grid to the top so extra space stays empty
    vbox.addStretch()

    return container
    # We return container instead of just the grid layout, because our label() func below expects a widget.

# Should be called create_label()
def label(
    master: QWidget,
    row: int,
    column: int,
    text: str,
    pad: int = PAD,
    tooltip: str | None = None,
    wide_tooltip: bool = False,
    wraplength: int = 0
):
    lbl = QLabel(text, master)
    # No direct "wraplength" in Qt, we can do word wrap
    if wraplength > 0:
        lbl.setWordWrap(True)
    grid = master.layout()
    if isinstance(grid, QGridLayout):
        grid.addWidget(lbl, row, column, 1, 1)
        # For padding, we can do spacing around
        lbl.setContentsMargins(pad, pad, pad, pad)

    if tooltip:
        add_tooltip(lbl, tooltip)

    return lbl

# Should be called create_textentry()
def entry(
    master: QWidget,
    row: int,
    column: int,
    ui_state: UIState,
    var_name: str,
    command: Callable[[], None] = None,
    tooltip: str = "",
    wide_tooltip: bool = False,
    width: int = 140,
    sticky: str = "new",
):
    """
    Creates a QLineEdit bound to ui_state's variable var_name.
    'command' is called whenever the variable changes or editing is finished.
    """
    var = ui_state.get_var(var_name)
    line_edit = QLineEdit(master)
    line_edit.setText(var.get())
    if width > 0:
        line_edit.setFixedWidth(width)

    grid = master.layout()
    if isinstance(grid, QGridLayout):
        grid.addWidget(line_edit, row, column, 1, 1)

    if tooltip:
        add_tooltip(line_edit, tooltip)

    def on_editing_finished():
        text_value = line_edit.text()
        var.set(text_value)
        if command:
            command()

    line_edit.editingFinished.connect(on_editing_finished)

    return line_edit


def file_entry(
    master: QWidget,
    row: int,
    column: int,
    ui_state: UIState,
    var_name: str,
    is_output: bool = False,
    path_modifier: Callable[[str], str] | None = None,
    allow_model_files: bool = True,
    allow_image_files: bool = False,
    command: Callable[[str], None] = None,
):
    """
    Combines a QLineEdit with a "..." button that opens a file dialog.
    """
    container = QFrame(master)
    container_layout = QGridLayout(container)
    container_layout.setContentsMargins(0,0,0,0)
    container_layout.setSpacing(PAD)
    container.setLayout(container_layout)

    grid = master.layout()
    if isinstance(grid, QGridLayout):
        grid.addWidget(container, row, column, 1, 1)

    line_edit = entry(
        container, 0, 0,
        ui_state, var_name,
        command=None,
        width=150
    )

    def open_dialog():
        filetypes = [("All Files", "*.*")]
        if allow_model_files:
            filetypes.extend([
                ("Diffusers", "model_index.json"),
                ("Checkpoint", "*.ckpt *.pt *.bin"),
                ("Safetensors", "*.safetensors"),
            ])
        if allow_image_files:
            exts = ' '.join([f"*.{x}" for x in supported_image_extensions()])
            filetypes.append(("Image", exts))

        dlg = QFileDialog(container)
        dlg.setNameFilters([f"{desc} ({pat})" for desc, pat in filetypes])

        if is_output:
            dlg.setAcceptMode(QFileDialog.AcceptSave)
        else:
            dlg.setAcceptMode(QFileDialog.AcceptOpen)

        if dlg.exec() == QFileDialog.Accepted:
            file_path = dlg.selectedFiles()[0]
            if path_modifier:
                file_path = path_modifier(file_path)
            line_edit.setText(file_path)
            ui_state.get_var(var_name).set(file_path)
            if command:
                command(file_path)

    btn = QPushButton("...", container)
    btn.setFixedWidth(40)
    container_layout.addWidget(btn, 0, 1, 1, 1)
    btn.clicked.connect(open_dialog)

    return container


def dir_entry(master: QWidget, row: int, column: int, ui_state: UIState, var_name: str,
              command: Callable[[str], None] = None):
    container = QFrame(master)
    container_layout = QGridLayout(container)
    container_layout.setContentsMargins(0,0,0,0)
    container_layout.setSpacing(PAD)
    container.setLayout(container_layout)

    grid = master.layout()
    if isinstance(grid, QGridLayout):
        grid.addWidget(container, row, column)

    line_edit = entry(container, 0, 0, ui_state, var_name, width=150)

    def open_dialog():
        dlg = QFileDialog(container)
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec() == QFileDialog.Accepted:
            selected = dlg.selectedFiles()[0]
            ui_state.get_var(var_name).set(selected)
            line_edit.setText(selected)
            if command:
                command(selected)

    btn = QPushButton("...", container)
    btn.setFixedWidth(40)
    container_layout.addWidget(btn, 0, 1)
    btn.clicked.connect(open_dialog)

    return container


def time_entry(
    master: QWidget, row: int, column: int,
    ui_state: UIState, var_name: str,
    unit_var_name: str,
    supports_time_units: bool = True
):
    container = QFrame(master)
    container_layout = QGridLayout(container)
    container_layout.setContentsMargins(0,0,0,0)
    container_layout.setSpacing(PAD)
    container.setLayout(container_layout)

    grid = master.layout()
    if isinstance(grid, QGridLayout):
        grid.addWidget(container, row, column, 1, 1)

    line_edit = entry(container, 0, 0, ui_state, var_name, width=50)

    all_values = [str(x) for x in list(TimeUnit)]
    if not supports_time_units:
        all_values = [str(x) for x in list(TimeUnit) if not x.is_time_unit()]

    combo = QComboBox(container)
    combo.addItems(all_values)
    container_layout.addWidget(combo, 0, 1)

    # bind the combo to var
    var_unit = ui_state.get_var(unit_var_name)

    def on_combo_change(idx):
        var_unit.set(combo.currentText())

    combo.currentIndexChanged.connect(on_combo_change)

    # we also trace var_unit => update combo
    def update_combo(*args):
        current_val = var_unit.get()
        if current_val in all_values:
            combo.setCurrentText(current_val)
        else:
            combo.setCurrentText(all_values[0])

    ## FIXLATER: var_unit.trace_add("write", lambda _0, _1, _2: update_combo())
    update_combo()

    return container


def icon_button(master: QWidget, row: int, column: int, text: str, command: Callable[[], None]):
    btn = QPushButton(text, master)
    btn.setFixedWidth(40)
    grid = master.layout()
    if isinstance(grid, QGridLayout):
        grid.addWidget(btn, row, column)
    btn.clicked.connect(command)
    return btn

# Should be called create_button()
def button(
    master: QWidget,
    row: int, column: int,
    text: str,
    command: Callable[[], None],
    tooltip: str | None = None
):
    btn = QPushButton(text, master)
    grid = master.layout()
    if isinstance(grid, QGridLayout):
        grid.addWidget(btn, row, column, 1, 1)
    btn.clicked.connect(command)

    if tooltip:
        add_tooltip(btn, tooltip)

    return btn

# Should be called create_options_box()
def options(
    master: QWidget,
    row: int,
    column: int,
    values: list[str],
    ui_state: UIState,
    var_name: str,
    command: Callable[[str], None] | None = None
):
    combo = QComboBox(master)
    combo.addItems(values)

    var = ui_state.get_var(var_name)

    def on_combo_changed(index):
        selected_text = combo.currentText()
        var.set(selected_text)
        if command:
            command(selected_text)

    combo.currentIndexChanged.connect(on_combo_changed)

    # Set the initial
    if var.get() in values:
        combo.setCurrentText(var.get())
    else:
        var.set(values[0])
        combo.setCurrentIndex(0)

    def var_callback(*args):
        val = var.get()
        if val in values:
            combo.setCurrentText(val)
        else:
            combo.setCurrentIndex(0)

    ## FIXLATER: var.trace_add("write", lambda _0, _1, _2: var_callback())

    grid = master.layout()
    if isinstance(grid, QGridLayout):
        grid.addWidget(combo, row, column, 1, 1)

    return combo


def options_adv(
    master: QWidget,
    row: int,
    column: int,
    values: list[str],
    ui_state: UIState,
    var_name: str,
    command: Callable[[str], None] | None = None,
    adv_command: Callable[[], None] | None = None
):
    container = QFrame(master)
    container_layout = QGridLayout(container)
    container_layout.setContentsMargins(0,0,0,0)
    container_layout.setSpacing(PAD)
    container.setLayout(container_layout)

    grid = master.layout()
    if isinstance(grid, QGridLayout):
        grid.addWidget(container, row, column, 1, 1)

    combo = options(container, 0, 0, values, ui_state, var_name, command=command)
    btn = QPushButton("…", container)
    btn.setFixedWidth(20)
    container_layout.addWidget(btn, 0, 1)
    if adv_command:
        btn.clicked.connect(adv_command)

    return container, {'component': combo, 'button_component': btn}


def options_kv(
    master: QWidget,
    row: int,
    column: int,
    values: list[tuple[str, Any]],
    ui_state: UIState,
    var_name: str,
    command: Callable[[Any], None] | None = None
):
    """
    Creates a QComboBox that shows 'keys' from values, but sets var to the 'value'.
    """
    combo = QComboBox(master)

    # Prepare data
    keys = [kv[0] for kv in values]
    combos_map = {kv[0]: kv[1] for kv in values}

    combo.addItems(keys)

    var = ui_state.get_var(var_name)

    def on_combo_change(index: int):
        selected_key = combo.currentText()
        internal_val = combos_map[selected_key]
        var.set(internal_val)
        if command:
            command(internal_val)

    combo.currentIndexChanged.connect(on_combo_change)

    # Set initial
    current_val = var.get()
    # find a key whose value matches current_val
    matched_key = None
    for k, v in values:
        if str(v) == str(current_val):
            matched_key = k
            break
    if matched_key is not None:
        combo.setCurrentText(matched_key)
    else:
        # default to the first
        combo.setCurrentIndex(0)
        var.set(values[0][1])
        if command:
            command(values[0][1])

    def var_callback(*args):
        new_val = var.get()
        # find the key
        found = False
        for k, v in values:
            if str(v) == str(new_val):
                combo.setCurrentText(k)
                found = True
                break
        if not found:
            # default again
            combo.setCurrentIndex(0)
            var.set(values[0][1])
            if command:
                command(values[0][1])

    ## FIXLATER: var.trace_add("write", lambda _0, _1, _2: var_callback())

    grid = master.layout()
    if isinstance(grid, QGridLayout):
        grid.addWidget(combo, row, column)

    return combo

# Should be called create_toggleswitch()
def switch(
    master: QWidget,
    row: int,
    column: int,
    ui_state: UIState,
    var_name: str,
    command: Callable[[], None] | None = None,
    text: str = "",
):
    """
    Creates a QCheckBox which toggles the boolean variable in ui_state[var_name].
    """
    var = ui_state.get_var(var_name)
    checkbox = QCheckBox(text, master)
    checkbox.setChecked(bool(var.get()))

    def on_state_changed(state: int):
        var.set(bool(state))
        if command:
            command()

    checkbox.stateChanged.connect(on_state_changed)

    def var_callback(*args):
        new_val = var.get()
        checkbox.setChecked(bool(new_val))

    ## FIXLATER: var.trace_add("write", lambda _0, _1, _2: var_callback())

    grid = master.layout()
    if isinstance(grid, QGridLayout):
        grid.addWidget(checkbox, row, column)

    return checkbox


def progress(master: QWidget, row: int, column: int):
    prog = QProgressBar(master)
    grid = master.layout()
    if isinstance(grid, QGridLayout):
        grid.addWidget(prog, row, column)
    return prog

# Should be called create_double_progress_bars()
def double_progress(master: QWidget, row: int, column: int, label_1: str, label_2: str):
    """
    Creates two labeled QProgressBars side by side (vertical stack).
    Returns two setter functions: set_1(value, max_value), set_2(value, max_value).
    """
    container = QFrame(master)
    container_layout = QGridLayout(container)
    container_layout.setContentsMargins(0, 0, 0, 0)
    container_layout.setSpacing(PAD)
    container.setLayout(container_layout)

    grid = master.layout()
    if isinstance(grid, QGridLayout):
        grid.addWidget(container, row, column)

    lbl_1 = QLabel(label_1, container)
    container_layout.addWidget(lbl_1, 0, 0)
    prog_1 = QProgressBar(container)
    container_layout.addWidget(prog_1, 0, 1)
    desc_1 = QLabel("", container)
    container_layout.addWidget(desc_1, 0, 2)

    lbl_2 = QLabel(label_2, container)
    container_layout.addWidget(lbl_2, 1, 0)
    prog_2 = QProgressBar(container)
    container_layout.addWidget(prog_2, 1, 1)
    desc_2 = QLabel("", container)
    container_layout.addWidget(desc_2, 1, 2)

    def set_1(value: int, max_value: int):
        if max_value > 0:
            prog_1.setRange(0, max_value)
            prog_1.setValue(value)
        else:
            prog_1.setRange(0, 0)
        desc_1.setText(f"{value}/{max_value}")

    def set_2(value: int, max_value: int):
        if max_value > 0:
            prog_2.setRange(0, max_value)
            prog_2.setValue(value)
        else:
            prog_2.setRange(0, 0)
        desc_2.setText(f"{value}/{max_value}")

    return set_1, set_2

import os
import platform
import subprocess
import traceback

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QCheckBox, QLineEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea, QFileDialog,
    QMessageBox, QSizePolicy
)
from PySide6.QtCore import Qt, QPoint, QRect, QSize
from PySide6.QtGui import QPixmap, QImage, QMouseEvent, QAction, QKeyEvent

# If you have your own modules, import them here:
from modules.module.Blip2Model import Blip2Model
from modules.module.BlipModel import BlipModel
from modules.module.ClipSegModel import ClipSegModel
from modules.module.MaskByColor import MaskByColor
from modules.module.RembgHumanModel import RembgHumanModel
from modules.module.RembgModel import RembgModel
from modules.module.WDModel import WDModel

from modules.ui.GenerateCaptionsWindow import GenerateCaptionsWindow
from modules.ui.GenerateMasksWindow import GenerateMasksWindow

from modules.util import path_util
from modules.util.torch_util import default_device
import torch

import cv2
import numpy as np
from PIL import Image, ImageDraw

class CaptionUI(QMainWindow):
    """
    A PySide6 translation of your customtkinter-based CaptionUI.
    """
    def __init__(
        self,
        parent=None,
        initial_dir: str|None = None,
        initial_include_subdirectories: bool = False
    ):
        super().__init__(parent)

        # -------------------------------------------------------------------
        # Basic window configuration
        # -------------------------------------------------------------------
        self.setWindowTitle("OneTrainer")
        # If you want a fixed size (like resizable(False, False) in Tk):
        # self.setFixedSize(1280, 980)
        self.show()
        self.raise_()

        # -------------------------------------------------------------------
        # Data / State
        # -------------------------------------------------------------------
        self.dir = initial_dir
        self.config_ui_data = {
            "include_subdirectories": initial_include_subdirectories
        }
        # You could still use a custom UIState if you wish, or just store directly:
        # self.config_ui_state = UIState(self, self.config_ui_data)

        # Note: this size MUST be smaller than screen size or very bad things happen.
        self.image_size = 650
        self.help_text = (
            "Keyboard shortcuts when focusing on the prompt input field:\n"
            "Up arrow: previous image\n"
            "Down arrow: next image\n"
            "Return: save\n"
            "Ctrl+M: only show the mask\n"
            "Ctrl+D: draw mask editing mode\n"
            "Ctrl+F: fill mask editing mode\n\n"
            "When editing masks:\n"
            "Left click: add mask\n"
            "Right click: remove mask\n"
            "Mouse wheel: increase or decrease brush size"
        )

        self.masking_model = None
        self.captioning_model = None

        # List of relative image paths in self.dir
        self.image_rel_paths = []
        self.current_image_index = -1

        # Image & mask data
        self.pil_image = None
        self.image_width = 0
        self.image_height = 0
        self.pil_mask = None
        self.mask_draw_x = 0
        self.mask_draw_y = 0
        self.mask_draw_radius = 0.01
        self.display_only_mask = False
        self.mask_editing_mode = 'draw'
        self.enable_mask_editing = False  # Instead of BooleanVar

        # For prompt editing
        self.prompt_text = ""

        self.brush_alpha = 1.0

        # -------------------------------------------------------------------
        # Main Layout
        # -------------------------------------------------------------------
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        # "Top bar" row
        self.top_bar_layout = QHBoxLayout()
        main_layout.addLayout(self.top_bar_layout)
        self.setup_top_bar()

        # "Bottom" area: left column = file list, right column = content
        self.bottom_layout = QHBoxLayout()
        main_layout.addLayout(self.bottom_layout, stretch=1)

        # Left column: scrollable list of images
        self.file_list_scroll_area = QScrollArea()
        self.file_list_scroll_area.setWidgetResizable(True)
        self.file_list_container = QWidget()
        self.file_list_layout = QVBoxLayout(self.file_list_container)
        self.file_list_scroll_area.setWidget(self.file_list_container)
        self.bottom_layout.addWidget(self.file_list_scroll_area, 0)

        # Center content (image, mask, caption)
        self.content_widget = QWidget()
        self.content_layout = QGridLayout(self.content_widget)
        self.bottom_layout.addWidget(self.content_widget, 1)
        self.setup_content_column()

        # -------------------------------------------------------------------
        # Initialize directory and image list
        # -------------------------------------------------------------------
        self.load_directory(self.config_ui_data["include_subdirectories"])

    # -----------------------------------------------------------------------
    # Top bar
    # -----------------------------------------------------------------------
    def setup_top_bar(self):
        open_button = QPushButton("Open")
        open_button.setToolTip("Open a new directory")
        open_button.clicked.connect(self.open_directory)
        self.top_bar_layout.addWidget(open_button)

        gen_masks_button = QPushButton("Generate Masks")
        gen_masks_button.setToolTip("Automatically generate masks")
        gen_masks_button.clicked.connect(self.open_mask_window)
        self.top_bar_layout.addWidget(gen_masks_button)

        gen_captions_button = QPushButton("Generate Captions")
        gen_captions_button.setToolTip("Automatically generate captions")
        gen_captions_button.clicked.connect(self.open_caption_window)
        self.top_bar_layout.addWidget(gen_captions_button)

        if platform.system() == "Windows":
            explorer_button = QPushButton("Open in Explorer")
            explorer_button.setToolTip("Open the current image in Explorer")
            explorer_button.clicked.connect(self.open_in_explorer)
            self.top_bar_layout.addWidget(explorer_button)

        # "Include subdirectories" switch -> QCheckBox
        self.include_subdirectories_checkbox = QCheckBox("include subdirectories")
        self.include_subdirectories_checkbox.setChecked(
            self.config_ui_data["include_subdirectories"]
        )
        self.include_subdirectories_checkbox.stateChanged.connect(
            self.toggle_include_subdirectories
        )
        self.top_bar_layout.addWidget(self.include_subdirectories_checkbox)

        # Add some stretch to push the "Help" button to the right
        self.top_bar_layout.addStretch(1)

        help_button = QPushButton("Help")
        help_button.setToolTip(self.help_text)
        help_button.clicked.connect(self.print_help)
        self.top_bar_layout.addWidget(help_button)

    def toggle_include_subdirectories(self, state):
        self.config_ui_data["include_subdirectories"] = bool(state)
        # If you want to refresh automatically each time it's toggled:
        # self.load_directory(self.config_ui_data["include_subdirectories"])

    # -----------------------------------------------------------------------
    # File list (left column)
    # -----------------------------------------------------------------------
    def rebuild_file_list_ui(self):
        """
        Clears out the old file-list layout, rebuilds labels for each image.
        """
        # Clear old items
        while self.file_list_layout.count() > 0:
            child = self.file_list_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Rebuild
        for i, rel_path in enumerate(self.image_rel_paths):
            label = ClickableLabel(rel_path)
            label.clicked_index.connect(self.switch_image)
            label.set_index(i)
            label.setText(rel_path)
            # Use a little indentation
            label.setContentsMargins(5, 2, 5, 2)
            self.file_list_layout.addWidget(label)

        self.file_list_layout.addStretch(1)

    # -----------------------------------------------------------------------
    # Content column (center areas)
    # -----------------------------------------------------------------------
    def setup_content_column(self):
        """
        This sets up the image area, mask editing controls, prompt entry, etc.
        """
        # -- Row 0: top row of buttons/checkbox to set mask mode --
        draw_button = QPushButton("Draw")
        draw_button.setToolTip("Draw a mask using a brush")
        draw_button.clicked.connect(self.draw_mask_editing_mode)
        self.content_layout.addWidget(draw_button, 0, 0)

        fill_button = QPushButton("Fill")
        fill_button.setToolTip("Draw a mask using a fill tool")
        fill_button.clicked.connect(self.fill_mask_editing_mode)
        self.content_layout.addWidget(fill_button, 0, 1)

        self.enable_mask_editing_checkbox = QCheckBox("Enable Mask Editing")
        self.enable_mask_editing_checkbox.setChecked(False)
        self.enable_mask_editing_checkbox.stateChanged.connect(self.set_mask_editing_enabled)
        self.content_layout.addWidget(self.enable_mask_editing_checkbox, 0, 2)

        self.brush_alpha_line = QLineEdit()
        self.brush_alpha_line.setFixedWidth(40)
        self.brush_alpha_line.setText("1.0")
        self.brush_alpha_line.returnPressed.connect(self.update_brush_alpha)
        self.content_layout.addWidget(self.brush_alpha_line, 0, 3)

        alpha_label = QLabel("Brush Alpha")
        self.content_layout.addWidget(alpha_label, 0, 4)

        # -- Row 1: image display --
        self.image_label = ClickableLabel("")
        # Let the label expand only to a certain size
        self.image_label.setFixedSize(self.image_size, self.image_size)
        # Enable mouse tracking to receive mouseMoveEvents
        self.image_label.setMouseTracking(True)
        self.image_label.mouse_moved.connect(self.edit_mask_mouse_move)
        self.image_label.mouse_pressed.connect(self.edit_mask_mouse_press)
        self.image_label.wheel_scrolled.connect(self.draw_mask_radius)
        self.content_layout.addWidget(self.image_label, 1, 0, 1, 5)

        # -- Row 2: prompt entry --
        self.prompt_line = QLineEdit()
        self.prompt_line.returnPressed.connect(self.save_current)
        self.content_layout.addWidget(self.prompt_line, 2, 0, 1, 5)

    def set_mask_editing_enabled(self, state):
        self.enable_mask_editing = bool(state)

    def update_brush_alpha(self):
        try:
            val = float(self.brush_alpha_line.text())
            self.brush_alpha = max(0.0, min(val, 1.0))
        except ValueError:
            self.brush_alpha = 1.0

    # -----------------------------------------------------------------------
    # Directory scanning and loading
    # -----------------------------------------------------------------------
    def load_directory(self, include_subdirectories=False):
        self.scan_directory(include_subdirectories)
        self.rebuild_file_list_ui()

        if len(self.image_rel_paths) > 0:
            self.switch_image(0)
        else:
            self.switch_image(-1)

        # Focus on the prompt line
        self.prompt_line.setFocus()

    def scan_directory(self, include_subdirectories=False):
        def __is_supported_image_extension(filename):
            name, ext = os.path.splitext(filename)
            return path_util.is_supported_image_extension(ext) and not name.endswith("-masklabel")

        self.image_rel_paths.clear()

        if not self.dir or not os.path.isdir(self.dir):
            return

        if include_subdirectories:
            for root, _, files in os.walk(self.dir):
                for filename in files:
                    if __is_supported_image_extension(filename):
                        rel = os.path.relpath(os.path.join(root, filename), self.dir)
                        self.image_rel_paths.append(rel)
        else:
            for filename in os.listdir(self.dir):
                if __is_supported_image_extension(filename):
                    rel = os.path.relpath(os.path.join(self.dir, filename), self.dir)
                    self.image_rel_paths.append(rel)

    # -----------------------------------------------------------------------
    # Image, Mask, Prompt loading
    # -----------------------------------------------------------------------
    def load_image(self):
        if (
            len(self.image_rel_paths) > 0 and
            0 <= self.current_image_index < len(self.image_rel_paths)
        ):
            image_rel = self.image_rel_paths[self.current_image_index]
            fullpath = os.path.join(self.dir, image_rel)
            try:
                return Image.open(fullpath).convert('RGB')
            except Exception:
                traceback.print_exc()
                print(f"Could not open image {fullpath}")
        # Fallback
        return Image.new("RGB", (512, 512), (0, 0, 0))

    def load_mask(self):
        if (
            len(self.image_rel_paths) > 0 and
            0 <= self.current_image_index < len(self.image_rel_paths)
        ):
            image_rel = self.image_rel_paths[self.current_image_index]
            mask_path = os.path.splitext(image_rel)[0] + "-masklabel.png"
            mask_path = os.path.join(self.dir, mask_path)
            if os.path.exists(mask_path):
                try:
                    return Image.open(mask_path).convert('RGB')
                except Exception:
                    return None
        return None

    def load_prompt(self):
        if (
            len(self.image_rel_paths) > 0 and
            0 <= self.current_image_index < len(self.image_rel_paths)
        ):
            image_rel = self.image_rel_paths[self.current_image_index]
            prompt_path = os.path.splitext(image_rel)[0] + ".txt"
            prompt_path = os.path.join(self.dir, prompt_path)
            if os.path.exists(prompt_path):
                try:
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        return f.read().splitlines()[0]
                except Exception:
                    return ""
        return ""

    # -----------------------------------------------------------------------
    # Image switching
    # -----------------------------------------------------------------------
    def switch_image(self, index: int):
        # Deselect old label color
        old_label = self.find_label_by_index(self.current_image_index)
        if old_label:
            old_label.setStyleSheet("color: black;")

        self.current_image_index = index
        label = self.find_label_by_index(index)
        if label:
            # highlight in red
            label.setStyleSheet("color: red;")

        if index < 0 or index >= len(self.image_rel_paths):
            # no images
            blank_img = Image.new("RGB", (512, 512), (0, 0, 0))
            self.set_image_on_label(blank_img)
            return

        self.pil_image = self.load_image()
        self.pil_mask = self.load_mask()
        self.prompt_text = self.load_prompt()
        self.prompt_line.setText(self.prompt_text)

        if self.pil_image:
            self.image_width = self.pil_image.width
            self.image_height = self.pil_image.height
            scale = self.image_size / max(self.pil_image.width, self.pil_image.height)
            new_w = int(self.pil_image.width * scale)
            new_h = int(self.pil_image.height * scale)

            self.pil_image = self.pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.refresh_image()
        else:
            blank_img = Image.new("RGB", (512, 512), (0, 0, 0))
            self.set_image_on_label(blank_img)

    def find_label_by_index(self, index: int):
        """
        Returns the label in the file-list layout with that index, if any.
        (We stored the index in ClickableLabel.)
        """
        for i in range(self.file_list_layout.count()):
            item = self.file_list_layout.itemAt(i)
            if item and item.widget():
                w = item.widget()
                if isinstance(w, ClickableLabel) and w.index == index:
                    return w
        return None

    def refresh_image(self):
        """ Combine self.pil_image + self.pil_mask if necessary and show on the image_label. """
        if not self.pil_image:
            return
        if self.pil_mask:
            resized_mask = self.pil_mask.resize(
                (self.pil_image.width, self.pil_image.height),
                Image.Resampling.NEAREST
            )
            if self.display_only_mask:
                self.set_image_on_label(resized_mask)
            else:
                np_image = np.array(self.pil_image).astype(np.float32) / 255.0
                np_mask = np.array(resized_mask).astype(np.float32) / 255.0
                # normalize mask between 0.3 - 1.0
                norm_min = 0.3
                mask_min = np_mask.min()
                if mask_min == 0:
                    np_mask = np_mask * (1.0 - norm_min) + norm_min
                elif mask_min < 1:
                    np_mask = ((np_mask - mask_min) / (1.0 - mask_min)) * (1.0 - norm_min) + norm_min

                np_masked_image = (np_image * np_mask * 255.0).astype(np.uint8)
                masked_pil = Image.fromarray(np_masked_image, mode="RGB")
                self.set_image_on_label(masked_pil)
        else:
            self.set_image_on_label(self.pil_image)

    def set_image_on_label(self, pil_image):
        """
        Convert PIL -> QPixmap, set on self.image_label.
        """
        data = pil_image.tobytes("raw", "RGB")
        qimg = QImage(
            data, pil_image.width, pil_image.height,
            3 * pil_image.width, QImage.Format_RGB888
        )
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix)

    # -----------------------------------------------------------------------
    # Mouse/Keyboard handling for next/prev images and mask editing
    # -----------------------------------------------------------------------
    def keyPressEvent(self, event: QKeyEvent):
        """
        Overall keypress handling for the QMainWindow.
        If your prompt line is focused, you can also handle it inside
        a custom event filter or by subclassing QLineEdit.
        """
        if event.key() == Qt.Key_Up:
            self.previous_image()
            return
        elif event.key() == Qt.Key_Down:
            self.next_image()
            return
        elif event.key() == Qt.Key_Return:
            self.save_current()
            return

        # Check modifiers for Ctrl+D, Ctrl+F, Ctrl+M
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_D:
                self.draw_mask_editing_mode()
                return
            elif event.key() == Qt.Key_F:
                self.fill_mask_editing_mode()
                return
            elif event.key() == Qt.Key_M:
                self.toggle_mask()
                return

        super().keyPressEvent(event)

    def previous_image(self):
        if self.current_image_index - 1 >= 0:
            self.switch_image(self.current_image_index - 1)

    def next_image(self):
        if self.current_image_index + 1 < len(self.image_rel_paths):
            self.switch_image(self.current_image_index + 1)

    # Mask editing mouse events come from ClickableLabel signals
    def edit_mask_mouse_move(self, pos: QPoint):
        if not self.enable_mask_editing:
            return
        self.edit_mask_common(pos, is_press=False)

    def edit_mask_mouse_press(self, pos: QPoint, button: int):
        if not self.enable_mask_editing:
            return
        self.edit_mask_common(pos, is_press=True, button=button)

    def edit_mask_common(self, pos: QPoint, is_press=False, button=None):
        if (
            len(self.image_rel_paths) == 0 or
            self.current_image_index >= len(self.image_rel_paths) or
            self.current_image_index < 0
        ):
            return

        # Convert from label coords to actual image coords
        event_x = pos.x()
        event_y = pos.y()
        # Map to full-size image
        start_x = int(event_x / self.pil_image.width * self.image_width)
        start_y = int(event_y / self.pil_image.height * self.image_height)
        end_x = int(self.mask_draw_x / self.pil_image.width * self.image_width)
        end_y = int(self.mask_draw_y / self.pil_image.height * self.image_height)

        self.mask_draw_x = event_x
        self.mask_draw_y = event_y

        # Distinguish left vs right
        is_left = (button == Qt.LeftButton)
        is_right = (button == Qt.RightButton)

        if self.mask_editing_mode == 'draw':
            if is_press:  # Only draw on press or drag? Up to you
                self.draw_mask(start_x, start_y, end_x, end_y, is_left, is_right)
        elif self.mask_editing_mode == 'fill':
            if is_press:
                self.fill_mask(start_x, start_y, end_x, end_y, is_left, is_right)

    # -----------------------------------------------------------------------
    # Mask editing modes
    # -----------------------------------------------------------------------
    def draw_mask(self, start_x, start_y, end_x, end_y, is_left, is_right):
        color = None
        if is_left:
            rgb_value = int(self.brush_alpha * 255)
            color = (rgb_value, rgb_value, rgb_value)
        elif is_right:
            color = (0, 0, 0)

        if color is not None:
            if self.pil_mask is None:
                # If we have no mask, create one
                # If we're adding -> black background. If removing -> white background.
                if is_left:
                    self.pil_mask = Image.new('RGB', (self.image_width, self.image_height), (0, 0, 0))
                else:
                    self.pil_mask = Image.new('RGB', (self.image_width, self.image_height), (255, 255, 255))

            radius = int(self.mask_draw_radius * max(self.pil_mask.width, self.pil_mask.height))

            draw_obj = ImageDraw.Draw(self.pil_mask)
            # line
            draw_obj.line((start_x, start_y, end_x, end_y), fill=color, width=(2*radius + 1))
            # start ellipse
            draw_obj.ellipse((start_x - radius, start_y - radius,
                              start_x + radius, start_y + radius), fill=color)
            # end ellipse
            draw_obj.ellipse((end_x - radius, end_y - radius,
                              end_x + radius, end_y + radius), fill=color)

            self.refresh_image()

    def fill_mask(self, start_x, start_y, end_x, end_y, is_left, is_right):
        color = None
        if is_left:
            rgb_value = int(self.brush_alpha * 255)
            color = (rgb_value, rgb_value, rgb_value)
        elif is_right:
            color = (0, 0, 0)

        if color is not None:
            if self.pil_mask is None:
                if is_left:
                    self.pil_mask = Image.new('RGB', (self.image_width, self.image_height), (0, 0, 0))
                else:
                    self.pil_mask = Image.new('RGB', (self.image_width, self.image_height), (255, 255, 255))

            np_mask = np.array(self.pil_mask, dtype=np.uint8)
            h, w, _ = np_mask.shape
            if 0 <= start_x < w and 0 <= start_y < h:
                cv2.floodFill(np_mask, None, (start_x, start_y), color)
                self.pil_mask = Image.fromarray(np_mask, 'RGB')
                self.refresh_image()

    def draw_mask_radius(self, delta):
        # Wheel up = Increase radius, wheel down = Decrease radius
        multiplier = 1.0 + (delta * 0.05)
        self.mask_draw_radius = max(0.0025, self.mask_draw_radius * multiplier)

    def draw_mask_editing_mode(self):
        self.mask_editing_mode = 'draw'

    def fill_mask_editing_mode(self):
        self.mask_editing_mode = 'fill'

    def toggle_mask(self):
        self.display_only_mask = not self.display_only_mask
        self.refresh_image()

    # -----------------------------------------------------------------------
    # Saving
    # -----------------------------------------------------------------------
    def save_current(self):
        if (
            len(self.image_rel_paths) == 0 or
            self.current_image_index >= len(self.image_rel_paths) or
            self.current_image_index < 0
        ):
            return
        image_rel = self.image_rel_paths[self.current_image_index]

        prompt_path = os.path.splitext(image_rel)[0] + ".txt"
        prompt_path = os.path.join(self.dir, prompt_path)
        mask_path = os.path.splitext(image_rel)[0] + "-masklabel.png"
        mask_path = os.path.join(self.dir, mask_path)

        self.prompt_text = self.prompt_line.text()

        # Save prompt
        try:
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(self.prompt_text)
        except Exception:
            traceback.print_exc()

        # Save mask
        if self.pil_mask:
            try:
                self.pil_mask.save(mask_path)
            except Exception:
                traceback.print_exc()

    # -----------------------------------------------------------------------
    # Directory / File open
    # -----------------------------------------------------------------------
    def open_directory(self):
        new_dir = QFileDialog.getExistingDirectory(self, "Select Directory", self.dir or "")
        if new_dir:
            self.dir = new_dir
            self.load_directory(self.config_ui_data["include_subdirectories"])

    def open_mask_window(self):
        # Show GenerateMasksWindow (PySide version would typically be QDialog)
        dialog = GenerateMasksWindow(self, self.dir, self.config_ui_data["include_subdirectories"])
        dialog.exec()
        self.switch_image(self.current_image_index)

    def open_caption_window(self):
        dialog = GenerateCaptionsWindow(self, self.dir, self.config_ui_data["include_subdirectories"])
        dialog.exec()
        self.switch_image(self.current_image_index)

    def open_in_explorer(self):
        try:
            if (
                len(self.image_rel_paths) == 0 or
                self.current_image_index >= len(self.image_rel_paths) or
                self.current_image_index < 0
            ):
                return
            image_rel = self.image_rel_paths[self.current_image_index]
            fullpath = os.path.realpath(os.path.join(self.dir, image_rel))
            subprocess.Popen(f"explorer /select,{fullpath}")
        except Exception:
            traceback.print_exc()

    def load_masking_model(self, model):
        self.captioning_model = None

        if model == "ClipSeg":
            if not isinstance(self.masking_model, ClipSegModel):
                print("loading ClipSeg model, this may take a while")
                self.masking_model = ClipSegModel(default_device, torch.float32)
        elif model == "Rembg":
            if not isinstance(self.masking_model, RembgModel):
                print("loading Rembg model, this may take a while")
                self.masking_model = RembgModel(default_device, torch.float32)
        elif model == "Rembg-Human":
            if not isinstance(self.masking_model, RembgHumanModel):
                print("loading Rembg-Human model, this may take a while")
                self.masking_model = RembgHumanModel(default_device, torch.float32)
        elif model == "Hex Color":
            if not isinstance(self.masking_model, MaskByColor):
                self.masking_model = MaskByColor(default_device, torch.float32)

    def load_captioning_model(self, model):
        self.masking_model = None

        if model == "Blip":
            if not isinstance(self.captioning_model, BlipModel):
                print("loading Blip model, this may take a while")
                self.captioning_model = BlipModel(default_device, torch.float16)
        elif model == "Blip2":
            if not isinstance(self.captioning_model, Blip2Model):
                print("loading Blip2 model, this may take a while")
                self.captioning_model = Blip2Model(default_device, torch.float16)
        elif model == "WD14 VIT v2":
            if not isinstance(self.captioning_model, WDModel):
                print("loading WD14_VIT_v2 model, this may take a while")
                self.captioning_model = WDModel(default_device, torch.float16)

    # -----------------------------------------------------------------------
    # Help
    # -----------------------------------------------------------------------
    def print_help(self):
        QMessageBox.information(self, "Help", self.help_text)

# -----------------------------------------------------------------------------
# Helper: a custom clickable label that can emit signals for press/move/wheel
# -----------------------------------------------------------------------------
from PySide6.QtCore import Signal

class ClickableLabel(QLabel):
    """
    A QLabel that can emit a clicked signal and also
    let us track mouseMove and wheel events.
    """
    clicked_index = Signal(int)
    mouse_moved = Signal(QPoint)
    mouse_pressed = Signal(QPoint, int)
    wheel_scrolled = Signal(float)

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.index = -1
        self.setCursor(Qt.ArrowCursor)

    def set_index(self, index: int):
        self.index = index

    def mousePressEvent(self, event: QMouseEvent):
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton or event.button() == Qt.RightButton:
            # If the label is used as a file list entry, we just want to switch_image on left click
            if self.index >= 0:
                self.clicked_index.emit(self.index)
            # If the label is the main image_label, we want to handle mask editing
            self.mouse_pressed.emit(event.pos(), event.button())

    def mouseMoveEvent(self, event: QMouseEvent):
        super().mouseMoveEvent(event)
        self.mouse_moved.emit(event.pos())

    def wheelEvent(self, event):
        """
        The direction of the wheel can be read from event.angleDelta().y()
        Typically a positive value = wheel up, negative = wheel down
        """
        delta_y = event.angleDelta().y()
        # Normalize to +1 or -1
        direction = 1 if delta_y > 0 else -1
        self.wheel_scrolled.emit(direction)
        event.accept()



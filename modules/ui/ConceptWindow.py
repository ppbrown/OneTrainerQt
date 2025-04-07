import math
import multiprocessing
import os
import pathlib
import random
import time
import traceback

import torch
from torchvision.transforms import functional

from matplotlib import pyplot as plt
# Use the QtAgg backend for Matplotlib instead of the TkAgg backend:
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PIL import Image

from PySide6.QtWidgets import (
    QDialog, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QCheckBox, QPushButton, QScrollArea,
    QFrame, QPlainTextEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QFont

from modules.util import concept_stats, path_util
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.enum.BalancingStrategy import BalancingStrategy
from modules.util.ui import components
from modules.util.ui.UIState import UIState

# mgds pipeline modules
from mgds.LoadingPipeline import LoadingPipeline
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModules.RandomBrightness import RandomBrightness
from mgds.pipelineModules.RandomCircularMaskShrink import RandomCircularMaskShrink
from mgds.pipelineModules.RandomContrast import RandomContrast
from mgds.pipelineModules.RandomFlip import RandomFlip
from mgds.pipelineModules.RandomHue import RandomHue
from mgds.pipelineModules.RandomMaskRotateCrop import RandomMaskRotateCrop
from mgds.pipelineModules.RandomRotate import RandomRotate
from mgds.pipelineModules.RandomSaturation import RandomSaturation
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class InputPipelineModule(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, data: dict):
        super().__init__()
        self.data = data

    def length(self) -> int:
        return 1

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return list(self.data.keys())

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return self.data


class ConceptWindow(QDialog):
    def __init__(
        self,
        parent,
        concept,
        ui_state: UIState,
        image_ui_state: UIState,
        text_ui_state: UIState,
        *args,
        **kwargs
    ):
        super().__init__()
        self.concept = concept
        self.ui_state = ui_state
        self.image_ui_state = image_ui_state
        self.text_ui_state = text_ui_state
        self.image_preview_file_index = 0

        # Setup QDialog
        self.setWindowTitle("Concept")
        self.resize(800, 700)

        # Main layout for the entire dialog
        self.main_layout = QGridLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(5)
        self.setLayout(self.main_layout)

        # QTabWidget in row=0
        self.tabview = QTabWidget(self)
        self.main_layout.addWidget(self.tabview, 0, 0, 1, 1)

        # Add tabs
        self.general_tab_widget = QWidget()
        self.tabview.addTab(self.general_tab_widget, "general")

        self.image_augmentation_widget = QWidget()
        self.tabview.addTab(self.image_augmentation_widget, "image augmentation")

        self.text_augmentation_widget = QWidget()
        self.tabview.addTab(self.text_augmentation_widget, "text augmentation")

        self.concept_stats_tab_widget = QWidget()
        self.tabview.addTab(self.concept_stats_tab_widget, "statistics")

        # Build each tab
        self.__general_tab(self.general_tab_widget)
        self.__image_augmentation_tab(self.image_augmentation_widget)
        self.__text_augmentation_tab(self.text_augmentation_widget)
        self.__concept_stats_tab(self.concept_stats_tab_widget)

        # "OK" button at row=1
        self.ok_button = QPushButton("ok", self)
        self.ok_button.clicked.connect(self.__ok)
        self.main_layout.addWidget(self.ok_button, 1, 0, 1, 1, alignment=Qt.AlignRight)

    def __general_tab(self, master: QWidget):

        scroll_area = QScrollArea(master)
        scroll_area.setWidgetResizable(True)
        layout_master = QVBoxLayout(master)
        layout_master.setContentsMargins(0, 0, 0, 0)
        layout_master.setSpacing(5)
        master.setLayout(layout_master)
        layout_master.addWidget(scroll_area)

        # Container inside the scroll
        container = QFrame()
        container_layout = QGridLayout(container)
        container_layout.setContentsMargins(5, 5, 5, 5)
        container_layout.setSpacing(5)
        container.setLayout(container_layout)
        scroll_area.setWidget(container)

        # name
        components.label(container, 0, 0, "Name", tooltip="Name of the concept")
        components.entry(container, 0, 1, self.ui_state, "name")

        # enabled
        components.label(container, 1, 0, "Enabled", tooltip="Enable or disable this concept")
        components.switch(container, 1, 1, self.ui_state, "enabled")

        # validation_concept
        components.label(container, 2, 0, "Validation concept", tooltip="Use concept for validation instead of training")
        components.switch(container, 2, 1, self.ui_state, "validation_concept")

        # path
        components.label(container, 3, 0, "Path", tooltip="Path for training data")
        components.dir_entry(container, 3, 1, self.ui_state, "path")

        # prompt source
        components.label(container, 4, 0, "Prompt Source",
                         tooltip="The source of prompts used. 'From text file per sample', etc.")
        prompt_path_entry = components.file_entry(container, 4, 2, self.text_ui_state, "prompt_path")

        def set_prompt_path_entry_enabled(option: str):
            # Enable the file entry's sub-widgets only if 'concept' is chosen.
            for child in prompt_path_entry.children():
                child.setEnabled(option == 'concept')

        components.options_kv(
            container, 4, 1,
            [
                ("From text file per sample", 'sample'),
                ("From single text file", 'concept'),
                ("From image file name", 'filename'),
            ],
            self.text_ui_state, "prompt_source", command=set_prompt_path_entry_enabled
        )
        set_prompt_path_entry_enabled(self.concept.text.prompt_source)

        # include subdirectories
        components.label(container, 5, 0, "Include Subdirectories")
        components.switch(container, 5, 1, self.ui_state, "include_subdirectories")

        # image variations
        components.label(container, 6, 0, "Image Variations")
        components.entry(container, 6, 1, self.ui_state, "image_variations")

        # text variations
        components.label(container, 7, 0, "Text Variations")
        components.entry(container, 7, 1, self.ui_state, "text_variations")

        # balancing
        components.label(container, 8, 0, "Balancing")
        components.entry(container, 8, 1, self.ui_state, "balancing")
        components.options(container, 8, 2, [str(x) for x in list(BalancingStrategy)], self.ui_state, "balancing_strategy")

        # loss weight
        components.label(container, 9, 0, "Loss Weight")
        components.entry(container, 9, 1, self.ui_state, "loss_weight")

    def __image_augmentation_tab(self, master: QWidget):
        """
        Window with a grid for the 'Random' and 'Fixed' columns, plus an image preview.
        """
        scroll_area = QScrollArea(master)
        scroll_area.setWidgetResizable(True)
        layout_master = QVBoxLayout(master)
        layout_master.setContentsMargins(0, 0, 0, 0)
        layout_master.setSpacing(5)
        master.setLayout(layout_master)
        layout_master.addWidget(scroll_area)

        container = QFrame()
        container_layout = QGridLayout(container)
        container_layout.setContentsMargins(5, 5, 5, 5)
        container_layout.setSpacing(5)
        container.setLayout(container_layout)
        scroll_area.setWidget(container)

        # header
        components.label(container, 0, 1, "Random")
        components.label(container, 0, 2, "Fixed")

        # crop jitter
        components.label(container, 1, 0, "Crop Jitter")
        components.switch(container, 1, 1, self.image_ui_state, "enable_crop_jitter")

        # random flip
        components.label(container, 2, 0, "Random Flip")
        components.switch(container, 2, 1, self.image_ui_state, "enable_random_flip")
        components.switch(container, 2, 2, self.image_ui_state, "enable_fixed_flip")

        # random rotate
        components.label(container, 3, 0, "Random Rotation")
        components.switch(container, 3, 1, self.image_ui_state, "enable_random_rotate")
        components.switch(container, 3, 2, self.image_ui_state, "enable_fixed_rotate")
        components.entry(container, 3, 3, self.image_ui_state, "random_rotate_max_angle")

        # brightness
        components.label(container, 4, 0, "Random Brightness")
        components.switch(container, 4, 1, self.image_ui_state, "enable_random_brightness")
        components.switch(container, 4, 2, self.image_ui_state, "enable_fixed_brightness")
        components.entry(container, 4, 3, self.image_ui_state, "random_brightness_max_strength")

        # contrast
        components.label(container, 5, 0, "Random Contrast")
        components.switch(container, 5, 1, self.image_ui_state, "enable_random_contrast")
        components.switch(container, 5, 2, self.image_ui_state, "enable_fixed_contrast")
        components.entry(container, 5, 3, self.image_ui_state, "random_contrast_max_strength")

        # saturation
        components.label(container, 6, 0, "Random Saturation")
        components.switch(container, 6, 1, self.image_ui_state, "enable_random_saturation")
        components.switch(container, 6, 2, self.image_ui_state, "enable_fixed_saturation")
        components.entry(container, 6, 3, self.image_ui_state, "random_saturation_max_strength")

        # hue
        components.label(container, 7, 0, "Random Hue")
        components.switch(container, 7, 1, self.image_ui_state, "enable_random_hue")
        components.switch(container, 7, 2, self.image_ui_state, "enable_fixed_hue")
        components.entry(container, 7, 3, self.image_ui_state, "random_hue_max_strength")

        # circular mask shrink
        components.label(container, 8, 0, "Circular Mask Generation")
        components.switch(container, 8, 1, self.image_ui_state, "enable_random_circular_mask_shrink")

        # random rotate/crop
        components.label(container, 9, 0, "Random Rotate and Crop")
        components.switch(container, 9, 1, self.image_ui_state, "enable_random_mask_rotate_crop")

        # resolution override
        components.label(container, 10, 0, "Resolution Override")
        components.switch(container, 10, 2, self.image_ui_state, "enable_resolution_override")
        components.entry(container, 10, 3, self.image_ui_state, "resolution_override")

        # image preview
        image_preview, filename_preview, caption_preview = self.__get_preview_image()
        self.preview_pixmap = self.__pil_to_qpixmap(image_preview)

        self.image_label = QLabel(container)
        self.image_label.setPixmap(self.preview_pixmap)
        self.image_label.setFixedSize(min(image_preview.width, 300), min(image_preview.height, 300))
        self.image_label.setScaledContents(True)
        container_layout.addWidget(self.image_label, 0, 4, 6, 1)  # row=0..5

        # refresh preview buttons
        update_button_frame = QFrame(container)
        update_button_layout = QGridLayout(update_button_frame)
        update_button_layout.setContentsMargins(0, 0, 0, 0)
        update_button_layout.setSpacing(5)
        update_button_frame.setLayout(update_button_layout)
        container_layout.addWidget(update_button_frame, 6, 4, 1, 1)

        self.prev_preview_button = QPushButton("<", update_button_frame)
        self.prev_preview_button.setFixedWidth(40)
        self.prev_preview_button.clicked.connect(self.__prev_image_preview)
        update_button_layout.addWidget(self.prev_preview_button, 0, 0)

        self.update_preview_button = QPushButton("Update Preview", update_button_frame)
        self.update_preview_button.clicked.connect(self.__update_image_preview)
        update_button_layout.addWidget(self.update_preview_button, 0, 1)

        self.next_preview_button = QPushButton(">", update_button_frame)
        self.next_preview_button.setFixedWidth(40)
        self.next_preview_button.clicked.connect(self.__next_image_preview)
        update_button_layout.addWidget(self.next_preview_button, 0, 2)

        # filename preview
        self.filename_preview_label = QLabel(container)
        self.filename_preview_label.setText(filename_preview)
        self.filename_preview_label.setFixedWidth(300)
        container_layout.addWidget(self.filename_preview_label, 7, 4, 1, 1)

        # caption preview
        self.caption_preview = QPlainTextEdit(container)
        self.caption_preview.setFixedSize(300, 150)
        self.caption_preview.setReadOnly(True)
        self.caption_preview.insertPlainText(caption_preview)
        container_layout.addWidget(self.caption_preview, 8, 4, 4, 1)

    def __text_augmentation_tab(self, master: QWidget):
        """
        Another QScrollArea with a container that has a QGridLayout for text augmentation controls.
        """
        scroll_area = QScrollArea(master)
        scroll_area.setWidgetResizable(True)
        layout_master = QVBoxLayout(master)
        layout_master.setContentsMargins(0, 0, 0, 0)
        layout_master.setSpacing(5)
        master.setLayout(layout_master)
        layout_master.addWidget(scroll_area)

        container = QFrame()
        container_layout = QGridLayout(container)
        container_layout.setContentsMargins(5, 5, 5, 5)
        container_layout.setSpacing(5)
        container.setLayout(container_layout)
        scroll_area.setWidget(container)

        # tag shuffling
        components.label(container, 0, 0, "Tag Shuffling")
        components.switch(container, 0, 1, self.text_ui_state, "enable_tag_shuffling")

        # tag delimiter
        components.label(container, 1, 0, "Tag Delimiter")
        components.entry(container, 1, 1, self.text_ui_state, "tag_delimiter")

        # keep_tags_count
        components.label(container, 2, 0, "Keep Tag Count")
        components.entry(container, 2, 1, self.text_ui_state, "keep_tags_count")

        # tag dropout
        components.label(container, 3, 0, "Tag Dropout")
        components.switch(container, 3, 1, self.text_ui_state, "tag_dropout_enable")

        # dropout mode
        components.label(container, 4, 0, "Dropout Mode")
        components.options_kv(
            container, 4, 1,
            [
                ("Full", 'FULL'),
                ("Random", 'RANDOM'),
                ("Random Weighted", 'RANDOM WEIGHTED'),
            ],
            self.text_ui_state, "tag_dropout_mode", None
        )
        components.label(container, 4, 2, "Probability")
        components.entry(container, 4, 3, self.text_ui_state, "tag_dropout_probability")

        # special dropout tags
        components.label(container, 5, 0, "Special Dropout Tags")
        components.options_kv(
            container, 5, 1,
            [
                ("None", 'NONE'),
                ("Blacklist", 'BLACKLIST'),
                ("Whitelist", 'WHITELIST'),
            ],
            self.text_ui_state, "tag_dropout_special_tags_mode", None
        )
        components.entry(container, 5, 2, self.text_ui_state, "tag_dropout_special_tags")

        # special tags regex
        components.label(container, 6, 0, "Special Tags Regex")
        components.switch(container, 6, 1, self.text_ui_state, "tag_dropout_special_tags_regex")

        # randomize capitalization
        components.label(container, 7, 0, "Randomize Capitalization")
        components.switch(container, 7, 1, self.text_ui_state, "caps_randomize_enable")
        components.label(container, 7, 2, "Force Lowercase")
        components.switch(container, 7, 3, self.text_ui_state, "caps_randomize_lowercase")

        # capitalization mode
        components.label(container, 8, 0, "Capitalization Mode")
        components.entry(container, 8, 1, self.text_ui_state, "caps_randomize_mode")
        components.label(container, 8, 2, "Probability")
        components.entry(container, 8, 3, self.text_ui_state, "caps_randomize_probability")

    def __concept_stats_tab(self, master: QWidget):
        """
        Example of converting the old CustomTkinter-based scrollable frame and
        Tk-based Matplotlib canvas into pure PySide6.
        """
        scroll_area = QScrollArea(master)
        scroll_area.setWidgetResizable(True)
        layout_master = QVBoxLayout(master)
        layout_master.setContentsMargins(0, 0, 0, 0)
        layout_master.setSpacing(5)
        master.setLayout(layout_master)
        layout_master.addWidget(scroll_area)

        # Container inside the scroll
        container = QFrame()
        grid_layout = QGridLayout(container)
        grid_layout.setContentsMargins(5, 5, 5, 5)
        grid_layout.setSpacing(5)
        container.setLayout(grid_layout)

        scroll_area.setWidget(container)

        # We'll mimic the old "frame.grid_columnconfigure" calls:
        # for col in range(4):
        #     grid_layout.setColumnStretch(col, 0)
        #     grid_layout.setColumnMinimumWidth(col, 150)

        self.cancel_scan_flag = multiprocessing.Event()

        # Example for setting a label with underline via QFont:
        def set_underline(label: QLabel):
            font = label.font()
            font.setUnderline(True)
            label.setFont(font)

        # file size
        self.file_size_label = components.label(container, 1, 0, "Total Size", pad=0,
                         tooltip="Total size of all image, mask, and caption files in MB")
        set_underline(self.file_size_label)

        self.file_size_preview = components.label(container, 2, 0, pad=0, text="-")

        # directory count
        self.dir_count_label = components.label(container, 1, 1, "Directories", pad=0,
                         tooltip="Total number of directories...")
        set_underline(self.dir_count_label)
        self.dir_count_preview = components.label(container, 2, 1, pad=0, text="-")

        # image count
        self.image_count_label = components.label(container, 3, 0, "\nTotal Images", pad=0,
                         tooltip="Total number of image files...")
        set_underline(self.image_count_label)
        self.image_count_preview = components.label(container, 4, 0, pad=0, text="-")

        self.video_count_label = components.label(container, 3, 1, "\nTotal Videos", pad=0,
                         tooltip="Total number of video files...")
        set_underline(self.video_count_label)
        self.video_count_preview = components.label(container, 4, 1, pad=0, text="-")

        self.mask_count_label = components.label(container, 3, 2, "\nTotal Masks", pad=0,
                         tooltip="Total number of mask files...")
        set_underline(self.mask_count_label)
        self.mask_count_preview = components.label(container, 4, 2, pad=0, text="-")

        self.caption_count_label = components.label(container, 3, 3, "\nTotal Captions", pad=0,
                         tooltip="Total number of caption files...")
        set_underline(self.caption_count_label)
        self.caption_count_preview = components.label(container, 4, 3, pad=0, text="-")

        # advanced img/vid stats
        self.image_count_mask_label = components.label(container, 5, 0, "\nImages with Masks", pad=0,
                         tooltip="Total number of image files with an associated mask")
        set_underline(self.image_count_mask_label)
        self.image_count_mask_preview = components.label(container, 6, 0, pad=0, text="-")

        self.mask_count_label_unpaired = components.label(container, 5, 1, "\nUnpaired Masks", pad=0,
                         tooltip="Total number of mask files which lack a corresponding image file")
        set_underline(self.mask_count_label_unpaired)
        self.mask_count_preview_unpaired = components.label(container, 6, 1, pad=0, text="-")

        self.image_count_caption_label = components.label(container, 7, 0, "\nImages with Captions", pad=0,
                         tooltip="Total number of image files with an associated caption")
        set_underline(self.image_count_caption_label)
        self.image_count_caption_preview = components.label(container, 8, 0, pad=0, text="-")

        self.video_count_caption_label = components.label(container, 7, 1, "\nVideos with Captions", pad=0,
                         tooltip="Total number of video files with an associated caption")
        set_underline(self.video_count_caption_label)
        self.video_count_caption_preview = components.label(container, 8, 1, pad=0, text="-")

        self.caption_count_label_unpaired = components.label(container, 7, 2, "\nUnpaired Captions", pad=0,
                         tooltip="Total number of caption files which lack a corresponding image file...")
        set_underline(self.caption_count_label_unpaired)
        self.caption_count_preview_unpaired = components.label(container, 8, 2, pad=0, text="-")

        # resolution info
        self.pixel_max_label = components.label(container, 9, 0, "\nMax Pixels", pad=0,
                         tooltip="Largest image in the concept by number of pixels")
        set_underline(self.pixel_max_label)
        self.pixel_max_preview = components.label(container, 10, 0, pad=0, text="-", wraplength=150)

        self.pixel_avg_label = components.label(container, 9, 1, "\nAvg Pixels", pad=0,
                         tooltip="Average size of images in the concept by number of pixels")
        set_underline(self.pixel_avg_label)
        self.pixel_avg_preview = components.label(container, 10, 1, pad=0, text="-", wraplength=150)

        self.pixel_min_label = components.label(container, 9, 2, "\nMin Pixels", pad=0,
                         tooltip="Smallest image in the concept by number of pixels")
        set_underline(self.pixel_min_label)
        self.pixel_min_preview = components.label(container, 10, 2, pad=0, text="-", wraplength=150)

        # video length info
        self.length_max_label = components.label(container, 11, 0, "\nMax Length", pad=0,
                         tooltip="Longest video in the concept by number of frames")
        set_underline(self.length_max_label)
        self.length_max_preview = components.label(container, 12, 0, pad=0, text="-", wraplength=150)

        self.length_avg_label = components.label(container, 11, 1, "\nAvg Length", pad=0,
                         tooltip="Average length of videos in the concept by frames")
        set_underline(self.length_avg_label)
        self.length_avg_preview = components.label(container, 12, 1, pad=0, text="-", wraplength=150)

        self.length_min_label = components.label(container, 11, 2, "\nMin Length", pad=0,
                         tooltip="Shortest video in the concept by frames")
        set_underline(self.length_min_label)
        self.length_min_preview = components.label(container, 12, 2, pad=0, text="-", wraplength=150)

        # video fps info
        self.fps_max_label = components.label(container, 13, 0, "\nMax FPS", pad=0,
                         tooltip="Video in concept with highest fps")
        set_underline(self.fps_max_label)
        self.fps_max_preview = components.label(container, 14, 0, pad=0, text="-", wraplength=150)

        self.fps_avg_label = components.label(container, 13, 1, "\nAvg FPS", pad=0,
                         tooltip="Average fps of videos in the concept")
        set_underline(self.fps_avg_label)
        self.fps_avg_preview = components.label(container, 14, 1, pad=0, text="-", wraplength=150)

        self.fps_min_label = components.label(container, 13, 2, "\nMin FPS", pad=0,
                         tooltip="Video in concept with the lowest fps")
        set_underline(self.fps_min_label)
        self.fps_min_preview = components.label(container, 14, 2, pad=0, text="-", wraplength=150)

        # caption info
        self.caption_max_label = components.label(container, 15, 0, "\nMax Caption Length", pad=0,
                         tooltip="Largest caption in concept by character count")
        set_underline(self.caption_max_label)
        self.caption_max_preview = components.label(container, 16, 0, pad=0, text="-", wraplength=150)

        self.caption_avg_label = components.label(container, 15, 1, "\nAvg Caption Length", pad=0,
                         tooltip="Average length of caption in concept by character count")
        set_underline(self.caption_avg_label)
        self.caption_avg_preview = components.label(container, 16, 1, pad=0, text="-", wraplength=150)

        self.caption_min_label = components.label(container, 15, 2, "\nMin Caption Length", pad=0,
                         tooltip="Smallest caption in concept by character count")
        set_underline(self.caption_min_label)
        self.caption_min_preview = components.label(container, 16, 2, pad=0, text="-", wraplength=150)

        # aspect bucket info
        self.aspect_bucket_label = components.label(container, 17, 0, "\nAspect Bucketing", pad=0,
                         tooltip="Graph of possible aspect buckets...")
        set_underline(self.aspect_bucket_label)

        self.small_bucket_label = components.label(container, 17, 1, "\nSmallest Buckets", pad=0,
                         tooltip="Image buckets with the least nonzero total images...")
        set_underline(self.small_bucket_label)
        self.small_bucket_preview = components.label(container, 18, 1, pad=0, text="-")

        # Setup Matplotlib figure for aspect bucketing
        plt.set_loglevel('WARNING')  # hush messages about data type in bar chart
        self.bucket_fig, self.bucket_ax = plt.subplots(figsize=(7, 2))
        # Create a Qt-based figure canvas
        self.canvas = FigureCanvasQTAgg(self.bucket_fig)
        # Add it to layout at row=19, col=0..3
        grid_layout.addWidget(self.canvas, 19, 0, 2, 4)
        self.bucket_fig.tight_layout()

        # For the sake of minimal changes, we won't do fancy color theming here:
        # you can set colors if desired, or just leave defaults:
        # self.bucket_fig.set_facecolor("#ffffff")
        # self.bucket_ax.set_facecolor("#ffffff")

        # Refresh stats - must be after all labels are defined
        components.button(
            master=container,
            row=0, column=0, text="Refresh Basic",
            command=lambda: self.__get_concept_stats_threaded(False, 9999),
            tooltip="Reload basic statistics for the concept directory"
        )
        components.button(
            master=container,
            row=0, column=1, text="Refresh Advanced",
            command=lambda: [
                self.__get_concept_stats_threaded(False, 9999),
                self.__get_concept_stats(True, 9999)
            ],
            tooltip="Reload advanced statistics for the concept directory"
        )
        components.button(
            master=container,
            row=0, column=2, text="Abort Scan",
            command=lambda: self.__cancel_concept_stats(),
            tooltip="Stop the currently running scan..."
        )
        self.processing_time = components.label(container, 0, 3, text="-", tooltip="Time taken to process concept directory")

        # Instead of old .pack(...), we already have it in scroll_area’s layout.
        return container

    def __prev_image_preview(self):
        self.image_preview_file_index = max(self.image_preview_file_index - 1, 0)
        self.__update_image_preview()

    def __next_image_preview(self):
        self.image_preview_file_index += 1
        self.__update_image_preview()

    def __update_image_preview(self):
        image_preview, filename_preview, caption_preview = self.__get_preview_image()
        self.preview_pixmap = self.__pil_to_qpixmap(image_preview)
        self.image_label.setPixmap(self.preview_pixmap)
        self.image_label.setFixedSize(min(image_preview.width, 300), min(image_preview.height, 300))
        self.image_label.setScaledContents(True)

        self.filename_preview_label.setText(filename_preview)
        self.caption_preview.setReadOnly(False)
        self.caption_preview.clear()
        self.caption_preview.insertPlainText(caption_preview)
        self.caption_preview.setReadOnly(True)

    def __get_preview_image(self):
        """
        pick an image from the concept.path, apply MGDS pipeline.
        """
        preview_image_path = "resources/icons/icon.png"
        file_index = -1
        glob_pattern = "**/*.*" if self.concept.include_subdirectories else "*.*"

        if os.path.isdir(self.concept.path):
            for path in pathlib.Path(self.concept.path).glob(glob_pattern):
                extension = os.path.splitext(path)[1]
                if (path.is_file()
                    and path_util.is_supported_image_extension(extension)
                    and not path.name.endswith("-masklabel.png")):
                    preview_image_path = path_util.canonical_join(self.concept.path, path)
                    file_index += 1
                    if file_index == self.image_preview_file_index:
                        break

        image = Image.open(preview_image_path).convert("RGB")
        image_tensor = functional.to_tensor(image)

        splitext = os.path.splitext(preview_image_path)
        preview_mask_path = path_util.canonical_join(splitext[0] + "-masklabel.png")
        if not os.path.isfile(preview_mask_path):
            preview_mask_path = None

        if preview_mask_path:
            mask = Image.open(preview_mask_path).convert("L")
            mask_tensor = functional.to_tensor(mask)
        else:
            mask_tensor = torch.ones((1, image_tensor.shape[1], image_tensor.shape[2]))

        # Prepare pipeline input data
        input_module = InputPipelineModule({
            'true': True,
            'image': image_tensor,
            'mask': mask_tensor,
            'enable_random_flip': self.concept.image.enable_random_flip,
            'enable_fixed_flip': self.concept.image.enable_fixed_flip,
            'enable_random_rotate': self.concept.image.enable_random_rotate,
            'enable_fixed_rotate': self.concept.image.enable_fixed_rotate,
            'random_rotate_max_angle': self.concept.image.random_rotate_max_angle,
            'enable_random_brightness': self.concept.image.enable_random_brightness,
            'enable_fixed_brightness': self.concept.image.enable_fixed_brightness,
            'random_brightness_max_strength': self.concept.image.random_brightness_max_strength,
            'enable_random_contrast': self.concept.image.enable_random_contrast,
            'enable_fixed_contrast': self.concept.image.enable_fixed_contrast,
            'random_contrast_max_strength': self.concept.image.random_contrast_max_strength,
            'enable_random_saturation': self.concept.image.enable_random_saturation,
            'enable_fixed_saturation': self.concept.image.enable_fixed_saturation,
            'random_saturation_max_strength': self.concept.image.random_saturation_max_strength,
            'enable_random_hue': self.concept.image.enable_random_hue,
            'enable_fixed_hue': self.concept.image.enable_fixed_hue,
            'random_hue_max_strength': self.concept.image.random_hue_max_strength,
            'enable_random_circular_mask_shrink': self.concept.image.enable_random_circular_mask_shrink,
            'enable_random_mask_rotate_crop': self.concept.image.enable_random_mask_rotate_crop,
        })

        # MGDS modules
        circular_mask_shrink = RandomCircularMaskShrink(
            mask_name='mask', shrink_probability=1.0,
            shrink_factor_min=0.2, shrink_factor_max=1.0,
            enabled_in_name='enable_random_circular_mask_shrink'
        )
        random_mask_rotate_crop = RandomMaskRotateCrop(
            mask_name='mask',
            additional_names=['image'],
            min_size=512,
            min_padding_percent=10,
            max_padding_percent=30,
            max_rotate_angle=20,
            enabled_in_name='enable_random_mask_rotate_crop'
        )
        random_flip = RandomFlip(
            names=['image','mask'],
            enabled_in_name='enable_random_flip',
            fixed_enabled_in_name='enable_fixed_flip'
        )
        random_rotate = RandomRotate(
            names=['image','mask'],
            enabled_in_name='enable_random_rotate',
            fixed_enabled_in_name='enable_fixed_rotate',
            max_angle_in_name='random_rotate_max_angle'
        )
        random_brightness = RandomBrightness(
            names=['image'],
            enabled_in_name='enable_random_brightness',
            fixed_enabled_in_name='enable_fixed_brightness',
            max_strength_in_name='random_brightness_max_strength'
        )
        random_contrast = RandomContrast(
            names=['image'],
            enabled_in_name='enable_random_contrast',
            fixed_enabled_in_name='enable_fixed_contrast',
            max_strength_in_name='random_contrast_max_strength'
        )
        random_saturation = RandomSaturation(
            names=['image'],
            enabled_in_name='enable_random_saturation',
            fixed_enabled_in_name='enable_fixed_saturation',
            max_strength_in_name='random_saturation_max_strength'
        )
        random_hue = RandomHue(
            names=['image'],
            enabled_in_name='enable_random_hue',
            fixed_enabled_in_name='enable_fixed_hue',
            max_strength_in_name='random_hue_max_strength'
        )
        output_module = OutputPipelineModule(['image','mask'])

        modules = [
            input_module,
            circular_mask_shrink,
            random_mask_rotate_crop,
            random_flip,
            random_rotate,
            random_brightness,
            random_contrast,
            random_saturation,
            random_hue,
            output_module,
        ]

        from mgds.LoadingPipeline import LoadingPipeline
        pipeline = LoadingPipeline(
            device=torch.device('cpu'),
            modules=modules,
            batch_size=1,
            seed=random.randint(0, 2**30),
            state=None,
            initial_epoch=0,
            initial_index=0
        )

        data = pipeline.__next__()
        image_tensor = data['image']
        mask_tensor = data['mask']

        # filename + first line of base caption
        filename_output = os.path.basename(preview_image_path)
        try:
            if self.concept.text.prompt_source == "sample":
                with open(os.path.splitext(preview_image_path)[0] + ".txt", "r", encoding="utf-8") as prompt_file:
                    prompt_output = prompt_file.readline()
            elif self.concept.text.prompt_source == "filename":
                prompt_output = os.path.splitext(os.path.basename(preview_image_path))[0]
            elif self.concept.text.prompt_source == "concept":
                with open(self.concept.text.prompt_path, "r", encoding="utf-8") as prompt_file:
                    prompt_output = prompt_file.readline()
            else:
                prompt_output = "No caption found."
        except FileNotFoundError:
            prompt_output = "No caption found."

        mask_tensor = torch.clamp(mask_tensor, 0.3, 1)
        image_tensor = image_tensor * mask_tensor
        out_image = functional.to_pil_image(image_tensor)
        out_image.thumbnail((300, 300))

        return out_image, filename_output, prompt_output

    def __pil_to_qpixmap(self, pil_image: Image.Image) -> QPixmap:
        """
        Convert a PIL Image to a QPixmap for display in a QLabel.
        """
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        data = pil_image.tobytes("raw", "RGB")
        qimg = QImage(
            data, pil_image.width, pil_image.height,
            3 * pil_image.width, QImage.Format_RGB888
        )
        return QPixmap.fromImage(qimg)

    def __update_concept_stats(self):
        """
        Same logic as before, but replace old tkinter .configure(...) with setText(...).
        Also, we do not need ctk.CTkFont; we can handle underlines via QFont if needed.
        """
        # file size
        self.file_size_preview.setText(str(int(self.concept.concept_stats["file_size"] / 1048576)) + " MB")
        self.processing_time.setText(str(round(self.concept.concept_stats["processing_time"], 2)) + " s")

        # directory count
        self.dir_count_preview.setText(str(self.concept.concept_stats["directory_count"]))

        # image count
        self.image_count_preview.setText(str(self.concept.concept_stats["image_count"]))
        self.image_count_mask_preview.setText(str(self.concept.concept_stats["image_with_mask_count"]))
        self.image_count_caption_preview.setText(str(self.concept.concept_stats["image_with_caption_count"]))

        # video count
        self.video_count_preview.setText(str(self.concept.concept_stats["video_count"]))
        self.video_count_caption_preview.setText(str(self.concept.concept_stats["video_with_caption_count"]))

        # mask count
        self.mask_count_preview.setText(str(self.concept.concept_stats["mask_count"]))
        self.mask_count_preview_unpaired.setText(str(self.concept.concept_stats["unpaired_masks"]))

        # caption count
        self.caption_count_preview.setText(str(self.concept.concept_stats["caption_count"]))
        self.caption_count_preview_unpaired.setText(str(self.concept.concept_stats["unpaired_captions"]))

        # resolution info
        max_pixels = self.concept.concept_stats["max_pixels"]
        avg_pixels = self.concept.concept_stats["avg_pixels"]
        min_pixels = self.concept.concept_stats["min_pixels"]

        if any(isinstance(x, str) for x in [max_pixels, avg_pixels, min_pixels]) or \
           self.concept.concept_stats["image_count"] == 0:
            self.pixel_max_preview.setText("-")
            self.pixel_avg_preview.setText("-")
            self.pixel_min_preview.setText("-")
        else:
            self.pixel_max_preview.setText(
                f'{round(max_pixels[0]/1_000_000, 2)} MP, {max_pixels[2]}\n{max_pixels[1]}'
            )
            self.pixel_avg_preview.setText(
                f'{round(avg_pixels/1_000_000, 2)} MP, ~{int(math.sqrt(avg_pixels))}w x {int(math.sqrt(avg_pixels))}h'
            )
            self.pixel_min_preview.setText(
                f'{round(min_pixels[0]/1_000_000, 2)} MP, {min_pixels[2]}\n{min_pixels[1]}'
            )

        # video length and fps info
        max_length = self.concept.concept_stats["max_length"]
        avg_length = self.concept.concept_stats["avg_length"]
        min_length = self.concept.concept_stats["min_length"]
        max_fps = self.concept.concept_stats["max_fps"]
        avg_fps = self.concept.concept_stats["avg_fps"]
        min_fps = self.concept.concept_stats["min_fps"]

        if any(isinstance(x, str) for x in [max_length, avg_length, min_length]) or \
           self.concept.concept_stats["video_count"] == 0:
            self.length_max_preview.setText("-")
            self.length_avg_preview.setText("-")
            self.length_min_preview.setText("-")
            self.fps_max_preview.setText("-")
            self.fps_avg_preview.setText("-")
            self.fps_min_preview.setText("-")
        else:
            self.length_max_preview.setText(f'{int(max_length[0])} frames\n{max_length[1]}')
            self.length_avg_preview.setText(f'{int(avg_length)} frames')
            self.length_min_preview.setText(f'{int(min_length[0])} frames\n{min_length[1]}')

            self.fps_max_preview.setText(f'{int(max_fps[0])} fps\n{max_fps[1]}')
            self.fps_avg_preview.setText(f'{int(avg_fps)} fps')
            self.fps_min_preview.setText(f'{int(min_fps[0])} fps\n{min_fps[1]}')

        # caption info
        max_caption_length = self.concept.concept_stats["max_caption_length"]
        avg_caption_length = self.concept.concept_stats["avg_caption_length"]
        min_caption_length = self.concept.concept_stats["min_caption_length"]

        if any(isinstance(x, str) for x in [max_caption_length, avg_caption_length, min_caption_length]) or \
           self.concept.concept_stats["caption_count"] == 0:
            self.caption_max_preview.setText("-")
            self.caption_avg_preview.setText("-")
            self.caption_min_preview.setText("-")
        else:
            self.caption_max_preview.setText(
                f'{max_caption_length[0]} chars, {max_caption_length[2]} words\n{max_caption_length[1]}'
            )
            self.caption_avg_preview.setText(
                f'{int(avg_caption_length[0])} chars, {int(avg_caption_length[1])} words'
            )
            self.caption_min_preview.setText(
                f'{min_caption_length[0]} chars, {min_caption_length[2]} words\n{min_caption_length[1]}'
            )

        # aspect bucketing
        aspect_buckets = self.concept.concept_stats["aspect_buckets"]
        self.bucket_ax.cla()
        if len(aspect_buckets) != 0 and max(aspect_buckets.values()) > 0:
            aspects = [str(x) for x in list(aspect_buckets.keys())]
            counts = list(aspect_buckets.values())
            b = self.bucket_ax.bar(aspects, counts)
            # The color for the bar labels is set to black by default – you could change as needed:
            self.bucket_ax.bar_label(b)
        self.canvas.draw()

        # Possibly force a repaint:
        self.update()

    def __get_concept_stats(self, advanced_checks: bool, waittime: float):
        start_time = time.perf_counter()
        last_update = time.perf_counter()
        subfolders = [self.concept.path]
        stats_dict = concept_stats.init_concept_stats(self.concept, advanced_checks)

        for path in subfolders:
            stats_dict = concept_stats.folder_scan(path, stats_dict, advanced_checks, self.concept)
            stats_dict["processing_time"] = time.perf_counter() - start_time
            if self.concept.include_subdirectories:
                subfolders.extend([f for f in os.scandir(path) if f.is_dir()])
            self.concept.concept_stats = stats_dict

            # check for abort or time limit
            if (time.perf_counter() - start_time) > waittime or self.cancel_scan_flag.is_set():
                stats_dict = concept_stats.init_concept_stats(self.concept, advanced_checks)
                stats_dict["processing_time"] = time.perf_counter() - start_time
                self.concept.concept_stats = stats_dict
                self.cancel_scan_flag.clear()
                break

            # update GUI approx every half second
            if time.perf_counter() > (last_update + 0.5):
                last_update = time.perf_counter()
                self.__update_concept_stats()
                # In PySide6, self.update() suffices:
                self.update()

        self.__update_concept_stats()

    def __get_concept_stats_threaded(self, advanced_checks: bool, waittime: float):
        """
        Attempt a threaded or multiprocess approach to avoid blocking.
        This just spawns a process to call __get_concept_stats.
        """
        self.p = multiprocessing.Process(
            target=self.__get_concept_stats,  # pass the function itself
            args=(advanced_checks, waittime),
            daemon=True
        )
        self.p.start()

    def __cancel_concept_stats(self):
        self.cancel_scan_flag.set()

    def __auto_update_concept_stats(self):
        try:
            self.__update_concept_stats()
            if self.concept.concept_stats["image_count"] == 0:  # force rescan if zero images
                raise KeyError
        except KeyError:
            try:
                self.__get_concept_stats_threaded(False, 2)
                if self.concept.concept_stats["processing_time"] < 0.1:
                    self.__get_concept_stats_threaded(True, 2)
            except FileNotFoundError:
                pass

    def __ok(self):
        # self.concept.configure_element()
        self.accept()

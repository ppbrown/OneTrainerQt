# sample_frame.py

import os
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QLabel, QLineEdit, QCheckBox, QComboBox, QPushButton, QFileDialog
)
from PySide6.QtCore import Qt

from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.ui.UIState import UIState


class SampleFrame(QFrame):
    """
    A PySide6 equivalent of your customtkinter-based SampleFrame.
    Replaces the ctk.CTkFrame layout with QFrames and QGridLayouts,
    and uses standard Qt widgets in place of `components.*` calls.
    """

    def __init__(
        self,
        parent,
        sample: SampleConfig,
        ui_state: UIState,
        include_prompt: bool = True,
        include_settings: bool = True,
    ):
        super().__init__(parent)

        self.sample = sample
        self.ui_state = ui_state
        self.include_prompt = include_prompt
        self.include_settings = include_settings

        # Top-level layout for this frame
        self.layout_main = QGridLayout(self)
        self.layout_main.setContentsMargins(0, 0, 0, 0)
        self.layout_main.setSpacing(5)
        self.setLayout(self.layout_main)

        row_index = 0

        # If we want a "top_frame" for prompt
        if self.include_prompt:
            self.top_frame = QFrame(self)
            self.top_frame_layout = QGridLayout(self.top_frame)
            self.top_frame_layout.setContentsMargins(0, 0, 0, 0)
            self.top_frame_layout.setSpacing(5)
            self.top_frame.setLayout(self.top_frame_layout)

            # Place top_frame in row=0
            self.layout_main.addWidget(self.top_frame, row_index, 0, 1, 1, alignment=Qt.AlignTop)
            row_index += 1

        # If we want a "bottom_frame" for settings
        if self.include_settings:
            self.bottom_frame = QFrame(self)
            self.bottom_frame_layout = QGridLayout(self.bottom_frame)
            self.bottom_frame_layout.setContentsMargins(0, 0, 0, 0)
            self.bottom_frame_layout.setSpacing(5)
            self.bottom_frame.setLayout(self.bottom_frame_layout)

            # Place bottom_frame in row=1
            self.layout_main.addWidget(self.bottom_frame, row_index, 0, 1, 1, alignment=Qt.AlignTop)
            row_index += 1

        # Now replicate your "components" calls with standard Qt:
        if self.include_prompt:
            # prompt
            lbl_prompt = QLabel("prompt:")
            self.top_frame_layout.addWidget(lbl_prompt, 0, 0)
            self.prompt_line = QLineEdit()
            self.top_frame_layout.addWidget(self.prompt_line, 0, 1)

            # negative prompt
            lbl_negative = QLabel("negative prompt:")
            self.top_frame_layout.addWidget(lbl_negative, 1, 0)
            self.negative_line = QLineEdit()
            self.top_frame_layout.addWidget(self.negative_line, 1, 1)

        if self.include_settings:
            # row0: width, height
            lbl_width = QLabel("width:")
            self.bottom_frame_layout.addWidget(lbl_width, 0, 0)
            self.width_line = QLineEdit()
            self.bottom_frame_layout.addWidget(self.width_line, 0, 1)

            lbl_height = QLabel("height:")
            self.bottom_frame_layout.addWidget(lbl_height, 0, 2)
            self.height_line = QLineEdit()
            self.bottom_frame_layout.addWidget(self.height_line, 0, 3)

            # row1: frames, length
            lbl_frames = QLabel("frames:")
            lbl_frames.setToolTip("Number of frames to generate (for videos).")
            self.bottom_frame_layout.addWidget(lbl_frames, 1, 0)
            self.frames_line = QLineEdit()
            self.bottom_frame_layout.addWidget(self.frames_line, 1, 1)

            lbl_length = QLabel("length:")
            lbl_length.setToolTip("Length in seconds of audio output.")
            self.bottom_frame_layout.addWidget(lbl_length, 1, 2)
            self.length_line = QLineEdit()
            self.bottom_frame_layout.addWidget(self.length_line, 1, 3)

            # row2: seed, random seed
            lbl_seed = QLabel("seed:")
            self.bottom_frame_layout.addWidget(lbl_seed, 2, 0)
            self.seed_line = QLineEdit()
            self.bottom_frame_layout.addWidget(self.seed_line, 2, 1)

            lbl_random_seed = QLabel("random seed:")
            self.bottom_frame_layout.addWidget(lbl_random_seed, 2, 2)
            self.random_seed_check = QCheckBox()
            self.bottom_frame_layout.addWidget(self.random_seed_check, 2, 3)

            # row3: cfg scale
            lbl_cfg = QLabel("cfg scale:")
            self.bottom_frame_layout.addWidget(lbl_cfg, 3, 0)
            self.cfg_line = QLineEdit()
            self.bottom_frame_layout.addWidget(self.cfg_line, 3, 1)

            # row4: steps, sampler
            lbl_steps = QLabel("steps:")
            self.bottom_frame_layout.addWidget(lbl_steps, 4, 0)
            self.steps_line = QLineEdit()
            self.bottom_frame_layout.addWidget(self.steps_line, 4, 1)

            lbl_sampler = QLabel("sampler:")
            self.bottom_frame_layout.addWidget(lbl_sampler, 4, 2)
            self.sampler_combo = QComboBox()
            # add noise schedulers
            # e.g. for sched in [NoiseScheduler.DDIM, NoiseScheduler.EULER, ...]
            self.sampler_combo.addItem("DDIM", NoiseScheduler.DDIM)
            self.sampler_combo.addItem("Euler", NoiseScheduler.EULER)
            self.sampler_combo.addItem("Euler A", NoiseScheduler.EULER_A)
            self.sampler_combo.addItem("UniPC", NoiseScheduler.UNIPC)
            self.sampler_combo.addItem("Euler Karras", NoiseScheduler.EULER_KARRAS)
            self.sampler_combo.addItem("DPM++ Karras", NoiseScheduler.DPMPP_KARRAS)
            self.sampler_combo.addItem("DPM++ SDE Karras", NoiseScheduler.DPMPP_SDE_KARRAS)
            self.bottom_frame_layout.addWidget(self.sampler_combo, 4, 3)

            # row5: inpainting
            lbl_inpainting = QLabel("inpainting:")
            lbl_inpainting.setToolTip("Enables inpainting sampling if using an inpainting model.")
            self.bottom_frame_layout.addWidget(lbl_inpainting, 5, 0)
            self.inpaint_check = QCheckBox()
            self.bottom_frame_layout.addWidget(self.inpaint_check, 5, 1)

            # row6: base image path
            lbl_base_image = QLabel("base image path:")
            lbl_base_image.setToolTip("The base image used for inpainting.")
            self.bottom_frame_layout.addWidget(lbl_base_image, 6, 0, 1, 1)
            self.base_image_line = QLineEdit()
            self.bottom_frame_layout.addWidget(self.base_image_line, 6, 1, 1, 3)
            # if you want a "browse" button:
            # we can do:
            # self.base_image_browse = QPushButton("Browse")
            # self.bottom_frame_layout.addWidget(self.base_image_browse, 6, 4)
            # self.base_image_browse.clicked.connect(self.__browse_base_image)

            # row7: mask image path
            lbl_mask_image = QLabel("mask image path:")
            lbl_mask_image.setToolTip("The mask used for inpainting.")
            self.bottom_frame_layout.addWidget(lbl_mask_image, 7, 2)
            self.mask_image_line = QLineEdit()
            self.bottom_frame_layout.addWidget(self.mask_image_line, 7, 3)

    # If you want a browse function:
    # def __browse_base_image(self):
    #     file_dialog = QFileDialog(self, "Select Base Image")
    #     file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
    #     if file_dialog.exec() == QFileDialog.Accepted:
    #         selected = file_dialog.selectedFiles()[0]
    #         self.base_image_line.setText(selected)

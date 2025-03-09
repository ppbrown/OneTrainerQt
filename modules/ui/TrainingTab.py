# training_tab.py

from PySide6.QtWidgets import (
    QWidget, QScrollArea, QGridLayout, QVBoxLayout, QHBoxLayout, QFrame,
    QLabel, QLineEdit, QComboBox, QCheckBox, QPushButton
)
from PySide6.QtCore import Qt

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.AlignPropLoss import AlignPropLoss
from modules.util.enum.AttentionMechanism import AttentionMechanism
from modules.util.enum.DataType import DataType
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.GradientCheckpointingMethod import GradientCheckpointingMethod
from modules.util.enum.LearningRateScaler import LearningRateScaler
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.enum.LossScaler import LossScaler
from modules.util.enum.LossWeight import LossWeight
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TimestepDistribution import TimestepDistribution
from modules.util.optimizer_util import change_optimizer
from modules.util.ui.UIState import UIState

# If you have these classes, import them:
# from modules.ui.OffloadingWindow import OffloadingWindow
# from modules.ui.OptimizerParamsWindow import OptimizerParamsWindow
# from modules.ui.SchedulerParamsWindow import SchedulerParamsWindow
# from modules.ui.TimestepDistributionWindow import TimestepDistributionWindow


class TrainingTab(QWidget):
    """
    A PySide6-based replica of your customtkinter-based TrainingTab code.
    """

    def __init__(self, parent: QWidget, train_config: TrainConfig, ui_state: UIState):
        super().__init__(parent)

        self.train_config = train_config
        self.ui_state = ui_state

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)

        # Scroll area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        main_layout.addWidget(self.scroll_area)

        # Container widget for the entire UI
        self.scroll_container = QWidget()
        self.scroll_area.setWidget(self.scroll_container)

        # 3-column grid layout
        self.grid_layout = QGridLayout(self.scroll_container)
        self.grid_layout.setColumnStretch(0, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.setColumnStretch(2, 1)

        # For dynamic calls
        self.lr_scheduler_comp = None
        self.lr_scheduler_adv_comp = None

        self.refresh_ui()

    def refresh_ui(self):
        """
        Clears out existing layout items and rebuilds the UI based on model_type.
        """
        # Clear old items
        while self.grid_layout.count() > 0:
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Create 3 QFrames (or QWidgets) for columns
        col0 = QFrame(self.scroll_container)
        col0_layout = QVBoxLayout(col0)
        self.grid_layout.addWidget(col0, 0, 0)

        col1 = QFrame(self.scroll_container)
        col1_layout = QVBoxLayout(col1)
        self.grid_layout.addWidget(col1, 0, 1)

        col2 = QFrame(self.scroll_container)
        col2_layout = QVBoxLayout(col2)
        self.grid_layout.addWidget(col2, 0, 2)

        # Based on the model type, create frames in each column
        if self.train_config.model_type.is_stable_diffusion():
            self.__setup_stable_diffusion_ui(col0_layout, col1_layout, col2_layout)
        elif self.train_config.model_type.is_stable_diffusion_3():
            self.__setup_stable_diffusion_3_ui(col0_layout, col1_layout, col2_layout)
        elif self.train_config.model_type.is_stable_diffusion_xl():
            self.__setup_stable_diffusion_xl_ui(col0_layout, col1_layout, col2_layout)
        elif self.train_config.model_type.is_wuerstchen():
            self.__setup_wuerstchen_ui(col0_layout, col1_layout, col2_layout)
        elif self.train_config.model_type.is_pixart():
            self.__setup_pixart_alpha_ui(col0_layout, col1_layout, col2_layout)
        elif self.train_config.model_type.is_flux():
            self.__setup_flux_ui(col0_layout, col1_layout, col2_layout)
        elif self.train_config.model_type.is_sana():
            self.__setup_sana_ui(col0_layout, col1_layout, col2_layout)
        elif self.train_config.model_type.is_hunyuan_video():
            self.__setup_hunyuan_video_ui(col0_layout, col1_layout, col2_layout)

    # -------------------------------------------------------------------------
    # The specialized UI setups for each model type
    # -------------------------------------------------------------------------
    def __setup_stable_diffusion_ui(self, col0_layout, col1_layout, col2_layout):
        self.__create_base_frame(col0_layout)
        self.__create_text_encoder_frame(col0_layout)
        self.__create_embedding_frame(col0_layout)

        self.__create_base2_frame(col1_layout)
        self.__create_unet_frame(col1_layout)
        self.__create_noise_frame(col1_layout)

        self.__create_align_prop_frame(col2_layout)
        self.__create_masked_frame(col2_layout)
        self.__create_loss_frame(col2_layout)

    def __setup_stable_diffusion_3_ui(self, col0_layout, col1_layout, col2_layout):
        self.__create_base_frame(col0_layout)
        self.__create_text_encoder_1_frame(col0_layout, supports_include=True)
        self.__create_text_encoder_2_frame(col0_layout, supports_include=True)
        self.__create_text_encoder_3_frame(col0_layout, supports_include=True)
        self.__create_embedding_frame(col0_layout)

        self.__create_base2_frame(col1_layout)
        self.__create_transformer_frame(col1_layout)
        self.__create_noise_frame(col1_layout)

        self.__create_align_prop_frame(col2_layout)
        self.__create_masked_frame(col2_layout)
        self.__create_loss_frame(col2_layout)

    def __setup_stable_diffusion_xl_ui(self, col0_layout, col1_layout, col2_layout):
        self.__create_base_frame(col0_layout)
        self.__create_text_encoder_1_frame(col0_layout)
        self.__create_text_encoder_2_frame(col0_layout)
        self.__create_embedding_frame(col0_layout)

        self.__create_base2_frame(col1_layout)
        self.__create_unet_frame(col1_layout)
        self.__create_noise_frame(col1_layout)

        self.__create_align_prop_frame(col2_layout)
        self.__create_masked_frame(col2_layout)
        self.__create_loss_frame(col2_layout)

    def __setup_wuerstchen_ui(self, col0_layout, col1_layout, col2_layout):
        self.__create_base_frame(col0_layout)
        self.__create_text_encoder_frame(col0_layout)
        self.__create_embedding_frame(col0_layout)

        self.__create_base2_frame(col1_layout)
        self.__create_prior_frame(col1_layout)
        self.__create_noise_frame(col1_layout)

        self.__create_masked_frame(col2_layout)
        self.__create_loss_frame(col2_layout)

    def __setup_pixart_alpha_ui(self, col0_layout, col1_layout, col2_layout):
        self.__create_base_frame(col0_layout)
        self.__create_text_encoder_frame(col0_layout)
        self.__create_embedding_frame(col0_layout)

        self.__create_base2_frame(col1_layout)
        self.__create_prior_frame(col1_layout)
        self.__create_noise_frame(col1_layout)

        self.__create_align_prop_frame(col2_layout)
        self.__create_masked_frame(col2_layout)
        self.__create_loss_frame(col2_layout, supports_vb_loss=True)

    def __setup_flux_ui(self, col0_layout, col1_layout, col2_layout):
        self.__create_base_frame(col0_layout)
        self.__create_text_encoder_1_frame(col0_layout, supports_include=True)
        self.__create_text_encoder_2_frame(col0_layout, supports_include=True)
        self.__create_embedding_frame(col0_layout)

        self.__create_base2_frame(col1_layout)
        self.__create_transformer_frame(col1_layout, supports_guidance_scale=True)
        self.__create_noise_frame(col1_layout)

        self.__create_align_prop_frame(col2_layout)
        self.__create_masked_frame(col2_layout)
        self.__create_loss_frame(col2_layout)

    def __setup_sana_ui(self, col0_layout, col1_layout, col2_layout):
        self.__create_base_frame(col0_layout)
        self.__create_text_encoder_frame(col0_layout)
        self.__create_embedding_frame(col0_layout)

        self.__create_base2_frame(col1_layout)
        self.__create_prior_frame(col1_layout)
        self.__create_noise_frame(col1_layout)

        self.__create_masked_frame(col2_layout)
        self.__create_loss_frame(col2_layout)

    def __setup_hunyuan_video_ui(self, col0_layout, col1_layout, col2_layout):
        self.__create_base_frame(col0_layout)
        self.__create_text_encoder_1_frame(col0_layout, supports_include=True)
        self.__create_text_encoder_2_frame(col0_layout, supports_include=True)
        self.__create_embedding_frame(col0_layout)

        self.__create_base2_frame(col1_layout, video_training_enabled=True)
        self.__create_transformer_frame(col1_layout, supports_guidance_scale=True)
        self.__create_noise_frame(col1_layout)

        self.__create_masked_frame(col2_layout)
        self.__create_loss_frame(col2_layout)

    # -----------------------------------------------------------------------
    # The sub-frame creation methods
    # -----------------------------------------------------------------------
    def __create_base_frame(self, layout):
        """
        For "optimizer", "learning rate scheduler", "learning rate", etc.
        We'll replicate them in a QGridLayout inside a QFrame.
        """
        frame = QFrame()
        frame_layout = QGridLayout(frame)
        row = 0

        # 1) "Optimizer"
        lbl_opt = QLabel("Optimizer:")
        lbl_opt.setToolTip("The type of optimizer.")
        frame_layout.addWidget(lbl_opt, row, 0)
        combo_opt = QComboBox()
        for opt_val in Optimizer:
            combo_opt.addItem(str(opt_val), opt_val)
        # connect or bind to UIState if you like
        frame_layout.addWidget(combo_opt, row, 1)
        # We replicate your "options_adv" approach with an "advanced" button
        btn_opt_adv = QPushButton("...")
        btn_opt_adv.clicked.connect(self.__open_optimizer_params_window)
        frame_layout.addWidget(btn_opt_adv, row, 2)
        row += 1

        # 2) "Learning Rate Scheduler"
        lbl_lr_sched = QLabel("Learning Rate Scheduler")
        lbl_lr_sched.setToolTip("Scheduler that changes the learning rate during training.")
        frame_layout.addWidget(lbl_lr_sched, row, 0)

        combo_lr_sched = QComboBox()
        for sched_val in LearningRateScheduler:
            combo_lr_sched.addItem(str(sched_val), sched_val)
        frame_layout.addWidget(combo_lr_sched, row, 1)

        btn_lr_sched_adv = QPushButton("...")
        btn_lr_sched_adv.clicked.connect(self.__open_scheduler_params_window)
        frame_layout.addWidget(btn_lr_sched_adv, row, 2)
        self.lr_scheduler_comp = combo_lr_sched
        self.lr_scheduler_adv_comp = btn_lr_sched_adv
        row += 1

        # 3) "Learning Rate"
        lbl_lr = QLabel("Learning Rate")
        frame_layout.addWidget(lbl_lr, row, 0)
        line_lr = QLineEdit()
        frame_layout.addWidget(line_lr, row, 1)
        row += 1

        # etc. (rest of your code for "learning_rate_warmup_steps", "epochs", etc.)
        # row increments each time

        layout.addWidget(frame)

    def __create_base2_frame(self, layout, video_training_enabled=False):
        frame = QFrame()
        frame_layout = QGridLayout(frame)
        row = 0

        # Example: "Attention mechanism"
        lbl_attn = QLabel("Attention")
        lbl_attn.setToolTip("The attention mechanism used during training.")
        frame_layout.addWidget(lbl_attn, row, 0)
        combo_attn = QComboBox()
        for m in AttentionMechanism:
            combo_attn.addItem(str(m), m)
        frame_layout.addWidget(combo_attn, row, 1)
        row += 1

        # etc. add more lines for "EMA", "EMA Decay", "Train Dtype", etc.

        if video_training_enabled:
            # for "frames"
            lbl_frames = QLabel("Frames")
            lbl_frames.setToolTip("Number of frames used for training.")
            frame_layout.addWidget(lbl_frames, row, 0)
            line_frames = QLineEdit()
            frame_layout.addWidget(line_frames, row, 1)
            row += 1

        layout.addWidget(frame)

    def __create_text_encoder_frame(self, layout):
        frame = QFrame()
        fl = QGridLayout(frame)
        row = 0

        # "Train Text Encoder"
        lbl_tte = QLabel("Train Text Encoder")
        fl.addWidget(lbl_tte, row, 0)
        sw_tte = QCheckBox()
        fl.addWidget(sw_tte, row, 1)
        row += 1

        # etc.
        layout.addWidget(frame)

    def __create_text_encoder_1_frame(self, layout, supports_include=False):
        frame = QFrame()
        fl = QGridLayout(frame)
        row = 0

        if supports_include:
            lbl_inc = QLabel("Include Text Encoder 1")
            sw_inc = QCheckBox()
            fl.addWidget(lbl_inc, row, 0)
            fl.addWidget(sw_inc, row, 1)
            row += 1

        # train text encoder 1, etc.
        layout.addWidget(frame)

    def __create_text_encoder_2_frame(self, layout, supports_include=False):
        frame = QFrame()
        layout.addWidget(frame)

    def __create_text_encoder_3_frame(self, layout, supports_include=False):
        frame = QFrame()
        layout.addWidget(frame)

    def __create_embedding_frame(self, layout):
        frame = QFrame()
        layout.addWidget(frame)

    def __create_unet_frame(self, layout):
        frame = QFrame()
        layout.addWidget(frame)

    def __create_prior_frame(self, layout):
        frame = QFrame()
        layout.addWidget(frame)

    def __create_transformer_frame(self, layout, supports_guidance_scale=False):
        frame = QFrame()
        layout.addWidget(frame)

    def __create_noise_frame(self, layout):
        frame = QFrame()
        layout.addWidget(frame)

    def __create_align_prop_frame(self, layout):
        frame = QFrame()
        layout.addWidget(frame)

    def __create_masked_frame(self, layout):
        frame = QFrame()
        layout.addWidget(frame)

    def __create_loss_frame(self, layout, supports_vb_loss=False):
        frame = QFrame()
        layout.addWidget(frame)

    # -----------------------------------------------------------------------
    # Called when user clicks advanced "..." buttons or when we need to open subwindows
    # -----------------------------------------------------------------------
    def __open_optimizer_params_window(self):
        # If you have a PySide6 class called OptimizerParamsWindow
        # window = OptimizerParamsWindow(self, self.train_config, self.ui_state)
        # window.exec()
        pass

    def __open_scheduler_params_window(self):
        # window = SchedulerParamsWindow(self, self.train_config, self.ui_state)
        # window.exec()
        pass

    def __open_timestep_distribution_window(self):
        # window = TimestepDistributionWindow(self, self.train_config, self.ui_state)
        # window.exec()
        pass

    def __open_offloading_window(self):
        # window = OffloadingWindow(self, self.train_config, self.ui_state)
        # window.exec()
        pass

    # -----------------------------------------------------------------------
    # Restoration logic
    # -----------------------------------------------------------------------
    def __restore_optimizer_config(self, *args):
        optimizer_config = change_optimizer(self.train_config)
        # self.ui_state.get_var("optimizer").update(optimizer_config)

    def __restore_scheduler_config(self, variable):
        if not self.lr_scheduler_adv_comp:
            return

        if variable == "CUSTOM":
            self.lr_scheduler_adv_comp.setEnabled(True)
        else:
            self.lr_scheduler_adv_comp.setEnabled(False)

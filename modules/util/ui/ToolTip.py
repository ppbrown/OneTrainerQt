# tool_tip.py

"""
Thoughts from ppbrown:

This was autoconverted by ChatGPT, but it is obsolete.
Use components.add_tooltip() instead.
This is a custom tooltip implementation, which is not needed in PySide6.
"""

import sys
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QDialog, QGridLayout, QApplication
)
from PySide6.QtCore import (
    QObject, QTimer, QPoint, Qt, QEvent
)


class ToolTip(QObject):
    """
    A PySide6 replacement for the customtkinter-based ToolTip.
    Binds to a widget, shows a small popup QDialog with text on hover
    after a delay. Dismisses on leave or button press.
    """

    def __init__(
        self,
        widget: QWidget,
        text: str = "widget info",
        x_position: int = 20,
        wide: bool = False
    ):
        super().__init__(widget)
        self.widget = widget
        self.text = text
        self.x_position = x_position
        self.waittime = 500  # ms
        self.wraplength = 350 if wide else 180
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._show_tip)

        self._tooltip_dialog: Optional[QDialog] = None

        widget.installEventFilter(self)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj == self.widget:
            if event.type() == QEvent.Enter:
                self._schedule()
            elif event.type() == QEvent.Leave:
                self._unschedule()
                self._hide_tip()
            elif event.type() == QEvent.MouseButtonPress:
                # also hide on button press
                self._unschedule()
                self._hide_tip()
        return super().eventFilter(obj, event)

    def _schedule(self):
        self._unschedule()
        self._timer.start(self.waittime)

    def _unschedule(self):
        if self._timer.isActive():
            self._timer.stop()

    def _show_tip(self):
        if self._tooltip_dialog is not None:
            return  # already shown

        # Position near the widget
        global_pos = self.widget.mapToGlobal(QPoint(25, self.x_position))

        self._tooltip_dialog = QDialog(self.widget, flags=(
            Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        ))
        self._tooltip_dialog.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._tooltip_dialog.setObjectName("tooltipDialog")
        self._tooltip_dialog.setStyleSheet("""
        QDialog#tooltipDialog {
            background-color: #FFFFE0;
            border: 1px solid #A0A0A0;
        }
        """)
        layout = QVBoxLayout(self._tooltip_dialog)
        layout.setContentsMargins(8, 8, 8, 8)
        label = QLabel(self.text, self._tooltip_dialog)
        label.setWordWrap(True)
        label.setMaximumWidth(self.wraplength)
        layout.addWidget(label)
        self._tooltip_dialog.adjustSize()

        self._tooltip_dialog.move(global_pos)
        self._tooltip_dialog.show()

    def _hide_tip(self):
        if self._tooltip_dialog:
            self._tooltip_dialog.close()
            self._tooltip_dialog = None

# bind_mousewheel.py


"""
Usage:
def on_wheel(delta: int, event: QWheelEvent):
    print("Scrolled:", "up" if delta > 0 else "down")

bind_mousewheel(myWidget, {myWidget}, on_wheel)
"""

import sys
from collections.abc import Callable
from typing import Any, Optional

from PySide6.QtCore import QObject, QEvent
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QWidget

def bind_mousewheel(
    widget: QWidget,
    whitelist: Optional[set[QWidget]],
    callback: Callable[[int, QWheelEvent], None]
):
    """
    PySide6 equivalent of your Tkinter-based bind_mousewheel.
    Installs an event filter on 'widget' to capture wheel events,
    calls 'callback(delta, event)' with delta=+1 or -1.
    If 'whitelist' is not None, only calls callback if the event
    is from a widget in that set.
    """

    class MouseWheelFilter(QObject):
        def eventFilter(self, obj: QObject, event: QEvent) -> bool:
            if event.type() == QEvent.Type.Wheel and isinstance(event, QWheelEvent):
                if whitelist is not None and obj not in whitelist:
                    return False
                # angleDelta().y() is positive => scroll up, negative => scroll down
                delta_y = event.angleDelta().y()
                sign = 1 if delta_y > 0 else -1
                callback(sign, event)
                return True
            return super().eventFilter(obj, event)

    mw_filter = MouseWheelFilter(widget)
    widget.installEventFilter(mw_filter)

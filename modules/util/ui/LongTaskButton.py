
"""
A button class whose function is to spawn a potentially long-running task in the background.
It will spawn off a seperate thread, so that the UI can get updated normally while the task is running.
If the button is pressed during a run, it will signal the task to stop early.
"""
import threading
from PySide6.QtWidgets import QPushButton
from PySide6.QtCore import QThread, Qt


class LongTaskButton(QPushButton):
    def __init__(self, text_normal, text_running, callback, parent=None):
        """
        :param text_normal: The initial text of the button.
        :param text_running: The text displayed while the task is running.
        :param callback: The function to run in a separate thread.
                         It must accept a threading.Event as its single argument.
        """
        super().__init__(text_normal, parent)
        self._text_normal = text_normal
        self._text_running = text_running
        self._callback = callback
        self._thread = None
        self.stop_event = threading.Event()

        self.clicked.connect(self._handle_click)

    def _handle_click(self):
        if self._thread is None or not self._thread.isRunning():
            self.setCursor(Qt.WaitCursor)

            self.stop_event.clear()
 
            self._thread = TaskThread(self._callback, self.stop_event)
            self._thread.finished.connect(self._task_finished)
            self.setText(self._text_running)
            self._thread.start()
        else:
            if self.stop_event:
                self.stop_event.set()

    def _task_finished(self):
        self.unsetCursor()
        self.setText(self._text_normal)
        self._thread = None
        self.stop_event.clear()

    def stop_task(self):
        self.stop_event.set()
        if self._thread and self._thread.isRunning():
            self._thread.wait()

    # If the Button's window is deleted while the thread is still running, ensure it stops gracefully.
    def __del__(self):
        self.stop_task()

# QThread subclass that runs the callback function for LongTaskButton.
class TaskThread(QThread):
    def __init__(self, callback, stop_event, parent=None):
        super().__init__(parent)
        self.callback = callback
        self.stop_event = stop_event

    def run(self):
        # Run the callback, passing the stop event so it can check for early termination
        self.callback(self.stop_event)
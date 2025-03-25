from util.import_util import script_imports

script_imports()

from modules.ui.TrainUI import TrainUI

import sys
from PySide6.QtWidgets import QApplication

from modules.util.ui.ui_utils import set_window_icon

def main():
    app = QApplication(sys.argv)
    
    window = TrainUI()
    window.setWindowTitle("OneTrainer")
    #window.setIcon(QIcon("resources/icons/icon.png"))
    set_window_icon(window)
    window.showNormal()

    sys.exit(app.exec())
    

if __name__ == '__main__':
    main()

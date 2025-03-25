from util.import_util import script_imports

script_imports()

from modules.ui.TrainUI import TrainUI

import sys
from PySide6.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    
    window = TrainUI()
    window.setWindowTitle("OneTrainer")
    window.setIcon(QIcon("resources/icons/icon.png"))
    window.showNormal()

    sys.exit(app.exec())
    

if __name__ == '__main__':
    main()

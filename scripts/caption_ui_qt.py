from util.import_util import script_imports

script_imports()

from modules.ui.CaptionUI import CaptionUI
from modules.util.args.CaptionUIArgs import CaptionUIArgs

import sys
from PySide6.QtWidgets import QApplication
##from captionui import CaptionUI  # Ensure captionui.py uses PySide6 and has no top-level widget creation

def main():
    app = QApplication(sys.argv)

    window = CaptionUI()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()

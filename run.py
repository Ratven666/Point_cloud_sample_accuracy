import warnings

from PyQt6.QtWidgets import QApplication

from app.interface import UiPointCloudAnalizer

if __name__ == "__main__":
    import sys

    warnings.simplefilter(action="ignore", category=FutureWarning)
    app = QApplication(sys.argv)
    ui = UiPointCloudAnalizer()
    ui.show()
    sys.exit(app.exec())

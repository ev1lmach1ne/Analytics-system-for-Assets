import sys, os, ctypes
sys.path.insert(0, os.path.dirname(__file__))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from gui.main_window import MainWindow

def main():
    app_id = "AnalyticsSystemForAssets.App.1"
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName("Analytics System for Assets")
    app.setStyle("Fusion")

    icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
    app_icon = QIcon(icon_path)
    app.setWindowIcon(app_icon)

    window = MainWindow()
    window.setWindowIcon(app_icon)
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()

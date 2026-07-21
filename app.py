import sys, os, ctypes
sys.path.insert(0, os.path.dirname(__file__))

from core.config import APP_CONFIG_PATH

def _ensure_config():
    """Si no existe config.json, muestra el diálogo de primera apertura."""
    if os.path.exists(APP_CONFIG_PATH):
        return True
    from PyQt6.QtWidgets import QApplication
    from gui.dialogs.first_launch_dialog import FirstLaunchDialog
    app = QApplication.instance() or QApplication(sys.argv)
    dlg = FirstLaunchDialog()
    return dlg.exec() == dlg.DialogCode.Accepted

def main():
    app_id = "AnalyticsSystemForAssets.App.1"
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QIcon
    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Analytics System for Assets")
    app.setStyle("Fusion")

    icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
    app_icon = QIcon(icon_path)
    app.setWindowIcon(app_icon)

    if not _ensure_config():
        sys.exit(0)

    window = MainWindow()
    window.setWindowIcon(app_icon)
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()

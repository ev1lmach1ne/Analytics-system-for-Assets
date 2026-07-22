import sys, os, ctypes
sys.path.insert(0, os.path.dirname(__file__))

# Verificación temprana de dependencias: si falta alguna librería crítica
# (p. ej. porque la app se abrió con el Python del sistema en lugar del venv
# que prepara instalar.bat), avisar en claro antes de que cualquier import
# falle con un traceback. Debe ir antes del primer import de core/gui.
_CRITICAS = {
    'PyQt6': 'PyQt6', 'dotenv': 'python-dotenv', 'pandas': 'pandas',
    'numpy': 'numpy', 'numba': 'numba', 'scipy': 'scipy',
    'matplotlib': 'matplotlib', 'requests': 'requests',
    'ccxt': 'ccxt', 'yfinance': 'yfinance',
}


def _dependencias_faltantes():
    import importlib.util
    return [pip for mod, pip in _CRITICAS.items()
            if importlib.util.find_spec(mod) is None]


_faltan = _dependencias_faltantes()
if _faltan:
    _msg = ("Faltan librerías necesarias para la aplicación:\n\n  - "
            + "\n  - ".join(_faltan)
            + "\n\nEjecuta instalar.bat (está en la carpeta del programa) y "
              "abre la app con el acceso directo del escritorio o con "
              "launcher.vbs.")
    try:
        ctypes.windll.user32.MessageBoxW(
            None, _msg, "Analytics System — faltan dependencias", 0x10)
    except Exception:
        print(_msg)
    sys.exit(1)

# QtWebEngineWidgets (vista "Moderna" de la pestaña Resultados, gui/widgets/
# lwc_chart.py) EXIGE importarse antes de crear cualquier QApplication — si no,
# PyQt6 lanza ImportError. El orden de imports de main_window ya lo cumple hoy
# por casualidad; se fija aquí explícitamente para no depender de ese orden.
try:
    import PyQt6.QtWebEngineWidgets  # noqa: F401
except ImportError:
    pass   # PyQt6-WebEngine no instalado -> lwc_chart.WEBENGINE_OK queda False

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

    app = QApplication(sys.argv)
    app.setApplicationName("Analytics System for Assets")
    app.setStyle("Fusion")

    icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
    app_icon = QIcon(icon_path)
    app.setWindowIcon(app_icon)

    if not _ensure_config():
        sys.exit(0)

    # Importar DESPUÉS de _ensure_config(): los módulos de las pestañas copian
    # LIMPIADOS_DIR al importarse, y en el primer arranque BASE_DATA no queda
    # fijado hasta que el usuario elige carpeta en el FirstLaunchDialog.
    from gui.main_window import MainWindow

    window = MainWindow()
    window.setWindowIcon(app_icon)
    window.show()

    from core.config import get_tutorial_visto, set_tutorial_visto
    if not get_tutorial_visto():
        from gui.dialogs.tutorial_dialog import TutorialDialog
        TutorialDialog(window).exec()
        set_tutorial_visto()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()

import sys, os, ctypes, time, threading
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

def _modo_script():
    """Rama «intérprete» del ejecutable empaquetado.

    Con PyInstaller --onefile no hay ningún python.exe al que llamar para
    los subprocesos de descarga/importación/análisis: sys.executable es el
    propio .exe. La GUI lo relanza con «--run-script <ruta>» (ver
    _comando_script en gui/widgets/console_widget.py) y aquí se ejecuta ese
    script y se sale, sin llegar a abrir ninguna ventana.

    Devuelve True si se consumió como script (el llamador debe salir)."""
    if len(sys.argv) < 3 or sys.argv[1] != '--run-script':
        return False
    script = sys.argv[2]

    # Con --windowed, PyInstaller puede dejar sys.stdout/stderr a None. El
    # padre (QProcess) sí nos pasa tuberías válidas en los descriptores 1 y
    # 2, así que se reabren a mano; sin esto la consola de la GUI se
    # quedaría vacía. line_buffering para que el progreso llegue en vivo y
    # no de golpe al terminar.
    #
    # La codificación hay que forzarla AQUÍ, en el propio proceso: el
    # bootloader de PyInstaller arranca Python con una configuración
    # «isolated» (use_environment = 0), así que el PYTHONIOENCODING que la
    # GUI mete en el entorno del hijo NO tiene ningún efecto en el .exe
    # congelado. Sin esto la salida se emite en la codepage ANSI (cp1252) y
    # el padre, que la decodifica como UTF-8, muestra las tablas con «?» en
    # los caracteres de caja y «�» en las acentuadas.
    for nombre, fd in (('stdout', 1), ('stderr', 2)):
        flujo = getattr(sys, nombre, None)
        if flujo is None:
            try:
                setattr(sys, nombre, os.fdopen(fd, 'w', buffering=1,
                                               encoding='utf-8', errors='replace'))
            except OSError:
                setattr(sys, nombre, open(os.devnull, 'w'))
        else:
            try:
                flujo.reconfigure(encoding='utf-8', errors='replace',
                                  line_buffering=True)
            except Exception:
                # Flujo sin reconfigure() (p. ej. un envoltorio nulo de
                # PyInstaller): se reenvuelve el búfer binario si lo tiene.
                try:
                    import io
                    setattr(sys, nombre, io.TextIOWrapper(
                        flujo.buffer, encoding='utf-8', errors='replace',
                        line_buffering=True))
                except Exception:
                    pass

    # La ruta llega calculada contra el _MEIPASS del proceso PADRE; este
    # proceso tiene el suyo propio, así que si no existe se resuelve por
    # nombre dentro del paquete de este.
    if not os.path.exists(script):
        from core.config import SCRIPTS_DIR
        alternativa = os.path.join(SCRIPTS_DIR, os.path.basename(script))
        if os.path.exists(alternativa):
            script = alternativa

    # runpy.run_path NO añade la carpeta del script a sys.path (a diferencia
    # de «python script.py»), así que los scripts que importan a un hermano
    # —p.ej. categorias_comun— no lo encontrarían.
    carpeta = os.path.dirname(os.path.abspath(script))
    if carpeta not in sys.path:
        sys.path.insert(0, carpeta)

    sys.argv = [script] + sys.argv[3:]
    import runpy
    runpy.run_path(script, run_name='__main__')
    return True


def main():
    if _modo_script():
        return

    app_id = "AnalyticsSystemForAssets.App.1"
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass

    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QSplashScreen
    from PyQt6.QtGui import QIcon, QPalette, QColor, QPixmap

    # QtWebEngineWidgets debe importarse antes de crear QApplication
    try:
        import PyQt6.QtWebEngineWidgets  # noqa: F401
    except ImportError:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName("Analytics System for Assets")
    app.setStyle("Fusion")

    # Fusion no define paleta propia: sin esto, el color de selección de
    # texto/tablas/listas sin :selected explícito hereda el acento del
    # sistema (en Windows puede salir rosa). Se fija al mismo azul cielo
    # (#2a4a6a fondo / #4fc3f7 texto) ya usado en el resto de la app.
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#2a4a6a"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#4fc3f7"))
    app.setPalette(palette)

    icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
    app_icon = QIcon(icon_path)
    app.setWindowIcon(app_icon)

    # Si falta la configuración inicial, mostrar el diálogo sin splash para
    # que no quede una ventana encima del diálogo de primera apertura.
    if not _ensure_config():
        sys.exit(0)

    # Splash screen con fade-in mientras carga la app
    # Usamos pixmap(256,256) para obtener la mejor resolución del .ico
    splash_px = app_icon.pixmap(256, 256).scaled(300, 300,
        Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    splash = QSplashScreen(splash_px)
    splash.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
    splash.setWindowOpacity(0.0)
    splash.show()
    for i in range(1, 11):
        splash.setWindowOpacity(i / 10.0)
        app.processEvents()
        time.sleep(0.03)

    # Importar DESPUÉS de _ensure_config(): los módulos de las pestañas copian
    # LIMPIADOS_DIR al importarse, y en el primer arranque BASE_DATA no queda
    # fijado hasta que el usuario elige carpeta en el FirstLaunchDialog.
    #
    # Este import tarda varios segundos (scipy/pandas/matplotlib) — si se
    # hiciera de forma síncrona, la app dejaría de procesar eventos durante
    # todo ese tiempo y Windows la marcaría "sin responder", dibujando
    # ventanas fantasma detrás del splash. Se hace en un hilo aparte
    # (solo importa módulos Python, no crea widgets) mientras el hilo
    # principal sigue bombeando el bucle de eventos.
    _resultado_import = {}

    def _importar_main_window():
        from gui.main_window import MainWindow
        _resultado_import['MainWindow'] = MainWindow
        # Calentar matplotlib fuera del hilo principal: la primera Figure
        # inicializa el gestor de fuentes y el backend (1-3 s en equipos
        # lentos). Hacerlo aquí evita que esa inicialización congele la UI la
        # primera vez que se construye el Backtester.
        try:
            from matplotlib.figure import Figure
            _fig = Figure(figsize=(1, 1))
            _fig.add_subplot(111)
        except Exception:
            pass

    hilo_import = threading.Thread(target=_importar_main_window, daemon=True)
    hilo_import.start()
    while hilo_import.is_alive():
        app.processEvents()
        time.sleep(0.03)
    hilo_import.join()

    window = _resultado_import['MainWindow']()
    window.setWindowIcon(app_icon)

    # Fade out del splash — luego mostramos la ventana
    for i in range(10, -1, -1):
        splash.setWindowOpacity(i / 10.0)
        app.processEvents()
        time.sleep(0.04)
    splash.close()

    window.show()

    # La ventana aparece ya interactiva; las pestañas aún pendientes se
    # precargan bajo el overlay de carga (spinner + texto). Si tarda más del
    # tope, la ventana queda despejada y el resto se carga pestaña a pestaña
    # en su primera visita.
    window.precargar_tabs()

    from core.config import get_tutorial_visto, set_tutorial_visto
    if not get_tutorial_visto():
        from gui.dialogs.tutorial_dialog import TutorialDialog
        TutorialDialog(window).exec()
        set_tutorial_visto()

    rc = app.exec()

    # La ventana ya se cerró (closeEvent → _apagar_hilos esperó a los hilos y
    # los destruyó), pero quedan eventos encolados y animaciones de fade de los
    # overlays. Si el intérprete destruye el árbol de Qt con eso pendiente,
    # aborta (0xC0000409). Se asienta el cierre aquí, con el árbol aún vivo.
    try:
        for _ in range(120):
            app.processEvents()
            time.sleep(0.01)
    except Exception:
        pass

    sys.exit(rc)

if __name__ == "__main__":
    main()

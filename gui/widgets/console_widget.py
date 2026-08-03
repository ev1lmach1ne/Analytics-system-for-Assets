import os
import re
import sys
from PyQt6.QtWidgets import QTextEdit, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QProcess, QProcessEnvironment, pyqtSignal, QByteArray
from PyQt6.QtGui import QTextCursor, QTextCharFormat, QColor, QFont


def _python_exe():
    """Intérprete para los subprocesos: el mismo con el que corre la app
    (venv incluido), nunca el 'python' que haya en el PATH del sistema —
    ese puede ser otro Python sin las dependencias instaladas."""
    exe = sys.executable
    if os.path.basename(exe).lower() == 'pythonw.exe':
        candidato = os.path.join(os.path.dirname(exe), 'python.exe')
        if os.path.exists(candidato):
            exe = candidato
    return exe


def _entorno_hijo(env):
    """Entorno del subproceso: el del sistema MÁS las claves de 'env'.

    Debe partir de systemEnvironment() y no de processEnvironment(): en un
    QProcess recién creado esta última devuelve un entorno VACÍO, así que
    insertar ahí las claves dejaba al hijo sin TEMP/TMP/SystemRoot. Con la
    app empaquetada eso reventaba la descarga, porque el hijo es el propio
    .exe y su bootloader necesita TEMP para descomprimirse ("Could not
    create temporary directory")."""
    qenv = QProcessEnvironment.systemEnvironment()
    # _on_stdout decodifica siempre como UTF-8, así que el hijo tiene que
    # emitir UTF-8 sí o sí. Por defecto Python usa la codepage ANSI (cp1252
    # en Windows), que no tiene los caracteres de caja de las tablas ni
    # coincide byte a byte con UTF-8 en las acentuadas. Se fija aquí, para
    # todos los scripts, y no en cada llamada suelta. Ojo: en el .exe
    # congelado estas variables NO bastan (PyInstaller arranca Python
    # ignorando el entorno) — ese caso lo cubre _modo_script() en app.py.
    qenv.insert('PYTHONIOENCODING', 'utf-8')
    qenv.insert('PYTHONUTF8', '1')
    for k, v in env.items():
        qenv.insert(k, str(v))
    return qenv


def _comando_script(script_path, extra_args=()):
    """Programa y argumentos para lanzar un script hijo.

    Empaquetado con PyInstaller --onefile no existe ningún intérprete de
    Python al que llamar: sys.executable ES el propio .exe. Se le relanza
    con el centinela --run-script, que app.py intercepta para ejecutar el
    script en vez de abrir una segunda ventana de la aplicación."""
    if getattr(sys, 'frozen', False):
        return sys.executable, ['--run-script', script_path, *extra_args]
    return _python_exe(), ['-u', script_path, *extra_args]

STYLE_CONSOLE = """
QTextEdit {
    background-color: #0d1424; color: #aabbcc;
    border: 1px solid #253a60; border-radius: 4px;
    padding: 8px; font-family: Consolas, monospace; font-size: 11px;
}
"""

ANSI_COLORS = {
    90: QColor('#7f8c8d'),
    91: QColor('#e74c3c'),
    92: QColor('#27ae60'),
    93: QColor('#f1c40f'),
    94: QColor('#3498db'),
    95: QColor('#9b59b6'),
    96: QColor('#1abc9c'),
    97: QColor('#c8d6e5'),
}

_ANSI_RE = re.compile(r'\033\[([\d;]+)m')

class ConsoleWidget(QWidget):
    finished = pyqtSignal(int)
    progress = pyqtSignal(int)

    _PROGRESS_RE = re.compile(r'\[(\d+)/(\d+)\]')

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        header = QLabel("Consola")
        header.setStyleSheet("color: #3a5a7a; font-size: 11px; font-weight: bold;")
        layout.addWidget(header)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet(STYLE_CONSOLE)
        self.output.document().setMaximumBlockCount(5000)
        self.output.document().setDefaultStyleSheet(
            "* { line-height: 1.1; }"
        )
        font = QFont("Consolas", 11)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.output.setFont(font)
        layout.addWidget(self.output)

        self._process = None
        self._buffer = ""
        self._current_fg = None

    def _default_fmt(self):
        fmt = QTextCharFormat()
        fmt.setFontFamily("Consolas")
        fmt.setFontFixedPitch(True)
        fmt.setFontPointSize(11)
        fmt.setForeground(self._current_fg or QColor('#aabbcc'))
        return fmt

    def _append_ansi(self, text):
        cursor = self.output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = self._default_fmt()
        parts = _ANSI_RE.split(text)
        for part in parts:
            if not part:
                continue
            if _ANSI_RE.fullmatch(f'\033[{part}m'):
                codes = [int(c) for c in part.split(';')]
                for c in codes:
                    if c == 0:
                        self._current_fg = None
                        fmt.setForeground(QColor('#aabbcc'))
                    elif c in ANSI_COLORS:
                        self._current_fg = ANSI_COLORS[c]
                        fmt.setForeground(ANSI_COLORS[c])
            else:
                cursor.insertText(part, fmt)
        self.output.setTextCursor(cursor)
        sb = self.output.verticalScrollBar()
        if sb:
            sb.setValue(sb.maximum())

    def run(self, script_path, env=None, clear=True):
        if clear:
            self.output.clear()
            self._buffer = ""
        self._current_fg = None
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.setProcessEnvironment(_entorno_hijo(env or {}))
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.finished.connect(self._on_finished)
        programa, args = _comando_script(script_path)
        self._process.start(programa, args)

    def run_with_args(self, script_path, args, env=None):
        self.output.clear()
        self._buffer = ""
        self._current_fg = None
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.setProcessEnvironment(_entorno_hijo(env or {}))
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.finished.connect(self._on_finished)
        programa, argumentos = _comando_script(script_path, args)
        self._process.start(programa, argumentos)

    def _on_stdout(self):
        data = self._process.readAllStandardOutput()
        text = QByteArray(data).data().decode('utf-8', errors='replace')
        self._buffer += text
        self._append_ansi(text)
        m = self._PROGRESS_RE.search(text)
        if m:
            current, total = int(m.group(1)), int(m.group(2))
            self.progress.emit(int(current / total * 100))

    def _on_finished(self, exit_code, exit_status):
        self.finished.emit(exit_code)

    def stop(self):
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()

    @property
    def full_output(self):
        return self._buffer

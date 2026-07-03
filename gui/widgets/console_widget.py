import re
from PyQt6.QtWidgets import QTextEdit, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QProcess, pyqtSignal, QByteArray
from PyQt6.QtGui import QTextCursor, QTextCharFormat, QColor, QFont

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
        if env:
            qenv = self._process.processEnvironment()
            for k, v in env.items():
                qenv.insert(k, str(v))
            self._process.setProcessEnvironment(qenv)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.finished.connect(self._on_finished)
        self._process.start("python", ["-u", script_path])

    def run_with_args(self, script_path, args, env=None):
        self.output.clear()
        self._buffer = ""
        self._current_fg = None
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        if env:
            qenv = self._process.processEnvironment()
            for k, v in env.items():
                qenv.insert(k, str(v))
            self._process.setProcessEnvironment(qenv)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.finished.connect(self._on_finished)
        self._process.start("python", ["-u", script_path] + args)

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

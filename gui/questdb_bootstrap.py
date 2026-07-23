"""
gui/questdb_bootstrap.py
Envoltorio Qt de core/questdb_manager: hilo con progreso + diálogo modal
ligero que se muestra SOLO si hace falta preparar QuestDB — si ya estaba
disponible, ensure_running() es instantáneo y no se muestra nada.
"""
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QMessageBox

from core.questdb_manager import ensure_running, is_reachable

# cuánto se deja visible el mensaje final de éxito antes de cerrar solo —
# para que no parezca que el diálogo "desaparece de golpe" sin confirmar
_MS_CONFIRMACION_FINAL = 600


class _QuestDBBootstrapThread(QThread):
    progreso = pyqtSignal(str, int)   # mensaje, porcentaje (-1 = indeterminado)
    terminado = pyqtSignal(bool, str)

    def _emitir_progreso(self, mensaje, porcentaje):
        self.progreso.emit(mensaje, -1 if porcentaje is None else porcentaje)

    def run(self):
        try:
            ok, mensaje = ensure_running(progress_cb=self._emitir_progreso)
        except Exception as e:
            ok, mensaje = False, f"Error inesperado preparando QuestDB: {e}"
        self.terminado.emit(ok, mensaje)


class _DialogoBootstrap(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preparando QuestDB")
        self.setModal(True)
        self.setFixedWidth(420)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        lay = QVBoxLayout(self)
        self.lbl = QLabel(
            "Preparando la base de datos local (solo la primera vez, puede "
            "tardar según tu conexión)…")
        self.lbl.setWordWrap(True)
        lay.addWidget(self.lbl)
        self.barra = QProgressBar()
        self.barra.setRange(0, 0)   # indeterminada hasta que haya un % real
        lay.addWidget(self.barra)

    def set_progreso(self, texto, porcentaje):
        self.lbl.setText(texto)
        if porcentaje is not None and porcentaje >= 0:
            self.barra.setRange(0, 100)
            self.barra.setValue(porcentaje)
        else:
            self.barra.setRange(0, 0)   # fase sin % conocido (extraer, arrancar)

    def set_completado(self):
        self.lbl.setText("QuestDB lista ✓")
        self.barra.setRange(0, 100)
        self.barra.setValue(100)


def mostrar_bootstrap_questdb(parent=None) -> bool:
    """Si QuestDB ya está disponible, no muestra nada y devuelve True al
    instante — caso normal. Si no, descarga/arranca con una barra de
    progreso modal (con % real durante la descarga) y bloquea hasta
    terminar. Devuelve si quedó lista."""
    if is_reachable():
        return True

    resultado = {'ok': False, 'mensaje': ''}
    dlg = _DialogoBootstrap(parent)
    th = _QuestDBBootstrapThread(dlg)

    def _on_progreso(texto, porcentaje):
        dlg.set_progreso(texto, None if porcentaje < 0 else porcentaje)

    def _on_terminado(ok, mensaje):
        resultado['ok'] = ok
        resultado['mensaje'] = mensaje
        if ok:
            dlg.set_completado()
            QTimer.singleShot(_MS_CONFIRMACION_FINAL, dlg.accept)
        else:
            dlg.accept()

    th.progreso.connect(_on_progreso)
    th.terminado.connect(_on_terminado)
    th.start()
    dlg.exec()
    th.wait()

    if not resultado['ok']:
        QMessageBox.warning(
            parent, "QuestDB no disponible",
            f"{resultado['mensaje']}\n\n"
            "El paso «Limpiar» necesita QuestDB para funcionar.")
    return resultado['ok']

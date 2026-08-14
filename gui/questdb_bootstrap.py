"""
gui/questdb_bootstrap.py
Envoltorio Qt de core/questdb_manager: hilo con progreso + diálogo modal
ligero que se muestra SOLO si hace falta preparar QuestDB — si ya estaba
disponible, ensure_running() es instantáneo y no se muestra nada.
"""
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtWidgets import (QDialog, QLabel, QMessageBox, QWidget, QHBoxLayout,
                             QVBoxLayout)

from gui.widgets.plot_common import PanelFlotanteDialog, montar_panel_flotante
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


class _SpinnerCircular(QWidget):
    """Anillo de carga giratorio dibujado con QPainter (sin assets).

    Un círculo tenue de fondo y un arco brillante `#4fc3f7` que rota; se usa
    mientras la duración de la operación es desconocida o solo se conoce un
    porcentaje."""
    def __init__(self, parent=None, size=40, grosor=3, color='#4fc3f7'):
        super().__init__(parent)
        self._angulo = 0
        self._color = QColor(color)
        self.setFixedSize(size, size)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._avanzar)
        self._timer.start(20)   # ~50 fps: rotación suave y barata

    def _avanzar(self):
        self._angulo = (self._angulo + 6) % 360
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(4, 4, -4, -4)
        base = QPen(QColor(79, 195, 247, 60))
        base.setWidth(3)
        base.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(base)
        p.drawArc(r, 0, 360 * 16)
        brillo = QPen(self._color)
        brillo.setWidth(3)
        brillo.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(brillo)
        p.drawArc(r, -self._angulo * 16, 300 * 16)


class _DialogoBootstrap(PanelFlotanteDialog):
    """Diálogo sin marco de Windows con spinner circular y halo pulsante.

    Panel flotante con cabecera propia; el halo azul late mientras la
    descarga/arranque está en curso y se queda quieto al terminar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Preparando QuestDB")   # oculto (sin marco nativo)
        lay, self.lbl_sub, halo = montar_panel_flotante(
            self, 'Preparando QuestDB', ancho=420,
            subtitulo='Iniciando el motor de datos…', boton_cerrar=False)
        self._halo = halo
        fila = QHBoxLayout()
        fila.setSpacing(12)
        self.spinner = _SpinnerCircular(self)
        fila.addWidget(self.spinner)
        col = QVBoxLayout()
        col.setSpacing(4)
        self.lbl = QLabel(
            "Preparando la base de datos local (solo la primera vez, puede "
            "tardar según tu conexión)…")
        self.lbl.setWordWrap(True)
        self.lbl.setObjectName("textoPanel")
        col.addWidget(self.lbl)
        self.lbl_pct = QLabel("")
        self.lbl_pct.setObjectName("subPanel")
        col.addWidget(self.lbl_pct)
        fila.addLayout(col, 1)
        lay.addLayout(fila)
        self._glow_anim = None
        self._pulsar_halo()

    # ── halo pulsante mientras carga ──

    def _pulsar_halo(self):
        glow = self._halo.graphicsEffect()
        if glow is None:
            return
        anim = QPropertyAnimation(glow, b"blurRadius", self)
        anim.setStartValue(12)
        anim.setEndValue(28)
        anim.setDuration(900)
        anim.setLoopCount(-1)
        anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        anim.start()
        self._glow_anim = anim

    def _detener_pulso(self):
        anim = getattr(self, '_glow_anim', None)
        if anim is not None:
            anim.stop()
        glow = self._halo.graphicsEffect()
        if glow is not None:
            glow.setBlurRadius(18)

    # ── estado en vivo ──

    def set_progreso(self, texto, porcentaje):
        self.lbl.setText(texto)
        if porcentaje is not None and porcentaje >= 0:
            self.lbl_pct.setText(f"{porcentaje} %")
        else:
            self.lbl_pct.setText("")

    def set_completado(self):
        self._detener_pulso()
        self.lbl.setText("QuestDB lista ✓")
        self.lbl_pct.setText("")
        self.lbl.setStyleSheet(
            "color: #2ecc71; font-size: 11px; font-weight: bold;")


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

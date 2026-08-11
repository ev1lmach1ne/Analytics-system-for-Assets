"""Bombeo de eventos durante construcciones pesadas.

Al construir pestañas con cientos de widgets (p. ej. el Constructor del
Backtester) el hilo principal queda bloqueado; si el bloqueo dura varios
segundos, Windows marca la ventana como "No responde" y dibuja copias
fantasma grises encima. Este helper procesa timers y pintados (el spinner
del overlay sigue girando y la ventana sigue repintándose) SIN entregar
entrada de usuario, de modo que no hay reentrada de clics en mitad de la
construcción.
"""
from PyQt6.QtCore import QEventLoop
from PyQt6.QtWidgets import QApplication


def bombear_eventos():
    QApplication.processEvents(
        QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
        | QEventLoop.ProcessEventsFlag.ExcludeSocketNotifiers)

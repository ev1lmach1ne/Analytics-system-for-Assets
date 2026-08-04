"""La entrada por orden límite queda a la vista pero fuera de uso.

El motor la simula entera (ver test_strategias_fibonacci.py y
test_backtest_ordenes_limite.py), pero la opción no se abre todavía en la
interfaz. Lo que se fija aquí es la forma de bloquearla: item deshabilitado en
el modelo del combo, NO item retirado — un setup guardado con `limite_fib` tiene
que seguir mostrando su tipo real en vez de decir "A mercado" y cambiar de
estrategia sin avisar.
"""
import os

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('PyQt6.QtWidgets')
from PyQt6.QtCore import Qt                                  # noqa: E402
from PyQt6.QtWidgets import QApplication                     # noqa: E402

from gui.widgets.tab_backtest import (                       # noqa: E402
    OptimizadorWidget, _ETIQUETA_LIMITE_FIB, _MAPA_TIPO_ENTRADA,
)


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope='module')
def combo(app):
    return OptimizadorWidget().cmb_tipo_entrada


def test_la_etiqueta_avisa_de_que_esta_por_llegar(combo):
    assert '(próximamente)' in _ETIQUETA_LIMITE_FIB
    assert combo.findText(_ETIQUETA_LIMITE_FIB) >= 0, \
        "la opción sigue listada: oculta del todo no se entendería que llegará"


def test_la_opcion_esta_deshabilitada_y_apagada(combo):
    item = combo.model().item(combo.findText(_ETIQUETA_LIMITE_FIB))
    assert not item.isEnabled()
    assert item.data(Qt.ItemDataRole.ForegroundRole) is not None, \
        "sin color propio no se distingue de una opción usable"


def test_a_mercado_sigue_siendo_la_seleccion_de_partida(combo):
    assert combo.currentText() == 'A mercado'
    item_mercado = combo.model().item(combo.findText('A mercado'))
    assert item_mercado.isEnabled()


def test_un_setup_ya_guardado_conserva_su_tipo(combo):
    """setCurrentIndex sí puede posarse en un item deshabilitado; es lo que
    permite que cargar un sistema antiguo no le cambie la entrada."""
    combo.setCurrentText(_ETIQUETA_LIMITE_FIB)
    assert _MAPA_TIPO_ENTRADA[combo.currentText()] == 'limite_fib'
    combo.setCurrentText('A mercado')

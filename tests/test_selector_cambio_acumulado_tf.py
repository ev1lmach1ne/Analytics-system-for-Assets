"""Selector de horizonte del «Cambio acumulado medio».

En TF diario o mayor (>= 1d) el horizonte 'Día' no tiene sentido — una sola
vela por día deja la curva 00:00→24:00 vacía —, así que el botón se
deshabilita y la selección por defecto pasa a 'Semana'.
"""
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

pytest.importorskip('PyQt6.QtWidgets')
from PyQt6.QtWidgets import QApplication   # noqa: E402

from gui.widgets.analisis_graficos import _Seccion   # noqa: E402


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def _seccion(app):
    return _Seccion(
        "Cambio acumulado", ("lógica", "significado", "uso", "resultados"),
        lambda *a: None, selector_opciones=('Día', 'Semana', 'Mes', 'Año'))


def test_diario_deshabilita_dia_y_selecciona_semana(app):
    s = _seccion(app)
    s._sincronizar_selector_tf({'_meta': {'tf': '1d', 'activo': 'STOCK'}})
    assert not s._botones_opcion['Día'].isEnabled()
    assert not s._botones_opcion['Día'].isChecked()
    assert s.selector_valor == 'Semana'
    assert s._botones_opcion['Semana'].isChecked()
    assert 'TF diario' in s._botones_opcion['Día'].toolTip()


def test_diario_respeta_seleccion_manual(app):
    s = _seccion(app)
    s.selector_valor = 'Año'
    s._botones_opcion['Año'].setChecked(True)
    s._sincronizar_selector_tf({'_meta': {'tf': '1d'}})
    assert not s._botones_opcion['Día'].isEnabled()
    assert s.selector_valor == 'Año'
    assert s._botones_opcion['Año'].isChecked()


def test_intradia_mantiene_dia_habilitado(app):
    s = _seccion(app)
    s._sincronizar_selector_tf({'_meta': {'tf': '1h'}})
    assert s._botones_opcion['Día'].isEnabled()
    assert s.selector_valor == 'Día'
    assert s._botones_opcion['Día'].isChecked()


def test_sin_meta_deja_dia_habilitado(app):
    s = _seccion(app)
    s._sincronizar_selector_tf({})
    assert s._botones_opcion['Día'].isEnabled()
    assert s.selector_valor == 'Día'
    s._sincronizar_selector_tf(None)
    assert s._botones_opcion['Día'].isEnabled()
"""Periodo cubierto en el título de la pestaña Resultados.

Se prueba el helper puro y no el widget: construir un ResultadosWidget y
llamar a mostrar() arrastra el dibujado completo (equity, WFA, Montecarlo,
MFE/MAE) para verificar una cadena de texto.

La duración es aproximada por diseño (ver el docstring de _texto_periodo), así
que los tests afirman sobre la unidad elegida y el redondeo, no sobre décimas.
"""
import os

import pandas as pd
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('PyQt6.QtWidgets')

from gui.widgets.tab_backtest import _texto_periodo   # noqa: E402


def _rango(inicio, fin, freq='1h'):
    return pd.DatetimeIndex(pd.date_range(inicio, fin, freq=freq, tz='UTC'))


def test_varios_anios_muestra_fechas_y_anios():
    txt = _texto_periodo(_rango('2020-01-01', '2025-03-15', freq='1D'))
    assert '01/01/2020' in txt
    assert '15/03/2025' in txt
    assert '5 años' in txt


def test_anio_justo_no_arrastra_meses_a_cero():
    txt = _texto_periodo(_rango('2024-01-01', '2024-12-31', freq='1D'))
    assert '1 año' in txt
    assert 'mes' not in txt


def test_rango_de_meses_no_habla_de_anios():
    txt = _texto_periodo(_rango('2024-01-01', '2024-08-01', freq='1D'))
    assert 'meses' in txt
    assert 'año' not in txt


def test_rango_corto_en_dias():
    txt = _texto_periodo(_rango('2024-05-01', '2024-05-19'))
    assert 'días' in txt
    assert 'mes' not in txt


def test_sin_velas_devuelve_cadena_vacia():
    assert _texto_periodo(pd.DatetimeIndex([])) == ''

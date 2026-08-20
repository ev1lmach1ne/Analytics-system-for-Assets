"""El render de la tabla de métricas del Backtester aguanta las claves
nuevas (CVaR, Serenity, Sharpe smart, estructura de trades) y les aplica
los formatos especiales (%, €)."""
import os

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('PyQt6.QtWidgets')
from PyQt6.QtWidgets import QApplication, QTableWidget  # noqa: E402

from gui.widgets.tab_backtest import (  # noqa: E402
    render_tabla_metricas, _FILAS_METRICAS,
)
from core.backtest import calcular_metricas  # noqa: E402


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def _resultado():
    n = 40
    return {
        'equity': np.full(n, 10000.0),
        'trades': {
            'idx_entrada': np.array([1, 2, 3, 4, 5], dtype=np.int64),
            'idx_salida': np.array([6, 7, 8, 9, 10], dtype=np.int64),
            'dir': np.array([1, 1, -1, -1, 1], dtype=np.int8),
            'pnl': np.array([100.0, -50.0, 80.0, -30.0, 200.0]),
            'ret_pct': np.array([0.01, -0.005, 0.008, -0.003, 0.02]),
            'r_multiple': np.array([1.0, -0.5, 0.8, -0.3, 2.0]),
            'setup': np.zeros(5, dtype=np.int64),
            'notional_redondo': np.array([10000.0] * 5),
            'costo_comision': np.array([1.0] * 5),
            'mfe_r': np.array([1.0] * 5), 'mae_r': np.array([0.5] * 5),
            'etd_r': np.array([0.1] * 5),
            'eficiencia_entrada': np.array([0.5] * 5),
            'eficiencia_salida': np.array([0.5] * 5),
        },
        'drawdown': np.zeros(n), 'capital_final': 10000.0, 'n_trades': 5,
    }


def test_render_tabla_metricas_con_claves_nuevas(app):
    r = _resultado()
    met = {'IS': calcular_metricas(r, 0, 20, 365.0, 1440.0),
           'OOS': calcular_metricas(r, 20, 40, 365.0, 1440.0),
           'Total': calcular_metricas(r, 0, 40, 365.0, 1440.0)}
    tab = QTableWidget(len(_FILAS_METRICAS), 4)
    render_tabla_metricas(tab, met, ('IS', 'OOS', 'Total'))
    nombres = [tab.item(f, 0).text() for f in range(tab.rowCount())
               if tab.item(f, 0)]
    for esperado in ('CVaR drawdown', 'CVaR retorno', 'Serenity Index',
                     'Sharpe smart', 'Win rate largos', 'Win rate cortos',
                     'Racha ganadora', 'Mayor ganancia', 'Mayor pérdida',
                     'Ganancia bruta', 'Pérdida bruta', 'Trades por día'):
        assert any(esperado in t for t in nombres), esperado

    # formatos especiales
    def _celda(nombre, col):
        fila = next(i for i, t in enumerate(nombres) if t == nombre)
        return tab.item(fila, col)

    assert _celda('Win rate largos', 3).text().endswith('%')
    assert _celda('Mayor ganancia', 3).text().endswith('€')
    assert _celda('Racha ganadora máx.', 3).text() == '1'
    # CVaR de una equity plana es 0.00 %
    assert _celda('CVaR retorno (peor 5% de velas)', 3).text() == '0.00 %'

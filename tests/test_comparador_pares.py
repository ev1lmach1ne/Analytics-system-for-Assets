"""El panel de Pares del Comparador usa el benchmark elegido en el panel de
Dispersión (el MISMO combo) y lo muestra de forma visible: el activo elegido
sale con beta ≈ 1.00 (regresión contra sí mismo), el par sintético con su
beta real, y el label/título de la pestaña nombran el benchmark."""
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('PyQt6.QtWidgets')
from PyQt6.QtWidgets import QApplication  # noqa: E402

from gui.widgets.tab_comparador import TabComparador  # noqa: E402


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def _sinteticos(app):
    """3 activos: ethusd = 1.5·btcusd + ruido (par con beta real),
    solusd independiente. Se descarta la sesión restaurada del disco para
    que el test sea determinista."""
    w = TabComparador()
    w._assets.clear()
    w._orden.clear()
    w._tf = '1h'
    rng = np.random.default_rng(3)
    n = 400
    ts = pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC')
    base = np.cumsum(rng.normal(0.0001, 0.005, n))
    series = {
        'btcusd': pd.Series(100 * np.exp(base), index=ts),
        'ethusd': pd.Series(100 * np.exp(1.5 * base + rng.normal(0, 0.001, n)),
                            index=ts),
        'solusd': pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.005, n))),
                            index=ts),
    }
    for i, (nom, close) in enumerate(series.items()):
        ret = np.log(close / close.shift(1)).dropna()
        path = f'fake/{nom}'
        w._assets[path] = {
            'label': f'{nom} 1h', 'color': '#4fc3f7',
            'metrics': {'cagr': 0.1, 'vol': 0.2, 'max_dd': -0.1,
                        'sharpe': 1.0, 'calmar': 1.0},
            'base': close, 'tf_cache': {}, 'chip': None, 'error': None}
        w._assets[path]['tf_cache']['1h'] = {
            'close': close, 'ret': ret, 'cagr': 0.1, 'vol': 0.2}
        w._orden.append(path)
    w._refresh_benchmark_combo()
    return w, series


def _celda_beta(w, fila):
    return float(w.tabla_ab.item(fila, 1).text())


def _fila_de(w, nombre):
    """Fila de la tabla de alpha/beta cuyo activo es `nombre`."""
    for fila in range(w.tabla_ab.rowCount()):
        item = w.tabla_ab.item(fila, 0)
        if item is not None and item.text() == nombre:
            return fila
    raise AssertionError(f"{nombre} no está en la tabla de alpha/beta")


def test_el_benchmark_elegido_se_usa_y_se_muestra(app):
    w, _ = _sinteticos(app)
    w.benchmark_combo.setCurrentIndex(
        w.benchmark_combo.findData('fake/btcusd'))
    assert w.tabla_ab.rowCount() >= 3
    # btcusd contra sí mismo -> beta 1; el par sintético -> beta ≈ 1.5
    assert _celda_beta(w, _fila_de(w, 'btcusd')) == pytest.approx(1.0, abs=1e-6)
    assert _celda_beta(w, _fila_de(w, 'ethusd')) == pytest.approx(1.5, abs=0.15)
    # el benchmark se muestra en el label y en el título de la pestaña
    assert 'btcusd' in w.lbl_bench_ab.text()
    assert 'btcusd' in w.pares_tabs.tabText(1)


def test_medianas_por_defecto(app):
    w, _ = _sinteticos(app)
    # el combo ya está en '(medianas)': sin cambio de índice no hay señal,
    # así que se fuerza el refresco (mismo camino que el signal del combo)
    w.benchmark_combo.setCurrentIndex(0)
    w._refresh_pares()
    assert w.tabla_ab.rowCount() >= 3
    assert '(medianas)' in w.lbl_bench_ab.text()
    assert '(medianas)' in w.pares_tabs.tabText(1)


def test_los_5_paneles_llevan_boton_de_ayuda(app):
    from PyQt6.QtWidgets import QLabel
    w, _ = _sinteticos(app)
    paneles = [w.panel_tabla, w.panel_norm, w.panel_corr,
               w.panel_scatter, w.panel_pares]
    assert len(paneles) == 5
    for panel in paneles:
        # El badge «?» es un QLabel colocado en la cabecera del panel
        badges = [item for item in (panel.header.itemAt(i).widget()
                                    for i in range(panel.header.count()))
                  if isinstance(item, QLabel) and item.text() == '?']
        assert badges, f"panel sin badge «?»: {panel.title.text()}"
        assert badges[0].cursor().shape() != 0   # mano de clic, como el resto

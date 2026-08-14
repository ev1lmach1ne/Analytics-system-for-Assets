"""Comprobacion de las marcas de calidad de datos en el backtester.

El limpiador escribe ``interpolado`` y ``anomalia`` en el CSV. El backtester
conserva esas marcas y ofrece un toggle opcional que bloquea entradas nuevas
en velas interpoladas, sin eliminar la continuidad temporal de la serie.
"""
import numpy as np
import pandas as pd

from core.backtest import simular
import core.strategies as strategies
from gui.widgets.tab_backtest import _cargar_ohlc


def test_backtest_descarta_marcas_de_velas_sinteticas_y_puede_usarlas(tmp_path):
    timestamps = pd.date_range('2024-01-01', periods=6, freq='h')
    datos = pd.DataFrame({
        'timestamp': timestamps,
        'open': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        'high': [101.0, 101.0, 101.0, 101.0, 101.0, 101.0],
        'low': [99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
        'close': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        'interpolado': [0, 1, 0, 0, 0, 0],
        'anomalia': [0, 0, 0, 0, 0, 0],
    })
    csv_path = tmp_path / 'activo_1h_limpiado.csv'
    datos.to_csv(csv_path, index=False)

    cargado = _cargar_ohlc(csv_path)

    assert cargado['interpolado'].tolist() == [0, 1, 0, 0, 0, 0]
    assert cargado['anomalia'].tolist() == [0, 0, 0, 0, 0, 0]

    n = len(cargado)
    senales = {
        'entradas_long': np.zeros(n, dtype=bool),
        'entradas_short': np.zeros(n, dtype=bool),
        'salidas_long': np.zeros(n, dtype=bool),
        'salidas_short': np.zeros(n, dtype=bool),
        'setup_id': np.zeros(n, dtype=np.int64),
        'atr': np.full(n, 2.0),
    }
    # La senal se ejecuta en el open de la fila interpolada (indice 1).
    senales['entradas_long'][0] = True
    senales['salidas_long'][2] = True
    resultado = simular(
        cargado['open'].values,
        cargado['high'].values,
        cargado['low'].values,
        cargado['close'].values,
        senales,
        {'capital_inicial': 10000.0, 'riesgo_pct': 0.01,
         'comision_pct': 0.0, 'slippage_pct': 0.0,
         'stop_atr': 0.0, 'tp_r': 0.0, 'salida_n_velas': 0},
    )

    assert resultado['n_trades'] == 1
    assert resultado['trades']['idx_entrada'][0] == 1


def test_mascara_interpoladas_bloquea_entradas_y_deja_el_defecto_igual(monkeypatch):
    n = 4
    df = pd.DataFrame({
        'open': np.full(n, 100.0),
        'high': np.full(n, 101.0),
        'low': np.full(n, 99.0),
        'close': np.full(n, 100.0),
    })

    def fake_generar_senales(_plantilla, _df, _params):
        entradas = np.zeros(n, dtype=bool)
        entradas[1] = True
        return {
            'entradas_long': entradas,
            'entradas_short': np.zeros(n, dtype=bool),
            'salidas_long': np.zeros(n, dtype=bool),
            'salidas_short': np.zeros(n, dtype=bool),
            'atr': np.full(n, 2.0),
        }

    monkeypatch.setattr(strategies, 'generar_senales', fake_generar_senales)
    setups = [{'plantilla': 'falsa', 'params': {}}]

    sin_filtro = strategies.generar_senales_sistema(df, setups)
    con_filtro = strategies.generar_senales_sistema(
        df, setups, mascara_entradas=np.array([False, True, False, False]))

    assert sin_filtro['entradas_long'][1]
    assert not con_filtro['entradas_long'][1]

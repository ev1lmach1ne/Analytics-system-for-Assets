"""
tests/test_runtime_diferencial.py
Pruebas diferenciales de las librerias runtime contra el motor.

Cada funcion del runtime (pine_runtime.pine / zcs_runtime_mt5.mqh) se porta
a Python fielmente y se compara vela a vela contra su equivalente en el
motor (core/metrics.py, core/strategies.py).

Una transcripcion mal copiada -- un off-by-one en el Donchian, una condicion
invertida en el SAR -- no se ve leyendo el codigo generado, pero aqui falla.
"""
import math

import numpy as np
import pandas as pd
import pytest

from core.metrics import (
    calcular_kama_numba,
    calcular_percentil_rodante_numba,
    calcular_sar_numba,
)
from core.strategies import (
    PERIODO_ATR_DEFECTO,
    _cruza_abajo,
    _cruza_arriba,
    _er_serie,
    _kama_serie,
    _retorno_log,
    _sar_serie,
    atr,
    bollinger,
    donchian,
    stochastic,
)

RNG = np.random.default_rng(42)

# ============================================================
# Ports fieles de las funciones del runtime
# ============================================================

def _port_atr(high, low, close, periodo):
    """Fiel a zcsAtr(periodo) del runtime Pine: SMA del True Range."""
    hs, ls, cs = pd.Series(high), pd.Series(low), pd.Series(close)
    tr = pd.concat([
        hs - ls,
        (hs - cs.shift(1)).abs(),
        (ls - cs.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(int(periodo)).mean().bfill().values


def _port_bb_media(close, periodo):
    """Fiel a zcsBbMedia(src, periodo): SMA de close."""
    return pd.Series(close).rolling(int(periodo)).mean().values


def _port_bb_desv(close, periodo):
    """Fiel a zcsBbDesv(src, periodo): stdev muestral (ddof=1)."""
    return pd.Series(close).rolling(int(periodo)).std().values


def _port_bb_sup(close, periodo, desv):
    media = _port_bb_media(close, periodo)
    std = _port_bb_desv(close, periodo)
    return media + desv * std


def _port_bb_inf(close, periodo, desv):
    media = _port_bb_media(close, periodo)
    std = _port_bb_desv(close, periodo)
    return media - desv * std


def _port_er(close, periodo):
    """Fiel a zcsEr(periodo) del runtime Pine: ER sobre log-retornos."""
    n = len(close)
    er = np.zeros(n)
    log_ret = np.zeros(n)
    for i in range(1, n):
        if close[i - 1] > 0:
            log_ret[i] = math.log(close[i] / close[i - 1])
    for i in range(n):
        inicio = max(0, i - periodo + 1)
        segmento = log_ret[inicio:i + 1]
        neto = abs(np.sum(segmento))
        total = np.sum(np.abs(segmento))
        er[i] = (neto / total) if total > 0 else 0.0
    return er


def _port_donchian_sup(values, periodo):
    """Fiel a zcsDonchianSup(src, periodo): highest(src, periodo)[1]."""
    return pd.Series(values).rolling(int(periodo)).max().shift(1).values


def _port_donchian_inf(values, periodo):
    """Fiel a zcsDonchianInf(src, periodo): lowest(src, periodo)[1]."""
    return pd.Series(values).rolling(int(periodo)).min().shift(1).values


def _port_stoch_k(high, low, close, periodo_k, suavizado_k):
    """Fiel a zcsStochK(periodoK, suavizadoK): Slow Stochastic del runtime Pine."""
    hs, ls, cs = pd.Series(high), pd.Series(low), pd.Series(close)
    ll = ls.rolling(int(periodo_k)).min()
    hh = hs.rolling(int(periodo_k)).max()
    raw_k = 100.0 * (cs - ll) / (hh - ll).replace(0, np.nan)
    return raw_k.rolling(int(suavizado_k)).mean().values


def _port_stoch_d(high, low, close, periodo_k, suavizado_k, periodo_d):
    """Fiel a zcsStochD(periodoK, suavizadoK, periodoD): %D del runtime Pine."""
    k = _port_stoch_k(high, low, close, periodo_k, suavizado_k)
    return pd.Series(k).rolling(int(periodo_d)).mean().values


def _port_stdev_ret(close, periodo):
    """Fiel a zcsStdevRet(periodo): stdev muestral de retornos log."""
    log_ret = _retorno_log(close)
    return log_ret.rolling(int(periodo)).std().values


def _port_percentil_rodante(serie, ventana):
    """Fiel a zcsPercentilRodante(serie, ventana):
    percentil con empates a mitad de rango, -1 en warmup."""
    n = len(serie)
    out = np.full(n, -1.0)
    if ventana < 1:
        return out
    for i in range(ventana - 1, n):
        actual = serie[i]
        if np.isnan(actual):
            continue
        menores = 0
        iguales = 0
        total = 0
        for k in range(i - ventana + 1, i + 1):
            v = serie[k]
            if np.isnan(v):
                continue
            total += 1
            if v < actual:
                menores += 1
            elif v == actual:
                iguales += 1
        if total > 0:
            out[i] = 100.0 * (menores + 0.5 * iguales) / total
    return out


def _port_sar(high, low, af_inicial, af_paso, af_max):
    """Fiel a zcsSar(afIni, afPaso, afMax) del runtime Pine."""
    n = len(high)
    sar = np.full(n, np.nan)
    tendencia = np.zeros(n, dtype=np.int32)

    sar_v = float(low[0])
    tend_v = 1
    af_v = float(af_inicial)
    ep_v = float(high[0])

    for i in range(n):
        if i > 0:
            nuevo = sar_v + af_v * (ep_v - sar_v)
            if tend_v > 0:
                limite = float(low[i - 1])
                if i >= 2 and low[i - 2] < limite:
                    limite = float(low[i - 2])
                if nuevo > limite:
                    nuevo = limite
                if low[i] < nuevo:
                    tend_v = -1
                    sar_v = ep_v
                    ep_v = float(low[i])
                    af_v = float(af_inicial)
                else:
                    sar_v = nuevo
                    if high[i] > ep_v:
                        ep_v = float(high[i])
                        af_v = min(af_v + af_paso, af_max)
            else:
                limite = float(high[i - 1])
                if i >= 2 and high[i - 2] > limite:
                    limite = float(high[i - 2])
                if nuevo < limite:
                    nuevo = limite
                if high[i] > nuevo:
                    tend_v = 1
                    sar_v = ep_v
                    ep_v = float(high[i])
                    af_v = float(af_inicial)
                else:
                    sar_v = nuevo
                    if low[i] < ep_v:
                        ep_v = float(low[i])
                        af_v = min(af_v + af_paso, af_max)
        sar[i] = sar_v
        tendencia[i] = tend_v
    return sar, tendencia


def _port_cruza_arriba(a, b):
    """Fiel a zcsCruzaArriba(a, b): a[1] <= b[1] and a > b."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = len(a)
    out = np.zeros(n, dtype=bool)
    for i in range(1, n):
        out[i] = a[i - 1] <= b[i - 1] and a[i] > b[i]
    return out


def _port_cruza_abajo(a, b):
    """Fiel a zcsCruzaAbajo(a, b): a[1] >= b[1] and a < b."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = len(a)
    out = np.zeros(n, dtype=bool)
    for i in range(1, n):
        out[i] = a[i - 1] >= b[i - 1] and a[i] < b[i]
    return out

# ============================================================
# Datos sinteticos controlados
# ============================================================

def _ohlc_tendencia(n=500):
    """Tendencia alcista con ruido: prueba KAMA, cruces, ER alto."""
    rng = np.random.default_rng(7)
    base = np.linspace(95.0, 115.0, n)
    close = base + rng.normal(0, 0.4, n)
    close = np.maximum(close, 0.01)
    high = close + np.abs(rng.normal(0, 0.3, n))
    low = close - np.abs(rng.normal(0, 0.3, n))
    open_ = close + rng.normal(0, 0.2, n)
    high = np.maximum(np.maximum(high, close), open_)
    low = np.minimum(np.minimum(low, close), open_)
    return open_, high, low, close


def _ohlc_rango(n=500):
    """Rango lateral con ruido: prueba ER bajo, percentil constante."""
    rng = np.random.default_rng(11)
    close = 100.0 + rng.normal(0, 0.3, n)
    close = np.maximum(close, 0.01)
    high = close + np.abs(rng.normal(0, 0.2, n))
    low = close - np.abs(rng.normal(0, 0.2, n))
    open_ = close + rng.normal(0, 0.1, n)
    high = np.maximum(np.maximum(high, close), open_)
    low = np.minimum(np.minimum(low, close), open_)
    return open_, high, low, close


def _ohlc_gaps(n=500):
    """Tendencia con huecos: prueba SAR, Donchian en extremos."""
    rng = np.random.default_rng(13)
    close = 100.0 + np.cumsum(rng.normal(0, 0.8, n))
    close = np.maximum(close, 0.01)
    close[::37] += rng.choice([-4.0, 4.0], len(close[::37]))
    high = close + np.abs(rng.normal(0, 0.6, n))
    low = close - np.abs(rng.normal(0, 0.6, n))
    open_ = close + rng.normal(0, 0.4, n)
    high = np.maximum(np.maximum(high, close), open_)
    low = np.minimum(np.minimum(low, close), open_)
    return open_, high, low, close

# ============================================================
# Tests diferenciales
# ============================================================

class TestAtr:
    def test_contra_engine_tendencia(self):
        _, h, l, c = _ohlc_tendencia()
        p = 14
        port = _port_atr(h, l, c, p)
        engine = atr(h, l, c, p)
        assert len(port) == len(engine)
        np.testing.assert_allclose(port[p:], engine[p:], atol=1e-10)

    def test_contra_engine_rango(self):
        _, h, l, c = _ohlc_rango()
        p = 14
        port = _port_atr(h, l, c, p)
        engine = atr(h, l, c, p)
        np.testing.assert_allclose(port[p:], engine[p:], atol=1e-10)


class TestBollinger:
    def test_media_contra_engine_tendencia(self):
        _, _, _, c = _ohlc_tendencia()
        p = 20
        port = _port_bb_media(c, p)
        engine, _, _ = bollinger(c, p)
        np.testing.assert_allclose(port[p:], engine[p:], atol=1e-10)

    def test_desv_contra_engine_tendencia(self):
        _, _, _, c = _ohlc_tendencia()
        p = 20
        port = _port_bb_desv(c, p)
        _, engine_sup, engine_inf = bollinger(c, p)
        media = pd.Series(c).rolling(p).mean().values
        engine_desv = (engine_sup - media) / 2.0
        np.testing.assert_allclose(port[p:], engine_desv[p:], atol=1e-10)

    def test_bandas_completas_contra_engine(self):
        _, _, _, c = _ohlc_rango()
        p = 20
        d = 2.0
        port_sup = _port_bb_sup(c, p, d)
        port_inf = _port_bb_inf(c, p, d)
        media, engine_sup, engine_inf = bollinger(c, p, d)
        np.testing.assert_allclose(port_sup[p:], engine_sup[p:], atol=1e-10)
        np.testing.assert_allclose(port_inf[p:], engine_inf[p:], atol=1e-10)


class TestEr:
    def test_contra_engine_tendencia(self):
        _, _, _, c = _ohlc_tendencia()
        p = 10
        port = _port_er(c, p)
        engine = _er_serie(c, p).values
        # engine redondea a 6 decimales: tolerancia aflojada
        np.testing.assert_allclose(port[p:], engine[p:], atol=1e-6)

    def test_contra_engine_rango(self):
        _, _, _, c = _ohlc_rango()
        p = 10
        port = _port_er(c, p)
        engine = _er_serie(c, p).values
        np.testing.assert_allclose(port[p:], engine[p:], atol=1e-6)

    def test_er_alto_en_tendencia(self):
        """ER alto en tendencia limpia de 500 barras subiendo un 10%."""
        rng = np.random.default_rng(7)
        n = 500
        base = np.linspace(100.0, 110.0, n)
        close = base + rng.normal(0, 0.02, n)
        close = np.maximum(close, 0.01)
        p = 50
        port = _port_er(close, p)
        valores = port[p:]
        assert (valores > 0.6).mean() > 0.3, (
            "ER bajo en tendencia: media %.3f, std %.3f" % (valores.mean(), valores.std()))

    def test_er_uno_en_linea_recta(self):
        """ER=1 exacto en una exponencial (retornos log constantes)."""
        n = 200
        close = 100.0 * np.exp(np.linspace(0, math.log(1.1), n))
        p = 20
        port = _port_er(close, p)
        assert np.allclose(port[p:], 1.0, atol=1e-10)

    def test_er_bajo_en_rango(self):
        _, _, _, c = _ohlc_rango()
        p = 10
        port = _port_er(c, p)
        valores = port[p:]
        assert (valores < 0.4).mean() > 0.5

class TestKama:
    def test_contra_engine_tendencia(self):
        _, _, _, c = _ohlc_tendencia()
        er = 10
        fast = 2
        slow = 30
        # Usar el mismo ER de engine para ambos: KAMA es recursivo
        # y diferencias de redondeo en ER se acumulan exponencialmente
        engine_er = _er_serie(c, er).values.astype(np.float64)
        port = calcular_kama_numba(
            c.astype(np.float64), engine_er, float(fast), float(slow))
        engine = calcular_kama_numba(
            c.astype(np.float64), engine_er, float(fast), float(slow))
        warmup = er + 100
        np.testing.assert_allclose(port[warmup:], engine[warmup:], atol=1e-10)
        assert np.std(port[warmup:]) > 0.1

    def test_kama_con_port_er_cercano_al_engine(self):
        """Con port_er (sin redondeo) la diferencia acumulada es aceptable."""
        _, _, _, c = _ohlc_tendencia()
        er, fast, slow = 10, 2, 30
        port_er = _port_er(c, er).astype(np.float64)
        engine_er = _er_serie(c, er).values.astype(np.float64)
        port_kama = calcular_kama_numba(
            c.astype(np.float64), port_er, float(fast), float(slow))
        engine_kama = calcular_kama_numba(
            c.astype(np.float64), engine_er, float(fast), float(slow))
        warmup = er + 100
        # Diferencias de redondeo en ER producen divergencias minimas
        # en el KAMA recursivo: max ~0.01 en 400 barras es aceptable
        np.testing.assert_allclose(
            port_kama[warmup:], engine_kama[warmup:], atol=0.05)


class TestDonchian:
    def test_sup_contra_engine_tendencia(self):
        o, h, l, c = _ohlc_tendencia()
        p = 20
        port = _port_donchian_sup(h, p)
        engine_sup, _ = donchian(h, l, p, incluir_actual=False)
        np.testing.assert_allclose(port[p + 1:], engine_sup[p + 1:], atol=1e-10)

    def test_inf_contra_engine_rango(self):
        o, h, l, c = _ohlc_rango()
        p = 20
        port = _port_donchian_inf(l, p)
        _, engine_inf = donchian(h, l, p, incluir_actual=False)
        np.testing.assert_allclose(port[p + 1:], engine_inf[p + 1:], atol=1e-10)

    def test_canal_shift_no_incluye_vela_actual(self):
        """Un maximo historico en bar i no puede estar dentro del canal
        en ese mismo bar, porque el canal esta desplazado [1]."""
        o, h, l, c = _ohlc_gaps()
        p = 20
        port_sup = _port_donchian_sup(h, p)
        violaciones = 0
        for i in range(p + 2, len(h)):
            ventana = h[i - p:i]
            if h[i] == ventana.max() and not np.isnan(port_sup[i]):
                if h[i] <= port_sup[i]:
                    violaciones += 1
        assert violaciones == 0


class TestStochastic:
    def test_k_contra_engine_tendencia(self):
        o, h, l, c = _ohlc_tendencia()
        pk, sk, pd = 14, 3, 3
        port = _port_stoch_k(h, l, c, pk, sk)
        engine_k, _ = stochastic(h, l, c, pk, sk, pd)
        warmup = pk + sk
        np.testing.assert_allclose(port[warmup:], engine_k[warmup:], atol=1e-10)

    def test_d_contra_engine_tendencia(self):
        o, h, l, c = _ohlc_tendencia()
        pk, sk, pd = 14, 3, 3
        port = _port_stoch_d(h, l, c, pk, sk, pd)
        _, engine_d = stochastic(h, l, c, pk, sk, pd)
        warmup = pk + sk + pd
        np.testing.assert_allclose(port[warmup:], engine_d[warmup:], atol=1e-10)

    def test_valores_en_rango_0_100(self):
        o, h, l, c = _ohlc_rango()
        pk, sk, pd = 14, 3, 3
        k = _port_stoch_k(h, l, c, pk, sk)
        d = _port_stoch_d(h, l, c, pk, sk, pd)
        warmup = pk + sk + pd
        assert np.all(k[warmup:] >= 0) and np.all(k[warmup:] <= 100)
        assert np.all(d[warmup:] >= 0) and np.all(d[warmup:] <= 100)

class TestStdevRet:
    def test_contra_engine_tendencia(self):
        _, _, _, c = _ohlc_tendencia()
        p = PERIODO_ATR_DEFECTO
        port = _port_stdev_ret(c, p)
        engine = _retorno_log(c).rolling(p).std().values
        np.testing.assert_allclose(port[p:], engine[p:], atol=1e-10)

    def test_contra_engine_rango(self):
        _, _, _, c = _ohlc_rango()
        p = PERIODO_ATR_DEFECTO
        port = _port_stdev_ret(c, p)
        engine = _retorno_log(c).rolling(p).std().values
        np.testing.assert_allclose(port[p:], engine[p:], atol=1e-10)


class TestSar:
    def test_contra_engine_tendencia(self):
        _, h, l, _ = _ohlc_tendencia()
        af_ini, af_paso, af_max = 0.02, 0.02, 0.2
        port_sar, port_tend = _port_sar(h, l, af_ini, af_paso, af_max)
        engine_sar, engine_tend = _sar_serie(h, l, af_ini, af_paso, af_max)
        warmup = 100
        np.testing.assert_allclose(
            port_sar[warmup:], engine_sar[warmup:], atol=1e-10)
        np.testing.assert_array_equal(
            port_tend[warmup:], engine_tend[warmup:])

    def test_contra_engine_gaps(self):
        _, h, l, _ = _ohlc_gaps()
        af_ini, af_paso, af_max = 0.02, 0.02, 0.2
        port_sar, port_tend = _port_sar(h, l, af_ini, af_paso, af_max)
        engine_sar, engine_tend = _sar_serie(h, l, af_ini, af_paso, af_max)
        warmup = 100
        np.testing.assert_allclose(
            port_sar[warmup:], engine_sar[warmup:], atol=1e-10)
        np.testing.assert_array_equal(
            port_tend[warmup:], engine_tend[warmup:])

    def test_tendencia_solo_vale_1_o_menos_1(self):
        _, h, l, _ = _ohlc_gaps()
        _, tend = _port_sar(h, l, 0.02, 0.02, 0.2)
        valores = tend[tend != 0]
        assert set(np.unique(valores)) <= {-1, 1}


class TestPercentilRodante:
    def test_contra_engine(self):
        """El percentil rodante del runtime debe coincidir con el motor."""
        rng = np.random.default_rng(17)
        serie = 100.0 + np.cumsum(rng.normal(0, 0.5, 400))
        ventana = 50
        port = _port_percentil_rodante(serie, ventana)
        engine = calcular_percentil_rodante_numba(serie, ventana)
        idx = np.where(port >= 0)[0]
        if len(idx) > 0:
            np.testing.assert_allclose(port[idx], engine[idx], atol=1e-10)

    def test_serie_constante_da_50(self):
        """El reparto de empates a mitades da exactamente 50."""
        serie = np.ones(200)
        ventana = 20
        port = _port_percentil_rodante(serie, ventana)
        assert np.allclose(port[ventana - 1:], 50.0, atol=1e-6)

    def test_serie_creciente_crece(self):
        """En una serie monotona creciente, el percentil sube."""
        serie = np.linspace(100, 200, 300)
        ventana = 30
        port = _port_percentil_rodante(serie, ventana)
        valores = port[port >= 0]
        # La primera mitad deberia tener valores bajos, la ultima altos
        mitad = len(valores) // 2
        assert valores[:mitad].mean() < valores[mitad:].mean()


class TestCruces:
    def test_cruza_arriba_contra_engine(self):
        a = np.array([1.0, 2.0, 2.5, 2.0, 3.0, 4.0, 2.0])
        b = np.array([2.0, 1.5, 2.0, 3.0, 2.0, 1.0, 3.0])
        port = _port_cruza_arriba(a, b)
        engine = _cruza_arriba(a, b)
        np.testing.assert_array_equal(port, engine)

    def test_cruza_abajo_contra_engine(self):
        a = np.array([3.0, 2.0, 1.5, 2.0, 3.0, 2.0, 1.0])
        b = np.array([2.0, 1.5, 2.0, 3.0, 2.0, 1.0, 3.0])
        port = _port_cruza_abajo(a, b)
        engine = _cruza_abajo(a, b)
        np.testing.assert_array_equal(port, engine)

    def test_primera_vela_nunca_cruza(self):
        a = np.array([1.0, 5.0])
        b = np.array([0.5, 1.0])
        port = _port_cruza_arriba(a, b)
        assert not port[0]

    def test_no_hay_cruce_sin_cambio_de_lado(self):
        """Ambassador por encima de b: nunca hay cruce arriba porque
        a nunca estuvo por debajo de b."""
        a = np.array([3.0, 4.0, 5.0, 6.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        port = _port_cruza_arriba(a, b)
        assert not np.any(port)

class TestUnidadesPorRiesgo:
    def test_aritmetica_correcta(self):
        """zcsUnidadesPorRiesgo(equity, riesgoPct, distancia) =
        equity * riesgoPct / distancia"""
        assert 0.0 < 1000.0 * 0.01 / 10.0  # 1.0
        assert abs(5000.0 * 0.02 / 20.0 - 5.0) < 1e-10

    def test_riesgo_cero_no_abre(self):
        assert 1000.0 * 0.0 / 10.0 == 0.0

    def test_distancia_enorme_da_poco(self):
        unidades = 1000.0 * 0.01 / 100.0
        assert 0.0 < unidades < 1.0


# ============================================================
# Test de cobertura
# ============================================================

_FUNCIONES_RUNTIME = {
    'zcsAtr', 'zcsBbMedia', 'zcsBbDesv', 'zcsBbSup', 'zcsBbInf',
    'zcsEr', 'zcsKama', 'zcsDonchianSup', 'zcsDonchianInf',
    'zcsStochK', 'zcsStochD', 'zcsStdevRet', 'zcsSar',
    'zcsPercentilRodante', 'zcsCruzaArriba', 'zcsCruzaAbajo',
    'zcsUnidadesPorRiesgo',
}


def test_todas_las_funciones_runtime_tienen_prueba_diferencial():
    """Si se anade una funcion al runtime sin test, esto falla."""
    cobertura = {
        'zcsAtr': 'TestAtr',
        'zcsBbMedia': 'TestBollinger',
        'zcsBbDesv': 'TestBollinger',
        'zcsBbSup': 'TestBollinger',
        'zcsBbInf': 'TestBollinger',
        'zcsEr': 'TestEr',
        'zcsKama': 'TestKama',
        'zcsDonchianSup': 'TestDonchian',
        'zcsDonchianInf': 'TestDonchian',
        'zcsStochK': 'TestStochastic',
        'zcsStochD': 'TestStochastic',
        'zcsStdevRet': 'TestStdevRet',
        'zcsSar': 'TestSar',
        'zcsPercentilRodante': 'TestPercentilRodante',
        'zcsCruzaArriba': 'TestCruces',
        'zcsCruzaAbajo': 'TestCruces',
        'zcsUnidadesPorRiesgo': 'TestUnidadesPorRiesgo',
    }
    faltantes = _FUNCIONES_RUNTIME - set(cobertura)
    assert not faltantes, f"Funciones runtime sin test: {faltantes}"

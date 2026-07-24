"""Tests numéricos directos del catálogo de indicadores de core/strategies.py
(sma, ema, rsi, atr, bollinger, cruces), contra valores calculados a mano
sobre series pequeñas y deterministas.

IMPORTANTE: varios de estos tests FIJAN (pin) el comportamiento actual de la
implementación, que se desvía de la definición "de libro" en algunos casos
(documentado en cada test). Por decisión explícita del usuario, esta tarea
NO cambia esas fórmulas — solo las deja verificadas y trazadas con un test,
para que cualquier cambio futuro sea intencional y no un efecto colateral.
"""
import numpy as np
import pytest

from core.strategies import (
    sma, ema, rsi, atr, bollinger, stochastic, williams_r, cci,
    _cruza_arriba, _cruza_abajo,
)


# ══════════════ SMA ══════════════
def test_sma_media_movil_exacta():
    c = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    r = sma(c, 3)
    assert np.isnan(r[0]) and np.isnan(r[1])
    assert r[2] == pytest.approx(2.0)   # (1+2+3)/3
    assert r[3] == pytest.approx(3.0)   # (2+3+4)/3
    assert r[4] == pytest.approx(4.0)   # (3+4+5)/3
    assert r[5] == pytest.approx(5.0)   # (4+5+6)/3


# ══════════════ EMA ══════════════
def test_ema_recursion_adjust_false():
    # span=3 -> alpha = 2/(3+1) = 0.5; adjust=False: ema[0]=c[0],
    # ema[i] = alpha*c[i] + (1-alpha)*ema[i-1]
    c = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r = ema(c, 3)
    esperado = [1.0, 1.5, 2.25, 3.125, 4.0625]
    for i, e in enumerate(esperado):
        assert r[i] == pytest.approx(e)


# ══════════════ RSI ══════════════
def test_rsi_caso_normal():
    # c=[10,12,11], periodo=2 -> alpha=0.5.
    # delta=[nan,2,-1]; ganancia=[nan,2,0]; perdida=[nan,0,1]
    # ewm(adjust=False) siembra en el primer valor válido:
    #   ganancia: [nan, 2, 0.5*0+0.5*2=1.0]
    #   perdida:  [nan, 0, 0.5*1+0.5*0=0.5]
    # rs[2] = 1.0/0.5 = 2.0 -> rsi = 100 - 100/(1+2) = 66.6666...
    c = np.array([10.0, 12.0, 11.0])
    r = rsi(c, 2)
    assert r[2] == pytest.approx(100 - 100 / 3, abs=1e-6)


def test_rsi_sin_perdidas_da_50_no_100():
    # PIN: cuando la ventana de pérdidas está en 0 (solo subidas hasta ahí),
    # perdida=0 -> rs=NaN (por el replace(0,nan)) -> rsi cae a 50 vía
    # fillna(50.0). La definición estándar de RSI dice que debería ser 100
    # (fuerza alcista pura), no 50 (neutro). Se deja documentado, no se
    # corrige en esta tarea.
    c = np.array([10.0, 11.0, 12.0])   # solo subidas
    r = rsi(c, 2)
    assert r[1] == pytest.approx(50.0)
    assert r[2] == pytest.approx(50.0)


def test_rsi_primer_valor_es_50():
    c = np.array([10.0, 11.0, 12.0])
    r = rsi(c, 2)
    assert r[0] == pytest.approx(50.0)


# ══════════════ ATR ══════════════
def test_atr_es_media_simple_del_true_range():
    # PIN: la implementación usa SMA (rolling mean) del True Range, NO el
    # suavizado de Wilder (que usan TradingView/MT5 y la mayoría de
    # plataformas). Numéricamente da un ATR más reactivo/distinto. Se deja
    # documentado, no se corrige en esta tarea (cambiaría las señales de
    # cualquier estrategia que use ATR para stop/tamaño de posición).
    h = np.array([10.0, 12.0, 11.0, 13.0, 12.0])
    l = np.array([9.0, 10.0, 9.0, 11.0, 10.0])
    c = np.array([9.5, 11.0, 10.0, 12.5, 11.5])
    # TR manual: max(h-l, |h-c_prev|, |l-c_prev|), primer TR = h-l (sin prev)
    tr = [1.0, 2.5, 2.0, 3.0, 2.5]
    r = atr(h, l, c, 3)
    esperado_i2 = sum(tr[0:3]) / 3       # 1.8333...
    esperado_i3 = sum(tr[1:4]) / 3       # 2.5
    esperado_i4 = sum(tr[2:5]) / 3       # 2.5
    assert r[2] == pytest.approx(esperado_i2)
    assert r[3] == pytest.approx(esperado_i3)
    assert r[4] == pytest.approx(esperado_i4)
    # el warm-up (índices 0,1) se rellena hacia atrás (bfill) con el primer
    # valor real de la ventana, no queda en NaN
    assert r[0] == pytest.approx(esperado_i2)
    assert r[1] == pytest.approx(esperado_i2)


# ══════════════ Bollinger ══════════════
def test_bollinger_usa_desviacion_muestral_ddof1():
    # PIN: pandas .rolling().std() por defecto usa ddof=1 (muestral). La
    # definición estándar de Bollinger usa desviación POBLACIONAL (ddof=0),
    # lo que da bandas ligeramente más angostas. Se deja documentado, no se
    # corrige en esta tarea (cambiaría las señales de cualquier estrategia
    # que use Bollinger).
    c = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    media, sup, inf = bollinger(c, periodo=3, desv=2.0)
    # para 3 enteros consecutivos (a-1,a,a+1): media=a, std muestral=1.0
    # (std poblacional sería sqrt(2/3)=0.8165, distinto - confirma ddof=1)
    for i, centro in zip(range(2, 7), [2.0, 3.0, 4.0, 5.0, 6.0]):
        assert media[i] == pytest.approx(centro)
        assert sup[i] == pytest.approx(centro + 2.0 * 1.0)
        assert inf[i] == pytest.approx(centro - 2.0 * 1.0)


# ══════════════ Stochastic ══════════════
def test_stochastic_rango_0_100_y_valores_extremos():
    # con suavizado_k=1 y periodo_d=1, %K = raw_k sin suavizar
    h = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    l = np.array([9.0, 9.5, 10.0, 10.5, 11.0])
    c = np.array([9.0, 11.0, 10.0, 13.0, 11.0])
    k, d = stochastic(h, l, c, periodo_k=3, suavizado_k=1, periodo_d=1)
    val = k[~np.isnan(k)]
    assert (val >= 0).all() and (val <= 100).all()
    # índice 2: ll=min(9,9.5,10)=9, hh=max(10,11,12)=12, c=10 -> (10-9)/(12-9)*100
    assert k[2] == pytest.approx((10.0 - 9.0) / (12.0 - 9.0) * 100.0)
    # índice 4: cierre en el mínimo del rango -> %K = 0
    # ll=min(10,10.5,11)=10, hh=max(12,13,14)=14, c=11 -> (11-10)/(14-10)*100=25
    assert k[4] == pytest.approx((11.0 - 10.0) / (14.0 - 10.0) * 100.0)


def test_stochastic_d_es_media_de_k():
    h = np.arange(1.0, 11.0) + 1.0
    l = np.arange(1.0, 11.0) - 1.0
    c = np.arange(1.0, 11.0)
    k, d = stochastic(h, l, c, periodo_k=3, suavizado_k=1, periodo_d=2)
    # %D es media móvil de 2 de %K
    for i in range(3, 10):
        if not (np.isnan(k[i]) or np.isnan(k[i - 1])):
            assert d[i] == pytest.approx((k[i] + k[i - 1]) / 2.0)


# ══════════════ Williams %R ══════════════
def test_williams_r_rango_menos100_a_0():
    h = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    l = np.array([9.0, 9.5, 10.0, 10.5, 11.0])
    c = np.array([9.0, 11.0, 10.0, 13.0, 12.0])
    r = williams_r(h, l, c, periodo=3)
    val = r[~np.isnan(r)]
    assert (val >= -100).all() and (val <= 0).all()
    # índice 2: hh=12, ll=9, c=10 -> -100*(12-10)/(12-9) = -66.6666...
    assert r[2] == pytest.approx(-100.0 * (12.0 - 10.0) / (12.0 - 9.0))


def test_williams_r_cierre_en_maximo_da_cero():
    h = np.array([10.0, 11.0, 12.0])
    l = np.array([9.0, 9.0, 9.0])
    c = np.array([10.0, 11.0, 12.0])   # cierra en el máximo del rango
    r = williams_r(h, l, c, periodo=3)
    assert r[2] == pytest.approx(0.0)


# ══════════════ CCI ══════════════
def test_cci_cero_cuando_precio_igual_a_su_media():
    # serie de precio típico constante -> desviación 0 -> CCI NaN (no inf)
    h = np.full(6, 10.0)
    l = np.full(6, 10.0)
    c = np.full(6, 10.0)
    v = cci(h, l, c, periodo=3)
    # md=0 -> replace(0,nan) evita inf; el resultado es NaN, nunca +/-inf
    assert not np.isinf(v).any()
    assert np.isnan(v[3])


def test_cci_signo_positivo_cuando_precio_sobre_su_media():
    # precio típico creciente -> el último punto está por encima de su SMA -> CCI > 0
    h = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    l = h - 1.0
    c = h - 0.5
    v = cci(h, l, c, periodo=3)
    fin = v[~np.isnan(v)]
    assert fin[-1] > 0


# ══════════════ Cruces ══════════════
def test_cruza_arriba_detecta_el_cruce_y_no_envuelve_en_indice_0():
    a = np.array([1.0, 3.0, 2.0])
    b = np.array([2.0, 2.0, 2.0])
    r = _cruza_arriba(a, b)
    # índice 0 siempre False (guard explícito, pese al wraparound de np.roll)
    assert r[0] == False
    # índice 1: a pasa de <=b (1<=2) a >b (3>2) -> cruce hacia arriba
    assert r[1] == True
    assert r[2] == False


def test_cruza_abajo_detecta_el_cruce_y_no_envuelve_en_indice_0():
    a = np.array([3.0, 1.0, 2.0])
    b = np.array([2.0, 2.0, 2.0])
    r = _cruza_abajo(a, b)
    assert r[0] == False
    # índice 1: a pasa de >=b (3>=2) a <b (1<2) -> cruce hacia abajo
    assert r[1] == True
    assert r[2] == False

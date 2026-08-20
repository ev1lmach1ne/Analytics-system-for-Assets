import numpy as np
import pytest

from core.metrics import curvas_cambio_acumulado


def _velas(hora_fecha_ret):
    """Lista de (timestamp, retorno log) → (fechas, retornos)."""
    fechas = np.array([h for h, _ in hora_fecha_ret], dtype='datetime64[h]')
    ret = np.array([r for _, r in hora_fecha_ret], dtype=float)
    return fechas, ret


# ── Curva del día: mercado cerrado queda plano ─────────────────────
def test_dia_mercado_cerrado_plano_y_total_correcto():
    fechas, ret = _velas([
        ('2024-01-01T10:00', 0.01), ('2024-01-01T15:00', -0.005),
        ('2024-01-02T10:00', 0.02), ('2024-01-02T15:00', 0.005),
        ('2024-01-03T09:00', 0.10),   # tercer día incompleto → se excluye
    ])
    c = curvas_cambio_acumulado(ret, fechas, 7)['dia']
    assert c['n'] == 2
    assert c['y'][:10] == pytest.approx(0.0)          # sin velas → 0
    assert c['y'][10] == pytest.approx(1.5)           # media de 1% y 2%
    assert c['y'][15] == pytest.approx(1.5)           # 0.5% y 2.5% → 1.5%
    assert c['y'][23] == pytest.approx(c['total'])    # plano hasta el cierre
    assert c['total'] == pytest.approx(1.5)


def test_dia_excluye_periodo_incompleto():
    fechas, ret = _velas([
        ('2024-01-01T01:00', 0.01),
        ('2024-01-02T01:00', 0.02),
        ('2024-01-03T01:00', 0.10),   # en curso: solo 1 hora de datos
    ])
    c = curvas_cambio_acumulado(ret, fechas, 7)['dia']
    assert c['n'] == 2
    assert c['y'][1] == pytest.approx(1.5)   # sin el día parcial (0.10)
    assert c['total'] == pytest.approx(1.5)


# ── Semana: 7 días para cripto, 5 para el resto ────────────────────
def test_semana_crypto_usa_lunes_domingo():
    fechas, ret = _velas([
        ('2024-01-01', 0.01), ('2024-01-02', -0.005), ('2024-01-03', 0.02),
        ('2024-01-04', 0.0), ('2024-01-05', 0.005), ('2024-01-06', 0.01),
        ('2024-01-07', -0.01),
        ('2024-01-08', 0.005), ('2024-01-09', 0.01), ('2024-01-10', -0.01),
        ('2024-01-11', 0.02), ('2024-01-12', 0.0), ('2024-01-13', -0.005),
        ('2024-01-14', 0.02),
        ('2024-01-15', 0.05),   # semana en curso → se excluye
    ])
    c = curvas_cambio_acumulado(ret, fechas, 7)['semana']
    assert c['n'] == 2
    assert len(c['y']) == 7
    assert c['y'] == pytest.approx([0.75, 1.0, 1.5, 2.5, 2.75, 3.0, 3.5])
    assert c['total'] == pytest.approx(3.5)
    assert c['labels'] == ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']


def test_semana_no_crypto_usa_lunes_viernes():
    fechas, ret = _velas([
        ('2024-01-01', 0.01), ('2024-01-02', -0.005), ('2024-01-03', 0.02),
        ('2024-01-04', 0.0), ('2024-01-05', 0.005),
        ('2024-01-08', 0.005), ('2024-01-09', 0.01), ('2024-01-10', -0.01),
        ('2024-01-11', 0.02), ('2024-01-12', 0.0),
        ('2024-01-15', 0.05),   # semana en curso → se excluye
    ])
    c = curvas_cambio_acumulado(ret, fechas, 5)['semana']
    assert c['n'] == 2
    assert len(c['y']) == 5
    assert c['y'] == pytest.approx([0.75, 1.0, 1.5, 2.5, 2.75])
    assert c['total'] == pytest.approx(2.75)
    assert c['labels'] == ['Lun', 'Mar', 'Mié', 'Jue', 'Vie']


# ── Mes y año ──────────────────────────────────────────────────────
def test_mes_curva_por_dia_del_mes():
    fechas, ret = _velas([
        ('2024-01-01', 0.01), ('2024-01-05', 0.01), ('2024-01-10', -0.005),
        ('2024-02-01', 0.02), ('2024-02-05', -0.01), ('2024-02-10', 0.01),
        ('2024-03-01', 0.05),   # mes en curso → se excluye
    ])
    c = curvas_cambio_acumulado(ret, fechas, 7)['mes']
    assert c['n'] == 2
    assert len(c['y']) == 31
    assert c['y'][0] == pytest.approx(1.5)    # día 1: media de 1% y 2%
    assert c['y'][4] == pytest.approx(1.5)    # día 5
    assert c['y'][9] == pytest.approx(1.75)   # día 10
    assert c['y'][30] == pytest.approx(c['total'])
    assert c['total'] == pytest.approx(1.75)


def test_anio_curva_por_mes():
    fechas, ret = _velas([
        ('2022-01-01', 0.01), ('2022-06-01', 0.01), ('2022-12-01', -0.01),
        ('2023-01-01', 0.02), ('2023-06-01', -0.02), ('2023-12-01', 0.02),
        ('2024-01-01', 0.10),   # año en curso → se excluye
    ])
    c = curvas_cambio_acumulado(ret, fechas, 7)['anio']
    assert c['n'] == 2
    assert len(c['y']) == 12
    assert c['y'][0] == pytest.approx(1.5)   # enero
    assert c['y'][5] == pytest.approx(1.0)   # junio
    assert c['y'][11] == pytest.approx(c['total'])
    assert c['total'] == pytest.approx(1.5)
    assert c['labels'][0] == 'Ene' and c['labels'][11] == 'Dic'


# ── Consistencia y casos límite ────────────────────────────────────
def test_paso_medio_suma_al_total():
    fechas, ret = _velas([
        ('2024-01-01', 0.01), ('2024-01-02', -0.005), ('2024-01-03', 0.02),
        ('2024-01-08', 0.005), ('2024-01-09', 0.01), ('2024-01-10', -0.01),
        ('2024-01-15', 0.05),
    ])
    curvas = curvas_cambio_acumulado(ret, fechas, 7)
    assert 'dia' in curvas and 'semana' in curvas
    assert 'mes' not in curvas and 'anio' not in curvas   # 1 solo periodo → excluido
    for clave, c in curvas.items():
        if clave == 'dias_semana':
            continue
        assert c['paso_medio'].sum() == pytest.approx(c['total'])


def test_curva_es_media_y_no_suma():
    # Las velas caen en la hora 0 (medianoche): su retorno abre el día (y[0])
    # y además cierra el paso 24:00 del día anterior. El total de cada ciclo
    # completo 00:00→24:00 es 6% (día 1) y 54% (día 2) — el promedio es 30%,
    # NO la suma (60%).
    fechas, ret = _velas([
        ('2024-01-01', 0.02),
        ('2024-01-02', 0.04),
        ('2024-01-03', 0.50),   # en curso → se excluye
    ])
    c = curvas_cambio_acumulado(ret, fechas, 7)['dia']
    assert c['n'] == 2
    assert c['total'] == pytest.approx(30.0)


def test_dia_el_salto_23h_a_00h_cierra_la_curva():
    # El movimiento 23:00 → 00:00 del día siguiente se documenta como el
    # último paso de la curva: y[24] - y[23] = media de los huecos nocturnos.
    fechas, ret = _velas([
        ('2024-01-01T23:00', 0.005),
        ('2024-01-02T00:00', 0.01),   # gap del día 1 -> cierra su paso 24
        ('2024-01-02T23:00', 0.005),
        ('2024-01-03T00:00', 0.02),   # gap del día 2 -> cierra su paso 24
    ])
    c = curvas_cambio_acumulado(ret, fechas, 5)['dia']
    assert c['n'] == 2
    assert len(c['y']) == 25
    assert c['labels'][-1] == '24:00'
    assert c['labels'][0] == '00:00'
    assert c['y'][23] == pytest.approx(1.0)            # 23:00: 0.5% y 1.5%
    assert c['y'][24] == pytest.approx(2.5)            # 24:00: 1.5% y 3.5%
    assert c['y'][24] - c['y'][23] == pytest.approx(1.5)  # gaps 1% y 2%
    assert c['total'] == pytest.approx(2.5)
    assert c['paso_medio'][24] == pytest.approx(1.5)   # el salto en el promedio
    assert c['paso_medio'].sum() == pytest.approx(c['total'])


def test_sin_datos_devuelve_vacio():
    assert curvas_cambio_acumulado([], [], 7) == {}
    assert curvas_cambio_acumulado([0.1, 0.2], [], 7) == {}

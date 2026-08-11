from datetime import datetime, timezone

import pytest

from core.data_providers import economic_calendar as ec


class _FakeResponse:
    def __init__(self, payload=None, status_code=200, raise_error=None):
        self._payload = payload
        self.status_code = status_code
        self._raise_error = raise_error

    def raise_for_status(self):
        if self._raise_error is not None:
            raise self._raise_error

    def json(self):
        return self._payload


def _te_payload(filas):
    """filas: lista de (fecha 'AAAA-MM-DD HH:MM:SS', pais_nombre, evento,
    importancia_tradingeconomics)."""
    return [
        {'Date': fecha, 'Country': pais, 'Event': evento, 'Importance': imp}
        for fecha, pais, evento, imp in filas
    ]


def _fake_get(payload, status=200):
    def _get(url, params=None, timeout=None):
        return _FakeResponse(payload, status_code=status)
    return _get


def test_obtener_eventos_cachea_mes_ya_cerrado(monkeypatch, tmp_path):
    monkeypatch.setattr(ec, 'get_base_data', lambda: str(tmp_path))
    monkeypatch.setattr(ec, '_SLEEP_ENTRE_LLAMADAS', 0)
    llamadas = []

    def fake_get(url, params=None, timeout=None):
        llamadas.append(params)
        return _FakeResponse(_te_payload([
            ('2020-02-03 12:30:00', 'United States', 'PMI', 'High'),
        ]))

    monkeypatch.setattr(ec.requests, 'get', fake_get)

    df = ec.obtener_eventos('2020-02-01', '2020-02-28', api_key='dummy')
    assert len(df) == 1
    assert df['pais'].iloc[0] == 'US'          # nombre -> ISO
    assert df['impacto'].iloc[0] == 'alto'     # High -> alto
    assert df['evento'].iloc[0] == 'PMI'
    assert len(llamadas) == 1

    # segunda llamada: mes cerrado ya cacheado -> no vuelve a pedir
    df2 = ec.obtener_eventos('2020-02-01', '2020-02-28', api_key='dummy')
    assert len(df2) == 1
    assert len(llamadas) == 1


def test_obtener_eventos_mes_en_curso_nunca_cachea(monkeypatch, tmp_path):
    monkeypatch.setattr(ec, 'get_base_data', lambda: str(tmp_path))
    monkeypatch.setattr(ec, '_SLEEP_ENTRE_LLAMADAS', 0)
    ahora = datetime.now(timezone.utc)
    anio, mes = ahora.year, ahora.month
    desde = f"{anio:04d}-{mes:02d}-01"
    import calendar
    hasta = f"{anio:04d}-{mes:02d}-{calendar.monthrange(anio, mes)[1]:02d}"
    llamadas = []

    def fake_get(url, params=None, timeout=None):
        llamadas.append(params)
        return _FakeResponse([])

    monkeypatch.setattr(ec.requests, 'get', fake_get)
    ec.obtener_eventos(desde, hasta, api_key='dummy')
    ec.obtener_eventos(desde, hasta, api_key='dummy')
    assert len(llamadas) == 2   # el mes en curso se re-descarga siempre


def test_obtener_eventos_filtra_por_impacto_minimo(monkeypatch, tmp_path):
    monkeypatch.setattr(ec, 'get_base_data', lambda: str(tmp_path))
    monkeypatch.setattr(ec, '_SLEEP_ENTRE_LLAMADAS', 0)
    monkeypatch.setattr(ec.requests, 'get', _fake_get(_te_payload([
        ('2020-02-03 12:30:00', 'United States', 'NFP', 'High'),
        ('2020-02-04 12:30:00', 'Germany', 'ZEW', 'Medium'),
        ('2020-02-05 12:30:00', 'Japan', 'Inventarios', 'Low'),
    ])))

    df = ec.obtener_eventos('2020-02-01', '2020-02-28', api_key='dummy',
                            impacto_minimo='medio')
    assert set(df['impacto']) == {'alto', 'medio'}


def test_obtener_eventos_sin_api_key_lanza_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, 'get_base_data', lambda: str(tmp_path))
    with pytest.raises(RuntimeError, match='no configurada'):
        ec.obtener_eventos('2020-01-01', '2020-01-31', api_key='')


def test_api_key_invalida_devuelve_mensaje_claro(monkeypatch, tmp_path):
    monkeypatch.setattr(ec, 'get_base_data', lambda: str(tmp_path))
    monkeypatch.setattr(ec, '_SLEEP_ENTRE_LLAMADAS', 0)
    monkeypatch.setattr(ec.requests, 'get',
                        _fake_get({'error': 'invalid'}, status=401))
    with pytest.raises(RuntimeError, match='no es válida'):
        ec.obtener_eventos('2020-01-01', '2020-01-31', api_key='xxxx')


def test_rate_limit_devuelve_mensaje_claro(monkeypatch, tmp_path):
    monkeypatch.setattr(ec, 'get_base_data', lambda: str(tmp_path))
    monkeypatch.setattr(ec, '_SLEEP_ENTRE_LLAMADAS', 0)
    monkeypatch.setattr(ec.requests, 'get',
                        _fake_get({}, status=429))
    with pytest.raises(RuntimeError, match='Límite'):
        ec.obtener_eventos('2020-01-01', '2020-01-31', api_key='xxxx')


def test_pais_no_reconocido_se_mantiene_tal_cual(monkeypatch, tmp_path):
    monkeypatch.setattr(ec, 'get_base_data', lambda: str(tmp_path))
    monkeypatch.setattr(ec, '_SLEEP_ENTRE_LLAMADAS', 0)
    monkeypatch.setattr(ec.requests, 'get', _fake_get(_te_payload([
        ('2020-02-03 12:30:00', 'Atlantis', 'Evento', 'High'),
    ])))
    df = ec.obtener_eventos('2020-02-01', '2020-02-28', api_key='dummy')
    assert df['pais'].iloc[0] == 'Atlantis'


@pytest.mark.parametrize('nombre,esperado', [
    ('EURUSD', {'EUR', 'USD'}),
    ('EUR_USD', {'EUR', 'USD'}),
    ('XAUUSD', {'USD'}),
    ('US30', {'USD'}),
    ('GER40', {'EUR'}),
    ('ABCXYZ', None),
])
def test_monedas_de_instrumento(nombre, esperado):
    resultado = ec.monedas_de_instrumento(nombre)
    if esperado is None:
        assert resultado is None
    else:
        assert set(resultado) == esperado

from datetime import datetime, timezone

import pytest

from core.data_providers import economic_calendar as ec


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _fake_finnhub_payload(filas):
    """filas: lista de (fecha 'AAAA-MM-DD', pais, evento, impacto_finnhub)."""
    return {
        'economicCalendar': [
            {'time': f"{fecha} 12:30:00", 'country': pais, 'event': evento, 'impact': impacto}
            for fecha, pais, evento, impacto in filas
        ]
    }


def test_obtener_eventos_cachea_mes_ya_cerrado(monkeypatch, tmp_path):
    monkeypatch.setattr(ec, 'get_base_data', lambda: str(tmp_path))
    monkeypatch.setattr(ec, '_SLEEP_ENTRE_LLAMADAS', 0)
    llamadas = []

    def fake_get(url, params=None, timeout=None):
        llamadas.append(params)
        return _FakeResponse(_fake_finnhub_payload([
            ('2020-01-10', 'US', 'Non-Farm Payrolls', 'high'),
        ]))

    monkeypatch.setattr(ec.requests, 'get', fake_get)

    df1 = ec.obtener_eventos('2020-01-01', '2020-01-31', api_key='dummy')
    assert len(llamadas) == 1
    assert len(df1) == 1
    assert df1.iloc[0]['impacto'] == 'alto'

    # segunda llamada al mismo mes (cerrado) no debe volver a pedir a la API
    df2 = ec.obtener_eventos('2020-01-01', '2020-01-31', api_key='dummy')
    assert len(llamadas) == 1
    assert len(df2) == 1
    assert (tmp_path / 'Noticias' / 'economico_2020-01.json').exists()


def test_obtener_eventos_mes_en_curso_nunca_cachea(monkeypatch, tmp_path):
    monkeypatch.setattr(ec, 'get_base_data', lambda: str(tmp_path))
    monkeypatch.setattr(ec, '_SLEEP_ENTRE_LLAMADAS', 0)
    llamadas = []

    def fake_get(url, params=None, timeout=None):
        llamadas.append(params)
        return _FakeResponse(_fake_finnhub_payload([]))

    monkeypatch.setattr(ec.requests, 'get', fake_get)

    hoy = datetime.now(timezone.utc).date()
    ec.obtener_eventos(hoy.replace(day=1), hoy, api_key='dummy')
    ec.obtener_eventos(hoy.replace(day=1), hoy, api_key='dummy')
    assert len(llamadas) == 2   # el mes en curso se refresca siempre, nunca cachea


def test_obtener_eventos_filtra_por_impacto_minimo(monkeypatch, tmp_path):
    monkeypatch.setattr(ec, 'get_base_data', lambda: str(tmp_path))
    monkeypatch.setattr(ec, '_SLEEP_ENTRE_LLAMADAS', 0)

    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(_fake_finnhub_payload([
            ('2020-02-05', 'US', 'CPI', 'high'),
            ('2020-02-10', 'US', 'Retail Sales', 'medium'),
            ('2020-02-15', 'US', 'Housing Starts', 'low'),
        ]))

    monkeypatch.setattr(ec.requests, 'get', fake_get)

    df = ec.obtener_eventos('2020-02-01', '2020-02-28', api_key='dummy',
                             impacto_minimo='medio')
    assert set(df['impacto']) == {'alto', 'medio'}


def test_obtener_eventos_sin_api_key_lanza_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, 'get_base_data', lambda: str(tmp_path))
    with pytest.raises(RuntimeError):
        ec.obtener_eventos('2020-01-01', '2020-01-31', api_key='')


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

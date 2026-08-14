"""Comprobaciones de errores y consultas auxiliares contra QuestDB.

El wire protocol PostgreSQL de QuestDB no devuelve SQLSTATE de forma fiable,
así que la clasificación combina código y texto, y la comprobación de
existencia de tabla usa HTTP /exec (la vía nativa).
"""
import pytest

import core.questdb_errors as qe


class _PgError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.pgcode = code


def test_codigo_postgres_se_reconoce_directo_y_envuelto():
    directo = _PgError('42P01')
    envuelto = Exception('pandas wrapper')
    envuelto.orig = _PgError('42703')
    assert qe.pgcode(directo) == '42P01'
    assert qe.pgcode(envuelto) == '42703'


def test_tabla_inexistente_se_reconoce_por_codigo():
    assert qe.es_tabla_inexistente(_PgError('42P01'))
    assert not qe.es_tabla_inexistente(_PgError('42703'))


def test_tabla_inexistente_se_reconoce_por_mensaje_de_questdb():
    # QuestDB puede responder por PG wire sin SQLSTATE: el texto es lo que
    # queda. Mismo caso para el error HTTP de /exec.
    assert qe.es_tabla_inexistente(
        RuntimeError('table does not exist [name=x]'))
    assert qe.es_tabla_inexistente(
        RuntimeError('la tabla no existe'))
    assert not qe.es_tabla_inexistente(RuntimeError('invalid password'))


def test_columna_inexistente_se_reconoce_por_codigo_y_mensaje():
    assert qe.es_columna_inexistente(_PgError('42703'))
    assert not qe.es_columna_inexistente(_PgError('42P01'))
    assert qe.es_columna_inexistente(
        RuntimeError('column spread does not exist'))
    assert qe.es_columna_inexistente(
        RuntimeError('Invalid column: spread'))
    assert not qe.es_columna_inexistente(RuntimeError('connection refused'))


class _Resp:
    def __init__(self, status_code, text, json_datos=None):
        self.status_code = status_code
        self.text = text
        self._json = json_datos

    def json(self):
        if self._json is None:
            raise ValueError('not json')
        return self._json


def _monkeypatch_get(monkeypatch, resp):
    llamado = {'params': None}

    def fake_get(url, params=None, timeout=None):
        llamado['url'] = url
        llamado['params'] = params
        llamado['timeout'] = timeout
        return resp

    monkeypatch.setattr(qe.requests, 'get', fake_get)
    return llamado


def test_filas_en_tabla_200_devuelve_el_conteo(monkeypatch):
    llamado = _monkeypatch_get(monkeypatch, _Resp(
        200, '{"dataset":[[12]]}', {'dataset': [[12]]}))
    assert qe.filas_en_tabla('localhost', 19000, 'btc_candles_1h') == 12
    assert 'count()' in llamado['params']['query']
    assert 'btc_candles_1h' in llamado['params']['query']


def test_filas_en_tabla_tabla_inexistente_devuelve_cero(monkeypatch):
    _monkeypatch_get(monkeypatch, _Resp(
        400, '{"error":"table does not exist [name=btc_candles_1h]"}'))
    assert qe.filas_en_tabla('localhost', 19000, 'btc_candles_1h') == 0


def test_filas_en_tabla_conexion_rota_lanza(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        raise qe.requests.exceptions.ConnectionError('connection refused')

    monkeypatch.setattr(qe.requests, 'get', fake_get)
    with pytest.raises(RuntimeError, match='No se pudo consultar QuestDB'):
        qe.filas_en_tabla('localhost', 19000, 'btc_candles_1h')


def test_filas_en_tabla_http_500_lanza(monkeypatch):
    _monkeypatch_get(monkeypatch, _Resp(500, 'boom'))
    with pytest.raises(RuntimeError, match='HTTP 500'):
        qe.filas_en_tabla('localhost', 19000, 'btc_candles_1h')

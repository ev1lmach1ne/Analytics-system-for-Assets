import json

from core.config import PRESETS_FRICCION, tipo_activo_de_csv


def _crear_csv(tmp_path, nombre, carpeta=None, meta=None):
    d = tmp_path / carpeta if carpeta else tmp_path
    d.mkdir(parents=True, exist_ok=True)
    csv_path = d / nombre
    csv_path.write_text('open,high,low,close\n')
    if meta is not None:
        (d / (nombre + '.meta.json')).write_text(json.dumps(meta))
    return str(csv_path)


def test_presets_friccion_tiene_las_4_clases():
    assert set(PRESETS_FRICCION) == {'CRYPTO', 'STOCK', 'FUTURO', 'FOREX'}
    for datos in PRESETS_FRICCION.values():
        assert 'slippage_pct' in datos and 'comision_pct' in datos


def test_tipo_activo_desde_meta_json(tmp_path):
    p = _crear_csv(tmp_path, 'x_1h_limpiado.csv', meta={'activo': 'CRYPTO'})
    assert tipo_activo_de_csv(p) == 'CRYPTO'


def test_tipo_activo_meta_json_corrupto_cae_a_nombre(tmp_path):
    p = _crear_csv(tmp_path, 'btc_1h_limpiado.csv')
    (tmp_path / 'btc_1h_limpiado.csv.meta.json').write_text('{not valid json')
    assert tipo_activo_de_csv(p) == 'CRYPTO'


def test_tipo_activo_meta_json_sin_clave_activo_cae_a_nombre(tmp_path):
    p = _crear_csv(tmp_path, 'btc_1h_limpiado.csv', meta={'tf': '1h'})
    assert tipo_activo_de_csv(p) == 'CRYPTO'


def test_tipo_activo_fallback_crypto(tmp_path):
    p = _crear_csv(tmp_path, 'btc_1h_limpio.csv')
    assert tipo_activo_de_csv(p) == 'CRYPTO'


def test_tipo_activo_fallback_forex(tmp_path):
    p = _crear_csv(tmp_path, 'XAUUSD_H1_202003130100_202506122200_preparado.csv')
    assert tipo_activo_de_csv(p) == 'FOREX'


def test_tipo_activo_fallback_futuro(tmp_path):
    p = _crear_csv(tmp_path, 'us500_1d.csv')
    assert tipo_activo_de_csv(p) == 'FUTURO'


def test_tipo_activo_fallback_por_carpeta(tmp_path):
    p = _crear_csv(tmp_path, 'datos_1h.csv', carpeta='crypto_btc')
    assert tipo_activo_de_csv(p) == 'CRYPTO'


def test_tipo_activo_desconocido(tmp_path):
    p = _crear_csv(tmp_path, 'datos_1h.csv')
    assert tipo_activo_de_csv(p) is None


def test_tipo_activo_meta_json_con_activo_desconocido_cae_a_nombre(tmp_path):
    p = _crear_csv(tmp_path, 'btc_1h_limpiado.csv', meta={'activo': 'RARO'})
    assert tipo_activo_de_csv(p) == 'CRYPTO'

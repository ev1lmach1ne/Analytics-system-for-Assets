"""
tests/test_favorito_serializable.py
Guardar un favorito no puede dejar un archivo a medias.

La config que el hilo de backtest mete en el payload lleva, además de la
cuenta, las máscaras precomputadas de condiciones: listas de arrays numpy.
json.dump escribe según recorre, así que al toparse con la primera lanzaba
TypeError y dejaba el favorito.json cortado por la mitad. Como @_no_crash se
traga la excepción y _leer_favoritos descarta la carpeta al fallar el
json.load, el usuario solo veía que el favorito no aparecía en la lista.
"""
import json
import os

import numpy as np
import pytest

pytest.importorskip('PyQt6.QtWidgets')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import gui.widgets.tab_backtest as tb   # noqa: E402


def _config_de_motor(n=50):
    """Config tal y como la arma _BacktestThread.run: escalares de cuenta,
    config_por_setup y las seis listas de máscaras numpy."""
    mascara = [np.ones(n, dtype=bool)]
    return {
        'capital_inicial': 10000.0,
        'comision_pct': 0.0003,
        'slippage_pct': 0.0002,
        'config_por_setup': {0: {'riesgo_pct': 0.004, 'stop_atr': 2.0}},
        'parciales_masks_long': [mascara],
        'parciales_masks_short': [mascara],
        'tramos_masks_long': [mascara],
        'tramos_masks_short': [mascara],
        'mecanismos_masks_long': [mascara],
        'mecanismos_masks_short': [mascara],
    }


# ══════════════ filtrado de la config ══════════════

def test_las_mascaras_numpy_no_llegan_al_json():
    limpio = tb._config_serializable(_config_de_motor())
    for clave in limpio:
        assert 'masks' not in clave, clave
    json.dumps(limpio)   # tiene que poder serializarse entero


def test_la_config_conserva_los_escalares_que_la_recarga_necesita():
    """_cargar_favorito_nombre solo lee capital/comisión/slippage: si el
    filtro se los llevara, el favorito se recargaría con otra cuenta."""
    limpio = tb._config_serializable(_config_de_motor())
    assert limpio['capital_inicial'] == 10000.0
    assert limpio['comision_pct'] == 0.0003
    assert limpio['slippage_pct'] == 0.0002


def test_el_filtro_no_nombra_las_claves_conocidas():
    """Se descarta por serializabilidad, no por lista de nombres: una máscara
    nueva con otro nombre tiene que caer igual, sin tocar este código."""
    config = {'capital_inicial': 1.0, 'mascara_inventada': [np.zeros(3)]}
    assert tb._config_serializable(config) == {'capital_inicial': 1.0}


def test_una_config_vacia_o_ausente_no_rompe():
    assert tb._config_serializable({}) == {}
    assert tb._config_serializable(None) == {}


# ══════════════ escritura atómica ══════════════

def test_el_favorito_se_relee_entero(tmp_path):
    """El caso que fallaba: el mismo dict que arma _guardar_favorito, escrito
    y vuelto a leer sin perder nada."""
    ruta = tmp_path / 'favorito.json'
    datos = {'nombre': 'zc rsi', 'csv': 'D:/datos/zc_1d.csv', 'tf': '1d',
             'setups': [{'nombre': 'RSI', 'plantilla': 'RSI',
                         'params': {'periodo': 9}}],
             'config': tb._config_serializable(_config_de_motor())}
    tb._guardar_json_atomico(str(ruta), datos)

    with open(ruta, encoding='utf-8') as f:
        releido = json.load(f)
    assert releido['nombre'] == 'zc rsi'
    assert releido['setups'][0]['params']['periodo'] == 9
    assert releido['config']['capital_inicial'] == 10000.0


def test_un_fallo_de_escritura_no_destruye_el_favorito_anterior(tmp_path):
    """La garantía de la escritura atómica: antes, un error a mitad dejaba el
    archivo truncado e ilegible para siempre."""
    ruta = tmp_path / 'favorito.json'
    tb._guardar_json_atomico(str(ruta), {'nombre': 'bueno'})

    with pytest.raises(TypeError):
        tb._guardar_json_atomico(str(ruta), {'roto': np.zeros(3)})

    with open(ruta, encoding='utf-8') as f:
        assert json.load(f) == {'nombre': 'bueno'}


def test_no_queda_ningun_temporal_tras_un_guardado_correcto(tmp_path):
    ruta = tmp_path / 'favorito.json'
    tb._guardar_json_atomico(str(ruta), {'nombre': 'x'})
    assert [p.name for p in tmp_path.iterdir()] == ['favorito.json']


def test_guardar_dos_veces_sobrescribe_sin_dejar_restos(tmp_path):
    ruta = tmp_path / 'favorito.json'
    tb._guardar_json_atomico(str(ruta), {'nombre': 'v1'})
    tb._guardar_json_atomico(str(ruta), {'nombre': 'v2'})
    with open(ruta, encoding='utf-8') as f:
        assert json.load(f)['nombre'] == 'v2'
    assert [p.name for p in tmp_path.iterdir()] == ['favorito.json']


# ══════════════ el favorito corrupto que ya existe en disco ══════════════

def test_un_favorito_truncado_no_tumba_la_lectura(tmp_path, monkeypatch):
    """_leer_favoritos ya salta el JSONDecodeError, y por eso el síntoma era
    silencioso. Se fija el comportamiento para que el arreglo no lo cambie:
    los favoritos sanos tienen que seguir cargándose aunque quede alguno
    corrupto de antes en la carpeta."""
    monkeypatch.setattr(tb, 'FAVORITOS_DIR', str(tmp_path))

    bueno = tmp_path / 'sano'
    bueno.mkdir()
    tb._guardar_json_atomico(str(bueno / 'favorito.json'), {'nombre': 'sano'})

    roto = tmp_path / 'truncado'
    roto.mkdir()
    (roto / 'favorito.json').write_text(
        '{\n  "nombre": "roto",\n  "config": {\n    "parciales_masks_long": [\n      [',
        encoding='utf-8')

    # _leer_favoritos no toca self: se llama sin construir el widget entero
    # (misma regla que tests/test_resultados_periodo.py — probar el helper,
    # no la GUI)
    favoritos = tb.OptimizadorWidget._leer_favoritos(None)
    assert list(favoritos) == ['sano']

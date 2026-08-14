"""Persistencia del Rf: sidecar al importar y auto-relleno en el Limpiador.

El Rf se recuerda en dos sitios: un sidecar <csv>.import_info junto al archivo
de origen (que Importar restaura al re-seleccionarlo) y el <csv>.meta.json del
archivo limpiado (que el Limpiador lee para rellenar su campo Rf).
"""
import json
import os

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PyQt6.QtWidgets')

from PyQt6.QtWidgets import QApplication, QLabel, QLineEdit  # noqa: E402

from gui.widgets.tab_importar import _guardar_rf_sidecar  # noqa: E402
import gui.widgets.tab_limpiados as tl                      # noqa: E402

class _Emit:
    def emit(self, *a, **k):
        pass


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def test_guardar_rf_sidecar_escribe_el_valor(tmp_path):
    csv_path = str(tmp_path / 'xauusd_1h.csv')
    _guardar_rf_sidecar(csv_path, 4.5)
    with open(csv_path + '.import_info', encoding='utf-8') as f:
        assert json.load(f) == {'rf_rate': 4.5}


def test_guardar_rf_sidecar_sin_rf_no_escribe_nada(tmp_path):
    csv_path = str(tmp_path / 'x.csv')
    _guardar_rf_sidecar(csv_path, None)
    assert not os.path.exists(csv_path + '.import_info')


def test_save_meta_de_importar_crea_la_carpeta_destino(tmp_path):
    import gui.widgets.tab_importar as ti
    limp = tmp_path / 'Limpiados'
    ti.LIMPIADOS_DIR = str(limp)
    w = ti.TabImportar.__new__(ti.TabImportar)
    w._config = {'nombre': 'btc', 'tf': '1h', 'activo': 'CRYPTO',
                 'categoria': 'CRYPTO', 'rf_rate': 2.5}
    w._save_meta()
    meta_path = limp / 'CRYPTO' / 'btc' / 'btc_1h_limpiado.csv.meta.json'
    assert meta_path.exists()
    assert json.loads(meta_path.read_text(encoding='utf-8'))['rf_rate'] == 2.5


def test_persistir_meta_rf_crea_y_actualiza_conservando_claves(tmp_path):
    csv_path = str(tmp_path / 'btc_1h_limpiado.csv')
    open(csv_path, 'w').write('timestamp,open\n')
    tl._persistir_meta_rf(csv_path, 'btc', '1h', 'CRYPTO', 3.0)
    meta = json.loads(open(csv_path + '.meta.json', encoding='utf-8').read())
    assert meta['rf_rate'] == 3.0

    # segunda llamada con otro Rf: conserva lo anterior y actualiza el Rf
    tl._persistir_meta_rf(csv_path, 'btc', '1h', 'CRYPTO', 1.5)
    meta = json.loads(open(csv_path + '.meta.json', encoding='utf-8').read())
    assert meta['rf_rate'] == 1.5
    assert meta['nombre'] == 'btc'


def test_registro_rf_guardar_y_leer(tmp_path):
    from core.rf_registry import guardar_rf, leer_rf
    limp = str(tmp_path / 'Limpiados')
    csv_path = os.path.join(limp, 'CRYPTO', 'btc', 'btc_1h_limpiado.csv')
    assert leer_rf(limp, csv_path) is None
    guardar_rf(limp, csv_path, 2.5)
    assert leer_rf(limp, csv_path) == 2.5
    guardar_rf(limp, csv_path, 1.0)
    assert leer_rf(limp, csv_path) == 1.0
    assert leer_rf(limp, os.path.join(limp, 'otro', 'x.csv')) is None


def test_read_meta_cae_al_registro_cuando_el_meta_no_tiene_rf(app, tmp_path):
    from core.rf_registry import guardar_rf
    limp = str(tmp_path / 'Limpiados')
    os.makedirs(os.path.join(limp, 'CRYPTO', 'btc'), exist_ok=True)
    csv_path = os.path.join(limp, 'CRYPTO', 'btc', 'btc_1h_limpiado.csv')
    open(csv_path, 'w').write('timestamp,open\n')
    # meta legacy SIN clave rf_rate
    with open(csv_path + '.meta.json', 'w', encoding='utf-8') as f:
        json.dump({'nombre': 'btc', 'tf': '1h', 'activo': 'CRYPTO'}, f)
    guardar_rf(limp, csv_path, 3.5)

    w = tl.TabLimpiados.__new__(tl.TabLimpiados)
    w.rf_input = QLineEdit()
    w.rf_input.setText('0')
    w.lbl_info = QLabel('')
    w.file_selected = _Emit()
    w._rf_rate = 0.0
    w._limpiados_dir = limp
    w._read_meta(csv_path)
    assert w.rf_input.text() == '3.5'
    assert w._rf_rate == 3.5


def test_read_meta_prefiere_el_meta_sobre_el_registro(app, tmp_path):
    from core.rf_registry import guardar_rf
    limp = str(tmp_path / 'Limpiados')
    os.makedirs(os.path.join(limp, 'CRYPTO', 'btc'), exist_ok=True)
    csv_path = os.path.join(limp, 'CRYPTO', 'btc', 'btc_1h_limpiado.csv')
    open(csv_path, 'w').write('timestamp,open\n')
    with open(csv_path + '.meta.json', 'w', encoding='utf-8') as f:
        json.dump({'nombre': 'btc', 'tf': '1h', 'activo': 'CRYPTO',
                   'rf_rate': 4.5}, f)
    guardar_rf(limp, csv_path, 9.9)

    w = tl.TabLimpiados.__new__(tl.TabLimpiados)
    w.rf_input = QLineEdit()
    w.rf_input.setText('0')
    w.lbl_info = QLabel('')
    w.file_selected = _Emit()
    w._rf_rate = 0.0
    w._limpiados_dir = limp
    w._read_meta(csv_path)
    assert w.rf_input.text() == '4.5'


def _limpiados_stub(tmp_path, meta):
    w = tl.TabLimpiados.__new__(tl.TabLimpiados)
    w.rf_input = QLineEdit()
    w.rf_input.setText('7')
    w.lbl_info = QLabel('')
    w.file_selected = _Emit()
    w._rf_rate = 0.0
    w._limpiados_dir = str(tmp_path)
    p = str(tmp_path / 'btc_1h_limpiado.csv')
    open(p, 'w').write('timestamp,open\n')
    if meta is not None:
        with open(p + '.meta.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f)
    w._read_meta(p)
    return w


def test_read_meta_rellena_con_rf_no_nulo(app, tmp_path):
    w = _limpiados_stub(tmp_path, {'nombre': 'btc', 'tf': '1h',
                                   'activo': 'CRYPTO', 'rf_rate': 4.5})
    assert w.rf_input.text() == '4.5'
    assert w._rf_rate == 4.5


def test_read_meta_rellena_tambien_con_rf_cero(app, tmp_path):
    w = _limpiados_stub(tmp_path, {'nombre': 'btc', 'tf': '1h',
                                   'activo': 'CRYPTO', 'rf_rate': 0.0})
    assert w.rf_input.text() == '0.0'
    assert w._rf_rate == 0.0


def test_read_meta_sin_clave_rf_no_pisa_el_campo(app, tmp_path):
    w = _limpiados_stub(tmp_path, {'nombre': 'btc', 'tf': '1h',
                                   'activo': 'CRYPTO'})
    assert w.rf_input.text() == '7'


def test_read_meta_sin_archivo_meta_no_pisa_el_campo(app, tmp_path):
    w = _limpiados_stub(tmp_path, None)
    assert w.rf_input.text() == '7'


def test_read_meta_con_meta_corrupto_no_rompe_ni_pisa(app, tmp_path):
    p = str(tmp_path / 'btc_1h_limpiado.csv')
    open(p, 'w').write('timestamp,open\n')
    with open(p + '.meta.json', 'w', encoding='utf-8') as f:
        f.write('{corrupto')
    w = tl.TabLimpiados.__new__(tl.TabLimpiados)
    w.rf_input = QLineEdit()
    w.rf_input.setText('7')
    w.lbl_info = QLabel('')
    w.file_selected = _Emit()
    w._rf_rate = 0.0
    w._limpiados_dir = str(tmp_path)
    w._read_meta(p)
    assert w.rf_input.text() == '7'
    assert w._rf_rate == 0.0
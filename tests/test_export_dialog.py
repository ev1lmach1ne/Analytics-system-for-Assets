"""
tests/test_export_dialog.py
El diálogo de exportación.

Lo que más importa aquí no es que la ventana se dibuje, sino que **cancelar no
deje ningún archivo escrito** y que el usuario vea qué se pierde ANTES de que
nada toque el disco.
"""
import os

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PyQt6.QtWidgets')

from PyQt6.QtWidgets import QApplication, QDialog   # noqa: E402

import core.codegen as codegen                       # noqa: E402
from core.codegen import fidelidad                   # noqa: E402
from core.strategies import _filtros_por_defecto     # noqa: E402
from gui.dialogs.export_codigo_dialog import (       # noqa: E402
    DialogoConfirmarPerdida, DialogoExportarCodigo,
)

META = {'sistema': 'zc rsi', 'activo': 'ZCMAIZ', 'tf': '1d'}
CUENTA = {'capital_inicial': 10000.0, 'comision_pct': 0.0003,
          'slippage_pct': 0.0002}


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def _setup(filtros=None, **extra):
    s = {'nombre': 'RSI', 'plantilla': 'RSI', 'riesgo_pct': 0.004,
         'stop_atr': 2.0, 'filtros': filtros or _filtros_por_defecto()}
    s.update(extra)
    return s


def _con_noticias():
    f = _filtros_por_defecto()
    f['noticias'] = dict(f['noticias'], activo=True)
    return _setup(f)


# ══════════════ la cuadrícula ══════════════

def test_la_cuadricula_lista_todas_las_plataformas(app):
    """Las que aún no se pueden generar se enseñan deshabilitadas a
    propósito: es la forma honesta de decir hasta dónde llega la función."""
    dlg = DialogoExportarCodigo([_setup()], CUENTA, META)
    assert len(dlg._tarjetas) == len(codegen.PLATAFORMAS)
    disponibles = [c for c, t in dlg._tarjetas.items() if t.disponible]
    assert sorted(disponibles) == sorted(codegen.plataformas_disponibles())
    for clave, tarjeta in dlg._tarjetas.items():
        assert tarjeta.chk.isEnabled() == tarjeta.disponible, clave
    dlg.deleteLater()


def test_las_plataformas_no_disponibles_no_se_pueden_marcar(app):
    dlg = DialogoExportarCodigo([_setup()], CUENTA, META)
    tarjeta = dlg._tarjetas['ninjatrader']
    tarjeta.chk.setChecked(True)          # la GUI no debería permitirlo
    assert not tarjeta.marcada()
    assert 'ninjatrader' not in dlg.plataformas_marcadas()
    dlg.deleteLater()


def test_las_disponibles_vienen_marcadas_de_inicio(app):
    """Es lo que casi siempre se quiere, y deja el informe de fidelidad
    visible desde el primer momento en vez de con un panel vacío."""
    dlg = DialogoExportarCodigo([_setup()], CUENTA, META)
    assert sorted(dlg.plataformas_marcadas()) == sorted(
        codegen.plataformas_disponibles())
    dlg.deleteLater()


def test_sin_ninguna_marcada_no_se_puede_exportar(app):
    dlg = DialogoExportarCodigo([_setup()], CUENTA, META)
    for tarjeta in dlg._tarjetas.values():
        tarjeta.chk.setChecked(False)
    assert not dlg.btn_exportar.isEnabled()
    dlg.deleteLater()


# ══════════════ el distintivo de fidelidad ══════════════

def test_un_sistema_con_noticias_marca_tradingview_en_rojo(app):
    """El caso que motiva la función: Pine no tiene calendario económico, y
    eso hay que verlo ANTES de elegir la plataforma."""
    dlg = DialogoExportarCodigo([_con_noticias()], CUENTA, META)
    assert dlg._tarjetas['tradingview'].lbl_badge.text() == \
        fidelidad.ICONOS[fidelidad.NIVEL_OMITIDO]
    # en MQL5 el filtro sí existe (calendario nativo): solo se aproxima
    assert dlg._tarjetas['mt5'].lbl_badge.text() == \
        fidelidad.ICONOS[fidelidad.NIVEL_APROXIMADO]
    dlg.deleteLater()


def test_el_distintivo_se_calcula_aunque_la_plataforma_no_este_marcada(app):
    """Sirve justo para decidir cuál marcar: si solo se pintara la marcada,
    llegaría tarde."""
    dlg = DialogoExportarCodigo([_con_noticias()], CUENTA, META)
    dlg._tarjetas['tradingview'].chk.setChecked(False)
    assert dlg._tarjetas['tradingview'].lbl_badge.text() != ''
    dlg.deleteLater()


def test_el_panel_nombra_el_setup_y_la_consecuencia(app):
    dlg = DialogoExportarCodigo([_con_noticias()], CUENTA, META)
    texto = dlg._panel.text()
    assert 'noticias' in texto.lower()
    assert 'RSI' in texto
    dlg.deleteLater()


def test_un_setup_bloqueado_se_anuncia_en_el_panel(app):
    """Los patrones de vela aún no se emiten: el setup no genera archivo y
    hay que decirlo antes, no después."""
    setup = {'nombre': 'patrones', 'plantilla': 'Patrones de velas',
             'params': {'patrones': ['Martillo'], 'lag_salida': 5},
             'filtros': _filtros_por_defecto()}
    dlg = DialogoExportarCodigo([setup], CUENTA, META)
    assert 'no se exportará' in dlg._panel.text()
    dlg.deleteLater()


# ══════════════ la confirmación ══════════════

def test_con_omisiones_hay_que_marcar_la_casilla_para_continuar(app):
    """Si hay ❌, el código NO reproduce el sistema backtesteado, y eso se
    acepta explícitamente o no se acepta."""
    analisis = codegen.analizar_sistema([_con_noticias()], CUENTA,
                                        ['tradingview'])
    dlg = DialogoConfirmarPerdida(analisis)
    assert dlg.chk_entiendo.isVisible() or dlg._hay_omisiones
    assert not dlg.btn_seguir.isEnabled()
    dlg.chk_entiendo.setChecked(True)
    assert dlg.btn_seguir.isEnabled()
    dlg.deleteLater()


def test_sin_omisiones_basta_con_pulsar(app):
    """Lo aproximado avisa pero no bloquea: pedir conformidad por todo
    enseñaría al usuario a marcar la casilla sin leerla."""
    analisis = codegen.analizar_sistema([_setup()], CUENTA, ['tradingview'])
    dlg = DialogoConfirmarPerdida(analisis)
    assert not dlg._hay_omisiones
    assert dlg.btn_seguir.isEnabled()
    dlg.deleteLater()


def test_la_confirmacion_nombra_la_plataforma_y_la_consecuencia(app):
    analisis = codegen.analizar_sistema([_con_noticias()], CUENTA,
                                        ['tradingview'])
    dlg = DialogoConfirmarPerdida(analisis)
    html = dlg._html(analisis)
    assert 'TradingView' in html
    assert 'calendario' in html.lower()
    dlg.deleteLater()


# ══════════════ lo que de verdad importa: cancelar no escribe ══════════════

def test_cancelar_la_confirmacion_no_deja_ningun_archivo(app, tmp_path,
                                                         monkeypatch):
    """La garantía del diálogo: nada toca el disco hasta que el usuario
    acepta."""
    dlg = DialogoExportarCodigo([_con_noticias()], CUENTA, META)
    dlg.txt_destino.setText(str(tmp_path))
    monkeypatch.setattr(DialogoConfirmarPerdida, 'exec',
                        lambda self: QDialog.DialogCode.Rejected)
    dlg._exportar()
    assert dlg.resultado is None
    assert list(tmp_path.iterdir()) == []
    dlg.deleteLater()


def test_aceptar_la_confirmacion_escribe_el_arbol(app, tmp_path, monkeypatch):
    dlg = DialogoExportarCodigo([_setup()], CUENTA, META)
    dlg.txt_destino.setText(str(tmp_path))
    dlg.txt_nombre.setText('zc rsi')
    monkeypatch.setattr(DialogoConfirmarPerdida, 'exec',
                        lambda self: QDialog.DialogCode.Accepted)
    dlg._exportar()
    assert dlg.resultado is not None
    escritos = dlg.resultado['archivos']
    assert 'NOTAS_DE_FIDELIDAD.md' in escritos
    assert any(a.endswith('.pine') for a in escritos)
    assert any(a.endswith('.mq5') for a in escritos)
    assert os.path.isdir(tmp_path / 'zc_rsi')
    dlg.deleteLater()


# ══════════════ copiar al portapapeles ══════════════

def test_copiar_pine_deja_el_codigo_listo_para_pegar(app, monkeypatch):
    """TradingView no importa archivos: el flujo real es portapapeles. Que el
    usuario tenga que abrir el .pine con un editor para copiarlo es un rodeo
    que este botón elimina."""
    monkeypatch.setattr(
        'gui.dialogs.export_codigo_dialog.informacion',
        lambda *a, **k: None)
    dlg = DialogoExportarCodigo([_setup()], CUENTA, META)
    dlg._copiar_pine()
    texto = QApplication.clipboard().text()
    assert texto.splitlines()[0] == '//@version=6'
    assert 'strategy(' in texto
    assert dlg.btn_copiar.text().startswith('✓')
    dlg.deleteLater()


def test_copiar_pine_no_escribe_nada_en_disco(app, tmp_path, monkeypatch):
    monkeypatch.setattr(
        'gui.dialogs.export_codigo_dialog.informacion',
        lambda *a, **k: None)
    dlg = DialogoExportarCodigo([_setup()], CUENTA, META)
    dlg.txt_destino.setText(str(tmp_path))
    dlg._copiar_pine()
    assert list(tmp_path.iterdir()) == []
    assert dlg.resultado is None
    dlg.deleteLater()


def test_copiar_pine_avisa_si_hay_varios_setups(app, monkeypatch):
    """Cada setup es un script independiente: concatenarlos daría un archivo
    que no compila, así que se copia uno y se dice cuál."""
    mensajes = []
    monkeypatch.setattr(
        'gui.dialogs.export_codigo_dialog.informacion',
        lambda *a, **k: mensajes.append(a[2]))
    dlg = DialogoExportarCodigo([_setup(), _setup()], CUENTA, META)
    dlg._copiar_pine()
    assert mensajes and 'S0' in mensajes[0]
    dlg.deleteLater()


def test_copiar_pine_avisa_cuando_no_hay_nada_generable(app, monkeypatch):
    avisos = []
    monkeypatch.setattr(
        'gui.dialogs.export_codigo_dialog.aviso',
        lambda *a, **k: avisos.append(a[2]))
    setup = {'nombre': 'patrones', 'plantilla': 'Patrones de velas',
             'params': {'patrones': ['Martillo'], 'lag_salida': 5},
             'filtros': _filtros_por_defecto()}
    dlg = DialogoExportarCodigo([setup], CUENTA, META)
    dlg._copiar_pine()
    assert avisos and 'señal' in avisos[0]
    dlg.deleteLater()


def test_codigo_de_setup_no_toca_el_disco(tmp_path):
    """El generador sin efectos secundarios que usa el botón."""
    codigo = codegen.codigo_de_setup([_setup()], CUENTA, 'tradingview', 0, META)
    assert codigo.startswith('//@version=6')
    assert list(tmp_path.iterdir()) == []


def test_codigo_de_setup_se_niega_con_un_setup_bloqueado():
    setup = {'nombre': 'patrones', 'plantilla': 'Patrones de velas',
             'params': {'patrones': ['Martillo'], 'lag_salida': 5},
             'filtros': _filtros_por_defecto()}
    with pytest.raises(ValueError, match='señal'):
        codegen.codigo_de_setup([setup], CUENTA, 'tradingview', 0, META)


def test_sin_nombre_no_se_exporta(app, tmp_path, monkeypatch):
    dlg = DialogoExportarCodigo([_setup()], CUENTA, META)
    dlg.txt_destino.setText(str(tmp_path))
    dlg.txt_nombre.setText('   ')
    monkeypatch.setattr(
        'gui.dialogs.export_codigo_dialog.aviso',
        lambda *a, **k: None)
    dlg._exportar()
    assert dlg.resultado is None
    assert list(tmp_path.iterdir()) == []
    dlg.deleteLater()

"""Paneles flotantes sin marco de Windows y popup sin barras de scroll.

La ventana de QuestDB y la lista completa de trades usan diálogos sin marco
nativo con cabecera propia; el popup de ayuda muestra todo el texto de golpe
(sin barras de desplazamiento).
"""
import os

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PyQt6.QtWidgets')

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QScrollArea, QPushButton, QLabel  # noqa: E402

from gui.widgets.plot_common import (  # noqa: E402
    _PopupAyuda, panel_flotante_dialog, PanelFlotanteDialog,
)
from gui.questdb_bootstrap import _DialogoBootstrap, _SpinnerCircular  # noqa: E402


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def test_popup_sin_barras_de_scroll(app):
    popup = _PopupAyuda([('Lógica', 'texto de lógica para la primera pestaña'),
                         ('Uso', 'texto de uso para la segunda pestaña')])
    assert not popup.findChildren(QScrollArea), \
        "el popup no debe tener barras de desplazamiento"
    assert popup.findChildren(QLabel)


def test_popup_se_anima_al_mostrar(app):
    popup = _PopupAyuda([('Ayuda', 'explicación breve')])
    assert popup.windowOpacity() == 1.0
    popup.show()
    assert hasattr(popup, '_anim_fade')
    assert hasattr(popup, '_anim_slide')


def test_dialogo_flotante_sin_marco_con_boton_cerrar(app):
    dlg, lay, lbl_sub, halo = panel_flotante_dialog(
        'Título de prueba', 400, alto=300, subtitulo='subtítulo',
        boton_cerrar=True)
    assert dlg.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert [b for b in dlg.findChildren(QPushButton) if b.text() == '✕']
    assert lbl_sub.text() == 'subtítulo'
    dlg.close()


def test_dialogo_flotante_sin_cerrar_si_no_se_pide(app):
    dlg, lay, lbl_sub, halo = panel_flotante_dialog(
        'Título', 300, alto=120, boton_cerrar=False)
    assert not [b for b in dlg.findChildren(QPushButton) if b.text() == '✕']
    dlg.close()


def test_bootstrap_sin_marco_con_spinner_y_sin_cierre(app):
    dlg = _DialogoBootstrap()
    assert isinstance(dlg, PanelFlotanteDialog)
    assert dlg.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert dlg.findChildren(_SpinnerCircular), "debe mostrarse el spinner"
    assert not [b for b in dlg.findChildren(QPushButton) if b.text() == '✕'], \
        "el arranque no se puede cancelar"
    dlg.set_progreso('Descargando motor…', 45)
    assert '45' in dlg.lbl_pct.text()
    dlg.set_completado()
    assert '✓' in dlg.lbl.text()

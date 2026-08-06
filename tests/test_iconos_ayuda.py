"""Los iconos «?» de ayuda responden todos al mismo gesto.

La app tenía dos badges «?» pixel a pixel idénticos con comportamientos
distintos: los de Análisis y los gráficos abrían un panel al hacer clic, y los
del Backtester solo enseñaban un tooltip al pasar el ratón. Clicar uno de estos
últimos no hacía nada y se leía como un icono roto.

Lo que se fija aquí es que cualquier icono de ayuda haga algo al clicarlo.
"""
import os

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('PyQt6.QtWidgets')
from PyQt6.QtWidgets import QApplication, QLabel, QTabWidget   # noqa: E402

from gui.widgets.plot_common import icono_ayuda, icono_ayuda_texto  # noqa: E402
from gui.widgets.tab_backtest import _icono_ayuda                   # noqa: E402


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


_VIVOS = []   # el popup es hijo del icono: si el icono se recolecta, Qt
              # destruye el objeto C++ del popup y consultarlo revienta


def _clicar(icono):
    _VIVOS.append(icono)
    icono.mousePressEvent(None)
    return icono._popup


def test_el_icono_del_backtester_abre_panel_al_clicarlo(app):
    icono = _icono_ayuda('Qué hace esta sección')
    popup = _clicar(icono)
    assert popup.isVisible()
    textos = [w.text() for w in popup.findChildren(QLabel)]
    assert 'Qué hace esta sección' in textos


def test_el_icono_del_backtester_conserva_el_tooltip(app):
    """El popup se suma al tooltip, no lo sustituye: quien ya estaba
    acostumbrado a pasar el ratón por encima sigue viendo la explicación."""
    assert _icono_ayuda('Texto de ayuda').toolTip() == 'Texto de ayuda'


def test_una_sola_seccion_no_monta_barra_de_pestanas(app):
    """Una pestaña solitaria se lee como una interfaz a medio hacer."""
    popup = _clicar(icono_ayuda_texto('Explicación única'))
    assert not popup.findChildren(QTabWidget)


def test_el_icono_de_cuatro_secciones_sigue_con_sus_pestanas(app):
    popup = _clicar(icono_ayuda('lógica', 'significado', 'uso', 'resultados'))
    tabs = popup.findChildren(QTabWidget)
    assert len(tabs) == 1
    assert [tabs[0].tabText(i) for i in range(tabs[0].count())] == \
        ['Lógica', 'Significado', 'Uso', 'Resultados']

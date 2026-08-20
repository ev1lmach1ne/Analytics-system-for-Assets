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
from PyQt6.QtWidgets import (QApplication, QLabel, QTabWidget,
                             QWidget, QVBoxLayout)                  # noqa: E402

from gui.widgets.plot_common import icono_ayuda, icono_ayuda_texto  # noqa: E402
from gui.widgets.tab_backtest import _icono_ayuda                   # noqa: E402


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


_VIVOS = []   # el overlay es hijo de la ventana del icono: si el icono se
              # recolecta, Qt destruye el objeto C++ y consultarlo revienta


def _clicar(icono):
    _VIVOS.append(icono)
    icono.mousePressEvent(None)
    return icono._overlay_ayuda


def _icono_visible(icono):
    """El overlay hereda la visibilidad de sus ancestros (ya no es una ventana
    propia), así que para comprobarlo hay que meter el icono en una ventana
    mostrada."""
    cont = QWidget()
    _VIVOS.append(cont)
    lay = QVBoxLayout(cont)
    lay.addWidget(icono)
    cont.show()
    return cont


def test_el_icono_del_backtester_abre_panel_al_clicarlo(app):
    icono = _icono_ayuda('Qué hace esta sección')
    cont = _icono_visible(icono)
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

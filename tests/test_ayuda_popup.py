"""Los iconos «?» de ayuda abren su panel DONDE SE PUEDE VER.

El popup se ancla a la esquina inferior izquierda del icono y mide 360 px de
ancho. En Análisis eso nunca dio problema porque allí el icono va pegado al
título, a la izquierda, con toda la ventana por delante. En el Backtester va
alineado a la DERECHA de su grupo (`_fila_ayuda`), o sea contra el borde de la
ventana: con la ventana maximizada el panel entero caía fuera del monitor. El
icono respondía al clic y el panel se creaba con su texto — simplemente no
había dónde verlo, y se leía como un icono muerto.

Por eso los tests miran la GEOMETRÍA final contra la pantalla, y no si el popup
existe o si tiene texto: eso ya funcionaba cuando el icono estaba roto de cara
al usuario.
"""
import os

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('PyQt6.QtWidgets')
from PyQt6.QtCore import QPoint, QRect                     # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QHBoxLayout  # noqa: E402

from gui.widgets.plot_common import (                      # noqa: E402
    icono_ayuda, icono_ayuda_texto, _PopupAyuda, _posicion_popup,
)


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def libre(app):
    return app.primaryScreen().availableGeometry()


def _icono_en(x, y, app, secciones_largas=False):
    """Un badge de ayuda colocado en (x, y) de la pantalla, como si fuera el
    del Backtester pegado al borde derecho."""
    cont = QWidget()
    lay = QHBoxLayout(cont)
    icono = (icono_ayuda('lógica', 'significado', 'uso', 'resultados')
             if secciones_largas else icono_ayuda_texto('explicación breve'))
    lay.addWidget(icono)
    cont.setGeometry(x, y, 60, 40)
    cont.show()
    app.processEvents()
    return cont, icono


def _popup_de(icono):
    icono.mousePressEvent(None)
    return icono._popup


def test_pegado_al_borde_derecho_el_panel_sigue_dentro(app, libre):
    """El caso que rompía el Backtester: icono contra el borde derecho."""
    cont, icono = _icono_en(libre.right() - 40, libre.top() + 40, app)
    popup = _popup_de(icono)
    popup.move(_posicion_popup(icono, popup))
    assert libre.contains(popup.geometry()), \
        f"panel en {popup.geometry().getRect()} fuera de {libre.getRect()}"
    cont.deleteLater()


def test_pegado_al_borde_inferior_el_panel_se_abre_hacia_arriba(app, libre):
    cont, icono = _icono_en(libre.left() + 40, libre.bottom() - 40, app,
                            secciones_largas=True)
    popup = _popup_de(icono)
    pos = _posicion_popup(icono, popup)
    popup.move(pos)
    assert libre.contains(popup.geometry()), \
        f"panel en {popup.geometry().getRect()} fuera de {libre.getRect()}"
    assert pos.y() < icono.mapToGlobal(icono.rect().bottomLeft()).y(), \
        "sin sitio debajo, el panel debería abrirse por encima del icono"
    cont.deleteLater()


def test_en_medio_de_la_pantalla_cuelga_del_icono(app, libre):
    """Donde hay sitio no se toca nada: el panel cuelga de la esquina inferior
    izquierda del icono, que es la posición natural. (Icono pegado a la
    izquierda para que el panel de 420 px quepa sin recortes.)"""
    cont, icono = _icono_en(libre.left() + 40, libre.center().y(), app)
    popup = _popup_de(icono)
    esperado = icono.mapToGlobal(icono.rect().bottomLeft())
    assert _posicion_popup(icono, popup) == esperado
    cont.deleteLater()


def test_un_panel_mas_alto_que_la_pantalla_no_se_va_por_arriba(app, libre):
    """Ni arriba ni abajo cabe: el recorte final manda, y el panel empieza
    dentro de la pantalla en vez de salirse por el borde superior."""
    cont, icono = _icono_en(libre.left() + 40, libre.center().y(), app)
    popup = _PopupAyuda([('Ayuda', 'x')], icono)
    popup.setFixedHeight(libre.height() + 400)
    pos = _posicion_popup(icono, popup)
    assert pos.y() >= libre.top()
    assert pos.x() >= libre.left()
    cont.deleteLater()


# ── que el contenido siga estando, no solo la posición ──

@pytest.mark.parametrize('nombre', ['OptimizadorWidget', 'ResultadosWidget'])
def test_los_iconos_del_backtester_tienen_texto(app, nombre):
    """Constructor y Resultados: cada «?» abre un panel con explicación real.
    Un badge sin texto es tan inútil como uno que no se ve."""
    import gui.widgets.tab_backtest as tb
    w = getattr(tb, nombre)()
    badges = [l for l in w.findChildren(QLabel) if l.text() == '?']
    assert badges, f"{nombre} no tiene ningún icono de ayuda"
    for b in badges:
        popup = _popup_de(b)
        textos = [l.text() for l in popup.findChildren(QLabel)]
        assert textos and all(t.strip() for t in textos), \
            f"un icono de {nombre} abre un panel vacío"

"""Los iconos «?» de ayuda abren su panel DONDE SE PUEDE VER.

El overlay se dibuja DENTRO de la ventana principal como widget hijo (sin
ventana propia, sin translucencia de OS: en Windows ya no puede aparecer ningún
borde/sombra alrededor). Se ancla a la esquina inferior izquierda del icono y
mide 420 px de ancho. En Análisis eso nunca dio problema porque allí el icono
va pegado al título, a la izquierda, con toda la ventana por delante. En el
Backtester va alineado a la DERECHA de su grupo (`_fila_ayuda`), o sea contra
el borde de la ventana: con la ventana maximizada el panel entero caería fuera.
El icono respondía al clic y el panel se creaba con su texto — simplemente no
había dónde verlo, y se leía como un icono muerto.

Por eso los tests miran la GEOMETRÍA final contra la ventana principal, y no si
el overlay existe o si tiene texto: eso ya funcionaba cuando el icono estaba
roto de cara al usuario.
"""
import os

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('PyQt6.QtWidgets')
from PyQt6.QtCore import QPoint                          # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel, QWidget  # noqa: E402

from gui.widgets.plot_common import (                    # noqa: E402
    icono_ayuda, icono_ayuda_texto, _OverlayAyuda, _posicion_overlay,
)


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def _ventana_con_icono(app, x_icono, y_icono, ancho=800, alto=600,
                       secciones_largas=False):
    """Ventana principal de prueba con un badge «?» colocado en (x_icono,
    y_icono) dentro de ella, como si fuera el del Backtester pegado al borde."""
    ventana = QWidget()
    ventana.setGeometry(0, 0, ancho, alto)
    icono = (icono_ayuda('lógica', 'significado', 'uso', 'resultados')
             if secciones_largas else icono_ayuda_texto('explicación breve'))
    icono.setParent(ventana)
    icono.move(x_icono, y_icono)
    ventana.show()
    app.processEvents()
    return ventana, icono


def _abrir_overlay(icono):
    icono.mousePressEvent(None)
    return icono._overlay_ayuda


def test_pegado_al_borde_derecho_el_panel_sigue_dentro(app):
    """El caso que rompía el Backtester: icono contra el borde derecho."""
    ventana, icono = _ventana_con_icono(app, 760, 40)
    overlay = _abrir_overlay(icono)
    assert ventana.rect().contains(overlay.geometry()), \
        f"panel en {overlay.geometry().getRect()} fuera de {ventana.rect().getRect()}"
    ventana.deleteLater()


def test_pegado_al_borde_inferior_el_panel_se_abre_hacia_arriba(app):
    ventana, icono = _ventana_con_icono(app, 40, 560, secciones_largas=True)
    overlay = _abrir_overlay(icono)
    assert ventana.rect().contains(overlay.geometry()), \
        f"panel en {overlay.geometry().getRect()} fuera de {ventana.rect().getRect()}"
    assert overlay.y() < icono.y() + icono.height(), \
        "sin sitio debajo, el panel debería abrirse por encima del icono"
    ventana.deleteLater()


def test_en_medio_de_la_ventana_cuelga_del_icono(app):
    """Donde hay sitio no se toca nada: el panel cuelga de la esquina inferior
    izquierda del icono, que es la posición natural. (Icono pegado a la
    izquierda para que el panel de 420 px quepa sin recortes.)"""
    ventana, icono = _ventana_con_icono(app, 40, 300)
    overlay = _abrir_overlay(icono)
    esperado = ventana.mapFromGlobal(icono.mapToGlobal(icono.rect().bottomLeft()))
    assert _posicion_overlay(icono, overlay) == esperado
    ventana.deleteLater()


def test_un_panel_mas_alto_que_la_ventana_no_se_va_por_arriba(app):
    """Ni arriba ni abajo cabe: el recorte final manda, y el panel empieza
    dentro de la ventana en vez de salirse por el borde superior."""
    ventana, icono = _ventana_con_icono(app, 40, 300)
    overlay = _abrir_overlay(icono)
    overlay.setFixedHeight(ventana.height() + 400)
    pos = _posicion_overlay(icono, overlay)
    assert pos.y() >= ventana.rect().top()
    assert pos.x() >= ventana.rect().left()
    ventana.deleteLater()


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
        overlay = _abrir_overlay(b)
        textos = [l.text() for l in overlay.findChildren(QLabel)]
        assert textos and all(t.strip() for t in textos), \
            f"un icono de {nombre} abre un panel vacío"


def test_el_overlay_no_crea_ninguna_ventana(app):
    """El rediseño pinta dentro de la ventana principal: el overlay NO es una
    ventana propia (ahí era donde Windows pintaba bordes/sombras)."""
    ventana, icono = _ventana_con_icono(app, 40, 40)
    overlay = _abrir_overlay(icono)
    assert not overlay.isWindow(), \
        "el overlay debe ser un widget hijo de la ventana, no una ventana"
    assert overlay.parentWidget() is ventana
    ventana.deleteLater()

"""Icono «?» de las categorías de la pestaña Métricas del Analizador.

Lo que se fija aquí es que la ayuda se localiza por el PREFIJO NUMÉRICO del
título y no por el título completo: cuando hay una ventana seleccionada, la GUI
reescribe el título a "7. Ratio Eficiencia … — Scalping" (ver
`TabAnalisis._render_metrics`), así que buscar por título exacto dejaría sin
icono a media pestaña en cuanto el usuario cambia de horizonte.

El último test compara las claves con los títulos que genera de verdad
`analisis_descriptivo.py`: si mañana se añade una categoría al informe, salta
aquí en vez de aparecer muda en la interfaz.
"""
import os
import re

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('PyQt6.QtWidgets')
from PyQt6.QtWidgets import QApplication, QLabel, QTabWidget   # noqa: E402

from gui.widgets.tab_analisis import (                          # noqa: E402
    AYUDA_CATEGORIAS, AYUDA_VENTANA, CategoryGroup, TabAnalisis,
    _ayuda_categoria,
)

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALIZADOR = os.path.join(RAIZ, 'library', 'scripts_utiles', 'analisis_descriptivo.py')


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def _iconos(widget):
    return [w for w in widget.findChildren(QLabel) if w.text() == '?']


def test_todas_las_secciones_tienen_texto():
    for clave, secciones in AYUDA_CATEGORIAS.items():
        assert len(secciones) == 4, f'categoría {clave}: se esperan 4 pestañas'
        for texto in secciones:
            assert isinstance(texto, str) and texto.strip(), \
                f'categoría {clave}: pestaña vacía'


@pytest.mark.parametrize('titulo, clave', [
    ('1. Información General y tipo de muestreo', '1'),
    ('7. Ratio Eficiencia (ER) y Exponente de Hurst', '7'),
    ('7. Ratio Eficiencia (ER) y Exponente de Hurst — Scalping', '7'),
    ('14. NATR, correlación Multi-TF', '14'),
    ('14.5. NATR Z-score, Ratio por horizonte', '14.5'),
])
def test_clave_por_prefijo_numerico(titulo, clave):
    # 14 y 14.5 son categorías distintas: el regex no puede confundirlas.
    assert _ayuda_categoria(titulo) is AYUDA_CATEGORIAS[clave]


def test_categoria_desconocida_no_pinta_icono(app):
    grupo = CategoryGroup('99. Categoría inventada', {'x': '1'})
    assert _iconos(grupo) == []


def test_categoria_conocida_pinta_un_solo_icono_con_cuatro_pestanas(app):
    grupo = CategoryGroup('12. Test de Estacionariedad (ADF / KPSS)',
                          {'Veredicto (Precio)': 'No estacionario'})
    iconos = _iconos(grupo)
    assert len(iconos) == 1

    icono = iconos[0]
    icono.mousePressEvent(None)          # el handler ignora el evento
    tabs = icono._overlay_ayuda.findChild(QTabWidget)
    assert [tabs.tabText(i) for i in range(tabs.count())] == \
        ['Lógica', 'Significado', 'Uso', 'Resultados']


def test_selector_ventana_tiene_icono_junto_al_combo(app):
    tab = TabAnalisis()
    barra = tab.horizon.parent().layout().itemAt(0).layout()
    posiciones = {barra.itemAt(i).widget(): i for i in range(barra.count())
                  if barra.itemAt(i).widget() is not None}
    # Detrás del combo: si alguien lo mueve delante, separaría la etiqueta
    # "Ventana" de su propio control.
    assert posiciones[tab.icono_ventana] == posiciones[tab.horizon] + 1

    tab.icono_ventana.mousePressEvent(None)
    tabs = tab.icono_ventana._overlay_ayuda.findChild(QTabWidget)
    assert tabs.count() == 4
    assert all(t.strip() for t in AYUDA_VENTANA)


def test_cobertura_frente_a_las_categorias_del_informe():
    if not os.path.exists(ANALIZADOR):
        pytest.skip('analisis_descriptivo.py no disponible')
    with open(ANALIZADOR, encoding='utf-8') as f:
        fuente = f.read()

    # Categorías registradas con _mostrar_categoria('N. ...') y las que se
    # inyectan a mano en el dict `metricas` (14 y 14.5).
    claves = set(re.findall(r"_mostrar_categoria\(\s*'(\d+(?:\.\d+)?)\.", fuente))
    claves |= set(re.findall(r"metricas\[\s*'(\d+(?:\.\d+)?)\.", fuente))

    faltan = claves - set(AYUDA_CATEGORIAS)
    assert not faltan, f'categorías del informe sin ayuda en la GUI: {sorted(faltan)}'

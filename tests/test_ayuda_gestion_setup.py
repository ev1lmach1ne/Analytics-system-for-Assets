"""Los seis campos de gestión del setup se explican solos, y se llaman igual
en toda la app.

El bocadillo tiene que estar en la ETIQUETA, no solo en el control: `addRow`
fabrica la QLabel por dentro y nadie le ponía tooltip, así que pasar el ratón
por el nombre del campo —que es lo que hace quien no sabe qué es «Break Even»—
no decía nada. Dos de los seis (Stop Loss y Take Profit) no tenían ayuda ni en
la etiqueta ni en el control.
"""
import os

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('PyQt6.QtWidgets')
from PyQt6.QtWidgets import (QApplication, QLabel, QFormLayout,   # noqa: E402
                             QRadioButton)

from core.backtest import MOTIVOS_SALIDA                          # noqa: E402
import gui.widgets.tab_backtest as tb                             # noqa: E402

# los seis conceptos, por el atributo del control de su fila
CAMPOS = ['sp_riesgo', 'sp_stop', 'sp_tp', 'sp_tiempo', 'sp_be', 'sp_trailing']

# nomenclatura vieja que no debe quedar en ningún texto visible
VIEJAS = ('Take-profit', 'Break-even', 'Trailing stop', 'Stop-loss', 'Stop loss')


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope='module')
def constructor(app):
    return tb.OptimizadorWidget()


def _etiqueta_de(widget):
    """La QLabel que el QFormLayout puso delante de `widget`.

    `labelForField` solo resuelve cuando el campo de la fila ES el widget. La
    fila de Break Even mete el spin dentro de un QHBoxLayout (spin + combo de
    unidad), así que ahí hay que buscar por la fila que CONTIENE al widget."""
    for form in widget.window().findChildren(QFormLayout):
        etiqueta = form.labelForField(widget)
        if etiqueta is not None:
            return etiqueta
        for fila in range(form.rowCount()):
            item = form.itemAt(fila, QFormLayout.ItemRole.FieldRole)
            campo = item.layout() if item is not None else None
            if campo is not None:
                hijos = (campo.itemAt(i).widget() for i in range(campo.count()))
                if any(h is widget for h in hijos):
                    item_lbl = form.itemAt(fila, QFormLayout.ItemRole.LabelRole)
                    return item_lbl.widget() if item_lbl is not None else None
            campo_widget = item.widget() if item is not None else None
            if campo_widget is not None and (
                    campo_widget is widget
                    or widget in campo_widget.findChildren(type(widget))):
                item_lbl = form.itemAt(fila, QFormLayout.ItemRole.LabelRole)
                return item_lbl.widget() if item_lbl is not None else None
    return None


@pytest.mark.parametrize('attr', CAMPOS)
def test_la_etiqueta_del_campo_explica_el_concepto(constructor, attr):
    campo = getattr(constructor, attr)
    etiqueta = _etiqueta_de(campo)
    assert etiqueta is not None, f"{attr} no está en ningún QFormLayout"
    ayuda = etiqueta.toolTip()
    assert ayuda.strip(), f"la etiqueta de {attr} no tiene bocadillo"
    assert 'Para qué sirve' in ayuda, \
        f"el bocadillo de {attr} dice qué es pero no para qué se usa"


@pytest.mark.parametrize('attr', CAMPOS)
def test_el_control_tambien_lleva_bocadillo(constructor, attr):
    """Stop Loss y Take Profit no tenían ninguno; el resto sí."""
    assert getattr(constructor, attr).toolTip().strip(), \
        f"{attr} se quedó sin bocadillo en el control"


def test_excluir_interpoladas_esta_activado_por_defecto(constructor):
    assert constructor.chk_excluir_interpoladas.isChecked()
    assert 'interpoladas' in constructor.chk_excluir_interpoladas.toolTip()


def test_modo_stop_esta_a_la_derecha_y_solo_funciona_con_stop(constructor):
    assert constructor.cmb_stop_modo.parentWidget() is constructor._stop_row
    assert constructor.sp_stop.value() == 0.0
    assert not constructor.cmb_stop_modo.isEnabled()

    constructor.sp_stop.setValue(1.0)
    assert constructor.cmb_stop_modo.isEnabled()
    constructor.sp_stop.setValue(0.0)
    assert not constructor.cmb_stop_modo.isEnabled()


def test_el_identificador_hurst_de_la_gui_es_el_canonico():
    """La GUI y el core deben hablar el mismo id para la reversión de Hurst:
    'hurst_reversion', nunca 'hurst_reversión' (con tilde)."""
    assert tb._MAPA_REGIMEN['Reversión (Hurst)'] == 'hurst_reversion'
    assert 'hurst_reversión' not in tb._MAPA_REGIMEN.values()
    assert 'hurst_reversion' in tb.BANDA_REGIMEN
    assert 'hurst_reversión' not in tb.BANDA_REGIMEN
    assert tb._normalizar_metodo_regimen('hurst_reversión') == 'hurst_reversion'


def test_el_combo_de_unidad_conserva_su_propia_explicacion(constructor):
    """La fila de Break Even lleva spin + combo de unidad. El combo explica
    otra cosa (× ATR frente a R): el texto de la fila no debe pisarlo."""
    propio = constructor.cmb_be_unidad.toolTip()
    assert '× ATR' in propio and ' R ' in propio.replace('\n', ' ')
    assert propio != constructor.sp_be.toolTip()


# ── nomenclatura ──

def test_las_etiquetas_usan_la_nomenclatura_nueva(constructor):
    esperadas = {'sp_stop': 'Stop Loss', 'sp_tp': 'Take Profit',
                 'sp_be': 'Break Even', 'sp_trailing': 'Trailing Stop'}
    for attr, termino in esperadas.items():
        texto = _etiqueta_de(getattr(constructor, attr)).text()
        assert termino in texto, f"{attr} se llama «{texto}»"


def test_no_queda_nomenclatura_vieja_a_la_vista(constructor):
    """Recorre TODO texto visible del Constructor: etiquetas, radios y los
    valores especiales de los spins ('Sin stop'...)."""
    textos = [w.text() for w in constructor.findChildren(QLabel)]
    textos += [w.text() for w in constructor.findChildren(QRadioButton)]
    textos += [getattr(constructor, a).specialValueText() for a in CAMPOS]
    for texto in textos:
        for vieja in VIEJAS:
            assert vieja not in texto, f"queda «{vieja}» en: {texto!r}"


def test_los_mecanismos_de_salida_se_llaman_igual():
    etiquetas = [v[0] for v in tb._MECANISMOS_SALIDA_UI.values()]
    assert 'Stop Loss' in etiquetas and 'Take Profit' in etiquetas
    assert 'Break Even' in etiquetas and 'Trailing Stop' in etiquetas


def test_la_columna_motivo_de_los_trades_usa_la_nomenclatura_nueva():
    """MOTIVOS_SALIDA se indexa por el entero del motor; el texto es solo para
    enseñarlo en la tabla de trades, así que puede cambiar sin romper nada."""
    assert MOTIVOS_SALIDA[1] == 'Stop Loss'
    assert MOTIVOS_SALIDA[2] == 'Take Profit'
    assert set(MOTIVOS_SALIDA) == {0, 1, 2, 3, 4, 5}

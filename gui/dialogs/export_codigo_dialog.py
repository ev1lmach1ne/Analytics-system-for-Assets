"""
gui/dialogs/export_codigo_dialog.py
Diálogo de exportación del sistema a código de plataformas de trading.

Dos pantallas:

  DialogoExportarCodigo   elegir plataformas y carpeta, con el informe de
                          fidelidad en vivo para cada una
  DialogoConfirmarPerdida lo que se pierde al traducir, ANTES de escribir
                          nada en disco

La segunda no es un trámite. Un filtro de sesión o de noticias que desaparece
en la traducción convierte el código exportado en un sistema distinto del que
se backtesteó, y mirando el .mq5 o el .pine no se nota: lo que falta no se ve.
Por eso el botón por defecto es Cancelar y, si hay algo omitido, hace falta
marcar una casilla de conformidad para continuar.
"""
import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QVBoxLayout,
)

import core.codegen as codegen
from core.codegen import fidelidad
from core.config import SISTEMAS_DIR

# Se escribe directamente bajo Sistemas/, sin un nivel intermedio: el árbol ya
# tiene que llevar dentro la estructura que espera cada plataforma (MetaTrader
# necesita MQL5/Experts y MQL5/Include para poder copiarse de una vez), así
# que cada carpeta de más son clics de más para llegar al archivo.
#
# No choca con los sistemas guardados aunque coincida el nombre: _leer_guardadas
# solo mira las carpetas que tienen sistema.json dentro.
CARPETA_DEFECTO = SISTEMAS_DIR

# Paleta del banner de avisos de Resultados, para no inventar colores nuevos.
_COLORES = {
    fidelidad.NIVEL_EXACTO: ('#10240f', '#7ed321', '#2a5a20'),
    fidelidad.NIVEL_APROXIMADO: ('#2a2010', '#ffb74d', '#5a4520'),
    fidelidad.NIVEL_OMITIDO: ('#2a1010', '#ff6b6b', '#5a2020'),
}

STYLE = """
QDialog { background-color: #141e30; }
QLabel { color: #c8d6e5; }
QLabel#title { color: #4fc3f7; font-size: 15px; font-weight: bold; }
QLabel#desc { color: #5a7a9a; font-size: 11px; }
QLabel#lenguaje { color: #5a7a9a; font-size: 10px; }
QLineEdit {
    background-color: #1a2a45; color: #c8d6e5; border: 1px solid #253a60;
    padding: 8px 10px; border-radius: 4px; font-size: 12px;
}
QPushButton {
    background-color: #2a4a6a; color: #4fc3f7; border: none;
    padding: 9px 20px; border-radius: 4px; font-size: 12px; font-weight: bold;
}
QPushButton:hover { background-color: #3a5a8a; }
QPushButton:disabled { background-color: #1a2a45; color: #3a5a7a; }
QPushButton#accept { background-color: #0f2a1a; color: #2ecc71; }
QPushButton#accept:hover { background-color: #1a3a2a; }
QPushButton#browse { background-color: #1a2a45; color: #7aaccc; }
QPushButton#peligro { background-color: #2a1010; color: #ff6b6b; }
QPushButton#peligro:hover { background-color: #3a1a1a; }
QCheckBox { color: #c8d6e5; font-size: 12px; }
QCheckBox:disabled { color: #3a5a7a; }
QFrame#tarjeta {
    background-color: #1a2a45; border: 1px solid #253a60; border-radius: 4px;
}
QFrame#tarjetaOff { background-color: #16203a; border: 1px solid #1e2d4a;
    border-radius: 4px; }
QScrollArea { border: 1px solid #253a60; border-radius: 4px;
    background-color: #0d1424; }
"""


class _TarjetaPlataforma(QFrame):
    """Una plataforma de la cuadrícula: casilla, nombre, lenguaje y el
    distintivo de fidelidad de ESTE sistema en ESA plataforma.

    El distintivo se ve antes de elegir a propósito: así se sabe qué
    plataforma respeta el sistema y cuál no sin tener que exportar primero."""

    def __init__(self, info, parent=None):
        super().__init__(parent)
        self.info = info
        self.disponible = info['estado'] == 'disponible'
        self.setObjectName('tarjeta' if self.disponible else 'tarjetaOff')
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(2)

        fila = QHBoxLayout()
        self.chk = QCheckBox(info['nombre'])
        self.chk.setEnabled(self.disponible)
        fila.addWidget(self.chk, 1)
        self.lbl_badge = QLabel("")
        fila.addWidget(self.lbl_badge)
        lay.addLayout(fila)

        sub = QLabel(info['lenguaje'] if self.disponible
                     else f"{info['lenguaje']} · próximamente")
        sub.setObjectName('lenguaje')
        lay.addWidget(sub)

    def marcada(self):
        return self.disponible and self.chk.isChecked()

    def pintar_fidelidad(self, nivel, n_avisos, bloqueados):
        """Distintivo y tooltip. Se resume en una línea porque el detalle va
        en el panel de abajo y, sobre todo, en el diálogo de confirmación."""
        if not self.disponible:
            self.lbl_badge.setText("")
            return
        icono = fidelidad.ICONOS[nivel]
        self.lbl_badge.setText(icono)
        _fondo, texto, _borde = _COLORES[nivel]
        self.lbl_badge.setStyleSheet(f"color: {texto}; font-size: 13px;")
        if nivel == fidelidad.NIVEL_EXACTO:
            ayuda = "Reproduce el sistema sin diferencias conocidas."
        elif nivel == fidelidad.NIVEL_APROXIMADO:
            ayuda = f"{n_avisos} diferencia(s), ninguna omisión."
        else:
            ayuda = f"{n_avisos} diferencia(s), con omisiones."
        if bloqueados:
            ayuda += (f"\nNo se exportarán los setups: "
                      f"{', '.join(str(b) for b in bloqueados)}.")
        self.lbl_badge.setToolTip(ayuda)


class DialogoExportarCodigo(QDialog):
    """Elegir plataformas y destino. No escribe nada hasta que el usuario pasa
    por el diálogo de confirmación."""

    def __init__(self, setups, config_global, meta, parent=None):
        super().__init__(parent)
        self._setups = setups
        self._config = config_global
        self._meta = dict(meta or {})
        self.resultado = None          # lo que devuelve exportar_sistema()

        self.setWindowTitle("Analytics System · Exportar código")
        self.setModal(True)
        self.resize(720, 640)
        self.setStyleSheet(STYLE)

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(24, 20, 24, 20)
        raiz.setSpacing(12)

        titulo = QLabel("📤  Exportar el sistema a otra plataforma")
        titulo.setObjectName('title')
        raiz.addWidget(titulo)

        desc = QLabel(
            f"Sistema «{self._meta.get('sistema', '')}» · "
            f"{self._meta.get('activo', '?')} · {self._meta.get('tf', '?')}\n"
            "Cada setup se exporta a un archivo independiente. Los parámetros "
            "están ajustados a este activo y esta temporalidad.")
        desc.setObjectName('desc')
        desc.setWordWrap(True)
        raiz.addWidget(desc)

        # ── cuadrícula de plataformas ──
        rejilla = QGridLayout()
        rejilla.setSpacing(8)
        self._tarjetas = {}
        for i, info in enumerate(codegen.PLATAFORMAS):
            tarjeta = _TarjetaPlataforma(info)
            tarjeta.chk.stateChanged.connect(self._refrescar)
            self._tarjetas[info['clave']] = tarjeta
            rejilla.addWidget(tarjeta, i // 2, i % 2)
        raiz.addLayout(rejilla)

        # ── panel de fidelidad ──
        raiz.addWidget(QLabel("Qué se pierde al traducir:"))
        self._panel = QLabel("")
        self._panel.setWordWrap(True)
        self._panel.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._panel.setTextFormat(Qt.TextFormat.RichText)
        self._panel.setContentsMargins(10, 8, 10, 8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._panel)
        scroll.setMinimumHeight(150)
        raiz.addWidget(scroll, 1)

        # ── destino ──
        fila_destino = QHBoxLayout()
        fila_destino.addWidget(QLabel("Carpeta:"))
        self.txt_destino = QLineEdit(CARPETA_DEFECTO)
        fila_destino.addWidget(self.txt_destino, 1)
        btn_examinar = QPushButton("Examinar…")
        btn_examinar.setObjectName('browse')
        btn_examinar.clicked.connect(self._examinar)
        fila_destino.addWidget(btn_examinar)
        raiz.addLayout(fila_destino)

        fila_nombre = QHBoxLayout()
        fila_nombre.addWidget(QLabel("Nombre:"))
        self.txt_nombre = QLineEdit(self._meta.get('sistema', 'sistema'))
        fila_nombre.addWidget(self.txt_nombre, 1)
        raiz.addLayout(fila_nombre)

        # ── botones ──
        botones = QHBoxLayout()
        # TradingView no importa archivos: el código se pega. Guardar un .pine
        # para que el usuario lo abra con un editor y lo copie es un rodeo, así
        # que el portapapeles va como acción de primera clase.
        self.btn_copiar = QPushButton("📋 Copiar Pine")
        self.btn_copiar.setObjectName('browse')
        self.btn_copiar.setToolTip(
            "Pone el código Pine en el portapapeles, listo para pegarlo en el "
            "Pine Editor de TradingView. No hace falta guardar el archivo.")
        self.btn_copiar.clicked.connect(self._copiar_pine)
        botones.addWidget(self.btn_copiar)
        botones.addStretch(1)
        btn_cancelar = QPushButton("Cancelar")
        btn_cancelar.clicked.connect(self.reject)
        botones.addWidget(btn_cancelar)
        self.btn_exportar = QPushButton("Exportar")
        self.btn_exportar.setObjectName('accept')
        self.btn_exportar.clicked.connect(self._exportar)
        botones.addWidget(self.btn_exportar)
        raiz.addLayout(botones)

        # marcar por defecto todas las disponibles: es lo que casi siempre se
        # quiere y deja el informe de fidelidad visible desde el primer momento
        for tarjeta in self._tarjetas.values():
            if tarjeta.disponible:
                tarjeta.chk.setChecked(True)
        self._refrescar()

    # ══════════════ estado ══════════════

    def plataformas_marcadas(self):
        return [clave for clave, t in self._tarjetas.items() if t.marcada()]

    def _plataformas_bloqueadas_por_noticias(self, analisis):
        """Plataformas cuyo análisis tiene el filtro de noticias en nivel
        OMITIDO (TradingView/MT4): el robot generado no evitaría las ventanas
        de noticias que el backtest sí evita, así que el export a esa
        plataforma queda bloqueado mientras el sistema use «Evitar noticias»."""
        return {
            clave for clave, info in analisis.items()
            if any(a.get('clave') == 'noticias'
                   and a['nivel'] == fidelidad.NIVEL_OMITIDO
                   for a in info['avisos'])
        }

    def _analisis(self, claves):
        return codegen.analizar_sistema(self._setups, self._config, claves)

    def _refrescar(self):
        """Recalcula los distintivos de TODAS las disponibles (no solo las
        marcadas: el distintivo sirve justo para decidir cuál marcar) y
        redibuja el panel con las marcadas."""
        disponibles = [c for c, t in self._tarjetas.items() if t.disponible]
        analisis = self._analisis(disponibles)
        bloqueadas = self._plataformas_bloqueadas_por_noticias(analisis)
        for clave in disponibles:
            info = analisis[clave]
            tarjeta = self._tarjetas[clave]
            tarjeta.pintar_fidelidad(
                info['nivel'], len(info['avisos']), info['bloqueados'])
            bloqueada = clave in bloqueadas
            tarjeta.chk.setEnabled(tarjeta.disponible and not bloqueada)
            if bloqueada:
                nombre = codegen.plataforma(clave)['nombre']
                tarjeta.chk.setChecked(False)
                tarjeta.chk.setToolTip(
                    f"El filtro «Evitar noticias» no está permitido para "
                    f"{nombre}: el robot no funcionará siguiendo las mismas "
                    f"pautas que el backtest. Desactiva el filtro de "
                    f"noticias o usa otra plataforma (MT5).")
            else:
                tarjeta.chk.setToolTip("")

        marcadas = self.plataformas_marcadas()
        self.btn_exportar.setEnabled(bool(marcadas))
        self._panel.setText(self._html_panel(marcadas, analisis))

    def _html_panel(self, marcadas, analisis):
        if not marcadas:
            return ("<span style='color:#5a7a9a'>Marca al menos una "
                    "plataforma.</span>")
        trozos = []
        for clave in marcadas:
            info = analisis[clave]
            nombre = codegen.plataforma(clave)['nombre']
            _fondo, color, _borde = _COLORES[info['nivel']]
            trozos.append(
                f"<p style='color:{color}; margin:6px 0 2px 0'>"
                f"<b>{nombre}</b> {fidelidad.ICONOS[info['nivel']]}</p>")
            for indice in info['bloqueados']:
                trozos.append(
                    f"<p style='color:#ff6b6b; margin:0 0 0 12px'>"
                    f"El setup {indice} no se exportará: falta su propia "
                    f"señal.</p>")
            if not info['avisos']:
                trozos.append("<p style='color:#7ed321; margin:0 0 0 12px'>"
                              "Sin diferencias conocidas.</p>")
            for aviso in info['avisos']:
                _f, c, _b = _COLORES[aviso['nivel']]
                trozos.append(
                    f"<p style='color:{c}; margin:0 0 0 12px'>"
                    f"{_escapar(fidelidad.texto_aviso(aviso))}</p>")
        return "".join(trozos)

    # ══════════════ acciones ══════════════

    def _copiar_pine(self):
        """Deja el código Pine en el portapapeles, sin escribir nada en disco.

        Con varios setups se copia el primero exportable y se dice cuál: son
        scripts independientes y concatenarlos daría un archivo que no
        compila."""
        analisis = self._analisis(['tradingview'])['tradingview']
        if self._plataformas_bloqueadas_por_noticias(
                {'tradingview': analisis}):
            QMessageBox.warning(
                self, "Filtro de noticias no soportado",
                "El filtro «Evitar noticias» no está permitido para "
                "TradingView: el robot no funcionará siguiendo las mismas "
                "pautas que el backtest. Desactiva el filtro de noticias o "
                "usa otra plataforma (MT5).")
            return
        exportables = [s['indice'] for s in
                       codegen.ir.ir_sistema(self._setups, self._config)['setups']
                       if s['indice'] not in analisis['bloqueados']]
        if not exportables:
            QMessageBox.warning(
                self, "No hay nada que copiar",
                "Ningún setup de este sistema se puede generar todavía en "
                "Pine Script: lo que falta es su propia señal.")
            return
        indice = exportables[0]
        try:
            codigo = codegen.codigo_de_setup(
                self._setups, self._config, 'tradingview', indice, self._meta)
        except Exception as e:                      # noqa: BLE001
            QMessageBox.critical(self, "No se ha podido generar",
                                 f"{type(e).__name__}: {e}")
            return
        QApplication.clipboard().setText(codigo)

        aviso = ""
        if len(exportables) > 1:
            aviso = (f"\n\nEste sistema tiene {len(exportables)} setups y cada "
                     f"uno es un script aparte: se ha copiado el S{indice}. "
                     f"Para los demás, usa Exportar.")
        omitido = ("\n\nOJO: hay diferencias con el backtest. Pulsa Exportar "
                   "para verlas y para tener NOTAS_DE_FIDELIDAD.md."
                   if analisis['avisos'] else "")
        self.btn_copiar.setText("✓ Copiado")
        QMessageBox.information(
            self, "Pine copiado al portapapeles",
            f"Pega con Ctrl+V en el Pine Editor de TradingView "
            f"(gráfico de {self._meta.get('activo', '?')}, "
            f"{self._meta.get('tf', '?')}).{aviso}{omitido}")

    def _examinar(self):
        actual = self.txt_destino.text().strip()
        if not os.path.isdir(actual):
            actual = SISTEMAS_DIR
        ruta = QFileDialog.getExistingDirectory(
            self, "Carpeta donde escribir el código", actual)
        if ruta:
            self.txt_destino.setText(ruta)

    def _exportar(self):
        claves = self.plataformas_marcadas()
        if not claves:
            return
        nombre = self.txt_nombre.text().strip()
        if not nombre:
            QMessageBox.warning(self, "Falta el nombre",
                                "Pon un nombre para la carpeta del sistema.")
            return
        destino = self.txt_destino.text().strip() or CARPETA_DEFECTO

        analisis = self._analisis(claves)
        # Bloqueo duro: el filtro de noticias no se reproduce en esas
        # plataformas y el robot operaría dentro de las ventanas que el
        # backtest evitaba — no es una diferencia menor, es otra estrategia.
        bloqueadas = self._plataformas_bloqueadas_por_noticias(analisis)
        if bloqueadas:
            nombres = ', '.join(
                codegen.plataforma(c)['nombre'] for c in sorted(bloqueadas))
            QMessageBox.warning(
                self, "Filtro de noticias no soportado",
                f"El filtro «Evitar noticias» no está permitido para "
                f"{nombres}: el robot no funcionará siguiendo las mismas "
                f"pautas que el backtest. Desmarca {nombres} o desactiva el "
                f"filtro de noticias.")
            return
        # Nada se escribe hasta que el usuario ve qué se pierde y lo acepta.
        if any(a['avisos'] for a in analisis.values()):
            confirmacion = DialogoConfirmarPerdida(analisis, self)
            if confirmacion.exec() != QDialog.DialogCode.Accepted:
                return

        try:
            self.resultado = codegen.exportar_sistema(
                self._setups, self._config, claves, destino, nombre,
                self._meta)
        except Exception as e:                      # noqa: BLE001
            QMessageBox.critical(
                self, "No se ha podido exportar",
                f"{type(e).__name__}: {e}")
            return
        self.accept()


class DialogoConfirmarPerdida(QDialog):
    """Lo que se pierde al traducir, agrupado por plataforma y por setup.

    Sale ANTES de escribir nada. Cancelar es el botón por defecto; si hay
    alguna omisión (❌) hace falta marcar la casilla de conformidad, porque en
    ese caso el código generado NO reproduce el sistema backtesteado."""

    def __init__(self, analisis, parent=None):
        super().__init__(parent)
        self._hay_omisiones = any(
            fidelidad.hay_omisiones(a['avisos']) for a in analisis.values())

        self.setWindowTitle("Analytics System · Antes de exportar")
        self.setModal(True)
        self.resize(680, 560)
        self.setStyleSheet(STYLE)

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(24, 20, 24, 20)
        raiz.setSpacing(12)

        titulo = QLabel("Esto no se reproduce igual que en el backtest")
        titulo.setObjectName('title')
        raiz.addWidget(titulo)

        cuerpo = QLabel(self._html(analisis))
        cuerpo.setWordWrap(True)
        cuerpo.setTextFormat(Qt.TextFormat.RichText)
        cuerpo.setAlignment(Qt.AlignmentFlag.AlignTop)
        cuerpo.setContentsMargins(10, 8, 10, 8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(cuerpo)
        raiz.addWidget(scroll, 1)

        self.chk_entiendo = QCheckBox(
            "Entiendo que el código generado no reproduce el sistema "
            "backtesteado")
        self.chk_entiendo.setVisible(self._hay_omisiones)
        self.chk_entiendo.stateChanged.connect(self._refrescar_boton)
        raiz.addWidget(self.chk_entiendo)

        botones = QHBoxLayout()
        botones.addStretch(1)
        btn_cancelar = QPushButton("Cancelar")
        btn_cancelar.setDefault(True)      # el seguro es no escribir nada
        btn_cancelar.clicked.connect(self.reject)
        botones.addWidget(btn_cancelar)
        self.btn_seguir = QPushButton("Exportar igualmente")
        self.btn_seguir.setObjectName('peligro' if self._hay_omisiones
                                      else 'accept')
        self.btn_seguir.clicked.connect(self.accept)
        botones.addWidget(self.btn_seguir)
        raiz.addLayout(botones)

        self._refrescar_boton()
        btn_cancelar.setFocus()

    def _refrescar_boton(self):
        self.btn_seguir.setEnabled(
            not self._hay_omisiones or self.chk_entiendo.isChecked())

    def _html(self, analisis):
        trozos = []
        for clave, info in analisis.items():
            if not info['avisos'] and not info['bloqueados']:
                continue
            nombre = codegen.plataforma(clave)['nombre']
            trozos.append(f"<h3 style='color:#4fc3f7'>{nombre}</h3>")
            for indice in info['bloqueados']:
                trozos.append(
                    f"<p style='color:#ff6b6b'><b>El setup {indice} no se "
                    f"exportará.</b> Lo que falta es su propia señal, así que "
                    f"el archivo sería un robot incapaz de abrir una "
                    f"operación.</p>")
            for aviso in info['avisos']:
                _f, color, _b = _COLORES[aviso['nivel']]
                # El filtro de noticias merece su propio banner: es la
                # diferencia que más cambia el comportamiento del robot.
                if aviso.get('clave') == 'noticias':
                    if aviso['nivel'] == fidelidad.NIVEL_OMITIDO:
                        trozos.append(
                            f"<p style='color:#ff6b6b; background:#2a1010; "
                            f"padding:8px; border-radius:4px;'><b>El filtro "
                            f"«Evitar noticias» NO se reproduce en {nombre}:</b> "
                            f"el robot operará también dentro de las ventanas "
                            f"de noticias que el backtest evitaba — hará más "
                            f"trades y con peor slippage que el backtest.</p>")
                    elif aviso['nivel'] == fidelidad.NIVEL_APROXIMADO:
                        trozos.append(
                            f"<p style='color:#ffb74d; background:#2a2010; "
                            f"padding:8px; border-radius:4px;'><b>El filtro "
                            f"«Evitar noticias» es aproximado en {nombre}:</b> "
                            f"usa el calendario de la plataforma; bloqueará "
                            f"ventanas parecidas, pero no exactamente las "
                            f"mismas velas que el backtest.</p>")
                trozos.append(
                    f"<p style='color:{color}; margin:8px 0'>"
                    f"{_escapar(fidelidad.texto_aviso(aviso))}</p>")
        return "".join(trozos) or "<p>Sin diferencias.</p>"


def _escapar(texto):
    return (str(texto).replace('&', '&amp;').replace('<', '&lt;')
            .replace('>', '&gt;'))

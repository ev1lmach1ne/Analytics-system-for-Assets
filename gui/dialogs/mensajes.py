"""
gui/dialogs/mensajes.py
Sustitutos tematizados de QMessageBox / QInputDialog / QFileDialog: sin barra
de título de Windows y con la paleta oscura de la app.

Las firmas y valores de retorno imitan a las versiones originales para que
los puntos de llamada solo cambien el nombre de la función.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QFileDialog, QLabel, QLineEdit,
                             QPushButton, QHBoxLayout)

from gui.dialogs.base import crear_dialogo

_ICONOS = {
    'info': 'ℹ',
    'ok': '✓',
    'aviso': '⚠',
    'error': '✕',
}
_ACENTOS = {
    'info': '#4fc3f7',
    'ok': '#2ecc71',
    'aviso': '#ffb74d',
    'error': '#e74c3c',
}


def _mensaje(tipo, parent, titulo, texto, subtitulo='', ancho=400):
    dlg, contenido, _ = crear_dialogo(titulo, parent, subtitulo=subtitulo,
                                      ancho=ancho)
    fila = QHBoxLayout()
    fila.setSpacing(12)
    icono = QLabel(_ICONOS[tipo])
    icono.setStyleSheet(f"color: {_ACENTOS[tipo]}; font-size: 30px;")
    fila.addWidget(icono, 0, Qt.AlignmentFlag.AlignTop)
    lbl = QLabel(texto)
    lbl.setWordWrap(True)
    lbl.setObjectName("dlgTexto")
    fila.addWidget(lbl, 1)
    contenido.addLayout(fila)

    botones = QHBoxLayout()
    botones.addStretch()
    btn = QPushButton("Aceptar")
    btn.setObjectName("accionOk" if tipo in ('ok', 'info') else "accionNeutra")
    btn.setDefault(True)
    btn.clicked.connect(dlg.accept)
    botones.addWidget(btn)
    contenido.addLayout(botones)

    dlg.exec()
    return True


def informacion(parent, titulo, texto, subtitulo=''):
    return _mensaje('info', parent, titulo, texto, subtitulo)


def aviso(parent, titulo, texto, subtitulo=''):
    return _mensaje('aviso', parent, titulo, texto, subtitulo)


def error(parent, titulo, texto, subtitulo=''):
    return _mensaje('error', parent, titulo, texto, subtitulo)


def confirmar(parent, titulo, texto, subtitulo='', texto_si='Sí',
              texto_no='No'):
    """Pregunta Sí/No. Devuelve True si el usuario confirma."""
    dlg, contenido, _ = crear_dialogo(titulo, parent, subtitulo=subtitulo,
                                      ancho=420)
    fila = QHBoxLayout()
    fila.setSpacing(12)
    icono = QLabel('?')
    icono.setStyleSheet("color: #4fc3f7; font-size: 30px;")
    fila.addWidget(icono, 0, Qt.AlignmentFlag.AlignTop)
    lbl = QLabel(texto)
    lbl.setWordWrap(True)
    lbl.setObjectName("dlgTexto")
    fila.addWidget(lbl, 1)
    contenido.addLayout(fila)

    botones = QHBoxLayout()
    botones.addStretch()
    btn_no = QPushButton(texto_no)
    btn_no.setObjectName("accionNeutra")
    btn_no.clicked.connect(dlg.reject)
    botones.addWidget(btn_no)
    btn_si = QPushButton(texto_si)
    btn_si.setObjectName("accionPeligro")
    btn_si.clicked.connect(dlg.accept)
    botones.addWidget(btn_si)
    contenido.addLayout(botones)

    return dlg.exec() == QDialog.DialogCode.Accepted


def pedir_texto(parent, titulo, etiqueta, inicial='', placeholder=''):
    """Pide un texto. Devuelve (texto, ok) como QInputDialog.getText."""
    dlg, contenido, _ = crear_dialogo(titulo, parent, ancho=400)
    lbl = QLabel(etiqueta)
    lbl.setObjectName("dlgTexto")
    lbl.setWordWrap(True)
    contenido.addWidget(lbl)
    edit = QLineEdit(inicial)
    if placeholder:
        edit.setPlaceholderText(placeholder)
    edit.selectAll()
    contenido.addWidget(edit)

    botones = QHBoxLayout()
    botones.addStretch()
    btn_c = QPushButton("Cancelar")
    btn_c.setObjectName("accionNeutra")
    btn_c.clicked.connect(dlg.reject)
    botones.addWidget(btn_c)
    btn_ok = QPushButton("Aceptar")
    btn_ok.setObjectName("accionOk")
    btn_ok.setDefault(True)
    btn_ok.clicked.connect(dlg.accept)
    botones.addWidget(btn_ok)
    contenido.addLayout(botones)

    edit.returnPressed.connect(dlg.accept)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        return edit.text(), True
    return "", False


def abrir_archivo(parent, titulo, directorio, filtro=""):
    """Selector de archivo no nativo, tematizado. Devuelve (ruta, "") como
    QFileDialog.getOpenFileName."""
    dlg = QFileDialog(parent, titulo, directorio, filtro)
    dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    if dlg.exec() == QFileDialog.DialogCode.Accepted and dlg.selectedFiles():
        return dlg.selectedFiles()[0], ""
    return "", ""


def guardar_archivo(parent, titulo, nombre_sugerido, filtro=""):
    """Selector de guardado no nativo, tematizado. Devuelve (ruta, "") como
    QFileDialog.getSaveFileName."""
    dlg = QFileDialog(parent, titulo, nombre_sugerido, filtro)
    dlg.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
    dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    if dlg.exec() == QFileDialog.DialogCode.Accepted and dlg.selectedFiles():
        return dlg.selectedFiles()[0], ""
    return "", ""


def elegir_directorio(parent, titulo, directorio=""):
    """Selector de carpeta no nativo, tematizado. Devuelve la ruta o None."""
    dlg = QFileDialog(parent, titulo, directorio)
    dlg.setFileMode(QFileDialog.FileMode.Directory)
    dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
    dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    if dlg.exec() == QFileDialog.DialogCode.Accepted and dlg.selectedFiles():
        return dlg.selectedFiles()[0]
    return None
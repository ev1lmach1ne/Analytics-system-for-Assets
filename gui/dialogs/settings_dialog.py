import os
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QLineEdit, QPushButton, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

from core.config import (
    get_base_data, set_base_data, PROJECT_ROOT,
)
from core.questdb_manager import is_reachable, detener_bundled
from gui.questdb_bootstrap import mostrar_bootstrap_questdb
from gui.dialogs.mensajes import elegir_directorio, aviso, informacion

STYLE = """
QDialog { background-color: #141e30; }
QLabel { color: #c8d6e5; }
QLabel#title { color: #4fc3f7; font-size: 15px; font-weight: bold; letter-spacing: 1px; }
QLabel#subtitle { color: #3a5a7a; font-size: 11px; }
QLabel#desc { color: #5a7a9a; font-size: 11px; }
QLineEdit {
    background-color: #1a2a45; color: #c8d6e5; border: 1px solid #253a60;
    padding: 10px 12px; border-radius: 4px; font-size: 12px;
}
QPushButton {
    background-color: #2a4a6a; color: #4fc3f7; border: none;
    padding: 10px 22px; border-radius: 4px; font-size: 12px; font-weight: bold;
}
QPushButton:hover { background-color: #3a5a8a; }
QPushButton:pressed { padding-top: 12px; padding-bottom: 8px; }
QPushButton#accept { background-color: #0f2a1a; color: #2ecc71; }
QPushButton#accept:hover { background-color: #1a3a2a; }
QPushButton#browse { background-color: #1a2a45; color: #7aaccc; }
QPushButton#browse:hover { background-color: #253a60; color: #c8d6e5; }
QFrame#sep { background-color: #253a60; max-height: 1px; }
QLabel#estadoOk { color: #2ecc71; font-size: 12px; font-weight: bold; }
QLabel#estadoMal { color: #e74c3c; font-size: 12px; font-weight: bold; }
"""


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analytics System · Ajustes")
        self.setModal(True)
        self.resize(560, 520)
        self.setStyleSheet(STYLE)

        icon_path = os.path.join(PROJECT_ROOT, "icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(14)

        title = QLabel("⚙  Ajustes")
        title.setObjectName("title")
        layout.addWidget(title)

        sep = QFrame()
        sep.setObjectName("sep")
        sep.setFixedHeight(1)
        layout.addWidget(sep)

        subtitle = QLabel("Carpeta del repositorio de datos:")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        path_layout = QHBoxLayout()
        path_layout.setSpacing(8)
        self.orig_path = get_base_data()
        self.path_input = QLineEdit(self.orig_path)
        path_layout.addWidget(self.path_input, 1)

        btn_browse = QPushButton(" Examinar...")
        btn_browse.setObjectName("browse")
        btn_browse.clicked.connect(self._browse)
        path_layout.addWidget(btn_browse)
        layout.addLayout(path_layout)

        desc = QLabel(
            " Carpeta raíz donde se guardan descargas, importaciones, limpiados e informes."
        )
        desc.setObjectName("desc")
        layout.addWidget(desc)

        sep2 = QFrame()
        sep2.setObjectName("sep")
        sep2.setFixedHeight(1)
        layout.addWidget(sep2)

        subtitle_qdb = QLabel("QuestDB (base de datos local):")
        subtitle_qdb.setObjectName("subtitle")
        layout.addWidget(subtitle_qdb)

        qdb_layout = QHBoxLayout()
        qdb_layout.setSpacing(8)
        self.lbl_qdb_estado = QLabel()
        qdb_layout.addWidget(self.lbl_qdb_estado, 1)
        btn_qdb_reiniciar = QPushButton(" Reiniciar")
        btn_qdb_reiniciar.setObjectName("browse")
        btn_qdb_reiniciar.clicked.connect(self._reiniciar_questdb)
        qdb_layout.addWidget(btn_qdb_reiniciar)
        layout.addLayout(qdb_layout)

        desc_qdb = QLabel(
            " La usa el paso «Limpiar» de Importar. Se descarga y arranca sola "
            "la primera vez que hace falta — este botón solo la reinicia si "
            "algo se ha quedado atascado."
        )
        desc_qdb.setObjectName("desc")
        desc_qdb.setWordWrap(True)
        layout.addWidget(desc_qdb)
        self._refrescar_estado_questdb()

        sep3 = QFrame()
        sep3.setObjectName("sep")
        sep3.setFixedHeight(1)
        layout.addWidget(sep3)

        subtitle_noticias = QLabel("Calendario económico:")
        subtitle_noticias.setObjectName("subtitle")
        layout.addWidget(subtitle_noticias)

        te_prox = QLabel(
            "Próximamente: el calendario económico está en revisión (sin "
            "proveedor gratuito fiable). El filtro «Evitar noticias» y las "
            "líneas de eventos en Resultados volverán cuando se estabilice "
            "la fuente de datos."
        )
        te_prox.setObjectName("desc")
        te_prox.setWordWrap(True)
        layout.addWidget(te_prox)

        layout.addStretch()

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        btn_apply = QPushButton(" Aplicar")
        btn_apply.setObjectName("accept")
        btn_apply.clicked.connect(self._apply)
        btn_layout.addWidget(btn_apply)

        btn_cancel = QPushButton(" Cancelar")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        layout.addLayout(btn_layout)

    def _refrescar_estado_questdb(self):
        if is_reachable():
            self.lbl_qdb_estado.setText("●  En ejecución")
            self.lbl_qdb_estado.setObjectName("estadoOk")
        else:
            self.lbl_qdb_estado.setText("●  No disponible")
            self.lbl_qdb_estado.setObjectName("estadoMal")
        # reaplicar el estilo tras cambiar objectName (Qt no lo hace solo)
        self.lbl_qdb_estado.style().unpolish(self.lbl_qdb_estado)
        self.lbl_qdb_estado.style().polish(self.lbl_qdb_estado)

    def _reiniciar_questdb(self):
        detener_bundled()
        mostrar_bootstrap_questdb(self)
        self._refrescar_estado_questdb()

    def _browse(self):
        cur = self.path_input.text() or get_base_data()
        if not os.path.isdir(cur):
            cur = ""
        path = elegir_directorio(self, "Seleccionar carpeta de datos", cur)
        if path:
            self.path_input.setText(path)

    def _apply(self):
        path = self.path_input.text().strip()
        if not path:
            return
        if not os.path.exists(path):
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                aviso(self, "Error", f"No se pudo crear la carpeta:\n{e}")
                return
        try:
            for sub in ("Limpiados", os.path.join("Limpiados", "Informes")):
                os.makedirs(os.path.join(path, sub), exist_ok=True)
        except Exception:
            pass
        if path != self.orig_path:
            set_base_data(path)
            informacion(
                self, "Ruta actualizada",
                "La nueva ruta de datos se ha guardado.\n\n"
                "Se aplicará al REINICIAR la aplicación: hasta entonces las "
                "pestañas abiertas siguen usando la carpeta anterior, para no "
                "dejar archivos a medias entre dos rutas.\n\n"
                "Cierra la aplicación y ábrela de nuevo cuando quieras "
                "trabajar con la nueva carpeta."
            )
        self.accept()
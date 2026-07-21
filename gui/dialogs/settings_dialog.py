import os
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QLineEdit, QPushButton, QFileDialog, QFrame,
                              QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

from core.config import get_base_data, set_base_data, PROJECT_ROOT

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
"""


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analytics System · Ajustes")
        self.setModal(True)
        self.resize(560, 320)
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

    def _browse(self):
        cur = self.path_input.text() or get_base_data()
        if not os.path.isdir(cur):
            cur = ""
        path = QFileDialog.getExistingDirectory(
            self, "Seleccionar carpeta de datos", cur
        )
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
                QMessageBox.warning(self, "Error", f"No se pudo crear la carpeta:\n{e}")
                return
        try:
            for sub in ("Limpiados", os.path.join("Limpiados", "Informes")):
                os.makedirs(os.path.join(path, sub), exist_ok=True)
        except Exception:
            pass
        if path != self.orig_path:
            set_base_data(path)
            QMessageBox.information(
                self, "Ruta actualizada",
                "La nueva ruta se ha guardado.\n"
                "Los cambios se aplicarán a las próximas operaciones."
            )
        self.accept()
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QLineEdit, QPushButton, QFormLayout,
                             QFrame, QDialogButtonBox)
from PyQt6.QtCore import Qt

TF_LABELS   = ['30s','1m','3m','5m','15m','30m','1h','2h','4h','1d']
TIPO_LABELS = ['Futuro/Cfd', 'Forex', 'Stock', 'Crypto']

STYLE_DIALOG = """
QDialog { background-color: #141e30; }
QLabel { color: #c8d6e5; font-size: 12px; }
QLineEdit, QComboBox {
    background-color: #1a2a45; color: #c8d6e5; border: 1px solid #253a60;
    padding: 6px 10px; border-radius: 4px; font-size: 12px;
}
QLineEdit:focus, QComboBox:focus { border-color: #4fc3f7; }
QComboBox::drop-down { border: none; background: #253a60; width: 24px; }
QComboBox QAbstractItemView {
    background-color: #1a2a45; color: #c8d6e5; selection-background-color: #2a4a6a;
    border: 1px solid #253a60;
}
QPushButton { background-color: #2a4a6a; color: #4fc3f7; border: none;
              padding: 8px 24px; border-radius: 4px; font-size: 12px; font-weight: bold; }
QPushButton:hover { background-color: #3a5a8a; }
QPushButton#cancel { background-color: #222a3a; color: #5a7a9a; }
QPushButton#cancel:hover { background-color: #2a3a4a; }
QFrame#sep { background-color: #253a60; max-height: 1px; }
"""

class ConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configurar Importacion")
        self.setFixedSize(420, 340)
        self.setStyleSheet(STYLE_DIALOG)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        title = QLabel("Configuracion del Activo")
        title.setStyleSheet("color: #4fc3f7; font-size: 15px; font-weight: bold;")
        layout.addWidget(title)

        sep = QFrame()
        sep.setObjectName("sep")
        layout.addWidget(sep)

        form = QFormLayout()
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.nombre = QLineEdit()
        self.nombre.setPlaceholderText("ej. xauusd, btc, spy")
        form.addRow("Nombre:", self.nombre)

        self.tipo = QComboBox()
        self.tipo.addItems(TIPO_LABELS)
        form.addRow("Tipo activo:", self.tipo)

        self.tf = QComboBox()
        self.tf.addItems(TF_LABELS)
        self.tf.setCurrentText("1h")
        form.addRow("Timeframe:", self.tf)

        layout.addLayout(form)
        layout.addStretch()

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_btn = QPushButton("Cancelar")
        cancel_btn.setObjectName("cancel")
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton("Importar")
        ok_btn.clicked.connect(self.accept)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(ok_btn)
        layout.addLayout(buttons)

    def get_config(self):
        tipo_map = {'Futuro/Cfd': 'FUTURO', 'Forex': 'FOREX',
                    'Stock': 'STOCK', 'Crypto': 'CRYPTO'}
        return {
            'nombre': self.nombre.text().strip().lower(),
            'tf': self.tf.currentText(),
            'activo': tipo_map[self.tipo.currentText()],
        }

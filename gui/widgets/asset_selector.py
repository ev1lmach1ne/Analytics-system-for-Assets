import os, re
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QComboBox,
                             QPushButton, QLabel, QCompleter, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal
from core.config import LIMPIADOS_DIR

STYLE_SELECTOR = """
QWidget { background-color: transparent; }
QComboBox {
    background-color: #1a2a45; color: #c8d6e5; border: none;
    padding: 6px 10px; border-radius: 4px; font-size: 12px; min-width: 220px;
}
QComboBox::drop-down { border: none; background: transparent; width: 22px; }
QComboBox::down-arrow { border: none; }
QComboBox QAbstractItemView {
    background-color: #1a2a45; color: #c8d6e5; selection-background-color: #2a4a6a;
    border: 1px solid #253a60; outline: none;
}
QPushButton {
    background-color: #2a4a6a; color: #4fc3f7; border: none;
    padding: 6px 14px; border-radius: 4px; font-size: 12px; font-weight: bold;
}
QPushButton:hover { background-color: #3a5a8a; }
QLabel { color: #3a5a7a; font-size: 10px; }
QFrame#sep { background-color: #253a60; max-height: 1px; }
"""

RE_LIMPIADO = re.compile(
    r'(.+?)_(\d+[smhd](?:\d+[smhd])?)_limpiado\.csv$', re.IGNORECASE
)
RE_LIMPIO = re.compile(
    r'(.+?)_(\d+[smhd](?:\d+[smhd])?)_limpio\.csv$', re.IGNORECASE
)

def scan_limpiados():
    """Recorre Limpiados/ recursivamente buscando *_limpiado.csv y *_limpio.csv.
    Returns list of (path, label, ticker, tf) sorted by label."""
    results = []
    if not os.path.isdir(LIMPIADOS_DIR):
        return results
    for root, dirs, files in os.walk(LIMPIADOS_DIR):
        if 'Informes' in root:
            continue
        for fname in files:
            fname_lower = fname.lower()
            if '_preparado' in fname_lower:
                continue
            m = RE_LIMPIADO.match(fname)
            if not m:
                m = RE_LIMPIO.match(fname)
            if m:
                ticker = m.group(1).lower()
                tf = m.group(2).lower()
                label = f"{ticker} {tf}"
                path = os.path.join(root, fname)
                results.append((path, label, ticker, tf))
    results.sort(key=lambda x: x[1])
    return results


class AssetSelector(QWidget):
    asset_selected = pyqtSignal(str, str, str)  # path, ticker, tf

    def __init__(self, parent=None):
        super().__init__(parent)
        self._assets = []  # [(path, label, ticker, tf)]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(6)

        lbl = QLabel("Activo:")
        lbl.setStyleSheet("color: #4fc3f7; font-size: 12px; font-weight: bold;")
        header.addWidget(lbl)

        self.combo = QComboBox()
        self.combo.setEditable(True)
        self.combo.setPlaceholderText("Buscar activo...")
        self.combo.currentIndexChanged.connect(self._on_selected)
        header.addWidget(self.combo, 1)

        self.btn_refresh = QPushButton("↻")
        self.btn_refresh.setFixedWidth(32)
        self.btn_refresh.clicked.connect(self.refresh)
        header.addWidget(self.btn_refresh)

        layout.addLayout(header)

        self.path_label = QLabel("")
        self.path_label.setObjectName("path")
        layout.addWidget(self.path_label)

        self.refresh()

    def refresh(self):
        old_label = self.combo.currentText()
        self._assets = scan_limpiados()
        self.combo.blockSignals(True)
        self.combo.clear()
        for path, label, ticker, tf in self._assets:
            self.combo.addItem(label, (path, ticker, tf))
        self.combo.blockSignals(False)

        # Restore selection if still exists
        idx = self.combo.findText(old_label)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)
        elif self._assets:
            self.combo.setCurrentIndex(0)
        else:
            self.combo.setEditText("")
            self.path_label.setText("No hay activos limpiados disponibles")

        # QCompleter with match-contains filter
        completer = QCompleter([a[1] for a in self._assets], self.combo)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.combo.setCompleter(completer)

    def current_asset(self):
        idx = self.combo.currentIndex()
        if 0 <= idx < len(self._assets):
            return self._assets[idx]
        return None

    def _on_selected(self, index):
        if 0 <= index < len(self._assets):
            path, label, ticker, tf = self._assets[index]
            self.path_label.setText(path)
            self.asset_selected.emit(path, ticker, tf)
        else:
            self.path_label.setText("")

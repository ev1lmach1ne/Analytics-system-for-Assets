"""
gui/dialogs/estilo.py
Hoja de estilos global (QSS) con la paleta oscura de la app para todos los
diálogos y ventanas auxiliares que no llevan estilos propios: QMessageBox,
QInputDialog, QFileDialog (modo no nativo), QCalendarWidget/QDateEdit, menús
contextuales, tooltips y barras de scroll.

Se aplica a nivel de aplicación (QApplication.setStyleSheet) desde app.py.
Los widgets que ya definen su propio setStyleSheet lo siguen usando: el QSS
del propio widget tiene prioridad sobre el global, así que las pestañas y los
diálogos ya tematizados no cambian de aspecto.
"""

ESTILO_GLOBAL = """
/* ── Contenedores modales sin tema propio ─────────────────────────── */
QMessageBox, QInputDialog, QFileDialog, QProgressDialog {
    background-color: #141e30;
}

/* ── Texto ────────────────────────────────────────────────────────── */
QMessageBox QLabel, QInputDialog QLabel, QFileDialog QLabel,
QProgressDialog QLabel {
    color: #c8d6e5; font-size: 12px;
}

/* ── Botones ──────────────────────────────────────────────────────── */
QMessageBox QPushButton, QInputDialog QPushButton, QFileDialog QPushButton,
QProgressDialog QPushButton {
    background-color: #2a4a6a; color: #4fc3f7; border: none;
    padding: 7px 18px; border-radius: 4px; font-size: 12px; font-weight: bold;
    min-width: 72px;
}
QMessageBox QPushButton:hover, QInputDialog QPushButton:hover,
QFileDialog QPushButton:hover, QProgressDialog QPushButton:hover {
    background-color: #3a5a8a;
}
QMessageBox QPushButton:pressed, QInputDialog QPushButton:pressed,
QFileDialog QPushButton:pressed, QProgressDialog QPushButton:pressed {
    background-color: #1e3a5a;
}

/* ── Campos de texto / combos de esos diálogos ────────────────────── */
QInputDialog QLineEdit, QFileDialog QLineEdit, QInputDialog QComboBox,
QFileDialog QComboBox, QInputDialog QSpinBox {
    background-color: #1a2a45; color: #c8d6e5;
    border: 1px solid #253a60; border-radius: 4px;
    padding: 6px 10px; font-size: 12px;
    selection-background-color: #2a4a6a; selection-color: #4fc3f7;
}
QInputDialog QLineEdit:focus, QFileDialog QLineEdit:focus,
QInputDialog QComboBox:focus, QFileDialog QComboBox:focus {
    border: 1px solid #3a5a8a;
}
QInputDialog QComboBox::drop-down, QFileDialog QComboBox::drop-down {
    border: none; background: transparent; width: 22px;
}

/* ── Listas/árboles/tablas del QFileDialog no nativo ──────────────── */
QFileDialog QListView, QFileDialog QTreeView, QFileDialog QTableView {
    background-color: #0d1424; color: #c8d6e5;
    border: 1px solid #253a60; outline: none;
}
QFileDialog QListView::item, QFileDialog QTreeView::item,
QFileDialog QTableView::item { padding: 3px 6px; }
QFileDialog QListView::item:selected, QFileDialog QTreeView::item:selected,
QFileDialog QTableView::item:selected {
    background-color: #2a4a6a; color: #4fc3f7;
}
QFileDialog QListView::item:hover, QFileDialog QTreeView::item:hover {
    background-color: #1e3050;
}
QFileDialog QHeaderView::section {
    background-color: #1a2a45; color: #aabbcc;
    border: none; padding: 4px 8px; font-size: 11px;
}
QFileDialog QToolButton {
    background-color: #1a2a45; color: #c8d6e5;
    border: 1px solid #253a60; border-radius: 4px; padding: 4px 8px;
}
QFileDialog QToolButton:hover { background-color: #253a60; }

/* ── Calendario (popup de QDateEdit) ──────────────────────────────── */
QCalendarWidget QWidget {
    background-color: #1a2a45; color: #c8d6e5;
}
QCalendarWidget QAbstractItemView {
    background-color: #1a2a45; color: #c8d6e5; outline: none;
    selection-background-color: #2a4a6a; selection-color: #4fc3f7;
}
QCalendarWidget QAbstractItemView:disabled {
    color: #3a5a7a;
}
QCalendarWidget QToolButton {
    background-color: #1a2a45; color: #4fc3f7; border: none;
    border-radius: 4px; padding: 4px 8px; font-size: 12px; font-weight: bold;
}
QCalendarWidget QToolButton:hover { background-color: #2a4a6a; }
QCalendarWidget QToolButton::menu-indicator { image: none; }
QCalendarWidget QSpinBox {
    background-color: #1a2a45; color: #c8d6e5; border: 1px solid #253a60;
    border-radius: 4px; padding: 2px 6px;
}
QCalendarWidget QSpinBox::up-button, QCalendarWidget QSpinBox::down-button {
    background-color: #2a4a6a; width: 16px; border: none;
}
QDateEdit {
    background-color: #1a2a45; color: #c8d6e5; border: 1px solid #253a60;
    border-radius: 4px; padding: 5px 8px; font-size: 12px;
}
QDateEdit:disabled { background-color: #101a2c; color: #3a5a7a; }
QDateEdit::drop-down { border: none; background: transparent; width: 22px; }

/* ── Menús contextuales ───────────────────────────────────────────── */
QMenu {
    background-color: #1a2a45; color: #c8d6e5;
    border: 1px solid #253a60; padding: 4px; font-size: 12px;
}
QMenu::item { padding: 6px 24px 6px 18px; border-radius: 4px; }
QMenu::item:selected { background-color: #2a4a6a; color: #4fc3f7; }
QMenu::item:disabled { color: #3a5a7a; }
QMenu::separator { height: 1px; background: #253a60; margin: 4px 8px; }

/* ── Tooltips ─────────────────────────────────────────────────────── */
QToolTip {
    background-color: #1a2a45; color: #c8d6e5;
    border: 1px solid #3a5a8a; padding: 4px 8px; font-size: 11px;
}

/* ── Barras de scroll ─────────────────────────────────────────────── */
QScrollBar:vertical { background: #1a2a45; width: 8px; }
QScrollBar::handle:vertical { background: #2a4a6a; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #1a2a45; height: 8px; }
QScrollBar::handle:horizontal { background: #2a4a6a; border-radius: 4px; min-width: 30px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::handle:hover { background: #3a5a8a; }
"""
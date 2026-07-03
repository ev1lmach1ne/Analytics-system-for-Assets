import os
from PyQt6.QtWidgets import (QListView, QWidget, QVBoxLayout, QPushButton,
                             QHBoxLayout, QToolButton)
from PyQt6.QtGui import QFileSystemModel, QPixmap, QPainter, QPen, QBrush, QColor, QFont, QIcon
from PyQt6.QtCore import QSortFilterProxyModel, Qt, QSize, QPoint

class CsvFilterModel(QSortFilterProxyModel):
    def __init__(self, exclude_patterns=None, parent=None):
        super().__init__(parent)
        self._exclude_patterns = exclude_patterns or ['_preparado', '_limpiado', '_preparado_preparado']

    def filterAcceptsRow(self, source_row, source_parent):
        model = self.sourceModel()
        if not model:
            return True
        index = model.index(source_row, 0, source_parent)
        file_name = model.fileName(index).lower()
        if model.isDir(index):
            return True
        if not (file_name.endswith('.csv') or file_name.endswith('.txt')):
            return False
        for pat in self._exclude_patterns:
            if pat in file_name:
                return False
        return True

class IncludeFilterModel(QSortFilterProxyModel):
    def __init__(self, include_patterns=None, parent=None):
        super().__init__(parent)
        self._include_patterns = include_patterns or ['_limpiado', '_limpio']

    def filterAcceptsRow(self, source_row, source_parent):
        model = self.sourceModel()
        if not model:
            return True
        index = model.index(source_row, 0, source_parent)
        file_name = model.fileName(index).lower()
        if model.isDir(index):
            return True
        if not (file_name.endswith('.csv') or file_name.endswith('.txt')):
            return False
        for pat in self._include_patterns:
            if pat in file_name:
                return True
        return False

class NoFilterModel(QSortFilterProxyModel):
    def filterAcceptsRow(self, source_row, source_parent):
        model = self.sourceModel()
        if not model:
            return True
        index = model.index(source_row, 0, source_parent)
        file_name = model.fileName(index).lower()
        if model.isDir(index):
            return True
        if not (file_name.endswith('.csv') or file_name.endswith('.txt')):
            return False
        return True

class FileExplorer(QWidget):
    def __init__(self, root_path, parent=None, mode='csv'):
        super().__init__(parent)
        self._root_path = root_path
        self._nav_stack = []

        self.setStyleSheet("""
            QPushButton, QToolButton {
                background-color: #2a4a6a; color: #4fc3f7; border: none;
                border-radius: 4px; font-weight: bold; font-size: 13px;
            }
QPushButton:hover, QToolButton:hover { background-color: #3a5a8a; }
QPushButton:pressed, QToolButton:pressed { padding-top: 10px; padding-bottom: 6px; }
QPushButton:disabled, QToolButton:disabled { background-color: #1a2a45; color: #3a5a7a; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        nav = QHBoxLayout()
        nav.setSpacing(4)

        self.btn_home = QPushButton()
        pix = QPixmap(20, 20)
        pix.fill(Qt.GlobalColor.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor('#4fc3f7'), 2)
        p.setPen(pen)
        p.setBrush(QBrush(QColor('#4fc3f7')))
        # Roof triangle
        pts = [QPoint(10, 2), QPoint(2, 10), QPoint(18, 10)]
        p.drawPolygon(pts)
        # Walls
        p.drawRect(4, 10, 12, 8)
        # Door
        p.setBrush(QBrush())
        p.drawRect(8, 13, 4, 5)
        p.end()
        self.btn_home.setIcon(QIcon(pix))
        self.btn_home.setIconSize(QSize(20, 20))
        self.btn_home.setToolTip("Ir a la carpeta raiz")
        self.btn_home.clicked.connect(self.go_home)
        self.btn_home.setFixedSize(30, 28)
        nav.addWidget(self.btn_home)

        self.btn_up = QToolButton()
        self.btn_up.setArrowType(Qt.ArrowType.UpArrow)
        self.btn_up.setToolTip("Subir al directorio padre")
        self.btn_up.clicked.connect(self.go_up)
        self.btn_up.setEnabled(False)
        self.btn_up.setFixedSize(30, 28)
        nav.addWidget(self.btn_up)

        nav.addStretch()
        layout.addLayout(nav)

        self.model = QFileSystemModel()
        self.model.setRootPath(root_path)
        self.model.setNameFilters(['*.csv', '*.txt', '*.CSV', '*.TXT'])
        self.model.setNameFilterDisables(False)

        if mode == 'include':
            self.filter_model = IncludeFilterModel(parent=self)
        elif mode == 'exclude':
            self.filter_model = CsvFilterModel(parent=self)
        else:
            self.filter_model = NoFilterModel(parent=self)
        self.filter_model.setSourceModel(self.model)

        self.list_view = QListView()
        self.list_view.setModel(self.filter_model)
        self.list_view.setRootIndex(self.filter_model.mapFromSource(self.model.index(root_path)))
        self.list_view.setViewMode(QListView.ViewMode.IconMode)
        self.list_view.setIconSize(QSize(64, 64))
        self.list_view.setGridSize(QSize(120, 100))
        self.list_view.setWordWrap(True)
        self.list_view.setSpacing(10)
        self.list_view.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self.list_view.doubleClicked.connect(self._on_double_click)

        layout.addWidget(self.list_view, 1)

    def _on_double_click(self, index):
        try:
            if not index.isValid():
                return
            proxy_index = self.filter_model.mapToSource(index)
            if not proxy_index.isValid():
                return
            path = self.model.filePath(proxy_index)
            if os.path.isdir(path):
                old_root = self.model.rootPath()
                if old_root != path:
                    self._nav_stack.append(old_root)
                self.model.setRootPath(path)
                new_root = self.filter_model.mapFromSource(self.model.index(path))
                self.list_view.setRootIndex(new_root)
                self.btn_up.setEnabled(True)
        except Exception:
            pass

    def go_up(self):
        if self._nav_stack:
            prev = self._nav_stack.pop()
            self.model.setRootPath(prev)
            self.list_view.setRootIndex(
                self.filter_model.mapFromSource(self.model.index(prev))
            )
            self.btn_up.setEnabled(len(self._nav_stack) > 0)

    def go_home(self):
        self.set_root_path(self._root_path)

    def set_root_path(self, root_path):
        self._nav_stack.clear()
        self._root_path = root_path
        self.model.setRootPath(root_path)
        self.list_view.setRootIndex(self.filter_model.mapFromSource(self.model.index(root_path)))
        self.btn_up.setEnabled(False)

    def refresh(self):
        root = self.model.rootPath()
        self.model.setRootPath('')
        self.model.setRootPath(root)
        self.list_view.setRootIndex(self.filter_model.mapFromSource(self.model.index(root)))
        self.btn_up.setEnabled(False)

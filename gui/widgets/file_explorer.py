import os
from PyQt6.QtWidgets import (QListView, QWidget, QVBoxLayout, QPushButton,
                             QHBoxLayout, QToolButton, QStyle, QFileIconProvider)
from PyQt6.QtGui import QFileSystemModel, QPixmap, QPainter, QPen, QBrush, QColor, QFont, QIcon
from PyQt6.QtCore import (QSortFilterProxyModel, Qt, QSize, QPoint,
                          QAbstractListModel, QModelIndex, QFileInfo, pyqtSignal)

_MAX_PROFUNDIDAD_BUSQUEDA = 6


def _asegurar_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass

class CsvFilterModel(QSortFilterProxyModel):
    def __init__(self, exclude_patterns=None, parent=None):
        super().__init__(parent)
        self._exclude_patterns = exclude_patterns or ['_preparado', '_limpiado', '_preparado_preparado']
        self._search_text = ''

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
        if self._search_text and self._search_text not in file_name:
            return False
        return True

class IncludeFilterModel(QSortFilterProxyModel):
    def __init__(self, include_patterns=None, parent=None):
        super().__init__(parent)
        self._include_patterns = include_patterns or ['_limpiado', '_limpio']
        self._search_text = ''

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
        if self._search_text and self._search_text not in file_name:
            return False
        return False

class NoFilterModel(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._search_text = ''

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
        if self._search_text and self._search_text not in file_name:
            return False
        return True

class _SearchResultsModel(QAbstractListModel):
    """Lista aplanada de rutas absolutas (resultados de búsqueda recursiva)."""

    PATH_ROLE = Qt.ItemDataRole.UserRole + 1

    def __init__(self, paths=None, parent=None):
        super().__init__(parent)
        self._paths = paths or []
        self._icon_provider = QFileIconProvider()

    def rowCount(self, parent=QModelIndex()):
        return len(self._paths)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._paths)):
            return None
        path = self._paths[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return os.path.basename(path)
        if role == Qt.ItemDataRole.DecorationRole:
            return self._icon_provider.icon(QFileInfo(path))
        if role == self.PATH_ROLE:
            return path
        return None

    def path_at(self, index):
        if not index.isValid():
            return None
        return index.data(self.PATH_ROLE)


class _ListaConAviso(QListView):
    """QListView que pinta un texto centrado cuando la carpeta está vacía."""

    def __init__(self, empty_hint=None, parent=None):
        super().__init__(parent)
        self._empty_hint = empty_hint

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._empty_hint:
            return
        model = self.model()
        if model is None or model.rowCount(self.rootIndex()) > 0:
            return
        painter = QPainter(self.viewport())
        painter.setPen(QColor('#5a7a9a'))
        painter.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter,
                         self._empty_hint)
        painter.end()


class FileExplorer(QWidget):
    file_chosen = pyqtSignal(str)

    def __init__(self, root_path, parent=None, mode='csv', empty_hint=None):
        super().__init__(parent)
        # La raíz debe existir siempre: con una ruta inexistente,
        # QFileSystemModel.index() devuelve un índice inválido y la vista
        # cae a mostrar la raíz del sistema de archivos.
        _asegurar_dir(root_path)
        self._root_path = root_path
        self._nav_stack = []
        self._mode = mode
        self._empty_hint = empty_hint
        self._search_text = ''
        self._searching = False

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

        self.btn_reload = QPushButton()
        self.btn_reload.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.btn_reload.setToolTip("Recargar explorador")
        self.btn_reload.clicked.connect(self.refresh)
        self.btn_reload.setFixedSize(30, 28)
        nav.addWidget(self.btn_reload)

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
        self._search_model = _SearchResultsModel(parent=self)

        self.list_view = _ListaConAviso(empty_hint=self._empty_hint)
        self.list_view.setModel(self.filter_model)
        self.list_view.setRootIndex(self.filter_model.mapFromSource(self.model.index(root_path)))
        self.list_view.setViewMode(QListView.ViewMode.IconMode)
        self.list_view.setIconSize(QSize(64, 64))
        self.list_view.setGridSize(QSize(120, 100))
        self.list_view.setWordWrap(True)
        self.list_view.setSpacing(10)
        self.list_view.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self.list_view.doubleClicked.connect(self._on_double_click)
        self.list_view.clicked.connect(self._on_item_clicked)

        layout.addWidget(self.list_view, 1)

    def _on_double_click(self, index):
        if self._searching:
            return
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
        if self._searching:
            self._search_text = ''
            self._salir_modo_busqueda()
        _asegurar_dir(root_path)
        self._nav_stack.clear()
        self._root_path = root_path
        self.model.setRootPath(root_path)
        self.list_view.setRootIndex(self.filter_model.mapFromSource(self.model.index(root_path)))
        self.btn_up.setEnabled(False)

    def refresh(self):
        if self._searching:
            # Re-ejecutar la búsqueda actual con el árbol de archivos actualizado
            self._entrar_modo_busqueda()
            return
        root = self.model.rootPath()
        self.model.setRootPath('')
        self.model.setRootPath(root)
        self.list_view.setRootIndex(self.filter_model.mapFromSource(self.model.index(root)))
        self.btn_up.setEnabled(False)

    def set_search_text(self, text: str):
        self._search_text = text.strip().lower()
        if self._search_text:
            self._entrar_modo_busqueda()
        else:
            self._salir_modo_busqueda()

    def _entrar_modo_busqueda(self):
        # Búsqueda recursiva desde la raíz: los datos están organizados una
        # subcarpeta por activo, así que buscar solo en la carpeta mostrada
        # (single-level, límite de QListView/QFileSystemModel) nunca encuentra
        # los CSV, que viven un nivel más abajo.
        self._searching = True
        resultados = self._buscar_recursivo(self._root_path, self._search_text)
        self._search_model.beginResetModel()
        self._search_model._paths = resultados
        self._search_model.endResetModel()
        if self.list_view.model() is not self._search_model:
            self.list_view.setModel(self._search_model)
        self.btn_home.setEnabled(False)
        self.btn_up.setEnabled(False)
        self.btn_reload.setEnabled(False)

    def _salir_modo_busqueda(self):
        self._searching = False
        if self.list_view.model() is not self.filter_model:
            self.list_view.setModel(self.filter_model)
        self.list_view.setRootIndex(
            self.filter_model.mapFromSource(self.model.index(self.model.rootPath()))
        )
        self.btn_home.setEnabled(True)
        self.btn_reload.setEnabled(True)
        self.btn_up.setEnabled(len(self._nav_stack) > 0)

    def _buscar_recursivo(self, root, texto):
        exclude_patterns = ['_preparado', '_limpiado', '_preparado_preparado']
        include_patterns = ['_limpiado', '_limpio']
        resultados = []
        root = os.path.normpath(root)
        base_depth = root.count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath.count(os.sep) - base_depth >= _MAX_PROFUNDIDAD_BUSQUEDA:
                dirnames[:] = []
                continue
            for fname in filenames:
                name_lower = fname.lower()
                if not (name_lower.endswith('.csv') or name_lower.endswith('.txt')):
                    continue
                if self._mode == 'exclude' and any(p in name_lower for p in exclude_patterns):
                    continue
                if self._mode == 'include' and not any(p in name_lower for p in include_patterns):
                    continue
                if texto not in name_lower:
                    continue
                resultados.append(os.path.join(dirpath, fname))
        resultados.sort(key=lambda p: os.path.basename(p).lower())
        return resultados

    def _on_item_clicked(self, index):
        path = self._resolver_ruta(index)
        if path and not os.path.isdir(path):
            self.file_chosen.emit(path)

    def _resolver_ruta(self, index):
        if not index.isValid():
            return None
        if self._searching:
            return self._search_model.path_at(index)
        source_index = self.filter_model.mapToSource(index)
        if not source_index.isValid():
            return None
        return self.model.filePath(source_index)

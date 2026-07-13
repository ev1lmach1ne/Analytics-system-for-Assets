import os
import gc
from PyQt6.QtWidgets import (QWidget, QScrollArea, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QToolButton, QSizePolicy)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap
from PyQt6.QtPdf import QPdfDocument, QPdfDocumentRenderOptions

STYLE_PDF = """
QWidget { background-color: transparent; }
QPushButton {
    background-color: #2a4a6a; color: #4fc3f7; border: none;
    padding: 6px 14px; border-radius: 4px; font-size: 11px; font-weight: bold;
}
QPushButton:hover { background-color: #3a5a8a; }
QPushButton:pressed { padding-top: 10px; padding-bottom: 6px; }
QPushButton:disabled { background-color: #1a2a45; color: #3a5a7a; }
QLabel#pageTitle {
    color: #4fc3f7; font-size: 11px; font-weight: bold;
    padding: 2px 0 0 0; margin: 0;
}
QLabel#pageImage {
    border: 1px solid #253a60; border-radius: 4px;
    padding: 0; margin: 0;
}
"""


class PdfViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._doc = None
        self._pdf_path = None
        self._pages = []
        self._page_images = []
        self._page_titles = []
        self._current_page = 0
        self._page_count = 0
        self._visible_pages = None  # None = todas; si no, lista ordenada de índices visibles
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QLabel("Graficos del informe")
        header.setStyleSheet("color: #4fc3f7; font-size: 12px; font-weight: bold;")
        layout.addWidget(header)

        nav = QHBoxLayout()
        nav.setSpacing(8)

        self.btn_prev = QToolButton()
        self.btn_prev.setArrowType(Qt.ArrowType.LeftArrow)
        self.btn_prev.setFixedSize(30, 28)
        self.btn_prev.clicked.connect(self._prev_page)
        nav.addWidget(self.btn_prev)

        self.lbl_page = QLabel("0 / 0")
        self.lbl_page.setStyleSheet("color: #aabbcc; font-size: 11px;")
        self.lbl_page.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav.addWidget(self.lbl_page)

        self.btn_next = QToolButton()
        self.btn_next.setArrowType(Qt.ArrowType.RightArrow)
        self.btn_next.setFixedSize(30, 28)
        self.btn_next.clicked.connect(self._next_page)
        nav.addWidget(self.btn_next)

        nav.addStretch()

        self.btn_open = QPushButton("Abrir PDF externo")
        self.btn_open.clicked.connect(self._open_pdf)
        nav.addWidget(self.btn_open)

        layout.addLayout(nav)

        self.gallery = QWidget()
        self.gallery.setStyleSheet("background-color: transparent;")
        self.gallery_layout = QVBoxLayout(self.gallery)
        self.gallery_layout.setContentsMargins(0, 0, 0, 0)
        self.gallery_layout.setSpacing(16)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.gallery)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self.scroll, 1)

    def load(self, pdf_path):
        self._pdf_path = pdf_path
        self._current_page = 0
        self._visible_pages = None

        # Cerrar el documento anterior explícitamente: si no, el archivo PDF
        # queda con el handle abierto y no se puede borrar ni sobrescribir en
        # Windows. Importante: el documento se crea SIN padre Qt (más abajo,
        # `QPdfDocument(None)`) — si se parenta al widget, Qt le transfiere el
        # ownership a C++ y ni `close()` ni `del` desde Python logran liberar
        # el archivo hasta que el propio widget se destruye.
        if self._doc is not None:
            self._doc.close()
            self._doc = None
            gc.collect()

        for w in self._pages:
            self.gallery_layout.removeWidget(w)
            w.deleteLater()
        self._pages.clear()
        self._page_images.clear()
        self._page_titles.clear()

        if not pdf_path or not os.path.exists(pdf_path):
            self._doc = None
            self._page_count = 0
            self.lbl_page.setText("Sin PDF")
            return

        doc = QPdfDocument(None)
        doc.load(pdf_path)
        self._doc = doc

        self._page_count = doc.pageCount()
        self.lbl_page.setText(f"{self._page_count} paginas")

        for i in range(self._page_count):
            pw, img, title = self._create_page_widget(i)
            self._pages.append(pw)
            self._page_images.append(img)
            self._page_titles.append(title)
            self.gallery_layout.addWidget(pw)

        self._update_page_titles()
        self._update_nav()
        self._render_pages()

    def _create_page_widget(self, index):
        w = QWidget()
        w_layout = QVBoxLayout(w)
        w_layout.setContentsMargins(0, 0, 0, 0)
        w_layout.setSpacing(4)

        # El texto se rellena en _update_page_titles(): el indice absoluto
        # dentro del PDF no sirve de titulo, porque con paginas cacheadas
        # por Ventana (Metricas/Regimen ER/Dashboard NATR se generan una vez
        # por cada Ventana) la posicion absoluta de una pagina visible puede
        # ser cualquiera (ej. la 7) aunque sea la primera que se ve al
        # filtrar por la Ventana seleccionada.
        title = QLabel("")
        title.setObjectName("pageTitle")
        w_layout.addWidget(title)

        img = QLabel()
        img.setObjectName("pageImage")
        img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img.setText("(cargando...)")
        w_layout.addWidget(img)

        return w, img, title

    def _update_page_titles(self):
        """Numera cada pagina VISIBLE por su posicion relativa dentro del
        filtro de Ventana actual (1, 2, 3...), no por su indice absoluto en
        el PDF completo."""
        vis = self._indices_visibles()
        for pos, idx in enumerate(vis):
            self._page_titles[idx].setText(f"Pagina {pos + 1} / {len(vis)}")

    def _target_size(self):
        avail = self.width() - 40
        return QSize(max(avail, 200), max(int(avail * 0.65), 200))

    def _render_pages(self):
        if not self._doc:
            return
        target = self._target_size()
        opts = QPdfDocumentRenderOptions()
        opts.setScaledSize(target)
        for i, img_label in enumerate(self._page_images):
            rendered = self._doc.render(i, target, opts)
            pix = QPixmap.fromImage(rendered)
            img_label.setPixmap(pix)

    def _indices_visibles(self):
        if self._visible_pages is None:
            return list(range(self._page_count))
        return self._visible_pages

    def set_visible_pages(self, indices):
        """Filtra qué páginas se muestran (None = todas).

        `indices` son índices 0-based del documento. Las páginas ocultas no se
        destruyen, solo se esconden: el cambio de Ventana es instantáneo.
        """
        if indices is None:
            self._visible_pages = None
        else:
            self._visible_pages = sorted(i for i in set(indices) if 0 <= i < self._page_count)
        vis = set(self._indices_visibles())
        for i, w in enumerate(self._pages):
            w.setVisible(i in vis)
        if vis and self._current_page not in vis:
            self._current_page = min(vis)
        self._update_page_titles()
        self._update_nav()

    def _update_nav(self):
        vis = self._indices_visibles()
        if not vis:
            self.lbl_page.setText("0 / 0")
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            return
        pos = vis.index(self._current_page) if self._current_page in vis else 0
        self.lbl_page.setText(f"{pos + 1} / {len(vis)}")
        self.btn_prev.setEnabled(pos > 0)
        self.btn_next.setEnabled(pos < len(vis) - 1)

    def _scroll_to_page(self, index):
        if 0 <= index < len(self._pages):
            self._current_page = index
            self.scroll.ensureWidgetVisible(self._pages[index], 0, 0)
            self._update_nav()

    @property
    def page_count(self):
        return self._page_count

    def go_to_page(self, index):
        self._scroll_to_page(index)

    def _prev_page(self):
        vis = self._indices_visibles()
        if self._current_page in vis:
            pos = vis.index(self._current_page)
            if pos > 0:
                self._scroll_to_page(vis[pos - 1])
        elif vis:
            self._scroll_to_page(vis[0])

    def _next_page(self):
        vis = self._indices_visibles()
        if self._current_page in vis:
            pos = vis.index(self._current_page)
            if pos < len(vis) - 1:
                self._scroll_to_page(vis[pos + 1])
        elif vis:
            self._scroll_to_page(vis[0])

    def _open_pdf(self):
        if self._pdf_path and os.path.exists(self._pdf_path):
            os.startfile(self._pdf_path)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Left:
            self._prev_page()
        elif event.key() == Qt.Key.Key_Right:
            self._next_page()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_pages()

import os
from PyQt6.QtWidgets import (QWidget, QScrollArea, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFrame, QSizePolicy)
from PyQt6.QtCore import Qt, QSize, QObject
from PyQt6.QtGui import QPixmap
from PyQt6.QtPdf import QPdfDocument, QPdfDocumentRenderOptions

STYLE_PDF = """
QWidget { background-color: transparent; }
QPushButton {
    background-color: #2a4a6a; color: #4fc3f7; border: none;
    padding: 6px 14px; border-radius: 4px; font-size: 11px; font-weight: bold;
}
QPushButton:hover { background-color: #3a5a8a; }
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

class PdfPageWidget(QWidget):
    def __init__(self, page_index, pdf_path, parent=None):
        super().__init__(parent)
        self._page_index = page_index
        self._pdf_path = pdf_path
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.title_label = QLabel(f"Pagina {page_index + 1}")
        self.title_label.setObjectName("pageTitle")
        layout.addWidget(self.title_label)

        self.image_label = QLabel()
        self.image_label.setObjectName("pageImage")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.image_label.setText("(cargando...)")
        layout.addWidget(self.image_label)

    def set_pixmap(self, pix):
        self.image_label.setPixmap(pix)

    def mousePressEvent(self, event):
        if self._pdf_path and os.path.exists(self._pdf_path):
            os.startfile(self._pdf_path)


class PdfViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._doc = None
        self._pdf_path = None
        self._pages = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QLabel("Graficos del informe")
        header.setStyleSheet("color: #4fc3f7; font-size: 12px; font-weight: bold;")
        layout.addWidget(header)

        nav = QHBoxLayout()
        nav.setSpacing(8)
        self.lbl_page = QLabel("0 / 0")
        self.lbl_page.setStyleSheet("color: #aabbcc; font-size: 11px;")
        nav.addWidget(self.lbl_page)
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

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.gallery)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(scroll, 1)

    def load(self, pdf_path):
        self._pdf_path = pdf_path
        for w in self._pages:
            self.gallery_layout.removeWidget(w)
            w.deleteLater()
        self._pages.clear()

        if not pdf_path or not os.path.exists(pdf_path):
            self.lbl_page.setText("Sin PDF")
            return

        parent = QObject(self)
        doc = QPdfDocument(parent)
        doc.load(pdf_path)
        self._doc = doc

        count = doc.pageCount()
        self.lbl_page.setText(f"{count} paginas")

        for i in range(count):
            pw = PdfPageWidget(i, pdf_path)
            self._pages.append(pw)
            self.gallery_layout.addWidget(pw)

        self._render_pages()

    def _target_size(self):
        avail = self.width() - 40
        return QSize(max(avail, 200), max(int(avail * 0.65), 200))

    def _render_pages(self):
        if not self._doc:
            return
        target = self._target_size()
        opts = QPdfDocumentRenderOptions()
        opts.setScaledSize(target)
        for i, pw in enumerate(self._pages):
            img = self._doc.render(i, target, opts)
            pix = QPixmap.fromImage(img)
            pw.set_pixmap(pix)

    def _open_pdf(self):
        if self._pdf_path and os.path.exists(self._pdf_path):
            os.startfile(self._pdf_path)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_pages()

import os, json, re
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QComboBox, QFrame, QFileDialog, QTextBrowser,
                             QTabWidget, QProgressBar, QScrollArea, QSizePolicy)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from gui.widgets.pdf_viewer import PdfViewer

STYLE_ANALISIS = """
QWidget { background-color: #141e30; }
QPushButton {
    background-color: #2a4a6a; color: #4fc3f7; border: none;
    padding: 8px 18px; border-radius: 4px; font-size: 12px; font-weight: bold;
}
QPushButton:hover { background-color: #3a5a8a; }
QPushButton:pressed { padding-top: 10px; padding-bottom: 6px; }
QPushButton:disabled { background-color: #1a2a45; color: #3a5a7a; }
QPushButton#export { background-color: #0f2a1a; color: #2ecc71; }
QPushButton#export:hover { background-color: #1a3a2a; }
QPushButton#export:pressed { padding-top: 10px; padding-bottom: 6px; }
QComboBox {
    background-color: #1a2a45; color: #c8d6e5; border: none;
    padding: 6px 10px; border-radius: 4px; font-size: 12px; min-width: 140px;
}
QComboBox::drop-down { border: none; background: transparent; width: 22px; }
QComboBox::down-arrow { border: none; }
QComboBox QAbstractItemView {
    background-color: #1a2a45; color: #c8d6e5; selection-background-color: #2a4a6a;
    border: 1px solid #253a60; outline: none;
}
QProgressBar {
    background-color: #1a2a45; border: none;
    border-radius: 7px; height: 16px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 rgba(255,255,255,0.30),
                                stop:0.4 #6dd5fa,
                                stop:0.7 #4fc3f7,
                                stop:1 rgba(0,0,0,0.2));
    border-radius: 6px;
}
QTabWidget::pane { background-color: #0d1424; border: 1px solid #253a60; border-top: none; }
QTabBar { background-color: #1a2a45; border: none; }
QTabBar::tab {
    background-color: #1a2a45; color: #5a7a9a; padding: 8px 20px;
    border: none; border-right: 1px solid #253a60; font-size: 11px;
}
QTabBar::tab:selected { background-color: #0d1424; color: #4fc3f7; font-weight: bold; }
QFrame#sep { background-color: #253a60; max-height: 1px; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
"""

_ANSI_RE = re.compile(r'\033\[([\d;]+)m')

ANSI_COLORS = {
    91:  '#e74c3c',
    92:  '#27ae60',
    93:  '#f1c40f',
    94:  '#3498db',
    95:  '#9b59b6',
    96:  '#1abc9c',
}

BAR_RE = re.compile(r'^([█░]{12})\s+(.*)$')

HORIZONTES = [
    ('General',     'Todas las metricas'),
    ('Scalping',    'TF <= 15m'),
    ('Daytrading',  'TF <= 1h'),
    ('Swingtrading','Todos los TF'),
    ('Position',    'TF >= 4h'),
]


def _rich_value(text):
    parts = _ANSI_RE.split(text)
    out = []
    open_tags = []
    for part in parts:
        if not part:
            continue
        m = _ANSI_RE.fullmatch(f'\033[{part}m')
        if m:
            codes = [int(c) for c in part.split(';')]
            for c in codes:
                if c == 0:
                    while open_tags:
                        t = open_tags.pop()
                        if t.startswith('<span'):
                            out.append('</span>')
                        elif t == '<b>':
                            out.append('</b>')
                elif c == 1:
                    out.append('<b>')
                    open_tags.append('<b>')
                elif c in ANSI_COLORS:
                    tag = f'<span style="color:{ANSI_COLORS[c]}">'
                    out.append(tag)
                    open_tags.append(tag)
        else:
            out.append(part.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))
    while open_tags:
        t = open_tags.pop()
        if t.startswith('<span'):
            out.append('</span>')
        elif t == '<b>':
            out.append('</b>')
    return ''.join(out)


SCROLL_STYLE = """
QWidget#metricsContainer {
    background-color: #0d1424;
}
QLabel#catHeader {
    color: #4fc3f7; font-size: 12px; font-weight: bold;
    padding: 0px; margin: 0px;
}
QLabel#metricName {
    color: #aabbcc; font-size: 11px;
}
QLabel#metricValue {
    color: #e0e0e0; font-size: 11px;
}
"""


class MetricRow(QWidget):
    def __init__(self, name, value_text, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(8)

        self.name_label = QLabel(name)
        self.name_label.setObjectName("metricName")
        self.name_label.setFixedWidth(220)

        m = BAR_RE.match(value_text)
        if m:
            filled = m.group(1).count('█')
            rest = m.group(2).strip()

            self.bar = QProgressBar()
            self.bar.setMinimum(0)
            self.bar.setMaximum(12)
            self.bar.setValue(filled)
            self.bar.setTextVisible(False)
            self.bar.setFixedHeight(16)

            self.value_label = QLabel(rest)
            self.value_label.setObjectName("metricValue")
            self.value_label.setFixedWidth(110)

            layout.addWidget(self.name_label)
            layout.addWidget(self.bar, 1)
            layout.addWidget(self.value_label)
        else:
            if '\033[' in value_text:
                rich = _rich_value(value_text)
                self.value_label = QLabel(rich)
            else:
                self.value_label = QLabel(value_text)
            self.value_label.setObjectName("metricValue")
            self.value_label.setWordWrap(True)

            layout.addWidget(self.name_label)
            layout.addWidget(self.value_label, 1)


class CategoryGroup(QWidget):
    def __init__(self, title, metrics, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(2)

        header = QLabel(f"\u25b6 {title}")
        header.setObjectName("catHeader")
        layout.addWidget(header)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #253a60;")
        layout.addWidget(sep)

        for metrica, valor in metrics.items():
            if not metrica.strip() or not str(valor).strip():
                continue
            row = MetricRow(metrica, str(valor))
            layout.addWidget(row)


class MetricsScroll(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.container = QWidget()
        self.container.setObjectName("metricsContainer")
        self.container.setStyleSheet(SCROLL_STYLE)
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(6)

        self.setWidget(self.container)

    def clear(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def populate(self, metricas):
        self.clear()
        period_text = ''
        first = True
        for cat, items in metricas.items():
            if not first:
                spacer = QFrame()
                spacer.setFixedHeight(4)
                spacer.setStyleSheet("background: transparent;")
                self.layout.addWidget(spacer)
            first = False

            if cat.startswith('1.'):
                per_val = items.get('Periodo', '')
                if per_val:
                    period_text = str(per_val)

            group = CategoryGroup(cat, items)
            self.layout.addWidget(group)

        self.layout.addStretch()
        return period_text


class TabAnalisis(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(STYLE_ANALISIS)
        self._pdf_path = None
        self._metrics_path = None
        self._all_metrics = None
        self._ticker = ''
        self._tf = ''

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.lbl_period = QLabel("")
        self.lbl_period.setStyleSheet("color: #3a5a7a; font-size: 11px;")
        toolbar.addWidget(self.lbl_period)

        self.lbl_asset = QLabel("Ningun activo seleccionado")
        self.lbl_asset.setStyleSheet("color: #4fc3f7; font-size: 16px; font-weight: bold; padding-left: 12px;")
        toolbar.addWidget(self.lbl_asset)

        toolbar.addStretch()

        lbl_horizon = QLabel("Ventana")
        lbl_horizon.setStyleSheet("color: #aabbcc; font-size: 11px; font-weight: bold; padding-right: 4px;")
        toolbar.addWidget(lbl_horizon)

        self.horizon = QComboBox()
        self.horizon.addItems(["General", "Scalping", "Daytrading", "Swingtrading", "Position"])
        self.horizon.setToolTip("Horizonte de analisis")
        toolbar.addWidget(self.horizon)

        self.btn_apply = QPushButton("Aplicar")
        self.btn_apply.clicked.connect(self._render_metrics)
        toolbar.addWidget(self.btn_apply)

        self.btn_export = QPushButton(" Exportar PDF")
        self.btn_export.setObjectName("export")
        self.btn_export.clicked.connect(self._export_pdf)
        self.btn_export.setEnabled(False)
        toolbar.addWidget(self.btn_export)

        layout.addLayout(toolbar)

        sep = QFrame()
        sep.setObjectName("sep")
        layout.addWidget(sep)

        self.inner_tabs = QTabWidget()
        self.inner_tabs.setDocumentMode(True)

        self.metrics_scroll = MetricsScroll()
        self.inner_tabs.addTab(self.metrics_scroll, "  Metricas  ")

        self.graphs_viewer = PdfViewer()
        self.inner_tabs.addTab(self.graphs_viewer, "  Graficos  ")

        layout.addWidget(self.inner_tabs, 1)

    @property
    def current_horizon(self):
        return self.horizon.currentText()

    def _update_horizon_items(self, tf):
        for i in range(self.horizon.count()):
            self.horizon.model().item(i).setEnabled(True)
        if tf:
            m = re.match(r'(\d+)([smhd])', tf)
            if m:
                num, unit = int(m.group(1)), m.group(2)
                seconds = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}[unit] * num
                if seconds >= 3600:
                    self.horizon.model().item(1).setEnabled(False)
                    if self.horizon.currentIndex() == 1:
                        self.horizon.setCurrentIndex(0)

    def load_results(self, pdf_path, metrics_path, ticker, tf):
        self._pdf_path = pdf_path if pdf_path and os.path.exists(pdf_path) else None
        self._metrics_path = metrics_path if metrics_path and os.path.exists(metrics_path) else None
        self._ticker = ticker or ''
        self._tf = tf or ''

        self.lbl_asset.setText(f"{ticker} {tf}" if ticker and tf else "Sin activo")
        self.btn_export.setEnabled(self._pdf_path is not None)
        self._update_horizon_items(tf)

        try:
            if self._metrics_path:
                with open(self._metrics_path, 'r', encoding='utf-8') as f:
                    self._all_metrics = json.load(f)
        except Exception:
            self._all_metrics = None

        self._render_metrics()
        if self._pdf_path:
            self.graphs_viewer.load(self._pdf_path)
            self.inner_tabs.setCurrentIndex(0)

    def _render_metrics(self):
        if not self._all_metrics:
            self.metrics_scroll.clear()
            self.lbl_period.setText("")
            return

        horizon = self.horizon.currentText() if self.horizon else 'General'
        metricas = dict(self._all_metrics)

        if horizon not in ('General', 'Scalping', 'Daytrading'):
            for key in ('11. Estimadores de Volatilidad OHLC',
                        '12. Test de Estacionariedad (ADF / KPSS)',
                        '13. Vida Media de Reversión (Half-Life OU)'):
                metricas.pop(key, None)

        period = self.metrics_scroll.populate(metricas)
        if period:
            self.lbl_period.setText(period)

    def _export_pdf(self):
        if not self._pdf_path:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar PDF",
            f"informe_{self._ticker}_{self._tf}.pdf",
            "PDF (*.pdf)"
        )
        if path:
            import shutil
            shutil.copy2(self._pdf_path, path)

import pandas as pd
from PyQt6.QtWidgets import (QVBoxLayout, QHBoxLayout, QLabel,
                             QDateEdit, QCheckBox, QPushButton, QFormLayout,
                             QFrame)
from PyQt6.QtCore import Qt, QDate

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.widgets import RangeSlider, Button, TextBox
from matplotlib.dates import num2date, date2num

from gui.dialogs.base import DialogoBase

STYLE_RANGO_DIALOG = """
QDialog { background-color: #141e30; }
QLabel { color: #c8d6e5; font-size: 12px; }
QLabel#warn { color: #e74c3c; font-size: 11px; }
QDateEdit {
    background-color: #1a2a45; color: #c8d6e5; border: none;
    padding: 6px 10px; border-radius: 4px; font-size: 12px;
}
QDateEdit:disabled { background-color: #101a2c; color: #3a5a7a; }
QDateEdit::drop-down { border: none; background: transparent; width: 22px; }
QCalendarWidget { background-color: #1a2a45; color: #c8d6e5; }
QCheckBox { color: #c8d6e5; font-size: 12px; }
QPushButton { background-color: #2a4a6a; color: #4fc3f7; border: none;
              padding: 8px 24px; border-radius: 4px; font-size: 12px; font-weight: bold; }
QPushButton:hover { background-color: #3a5a8a; }
QPushButton#cancel { background-color: #222a3a; color: #5a7a9a; }
QPushButton#cancel:hover { background-color: #2a3a4a; }
QFrame#sep { background-color: #253a60; max-height: 1px; }
"""


class RangoAnalisisDialog(DialogoBase):
    """Dialogo para elegir el rango de fechas a analizar (o todo el historico).

    Modo normal: grafico de precio + RangeSlider de matplotlib embebido
    (mismo look que el selector legacy standalone, pero dentro de la app).
    Modo fallback (si no se puede graficar la serie): QDateEdit manual.
    """

    def __init__(self, csv_path, parent=None):
        self.slider = None
        self._idx_min = None
        self._idx_max = None
        self._tz = None
        self._min_date = None
        self._max_date = None

        df = self._cargar_serie_close(csv_path)
        if df is not None:
            super().__init__("Rango de análisis", parent,
                             subtitulo="Periodo a analizar", ancho=800, alto=620)
            self._construir_modo_grafico(df)
        else:
            self._min_date, self._max_date = self._read_range(csv_path)
            super().__init__("Rango de análisis", parent,
                             subtitulo="Periodo a analizar", ancho=380, alto=320)
            self._construir_modo_fallback()

    # ── Carga de datos ──────────────────────────────────────────────
    def _cargar_serie_close(self, path):
        try:
            df = pd.read_csv(path, usecols=['timestamp', 'close'], parse_dates=['timestamp'])
            df = df.set_index('timestamp')
            df = df[df.index.notna()]
            if len(df) < 2:
                return None
            return df
        except Exception:
            return None

    def _read_range(self, path):
        """Lectura ligera de solo la primera y ultima fila (fallback sin grafico)."""
        try:
            first = pd.read_csv(path, nrows=1, parse_dates=['timestamp'], usecols=['timestamp'])
            t0 = first['timestamp'].iloc[0]
            with open(path, 'rb') as f:
                f.seek(-2000, 2)
                last_line = f.read().decode('utf-8', errors='replace').strip().split('\n')[-1]
            t1 = pd.to_datetime(last_line.split(',')[0])
            return t0.date(), t1.date()
        except Exception:
            return None, None

    # ── Modo gráfico + slider (caso normal) ─────────────────────────
    def _construir_modo_grafico(self, df):
        self.contenido.setContentsMargins(10, 8, 10, 8)
        self.contenido.setSpacing(8)

        max_puntos = 50000
        step = max(1, len(df) // max_puntos)
        df_plot = df.iloc[::step]

        fig = Figure(figsize=(7.6, 4.6))
        fig.patch.set_facecolor('#141e30')
        self.canvas = FigureCanvasQTAgg(fig)

        ax = fig.add_subplot(111)
        ax.set_facecolor('#0d1424')

        ax.fill_between(df_plot.index, df_plot['close'], alpha=0.25, color='#4fc3f7', zorder=1)
        ax.plot(df_plot.index, df_plot['close'], color='#4fc3f7', linewidth=1.2, zorder=2)

        ax.set_title("Arrastra el slider para elegir el rango — Analizar para confirmar",
                     color='#c8d6e5', fontsize=10, fontweight='bold')
        ax.set_xlabel('Fecha', color='#aabbcc')
        ax.set_ylabel('Precio', color='#aabbcc')
        ax.tick_params(colors='#aabbcc', labelsize=8)
        ax.grid(True, alpha=0.15, color='#253a60')
        for spine in ax.spines.values():
            spine.set_edgecolor('#253a60')

        idx_min = df_plot.index.min()
        idx_max = df_plot.index.max()
        x_min_num = date2num(idx_min)
        x_max_num = date2num(idx_max)
        tz = df_plot.index.tz
        ax.set_xlim(idx_min, idx_max)
        fig.subplots_adjust(bottom=0.20, top=0.90)
        pos = ax.get_position()

        highlight = [ax.axvspan(idx_min, idx_max, alpha=0.15, facecolor='#4fc3f7', zorder=0)]

        ax.text(0.01, 0.99, f"Velas totales: {len(df):,}",
                transform=ax.transAxes, ha='left', va='top',
                fontsize=8, color='#aabbcc',
                bbox=dict(facecolor='#141e30', alpha=0.7, edgecolor='#253a60', linewidth=0.5))

        ax_slider = fig.add_axes([pos.x0, 0.055, pos.width, 0.035])
        ax_slider.set_facecolor('#1a2a45')
        slider = RangeSlider(ax_slider, '', x_min_num, x_max_num,
                             valinit=(x_min_num, x_max_num), valfmt='')

        bx0 = pos.x0
        bw = pos.width
        ax_reset  = fig.add_axes([bx0, 0.01, 0.07, 0.032])
        ax_manual = fig.add_axes([bx0 + 0.08, 0.01, 0.09, 0.032])
        ax_cancel = fig.add_axes([bx0 + bw - 0.17, 0.01, 0.08, 0.032])
        ax_accept = fig.add_axes([bx0 + bw - 0.08, 0.01, 0.08, 0.032])

        for bax in [ax_reset, ax_manual, ax_cancel, ax_accept]:
            bax.set_facecolor('#1a2a45')

        btn_reset  = Button(ax_reset, 'Reset', color='#1a2a45', hovercolor='#2a3a55')
        btn_manual = Button(ax_manual, 'Manual', color='#1a2a45', hovercolor='#2a3a55')
        btn_cancel = Button(ax_cancel, 'Cancelar', color='#222a3a', hovercolor='#2a3a4a')
        btn_accept = Button(ax_accept, 'Analizar', color='#2a4a6a', hovercolor='#3a5a8a')

        for btn in [btn_reset, btn_manual, btn_cancel, btn_accept]:
            for t in btn.ax.texts:
                t.set_fontsize(7.5)

        btn_cancel.label.set_color('#e74c3c')
        btn_accept.label.set_color('#4fc3f7')

        fmt_dmy = '%d/%m/%Y'
        mx0 = pos.x0
        ax_start = fig.add_axes([mx0, 0.115, 0.15, 0.038])
        ax_end   = fig.add_axes([mx0 + 0.20, 0.115, 0.15, 0.038])
        ax_apply = fig.add_axes([mx0 + 0.38, 0.115, 0.07, 0.038])
        for bax in [ax_start, ax_end, ax_apply]:
            bax.set_facecolor('#1a2a45')
            bax.set_visible(False)

        start_box = TextBox(ax_start, 'Inicio:', initial=idx_min.strftime(fmt_dmy),
                            color='#1a2a45', hovercolor='#2a3a55')
        end_box   = TextBox(ax_end, 'Fin:', initial=idx_max.strftime(fmt_dmy),
                            color='#1a2a45', hovercolor='#2a3a55')
        btn_apply = Button(ax_apply, 'Ir', color='#2a4a6a', hovercolor='#3a5a8a')

        for box in [start_box, end_box]:
            box.label.set_color('#aabbcc')
            box.label.set_fontsize(7)
            if hasattr(box, 'text_disp'):
                box.text_disp.set_color('#c8d6e5')
            for t in box.ax.texts:
                t.set_fontsize(8)

        btn_apply.label.set_color('#4fc3f7')
        for t in btn_apply.ax.texts:
            t.set_fontsize(8)

        manual_visible = False

        def actualizar_todo(vmin_num, vmax_num):
            nonlocal manual_visible
            dmin = pd.Timestamp(num2date(vmin_num).replace(tzinfo=None)).tz_localize(tz)
            dmax = pd.Timestamp(num2date(vmax_num).replace(tzinfo=None)).tz_localize(tz)
            highlight[0].remove()
            highlight[0] = ax.axvspan(dmin, dmax, alpha=0.15, facecolor='#4fc3f7', zorder=0)
            ax.set_title(f"Rango: {dmin:%d %b %Y}  →  {dmax:%d %b %Y}",
                         color='#4fc3f7', fontsize=10, fontweight='bold')
            if manual_visible:
                start_box.set_val(dmin.strftime(fmt_dmy))
                end_box.set_val(dmax.strftime(fmt_dmy))
            self.canvas.draw_idle()

        def sincronizar_slider(dmin, dmax):
            dmin_num = date2num(dmin)
            dmax_num = date2num(dmax)
            slider.set_val((dmin_num, dmax_num))
            actualizar_todo(dmin_num, dmax_num)

        def on_slider(val):
            vmin, vmax = slider.val
            actualizar_todo(vmin, vmax)

        def on_manual_toggle(event):
            nonlocal manual_visible
            manual_visible = not manual_visible
            for bax in [ax_start, ax_end, ax_apply]:
                bax.set_visible(manual_visible)
            self.canvas.draw_idle()

        def on_apply(event):
            try:
                texto_inicio = start_box.text.strip()
                texto_fin = end_box.text.strip()
                dmin = pd.to_datetime(texto_inicio, format=fmt_dmy)
                dmax = pd.to_datetime(texto_fin, format=fmt_dmy)
                if tz is not None:
                    dmin = dmin.tz_localize(tz) if dmin.tz is None else dmin
                    dmax = dmax.tz_localize(tz) if dmax.tz is None else dmax
                sincronizar_slider(dmin, dmax)
            except Exception:
                pass

        def on_reset(event):
            sincronizar_slider(idx_min, idx_max)

        def on_accept(event):
            self.accept()

        def on_cancel(event):
            self.reject()

        def on_key(event):
            if event.key == 'escape':
                self.reject()

        slider.on_changed(on_slider)
        btn_reset.on_clicked(on_reset)
        btn_manual.on_clicked(on_manual_toggle)
        btn_accept.on_clicked(on_accept)
        btn_cancel.on_clicked(on_cancel)
        btn_apply.on_clicked(on_apply)
        for box in [start_box, end_box]:
            box.on_submit(lambda _: on_apply(None))
        self.canvas.mpl_connect('key_press_event', on_key)

        self.contenido.addWidget(self.canvas, 1)

        # Referencias fuertes: los widgets de matplotlib se desconectan si
        # sus objetos Python se recolectan (garbage collection).
        self._fig = fig
        self._ax = ax
        self.slider = slider
        self._btn_reset = btn_reset
        self._btn_manual = btn_manual
        self._btn_cancel = btn_cancel
        self._btn_accept = btn_accept
        self._btn_apply = btn_apply
        self._start_box = start_box
        self._end_box = end_box

        self._idx_min = idx_min
        self._idx_max = idx_max
        self._tz = tz

    # ── Modo fallback (sin gráfico) ──────────────────────────────────
    def _construir_modo_fallback(self):
        self.contenido.setSpacing(14)

        self.chk_full = QCheckBox("Usar todo el historico")
        self.chk_full.setChecked(True)
        self.chk_full.stateChanged.connect(self._toggle_dates)
        self.contenido.addWidget(self.chk_full)

        form = QFormLayout()
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.date_inicio = QDateEdit()
        self.date_inicio.setCalendarPopup(True)
        self.date_inicio.setDisplayFormat("yyyy-MM-dd")

        self.date_fin = QDateEdit()
        self.date_fin.setCalendarPopup(True)
        self.date_fin.setDisplayFormat("yyyy-MM-dd")

        if self._min_date and self._max_date:
            qmin = QDate(self._min_date.year, self._min_date.month, self._min_date.day)
            qmax = QDate(self._max_date.year, self._max_date.month, self._max_date.day)
            for w in (self.date_inicio, self.date_fin):
                w.setMinimumDate(qmin)
                w.setMaximumDate(qmax)
            self.date_inicio.setDate(qmin)
            self.date_fin.setDate(qmax)
        else:
            self.chk_full.setChecked(True)
            self.chk_full.setEnabled(False)

        form.addRow("Desde:", self.date_inicio)
        form.addRow("Hasta:", self.date_fin)
        self.contenido.addLayout(form)

        self.lbl_warn = QLabel("")
        self.lbl_warn.setObjectName("warn")
        self.lbl_warn.setWordWrap(True)
        if self._min_date and self._max_date:
            self.lbl_warn.setText("No se pudo generar el grafico de precio para este archivo.")
        else:
            self.lbl_warn.setText(
                "No se pudo leer el rango de fechas del archivo. "
                "Solo esta disponible el analisis de todo el historico."
            )
        self.contenido.addWidget(self.lbl_warn)

        self.contenido.addStretch()

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_btn = QPushButton("Cancelar")
        cancel_btn.setObjectName("cancel")
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton("Analizar")
        ok_btn.clicked.connect(self.accept)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(ok_btn)
        self.contenido.addLayout(buttons)

        self._toggle_dates()

    def _toggle_dates(self):
        enabled = not self.chk_full.isChecked() and self._min_date is not None
        self.date_inicio.setEnabled(enabled)
        self.date_fin.setEnabled(enabled)

    # ── Resultado ─────────────────────────────────────────────────
    def get_rango(self):
        """Devuelve (date_inicio, date_fin) o (None, None) si es todo el historico."""
        if self.slider is not None:
            vmin, vmax = self.slider.val
            dmin = pd.Timestamp(num2date(vmin).replace(tzinfo=None)).tz_localize(self._tz)
            dmax = pd.Timestamp(num2date(vmax).replace(tzinfo=None)).tz_localize(self._tz)

            tolerancia = pd.Timedelta(seconds=1)
            if abs(dmin - self._idx_min) > tolerancia or abs(dmax - self._idx_max) > tolerancia:
                return dmin.date(), dmax.date()
            return None, None

        if self.chk_full.isChecked() or self._min_date is None:
            return None, None
        return self.date_inicio.date().toPyDate(), self.date_fin.date().toPyDate()

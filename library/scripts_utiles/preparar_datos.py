import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
import os
import csv
import sys
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button
sys.stdout.reconfigure(encoding='utf-8')

# ── CONFIG ──────────────────────────────────────────
QUESTDB_HOST      = 'localhost'
QUESTDB_HTTP_PORT = 19000

# ── 1. FILE DIALOG ──────────────────────────────────
try:
    root = tk.Tk()
    root.withdraw()
    CSV_INPUT = filedialog.askopenfilename(title="Selecciona el archivo CSV", filetypes=[("CSV", "*.csv")])
    root.destroy()
except Exception:
    CSV_INPUT = None
if not CSV_INPUT:
    CSV_INPUT = r"D:\DATOS\Activos\XAUUSD_H1_202003130100_202506122200.csv"
    print(f"⚠️  Usando ruta por defecto: {CSV_INPUT}")

# ── 2. INTERACTIVE CONFIG WINDOW ────────────────────
SEL_NOMBRE  = ['']
SEL_TIPO    = [0]  # 0=Futuro, 1=Stock, 2=Crypto
SEL_TF      = [4]  # index in TF_LABELS
TF_LABELS   = ['1m','5m','15m','30m','1h','4h','1d','ticks','rango']
TIPO_LABELS = ['Futuro', 'Stock', 'Crypto']
CANCELADO   = [False]

def ventana_config():
    fig = plt.figure(figsize=(9, 7))
    fig.patch.set_facecolor('#E8F4FD')

    # ── helpers para radio buttons custom ──
    class CustomRadios:
        def __init__(self, fig, labels, positions, default, color_sel='#5B9BD5', callback=None):
            self.fig = fig
            self.labels = labels
            self.pos = positions
            self.selected = default
            self.color_sel = color_sel
            self.callback = callback
            self.circles = []
            self.txts = []
            for i, (lbl, (x, y)) in enumerate(zip(labels, positions)):
                c = Circle((x, y), 0.015, facecolor='white', edgecolor='#4A6FA5',
                           linewidth=1.5, transform=fig.transFigure, zorder=5)
                fig.patches.append(c)
                self.circles.append(c)
                t = fig.text(x + 0.028, y, lbl, fontsize=10, color='#4A6FA5',
                             va='center', transform=fig.transFigure, zorder=5)
                self.txts.append(t)
            self._draw()

        def _draw(self):
            for i, c in enumerate(self.circles):
                sel = i == self.selected
                c.set_radius(0.022 if sel else 0.015)
                c.set_facecolor(self.color_sel if sel else 'white')
                c.set_edgecolor(self.color_sel if sel else '#4A6FA5')
                c.set_linewidth(3 if sel else 1.5)
                c.set_alpha(0.9 if sel else 0.6)
                self.txts[i].set_fontsize(11 if sel else 10)
                self.txts[i].set_fontweight('bold' if sel else 'normal')
                self.txts[i].set_color('#1E3A5F' if sel else '#4A6FA5')
            self.fig.canvas.draw_idle()

        def hit(self, x, y):
            for i, c in enumerate(self.circles):
                cx, cy = c.center
                dx, dy = x - cx, y - cy
                if dx*dx + dy*dy < 0.003:
                    return i
            return -1

        def set_active(self, idx):
            if 0 <= idx < len(self.labels):
                self.selected = idx
                self._draw()
                if self.callback:
                    self.callback(self.labels[idx])

    # ── rectángulos de foco visual ──
    focus_rects = {}

    def make_focus_rect(key, x, y, w, h):
        r = plt.Rectangle((x, y), w, h, fill=False, linewidth=2.5,
                          edgecolor='#B0C4DE', visible=True, transform=fig.transFigure, zorder=2)
        fig.patches.append(r)
        focus_rects[key] = r
        return r

    # ── TITLE ──
    fig.text(0.5, 0.96, '☁  CONFIGURACIÓN DEL ACTIVO  ☁', ha='center', va='top',
             fontsize=16, fontweight='bold', color='#1E3A5F')

    # ── ARCHIVO ──
    nombre_csv = os.path.basename(CSV_INPUT)
    fig.text(0.12, 0.89, f'📁  {nombre_csv}', fontsize=10, color='#4A6FA5')

    # ── NOMBRE ACTIVO ──
    fig.text(0.12, 0.82, 'Nombre del activo:', fontsize=11, color='#4A6FA5', fontweight='bold')
    name_text = fig.text(0.12, 0.76, f'✏️  {SEL_NOMBRE[0]}', fontsize=13, color='#1E3A5F',
                         fontweight='bold')
    make_focus_rect('nombre', 0.115, 0.745, 0.515, 0.07)

    EDITANDO = [False]
    BUF_NOMBRE = ['']

    # ── TIPO DE ACTIVO ──
    fig.text(0.12, 0.66, 'Tipo de activo:', fontsize=11, color='#4A6FA5', fontweight='bold')
    pos_tipo = [(0.14, 0.60), (0.42, 0.60), (0.70, 0.60)]
    radio_tipo = CustomRadios(fig, TIPO_LABELS, pos_tipo, SEL_TIPO[0])
    make_focus_rect('tipo', 0.10, 0.565, 0.80, 0.07)

    # ── TIMEFRAME ──
    fig.text(0.12, 0.48, 'Timeframe:', fontsize=11, color='#4A6FA5', fontweight='bold')
    xs_r1 = [0.14, 0.32, 0.50, 0.68]
    pos_tf1 = [(x, 0.42) for x in xs_r1]
    radio_tf1 = CustomRadios(fig, TF_LABELS[:4], pos_tf1, -1 if SEL_TF[0] >= 4 else SEL_TF[0])
    xs_r2 = [0.14, 0.32, 0.50, 0.68, 0.86]
    pos_tf2 = [(x, 0.34) for x in xs_r2]
    radio_tf2 = CustomRadios(fig, TF_LABELS[4:], pos_tf2, max(0, SEL_TF[0] - 4) if SEL_TF[0] >= 4 else -1)
    make_focus_rect('tf', 0.10, 0.305, 0.82, 0.15)

    def _iniciar_edicion():
        EDITANDO[0] = True
        BUF_NOMBRE[0] = SEL_NOMBRE[0]
        name_text.set_text(f'✏️  {BUF_NOMBRE[0]}_')
        fig.canvas.draw_idle()

    def _confirmar_edicion():
        if BUF_NOMBRE[0].strip():
            SEL_NOMBRE[0] = BUF_NOMBRE[0].strip().lower()
        name_text.set_text(f'✏️  {SEL_NOMBRE[0]}')
        EDITANDO[0] = False
        fig.canvas.draw_idle()

    def _cancelar_edicion():
        name_text.set_text(f'✏️  {SEL_NOMBRE[0]}')
        EDITANDO[0] = False
        fig.canvas.draw_idle()

    def sync_tf():
        s1 = radio_tf1.selected
        s2 = radio_tf2.selected
        if s1 >= 0:
            SEL_TF[0] = s1
            radio_tf2.selected = -1
        elif s2 >= 0:
            SEL_TF[0] = 4 + s2
            radio_tf1.selected = -1
        else:
            SEL_TF[0] = 4
            radio_tf2.selected = 0
        radio_tf1._draw()
        radio_tf2._draw()

    def _click(event):
        tf = fig.transFigure.inverted()
        fx, fy = tf.transform((event.x, event.y))
        if 0.115 <= fx <= 0.63 and 0.745 <= fy <= 0.815:
            _iniciar_edicion()
            return
        h1 = radio_tf1.hit(fx, fy)
        if h1 >= 0:
            radio_tf1.selected = h1
            radio_tf2.selected = -1
            sync_tf()
            return
        h2 = radio_tf2.hit(fx, fy)
        if h2 >= 0:
            radio_tf2.selected = h2
            radio_tf1.selected = -1
            sync_tf()
            return
        ht = radio_tipo.hit(fx, fy)
        if ht >= 0:
            radio_tipo.set_active(ht)
            SEL_TIPO[0] = ht

    # ── BUTTONS ──
    ax_ok = plt.axes([0.25, 0.16, 0.15, 0.06])
    btn_ok = Button(ax_ok, '✅ Aceptar', color='#2E86C1', hovercolor='#1A6DA0')
    btn_ok.label.set_color('white')

    ax_cancel = plt.axes([0.55, 0.16, 0.15, 0.06])
    btn_cancel = Button(ax_cancel, '❌ Cancelar', color='#BDC3C7', hovercolor='#A0A0A0')

    OK_PRESSED = [False]

    def on_ok(event):
        OK_PRESSED[0] = True
        if EDITANDO[0]:
            _confirmar_edicion()
        if not SEL_NOMBRE[0].strip():
            print("⚠️  El nombre del activo no puede estar vacío.")
            OK_PRESSED[0] = False
            return
        SEL_NOMBRE[0] = SEL_NOMBRE[0].strip().lower()
        sync_tf()
        plt.close(fig)

    def on_cancel(event):
        CANCELADO[0] = True
        plt.close(fig)

    btn_ok.on_clicked(on_ok)
    btn_cancel.on_clicked(on_cancel)

    # ── KEYBOARD NAV ──
    FOCUS_GROUPS = ['nombre', 'tipo', 'tf', 'btn_ok', 'btn_cancel']
    focus_idx = [0]
    focus_order = [('nombre', 0), ('tipo', 1), ('tf', 2), ('btn_ok', 3), ('btn_cancel', 4)]

    def highlight_focus(idx):
        for key, r in focus_rects.items():
            if key in ('nombre', 'tipo', 'tf'):
                r.set_edgecolor('#4A90D9' if key == FOCUS_GROUPS[idx] else '#B0C4DE')
                r.set_linewidth(2.5 if key == FOCUS_GROUPS[idx] else 1)
        ax_ok.spines['bottom'].set_color('#4A90D9' if FOCUS_GROUPS[idx] == 'btn_ok' else '#B0C4DE')
        ax_cancel.spines['bottom'].set_color('#4A90D9' if FOCUS_GROUPS[idx] == 'btn_cancel' else '#B0C4DE')
        fig.canvas.draw_idle()

    def on_key(event):
        if EDITANDO[0]:
            if event.key == 'enter':
                _confirmar_edicion()
            elif event.key == 'escape':
                _cancelar_edicion()
            elif event.key == 'backspace':
                BUF_NOMBRE[0] = BUF_NOMBRE[0][:-1]
                name_text.set_text(f'✏️  {BUF_NOMBRE[0]}_')
                fig.canvas.draw_idle()
            elif event.key == 'space':
                BUF_NOMBRE[0] += '_'
                name_text.set_text(f'✏️  {BUF_NOMBRE[0]}_')
                fig.canvas.draw_idle()
            elif event.key == 'ctrl+u':
                BUF_NOMBRE[0] = ''
                name_text.set_text(f'✏️  _')
                fig.canvas.draw_idle()
            elif len(event.key) == 1:
                BUF_NOMBRE[0] += event.key.lower()
                name_text.set_text(f'✏️  {BUF_NOMBRE[0]}_')
                fig.canvas.draw_idle()
            return
        if event.key in ('down', 'tab'):
            focus_idx[0] = (focus_idx[0] + 1) % len(FOCUS_GROUPS)
            highlight_focus(focus_idx[0])
            event.guiEvent = None
        elif event.key == 'up':
            focus_idx[0] = (focus_idx[0] - 1) % len(FOCUS_GROUPS)
            highlight_focus(focus_idx[0])
        elif event.key == 'right':
            g = FOCUS_GROUPS[focus_idx[0]]
            if g == 'tipo':
                radio_tipo.set_active(min(len(TIPO_LABELS)-1, radio_tipo.selected + 1))
                SEL_TIPO[0] = radio_tipo.selected
            elif g == 'tf':
                n = SEL_TF[0]
                n = min(len(TF_LABELS)-1, n+1)
                if n < 4:
                    radio_tf1.selected = n
                    radio_tf2.selected = -1
                else:
                    radio_tf2.selected = n - 4
                    radio_tf1.selected = -1
                SEL_TF[0] = n
                radio_tf1._draw()
                radio_tf2._draw()
        elif event.key == 'left':
            g = FOCUS_GROUPS[focus_idx[0]]
            if g == 'tipo':
                radio_tipo.set_active(max(0, radio_tipo.selected - 1))
                SEL_TIPO[0] = radio_tipo.selected
            elif g == 'tf':
                n = SEL_TF[0]
                n = max(0, n-1)
                if n < 4:
                    radio_tf1.selected = n
                    radio_tf2.selected = -1
                else:
                    radio_tf2.selected = n - 4
                    radio_tf1.selected = -1
                SEL_TF[0] = n
                radio_tf1._draw()
                radio_tf2._draw()
        elif event.key == 'enter':
            g = FOCUS_GROUPS[focus_idx[0]]
            if g == 'nombre':
                _iniciar_edicion()
            elif g == 'btn_ok':
                on_ok(None)
            elif g == 'btn_cancel':
                on_cancel(None)
        elif event.key == 'escape':
            on_cancel(None)

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('button_press_event', _click)
    fig.canvas.mpl_connect('close_event', lambda e: CANCELADO.__setitem__(0, not OK_PRESSED[0]))

    # ── HINT ──
    fig.text(0.5, 0.06, '↑↓ entre grupos · ←→ cambia opción · Enter/Click confirma · Esc cancela',
             ha='center', fontsize=9, color='#7FB3D8', style='italic')

    highlight_focus(0)
    plt.show(block=True)

ventana_config()

if CANCELADO[0]:
    print("⚠️  Operación cancelada por el usuario.")
    sys.exit(0)

NOMBRE_ACTIVO = SEL_NOMBRE[0]
TIPO_ACTIVO   = TIPO_LABELS[SEL_TIPO[0]].upper()
TIMEFRAME     = TF_LABELS[SEL_TF[0]]

# ── 3. SESSION CONFIG ───────────────────────────────
CONFIG_PATH = r"D:\DATOS\Activos\sesion_config.json"
os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
with open(CONFIG_PATH, 'w') as f:
    json.dump({'nombre': NOMBRE_ACTIVO, 'tf': TIMEFRAME, 'activo': TIPO_ACTIVO}, f)
print(f"✅ Config guardada: {CONFIG_PATH}")

# ── 4. PROCESSING ───────────────────────────────────
TABLA_DESTINO = f"{NOMBRE_ACTIVO}_candles_{TIMEFRAME}"
_p = Path(CSV_INPUT)
CSV_OUTPUT = str(_p.parent / f"{_p.stem}_preparado{_p.suffix}")

print("="*50)
print("PREPARACIÓN DE DATOS")
print("="*50)

# Detectar separador automáticamente
with open(CSV_INPUT, 'r', encoding='utf-8') as f:
    primera_linea = f.readline()
    SEPARADOR = max(['\t', ',', ';', '|', ' '], key=lambda s: len(primera_linea.split(s)))
print(f"Separador detectado: {repr(SEPARADOR)}")

# [1/4] Lectura
df = pd.read_csv(CSV_INPUT, sep=SEPARADOR, low_memory=False)
print(f"Filas iniciales: {len(df)}")

# [2/4] Normalización
df.columns = df.columns.str.strip().str.replace('\ufeff', '').str.replace('<', '').str.replace('>', '')
print(f"      Columnas detectadas: {list(df.columns)}")

# Combinar DATE + TIME (MT4)
cols_upper = {c.upper(): c for c in df.columns}
if 'DATE' in cols_upper and 'TIME' in cols_upper:
    col_date = cols_upper['DATE']
    col_time = cols_upper['TIME']
    df['timestamp'] = pd.to_datetime(df[col_date].astype(str) + ' ' + df[col_time].astype(str), errors='coerce')
    df = df.drop(columns=[col_date, col_time])
    print(f"      Formato MT4 detectado — columnas DATE+TIME combinadas en 'timestamp'")

# Renombrar alias canónicos
ALIAS_COLUMNAS = {
    'timestamp': ['time', 'date', 'datetime', 'ts', 'fecha', 'open_time', 'Date', 'Time', 'Datetime', 'Timestamp', 'index', 'Open_time', 'Fecha'],
    'open':      ['Open', 'OPEN', 'o', 'precio_apertura', 'open_price', 'Apertura', 'apertura'],
    'high':      ['High', 'HIGH', 'h', 'max', 'maximo', 'máximo', 'precio_maximo'],
    'low':       ['Low',  'LOW',  'l', 'min', 'minimo', 'mínimo', 'precio_minimo'],
    'close':     ['Close','CLOSE','c', 'price', 'precio', 'last', 'ultimo', 'último', 'settle', 'settlement'],
    'volume':    ['Volume','VOLUME','vol','Vol','Vol.','VOL','qty','quantity', 'amount', 'volumen', 'tickvol', 'TICKVOL'],
    'spread':    ['Spread', 'SPREAD', 'spread'],
}
mapa_inverso = {alias.lower(): canonico for canonico, aliases in ALIAS_COLUMNAS.items() for alias in aliases}
renombrado = {col: mapa_inverso[col.lower()] for col in df.columns if col.lower() in mapa_inverso}
df = df.rename(columns=renombrado)
df = df.loc[:, ~df.columns.duplicated(keep='first')]

# [3/4] Conversión
if 'timestamp' not in df.columns:
    posibles_ts = [c for c in df.columns if any(k in c.lower() for k in ['date', 'time', 'fecha', 'datetime'])]
    if posibles_ts:
        df['timestamp'] = pd.to_datetime(df[posibles_ts[0]], errors='coerce')
        df = df.drop(columns=[c for c in posibles_ts if c != 'timestamp' and c in df.columns])
        print(f"      Columna '{posibles_ts[0]}' usada como timestamp")
    else:
        primera_col = df.columns[0]
        df['timestamp'] = pd.to_datetime(df[primera_col], errors='coerce')
        print(f"      Intentando usar columna '{primera_col}' como timestamp")
        if df['timestamp'].isna().all():
            print(f"      ⚠️ No se pudo parsear '{primera_col}' como fecha. Usando índice numérico.")
            df['timestamp'] = pd.Timestamp.now().normalize() + pd.to_timedelta(range(len(df)), unit='h')

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', dayfirst=True)
df = df.dropna(subset=['timestamp'])

# Estandarizar a UTC
if df['timestamp'].dt.tz is None:
    horas = df['timestamp'].dt.hour
    if horas.nunique() == 1 and horas.iloc[0] == 0:
        pass
    else:
        print(f"      ⚠️ Timestamps sin timezone. Asumiendo UTC.")
        print(f"         Si los datos están en hora local, ajusta manualmente.")
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
else:
    df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

if 'volume' not in df.columns: df['volume'] = 0
if 'spread' not in df.columns: df['spread'] = 0
cols_finales = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread']
df = df[[c for c in cols_finales if c in df.columns]]

# [4/4] Guardar CSV preparado
df.to_csv(CSV_OUTPUT, index=False)
print(f"✅ Archivo listo: {CSV_OUTPUT}")

# [5/5] Subir a QuestDB via HTTP /imp
print(f"\n[5/5] Subiendo {len(df):,} filas a QuestDB — Tabla: {TABLA_DESTINO} ...")
try:
    SUFIJOS = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000}
    def _parse_num(v):
        if pd.isna(v):
            return 0.0
        s = str(v).strip().replace(' ', '')
        if not s or s.lower() in ('nan', 'null', 'none', ''):
            return 0.0
        # Detectar formato europeo (punto= miles, coma=decimal)
        if ',' in s and '.' in s:
            if s.rfind(',') > s.rfind('.'):
                s = s.replace('.', '')
                s = s.replace(',', '.')
            else:
                s = s.replace(',', '')
        elif ',' in s:
            parts = s.split(',')
            if len(parts) == 2:
                last_clean = parts[1].rstrip('KkMmBb')
                if last_clean.isdigit() and len(last_clean) <= 2:
                    s = s.replace(',', '.')
        suf = s[-1].upper()
        if suf in SUFIJOS:
            try:
                return float(s[:-1]) * SUFIJOS[suf]
            except ValueError:
                return 0.0
        try:
            return float(s)
        except ValueError:
            return 0.0

    for c in df.columns:
        if c != 'timestamp':
            df[c] = df[c].apply(_parse_num)
            if c == 'volume':
                df[c] = df[c].astype(int)
    csv_buffer = df.to_csv(index=False, encoding='utf-8')
    cols_schema = ','.join([f'{c}:INT' if c == 'volume' else f'{c}:DOUBLE' for c in df.columns if c != 'timestamp'])
    url = f"http://{QUESTDB_HOST}:{QUESTDB_HTTP_PORT}/imp?name={TABLA_DESTINO}&overwrite=true&types=timestamp:TIMESTAMP,{cols_schema}"
    resp = requests.post(url, files={'data': ('data.csv', csv_buffer.encode('utf-8'), 'text/csv')}, timeout=120)
    if resp.status_code == 200:
        print(f"      ↳ {resp.text.strip()}")
        print(f"✅ Subida completada — {len(df):,} filas en {TABLA_DESTINO}")
    else:
        print(f"⚠️  HTTP {resp.status_code}: {resp.text.strip()}")
except Exception as e:
    print(f"⚠️  Error al subir a QuestDB: {e}")
    print("   El archivo preparado se ha guardado igualmente.")

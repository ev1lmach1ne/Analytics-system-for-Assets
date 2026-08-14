import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
import os
import csv
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from core.config import CONFIG_PATH, QUESTDB_HOST, QUESTDB_HTTP_PORT
from core.parsing import parse_columna_flexible
from core.questdb_errors import filas_en_tabla
import re

# tkinter/customtkinter se importan DENTRO de las ramas interactivas, no aquí:
# solo hacen falta cuando el script se usa suelto desde la línea de comandos.
# Lanzado desde la GUI siempre llegan CSV_INPUT y CONFIG_SKIP, así que esas
# ramas no se ejecutan — y al ir empaquetado con PyInstaller esos módulos no
# viajan dentro del .exe, con lo que importarlos arriba rompía la importación
# entera ("No module named 'tkinter'").
# errors='replace' explícito: reconfigure(encoding=...) a secas deja errors en
# 'strict', y eso deshace la tolerancia que app.py deja puesta al lanzar el
# script desde el .exe empaquetado.
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CSV_INPUT = os.environ.get('CSV_INPUT', '')


# ── 1. FILE DIALOG (skip if CSV_INPUT env is set) ──
if not CSV_INPUT:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        CSV_INPUT = filedialog.askopenfilename(title="Selecciona el archivo CSV", filetypes=[("CSV", "*.csv")])
        root.destroy()
    except Exception:
        CSV_INPUT = None
if not CSV_INPUT:
    from core.config import BASE_DATA
    CSV_INPUT = os.path.join(BASE_DATA, "XAUUSD_H1_202003130100_202506122200.csv")
    print(f"⚠️  Usando ruta por defecto: {CSV_INPUT}")

# ── 2. INTERACTIVE CONFIG WINDOW (skip if env vars are set) ──
CONFIG_SKIP = os.environ.get('CONFIG_SKIP', '')
if CONFIG_SKIP:
    NOMBRE_ACTIVO = os.environ.get('CONFIG_NOMBRE', 'xauusd')
    TIMEFRAME     = os.environ.get('CONFIG_TF', '1h')
    TIPO_ACTIVO   = os.environ.get('CONFIG_ACTIVO', 'FUTURO')
    CATEGORIA     = os.environ.get('CONFIG_CATEGORIA', '')
else:
    CATEGORIA = ''
    import customtkinter as ctk
    SEL_NOMBRE  = ['']
    SEL_TIPO    = [0]  # 0=Futuro, 1=Stock, 2=Crypto
    SEL_TF      = [6]  # index in TF_LABELS (default = 1h)
    TF_LABELS   = ['30s','1m','3m','5m','15m','30m','1h','2h','4h','1d','ticks','rango','Custom']
    TIPO_LABELS = ['Futuro/Cfd', 'Forex', 'Stock', 'Crypto']
    CANCELADO   = [False]
    TF_CUSTOM_STR = ['']  # custom timeframe string (ej. '3m', '10m', '30s')

    def ventana_config():
        ctk.set_appearance_mode("Light")
        ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("☁  CONFIGURADOR DE ARCHIVO  ☁")
    root.geometry("640x580")
    root.resizable(False, False)

    OK_PRESSED = [False]
    IS_CUSTOM = [SEL_TF[0] == 12 and bool(TF_CUSTOM_STR[0])]

    # ── TOP BAR: title + theme toggle ──
    top_bar = ctk.CTkFrame(root, fg_color="transparent")
    top_bar.pack(pady=(18, 0), padx=30, fill="x")

    ctk.CTkLabel(top_bar, text="☁  CONFIGURADOR DE ARCHIVO  ☁",
                 font=ctk.CTkFont(size=17, weight="bold"),
                 anchor="center").pack(side="left", expand=True, fill="x")

    def toggle_theme():
        n = "Dark" if ctk.get_appearance_mode() == "Light" else "Light"
        ctk.set_appearance_mode(n)
        theme_btn.configure(text="☀️" if n == "Light" else "🌙")

    theme_btn = ctk.CTkButton(top_bar, text="🌙", width=36, height=30, command=toggle_theme,
                              fg_color=("gray75", "gray40"), hover_color=("gray65", "gray50"),
                              corner_radius=8)
    theme_btn.pack(side="right", padx=(0, 5))

    # ── FILE INFO ──
    ctk.CTkLabel(root, text=f"📁  {os.path.basename(CSV_INPUT)}",
                 font=ctk.CTkFont(size=11)).pack(pady=(8, 12))

    # ── MAIN FRAME ──
    main = ctk.CTkFrame(root)
    main.pack(fill="both", expand=True, padx=35, pady=(0, 15))

    # ── NOMBRE ──
    ctk.CTkLabel(main, text="Nombre del activo:",
                 font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", pady=(18, 5))
    nombre_entry = ctk.CTkEntry(main, placeholder_text="xauusd", width=320)
    nombre_entry.pack(anchor="w", padx=10)

    # ── TIPO ──
    ctk.CTkLabel(main, text="Tipo de activo:",
                 font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", pady=(15, 5))
    tipo_frame = ctk.CTkFrame(main, fg_color="transparent")
    tipo_frame.pack(padx=10, anchor="w")
    tipo_var = ctk.StringVar(value=TIPO_LABELS[SEL_TIPO[0]])
    radios_tipo = []
    for lbl in TIPO_LABELS:
        rb = ctk.CTkRadioButton(tipo_frame, text=lbl, variable=tipo_var,
                                value=lbl, font=ctk.CTkFont(size=12))
        rb.pack(side="left", padx=(0, 20))
        radios_tipo.append(rb)

    def cycle_tipo(delta):
        i = TIPO_LABELS.index(tipo_var.get())
        tipo_var.set(TIPO_LABELS[(i + delta) % len(TIPO_LABELS)])

    for rb in radios_tipo:
        rb.bind("<Left>", lambda e: cycle_tipo(-1))
        rb.bind("<Right>", lambda e: cycle_tipo(1))

    # ── TIMEFRAME ──
    ctk.CTkLabel(main, text="Timeframe:",
                 font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", pady=(15, 5))
    tf_main = ctk.CTkFrame(main, fg_color="transparent")
    tf_main.pack(padx=5, fill="x")

    tf_sel = ctk.StringVar()

    def make_callback(btn, others):
        def cb(v):
            if v:
                for o in others:
                    o.set(None)
                tf_sel.set(v)
                _toggle_custom_frame(v)
            else:
                tf_sel.set("")
        return cb

    def _toggle_custom_frame(v):
        if v == "Custom":
            custom_frame.pack(padx=10, pady=(8, 0), anchor="w")
            custom_entry.focus_set()
        else:
            custom_frame.pack_forget()

    seg1 = ctk.CTkSegmentedButton(tf_main, values=TF_LABELS[:6],
                                  font=ctk.CTkFont(size=11), dynamic_resizing=False)
    seg1.pack(pady=2, fill="x")

    seg2 = ctk.CTkSegmentedButton(tf_main, values=TF_LABELS[6:10],
                                  font=ctk.CTkFont(size=11), dynamic_resizing=False)
    seg2.pack(pady=2, fill="x")

    seg3 = ctk.CTkSegmentedButton(tf_main, values=TF_LABELS[10:],
                                  font=ctk.CTkFont(size=11), dynamic_resizing=False)
    seg3.pack(pady=2, fill="x")

    seg1.configure(command=make_callback(seg1, [seg2, seg3]))
    seg2.configure(command=make_callback(seg2, [seg1, seg3]))
    seg3.configure(command=make_callback(seg3, [seg1, seg2]))

    # ── Restore initial selection ──
    if IS_CUSTOM[0]:
        seg3.set("Custom")
        tf_sel.set("Custom")
    else:
        idx = SEL_TF[0]
        if idx < 6:
            seg1.set(TF_LABELS[idx])
            tf_sel.set(TF_LABELS[idx])
        elif idx < 10:
            seg2.set(TF_LABELS[idx])
            tf_sel.set(TF_LABELS[idx])
        else:
            seg3.set(TF_LABELS[idx])
            tf_sel.set(TF_LABELS[idx])

    # ── CUSTOM TF ──
    custom_frame = ctk.CTkFrame(main, fg_color="transparent")
    ctk.CTkLabel(custom_frame, text="Custom:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 8))
    custom_entry = ctk.CTkEntry(custom_frame, placeholder_text="ej. 3m, 10m, 2h, 30s", width=200)
    custom_entry.pack(side="left")
    if IS_CUSTOM[0] and TF_CUSTOM_STR[0]:
        custom_entry.insert(0, TF_CUSTOM_STR[0])
        custom_frame.pack(padx=10, pady=(8, 0), anchor="w")

    # ── BUTTONS ──
    btn_frame = ctk.CTkFrame(main, fg_color="transparent")
    btn_frame.pack(pady=(20, 8))

    def on_ok():
        nombre = nombre_entry.get().strip().lower()
        if not nombre:
            print("⚠️  El nombre del activo no puede estar vacío.")
            nombre_entry.focus_set()
            return
        SEL_NOMBRE[0] = nombre
        SEL_TIPO[0] = TIPO_LABELS.index(tipo_var.get())

        selected = tf_sel.get()
        if selected == "Custom":
            custom_val = custom_entry.get().strip().lower()
            if not custom_val:
                print("⚠️  El timeframe personalizado está vacío.")
                custom_entry.focus_set()
                return
            if not re.match(r'^\d+(?:s|sec|min|m|h|d|w)$', custom_val):
                print("⚠️  Formato inválido: ej. 3m, 10m, 2h, 30s")
                custom_entry.focus_set()
                return
            TF_CUSTOM_STR[0] = custom_val
            SEL_TF[0] = 12
        else:
            SEL_TF[0] = TF_LABELS.index(selected)

        OK_PRESSED[0] = True
        root.destroy()

    def on_cancel():
        CANCELADO[0] = True
        root.destroy()

    ctk.CTkButton(btn_frame, text="✅  Aceptar", command=on_ok, width=120,
                  font=ctk.CTkFont(size=13)).pack(side="left", padx=8)
    ctk.CTkButton(btn_frame, text="✕  Cancelar", command=on_cancel, width=120,
                  fg_color="#5D6D7E", hover_color="#4A5A6B",
                  font=ctk.CTkFont(size=13)).pack(side="left", padx=8)

    # ── HINT ──
    ctk.CTkLabel(main, text="Tab entre grupos · ←→ cambia opción · Enter confirma · Esc cancela",
                 font=ctk.CTkFont(size=10), text_color="gray").pack(pady=(5, 10))

    # ── KEYBOARD BINDINGS ──
    root.bind("<Return>", lambda e: on_ok())
    root.bind("<Escape>", lambda e: on_cancel())

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.grab_set()
    root.mainloop()

    ventana_config()

    if CANCELADO[0]:
        print("⚠️  Operación cancelada por el usuario.")
        sys.exit(0)

    NOMBRE_ACTIVO = SEL_NOMBRE[0]
    _mapa_tipo = {'Futuro/Cfd': 'FUTURO', 'Forex': 'FOREX', 'Stock': 'STOCK', 'Crypto': 'CRYPTO'}
    TIPO_ACTIVO   = _mapa_tipo[TIPO_LABELS[SEL_TIPO[0]]]
    if SEL_TF[0] == 12 and TF_CUSTOM_STR[0].strip():
        TIMEFRAME = TF_CUSTOM_STR[0].strip().lower()
    else:
        TIMEFRAME = TF_LABELS[SEL_TF[0]]

# ── 3. SESSION CONFIG ───────────────────────────────
# 'categoria' la aporta la pestaña Importar (CONFIG_CATEGORIA): hay que
# conservarla al reescribir la config, porque limpieza_datos_er.py la usa
# para guardar el CSV limpio en la carpeta de su categoría. Antes se
# machacaba la config solo con nombre/tf/activo y el limpiador, sin
# categoría, mandaba todo a OTROS (mientras el meta.json sí iba a la
# carpeta correcta). Si no llega por entorno, se respeta la que ya hubiera
# en la config previa (lanzamientos sueltos desde CLI).
if not CATEGORIA and os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, encoding='utf-8') as f:
            CATEGORIA = (json.load(f) or {}).get('categoria', '')
    except Exception:
        CATEGORIA = ''
_sesion_cfg = {'nombre': NOMBRE_ACTIVO, 'tf': TIMEFRAME, 'activo': TIPO_ACTIVO}
if CATEGORIA:
    _sesion_cfg['categoria'] = CATEGORIA
os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
with open(CONFIG_PATH, 'w') as f:
    json.dump(_sesion_cfg, f)
print(f"✅ Config guardada: {CONFIG_PATH}")

# ── 4. PROCESSING ───────────────────────────────────
TABLA_DESTINO = f"{NOMBRE_ACTIVO}_candles_{TIMEFRAME}"
_p = Path(CSV_INPUT)
CSV_OUTPUT = str(_p.parent / f"{_p.stem}_preparado{_p.suffix}")

print(f"\n{'='*60}")
print(f"PREPARACIÓN DE DATOS — {NOMBRE_ACTIVO.upper()} ({TIMEFRAME})")
print(f"{'='*60}\n")

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

# Sanitizar columna timestamp antes de parsear
ts_raw = df['timestamp'].astype(str).str.strip().str.replace('\ufeff', '', regex=False)
ts_raw = ts_raw.str.replace(r'[^\x20-\x7E]', '', regex=True)
antes = len(df)
df['timestamp'] = pd.to_datetime(ts_raw, errors='coerce', format='ISO8601')
nat_count = df['timestamp'].isna().sum()
print(f"      ⚠️ Timestamps NaT tras parseo: {nat_count:,} de {antes:,} filas ({nat_count/antes:.1%})")
df = df.dropna(subset=['timestamp'])
print(f"      Filas tras dropna: {len(df):,} ({len(df)/antes:.1%})")

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
df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.000000Z')

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
    fallos_parseo = {}
    for c in df.columns:
        if c != 'timestamp':
            vals, motivos = parse_columna_flexible(df[c])
            df[c] = vals
            n_fallo = int((motivos == 'fallo').sum())
            if n_fallo > 0:
                fallos_parseo[c] = n_fallo
            if c == 'volume':
                df[c] = df[c].astype(int)

    if fallos_parseo:
        total_fallos = sum(fallos_parseo.values())
        print(f"⚠️  Celdas no parseables convertidas a 0.0 ({total_fallos:,} en total):")
        for col, n in fallos_parseo.items():
            print(f"      - {col}: {n:,} celdas")
    else:
        print("      Todas las celdas numéricas se parsearon correctamente.")

    # ── Comprobación de overwrite: ¿la tabla destino ya tiene datos? ──
    # Se consulta por HTTP /exec, la vía NATIVA de QuestDB (la misma que usa
    # la subida). El wire protocol PostgreSQL de QuestDB no devuelve SQLSTATE
    # de forma fiable para "tabla inexistente", y antes cualquier error se
    # tragaba como "tabla vacía", ocultando QuestDB caída o credenciales
    # inválidas.
    filas_existentes = filas_en_tabla(
        QUESTDB_HOST, QUESTDB_HTTP_PORT, TABLA_DESTINO)

    proceder_subida = True
    if filas_existentes > 0:
        if sys.stdin.isatty() and not CONFIG_SKIP:
            print(f"⚠️  La tabla '{TABLA_DESTINO}' ya existe con {filas_existentes:,} filas.")
            respuesta = input("   ¿Sobrescribir todos los datos existentes? [s/N]: ").strip().lower()
            proceder_subida = respuesta in ('s', 'si', 'sí', 'y', 'yes')
            if not proceder_subida:
                raise RuntimeError(
                    "Subida cancelada por el usuario; no se inicia la limpieza.")
        else:
            print(f"⚠️  La tabla '{TABLA_DESTINO}' ya existe con {filas_existentes:,} filas — "
                  f"se sobrescribirá automáticamente (modo no interactivo).")

    if proceder_subida:
        csv_buffer = df.to_csv(index=False, encoding='utf-8')
        cols_schema = ','.join([f'{c}:INT' if c == 'volume' else f'{c}:DOUBLE' for c in df.columns if c != 'timestamp'])
        url = f"http://{QUESTDB_HOST}:{QUESTDB_HTTP_PORT}/imp?name={TABLA_DESTINO}&overwrite=true&types=timestamp:TIMESTAMP,{cols_schema}"
        resp = requests.post(
            url,
            files={'data': ('data.csv', csv_buffer.encode('utf-8'), 'text/csv')},
            timeout=120)
        if not 200 <= resp.status_code < 300:
            raise RuntimeError(
                f"QuestDB respondió HTTP {resp.status_code}: {resp.text.strip()}")
        print(f"      ↳ {resp.text.strip()}")
        print(f"✅ Subida completada — {len(df):,} filas en {TABLA_DESTINO}")
except Exception as e:
    print(f"⚠️  Error al subir a QuestDB: {e}")
    print("   El archivo preparado se ha guardado, pero la importación ha fallado.")
    sys.exit(4)

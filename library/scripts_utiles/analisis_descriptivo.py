import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from core.config import CONFIG_PATH, INFORMES_DIR, LIMPIADOS_DIR, BASE_DATA, tf_to_minutes
from core.metrics import (calcular_er_series, calcular_kama_numba,
                          calcular_umbrales_er, contar_regimen_hurst,
                          calcular_hurst_array, curvas_cambio_acumulado)
from core.candle_patterns import SESIONES
from scipy import stats
from scipy.signal import fftconvolve
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
import json
import subprocess
import re
import math
import io
import pickle
import time
import traceback
import warnings
import seaborn as sns
warnings.filterwarnings('ignore', category=InterpolationWarning)
from matplotlib.widgets import SpanSelector
from matplotlib.dates import num2date, date2num, AutoDateLocator, ConciseDateFormatter
from matplotlib.widgets import RangeSlider, TextBox, Button

# region ── 1. CONFIGURACIÓN — solo cambiar estos valores ────────────────────────────
# ── CONFIG por defecto (se sobreescribe con sesion_config.json si existe) ──
CONFIG = {
    'activo':     'FUTURO',
    'nombre':     'xauusd',
    'tf':         '1h',
    'input_path': os.path.join(LIMPIADOS_DIR, "xauusd", "xauusd_1h_limpiado.csv"),
    'interactive': False,
    'horizonte':  'General',  # Ventana seleccionada en la GUI (filtra el output de terminal)
}

# ── Leer config desde archivo de sesión (lo escribe la GUI) ──
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        sesion = json.load(f)

    # Preferir input_path explícito desde sesion_config.json (lo escribe la GUI)
    if 'input_path' in sesion and sesion['input_path']:
        input_clean = sesion['input_path']
    else:
        input_clean = os.path.join(
            LIMPIADOS_DIR, sesion['nombre'],
            f"{sesion['nombre']}_{sesion['tf']}_limpiado.csv"
        )

    CONFIG = {
        'activo':     sesion.get('activo', 'FUTURO'),
        'nombre':     sesion.get('nombre', 'xauusd'),
        'tf':         sesion.get('tf', '1h'),
        'input_path': input_clean,
        'rf_rate':    sesion.get('rf_rate', 0.0),
        'interactive': True,
        'horizonte':  sesion.get('horizonte', 'General'),
    }

    # Sobreescribir con .meta.json si existe junto al archivo de entrada
    meta_path = input_clean + '.meta.json'
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if 'nombre' in meta:
                CONFIG['nombre'] = meta['nombre']
            if 'tf' in meta:
                CONFIG['tf'] = meta['tf']
            if 'activo' in meta:
                CONFIG['activo'] = meta['activo']
            if 'rf_rate' in meta:
                CONFIG['rf_rate'] = meta['rf_rate']
            print(f"↳ Config desde .meta.json: {meta.get('nombre')} {meta.get('tf')}")
        except Exception:
            pass

    print(f"↳ Config desde sesión: {CONFIG['nombre']} {CONFIG['tf']} → {input_clean}")
else:
    print("↳ Usando config manual (sin sesion_config.json)")

# endregion


# region ── 2. FUNCIONES AUXILIARES ──────────────────────────────────────────────────
def get_factores(tipo_activo, tf):
    minutos_dia = {'CRYPTO': 1440, 'FUTURO': 1440, 'STOCK': 390}.get(tipo_activo, 1440)
    dias_ano = {'CRYPTO': 365, 'FUTURO': 252, 'STOCK': 252}.get(tipo_activo, 252)
    # CRYPTO opera 7/7 (sin fines de semana de mercado cerrado): usar 5 dias/semana
    # y 21 dias/mes (convencion de dias de TRADING de STOCK/FUTURO) infraestima la
    # ventana semanal/mensual real en un ~30-40%. Se deriva de dias_ano para que
    # 'semanal'/'mensual'/'trimestral' sean consistentes con 'anual' en todos los
    # casos (para STOCK/FUTURO da exactamente 5/21/63 como antes, sin regresion).
    dias_semana = 7 if tipo_activo == 'CRYPTO' else 5
    tf = tf.lower()
    if tf.endswith('d'):      vd = 1
    elif tf.endswith('w'):    vd = 1/dias_semana
    elif tf.endswith('h'):    vd = minutos_dia / (float(tf.replace('h','')) * 60)
    elif tf.endswith('min'):  vd = minutos_dia / float(tf.replace('min',''))
    elif tf.endswith('m'):    vd = minutos_dia / float(tf.replace('m',''))
    elif tf.endswith('s'):    vd = minutos_dia * 60 / float(tf.replace('s',''))
    elif tf.endswith('sec'):  vd = minutos_dia * 60 / float(tf.replace('sec',''))
    else:                     raise ValueError(f"TF no reconocido: {tf}")
    return {'dia': vd, 'semanal': vd*dias_semana, 'mensual': vd*(dias_ano/12),
            'trimestral': vd*(dias_ano/4), 'anual': vd*dias_ano}


def seleccionar_rango_interactivo(df):
    if not CONFIG.get('interactive', False) or not isinstance(df.index, pd.DatetimeIndex):
        return None, None

    max_puntos = 50000
    step = max(1, len(df) // max_puntos)
    df_plot = df.iloc[::step]

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    ax.fill_between(df_plot.index, df_plot['close'], alpha=0.25, color='#00d4aa', zorder=1)
    ax.plot(df_plot.index, df_plot['close'], color='#00d4aa', linewidth=1.2, zorder=2)

    ax.set_title("Selecciona el rango con el slider — Aceptar para confirmar",
                 color='#e6edf3', fontsize=11, fontweight='bold')
    ax.set_xlabel('Fecha', color='#8b949e')
    ax.set_ylabel('Precio', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    ax.grid(True, alpha=0.12, color='#30363d')
    for spine in ax.spines.values():
        spine.set_edgecolor('#21262d')

    idx_min = df_plot.index.min()
    idx_max = df_plot.index.max()
    x_min_num = date2num(idx_min)
    x_max_num = date2num(idx_max)
    tz = df_plot.index.tz
    ax.set_xlim(idx_min, idx_max)
    fig.subplots_adjust(bottom=0.09)
    pos = ax.get_position()

    highlight = [ax.axvspan(idx_min, idx_max, alpha=0.12, facecolor='#00d4aa', zorder=0)]

    ax.text(0.01, 0.99, f"Velas totales: {len(df):,}",
            transform=ax.transAxes, ha='left', va='top',
            fontsize=9, color='#8b949e',
            bbox=dict(facecolor='#0d1117', alpha=0.7, edgecolor='#30363d', linewidth=0.5))

    ax_slider = fig.add_axes([pos.x0, 0.04, pos.width, 0.030])
    ax_slider.set_facecolor('#21262d')
    slider = RangeSlider(ax_slider, '', x_min_num, x_max_num,
                         valinit=(x_min_num, x_max_num), valfmt='')

    bx0 = pos.x0
    bw = pos.width
    ax_reset   = fig.add_axes([bx0, 0.01, 0.07, 0.028])
    ax_manual  = fig.add_axes([bx0 + 0.08, 0.01, 0.09, 0.028])
    ax_cancel  = fig.add_axes([bx0 + bw - 0.17, 0.01, 0.08, 0.028])
    ax_accept  = fig.add_axes([bx0 + bw - 0.08, 0.01, 0.08, 0.028])

    for bax in [ax_reset, ax_manual, ax_cancel, ax_accept]:
        bax.set_facecolor('#21262d')

    btn_reset  = Button(ax_reset, 'Reset', color='#21262d', hovercolor='#30363d')
    btn_manual = Button(ax_manual, 'Manual', color='#21262d', hovercolor='#30363d')
    btn_cancel = Button(ax_cancel, 'Cancelar', color='#21262d', hovercolor='#3d1a1a')
    btn_accept = Button(ax_accept, 'Aceptar', color='#1D9E75', hovercolor='#15805e')

    for btn in [btn_reset, btn_manual, btn_cancel, btn_accept]:
        for t in btn.ax.texts:
            t.set_fontsize(7.5)

    btn_cancel.label.set_color('#f85149')
    btn_accept.label.set_color('white')

    fmt_dmy = '%d/%m/%Y'
    mx0 = pos.x0
    ax_start = fig.add_axes([mx0, 0.085, 0.15, 0.035])
    ax_end   = fig.add_axes([mx0 + 0.20, 0.085, 0.15, 0.035])
    ax_apply = fig.add_axes([mx0 + 0.38, 0.085, 0.07, 0.035])
    for bax in [ax_start, ax_end, ax_apply]:
        bax.set_facecolor('#21262d')
        bax.set_visible(False)

    start_box = TextBox(ax_start, 'Inicio:', initial=idx_min.strftime(fmt_dmy),
                        color='#21262d', hovercolor='#30363d')
    end_box   = TextBox(ax_end, 'Fin:', initial=idx_max.strftime(fmt_dmy),
                        color='#21262d', hovercolor='#30363d')
    btn_apply = Button(ax_apply, 'Ir', color='#1D9E75', hovercolor='#15805e')

    for box in [start_box, end_box]:
        box.label.set_color('#8b949e')
        box.label.set_fontsize(7)
        if hasattr(box, 'text_disp'):
            box.text_disp.set_color('white')
        for t in box.ax.texts:
            t.set_fontsize(8)

    btn_apply.label.set_color('white')
    for t in btn_apply.ax.texts:
        t.set_fontsize(8)

    for box in [start_box, end_box]:
        box.on_submit(lambda _: on_apply(None))

    manual_visible = False
    accion = 'ninguna'

    def actualizar_todo(vmin_num, vmax_num):
        nonlocal accion
        dmin = pd.Timestamp(num2date(vmin_num).replace(tzinfo=None)).tz_localize(tz)
        dmax = pd.Timestamp(num2date(vmax_num).replace(tzinfo=None)).tz_localize(tz)
        highlight[0].remove()
        highlight[0] = ax.axvspan(dmin, dmax, alpha=0.12, facecolor='#00d4aa', zorder=0)
        ax.set_title(f"Rango: {dmin:%d %b %Y, %H:%M}  →  {dmax:%d %b %Y, %H:%M}",
                     color='#ff9900', fontsize=11, fontweight='bold')
        if manual_visible:
            start_box.set_val(dmin.strftime(fmt_dmy))
            end_box.set_val(dmax.strftime(fmt_dmy))
        fig.canvas.draw_idle()

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
        fig.subplots_adjust(bottom=0.14 if manual_visible else 0.09)
        fig.canvas.draw_idle()

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
        except Exception as e:
            print(f"      ⚠️ Error al aplicar fecha manual: {e}")

    def on_reset(event):
        sincronizar_slider(idx_min, idx_max)

    def on_accept(event):
        nonlocal accion
        accion = 'aceptar'
        plt.close(fig)

    def on_cancel(event):
        nonlocal accion
        accion = 'cancelar'
        plt.close(fig)

    def on_key(event):
        if event.key == 'escape':
            nonlocal accion
            accion = 'cancelar'
            plt.close(fig)

    def on_close(event):
        nonlocal accion
        if accion == 'ninguna':
            accion = 'cancelar'

    slider.on_changed(on_slider)
    btn_reset.on_clicked(on_reset)
    btn_manual.on_clicked(on_manual_toggle)
    btn_accept.on_clicked(on_accept)
    btn_cancel.on_clicked(on_cancel)
    btn_apply.on_clicked(on_apply)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('close_event', on_close)

    plt.show()

    if accion == 'cancelar':
        print("      ❌ Selección cancelada por el usuario.")
        sys.exit(0)

    vmin, vmax = slider.val
    dmin = pd.Timestamp(num2date(vmin).replace(tzinfo=None)).tz_localize(df.index.tz)
    dmax = pd.Timestamp(num2date(vmax).replace(tzinfo=None)).tz_localize(df.index.tz)

    tolerancia = pd.Timedelta(seconds=1)
    if abs(dmin - idx_min) > tolerancia or abs(dmax - idx_max) > tolerancia:
        print(f"      Rango seleccionado: {dmin} → {dmax}")
        return dmin, dmax

    print("        Sin selección — usando dataset completo.")
    return None, None
# endregion

if __name__ == "__main__":
    # region ── 3. VALIDACIÓN ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DESCRIPTIVO — {CONFIG['nombre']} ({CONFIG['tf']})")
    print(f"{'='*60}")
    print("[1/5] Validando configuración...")

    # Factores de anualización del TF base, calculados una única vez.
    # None si el TF no es reconocible; cada consumidor aplica su propio fallback.
    try:
        FACTORES_TF = get_factores(CONFIG['activo'], CONFIG['tf'])
    except (ValueError, KeyError, TypeError):
        FACTORES_TF = None
    # endregion
    # region ── 4. CARGA ──────────────────────────────────────────────────────────────────
    print("[2/5] Cargando archivo CSV...")
    df = pd.read_csv(CONFIG['input_path'], parse_dates=['timestamp']).set_index('timestamp')
    
    # --- PARCHE DE SEGURIDAD PARA LAS FECHAS ---
    # Nos aseguramos de que el índice sea Datetime y no un objeto o texto
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[df.index.notna()] 
    es_datetime_valido = isinstance(df.index, pd.DatetimeIndex)
    
    # Rango enviado por la GUI (variables de entorno, formato YYYY-MM-DD).
    # Si está presente tiene prioridad sobre el selector interactivo, que bajo la
    # GUI (backend Agg) nunca llega a mostrarse.
    gui_rango_inicio = os.environ.get('GUI_RANGO_INICIO')
    gui_rango_fin    = os.environ.get('GUI_RANGO_FIN')
    if gui_rango_inicio and gui_rango_fin and es_datetime_valido:
        print(f"      Filtrando al rango de la GUI: {gui_rango_inicio} → {gui_rango_fin}")
        df_rango = df.loc[gui_rango_inicio:gui_rango_fin]
        if len(df_rango) > 0:
            df = df_rango.copy()
        else:
            print("      ⚠ El rango de la GUI no contiene datos — usando dataset completo.")
        es_datetime_valido = isinstance(df.index, pd.DatetimeIndex) and df.index.is_monotonic_increasing
    else:
        # Selector interactivo de rango (modo standalone)
        start_sel, end_sel = seleccionar_rango_interactivo(df)
        if start_sel is not None:
            print(f"      Filtrando al rango seleccionado...")
            df = df.loc[start_sel:end_sel].copy()
            es_datetime_valido = isinstance(df.index, pd.DatetimeIndex) and df.index.is_monotonic_increasing


    # ── NOMBRE DEL PDF SEGÚN EL RANGO REAL DE DATOS ──
    nombre_activo = CONFIG['nombre'].replace('/', '_')
    inicio = df.index.min().strftime('%Y-%m-%d')
    fin = df.index.max().strftime('%Y-%m-%d')

    # En modo GUI usar directorio temporal, en standalone usar ruta fija
    if 'GUI_METRICS_OUTPUT' in os.environ:
        OUTPUT_DIR = os.path.dirname(os.environ['GUI_METRICS_OUTPUT'])
    else:
        OUTPUT_DIR = INFORMES_DIR

    OUTPUT_PDF = os.path.join(
        OUTPUT_DIR, nombre_activo,
        f"informe_{nombre_activo}_{CONFIG['tf']}_{inicio}_to_{fin}.pdf"
    )
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)
    # Se escribe primero a un .tmp y solo se renombra al nombre final si el PDF
    # se completa con éxito. Si el proceso muere a mitad de generación (p.ej. la
    # GUI se cierra mientras el análisis sigue corriendo y Qt mata el QProcess
    # hijo), el archivo final nunca queda con un PDF corrupto/truncado: como
    # mucho queda huérfano el .tmp.
    OUTPUT_PDF_TMP = OUTPUT_PDF + '.tmp'
    
    # ── DETECCIÓN Y COMPROBACIÓN TEMPORAL UNIFICADA ─────────────────────────────────────────────
    # [AÑADIDO] Sustituye la dependencia rígida de FACTORES (manual) por una
    # detección dinámica de la temporalidad real a partir de los timestamps del
    # propio archivo. Si el archivo no tiene timestamps fiables (p. ej. viene en
    # TICKS, volumen o rango en vez de tiempo fijo), las métricas que dependan del
    # tiempo se omiten mostrando un aviso explícito en vez de hacer fallar el script.
    velas_por_dia  = None
    velas_por_anio = None
    tipo_muestreo  = 'sin_tiempo'
    
    es_datetime_valido = isinstance(df.index, pd.DatetimeIndex) and df.index.is_monotonic_increasing
    
    if es_datetime_valido and len(df) > 2:
        diffs_seg = df.index.to_series().diff().dropna().dt.total_seconds()
        diffs_seg = diffs_seg[diffs_seg > 0]  # descarta timestamps duplicados
    
        if len(diffs_seg) > 0:
            mediana_seg = diffs_seg.median()
            cv = diffs_seg.std() / mediana_seg if mediana_seg > 0 else np.inf
            # Si el TF está definido, calcular el gap esperado en segundos
            _tf = CONFIG.get('tf', '')
            _m = re.match(r'(\d+)([a-z]+)', _tf.lower().strip())
            _mapa_unidad = {'s': 1, 'sec': 1, 'm': 60, 'min': 60, 'h': 3600, 'd': 86400, 'w': 604800}
            esperado_seg = None
            if _m:
                num, unidad = int(_m.group(1)), _m.group(2)
                if unidad in _mapa_unidad:
                    esperado_seg = num * _mapa_unidad[unidad]
            # Si la mediana coincide con el gap esperado (±20%), es tiempo fijo
            coincide_frecuencia = (esperado_seg is not None and
                                   0.8 * esperado_seg <= mediana_seg <= 1.2 * esperado_seg)
            tipo_muestreo = 'tiempo_fijo' if (cv < 0.15 or coincide_frecuencia) else 'evento'
    
            rango_total_seg  = (df.index[-1] - df.index[0]).total_seconds()
            rango_total_dias = rango_total_seg / 86400
    
            if rango_total_dias > 0:
                velas_por_dia  = len(df) / rango_total_dias
                velas_por_anio = velas_por_dia * 365

                # Aviso de divergencia: compara la densidad de velas REAL (medida
                # a partir de los huecos entre timestamps, extrapolada a un año)
                # contra la que asume get_factores() según CONFIG['activo']
                # (p.ej. 390min/día para STOCK = sesión regular NYSE). Se compara
                # en base ANUAL (no 'dia'): FACTORES_TF['dia'] es velas por día
                # DE TRADING, mientras que velas_por_dia empírico se mide sobre
                # días de CALENDARIO (incluye fines de semana) — comparar esas
                # dos directamente da un falso positivo sistemático de ~30% en
                # cualquier activo que no opere 7/7 (STOCK, FUTURO, FOREX). Al
                # multiplicar ambas por su respectivo conteo de días/año el
                # efecto de fin de semana se cancela y quedan en la misma base.
                # No sustituye la tabla estática (mantenerla da estabilidad/
                # comparabilidad entre corridas con distinto rango de fechas);
                # solo avisa cuando el supuesto no encaja.
                if FACTORES_TF:
                    velas_anio_esperado = FACTORES_TF['anual']
                    if velas_anio_esperado:
                        divergencia = abs(velas_por_anio - velas_anio_esperado) / velas_anio_esperado
                        if divergencia > 0.20:
                            print(f"  ⚠ Densidad de velas real (~{velas_por_anio:.0f}/anio) difiere "
                                  f"{divergencia:.0%} del supuesto para activo '{CONFIG['activo']}' "
                                  f"(~{velas_anio_esperado:.0f}/anio). Las metricas anualizadas "
                                  f"(Sharpe, Sortino, Calmar...) podrian no reflejar el horario "
                                  f"de sesion real de este activo.")

    # endregion
    # region ── 5. INTEGRIDAD y DETECCIÓN DE DATOS y COLUMNAS ────────────────────────────
    print("[3/5] Verificando integridad de datos...")
    if 'interpolado' not in df.columns: df['interpolado'] = 0
    if 'anomalia'    not in df.columns: df['anomalia']    = 0
    if 'ER'          not in df.columns: df['ER']          = 0.0
    
    # endregion
    # region ── 6. CÁLCULO DE RETORNOS ────────────────────────────────────────────────────
    print("[4/5] Calculando métricas estadísticas...")
    df['retorno'] = np.log(df['close'] / df['close'].shift(1))
    df['retorno'] = df['retorno'].fillna(0)
    r  = df['retorno']

    # Estadísticos base de la serie de retornos, calculados una única vez y
    # reutilizados en métricas, terminal y páginas del PDF.
    r_media = r.mean()
    r_std   = r.std()
    p05_r   = np.percentile(r, 5)
    p01_r   = np.percentile(r, 1)
    # CVaR de todo el periodo: media de los retornos por debajo del percentil 5
    _cola_r = r[r <= p05_r]
    p05_cola_r = float(_cola_r.mean()) if len(_cola_r) else float(p05_r)

    # endregion
    # region ── 7. AUDITORÍA DE CALIDAD Y CONTROL DE INTERPOLACIÓN ───────────────────────
    # ==============================================================================
    # [CONFIGURACIÓN] Cambia el nombre si tu columna se llama diferente (ej. 'is_interpolated')
    col_flag = 'interpolado' 
    
    if col_flag in df.columns:
        total_filas = len(df)
        total_interpoladas = int(df[col_flag].sum())
        total_originales = total_filas - total_interpoladas
        
        # Calculamos la distancia real por calendario basada en tus fechas exactas
        fecha_inicio = df.index.min()
        fecha_fin = df.index.max()
        segundos_reales = (fecha_fin - fecha_inicio).total_seconds()
        if segundos_reales > 0:
            vpd_teorico = FACTORES_TF['dia'] if FACTORES_TF else None
            if vpd_teorico is not None and vpd_teorico > 0:
                velas_teoricas = int((segundos_reales / 86400) * vpd_teorico) + 1
            else:
                velas_teoricas = int(segundos_reales / 3600) + 1
        else:
            velas_teoricas = total_filas
        
        # Cálculos de control
        filas_fantasma = max(0, velas_teoricas - total_filas)
        pct_completitud = total_filas / velas_teoricas * 100 if velas_teoricas > 0 else 0
    
        print(f"\n📊 [AUDITORÍA] DIAGNÓSTICO DE CONTROL DE CALIDAD:")
        print(f"======================================================================")
        print(f" • Velas teóricas por calendario:  {velas_teoricas:>12,} velas.")
        print(f" • Filas físicas en tu archivo:    {total_filas:>12,} velas.")
        print(f"   └── {'⚠️' if filas_fantasma > 0 else '✅'} Huecos temporales:          {filas_fantasma:>12,} velas ({pct_completitud:.2f}% completitud)")
        print(f" 📊 DESGLOSE DE LOS DATOS DEL ARCHIVO:")
        print(f" ======================================================================")
        print(f"    Total filas en el archivo:                    {total_filas:>10,}")
        print(f"    ├── 🟢 Datos reales (Exchange):               {total_originales:>10,}  ({total_originales/total_filas:.2%})")
        print(f"    └── 🔵 Datos rellenados (Script):             {total_interpoladas:>10,}  ({total_interpoladas/total_filas:.2%})")
        if 'anomalia' in df.columns:
            total_anomalias = int(df['anomalia'].sum())
            total_huecos_puros = total_interpoladas - total_anomalias
            print(f"        ├── 🟡 Bad Ticks (anomalías):            {total_anomalias:>10,}  ({total_anomalias/total_filas:.2%})")
            print(f"        └── ⚪ Huecos temporales:                 {total_huecos_puros:>10,}  ({total_huecos_puros/total_filas:.2%})")
        print(f" ======================================================================\n")
    else:
        print(f"⚠️ No se encontró la columna '{col_flag}'. Verifica el nombre exacto en tu CSV.")
    
    # endregion
    # region ── 8. CONFIGURACIÓN DE BLOQUES MULTI-TIMEFRAME ──────────────────────────────
    activo_cfg = CONFIG['activo']
    tf_cfg = CONFIG['tf']
    
    if FACTORES_TF is not None:
        bloques = FACTORES_TF
    else:
        bloques = {'dia': 1, 'semanal': 7, 'mensual': 30, 'trimestral': 90, 'anual': 365}
    
    def normalizar_serie(serie, bloques_agrupacion):
        s = serie.groupby(np.arange(len(serie)) // int(bloques_agrupacion)).sum()
        return s[np.isfinite(s)]
    
    series_temporales = {}
    stats_temporales = {}
    
    for periodo, n_bloques in bloques.items():
        if n_bloques < 1:
            continue
        series_temporales[periodo] = normalizar_serie(df['retorno'], n_bloques)
        s_actual = series_temporales[periodo]
        
        # Max DD Macro
        cum_ret_p = np.exp(s_actual.cumsum())
        mdd_p = ((cum_ret_p - cum_ret_p.cummax()) / cum_ret_p.cummax()).min()
        
        # Max DD Intra-periodo (vectorizado: cumsum/cummax por grupo en C)
        bloques_id = np.arange(len(df)) // int(n_bloques)
        _ret = df['retorno']
        _cum_log = _ret.groupby(bloques_id).cumsum()
        _cum_eq = np.exp(_cum_log)
        _cum_eq_max = _cum_eq.groupby(bloques_id).cummax().clip(lower=1.0)
        _dd = (_cum_eq - _cum_eq_max) / _cum_eq_max
        dds_por_bloque = _dd.groupby(bloques_id).min()
        
        peor_dd_intrabloque = dds_por_bloque.min()
        idx_peor_bloque = dds_por_bloque.idxmin()
        
        if es_datetime_valido and len(df) > 0:
            fechas_bloque = df.index[bloques_id == idx_peor_bloque]
            fecha_peor = fechas_bloque[0].strftime('%Y-%m-%d') if len(fechas_bloque) > 0 else "N/A"
        else:
            fecha_peor = "N/A"
        
        if len(s_actual) > 2:
            stat_jb_p, p_jb_p = stats.jarque_bera(s_actual)
        else:
            stat_jb_p, p_jb_p = 0.0, 1.0
        
        # 1. Crear el diccionario base
        stats_temporales[periodo] = {
            'media': s_actual.mean(),
            'std':   s_actual.std(),
            'max_dd': mdd_p,
            'max_dd_interno': peor_dd_intrabloque,
            'fecha_peor': fecha_peor,
            'p_value_normalidad': p_jb_p
        }
        
        # 2. Análisis de Dependencia Estadística (ACF/PACF)
        n_lags = min(40, len(s_actual) // 4) 
        if len(s_actual) > n_lags + 5:
            acf_vals = acf(s_actual, nlags=n_lags)
            pacf_vals = pacf(s_actual, nlags=n_lags)
            umbral_dependencia = 1.96 / np.sqrt(len(s_actual))
        else:
            acf_vals, pacf_vals, umbral_dependencia = [], [], 0.0
    
        # 3. Actualizar el diccionario existente
        stats_temporales[periodo].update({
            'acf_vals': acf_vals,
            'pacf_vals': pacf_vals,
            'umbral': umbral_dependencia
        })
        
    # Variables globales para acceso rápido
    r_diario_real = series_temporales['dia']
    ret_diario, vol_diaria = stats_temporales['dia']['media'], stats_temporales['dia']['std']
    ret_semanal, vol_semanal = stats_temporales['semanal']['media'], stats_temporales['semanal']['std']
    ret_mensual, vol_mensual = stats_temporales['mensual']['media'], stats_temporales['mensual']['std']
    ret_trimestral, vol_trimestral = stats_temporales['trimestral']['media'], stats_temporales['trimestral']['std']
    ret_anual, vol_anual = stats_temporales['anual']['media'], stats_temporales['anual']['std']
    
    # Función de validación temporal
    def validar_output_terminal(stats_temporales):
        print("\n" + "="*50)
        print("VALIDACIÓN: ANÁLISIS DE ARQUITECTURA TEMPORAL")
        print("="*50)
        print(f"{'Escala':<12} | {'Lag 1':<10} | {'Tipo':<15} | {'Umbral'}")
        print("-" * 50)
        orden = ['dia', 'semanal', 'mensual', 'trimestral', 'anual']
        for periodo in orden:
            if periodo in stats_temporales:
                data = stats_temporales[periodo]
                pacf = data['pacf_vals']
                umbral = data['umbral']
                val = pacf[1] if len(pacf) > 1 else 0.0
                if val > umbral: estado = "INERCIA"
                elif val < -umbral: estado = "REVERSIÓN"
                else: estado = "RUIDO"
                print(f"{periodo.capitalize():<12} | {val:>8.4f} | {estado:<15} | {umbral:.4f}")
        print("="*50 + "\n")
    
    validar_output_terminal(stats_temporales)
    # endregion
    # region ── 9. MÉTRICAS ──────────────────────────────────────────────────────────────
    
    # ==========================================================================
    # region PASO 1: Cálculos de Volatilidad y Retornos Anualizados/Diarios Compuestos
    # Calcula la dispersión del precio y el rendimiento compuesto esperado en base al régimen operativo anual del activo.
    if velas_por_anio is not None:
        velas_anual_regimen = FACTORES_TF['anual'] if FACTORES_TF else velas_por_anio
        vol_anual  = r_std * np.sqrt(velas_anual_regimen)
        vol_diaria = r_std * np.sqrt(velas_por_dia)
        
        media_log_diaria_real = r_diario_real.mean()
        ret_diario = np.exp(media_log_diaria_real) - 1
        
        dias_ano_regimen = 365 if CONFIG['activo'] == 'CRYPTO' else 252
        ret_anual  = np.exp(media_log_diaria_real * dias_ano_regimen) - 1
        
        rf_anual   = CONFIG.get('rf_rate', 0.0) / 100.0
        if rf_anual > 0:
            print(f"      Rf rate: {rf_anual:.4%} (configurado)")
        else:
            print(f"      Rf rate: 0.00% (sin tasa libre de riesgo)")
        sharpe     = (ret_anual - rf_anual) / vol_anual if (vol_anual is not None and vol_anual != 0 and vol_anual != 0.0) else 0
        excess_ret = r - (rf_anual / velas_anual_regimen)
        downside_std = excess_ret[excess_ret < 0].std() * np.sqrt(velas_anual_regimen)
        sortino = (ret_anual - rf_anual) / downside_std if downside_std != 0 else 0.0
        calmar_disponible = True
    else:
        # AQUÍ ESTÁ EL CAMBIO: Asignamos 0.0 en lugar de None
        vol_anual = vol_diaria = ret_anual = ret_diario = sharpe = sortino = 0.0
        calmar_disponible = False
        
    # endregion
    # ==========================================================================
    # region PASO 2: Cálculo del Máximo Drawdown (Max DD) Histórico
    # Mide la peor pérdida acumulada de pico a valle sufrida por el precio a lo largo del histórico.
    cum_returns = np.exp(r.cumsum())
    peak        = cum_returns.cummax()
    mdd         = ((cum_returns - peak) / peak).min()
    # endregion
    # ==========================================================================
    # region PASO 3: Cálculo de Tiempos de Recuperación en Drawdown
    # Mide la cantidad de velas y el tiempo exacto transcurrido entre el pico máximo y su break-even o recuperación completa.
    drawdown_series = (cum_returns - peak) / peak
    en_drawdown      = drawdown_series < 0
    
    bloques_dd       = (en_drawdown != en_drawdown.shift()).cumsum()
    duraciones_velas = en_drawdown.groupby(bloques_dd).sum()
    duraciones_velas = duraciones_velas[duraciones_velas > 0]

    # Piezas sueltas de la recuperación, que la tarjeta de Max Drawdown de la
    # GUI muestra por separado (no puede reaprovechar `recovery_str`, que ya
    # viene formateado en una línea para la tabla del PDF). Se inicializan aquí
    # porque las ramas de abajo no siempre llegan a definirlas: sin drawdown, o
    # con archivos de TICKS que no tienen temporalidad.
    duracion_str = None
    rango_fechas_str = None
    dd_recuperado = None   # True recuperado · False sigue en DD · None sin DD

    if len(duraciones_velas) > 0:
        # El bloque a reportar es el del DRAWDOWN MAS PROFUNDO (el mismo que
        # produce `mdd` en el PASO 2), no el de mayor DURACION — si no, el
        # "Tiempo de recuperacion" mostrado junto al Max Drawdown Historico
        # puede pertenecer a un episodio distinto (uno largo pero poco
        # profundo) en vez de al que realmente tuvo la caida de `mdd`.
        idx_mdd            = drawdown_series.idxmin()
        bloque_max_id       = bloques_dd.loc[idx_mdd]
        recovery_velas_max = int(duraciones_velas.loc[bloque_max_id])
    
        if es_datetime_valido:
            indices_bloque = en_drawdown[bloques_dd == bloque_max_id].index
    
            # Fecha del PICO (breakeven inicial): la última vela en máximo, justo antes de empezar a caer.
            pos_primera_en_dd = df.index.get_loc(indices_bloque[0])
            ts_pico = df.index[pos_primera_en_dd - 1] if pos_primera_en_dd > 0 else indices_bloque[0]
    
            # Fecha de RECUPERACIÓN COMPLETA: la primera vela donde el precio vuelve a superar (o igualar) el pico anterior.
            pos_ultima_en_dd = df.index.get_loc(indices_bloque[-1])
            dd_recuperado = pos_ultima_en_dd + 1 < len(df.index)
            if dd_recuperado:
                ts_recuperado = df.index[pos_ultima_en_dd + 1]
            else:
                ts_recuperado = indices_bloque[-1]  # aún no recuperado al final de los datos
    
            recovery_timedelta = ts_recuperado - ts_pico
            dias_rec    = recovery_timedelta.days
            horas_rec   = recovery_timedelta.seconds // 3600
            minutos_rec = (recovery_timedelta.seconds % 3600) // 60
    
            # Si la recuperación alcanza o supera 1 año (365 días), se descompone en años + el resto en d/h/m.
            if dias_rec >= 365:
                anios_rec    = dias_rec // 365
                dias_resto   = dias_rec % 365
                etiqueta_anio = "año" if anios_rec == 1 else "años"
                duracion_str = f"{anios_rec} {etiqueta_anio} {dias_resto}d {horas_rec}h {minutos_rec}m"
            else:
                duracion_str = f"{dias_rec}d {horas_rec}h {minutos_rec}m"
    
            # Formato de fecha: con hora solo si el timeframe es intradía
            fmt_fecha = '%Y-%m-%d %H:%M' if recovery_timedelta < pd.Timedelta(days=3) else '%Y-%m-%d'
            rango_fechas_str = f"{ts_pico.strftime(fmt_fecha)} → {ts_recuperado.strftime(fmt_fecha)}"
    
            recovery_str = f"{duracion_str}   ({rango_fechas_str})"
        else:
            recovery_str = "N/A (TICKS — sin temporalidad)"
    else:
        recovery_velas_max = 0
        recovery_str = "0d 0h 0m" if es_datetime_valido else "N/A (TICKS — sin temporalidad)"
    
    # endregion
    # ==========================================================================
    # region PASO 4: Cálculo de la Profundidad del Drawdown Medio
    # Obtiene el promedio del punto más bajo registrado de manera aislada en cada episodio consecutivo de caída.
    if en_drawdown.any():
        profundidad_por_episodio = drawdown_series[en_drawdown].groupby(bloques_dd[en_drawdown]).min()
        drawdown_medio = profundidad_por_episodio.mean()
        num_episodios_dd = len(profundidad_por_episodio)
    else:
        drawdown_medio = None
        num_episodios_dd = 0
    
    # endregion
    # ==========================================================================
    # region PASO 5: Cálculo del Calmar Ratio
    # Relaciona el retorno compuesto anualizado frente al máximo drawdown histórico para medir la recompensa por unidad de riesgo.
    if calmar_disponible and mdd != 0:
        calmar_ratio = ret_anual / abs(mdd)
    else:
        calmar_ratio = None
    
    # endregion
    # ==========================================================================
    # region PASO 6: Inicialización de Parámetros Auxiliares de Riesgo y Formatos
    # Establece las funciones de formateo numérico de strings y las variables de temporalidad del activo.
    tf_actual = CONFIG['tf']

    def fmt_pct(valor, decimales=2):
        return f"{valor*100:.{decimales}f}%" if valor is not None else "N/A (TICKS)"

    def fmt_num(valor, decimales=4):
        return f"{valor:.{decimales}f}" if valor is not None else "N/A (TICKS)"

    # endregion
    # ==========================================================================
    # region PASO 7: Test de Normalidad de Jarque-Bera y Lógica Adaptativa de Módulo VaR
    # Evalúa la asimetría y curtosis de la serie para decidir dinámicamente si aplicar un cálculo de VaR Histórico o Paramétrico.
    r_clean = r[np.isfinite(r)].dropna()
    
    if len(r_clean) > 0:
        stat_jb, p_jb = stats.jarque_bera(r_clean)
    else:
        stat_jb, p_jb = 0.0, 1.0 # Valores seguros si la serie está vacía
    
    if p_jb < 0.05:
        var_95 = np.percentile(r_clean, 5)
        var_99 = np.percentile(r_clean, 1)
        lbl_var_95, lbl_var_99 = f'VaR 95% Histórico ({tf_actual})', f'VaR 99% Histórico ({tf_actual})'
        # CVaR histórico: media de la cola más allá del VaR
        cola95 = r_clean[r_clean <= var_95]
        cola99 = r_clean[r_clean <= var_99]
        cvar_95 = float(cola95.mean()) if len(cola95) else float(var_95)
        cvar_99 = float(cola99.mean()) if len(cola99) else float(var_99)
        lbl_cvar_95, lbl_cvar_99 = f'CVaR 95% Histórico ({tf_actual})', f'CVaR 99% Histórico ({tf_actual})'
    else:
        z_95, z_99 = stats.norm.ppf(0.05), stats.norm.ppf(0.01)
        var_95 = r_clean.mean() + z_95 * r_clean.std()
        var_99 = r_clean.mean() + z_99 * r_clean.std()
        lbl_var_95, lbl_var_99 = f'VaR 95% Paramétrico ({tf_actual})', f'VaR 99% Paramétrico ({tf_actual})'
        # CVaR paramétrico: media - std·φ(z)/Φ(z)
        cvar_95 = r_clean.mean() - r_clean.std() * (stats.norm.pdf(1.645) / 0.95)
        cvar_99 = r_clean.mean() - r_clean.std() * (stats.norm.pdf(2.326) / 0.99)
        lbl_cvar_95, lbl_cvar_99 = f'CVaR 95% Paramétrico ({tf_actual})', f'CVaR 99% Paramétrico ({tf_actual})'
    
    val_var_95, val_var_99 = f"{-var_95*100:.4f}%", f"{-var_99*100:.4f}%"
    val_cvar_95, val_cvar_99 = f"{-cvar_95*100:.4f}%", f"{-cvar_99*100:.4f}%"
    
    # endregion
    # ==========================================================================
    # region PASO 8: Cálculo de Persistencia de Rangos vía Ratio de Eficiencia (ER)
    # Cuantifica el ruido direccional de las series analizando cuántas velas rompen los umbrales de tendencia o ruido locales.
    er_medio         = df['ER'].mean()
    er_std           = df['ER'].std()
    
    umbral_tendencia = min(0.95, er_medio + er_std)
    umbral_ruido     = max(0.05, er_medio - er_std)
    
    total_tendencia  = (df['ER'] > umbral_tendencia).sum()
    total_ruido      = (df['ER'] < umbral_ruido).sum()
    
    # Agrupa y contabiliza las ocurrencias del coeficiente Hurst en zonas de tendencia, paseo aleatorio o reversión a la media.
    
    total_tendencia_h = ((df['hurst'] > 0.58)).sum()
    total_aleatorio_h  = ((df['hurst'] >= 0.52) & (df['hurst'] <= 0.58)).sum()
    total_reversion_h  = ((df['hurst'] < 0.52)).sum()
    
    # endregion
    # ==========================================================================
    # region PASO 9: Análisis de Periodos Tendenciales (H > 0.6) y Mean Reversion (H < 0.52)
    # ==========================================================================
    mask_h_tend = df['hurst'] > 0.6
    mask_h_rev  = df['hurst'] < 0.52
    
    # --- Rachas consecutivas de tendencia ---
    en_tend = mask_h_tend
    rachas_tend = (en_tend != en_tend.shift()).cumsum()
    rachas_tend_len = en_tend.groupby(rachas_tend).sum()
    rachas_tend_len = rachas_tend_len[rachas_tend_len > 0]
    if len(rachas_tend_len) > 0:
        duracion_media_racha_tend = rachas_tend_len.mean()
        duracion_max_racha_tend   = rachas_tend_len.max()
    else:
        duracion_media_racha_tend = None
        duracion_max_racha_tend   = None
    
    pct_tiempo_tend = mask_h_tend.sum() / len(df)
    pct_tiempo_rev  = mask_h_rev.sum() / len(df)
    
    if velas_por_dia is not None and velas_por_dia > 0:
        velas_dia_tend = FACTORES_TF['dia'] if FACTORES_TF else velas_por_dia
        if mask_h_tend.any():
            ret_dia_tend = r[mask_h_tend].mean() * velas_dia_tend * 100
            vol_dia_tend = r[mask_h_tend].std() * np.sqrt(velas_dia_tend) * 100
            sharpe_tend = ret_dia_tend / vol_dia_tend if vol_dia_tend != 0 else 0.0
        else:
            ret_dia_tend = vol_dia_tend = sharpe_tend = None
        if mask_h_rev.any():
            ret_dia_rev = r[mask_h_rev].mean() * velas_dia_tend * 100
            vol_dia_rev = r[mask_h_rev].std() * np.sqrt(velas_dia_tend) * 100
            sharpe_rev = ret_dia_rev / vol_dia_rev if vol_dia_rev != 0 else 0.0
        else:
            ret_dia_rev = vol_dia_rev = sharpe_rev = None
    else:
        ret_dia_tend = vol_dia_tend = ret_dia_rev = vol_dia_rev = sharpe_tend = sharpe_rev = None
    
    # endregion
    # ==========================================================================
    # region PASO 10: ER (vía KAMA) y Exponente de Hurst por horizonte
    # Cada Ventana de la GUI tiene su propio periodo de ER (con línea KAMA usando
    # sus constantes rápida/lenta) y su propia ventana rodante de Hurst.
    # General sigue usando las columnas ER/hurst del CSV limpiado (PASO 8/9).
    HORIZON_ER_KAMA = {
        'Scalping':     {'er': 14, 'fast': 2, 'slow': 30},
        'Daytrading':   {'er': 10, 'fast': 2, 'slow': 30},
        'Swingtrading': {'er': 20, 'fast': 2, 'slow': 30},
        'Position':     {'er': 30, 'fast': 3, 'slow': 50},
    }
    HORIZON_HURST_RANGO = {'Scalping': (50, 100), 'Daytrading': (100, 150),
                           'Swingtrading': (200, 250), 'Position': (300, 500)}

    _ER_H = {}     # horizonte → serie ER propia + umbrales + % regímenes
    _KAMA_H = {}   # horizonte → línea KAMA + valor actual + señal
    _HURST_H = {}  # horizonte → serie Hurst propia + ventana usada + regímenes

    try:
        if 'retorno_log' in df.columns:
            _ret_log_h = df['retorno_log'].fillna(0.0)
        else:
            _ret_log_h = np.log(df['close'] / df['close'].shift(1)).fillna(0.0)
        _retornos_h = np.array(_ret_log_h.values, dtype=np.float64, copy=True)

        for _h, _p in HORIZON_ER_KAMA.items():
            er_h = calcular_er_series(_ret_log_h, _p['er'])
            umbrales_h = calcular_umbrales_er(er_h)
            _ER_H[_h] = {
                'serie': er_h,
                'periodo': _p['er'],
                'umbrales': umbrales_h,
                'pct_tendencia': float((er_h > umbrales_h['umbral_tendencia']).mean()),
                'pct_ruido': float((er_h < umbrales_h['umbral_ruido']).mean()),
            }

            kama_vals = calcular_kama_numba(
                np.array(df['close'].values, dtype=np.float64),
                np.array(er_h.values, dtype=np.float64),
                float(_p['fast']), float(_p['slow'])
            )
            kama_serie = pd.Series(kama_vals, index=df.index)
            precio_actual = float(df['close'].dropna().iloc[-1])
            kama_actual = float(kama_serie.dropna().iloc[-1])
            _KAMA_H[_h] = {
                'serie': kama_serie,
                'actual': kama_actual,
                'senal': 'ALCISTA (precio > KAMA)' if precio_actual > kama_actual
                         else 'BAJISTA (precio < KAMA)',
                'fast': _p['fast'], 'slow': _p['slow'],
            }

            # Ventana Hurst adaptativa: máximo del rango con histórico suficiente,
            # si no el mínimo; si tampoco llega, este horizonte se omite.
            v_min, v_max = HORIZON_HURST_RANGO[_h]
            if len(_retornos_h) >= v_max * 10:
                ventana_h = v_max
            elif len(_retornos_h) >= v_min * 10:
                ventana_h = v_min
            else:
                continue

            lags_h = np.array([l for l in (8, 16, 32, 64, 128, 256) if l < ventana_h],
                              dtype=np.int64)
            paso_h = max(2, ventana_h // 20)
            hurst_vals_h = calcular_hurst_array(_retornos_h, ventana_h, paso_h, lags_h)
            hurst_serie_h = pd.Series(hurst_vals_h, index=df.index)
            hurst_serie_h = hurst_serie_h.interpolate(method='linear').bfill().ffill().fillna(0.5)
            _HURST_H[_h] = {
                'serie': hurst_serie_h,
                'ventana': ventana_h,
                'regimen': contar_regimen_hurst(hurst_serie_h),
            }

        # 'General' reutiliza la serie ER ya presente en el CSV limpiado (la
        # misma que usan las páginas de métricas generales) en vez de
        # recalcular una nueva, pero le añade su propia línea KAMA — antes
        # la página de régimen ER para 'General' no tenía KAMA.
        _er_general = df['ER']
        _umbrales_general = calcular_umbrales_er(_er_general)
        _ER_H['General'] = {
            'serie': _er_general,
            'periodo': 'base',
            'umbrales': _umbrales_general,
            'pct_tendencia': float((_er_general > _umbrales_general['umbral_tendencia']).mean()),
            'pct_ruido': float((_er_general < _umbrales_general['umbral_ruido']).mean()),
        }
        _kama_general_vals = calcular_kama_numba(
            np.array(df['close'].values, dtype=np.float64),
            np.array(_er_general.values, dtype=np.float64),
            2.0, 30.0
        )
        _kama_general_serie = pd.Series(_kama_general_vals, index=df.index)
        _precio_actual_g = float(df['close'].dropna().iloc[-1])
        _kama_actual_g = float(_kama_general_serie.dropna().iloc[-1])
        _KAMA_H['General'] = {
            'serie': _kama_general_serie,
            'actual': _kama_actual_g,
            'senal': 'ALCISTA (precio > KAMA)' if _precio_actual_g > _kama_actual_g
                     else 'BAJISTA (precio < KAMA)',
            'fast': 2, 'slow': 30,
        }
    except Exception as e:
        print(f"  ⚠ PASO 10: Error en ER/KAMA/Hurst por horizonte: {e}")
    # endregion
    # ==========================================================================
    # region PASO 11: Cálculos Multitemporales de Volatilidad Histórica Rodante (Rolling Vol)
    # Extrae ventanas rodantes estandarizadas (7d, 30d, 90d, 365d) parametrizando las velas operativas según el tipo de producto.
    velas_dia = FACTORES_TF['dia']
    velas_anual = FACTORES_TF['anual']

    dias_ano = 365 if activo_cfg == 'CRYPTO' else 252
    dias_trimestre = 90 if activo_cfg == 'CRYPTO' else 63
    
    vol_historica_total = r_std * np.sqrt(velas_anual)
    
    hv_7d  = r.rolling(window=int(7 * velas_dia)).std() * np.sqrt(velas_anual)
    hv_30d = r.rolling(window=int(30 * velas_dia)).std() * np.sqrt(velas_anual)
    hv_90d = r.rolling(window=int(dias_trimestre * velas_dia)).std() * np.sqrt(velas_anual)
    hv_365d = r.rolling(window=int(dias_ano * velas_dia)).std() * np.sqrt(velas_anual)
    
    val_hv_7d   = hv_7d.iloc[-1] if len(hv_7d) >= (7 * velas_dia) else None
    val_hv_30d  = hv_30d.iloc[-1] if len(hv_30d) >= (30 * velas_dia) else None
    val_hv_90d  = hv_90d.iloc[-1] if len(hv_90d) >= (dias_trimestre * velas_dia) else None
    val_hv_365d = hv_365d.iloc[-1] if len(hv_365d) >= (dias_ano * velas_dia) else None
    
    # --- Comparativa de Regímenes Geométricos de Tendencia Estructural ---
    ventana_sma = int(200 * velas_dia)
    
    if len(df) > ventana_sma:
        df['SMA_Regimen'] = df['close'].rolling(window=ventana_sma).mean()
        es_bull = df['close'] > df['SMA_Regimen']
        es_bear = df['close'] <= df['SMA_Regimen']
        
        ret_bull = r[es_bull].mean() * velas_anual
        vol_bull = r[es_bull].std() * np.sqrt(velas_anual)
        
        ret_bear = r[es_bear].mean() * velas_anual
        vol_bear = r[es_bear].std() * np.sqrt(velas_anual)
        
        str_bull = f"Ret: {ret_bull*100:.2f}% | Vol: {vol_bull*100:.2f}%"
        str_bear = f"Ret: {ret_bear*100:.2f}% | Vol: {vol_bear*100:.2f}%"
    else:
        str_bull = "N/A"
        str_bear = "N/A"
    
    # endregion
    # ==========================================================================
    # region PASO 12: Análisis Estadístico de Correlaciones Temporales (Leverage Effect)
    # ==========================================================================
    ventana_vol = 0
    ventana_corr = 0
    corr_media_historica = 0.0
    corr_max = 0.0
    corr_min = 0.0
    tiempo_negativa = 0.0
    tiempo_positiva = 0.0
    hora_mas_volatil = 0
    vol_maxima = 0.0
    vol_por_hora = pd.Series(dtype=float)
    vol_relativa_asia = 0.0
    vol_relativa_london = 0.0
    vol_relativa_ny = 0.0
    vol_weekend_ratio = 0.0
    pivot_week = pd.DataFrame(dtype=float)
    pivot_month = pd.DataFrame(dtype=float)
    corr_24 = pd.DataFrame(dtype=float)
    overall_vol = 0.0
    
    try:
        if 'hora_utc' not in df.columns:
            df['hora_utc'] = df.index.hour

        # Hora LOCAL de cada plaza (vía IANA, tz_convert): incorpora el
        # cambio de horario de verano/invierno vela a vela, en vez de un
        # rango UTC fijo que solo es exacto la mitad del año. Misma
        # convención que SESIONES en core/candle_patterns.py (8:00-17:00
        # hora local de la plaza). Tokio no tiene horario de verano, pero
        # se calcula igual por uniformidad.
        _idx_utc = (df.index.tz_localize('UTC') if df.index.tz is None
                    else df.index.tz_convert('UTC'))
        df['hora_tokio'] = _idx_utc.tz_convert('Asia/Tokyo').hour
        df['hora_londres'] = _idx_utc.tz_convert(SESIONES['londres']['tz']).hour
        df['hora_ny'] = _idx_utc.tz_convert(SESIONES['ny']['tz']).hour

        velas_por_dia = FACTORES_TF['dia']
        ventana_vol = int(velas_por_dia)
        ventana_corr = int(velas_por_dia * 7)
        
        if CONFIG['tf'] == '1d':
            ventana_vol = 7
            ventana_corr = 30
    
        df['vol_rodante_tf'] = df['retorno'].rolling(window=ventana_vol).std()

        # Leverage effect: la correlación retorno-volatilidad es un fenómeno de
        # régimen diario. Calcular rolling().corr() sobre millones de velas
        # intradiarias es O(n·w) y domina el tiempo del PASO 11. Pre-agregando
        # a resolución diaria se reduce n de ~millones a ~miles sin pérdida
        # de información relevante (el leverage effect no se mide intradía).
        if CONFIG['tf'] != '1d' and velas_por_dia > 1:
            _daily = df['retorno'].groupby(df.index.date).agg(['sum', 'std'])
            _daily.columns = ['ret_d', 'vol_d']
            _ventana_corr_d = max(7, ventana_corr // int(velas_por_dia))
            _corr_series = _daily['ret_d'].rolling(window=_ventana_corr_d).corr(_daily['vol_d'])
            corr_media_historica = _corr_series.mean()
            corr_max = _corr_series.max()
            corr_min = _corr_series.min()
            _clean = _corr_series.dropna()
            tiempo_negativa = (_clean < 0).mean() * 100
            tiempo_positiva = (_clean > 0).mean() * 100
        else:
            df['corr_leverage'] = df['retorno'].rolling(window=ventana_corr).corr(df['vol_rodante_tf'])
            df_clean_corr = df.dropna(subset=['corr_leverage'])
            corr_media_historica = df_clean_corr['corr_leverage'].mean()
            corr_max = df_clean_corr['corr_leverage'].max()
            corr_min = df_clean_corr['corr_leverage'].min()
            tiempo_negativa = (df_clean_corr['corr_leverage'] < 0).mean() * 100
            tiempo_positiva = (df_clean_corr['corr_leverage'] > 0).mean() * 100

        # Leverage effect por horizonte de trading
        LEVERAGE_WINDOWS = {
            'Scalping':     {'vol_dias': 1,  'corr_dias': 2},
            'Daytrading':   {'vol_dias': 1,  'corr_dias': 7},
            'Swingtrading': {'vol_dias': 5,  'corr_dias': 30},
            'Position':     {'vol_dias': 21, 'corr_dias': 120},
        }
        _leverage_por_horizonte = {}
        if CONFIG['tf'] != '1d' and velas_por_dia > 1:
            for _hname, _w in LEVERAGE_WINDOWS.items():
                _vol_s = (_daily['vol_d'] if _w['vol_dias'] <= 1
                          else _daily['ret_d'].rolling(_w['vol_dias']).std())
                _corr_h = _daily['ret_d'].rolling(_w['corr_dias']).corr(_vol_s)
                _clean_h = _corr_h.dropna()
                _leverage_por_horizonte[_hname] = {
                    'media': _corr_h.mean(), 'max': _corr_h.max(), 'min': _corr_h.min(),
                    'neg_pct': (_clean_h < 0).mean() * 100, 'pos_pct': (_clean_h > 0).mean() * 100,
                    'vol_desc': ('intradía (realized)' if _w['vol_dias'] <= 1
                                 else f"rolling {_w['vol_dias']} días"),
                    'corr_desc': f"{_w['corr_dias']} días",
                }
        else:
            for _hname, _w in LEVERAGE_WINDOWS.items():
                _vol_h = df['retorno'].rolling(window=_w['vol_dias']).std()
                _corr_h = df['retorno'].rolling(window=_w['corr_dias']).corr(_vol_h)
                _clean_h = _corr_h.dropna()
                _leverage_por_horizonte[_hname] = {
                    'media': _corr_h.mean(), 'max': _corr_h.max(), 'min': _corr_h.min(),
                    'neg_pct': (_clean_h < 0).mean() * 100, 'pos_pct': (_clean_h > 0).mean() * 100,
                    'vol_desc': ('intradía (realized)' if _w['vol_dias'] <= 1
                                 else f"rolling {_w['vol_dias']} días"),
                    'corr_desc': f"{_w['corr_dias']} días",
                }
        
        if df['hora_utc'].nunique() > 1:
            vol_por_hora = df.groupby('hora_utc')['retorno'].std() * 100
            vol_por_hora = vol_por_hora.dropna()
            if vol_por_hora.empty:
                hora_mas_volatil = 0
                vol_maxima = 0.0
            else:
                hora_mas_volatil = vol_por_hora.idxmax()
                vol_maxima = vol_por_hora.max()
    
            # Estacionalidad temporal por sesión: 8:00-17:00 HORA LOCAL de
            # cada plaza (between es inclusivo: horas 8..16 = fin exclusivo
            # a las 17:00), misma convención que el filtro de sesión de la
            # pestaña Patrones. Al usar hora local convertida, el rango UTC
            # efectivo se ajusta solo con el DST de cada plaza.
            overall_vol = df['retorno'].std()
            if overall_vol > 0:
                asia = df[df['hora_tokio'].between(8, 16)]
                london = df[df['hora_londres'].between(8, 16)]
                ny = df[df['hora_ny'].between(8, 16)]
                vol_relativa_asia = asia['retorno'].std() / overall_vol if len(asia) > 50 else 0.0
                vol_relativa_london = london['retorno'].std() / overall_vol if len(london) > 50 else 0.0
                vol_relativa_ny = ny['retorno'].std() / overall_vol if len(ny) > 50 else 0.0
                weekday = df[df.index.dayofweek < 5]
                weekend = df[df.index.dayofweek >= 5]
                vol_weekend = weekend['retorno'].std() if len(weekend) > 20 else None
                vol_weekday = weekday['retorno'].std() if len(weekday) > 20 else None
                if vol_weekend is not None and vol_weekday is not None and vol_weekday > 0:
                    vol_weekend_ratio = vol_weekend / vol_weekday
    
            # ── Matrices para heatmaps ──
            df['retorno_pct'] = df['retorno'] * 100
            pivot_week = df.pivot_table(
                values='retorno_pct',
                index=df.index.dayofweek,
                columns='hora_utc',
                aggfunc='std'
            )
            pivot_week = pivot_week.reindex(columns=range(24), fill_value=0)
            pivot_month = df.pivot_table(
                values='retorno_pct',
                index=df.index.month,
                columns='hora_utc',
                aggfunc='std'
            )
            pivot_month = pivot_month.reindex(columns=range(24), fill_value=0)
            daily_profile = df.pivot_table(
                values='retorno',
                index=df.index.date,
                columns='hora_utc',
                aggfunc='first'
            ).reindex(columns=range(24), fill_value=0)
            daily_profile.columns = [f'hora_{h}' for h in daily_profile.columns]
            corr_24 = daily_profile.corr(method='spearman')
        else:
            print("      📅 Timeframe diario — no hay distribución horaria intradía.")
    
    except Exception as e:
        print(f"AVISO: Cálculo de correlaciones saltado: {e}")
    
    # endregion
    # ==========================================================================
    # region PASO 13: Desviación de la Ley de Escalado Fractal (Raíz de T)
    # ==========================================================================
    vol_diaria_real = stats_temporales['dia']['std']
    vol_mensual_real = stats_temporales['mensual']['std']
    # Buscamos la anomalía matemática. El bloque 'mensual' se agrupa en
    # FACTORES_TF['mensual'] velas (21 dias de trading para STOCK/FUTURO,
    # ~30.4 dias reales para CRYPTO — ver get_factores) mientras que 'dia' es
    # 1 bloque de referencia; usar un sqrt(30) fijo comparaba unidades
    # distintas (30 dias de calendario contra un bloque que en realidad
    # agrupa 21 o 30.4, segun el activo) y generaba una desviacion espuria
    # incluso cuando el activo sigue la ley de raiz de T perfectamente.
    factor_mensual_dias = (FACTORES_TF['mensual'] / FACTORES_TF['dia']) if FACTORES_TF else 30
    vol_mensual_teorica = vol_diaria_real * np.sqrt(factor_mensual_dias)
    desviacion_escalado = vol_mensual_real - vol_mensual_teorica
    
    # endregion
    # ==========================================================================
    # region PASO 14: Score de Confluencia Macro (Trend Confluence)
    # ==========================================================================
    score_tendencia = 0
    if series_temporales['mensual'].iloc[-1] > 0: score_tendencia += 1
    if series_temporales['semanal'].iloc[-1] > 0: score_tendencia += 1
    if series_temporales['dia'].iloc[-1] > 0:     score_tendencia += 1
    # endregion
    # ==========================================================================
    # region PASO 15: Volatility Clustering — Autocorrelación de Volatilidad (ARCH)
    # ==========================================================================
    retorno_sq = df['retorno'] ** 2
    r_sq_clean = retorno_sq.dropna()
    clustering_lag1 = np.nan
    clustering_lb_q = np.nan
    clustering_lb_p = np.nan
    clustering_presente = False
    
    if len(r_sq_clean) > 30:
        try:
            lags_sq = min(20, len(r_sq_clean) // 4 - 1)
            if lags_sq >= 2:
                acf_sq_full = acf(r_sq_clean, nlags=lags_sq)
                clustering_lag1 = acf_sq_full[1]
                lb_result = acf(r_sq_clean, nlags=min(12, lags_sq), qstat=True)
                clustering_lb_q = lb_result[1][-1]
                clustering_lb_p = lb_result[2][-1]
                clustering_presente = clustering_lb_p < 0.05
        except Exception:
            pass
    # endregion
    # ==========================================================================
    # region PASO 16: Estimadores de Volatilidad OHLC
    df_ohlc = df[['open', 'high', 'low', 'close']].replace(0, np.nan).dropna()

    log_hl = np.log(df_ohlc['high'] / df_ohlc['low'])
    log_co = np.log(df_ohlc['close'] / df_ohlc['open'])
    log_oc_prev = np.log(df_ohlc['open'] / df_ohlc['close'].shift(1))
    log_ho = np.log(df_ohlc['high'] / df_ohlc['open'])
    log_lo = np.log(df_ohlc['low'] / df_ohlc['open'])

    factor_park = 1 / (4 * np.log(2))
    parkinson_var = factor_park * (log_hl ** 2)
    parkinson_vol_periodo = np.sqrt(parkinson_var.mean())

    gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    gk_var = gk_var.clip(lower=0)
    gk_vol_periodo = np.sqrt(gk_var.mean())

    rs_var = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    rs_var = rs_var.clip(lower=0)
    rs_vol_periodo = np.sqrt(rs_var.mean())

    ctc_vol_periodo = r_std

    if velas_por_anio is not None:
        factor_anual_ohlc = FACTORES_TF['anual'] if FACTORES_TF else velas_por_anio
        parkinson_vol_anual = parkinson_vol_periodo * np.sqrt(factor_anual_ohlc)
        gk_vol_anual       = gk_vol_periodo * np.sqrt(factor_anual_ohlc)
        rs_vol_anual        = rs_vol_periodo * np.sqrt(factor_anual_ohlc)
        ctc_vol_anual        = ctc_vol_periodo * np.sqrt(factor_anual_ohlc)
    else:
        parkinson_vol_anual = gk_vol_anual = rs_vol_anual = ctc_vol_anual = None

    eficiencia_parkinson = (ctc_vol_periodo / parkinson_vol_periodo) ** 2 if parkinson_vol_periodo > 0 else None
    eficiencia_gk        = (ctc_vol_periodo / gk_vol_periodo) ** 2 if gk_vol_periodo > 0 else None
    eficiencia_rs         = (ctc_vol_periodo / rs_vol_periodo) ** 2 if rs_vol_periodo > 0 else None
    # endregion
    # ==========================================================================
    # region PASO 17: Tests de Estacionariedad (ADF/KPSS)
    MAX_OBS_STATIONARITY = 20000
    precio_test = df_ohlc['close']
    if len(precio_test) > MAX_OBS_STATIONARITY:
        paso_stat = len(precio_test) // MAX_OBS_STATIONARITY
        precio_test_sample = precio_test.iloc[::paso_stat]
        r_test_sample       = r.iloc[::paso_stat] if len(r) > MAX_OBS_STATIONARITY else r
    else:
        precio_test_sample = precio_test
        r_test_sample       = r

    try:
        adf_precio_stat, adf_precio_p, *_ = adfuller(precio_test_sample, autolag='AIC')
    except Exception:
        adf_precio_stat, adf_precio_p = np.nan, np.nan

    try:
        adf_ret_stat, adf_ret_p, *_ = adfuller(r_test_sample, autolag='AIC')
    except Exception:
        adf_ret_stat, adf_ret_p = np.nan, np.nan

    try:
        with np.errstate(all='ignore'):
            kpss_precio_stat, kpss_precio_p, *_ = kpss(precio_test_sample, regression='c', nlags='auto')
    except Exception:
        kpss_precio_stat, kpss_precio_p = np.nan, np.nan

    try:
        with np.errstate(all='ignore'):
            kpss_ret_stat, kpss_ret_p, *_ = kpss(r_test_sample, regression='c', nlags='auto')
    except Exception:
        kpss_ret_stat, kpss_ret_p = np.nan, np.nan

    def diagnostico_estacionariedad(adf_p, kpss_p):
        if np.isnan(adf_p) or np.isnan(kpss_p):
            return "N/A"
        adf_estacionaria  = adf_p < 0.05
        kpss_estacionaria = kpss_p > 0.05
        if adf_estacionaria and kpss_estacionaria:
            return "ESTACIONARIA (consenso)"
        elif not adf_estacionaria and not kpss_estacionaria:
            return "NO ESTACIONARIA (consenso)"
        else:
            return "AMBIGUA (tests discrepan)"

    veredicto_precio   = diagnostico_estacionariedad(adf_precio_p, kpss_precio_p)
    veredicto_retornos = diagnostico_estacionariedad(adf_ret_p, kpss_ret_p)
    # endregion
    # ==========================================================================
    # region PASO 18: Vida Media de Reversión (Half-Life OU)
    precio_log = np.log(df_ohlc['close'])
    delta_p    = precio_log.diff().dropna()
    p_lag      = precio_log.shift(1).dropna()

    idx_comun   = delta_p.index.intersection(p_lag.index)
    delta_p_hl  = delta_p.loc[idx_comun].values
    p_lag_hl    = p_lag.loc[idx_comun].values

    try:
        p_lag_mean   = p_lag_hl.mean()
        cov_hl       = np.mean((p_lag_hl - p_lag_mean) * (delta_p_hl - delta_p_hl.mean()))
        var_hl       = np.var(p_lag_hl)
        beta_hl      = cov_hl / var_hl if var_hl != 0 else 0.0

        if -1 < beta_hl < 0:
            half_life_velas = -np.log(2) / np.log(1 + beta_hl)
        else:
            half_life_velas = None

        if half_life_velas is not None and velas_por_dia is not None and velas_por_dia > 0:
            half_life_dias = half_life_velas / velas_por_dia
        else:
            half_life_dias = None
    except Exception:
        beta_hl = None
        half_life_velas = None
        half_life_dias = None
    # endregion
    # ==========================================================================
    # region PASO 19: Análisis de Volatilidad Relativa Inter-Temporal (NATR/ATR)
    # ==========================================================================

    TF_RULES = {
        '1min': '1min', '5min': '5min', '15min': '15min',
        '1h': '1h', '4h': '4h', '1d': '1D',
        '1w': '1W', '1mo': '1ME',
    }

    # Selección por minutos, no por nombre: el TF de la GUI puede venir como
    # '15m' (TF_LABELS) y no como la clave canónica '15min'.
    _base_tf_cfg = CONFIG['tf']
    _base_min_cfg = tf_to_minutes(_base_tf_cfg) or 0
    if _base_min_cfg <= 15:
        _general_pair = [('15min', '1h')]
    elif _base_min_cfg <= 60:
        _general_pair = [('1h', '4h')]
    elif _base_min_cfg <= 240:
        _general_pair = [('4h', '1d')]
    elif _base_min_cfg <= 1440:
        _general_pair = [('1d', '1w')]
    else:
        _general_pair = [('1w', '1mo')]

    HORIZON_PAIRS = {
        'General':      _general_pair,
        'Scalping':     [('1min', '5min'), ('5min', '15min')],
        'Daytrading':   [('15min', '1h'),  ('1h', '4h')],
        'Swingtrading': [('1h', '4h'),     ('4h', '1d')],
        'Position':     [('1d', '1w'),     ('1w', '1mo')],
    }

    MAX_LAGS = {
        'General': 20,
        'Scalping': 12,
        'Daytrading': 16,
        'Swingtrading': 20,
        'Position': 24,
    }

    _NATR_DATA = {}
    _NATR_CORR = None
    _NATR_PAIRS = {}

    def _tf_to_minutes(tf):
        return tf_to_minutes(tf)

    def _resample_ohlc(df, rule):
        return df.resample(rule).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()

    def _natr_key(tf):
        """Resuelve un TF de los pares (clave canónica, p.ej. '15min') a la clave
        real de _NATR_DATA con los mismos minutos (el TF base puede ser '15m')."""
        tf_min = _tf_to_minutes(tf)
        for k in _NATR_DATA:
            if _tf_to_minutes(k) == tf_min:
                return k
        return None

    def _calcular_atr_natr(df, period=14):
        d = df.copy()
        d['TR'] = np.maximum(
            d['high'] - d['low'],
            np.maximum(
                (d['high'] - d['close'].shift(1)).abs(),
                (d['low'] - d['close'].shift(1)).abs()
            )
        )
        d['ATR'] = d['TR'].rolling(window=period, min_periods=period).mean()
        d['NATR'] = d['ATR'] / d['close'] * 100
        return d

    base_tf = CONFIG['tf']
    base_min = _tf_to_minutes(base_tf)

    # 1. Compute base NATR
    try:
        df_natr = _calcular_atr_natr(df)
        _NATR_DATA[base_tf] = df_natr['NATR'].dropna()

        # 2. Determine which TFs to resample (all standard TFs >= base TF)
        all_tfs = ['1min', '5min', '15min', '1h', '4h', '1d', '1w', '1mo']
        _tf_min_used = {_tf_to_minutes(base_tf): base_tf}
        available_tfs = [base_tf]

        for tf in all_tfs:
            if tf == base_tf:
                continue
            tf_min = _tf_to_minutes(tf)
            if tf_min is None:
                continue
            # Skip if a TF with same minute value already exists (e.g. '1min' when base is '1m')
            if tf_min in _tf_min_used:
                continue
            if base_min and tf_min >= base_min:
                rule = TF_RULES.get(tf)
                if rule:
                    try:
                        df_resampled = _resample_ohlc(df, rule)
                        if len(df_resampled) >= 20:
                            df_res_natr = _calcular_atr_natr(df_resampled)
                            _NATR_DATA[tf] = df_res_natr['NATR'].dropna()
                            _tf_min_used[tf_min] = tf
                            available_tfs.append(tf)
                    except Exception:
                        pass

        available_tfs.sort(key=lambda x: _tf_to_minutes(x) or 0)

        # 3. Build correlation matrix
        if len(available_tfs) >= 2:
            natr_aligned = pd.DataFrame({tf: _NATR_DATA[tf] for tf in available_tfs}).dropna()
            if len(natr_aligned) > 1:
                _NATR_CORR = natr_aligned.corr(method='pearson')

        # 4. Compute pairs for ALL horizons
        for horizon_name, pairs in HORIZON_PAIRS.items():
            max_lag = MAX_LAGS.get(horizon_name, 20)
            horizon_results = []
            for tf_a, tf_b in pairs:
                tf_a, tf_b = _natr_key(tf_a), _natr_key(tf_b)
                if tf_a is None or tf_b is None:
                    continue
                both = pd.DataFrame({tf_a: _NATR_DATA[tf_a], tf_b: _NATR_DATA[tf_b]}).dropna()
                if len(both) < 20:
                    continue

                # Lead-lag cross-correlation (FFT: O(n log n) en vez de O(n²))
                ser_a = both[tf_a].values
                ser_b = both[tf_b].values
                mean_a, mean_b = ser_a.mean(), ser_b.mean()
                a_centered = ser_a - mean_a
                b_centered = ser_b - mean_b
                ccorr = fftconvolve(a_centered, b_centered[::-1], mode='full')
                denom = np.sqrt(np.sum((ser_a - mean_a)**2) * np.sum((ser_b - mean_b)**2))
                if denom > 0:
                    ccorr /= denom
                lags = np.arange(-len(ser_a) + 1, len(ser_a))
                pos_mask = (lags >= 0) & (lags <= max_lag)
                if pos_mask.any():
                    valid_ccorr = ccorr[pos_mask]
                    valid_lags = lags[pos_mask]
                    best_idx = np.argmax(valid_ccorr)
                    opt_lag = int(valid_lags[best_idx])
                    max_corr = float(valid_ccorr[best_idx])
                else:
                    opt_lag = 0
                    max_corr = 0.0

                horizon_results.append({
                    'pair': f'{tf_a}/{tf_b}',
                    'natr_base': float(_NATR_DATA[tf_a].mean()),
                    'natr_target': float(_NATR_DATA[tf_b].mean()),
                    'ratio': float((both[tf_a] / both[tf_b]).mean()),
                    'lag': opt_lag,
                    'lag_unit': tf_b,
                    'max_lag': max_lag,
                    'corr': max_corr,
                })
            _NATR_PAIRS[horizon_name] = horizon_results

    except Exception as e:
        print(f"  ⚠ PASO 19: Error en cálculo NATR: {e}")
    # endregion
    # ==========================================================================
    # region PASO 20: Z-score, Ratio por horizonte (NATR)
    HORIZON_NAMES = list(HORIZON_PAIRS.keys())
    WINDOW_ZSCORE_DAYS = {'General': 252, 'Scalping': 30, 'Daytrading': 90, 'Swingtrading': 180, 'Position': 252}
    _NATR_Z_SERIES = {h: {} for h in HORIZON_NAMES}
    _NATR_Z_CURRENT = {h: {} for h in HORIZON_NAMES}
    _NATR_RATIO_SERIES = {h: {} for h in HORIZON_NAMES}
    _NATR_RATIO_BB = {h: {} for h in HORIZON_NAMES}
    _NATR_THEORETICAL = {}
    if _NATR_DATA:
        tfs_sorted = sorted(_NATR_DATA.keys(), key=lambda x: _tf_to_minutes(x) or 0)
        base_tf_teo = tfs_sorted[0]
        base_natr = float(_NATR_DATA[base_tf_teo].mean())
        base_min = _tf_to_minutes(base_tf_teo) or 1
        for tf in tfs_sorted:
            tf_min = _tf_to_minutes(tf) or 1
            # La ley de escalado sqrt(T) implica que el NATR% CRECE con el
            # tamaño de vela (mas tiempo -> mayor rango relativo esperado),
            # no al reves: para tf_min > base_min el factor debe ser > 1.
            _NATR_THEORETICAL[tf] = base_natr * np.sqrt(tf_min / base_min)

    _base_min_z = _tf_to_minutes(CONFIG['tf'])
    # Densidad de velas por DIA DE MERCADO REAL (no de calendario): el pipeline
    # de limpieza (limpieza_datos_er.py) reindexa solo dentro del patron de
    # dias/horas realmente activos detectado en los datos — fines de semana u
    # horario fuera de sesion NUNCA generan filas (ni siquiera NaN), asi que
    # contar fechas unicas presentes en el indice ya excluye automaticamente
    # esos huecos sin necesidad de asumir 5 o 7 dias/semana. Para STOCK/FUTURO
    # esto da ~252-258 fechas/año (segun festivos reales de cada año en los
    # datos); para CRYPTO, que cotiza 7/7, da ~365 (coincide con contar dias
    # de calendario porque no hay huecos que excluir).
    _dias_mercado_unicos = pd.Index(df.index.date).nunique() if es_datetime_valido else None
    _velas_dia_base_trading = (len(df) / _dias_mercado_unicos) if _dias_mercado_unicos else None
    for horizon_name, pairs in HORIZON_PAIRS.items():
        window_dias_z = WINDOW_ZSCORE_DAYS.get(horizon_name, 252)
        for tf in _NATR_DATA:
            s = _NATR_DATA[tf].dropna()
            if len(s) < 30:
                continue
            # WINDOW_ZSCORE_DAYS esta en DIAS DE MERCADO, pero _NATR_DATA[tf]
            # son velas del TF resampleado (puede ser distinto al TF base,
            # p.ej. 1h cuando el TF base es 1min) — hay que convertir dias ->
            # velas de ESE tf especifico, no usar el numero de dias como si
            # fueran velas directamente (si no, "30d" para Scalping con datos
            # de 1min terminaba siendo literalmente 30 minutos).
            tf_min_z = _tf_to_minutes(tf) or _base_min_z
            velas_dia_tf = (_velas_dia_base_trading * _base_min_z / tf_min_z) if (_velas_dia_base_trading and _base_min_z and tf_min_z) else 1
            window_velas_z = max(1, int(round(window_dias_z * velas_dia_tf)))
            w = min(window_velas_z, len(s) // 2)
            s_ventana = s.iloc[-w:] if w > 0 else s
            mu, sigma = s_ventana.mean(), s_ventana.std()
            if sigma > 0:
                _NATR_Z_CURRENT[horizon_name][tf] = (s.iloc[-1] - mu) / sigma
                # Equivalente vectorizado de rolling(w).apply(lambda x: (x[-1]-x.mean())/x.std()):
                # rolling().apply() con una lambda en Python evalúa la función una vez por
                # cada una de las ~3M ventanas (llamada a nivel de intérprete por fila), lo
                # que en series de 1 minuto de varios años puede tardar minutos/horas y da
                # la sensación de que el script se ha quedado colgado. rolling().mean()/.std()
                # son operaciones vectorizadas en pandas y calculan lo mismo casi al instante.
                roll_mean = s.rolling(w, min_periods=w).mean()
                roll_std = s.rolling(w, min_periods=w).std()
                z_series = (s - roll_mean) / roll_std
                _NATR_Z_SERIES[horizon_name][tf] = z_series.where(roll_std > 0, 0.0)
            else:
                _NATR_Z_CURRENT[horizon_name][tf] = 0.0
                _NATR_Z_SERIES[horizon_name][tf] = pd.Series(0.0, index=s.index)
        for tf_a, tf_b in pairs:
            tf_a, tf_b = _natr_key(tf_a), _natr_key(tf_b)
            if tf_a is None or tf_b is None:
                continue
            both = pd.DataFrame({tf_a: _NATR_DATA[tf_a], tf_b: _NATR_DATA[tf_b]}).dropna()
            if len(both) < 30:
                continue
            ratio_s = (both[tf_a] / both[tf_b]).dropna()
            if len(ratio_s) < 30:
                continue
            _NATR_RATIO_SERIES[horizon_name][(tf_a, tf_b)] = ratio_s
            mu_r, sigma_r = ratio_s.mean(), ratio_s.std()
            _NATR_RATIO_BB[horizon_name][(tf_a, tf_b)] = {
                'mean': pd.Series(mu_r, index=ratio_s.index),
                'upper': pd.Series(mu_r + 2 * sigma_r, index=ratio_s.index),
                'lower': pd.Series(mu_r - 2 * sigma_r, index=ratio_s.index),
                'current': float(ratio_s.iloc[-1]),
            }
    # endregion
    # endregion (9. MÉTRICAS)
    # region ── 10. 📊 ESTRUCTURACIÓN Y PRESENTACIÓN DE MÉTRICAS FINANCIERAS EN TERMINAL
    # ── Determinaciones para análisis de dependencia ──
    def estado_adn(val, umbral):
        """Determina el estado táctico según la dependencia estadística."""
        if val > umbral:
            return "INERCIA"
        elif val < -umbral:
            return "REVERSIÓN"
        else:
            return "RUIDO"

    def _pacf1(stats):
        vals = stats.get('pacf_vals', [])
        return vals[1] if len(vals) > 1 else 0.0

    def _fmt_dependencia(stats):
        val = _pacf1(stats)
        return f"{val:.4f} → {estado_adn(val, stats['umbral'])}"

    # ── CONFIGURACIÓN DE ETIQUETAS DINÁMICAS (Para Ticks o Tiempo Fijo) ──────────
    if tipo_muestreo == 'tiempo_fijo':
        lbl_c, lbl_m, lbl_l = 'Diario', 'Semanal', 'Mensual'
    else:
        lbl_c, lbl_m, lbl_l = 'Bloque Corto', 'Bloque Medio', 'Bloque Largo'
    
    def barra(valor, total, ancho=12):
        if total is None or total <= 0: return '░' * ancho
        llenos = min(int(round(valor / total * ancho)), ancho)
        return '█' * llenos + '░' * (ancho - llenos)
    
    # Pre-cálculos para barras
    er_tend = int((df['ER'] > 0.5).sum())
    er_alet = int(((df['ER'] >= 0.3) & (df['ER'] <= 0.5)).sum())
    er_ruido = int((df['ER'] < 0.3).sum())
    er_total = er_tend + er_alet + er_ruido
    
    h_tend = int(total_tendencia_h)
    h_alet = int(total_aleatorio_h)
    h_rev = int(total_reversion_h)
    h_total = h_tend + h_alet + h_rev
    
    hv_vals = [v for v in [val_hv_7d, val_hv_30d, val_hv_90d, val_hv_365d] if v is not None]
    max_hv = max(hv_vals) if hv_vals else 1
    
    # Max reference for HV bars (relative to highest period)
    _hv_max = max(hv_vals) if hv_vals else 1
    
    # Max reference for session bars (relative to busiest session)
    _rel_hours = [v for v in [vol_relativa_asia, vol_relativa_london, vol_relativa_ny] if v > 0]
    _rel_h_max = max(_rel_hours) if _rel_hours else 1
    
    metricas = {}
    # metricas_pdf_por_horizonte guarda, para CADA ventana (no solo la
    # seleccionada al analizar), la version ya filtrada de cada categoria —
    # son las que se dibujan en las paginas estaticas "Metricas (N)" del PDF,
    # generando un juego completo de paginas por ventana (mismo patron que
    # ya usan "Precio por régimen ER" y "Dashboard NATR"), para que la GUI
    # pueda cambiar de Ventana sin re-analizar y sin que las métricas se
    # queden congeladas en la ventana que estaba seleccionada al analizar.
    # `metricas` (sin filtrar) sigue exportándose completo en el JSON para
    # el panel adaptativo de la GUI.
    metricas_pdf_por_horizonte = {h: {} for h in HORIZON_NAMES}

    def _filtrar_por_horizonte(datos, horizonte):
        prefijo = f'[{horizonte}]'
        sufijo = f'({horizonte})'
        display = {}
        tiene_horizonte = False
        for k, v in datos.items():
            if not k.startswith('['):
                display[k] = v
            elif k.startswith(prefijo):
                clean_k = k[len(prefijo):].lstrip()
                if clean_k.endswith(sufijo):
                    clean_k = clean_k[:-len(sufijo)].rstrip()
                display[clean_k] = v
                tiene_horizonte = True
        return display, tiene_horizonte

    def _mostrar_categoria(titulo, datos):
        metricas[titulo] = datos
        horizonte_sel = CONFIG.get('horizonte', 'General')
        for _hz in HORIZON_NAMES:
            display_hz, tiene_hz = _filtrar_por_horizonte(datos, _hz)
            titulo_hz = f'{titulo} — {_hz}' if (tiene_hz and _hz != 'General') else titulo
            metricas_pdf_por_horizonte[_hz][titulo_hz] = display_hz
            if _hz == horizonte_sel:
                # El log de terminal solo muestra la ventana seleccionada,
                # igual que antes.
                print(f"\n ▶ {titulo_hz}")
                print(f"  {'─' * (len(titulo_hz) + 2)}")
                for metrica, valor in display_hz.items():
                    if metrica.strip():
                        print(f"    {metrica:<40} : {valor}")
                sys.stdout.flush()

    _mostrar_categoria('1. Información General y tipo de muestreo', {
        'Periodo': f"{df.index.min()} → {df.index.max()}" if es_datetime_valido else f"{len(df):,} ticks (archivo sin temporalidad)",
        'Tipo de muestreo': tipo_muestreo,
        'Total velas': f"{len(df):,}"
    })

    _mostrar_categoria('2. Rendimiento y Retornos', {
        'Retorno anualizado (CAGR)': fmt_pct(ret_anual),
        f'Media retorno ({tf_actual})': f"{r_media*100:.6f}%",
        f'Mediana retorno ({tf_actual})': f"{r.median()*100:.6f}%",
        'Retorno diario promedio': f"{(ret_diario or 0)*100:.4f}%",
        'Retornos positivos': f"{(r > 0).sum() / r.count() * 100:.2f}%",
        'Retornos negativos': f"{(r < 0).sum() / r.count() * 100:.2f}%",
        'Alineación de marcos temporales': f"{barra(score_tendencia, 3)} {score_tendencia}/3"
    })

    _mostrar_categoria('3. Avanzada & Volatilidad Histórica', {
        'Volatilidad Histórica Total': f"{vol_historica_total*100:.2f}%",
        'HV 7d': f"{barra(val_hv_7d, _hv_max)} {val_hv_7d / vol_historica_total:.3f}x" if val_hv_7d is not None else "N/A",
        'HV 30d': f"{barra(val_hv_30d, _hv_max)} {val_hv_30d / vol_historica_total:.3f}x" if val_hv_30d is not None else "N/A",
        f'HV {dias_trimestre}d': f"{barra(val_hv_90d, _hv_max)} {val_hv_90d / vol_historica_total:.3f}x" if val_hv_90d is not None else "N/A",
        f'HV {dias_ano}d': f"{barra(val_hv_365d, _hv_max)} {val_hv_365d / vol_historica_total:.3f}x" if val_hv_365d is not None else "N/A",
        'Desviación Escalado Fractal': f"{desviacion_escalado:.4%}",
    })

    _mostrar_categoria('4. Riesgo y Volatilidad', {
        'Volatilidad anualizada': fmt_pct(vol_anual),
        'Volatilidad diaria': fmt_pct(vol_diaria),
        'Desv. estandar': f"{r_std*100:.6f}%",
        f'Ratio Sharpe (Rf=T-Bill 3m) [{rf_anual:.2%}]': (f"{'\033[92m' + f'{sharpe:.4f}' + '\033[0m'}" if sharpe and sharpe > 1
                                                           else f"{'\033[93m' + f'{sharpe:.4f}' + '\033[0m'}" if sharpe and sharpe > 0.5
                                                           else f"{'\033[91m' + f'{sharpe:.4f}' + '\033[0m'}" if sharpe is not None
                                                           else "N/A"),
        'Calmar Ratio': (f"{'\033[92m' + f'{calmar_ratio:.4f}' + '\033[0m'}" if calmar_ratio and calmar_ratio > 1
                         else f"{'\033[93m' + f'{calmar_ratio:.4f}' + '\033[0m'}" if calmar_ratio and calmar_ratio > 0.5
                         else f"{'\033[91m' + f'{calmar_ratio:.4f}' + '\033[0m'}" if calmar_ratio is not None
                         else "N/A (TICKS)"),
        'Sortino Ratio': (f"{'\033[92m' + f'{sortino:.4f}' + '\033[0m'}" if sortino and sortino > 1
                          else f"{'\033[93m' + f'{sortino:.4f}' + '\033[0m'}" if sortino and sortino > 0.5
                          else f"{'\033[91m' + f'{sortino:.4f}' + '\033[0m'}" if sortino is not None
                          else "N/A"),
    })

    _mostrar_categoria('5. Drawdown Analysis', {
        'Max Drawdown Histórico': f"{mdd*100:.2f}%",
        'Max Drawdown Interno (Semanal)': f"{stats_temporales['semanal']['max_dd_interno']*100:.2f}%",
        'Max Drawdown Interno (Diario)': f"{stats_temporales['dia']['max_dd_interno']*100:.2f}%  ({stats_temporales['dia']['fecha_peor']})",
        'Drawdown medio': f"{drawdown_medio*100:.2f}%" if drawdown_medio is not None else "N/A",
        'Episodios de drawdown': f"{num_episodios_dd:,}",
        'Tiempo recuperación (velas)': f"{recovery_velas_max:,}",
        'Tiempo recuperación (real)': recovery_str
    })

    _mostrar_categoria('6. VaR y Riesgo del Activo', {
        lbl_var_95: val_var_95,
        lbl_var_99: val_var_99,
        lbl_cvar_95: val_cvar_95,
        lbl_cvar_99: val_cvar_99,
        f'Peor caída en {tf_actual} (Mínimo)': f"{r.min()*100:.4f}%",
        f'Mayor subida en {tf_actual} (Máximo)': f"{r.max()*100:.4f}%",
        'Skewness': f"{r.skew():.4f}",
        'Kurtosis': f"{r.kurtosis():.4f}",
        'Jarque-Bera stat': f"{stat_jb:.2f}",
        'Jarque-Bera p-value': f"{p_jb:.6f}",
        f'Jarque-Bera p-value ({lbl_c})': f"{stats_temporales['dia']['p_value_normalidad']:.6f}",
        'Distribucion normal': f"{'NO (fat tails)' if p_jb < 0.05 else 'SI'}"
    })

    cat7 = {
        'ER medio': f"{er_medio:.4f}",
        'ER maximo': f"{df['ER'].max():.4f}",
        'ER minimo': f"{df['ER'].min():.4f}",
        'Periodos tendencia (ER>0.5)': f"{barra(er_tend, er_total)} {er_tend:,}",
        'Paseo aleatorio (ER 0.3-0.5)': f"{barra(er_alet, er_total)} {er_alet:,} (Random Walk)",
        'Periodos ruido (ER<0.3)': f"{barra(er_ruido, er_total)} {er_ruido:,}",
        ' ': '',
        'Hurst medio': f"{df['hurst'].mean():.4f}",
        'Hurst maximo': f"{df['hurst'].max():.4f}",
        'Hurst minimo': f"{df['hurst'].min():.4f}",
        'Periodos tendencia (H>0.58)': f"{barra(h_tend, h_total)} {h_tend:,}",
        'Paseo aleatorio (H 0.52-0.58)': f"{barra(h_alet, h_total)} {h_alet:,}",
        'Periodos mean reversion (H<0.52)': f"{barra(h_rev, h_total)} {h_rev:,}",
        ' ': '',
        '% Tiempo en tendencia pura': f"{pct_tiempo_tend:.2%}",
        'Retorno diario promedio en tendencia': f"{ret_dia_tend:.4f}%" if ret_dia_tend is not None else "N/A",
        'Volatilidad diaria en tendencia': f"{vol_dia_tend:.4f}%" if vol_dia_tend is not None else "N/A",
        'Sharpe en tendencia': fmt_num(sharpe_tend),
        'Duración media racha tendencial': f"{duracion_media_racha_tend:.1f} velas" if duracion_media_racha_tend is not None else "N/A",
        'Duración máxima racha tendencial': f"{duracion_max_racha_tend:.0f} velas" if duracion_max_racha_tend is not None else "N/A",
        ' ': '',
        '% Tiempo en mean reversion': f"{pct_tiempo_rev:.2%}",
        'Retorno diario promedio en mean rev.': f"{ret_dia_rev:.4f}%" if ret_dia_rev is not None else "N/A",
        'Volatilidad diaria en mean reversion': f"{vol_dia_rev:.4f}%" if vol_dia_rev is not None else "N/A",
        'Sharpe en mean reversion': fmt_num(sharpe_rev),
        'Mejora Sharpe (Tend vs Rev)': f"{(sharpe_tend - sharpe_rev):.4f}" if (sharpe_tend is not None and sharpe_rev is not None) else "N/A"
    }

    # ── Sustitución por Ventana (PASO 10) ──
    # Mismos nombres de fila que las generales: la GUI y el terminal sustituyen
    # el valor en su sitio cuando la ventana está seleccionada (y añaden la
    # ventana al título de la categoría). Las filas sin equivalente por ventana
    # (rachas, Sharpe por régimen...) conservan el valor general.
    for _h in HORIZON_ER_KAMA:
        eh = _ER_H.get(_h)
        if not eh:
            continue
        kh = _KAMA_H.get(_h)
        hh = _HURST_H.get(_h)
        _p, _s = f'[{_h}] ', f' ({_h})'
        er_s = eh['serie']
        er_tend_h  = int((er_s > 0.5).sum())
        er_alet_h  = int(((er_s >= 0.3) & (er_s <= 0.5)).sum())
        er_ruido_h = int((er_s < 0.3).sum())
        er_total_h = max(1, er_tend_h + er_alet_h + er_ruido_h)
        cat7[_p + 'ER medio' + _s]  = f"{er_s.mean():.4f}"
        cat7[_p + 'ER maximo' + _s] = f"{er_s.max():.4f}"
        cat7[_p + 'ER minimo' + _s] = f"{er_s.min():.4f}"
        cat7[_p + 'Periodos tendencia (ER>0.5)' + _s]  = f"{barra(er_tend_h, er_total_h)} {er_tend_h:,}"
        cat7[_p + 'Paseo aleatorio (ER 0.3-0.5)' + _s] = f"{barra(er_alet_h, er_total_h)} {er_alet_h:,} (Random Walk)"
        cat7[_p + 'Periodos ruido (ER<0.3)' + _s]      = f"{barra(er_ruido_h, er_total_h)} {er_ruido_h:,}"
        if kh:
            cat7[_p + 'Periodo ER' + _s]  = f"{eh['periodo']} velas — KAMA rápida {kh['fast']} / lenta {kh['slow']}"
            cat7[_p + 'KAMA actual' + _s] = f"{kh['actual']:.4f}"
            cat7[_p + 'Señal KAMA' + _s]  = kh['senal']
        if hh:
            hs = hh['serie']
            reg = hh['regimen']
            h_total_h = max(1, len(hs))
            cat7[_p + 'Hurst medio' + _s]  = f"{hs.mean():.4f}"
            cat7[_p + 'Hurst maximo' + _s] = f"{hs.max():.4f}"
            cat7[_p + 'Hurst minimo' + _s] = f"{hs.min():.4f}"
            cat7[_p + 'Periodos tendencia (H>0.58)' + _s]     = f"{barra(reg['total_tendencia'], h_total_h)} {reg['total_tendencia']:,}"
            cat7[_p + 'Paseo aleatorio (H 0.52-0.58)' + _s]   = f"{barra(reg['total_aleatorio'], h_total_h)} {reg['total_aleatorio']:,}"
            cat7[_p + 'Periodos mean reversion (H<0.52)' + _s] = f"{barra(reg['total_reversion'], h_total_h)} {reg['total_reversion']:,}"
            cat7[_p + '% Tiempo en tendencia pura' + _s]  = f"{float((hs > 0.6).mean()):.2%}"
            cat7[_p + '% Tiempo en mean reversion' + _s]  = f"{float((hs < 0.52).mean()):.2%}"
            cat7[_p + 'Ventana Hurst' + _s] = f"{hh['ventana']} velas"

    _mostrar_categoria('7. Ratio Eficiencia (ER) y Exponente de Hurst', cat7)

    def _color_signo(valor, decimales=4, sufijo=''):
        txt = f"{valor:.{decimales}f}{sufijo}"
        if valor > 0:
            return f"\033[92m{txt}\033[0m"
        if valor < 0:
            return f"\033[91m{txt}\033[0m"
        return f"\033[97m{txt}\033[0m"

    def _color_pct_tiempo_negativo(valor):
        txt = f"{valor:.2f}%"
        if valor > 50:
            return f"\033[91m{txt}\033[0m"
        if valor < 50:
            return f"\033[92m{txt}\033[0m"
        return f"\033[97m{txt}\033[0m"

    _mostrar_categoria('8. Análisis de Correlación y Estacionalidad', dict({
        'Contexto': f"Timeframe: {CONFIG['tf']} | Análisis Generalizado",
        'Ventana Volatilidad': f"{int(ventana_vol)} velas (~1 día)",
        'Ventana Correlación': f"{int(ventana_corr)} velas (~7 días)",
        'Correlación media Retorno-Vol': _color_signo(corr_media_historica),
        'Maxima correlación (FOMO)': _color_signo(corr_max),
        'Minima correlación (Panic)': _color_signo(corr_min),
        'Tiempo corr. negativa (%)': _color_pct_tiempo_negativo(tiempo_negativa),
    }, **({
        'Volatilidad promedio Tokio (08-17h local)': f"{barra(vol_relativa_asia, _rel_h_max)} {vol_relativa_asia:.2f}x" if vol_relativa_asia > 0 else f"{barra(0, 1)} N/A",
        'Volatilidad promedio Londres (08-17h local)': f"{barra(vol_relativa_london, _rel_h_max)} {vol_relativa_london:.2f}x" if vol_relativa_london > 0 else f"{barra(0, 1)} N/A",
        'Volatilidad promedio NY (08-17h local)': f"{barra(vol_relativa_ny, _rel_h_max)} {vol_relativa_ny:.2f}x" if vol_relativa_ny > 0 else f"{barra(0, 1)} N/A",
        'Vol. fin de semana vs laborable': f"{vol_weekend_ratio:.1%}" if vol_weekend_ratio > 0 else "N/A"
    } if len(vol_por_hora) > 0 else {})))

    # Estas filas por horizonte se añaden DESPUES de _mostrar_categoria (que ya
    # registró y filtró la categoría 8 para todas las ventanas), así que hay
    # que replicar aquí el mismo filtrado para cada ventana en
    # metricas_pdf_por_horizonte (si no, esa ventana nunca vería estas filas
    # en su propia página de Métricas).
    for _hname, _r in _leverage_por_horizonte.items():
        _pfx = f'[{_hname}]'
        _campos_h = {
            f'{_pfx} Ventana Volatilidad ({_hname})': _r['vol_desc'],
            f'{_pfx} Ventana Correlación ({_hname})': _r['corr_desc'],
            f'{_pfx} Correlación media Retorno-Vol ({_hname})': _color_signo(_r['media']),
            f'{_pfx} Maxima correlación (FOMO) ({_hname})': _color_signo(_r['max']),
            f'{_pfx} Minima correlación (Panic) ({_hname})': _color_signo(_r['min']),
            f'{_pfx} Tiempo corr. negativa (%) ({_hname})': _color_pct_tiempo_negativo(_r['neg_pct']),
        }
        metricas['8. Análisis de Correlación y Estacionalidad'].update(_campos_h)
        _sufijo_cat8 = f'({_hname})'
        for _k, _v in _campos_h.items():
            _clean_k = _k[len(_pfx):].lstrip()
            if _clean_k.endswith(_sufijo_cat8):
                _clean_k = _clean_k[:-len(_sufijo_cat8)].rstrip()
            metricas_pdf_por_horizonte[_hname]['8. Análisis de Correlación y Estacionalidad'][_clean_k] = _v


    _mostrar_categoria('9. Análisis de dependencia — Autocorrelación (ACF) y Parcial (PACF)', {
        'Contexto': f"Análisis basado en PACF Lag 1",
        'Significancia (Umbral)': f"{stats_temporales['dia']['umbral']:.4f}",
        'Dependencia Diaria': _fmt_dependencia(stats_temporales['dia']),
        'Dependencia Semanal': _fmt_dependencia(stats_temporales['semanal']),
        'Dependencia Mensual': _fmt_dependencia(stats_temporales['mensual']),
        'Dependencia Trimestral': _fmt_dependencia(stats_temporales['trimestral']),
        'Memoria Estructural': "Fuerte (D)" if abs(_pacf1(stats_temporales['dia'])) > stats_temporales['dia']['umbral'] else "Débil/Ruido"
    })

    _mostrar_categoria('10. Volatility Clustering (Efecto ARCH)', {
        'Contexto': f"ACF retornos² lag-1 | Test Ljung-Box (H₀: no clustering)",
        'Clustering Lag-1': f"{clustering_lag1:.4f}" if not np.isnan(clustering_lag1) else "N/A",
        'Ljung-Box Q-stat': f"{clustering_lb_q:.2f}" if not np.isnan(clustering_lb_q) else "N/A",
        'Ljung-Box p-valor': f"{clustering_lb_p:.4f}" if not np.isnan(clustering_lb_p) else "N/A",
        'Clustering detectado': "SÍ" if clustering_presente else "NO",
        'Interpretación': ("La volatilidad se agrupa" if clustering_presente
                          else "No hay evidencia de agrupación de volatilidad")
    })

    print()
    print(f"\n{'═'*70}\n")

    _mostrar_categoria('11. Estimadores de Volatilidad OHLC', {
        'Contexto': "Parkinson / Garman-Klass / Rogers-Satchell vs Close-to-Close",
        'Vol. Close-to-Close (anual)': fmt_pct(ctc_vol_anual) if ctc_vol_anual is not None else "N/A",
        'Vol. Parkinson (anual)': fmt_pct(parkinson_vol_anual) if parkinson_vol_anual is not None else "N/A",
        'Vol. Garman-Klass (anual)': fmt_pct(gk_vol_anual) if gk_vol_anual is not None else "N/A",
        'Vol. Rogers-Satchell (anual)': fmt_pct(rs_vol_anual) if rs_vol_anual is not None else "N/A",
        ' ': '',
        'Eficiencia Parkinson vs CtC': f"{eficiencia_parkinson:.2f}x" if eficiencia_parkinson else "N/A",
        'Eficiencia Garman-Klass vs CtC': f"{eficiencia_gk:.2f}x" if eficiencia_gk else "N/A",
        'Eficiencia Rogers-Satchell vs CtC': f"{eficiencia_rs:.2f}x" if eficiencia_rs else "N/A",
        'Estimador recomendado': "Rogers-Satchell (robusto a drift)" if rs_vol_periodo > 0 else "N/A"
    })

    _mostrar_categoria('12. Test de Estacionariedad (ADF / KPSS)', {
        'Contexto': "ADF: H0=raiz unitaria | KPSS: H0=estacionaria",
        'ADF stat (Precio)': f"{adf_precio_stat:.4f}" if not np.isnan(adf_precio_stat) else "N/A",
        'ADF p-valor (Precio)': f"{adf_precio_p:.4f}" if not np.isnan(adf_precio_p) else "N/A",
        'KPSS stat (Precio)': f"{kpss_precio_stat:.4f}" if not np.isnan(kpss_precio_stat) else "N/A",
        'KPSS p-valor (Precio)': f"{kpss_precio_p:.4f}" if not np.isnan(kpss_precio_p) else "N/A",
        'Veredicto (Precio)': veredicto_precio,
        ' ': '',
        'ADF stat (Retornos)': f"{adf_ret_stat:.4f}" if not np.isnan(adf_ret_stat) else "N/A",
        'ADF p-valor (Retornos)': f"{adf_ret_p:.4f}" if not np.isnan(adf_ret_p) else "N/A",
        'KPSS stat (Retornos)': f"{kpss_ret_stat:.4f}" if not np.isnan(kpss_ret_stat) else "N/A",
        'KPSS p-valor (Retornos)': f"{kpss_ret_p:.4f}" if not np.isnan(kpss_ret_p) else "N/A",
        'Veredicto (Retornos)': veredicto_retornos
    })

    _mostrar_categoria('13. Vida Media de Reversión (Half-Life OU)', {
        'Contexto': "Modelo Ornstein-Uhlenbeck discreto sobre log(precio)",
        'Beta (velocidad de reversión)': f"{beta_hl:.6f}" if beta_hl is not None else "N/A",
        'Half-Life (velas)': f"{half_life_velas:.1f}" if half_life_velas is not None else "N/A (no reversiva)",
        'Half-Life (días)': f"{half_life_dias:.2f}" if half_life_dias is not None else "N/A",
        'Interpretación': (
            "Serie mean-reverting: usese holding ≈ half-life" if half_life_velas is not None and half_life_velas > 0
            else "Sin reversión detectada (random walk / tendencia)"
        )
    })

    # ── 14/14.5. NATR, Z-score, Ratio — tabla con bordes ──

    # Helper: dibuja una tabla con bordes ┌─┐└─┘├─┤│
    def _natr_grid(headers, rows, caption=''):
        col_widths = [
            max(len(str(row[i])) for row in rows + [headers]) + 2
            for i in range(len(headers))
        ]
        sep = '├' + '┼'.join('─' * w for w in col_widths) + '┤'
        top = '┌' + '┬'.join('─' * w for w in col_widths) + '┐'
        bot = '└' + '┴'.join('─' * w for w in col_widths) + '┘'

        def _row(cells):
            return '│' + '│'.join(f'{str(c):^{w}}' for c, w in zip(cells, col_widths)) + '│'

        lines = [top, _row(headers), sep]
        for i, row in enumerate(rows):
            lines.append(_row(row))
            if i < len(rows) - 1:
                lines.append(sep)
        lines.append(bot)
        if caption:
            print(f"\n  {caption}")
        for line in lines:
            print(f"  {line}")
        sys.stdout.flush()

    for horizon_name in HORIZON_NAMES:
        label_h = f' [General]' if horizon_name == 'General' else f' [{horizon_name}]'
        titulo = f'14. NATR Multi-TF{label_h}'
        zs = _NATR_Z_CURRENT.get(horizon_name, {})
        pairs = HORIZON_PAIRS.get(horizon_name, [])
        pairs_data = _NATR_PAIRS.get(horizon_name, [])

        # ----- bloque A: tabla NATR por TF -----
        tfs = sorted(_NATR_DATA.keys(), key=lambda x: _tf_to_minutes(x) or 0)
        rows_a = []
        for tf in tfs:
            natr_val = _NATR_DATA[tf].mean()
            z_val = zs.get(tf, 0.0)
            teo = _NATR_THEORETICAL.get(tf, 0)
            dev = (natr_val / teo - 1) * 100 if teo > 0 else 0
            rows_a.append((
                f'{tf}',
                f'{natr_val:.4f}%',
                f'{z_val:+.2f}',
                f'{teo:.4f}%',
                f'{dev:+.1f}%',
            ))
        _natr_grid(
            ['TF', 'NATR', 'Z-score', 'Teórico √T', 'Desv.'],
            rows_a,
            caption=f'{titulo} — NATR por timeframe'
        )

        # ----- bloque B: lead-lag -----
        if pairs_data:
            rows_b = []
            for pr in pairs_data:
                rows_b.append((
                    pr['pair'],
                    f'{pr["natr_base"]:.4f}',
                    f'{pr["natr_target"]:.4f}',
                    f'{pr["ratio"]:.4f}',
                    f'~{pr["lag"]}v ({pr["lag_unit"]})',
                ))
            _natr_grid(
                ['Par', 'NATR base', 'NATR target', 'Ratio', 'Lead-Lag'],
                rows_b,
                caption=f'{titulo} — Pares de volatilidad'
            )

        # ----- bloque C: ratio con bandas -----
        for tf_a, tf_b in pairs:
            tf_a_k, tf_b_k = _natr_key(tf_a), _natr_key(tf_b)
            bb = _NATR_RATIO_BB.get(horizon_name, {}).get((tf_a_k, tf_b_k))
            if not bb:
                continue
            rows_c = [(
                f'{tf_a_k}/{tf_b_k}',
                f'{bb["current"]:.4f}',
                f'{bb["upper"].iloc[0]:.4f}',
                f'{bb["lower"].iloc[0]:.4f}',
            )]
            _natr_grid(
                ['Ratio', 'Actual', 'BB Sup (+2σ)', 'BB Inf (-2σ)'],
                rows_c,
                caption=f'{titulo} — Ratio con Bandas Bollinger'
            )

    # Poblar metricas para PDF (formato plano original, compatibilidad)
    natr_cat_pdf = {}
    for tf_name in sorted(_NATR_DATA.keys(), key=lambda x: _tf_to_minutes(x) or 0):
        natr_cat_pdf[f'NATR({tf_name})'] = f'{_NATR_DATA[tf_name].mean():.4f}'
    for horizon_name in HORIZON_NAMES:
        for pr in _NATR_PAIRS.get(horizon_name, []):
            prefix = f'[{horizon_name}] Par {pr["pair"]}'
            natr_cat_pdf[f'{prefix} - NATR base'] = f'{pr["natr_base"]:.4f}'
            natr_cat_pdf[f'{prefix} - NATR target'] = f'{pr["natr_target"]:.4f}'
            natr_cat_pdf[f'{prefix} - Ratio'] = f'{pr["ratio"]:.4f}'
            natr_cat_pdf[f'{prefix} - Lead-Lag'] = f'~{pr["lag"]} velas ({pr["lag_unit"]}) [max: {pr["max_lag"]} lags]'
    metricas['14. NATR, correlación Multi-TF'] = natr_cat_pdf

    natr_zr_pdf = {}
    for horizon_name in HORIZON_NAMES:
        pfx = f'[{horizon_name}]'
        for tf, zval in _NATR_Z_CURRENT.get(horizon_name, {}).items():
            natr_zr_pdf[f'{pfx}Z-score({tf})'] = f'{zval:+.3f}'
        for tf_a, tf_b in HORIZON_PAIRS.get(horizon_name, []):
            tf_a_k, tf_b_k = _natr_key(tf_a), _natr_key(tf_b)
            bb = _NATR_RATIO_BB.get(horizon_name, {}).get((tf_a_k, tf_b_k))
            if not bb:
                continue
            natr_zr_pdf[f'{pfx}Ratio({tf_a_k}/{tf_b_k})'] = f'{bb["current"]:.4f}'
            natr_zr_pdf[f'{pfx}Ratio BB sup({tf_a_k}/{tf_b_k})'] = f'{bb["upper"].iloc[0]:.4f}'
            natr_zr_pdf[f'{pfx}Ratio BB inf({tf_a_k}/{tf_b_k})'] = f'{bb["lower"].iloc[0]:.4f}'
    metricas['14.5. NATR Z-score, Ratio por horizonte'] = natr_zr_pdf

    print()
    print(f"\n{'═'*70}\n")

    # endregion
    # region ── 11. PREPARACIÓN PREVIA PDF ───────────────────────────────────────────────
    # Preparación máscaras de regimen
    mask_tend  = df['ER'] > 0.45
    mask_trans = (df['ER'] >= 0.30) & (df['ER'] <= 0.45)
    mask_ruido = df['ER'] < 0.30
    
    def color_regimen(er_val):
        if er_val > 0.45:   return '#1D9E75'
        elif er_val > 0.30: return '#888888'
        else:               return '#E24B4A'
    
    print("Generando PDF...")
    
    # ── CÁLCULO DE RENDIMIENTOS ESTACIONALES (Capitalización Compuesta) ──
    df['factor_crecimiento'] = np.exp(df['retorno'])
    
    orden_meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                   'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    orden_dias  = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    
    if es_datetime_valido:
        # Mapeo por número (independiente del locale del sistema)
        df['mes_nombre'] = df.index.month.map({
            1:'Enero', 2:'Febrero', 3:'Marzo', 4:'Abril', 5:'Mayo', 6:'Junio',
            7:'Julio', 8:'Agosto', 9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre'
        })
        df['dia_nombre'] = df.index.dayofweek.map({
            0:'Lunes', 1:'Martes', 2:'Miércoles', 3:'Jueves',
            4:'Viernes', 5:'Sábado', 6:'Domingo'
        })
    
        mes_stats = (df.groupby('mes_nombre')['factor_crecimiento'].prod() - 1) * 100
        dia_stats = (df.groupby('dia_nombre')['factor_crecimiento'].prod() - 1) * 100
        mes_stats = mes_stats.reindex(orden_meses)
        dia_stats = dia_stats.reindex(orden_dias)
    else:
        mes_stats = pd.Series(dtype=float)
        dia_stats = pd.Series(dtype=float)
    # endregion
    # region ── 12. REPRESENTACIONES VISUALES Y GRÁFICAS DE DATOS, GENERACIÓN DE PDF ─────
    # Las páginas de métricas son dinámicas: M = ceil(nº categorías / 6) páginas.
    # Las regiones siguientes se etiquetan como "PÁGINA M+n" (n = offset tras las métricas).
    
    # Se genera el PDF en memoria y se escribe a disco de una sola vez al final
    # (en vez de con una escritura incremental por cada página): así se evita
    # que el antivirus / Windows intercepte y trunque el archivo a mitad de la
    # generación, que es lo que produce PDFs "dañados, no se pueden reparar"
    # aunque el script termine sin ningún error.
    # ── Mapa de páginas del PDF ──
    # Cada página se registra en el momento de guardarse (los bloques van en
    # try/except y pueden fallar sin emitir página, así que el índice no puede
    # ser estático). horizonte=None → página general, visible siempre en la GUI;
    # horizonte='<Ventana>' → solo visible con esa Ventana seleccionada.
    # Para añadir una futura página por horizonte basta con llamar a
    # _registrar_pagina(titulo, horizonte=<Ventana>) justo tras su pdf.savefig.
    _PAGE_MAP = []
    _horizonte_sel = CONFIG.get('horizonte', 'General')
    def _registrar_pagina(titulo, horizonte=None):
        _PAGE_MAP.append({'pagina': len(_PAGE_MAP), 'titulo': titulo, 'horizonte': horizonte})
        # El log de terminal cuenta solo las páginas visibles con la Ventana
        # seleccionada (las de otros horizontes se generan en silencio: son la
        # caché que permite cambiar de ventana en la GUI sin re-analizar).
        if horizonte in (None, _horizonte_sel):
            n_visible = sum(1 for p in _PAGE_MAP if p['horizonte'] in (None, _horizonte_sel))
            print(f"Generado página {n_visible}/{TOTAL_PAGINAS_VISIBLES} — {titulo}")
            sys.stdout.flush()

    # ── Bundle de datos para los gráficos nativos de la GUI ──
    # La GUI dibuja las mismas figuras que este PDF, pero con matplotlib
    # embebido en Qt. Los datos de cada gráfico se calculan DENTRO de su
    # bloque de dibujo (decimados, agregados, con outliers recortados...), así
    # que en vez de recalcularlos aparte cada región va guardando en _PLOT el
    # array que acaba de plotear: la GUI dibuja exactamente lo mismo que el PDF
    # sin volver a pasar por el motor de análisis.
    #
    # Regla del formato: SOLO numpy / list / dict / str / float / int / bool,
    # nunca objetos pandas. El pickle lo lee otro proceso (la GUI) y no
    # queremos acoplar el formato del archivo a la versión de pandas instalada.
    # Las claves tupla (pares de TF) se serializan como "tf_a/tf_b".
    _EXPORTAR_PLOT = 'GUI_PLOTDATA_OUTPUT' in os.environ
    _PLOT = {
        '_version': 1,
        '_meta': {
            'nombre': CONFIG['nombre'],
            'tf': CONFIG['tf'],
            'activo': CONFIG['activo'],
            'archivo': os.path.basename(CONFIG['input_path']),
            'es_datetime_valido': bool(es_datetime_valido),
            'horizontes': list(HORIZON_NAMES),
        },
    }

    def _arr(x):
        """Serie/Index/array de pandas → np.ndarray desligado de pandas."""
        return np.asarray(getattr(x, 'values', x))

    def _eje(idx):
        """Eje X: DatetimeIndex → datetime64[ns]; cualquier otra cosa → array."""
        if es_datetime_valido and isinstance(idx, pd.DatetimeIndex):
            return np.asarray(idx, dtype='datetime64[ns]')
        return np.asarray(getattr(idx, 'values', idx))

    _pdf_buffer = io.BytesIO()
    with PdfPages(_pdf_buffer) as pdf:
    
    # region ── PÁGINAS 1..M — Métricas Estructuradas (MOTOR DINÁMICO MULTIPÁGINA) ──
        dy = 0.020

        # El reparto ya NO es "3 categorías fijas por columna": categorías
        # como "7. Ratio Eficiencia..." pueden tener muchas más filas que
        # otras (varía según la ventana), y con un reparto por CANTIDAD de
        # categorías una columna se llenaba de sobra mientras la siguiente
        # categoría quedaba empujada fuera de la página. Se empaqueta por
        # número REAL de filas por columna.
        MAX_FILAS_POR_COLUMNA = 36  # (0.83 - margen inferior) / dy, aprox.

        def _paginar_metricas(metricas_hz):
            def _filas_categoria(cat):
                return 1 + sum(1 for k in metricas_hz[cat] if str(k).strip())

            paginas_hz = []
            pagina_actual = {'izq': [], 'der': [], 'filas_izq': 0, 'filas_der': 0}
            for _cat in metricas_hz.keys():
                _filas = _filas_categoria(_cat)
                _col = 'izq' if pagina_actual['filas_izq'] <= pagina_actual['filas_der'] else 'der'
                _hay_contenido = pagina_actual['izq'] or pagina_actual['der']
                if _hay_contenido and pagina_actual[f'filas_{_col}'] + _filas > MAX_FILAS_POR_COLUMNA:
                    paginas_hz.append(pagina_actual)
                    pagina_actual = {'izq': [], 'der': [], 'filas_izq': 0, 'filas_der': 0}
                    _col = 'izq'
                pagina_actual[_col].append(_cat)
                pagina_actual[f'filas_{_col}'] += _filas
            if pagina_actual['izq'] or pagina_actual['der']:
                paginas_hz.append(pagina_actual)
            return paginas_hz

        # Paginacion pre-calculada para TODAS las ventanas (cada una puede
        # repartirse distinto, porque cada ventana añade sus propias filas a
        # categorías como "7." u "8."). Se generan y registran las páginas de
        # las 5 ventanas (mismo patrón que "Precio por régimen ER" y
        # "Dashboard NATR"): la GUI, al cambiar de Ventana, ya no necesita
        # re-analizar para ver las métricas actualizadas.
        paginas_por_horizonte = {hz: _paginar_metricas(metricas_pdf_por_horizonte[hz])
                                  for hz in HORIZON_NAMES}

        # Total de páginas VISIBLES con la ventana seleccionada (las que se
        # loguean): métricas + 9 generales sin etiqueta + 1 régimen ER + 1
        # dashboard NATR del horizonte. Internamente se generan más (una por
        # horizonte) como caché para el visor adaptativo.
        _er_vis = 1 if (_horizonte_sel == 'General' or _horizonte_sel in _ER_H) else 0
        _dash_vis = 1 if _horizonte_sel in HORIZON_NAMES else 0
        TOTAL_PAGINAS_VISIBLES = len(paginas_por_horizonte.get(_horizonte_sel, [])) + 9 + _er_vis + _dash_vis

        for _hz_pag in HORIZON_NAMES:
          metricas_hz = metricas_pdf_por_horizonte[_hz_pag]
          paginas_hz = paginas_por_horizonte[_hz_pag]
          for pag_num, _pagina in enumerate(paginas_hz, start=1):
            col_izq = _pagina['izq']
            col_der = _pagina['der']

            # Inicializar una nueva hoja en el PDF
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis('off')
            fig.patch.set_facecolor('#0f0f0f')

            # Encabezados principales
            texto_titulo = f"{CONFIG['nombre']} ({CONFIG['tf']}) — Analisis Descriptivo"
            if len(paginas_hz) > 1:
                texto_titulo += f" (Parte {pag_num - 1})"

            fig.text(0.5, 0.94, texto_titulo, ha='center', va='top', fontsize=16, fontweight='bold', color='white')
            fig.text(0.5, 0.89, f"Activo: {CONFIG['activo']} | Archivo: {os.path.basename(CONFIG['input_path'])}",
                     ha='center', va='top', fontsize=9, color='#888780')

            # --- RENDEREAR COLUMNA 1 (IZQUIERDA) ---
            y_current = 0.83
            for categoria in col_izq:
                fig.text(0.04, y_current, categoria, fontsize=10, color='#ff9900', fontweight='bold', va='center')
                y_current -= 0.022

                for idx, (k, v) in enumerate(metricas_hz[categoria].items()):
                    nombre_metrica = str(k).strip()
                    # Saltar separadores vacíos para no desperdiciar espacio
                    if nombre_metrica == '':
                        y_current -= dy * 0.5
                        continue

                    bg = '#161616' if idx % 2 == 0 else '#0d0d0d'
                    fig.patches.append(plt.Rectangle((0.02, y_current - 0.008), 0.46, dy,
                                                       transform=fig.transFigure, facecolor=bg, zorder=0))
                    key_fsize = 7.0 if len(nombre_metrica) > 33 else 8.5
                    fig.text(0.04, y_current, nombre_metrica, fontsize=key_fsize, color='#a0a0a0', va='center')
                    
                    # Valores por defecto para la columna izquierda
                    valor_str = re.sub(r'\033\[[0-9;]*m', '', str(v))
                    x_val = 0.33
                    # Encoger la fuente no basta para textos muy largos (ej.
                    # "Contexto"): igual invaden la columna derecha. Se
                    # truncan para que quepan siempre en el ancho disponible
                    # (el tamaño de fuente se decide ANTES de truncar, sobre
                    # la longitud original, para no acabar agrandando la
                    # fuente de un texto que ya era demasiado largo).
                    f_size = 5.5 if len(valor_str) > 45 else 7.0 if len(valor_str) > 30 else 9
                    if len(valor_str) > 45:
                        valor_str = valor_str[:42] + '…'
                    
                    # CORRECCIÓN: Match exacto para 'Periodo' y específico para la versión '(real)'
                    if nombre_metrica == 'Periodo':
                        valor_str = re.sub(r'\+\d{2}:\d{2}', '', valor_str)
                        x_val = 0.20
                        f_size = min(f_size, 7.0)
                    elif 'Tiempo recuperación (real)' in nombre_metrica:
                        x_val = 0.26
                        f_size = min(f_size, 6.5)
                    elif any(w in nombre_metrica for w in ['Periodo', 'Régimen', 'Fractal', 'Paseo', 'reversion']):
                        f_size = min(f_size, 8)
                        
                    fig.text(x_val, y_current, valor_str, fontsize=f_size, color='white', va='center', fontweight='bold')
                    y_current -= dy
                y_current -= 0.020  
    
            # --- RENDEREAR COLUMNA 2 (DERECHA) ---
            y_current = 0.83
            for categoria in col_der:
                fig.text(0.53, y_current, categoria, fontsize=10, color='#ff9900', fontweight='bold', va='center')
                y_current -= 0.022
                
                for idx, (k, v) in enumerate(metricas_hz[categoria].items()):
                    nombre_metrica = str(k).strip()
                    # Saltar separadores vacíos para no desperdiciar espacio
                    if nombre_metrica == '':
                        y_current -= dy * 0.5
                        continue

                    bg = '#161616' if idx % 2 == 0 else '#0d0d0d'
                    fig.patches.append(plt.Rectangle((0.52, y_current - 0.008), 0.46, dy,
                                                       transform=fig.transFigure, facecolor=bg, zorder=0))
                    key_fsize = 7.0 if len(nombre_metrica) > 33 else 8.5
                    fig.text(0.54, y_current, nombre_metrica, fontsize=key_fsize, color='#a0a0a0', va='center')
                    
                    # Valores por defecto para la columna derecha
                    valor_str = re.sub(r'\033\[[0-9;]*m', '', str(v))
                    x_val = 0.74
                    # Encoger la fuente no basta para textos muy largos (ej.
                    # "Contexto"): igual se salen de la pagina. Se truncan
                    # para que quepan siempre en el ancho disponible (el
                    # tamaño de fuente se decide ANTES de truncar).
                    f_size = 5.5 if len(valor_str) > 45 else 7.0 if len(valor_str) > 30 else 9
                    if len(valor_str) > 45:
                        valor_str = valor_str[:42] + '…'
                    
                    # CORRECCIÓN: Match exacto para 'Periodo' y específico para la versión '(real)'
                    if nombre_metrica == 'Periodo':
                        valor_str = re.sub(r'\+\d{2}:\d{2}', '', valor_str)
                        x_val = 0.61
                        f_size = min(f_size, 7.0)
                    elif 'Tiempo recuperación (real)' in nombre_metrica:
                        x_val = 0.70
                        f_size = min(f_size, 6.5)
                    elif any(w in nombre_metrica for w in ['real', 'Normalidad', 'p-value', 'Paseo', 'reversion']):
                        f_size = min(f_size, 7.5)
                        
                    fig.text(x_val, y_current, valor_str, fontsize=f_size, color='white', va='center', fontweight='bold')
                    y_current -= dy
                y_current -= 0.020  
    
            # Guardar la página actual
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close()

            _registrar_pagina(f'Métricas ({pag_num})', horizonte=_hz_pag)

    # endregion
    # region ── PÁGINA M+1 — Precio, Equity Curve y Underwater Drawdown ─────
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('#0f0f0f')
        gs  = gridspec.GridSpec(3, 1, hspace=0.50, height_ratios=[1.2, 1, 1])
        
        paso = 200
        df_plot = df.iloc[::paso]
        equity_plot = cum_returns.iloc[::paso]
        dd_plot = drawdown_series.iloc[::paso]
        eje_x_plot = df_plot.index if es_datetime_valido else np.arange(len(df_plot))
    
        ax1 = fig.add_subplot(gs[0])
        ax1.fill_between(eje_x_plot, df_plot['close'].min(), df_plot['close'],
                          color='#1D9E75', alpha=0.08, rasterized=True)
        ax1.plot(eje_x_plot, df_plot['close'], color='#1D9E75', linewidth=1.6, rasterized=True)
        if 'SMA_Regimen' in df_plot.columns:
            ax1.plot(eje_x_plot, df_plot['SMA_Regimen'], color='#d29922', linewidth=0.6, alpha=0.6, linestyle='--', rasterized=True)
            ax1.text(0.98, 0.98, 'SMA (200pp)', transform=ax1.transAxes, fontsize=8,
                     color='#d29922', ha='right', va='top',
                     bbox=dict(facecolor='#111111', alpha=0.7, edgecolor='#d29922', linewidth=0.5))
        ax1.set_facecolor('#111111')
        ax1.set_title(f"{CONFIG['nombre']} — Precio de Cierre ({CONFIG['tf']})",
                      color='white', fontsize=11)
        ax1.set_ylabel('Precio', color='#888780')
        ax1.tick_params(colors='#888780')
        ax1.grid(True, alpha=0.2, color='#444')
        for spine in ax1.spines.values(): spine.set_edgecolor('#333')

        ax2 = fig.add_subplot(gs[1])
        ax2.fill_between(eje_x_plot, 1, equity_plot, where=(equity_plot >= 1),
                          color='#1D9E75', alpha=0.15, step='pre', rasterized=True)
        ax2.fill_between(eje_x_plot, 1, equity_plot, where=(equity_plot < 1),
                          color='#E24B4A', alpha=0.15, step='pre', rasterized=True)
        ax2.plot(eje_x_plot, equity_plot, color='#58a6ff', linewidth=0.8, rasterized=True)
        ax2.axhline(1, color='#888780', linewidth=0.5, linestyle='--')
        ax2.set_facecolor('#111111')
        ax2.set_title(f"{CONFIG['nombre']} — Curva de Equity (Base 1.0)",
                      color='white', fontsize=11)
        ax2.set_ylabel('Capital (×)', color='#888780')
        ax2.tick_params(colors='#888780')
        ax2.grid(True, alpha=0.2, color='#444')
        for spine in ax2.spines.values(): spine.set_edgecolor('#333')

        ax3 = fig.add_subplot(gs[2])
        ax3.fill_between(eje_x_plot, 0, dd_plot * 100,
                          color='#E24B4A', alpha=0.4, step='pre', rasterized=True)
        ax3.plot(eje_x_plot, dd_plot * 100, color='#f85149', linewidth=0.5, rasterized=True)
        ax3.axhline(0, color='#888780', linewidth=0.5, linestyle='--')
        ax3.set_facecolor('#111111')
        ax3.set_title(f"{CONFIG['nombre']} — Underwater Drawdown (%)",
                      color='white', fontsize=11)
        ax3.set_ylabel('Drawdown %', color='#888780')
        ax3.set_xlabel('Tiempo' if es_datetime_valido else 'Nº de vela (TICKS)', color='#888780')
        ax3.tick_params(colors='#888780')
        ax3.grid(True, alpha=0.2, color='#444')
        for spine in ax3.spines.values(): spine.set_edgecolor('#333')
    
        if es_datetime_valido:
            for y in pd.date_range(df_plot.index[0], df_plot.index[-1], freq='YS'):
                ax1.axvline(y, color='#444', linewidth=0.4, alpha=0.4)
                ax2.axvline(y, color='#444', linewidth=0.4, alpha=0.4)
                ax3.axvline(y, color='#444', linewidth=0.4, alpha=0.4)

        if _EXPORTAR_PLOT:
            _PLOT['precio_equity'] = {
                'x': _eje(eje_x_plot),
                'close': _arr(df_plot['close']),
                'sma': (_arr(df_plot['SMA_Regimen'])
                        if 'SMA_Regimen' in df_plot.columns else None),
                'equity': _arr(equity_plot),
                'dd_pct': _arr(dd_plot) * 100,
            }

        pdf.savefig(fig, facecolor=fig.get_facecolor(), dpi=150)
        plt.close()
        _registrar_pagina('Precio, Equity y Underwater')
    
    # endregion
    # region ── PÁGINA M+2 — Análisis de Estacionalidad ────────
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('#0f0f0f')
    
        # ── TÍTULO PRINCIPAL ──
        fig.text(0.5, 0.97, f"{CONFIG['nombre']} ({CONFIG['tf']}) — Análisis de Estacionalidad",
                 ha='center', va='top', fontsize=16, fontweight='bold', color='white')
        fig.text(0.5, 0.93, f"Activo: {CONFIG['activo']} | Archivo: {os.path.basename(CONFIG['input_path'])}",
                 ha='center', va='top', fontsize=9, color='#888780')
    
        if es_datetime_valido and mes_stats.notna().any():
            gs = gridspec.GridSpec(2, 1, hspace=0.4)
    
            ax1 = fig.add_subplot(gs[0])
            mes_plot = mes_stats.fillna(0)
            cols_m = ['#1D9E75' if x >= 0 else '#E24B4A' for x in mes_plot]
            bars1 = ax1.bar(mes_plot.index, mes_plot, color=cols_m, alpha=0.8)
            ax1.set_title('Retorno Acumulado por Mes (%)', color='white', fontsize=12)
            ax1.tick_params(colors='#888780')
            ax1.grid(True, axis='y', alpha=0.2)
            ax1.set_facecolor('#111111')
            for spine in ax1.spines.values(): spine.set_edgecolor('#333')
            for bar in bars1:
                h = bar.get_height()
                if h != 0:
                    ax1.text(bar.get_x() + bar.get_width()/2, h,
                             f'{h:.1f}%', ha='center',
                             va='bottom' if h > 0 else 'top',
                             color='white', fontsize=8)
    
            ax2 = fig.add_subplot(gs[1])
            dia_plot = dia_stats.fillna(0)
            cols_d = ['#1D9E75' if x >= 0 else '#E24B4A' for x in dia_plot]
            bars2 = ax2.bar(dia_plot.index, dia_plot, color=cols_d, alpha=0.8)
            ax2.set_title('Retorno Acumulado por Día de la Semana (%)', color='white', fontsize=12)
            ax2.tick_params(colors='#888780')
            ax2.grid(True, axis='y', alpha=0.2)
            ax2.set_facecolor('#111111')
            for spine in ax2.spines.values(): spine.set_edgecolor('#333')
            for bar in bars2:
                h = bar.get_height()
                if h != 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, h,
                             f'{h:.1f}%', ha='center',
                             va='bottom' if h > 0 else 'top',
                             color='white', fontsize=8)

            if _EXPORTAR_PLOT:
                _PLOT['estacionalidad'] = {
                    'disponible': True,
                    'meses_labels': [str(i) for i in mes_plot.index],
                    'meses': _arr(mes_plot).astype(float),
                    'dias_labels': [str(i) for i in dia_plot.index],
                    'dias': _arr(dia_plot).astype(float),
                }
        else:
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.set_facecolor('#111111')
            ax.text(0.5, 0.5,
                    "Análisis de estacionalidad no disponible.\n\n"
                    "El archivo está basado en TICKS (sin temporalidad de calendario fija),\n"
                    "por lo que no es posible agrupar los retornos por mes o día de la semana.",
                    ha='center', va='center', fontsize=12, color='#888780',
                    transform=ax.transAxes)
            if _EXPORTAR_PLOT:
                _PLOT['estacionalidad'] = {'disponible': False}
    
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close()
        _registrar_pagina('Análisis de Estacionalidad')
        
    # endregion
    # region ── PÁGINA M+3 — Precio por Régimen ER (con KAMA, por Ventana) ────
        UMBRAL_TEND = 0.5
        UMBRAL_RUID = 0.3

        def color_regimen_dinamico(er_val):
            if er_val > UMBRAL_TEND:   return '#00d4aa'
            elif er_val > UMBRAL_RUID: return '#8b949e'
            else:                      return '#f85149'

        for _h in HORIZON_NAMES:
            eh = _ER_H.get(_h)
            if eh is None:
                continue
            try:
                er_serie_h = eh['serie']
                u_h = eh['umbrales']
                kama_h = _KAMA_H.get(_h)
                hh = _HURST_H.get(_h)

                fig = plt.figure(figsize=(11.69, 8.27))
                fig.patch.set_facecolor('#0d1117')
                gs = gridspec.GridSpec(2, 1, hspace=0.4, height_ratios=[3, 1])

                step_h = max(1, len(df) // 20000)
                df_sub_h = df.iloc[::step_h]
                er_sub_h = er_serie_h.iloc[::step_h]

                ax_p = fig.add_subplot(gs[0])
                ax_p.set_facecolor('#0d1117')
                titulo_h = (f"{CONFIG['activo']} — Precio por Régimen ER — {_h} "
                            f"(ER {eh['periodo']}, KAMA {kama_h['fast']}/{kama_h['slow']})"
                            if kama_h else
                            f"{CONFIG['activo']} — Precio por Régimen ER — {_h} (ER {eh['periodo']})")
                ax_p.set_title(titulo_h, color='#e6edf3', fontsize=11)
                ax_p.set_ylabel('Precio', color='#8b949e')
                ax_p.set_xlabel('Tiempo' if es_datetime_valido else 'Nº de vela (TICKS)', color='#8b949e')
                ax_p.tick_params(colors='#8b949e')
                ax_p.grid(True, alpha=0.12, color='#30363d')
                for spine in ax_p.spines.values():
                    spine.set_edgecolor('#21262d')

                # eje_x_h y x_kama DEBEN usar la misma representación
                # numérica del eje X: mezclar enteros int64 (ns) crudos para
                # el LineCollection con un DatetimeIndex nativo para el KAMA
                # confunde el conversor de fechas de matplotlib (desborda al
                # autoescalar, o dibuja el LineCollection fuera del rango
                # visible mientras el KAMA sí se ve — de ahí el "todo azul").
                # date2num() da a ambas líneas la misma unidad (días desde
                # una época de referencia), evitando el problema de raíz.
                eje_x_h = date2num(df_sub_h.index) if es_datetime_valido else np.arange(len(df_sub_h))
                points_h = np.array([eje_x_h, df_sub_h['close'].values]).T.reshape(-1, 1, 2)
                segments_h = np.concatenate([points_h[:-1], points_h[1:]], axis=1)
                # colors_h debe tener un elemento por SEGMENTO (N-1), no por
                # punto (N): con un color de más, LineCollection ignora el
                # array de colores por completo y cae al azul por defecto de
                # matplotlib para toda la línea — este era el otro bug real
                # detrás de "se ve todo azul".
                colors_h = [color_regimen_dinamico(er) for er in er_sub_h.values[:-1]]
                lc_h = LineCollection(segments_h, colors=colors_h, linewidth=0.8, alpha=0.95,
                                      zorder=2, rasterized=True)
                ax_p.add_collection(lc_h)
                ax_p.autoscale()
                if es_datetime_valido:
                    # eje_x_h son floats de date2num(), no un DatetimeIndex:
                    # matplotlib ya no los reconoce como fechas solo, hay que
                    # decirle explícitamente cómo formatear los ticks.
                    locator = AutoDateLocator()
                    ax_p.xaxis.set_major_locator(locator)
                    ax_p.xaxis.set_major_formatter(ConciseDateFormatter(locator))

                if kama_h is not None:
                    x_kama = eje_x_h if es_datetime_valido else np.arange(len(df_sub_h))
                    ax_p.plot(x_kama, kama_h['serie'].iloc[::step_h].values,
                              color='#58a6ff', linewidth=0.7, alpha=0.5,
                              zorder=1, rasterized=True)

                legend_h = [
                    Line2D([0], [0], color='#00d4aa', linewidth=2, label='Tendencia (ER>0.50)'),
                    Line2D([0], [0], color='#8b949e', linewidth=2, label='Transicion (0.30-0.50)'),
                    Line2D([0], [0], color='#f85149', linewidth=2, label='Ruido (ER<0.30)'),
                ]
                if kama_h is not None:
                    legend_h.append(Line2D([0], [0], color='#58a6ff', linewidth=2,
                                           label=f"KAMA {kama_h['fast']}/{kama_h['slow']}"))
                ax_p.legend(handles=legend_h, facecolor='#161b22', labelcolor='#e6edf3',
                            fontsize=8, loc='upper right')

                ax_e = fig.add_subplot(gs[1])
                sub_hurst = (f" · Hurst v{hh['ventana']}: {hh['serie'].mean():.3f}" if hh else "")
                ax_e.set_title(f"Histórico ER {_h} (periodo {eh['periodo']}){sub_hurst}",
                               color='#e6edf3', fontsize=10, pad=10)
                x_plot_h = df.index[::step_h] if es_datetime_valido else np.arange(len(df))[::step_h]
                er_plot_h = er_serie_h.values[::step_h]
                ax_e.fill_between(x_plot_h, er_plot_h, color='#d29922', alpha=0.15, linewidth=0, rasterized=True)
                ax_e.plot(x_plot_h, er_plot_h, color='#d29922', linewidth=0.3, alpha=0.5, rasterized=True)
                er_suav_h = er_serie_h.rolling(200).mean()
                ax_e.plot(x_plot_h, er_suav_h.iloc[::step_h].values, color='#e6edf3',
                          linewidth=1.0, label='Tendencia (SMA 200)', rasterized=True)
                ax_e.axhline(0.5, color='#00d4aa', linewidth=0.8, linestyle='--', label='Direccionalidad (0.5)')
                ax_e.axhline(0.3, color='#f85149', linewidth=0.8, linestyle='--', label='Alto Ruido (0.3)')
                ax_e.axhline(u_h['er_medio'], color='#58a6ff', linewidth=1.0, linestyle='-',
                             alpha=0.7, label=f"Media ({u_h['er_medio']:.2f})")
                ax_e.axhline(u_h['umbral_tendencia'], color='#58a6ff', linewidth=0.7, linestyle=':',
                             alpha=0.7, label=f"+1σ ({u_h['umbral_tendencia']:.2f})")
                ax_e.axhline(u_h['umbral_ruido'], color='#58a6ff', linewidth=0.7, linestyle=':',
                             alpha=0.7, label=f"-1σ ({u_h['umbral_ruido']:.2f})")
                ax_e.set_facecolor('#0d1117')
                ax_e.set_ylabel('ER', color='#8b949e')
                ax_e.set_ylim(0, 1)
                ax_e.tick_params(colors='#8b949e')
                ax_e.grid(True, alpha=0.12, color='#30363d')
                ax_e.legend(loc='upper right', facecolor='#161b22', labelcolor='#e6edf3', fontsize=7, ncol=2)
                for spine in ax_e.spines.values():
                    spine.set_edgecolor('#21262d')

                if _EXPORTAR_PLOT:
                    _PLOT.setdefault('regimen_er', {})[_h] = {
                        # panel superior: precio coloreado por régimen ER
                        'x': (_eje(df_sub_h.index) if es_datetime_valido
                              else np.arange(len(df_sub_h))),
                        'close': _arr(df_sub_h['close']).astype(float),
                        'er': _arr(er_sub_h).astype(float),
                        'kama': (_arr(kama_h['serie'].iloc[::step_h]).astype(float)
                                 if kama_h is not None else None),
                        'kama_fast': (kama_h['fast'] if kama_h is not None else None),
                        'kama_slow': (kama_h['slow'] if kama_h is not None else None),
                        # panel inferior: histórico ER + SMA 200
                        'x_er': (_eje(df.index[::step_h]) if es_datetime_valido
                                 else np.arange(len(df))[::step_h]),
                        'er_hist': np.asarray(er_plot_h, dtype=float),
                        'er_sma': _arr(er_suav_h.iloc[::step_h]).astype(float),
                        'periodo': eh['periodo'],
                        'umbrales': {k: float(v) for k, v in u_h.items()},
                        'hurst_ventana': (hh['ventana'] if hh else None),
                        'hurst_medio': (float(hh['serie'].mean()) if hh else None),
                    }

                pdf.savefig(fig, facecolor=fig.get_facecolor(), dpi=150)
                plt.close()
                _registrar_pagina(f'Precio por régimen ER — {_h}', horizonte=_h)
            except Exception as e:
                print(f"ERROR EN Página régimen ER {_h}: {e}")
                traceback.print_exc()
                plt.close()
    # endregion
    # region ── PÁGINA M+4 — Riesgo Diario/Anual + QQ-Plot ────────
        try:
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.patch.set_facecolor('#0f0f0f')
            gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.8], hspace=0.45)
    
            sigmas = [1, 2, 3]
            colores_sigmas = ['#1D9E75', '#BA7517', '#E24B4A']
    
            def dibujar_campana_p5(ax, media, std, titulo, es_anual=False):
                ax.set_facecolor('#111111')
                x = np.linspace(media - 4*std, media + 4*std, 200)
                y = stats.norm.pdf(x, media, std)
                ax.plot(x, y, color='#185FA5', linewidth=1.8, label='Distribución Proyectada', rasterized=True)
    
                ax.axvline(media, color='white', linestyle='--', linewidth=1, alpha=0.5)
                ax.text(media, -max(y) * 0.06, f"μ: {media:.2%}", color='white', fontsize=6.5,
                        ha='center', fontweight='bold', bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=1))
    
                for s in sigmas:
                    ax.fill_between(x, y, where=(x >= media-s*std) & (x <= media+s*std),
                                    color=colores_sigmas[s-1], alpha=0.12, rasterized=True)
                    for lado in [-1, 1]:
                        val = media + (lado * s * std)
                        ax.axvline(val, color=colores_sigmas[s-1], linestyle=':', alpha=0.4)
                        
                        if not es_anual:
                            prob = np.mean(r_diario_real <= val) if lado == -1 else np.mean(r_diario_real >= val)
                            txt = f"1 c/{1/prob:.1f}d" if prob > 0 else "No reg."
                        else:
                            factor = 365 if CONFIG['activo'] == 'CRYPTO' else 252
                            umbral = val / np.sqrt(factor)
                            prob = np.mean(r_diario_real <= umbral) if lado == -1 else np.mean(r_diario_real >= umbral)
                            txt = f"1 c/{(1/prob)/factor:.1f}añ" if prob > 0 else "No reg."
                        
                        etiqueta = f"{lado*s}σ: {val:.2%}\n({txt})"
                        ax.text(val, -max(y) * (0.05 if s % 2 != 0 else 0.14), etiqueta,
                                color=colores_sigmas[s-1], fontsize=5.5, ha='center', fontweight='bold',
                                bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=0.5))
    
                ax.set_ylim(-max(y)*0.24, max(y)*1.15)
                ax.set_title(titulo, color='white', fontsize=10, pad=6)
                ax.tick_params(colors='#888780', labelsize=6)
                ax.grid(True, alpha=0.03)
                ax.legend(facecolor='#222', labelcolor='white', fontsize=6, loc='upper right')
                for spine in ax.spines.values(): spine.set_edgecolor('#333')
    
            ax1 = fig.add_subplot(gs[0])
            dibujar_campana_p5(ax1, ret_diario, vol_diaria, "Retorno Diario (Proyección)", es_anual=False)
    
            ax2 = fig.add_subplot(gs[1])
            dibujar_campana_p5(ax2, ret_anual, vol_anual, "Retorno Anual (Proyección)", es_anual=True)
    
            ax3 = fig.add_subplot(gs[2])
            ax3.set_facecolor('#111111')
            (osm, osr), (slope, intercept, r_sq) = stats.probplot(r_clean, dist="norm")
            ax3.scatter(osm, osr, color='#58a6ff', s=8, alpha=0.4, rasterized=True)
            ax3.plot(osm, slope * osm + intercept, color='#E24B4A', linewidth=1.2, linestyle='--')
            ax3.set_title(f"QQ-Plot: Retornos {CONFIG['tf']} vs Normal", color='white', fontsize=10, pad=6)
            ax3.set_xlabel('Cuantiles teóricos', color='#888780', fontsize=7)
            ax3.set_ylabel('Cuantiles observados', color='#888780', fontsize=7)
            ax3.tick_params(colors='#888780', labelsize=6)
            ax3.grid(True, alpha=0.15, color='#444')
            for spine in ax3.spines.values(): spine.set_edgecolor('#333')
            ax3.text(0.98, 0.05, f"R² ajuste: {r_sq:.4f}", transform=ax3.transAxes, ha='right', va='bottom',
                     fontsize=7, color='#888780', fontweight='bold')

            if _EXPORTAR_PLOT:
                # Las etiquetas "1 c/Xd" de cada sigma salen de la frecuencia
                # empírica en r_diario_real. Se precalculan aquí (son 6 puntos)
                # en vez de exportar la serie entera y repetir la lógica de
                # es_anual en la GUI.
                _factor_anual = 365 if CONFIG['activo'] == 'CRYPTO' else 252

                def _probs_campana(media, std, es_anual):
                    out = {}
                    for s in sigmas:
                        for lado in (-1, 1):
                            val = media + (lado * s * std)
                            ref = val / np.sqrt(_factor_anual) if es_anual else val
                            prob = (np.mean(r_diario_real <= ref) if lado == -1
                                    else np.mean(r_diario_real >= ref))
                            out[str(lado * s)] = {'val': float(val), 'prob': float(prob)}
                    return out

                _PLOT['riesgo_dia_anual'] = {
                    'diario': {'media': float(ret_diario), 'std': float(vol_diaria),
                               'sigmas': _probs_campana(ret_diario, vol_diaria, False)},
                    'anual': {'media': float(ret_anual), 'std': float(vol_anual),
                              'sigmas': _probs_campana(ret_anual, vol_anual, True),
                              'factor_anual': int(_factor_anual)},
                    'qq': {'osm': np.asarray(osm, dtype=float),
                           'osr': np.asarray(osr, dtype=float),
                           'slope': float(slope), 'intercept': float(intercept),
                           'r_sq': float(r_sq)},
                }

            pdf.savefig(fig, facecolor=fig.get_facecolor(), dpi=150)
            plt.close()
            _registrar_pagina('Riesgo Diario/Anual + QQ-Plot')
        except Exception as e:
            print(f"ERROR EN PÁGINA M+4 (Riesgo Diario/Anual): {e}")
            plt.close()
    
    # endregion
    # region ── PÁGINA M+5 — Riesgo Intradiario + Rolling VaR ────────
        try:
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.patch.set_facecolor('#0f0f0f')
            gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.8], hspace=0.45)
    
            sigmas = [1, 2, 3]
            colores_sigmas = ['#1D9E75', '#BA7517', '#E24B4A']
            jb_stat_p6, jb_p_val_p6 = stats.jarque_bera(r)
            es_normal_p6 = jb_p_val_p6 > 0.05
    
            def dibujar_campana_micro(ax, media, std, titulo, tipo_grafico='tf_puro'):
                ax.set_facecolor('#111111')
                x = np.linspace(media - 4*std, media + 4*std, 200)
                y = stats.norm.pdf(x, media, std)
                ax.plot(x, y, color='#185FA5', linewidth=1.8, label='Distribución Proyectada', rasterized=True)
    
                if tipo_grafico == 'tf_puro':
                    for s in sigmas:
                        ax.fill_between(x, y, where=(x >= media-s*std) & (x <= media+s*std),
                                        color=colores_sigmas[s-1], alpha=0.12, rasterized=True)
                        for lado in [-1, 1]:
                            val = media + (lado * s * std)
                            ax.axvline(val, color=colores_sigmas[s-1], linestyle=':', alpha=0.4)
                            prob_evento = np.mean(r <= val) if lado == -1 else np.mean(r >= val)
                            texto_frecuencia = f"1 c/{1/prob_evento:.1f}vel" if prob_evento > 0 else "No reg."
                            etiqueta = f"{lado*s}σ: {val:.2%}\n({texto_frecuencia})"
                            ax.text(val, -max(y) * (0.05 if s % 2 != 0 else 0.14), etiqueta,
                                    color=colores_sigmas[s-1], fontsize=5.5, ha='center', fontweight='bold',
                                    bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=0.5))
                else:
                    niveles_var  = [0.95, 0.99]
                    colores_var  = ['#BA7517', '#E24B4A']
                    z_scores_var = [1.645, 2.326]
                    for idx, (conf, z) in enumerate(zip(niveles_var, z_scores_var)):
                        val_var = media - (z * std) if es_normal_p6 else np.percentile(r, (1-conf)*100)
                        ax.fill_between(x, y, where=(x <= val_var), color=colores_var[idx], alpha=0.18, rasterized=True)
                        ax.axvline(val_var, color=colores_var[idx], linestyle='-', linewidth=1.3, alpha=0.6)
                        prob_evento = np.mean(r <= val_var)
                        texto_frecuencia = f"1 c/{1/prob_evento:.1f}vel" if prob_evento > 0 else "No reg."
                        etiqueta = f"VaR ({int(conf*100)}%): {val_var:.2%}\n({texto_frecuencia})"
                        ax.text(val_var, -max(y) * (0.06 if idx == 0 else 0.16), etiqueta,
                                color=colores_var[idx], fontsize=5.5, ha='center', fontweight='bold',
                                bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=0.5))
                    # CVaR (Expected Shortfall): la media de la cola más allá
                    # del VaR. Paramétrico si la serie pasa normalidad
                    # (media - std·φ(z)/Φ(z)), histórico si no (media de los
                    # retornos <= al VaR). Se dibuja en línea discontinua, bajo
                    # las etiquetas del VaR.
                    for idx, (conf, z) in enumerate(zip(niveles_var, z_scores_var)):
                        val_var = media - (z * std) if es_normal_p6 else np.percentile(r, (1-conf)*100)
                        if es_normal_p6:
                            val_cvar = media - std * (stats.norm.pdf(z) / conf)
                        else:
                            cola = r[r <= val_var]
                            val_cvar = float(cola.mean()) if len(cola) else float(val_var)
                        prob_cola = np.mean(r <= val_cvar)
                        texto_freq_c = f"1 c/{1/prob_cola:.1f}vel" if prob_cola > 0 else "No reg."
                        etiqueta_c = f"CVaR ({int(conf*100)}%): {val_cvar:.2%}\n({texto_freq_c})"
                        ax.axvline(val_cvar, color=colores_var[idx], linestyle='--', linewidth=1.5, alpha=0.75)
                        ax.text(val_cvar, -max(y) * (0.30 if idx == 0 else 0.42), etiqueta_c,
                                color=colores_var[idx], fontsize=5.5, ha='center', fontweight='bold',
                                bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=0.5))
    
                ax.axvline(media, color='white', linestyle='--', linewidth=1, alpha=0.4)
                # el límite inferior se abre cuando la campana lleva CVaR (las
                # etiquetas de la cola cuelgan por debajo de las del VaR)
                if tipo_grafico == 'var':
                    ax.set_ylim(-max(y)*0.50, max(y)*1.15)
                else:
                    ax.set_ylim(-max(y)*0.24, max(y)*1.15)
                ax.set_title(titulo, color='white', fontsize=10, pad=6)
                ax.tick_params(colors='#888780', labelsize=6)
                ax.grid(True, alpha=0.03)
                ax.legend(facecolor='#222', labelcolor='white', fontsize=6, loc='upper right')
                for spine in ax.spines.values(): spine.set_edgecolor('#333')
    
            ax1 = fig.add_subplot(gs[0])
            dibujar_campana_micro(ax1, r_media, r_std,
                                  f"Retorno por Vela {CONFIG['tf']}", tipo_grafico='tf_puro')

            ax2 = fig.add_subplot(gs[1])
            dibujar_campana_micro(ax2, r_media, r_std,
                                  f"VaR {'Paramétrico' if es_normal_p6 else 'Histórico'} ({CONFIG['tf']})",
                                  tipo_grafico='var')
    
            ax3 = fig.add_subplot(gs[2])
            ax3.set_facecolor('#111111')
            rolling_window = max(20, int(velas_por_dia * 20)) if velas_por_dia is not None else 100
            # Ventana expresada en velas (varía mucho según el TF), pero la
            # intención es siempre "20 días" de histórico — se muestra la
            # equivalencia en días para que sea legible sin tener que hacer
            # la cuenta mentalmente.
            dias_ventana = rolling_window / velas_por_dia if velas_por_dia else None
            etiqueta_ventana = (f'{rolling_window}v ≈ {dias_ventana:.0f}d' if dias_ventana
                                else f'{rolling_window}v')
            if len(r) > rolling_window * 2:
                r_var95 = r.rolling(window=rolling_window).quantile(0.05)
                r_var99 = r.rolling(window=rolling_window).quantile(0.01)
                paso_rolling = max(1, len(r_var95) // 2000)
                idx_r = r_var95.index[::paso_rolling] if es_datetime_valido else np.arange(len(r_var95))[::paso_rolling]
                ax3.plot(idx_r, r_var95.iloc[::paso_rolling] * 100,
                         color='#BA7517', linewidth=0.6, label=f'VaR 95% ({etiqueta_ventana})', rasterized=True)
                ax3.plot(idx_r, r_var99.iloc[::paso_rolling] * 100,
                         color='#E24B4A', linewidth=0.6, label=f'VaR 99% ({etiqueta_ventana})', rasterized=True)
                # p05_r/p01_r son percentiles (P5/P1) de TODO el periodo, no
                # medias — sirven de referencia fija frente al VaR rodante
                # (que varia con la ventana movil de arriba).
                ax3.axhline(p05_r * 100, color='#BA7517', linewidth=0.8, linestyle='--', alpha=0.5,
                            label=f'VaR 95% histórico ($P_{{5}}$, todo el periodo): {p05_r*100:.2f}%')
                ax3.axhline(p01_r * 100, color='#E24B4A', linewidth=0.8, linestyle='--', alpha=0.5,
                            label=f'VaR 99% histórico ($P_{{1}}$, todo el periodo): {p01_r*100:.2f}%')
                ax3.fill_between(idx_r, r_var95.iloc[::paso_rolling] * 100, alpha=0.08, color='#BA7517')
                ax3.set_title(f"Rolling VaR (ventana {etiqueta_ventana})", color='white', fontsize=10, pad=6)
            else:
                ax3.text(0.5, 0.5, "Datos insuficientes para Rolling VaR",
                         ha='center', va='center', fontsize=10, color='#888780', transform=ax3.transAxes)
            ax3.set_xlabel('Tiempo' if es_datetime_valido else 'Nº de vela', color='#888780', fontsize=7)
            ax3.set_ylabel('Pérdida %', color='#888780', fontsize=7)
            ax3.tick_params(colors='#888780', labelsize=6)
            ax3.grid(True, alpha=0.15, color='#444')
            ax3.legend(facecolor='#222', labelcolor='white', fontsize=6, loc='lower left')
            for spine in ax3.spines.values(): spine.set_edgecolor('#333')

            if _EXPORTAR_PLOT:
                _sigmas_vela = {}
                for _s in sigmas:
                    for _lado in (-1, 1):
                        _val = r_media + (_lado * _s * r_std)
                        _prob = (np.mean(r <= _val) if _lado == -1 else np.mean(r >= _val))
                        _sigmas_vela[str(_lado * _s)] = {'val': float(_val), 'prob': float(_prob)}

                _vars = {}
                for _conf, _z in ((0.95, 1.645), (0.99, 2.326)):
                    _val_var = (r_media - _z * r_std if es_normal_p6
                                else np.percentile(r, (1 - _conf) * 100))
                    _vars[str(_conf)] = {'val': float(_val_var),
                                         'prob': float(np.mean(r <= _val_var))}
                    # CVaR (Expected Shortfall): la media de la cola más allá
                    # del VaR. Paramétrico si la serie pasa normalidad, histórico
                    # si no — misma lógica adaptativa que el VaR.
                    if es_normal_p6:
                        _val_cvar = r_media - r_std * (stats.norm.pdf(_z) / _conf)
                    else:
                        _cola = r[r <= _val_var]
                        _val_cvar = float(_cola.mean()) if len(_cola) else float(_val_var)
                    _vars[f'cvar{int(_conf * 100)}'] = {
                        'val': float(_val_cvar),
                        'prob': float(np.mean(r <= _val_cvar))}

                _rolling = None
                if 'r_var95' in locals() and len(r) > rolling_window * 2:
                    _rolling = {
                        'x': (_eje(r_var95.index[::paso_rolling]) if es_datetime_valido
                              else np.arange(len(r_var95))[::paso_rolling]),
                        'var95_pct': _arr(r_var95.iloc[::paso_rolling]).astype(float) * 100,
                        'var99_pct': _arr(r_var99.iloc[::paso_rolling]).astype(float) * 100,
                        'etiqueta_ventana': etiqueta_ventana,
                    }

                _PLOT['riesgo_intradia'] = {
                    'media': float(r_media), 'std': float(r_std),
                    'sigmas': _sigmas_vela,
                    'var': _vars,
                    'es_normal': bool(es_normal_p6),
                    'p05_pct': float(p05_r) * 100,
                    'p01_pct': float(p01_r) * 100,
                    'rolling': _rolling,
                }

            pdf.savefig(fig, facecolor=fig.get_facecolor(), dpi=150)
            plt.close()
            _registrar_pagina('Riesgo Intradiario + Rolling VaR')
    
        except Exception as e:
            print(f"ERROR EN PÁGINA M+5 (Riesgo Intradiario): {e}")
            traceback.print_exc()
            print(f"  TIPOS: r={type(r).__name__}, velas_por_dia={type(velas_por_dia).__name__}, vol_por_hora={type(vol_por_hora).__name__}")
            plt.close()
    
    # endregion

    # region ── PÁGINA M+6 — Boxplot + Retorno Esperado por Hora ──────────────────
        if 'vol_por_hora' not in locals():
            vol_por_hora = pd.Series(dtype=float)
        tiene_datos_hora = len(vol_por_hora) > 0

        try:
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.patch.set_facecolor('#0f0f0f')

            if tiene_datos_hora:
                gs = gridspec.GridSpec(2, 1, hspace=0.40, height_ratios=[1.5, 1])

                # ── (0) Boxplot: Desviación Estándar por Hora ──
                ax1 = fig.add_subplot(gs[0])
                ax1.set_facecolor('#111111')
                for spine in ax1.spines.values(): spine.set_edgecolor('#333')

                hourly_vol = df.groupby([pd.Grouper(freq='W-MON'), 'hora_utc'])['retorno'].std().dropna() * 100

                # Agrupar una sola vez por hora (evita 24 escaneos .xs sobre toda la serie)
                grupos_hora = {h: v.values for h, v in hourly_vol.groupby(level='hora_utc')}

                bxp_stats = []
                hourly_means = []
                hourly_n = []
                for h in range(24):
                    vals = grupos_hora.get(h, np.array([]))
                    n = len(vals)
                    hourly_n.append(n)
                    if n > 0:
                        q1, q2, q3 = np.percentile(vals, [25, 50, 75])
                        iqr = q3 - q1
                        umbral_iqr = 3.0 if n > 500 else 1.5
                        lower, upper = q1 - umbral_iqr*iqr, q3 + umbral_iqr*iqr
                        whislo = np.min(vals[vals >= lower])
                        whishi = np.max(vals[vals <= upper])
                        if n > 1000:
                            fliers = np.array([])
                        else:
                            fliers = vals[(vals < whislo) | (vals > whishi)]
                            if len(fliers) > 30:
                                fliers = np.random.choice(fliers, 30, replace=False)
                        mean_val = np.mean(vals)
                    else:
                        q1 = q2 = q3 = whislo = whishi = mean_val = 0.0
                        fliers = np.array([])
                    hourly_means.append(mean_val)
                    bxp_stats.append({'med': q2, 'q1': q1, 'q3': q3,
                                      'whislo': whislo, 'whishi': whishi,
                                      'fliers': fliers, 'mean': mean_val, 'n': n})

                bp = ax1.bxp(bxp_stats, positions=range(24), widths=0.6,
                             patch_artist=True,
                             boxprops=dict(facecolor='#185FA5', alpha=0.6, edgecolor='#4A8BC2', linewidth=0.8),
                             whiskerprops=dict(color='#4A8BC2', linewidth=0.8),
                             capprops=dict(color='#4A8BC2', linewidth=0.8),
                             medianprops=dict(color='#E24B4A', linewidth=1.5),
                             flierprops=dict(marker='o', markerfacecolor='#E24B4A', markersize=3, alpha=0.4))

                for i, mv in enumerate(hourly_means):
                    ax1.plot(i, mv, 'D', color='#E24B4A', markersize=5, zorder=5, alpha=0.9)

                # Fijar el ylim en base a los whiskers reales (no a los fliers): con millones
                # de velas de 1 minuto, un solo outlier extremo (flash-crash) puede quedar
                # dibujado como flier y estirar el autoscale de matplotlib, aplastando todas
                # las cajas. Al fijar el rango explícitamente los fliers fuera de rango se
                # recortan (clip_on=True por defecto) en vez de forzar el zoom-out del eje.
                whisk_los = [s['whislo'] for s in bxp_stats if s['n'] > 0]
                whisk_his = [s['whishi'] for s in bxp_stats if s['n'] > 0]
                if whisk_los and whisk_his:
                    rango = max(whisk_his) - min(whisk_los)
                    margen = rango * 0.15 if rango > 0 else max(whisk_his) * 0.1 or 1.0
                    ax1.set_ylim(min(0, min(whisk_los) - margen * 0.2), max(whisk_his) + margen)

                total_avg_vol = hourly_vol.mean()
                ax1.axhline(total_avg_vol, color='#BA7517', linewidth=1.2, linestyle='--',
                            alpha=0.8, label=f'Vol. media total: {total_avg_vol:.4f}%')
                ax1.plot(0, total_avg_vol, '>', color='#BA7517',
                         transform=ax1.get_yaxis_transform(), markersize=6, clip_on=False)

                y_top = ax1.get_ylim()[1]
                for hora, nombre, color in [(0, 'Tokio', '#E24B4A'), (8, 'Londres', '#58a6ff'), (13, 'NY', '#d29922')]:
                    ax1.axvline(hora, color=color, linewidth=1, linestyle=':', alpha=0.5)
                    ax1.text(hora + 0.3, y_top * 0.97, nombre, fontsize=6, color=color,
                             ha='left', va='top', alpha=0.8)

                ax1.set_title(f"{CONFIG['nombre']} — Desviación Estándar de Retornos por Hora UTC ({CONFIG['tf']})",
                              color='white', fontsize=11)
                ax1.set_xlabel('Hora UTC', color='#888780')
                ax1.set_ylabel('Desv. Est. 1h (%)', color='#888780')
                ax1.tick_params(colors='#888780')
                ax1.set_xticks(range(0, 24))
                ax1.set_xticklabels(range(24), fontsize=8)
                n_approx = int(round(np.mean(hourly_n), -2))
                ax1.text(0.01, 0.01, f'N ≈ {n_approx:,} por hora',
                         transform=ax1.transAxes, ha='left', va='bottom',
                         fontsize=9, color='#888780')
                ax1.grid(True, alpha=0.2, color='#444')
                legend_elements = [
                    Line2D([0], [0], color='#E24B4A', linewidth=1.5, label='Mediana'),
                    Line2D([0], [0], marker='D', color='w', markerfacecolor='#E24B4A', markersize=5, label='Media'),
                    Line2D([0], [0], color='#BA7517', linewidth=1.2, linestyle='--', label=f'Vol. media total: {total_avg_vol:.4f}%'),
                ]
                ax1.legend(handles=legend_elements, facecolor='#222', labelcolor='white', fontsize=8, loc='upper right')

                # ── (1) Perfil Horario de Retorno Acumulado ──
                ax2 = fig.add_subplot(gs[1])
                ax2.set_facecolor('#111111')
                ret_por_hora = df.groupby('hora_utc')['retorno'].sum() * 100
                vals = ret_por_hora.reindex(range(24), fill_value=0).values
                cols = ['#1D9E75' if v >= 0 else '#E24B4A' for v in vals]
                ax2.bar(range(24), vals, color=cols, alpha=0.7, width=0.8)
                ax2.axhline(0, color='#888780', linewidth=0.5, linestyle='--')

                # Línea horizontal del retorno acumulado medio por hora
                _ret_mean = float(np.mean(vals))
                ax2.axhline(_ret_mean, color='#BA7517', linewidth=1.0, linestyle=':',
                            alpha=0.7, label=f'Retorno medio: {_ret_mean:.2f}%')

                ax2.set_title(f"{CONFIG['nombre']} — Retorno Acumulado por Hora (%)",
                              color='white', fontsize=11, pad=6)
                ax2.set_xlabel('Hora UTC', color='#888780', fontsize=8)
                ax2.set_ylabel('Retorno acumulado %', color='#888780', fontsize=8)
                ax2.set_xticks(range(24))
                ax2.set_xticklabels([str(h) for h in range(24)], fontsize=6.5)
                ax2.tick_params(colors='#888780', labelsize=7)
                ax2.set_xlim(-0.5, 23.5)
                ax2.grid(True, axis='y', alpha=0.15, color='#444')

                # Top 3 horas por valor absoluto para resaltar
                _top3 = set(np.argsort(np.abs(vals))[-3:])

                y_top = ax2.get_ylim()[1]
                y_bot = ax2.get_ylim()[0]
                for hora, nombre, color in [(0, 'Tokio', '#E24B4A'), (8, 'Londres', '#58a6ff'), (13, 'NY', '#d29922')]:
                    ax2.axvline(hora, color=color, linewidth=0.8, linestyle='--', alpha=0.6)
                    ax2.text(hora + 0.3, y_top * 0.95, nombre, fontsize=6, color=color,
                             ha='left', va='top', alpha=0.8)
                _label_bbox = dict(boxstyle='round,pad=0.15', facecolor='#222', alpha=0.7, edgecolor='none')
                for h in range(24):
                    if abs(vals[h]) > 0.01:
                        is_top = h in _top3
                        lbl_color = '#d29922' if is_top else 'white'
                        lbl_fw = 'bold' if is_top else 'normal'
                        lbl_fs = 7 if is_top else 6.5
                        offset = abs(y_top - y_bot) * 0.02
                        ax2.text(h, vals[h] + (offset if vals[h] > 0 else -offset),
                                 f'{vals[h]:.2f}%', ha='center',
                                 va='bottom' if vals[h] > 0 else 'top',
                                 fontsize=lbl_fs, color=lbl_color, fontweight=lbl_fw,
                                 bbox=_label_bbox)
                ax2.legend(loc='upper right', fontsize=7, facecolor='#222',
                           labelcolor='#888780', framealpha=0.3)
                for spine in ax2.spines.values(): spine.set_edgecolor('#333')

                if _EXPORTAR_PLOT:
                    _PLOT['perfil_horario'] = {
                        'disponible': True,
                        # bxp_stats se pasa tal cual a ax.bxp() en la GUI
                        'bxp': [{'med': float(s['med']), 'q1': float(s['q1']),
                                 'q3': float(s['q3']), 'whislo': float(s['whislo']),
                                 'whishi': float(s['whishi']), 'mean': float(s['mean']),
                                 'fliers': np.asarray(s['fliers'], dtype=float),
                                 'n': int(s['n'])} for s in bxp_stats],
                        'medias': np.asarray(hourly_means, dtype=float),
                        'n_por_hora': np.asarray(hourly_n, dtype=int),
                        'vol_media_total': float(total_avg_vol),
                        'ret_por_hora': np.asarray(vals, dtype=float),
                    }

            else:
                if _EXPORTAR_PLOT:
                    _PLOT['perfil_horario'] = {'disponible': False}
                ax = fig.add_subplot(111)
                ax.axis('off')
                ax.set_facecolor('#111111')
                ax.text(0.5, 0.5,
                        "Análisis intradía no disponible.\n\n"
                        "Timeframe diario (1d): todas las velas tienen marca\n"
                        "de tiempo a las 00:00 UTC, no hay distribución horaria.",
                        ha='center', va='center', fontsize=12,
                        color='#888780', transform=ax.transAxes)

            pdf.savefig(fig, facecolor=fig.get_facecolor(), dpi=150)
            plt.close()
            _registrar_pagina('Boxplot + Retorno Esperado por Hora')

        except Exception as e:
            print(f"ERROR EN PÁGINA M+6 (Boxplot por Hora): {e}")
            plt.close()

    # endregion
    # region ── PÁGINA M+7 — Matriz de Correlación Intra-diaria 24×24 ────────────
        try:
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.patch.set_facecolor('#0f0f0f')
            tiene_corr = 'corr_24' in locals() and not corr_24.empty and corr_24.shape == (24, 24)
    
            if tiene_corr:
                ax = fig.add_subplot(111)
                ax.set_facecolor('#111111')
                mask = np.zeros_like(corr_24, dtype=bool)
                mask[np.triu_indices_from(mask, k=1)] = True
                sns.heatmap(corr_24, mask=mask, annot=False, cmap='coolwarm',
                            center=0, vmin=-1, vmax=1, square=True, ax=ax,
                            linewidths=0.3, linecolor='#333',
                            cbar_kws={'label': 'Correlación de Pearson', 'shrink': 0.8})
                for i in range(corr_24.shape[0]):
                    for j in range(corr_24.shape[1]):
                        if mask[i, j]:
                            continue
                        val = corr_24.iloc[i, j]
                        color = 'white' if abs(val) >= 0.5 else '#000'
                        ax.text(j + 0.5, i + 0.5, f'{val:.2f}',
                                ha='center', va='center', fontsize=4.5, color=color,
                                fontweight='bold' if abs(val) >= 0.5 else 'normal')
                ax.set_title(f"{CONFIG['nombre']} — Matriz de Correlación Intra-diaria (Spearman, {CONFIG['tf']})",
                             color='white', fontsize=12, pad=15)
                ax.set_xlabel('Hora UTC', color='#888780')
                ax.set_ylabel('Hora UTC', color='#888780')
                ax.tick_params(colors='#888780', labelsize=8)
                for spine in ax.spines.values(): spine.set_edgecolor('#333')
                cbar = ax.collections[0].colorbar
                cbar.ax.yaxis.label.set_color('#888780')
                cbar.ax.tick_params(colors='#888780')
                vmin, vmax = cbar.mappable.get_clim()
                cbar.set_ticks([vmin, (vmin+vmax)/2, vmax])
                cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
                if _EXPORTAR_PLOT:
                    _PLOT['corr_24'] = {
                        'm': corr_24.values.astype(float),
                        'labels': [str(c) for c in corr_24.columns],
                    }
            else:
                ax = fig.add_subplot(111)
                ax.axis('off')
                ax.set_facecolor('#111111')
                ax.text(0.5, 0.5, "Matriz de correlación intra-diaria no disponible.",
                        ha='center', va='center', fontsize=12, color='#888780', transform=ax.transAxes)
    
            pdf.savefig(fig, facecolor=fig.get_facecolor(), dpi=150)
            plt.close()
            _registrar_pagina('Matriz de Correlación Intra-diaria 24×24')
    
        except Exception as e:
            print(f"ERROR EN PÁGINA M+7 (Correlación 24×24): {e}")
            plt.close()
    
    # endregion
    # region ── PÁGINA M+8 — Heatmaps Semanal + Mensual ──
        try:
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.patch.set_facecolor('#0f0f0f')
    
            tiene_week = 'pivot_week' in locals() and not pivot_week.empty and pivot_week.shape[0] >= 3
            tiene_mes  = 'pivot_month' in locals() and not pivot_month.empty and pivot_month.shape[0] >= 3
    
            if tiene_week or tiene_mes:
                gs = gridspec.GridSpec(2, 1, hspace=0.40, height_ratios=[1, 1])
    
                # ── (0) Heatmap Semanal ──
                ax1 = fig.add_subplot(gs[0])
                ax1.set_facecolor('#111111')
                if tiene_week:
                    dias_labels = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
                    pw = pivot_week.copy()
                    pw.index = dias_labels[:len(pw)]
                    # La escala de color se capa al percentil 95: un único evento real
                    # extremo (ej. un flash-crash) no debe saturar el cmap y dejar sin
                    # contraste al resto de celdas. El valor anotado en cada celda sigue
                    # siendo el real, sin recortar.
                    vmax_pw = np.nanpercentile(pw.values, 95)
                    if not np.isfinite(vmax_pw) or vmax_pw <= 0:
                        vmax_pw = np.nanmax(pw.values)
                    sns.heatmap(pw, annot=False, cmap='YlOrRd', ax=ax1, vmin=0, vmax=vmax_pw,
                                linewidths=0.5, linecolor='#333', square=False,
                                cbar_kws={'label': 'Volatilidad (%)', 'shrink': 0.8, 'extend': 'max'})
                    for i in range(pw.shape[0]):
                        for j in range(pw.shape[1]):
                            val = pw.iloc[i, j]
                            ax1.text(j + 0.5, i + 0.5, f'{val:.3f}',
                                    ha='center', va='center', fontsize=5.5,
                                    color='#000', fontweight='bold')
                    ax1.set_title(f"{CONFIG['nombre']} — Estacionalidad Semanal: Volatilidad por Hora y Día ({CONFIG['tf']})",
                                  color='white', fontsize=12, pad=15)
                    ax1.set_xlabel('Hora UTC', color='#888780')
                    ax1.set_ylabel('Día de la semana', color='#888780')
                    ax1.tick_params(colors='#888780', labelsize=9)
                    cbar1 = ax1.collections[0].colorbar
                    cbar1.ax.yaxis.label.set_color('#888780')
                    cbar1.ax.tick_params(colors='#888780')
                    vmin1, vmax1 = cbar1.mappable.get_clim()
                    cbar1.set_ticks([vmin1, (vmin1+vmax1)/2, vmax1])
                    cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
                    if _EXPORTAR_PLOT:
                        _PLOT.setdefault('heatmaps', {})['week'] = {
                            'm': pw.values.astype(float),
                            'filas': [str(i) for i in pw.index],
                            'cols': [str(c) for c in pw.columns],
                            'vmax': float(vmax_pw),
                        }
                else:
                    ax1.text(0.5, 0.5, "No disponible", ha='center', va='center', fontsize=11, color='#888780', transform=ax1.transAxes)
                for spine in ax1.spines.values(): spine.set_edgecolor('#333')
    
                # ── (1) Heatmap Mensual ──
                ax2 = fig.add_subplot(gs[1])
                ax2.set_facecolor('#111111')
                if tiene_mes:
                    mes_labels = ['Ene','Feb','Mar','Abr','May','Jun',
                                  'Jul','Ago','Sep','Oct','Nov','Dic']
                    pm = pivot_month.copy()
                    pm.index = mes_labels[:len(pm)]
                    vmax_pm = np.nanpercentile(pm.values, 95)
                    if not np.isfinite(vmax_pm) or vmax_pm <= 0:
                        vmax_pm = np.nanmax(pm.values)
                    sns.heatmap(pm, annot=False, cmap='YlOrRd', ax=ax2, vmin=0, vmax=vmax_pm,
                                linewidths=0.5, linecolor='#333', square=False,
                                cbar_kws={'label': 'Volatilidad (%)', 'shrink': 0.8, 'extend': 'max'})
                    for i in range(pm.shape[0]):
                        for j in range(pm.shape[1]):
                            val = pm.iloc[i, j]
                            ax2.text(j + 0.5, i + 0.5, f'{val:.3f}',
                                    ha='center', va='center', fontsize=5.5,
                                    color='#000', fontweight='bold')
                    ax2.set_title(f"{CONFIG['nombre']} — Estacionalidad Mensual: Volatilidad por Hora y Mes ({CONFIG['tf']})",
                                  color='white', fontsize=12, pad=15)
                    ax2.set_xlabel('Hora UTC', color='#888780')
                    ax2.set_ylabel('Mes', color='#888780')
                    ax2.tick_params(colors='#888780', labelsize=9)
                    cbar2 = ax2.collections[0].colorbar
                    cbar2.ax.yaxis.label.set_color('#888780')
                    cbar2.ax.tick_params(colors='#888780')
                    vmin2, vmax2 = cbar2.mappable.get_clim()
                    cbar2.set_ticks([vmin2, (vmin2+vmax2)/2, vmax2])
                    cbar2.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
                    if _EXPORTAR_PLOT:
                        _PLOT.setdefault('heatmaps', {})['month'] = {
                            'm': pm.values.astype(float),
                            'filas': [str(i) for i in pm.index],
                            'cols': [str(c) for c in pm.columns],
                            'vmax': float(vmax_pm),
                        }
                else:
                    ax2.text(0.5, 0.5, "No disponible", ha='center', va='center', fontsize=11, color='#888780', transform=ax2.transAxes)
                for spine in ax2.spines.values(): spine.set_edgecolor('#333')
    
            else:
                ax = fig.add_subplot(111)
                ax.axis('off')
                ax.set_facecolor('#111111')
                ax.text(0.5, 0.5, "Heatmaps no disponibles.\nTF diario o sin datos horarios.",
                        ha='center', va='center', fontsize=12, color='#888780', transform=ax.transAxes)
    
            pdf.savefig(fig, facecolor=fig.get_facecolor(), dpi=150)
            plt.close()
            _registrar_pagina('Heatmaps Semanal + Mensual')
    
        except Exception as e:
            print(f"ERROR EN PÁGINA M+8 (Heatmaps): {e}")
            plt.close()
    
    # endregion
    # region ── PÁGINA M+8.5 — Cambio Acumulado Medio (Día/Semana/Mes/Año) ─────────
        try:
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.patch.set_facecolor('#0f0f0f')

            _idx_ca = (df.index.tz_localize('UTC') if df.index.tz is None
                       else df.index.tz_convert('UTC'))
            if _idx_ca.tz is not None:
                # a numpy las horas tz-aware no le caben en datetime64: pasar
                # a UTC naive evita el UserWarning de astype en metrics.py.
                _idx_ca = _idx_ca.tz_localize(None)
            dias_semana_ca = 7 if CONFIG['activo'] == 'CRYPTO' else 5
            curvas_ca = curvas_cambio_acumulado(
                df['retorno'].to_numpy(), _idx_ca.to_numpy(), dias_semana_ca)

            if not curvas_ca:
                ax = fig.add_subplot(111)
                ax.axis('off')
                ax.set_facecolor('#111111')
                ax.text(0.5, 0.5, "Cambio acumulado medio no disponible.\nSin retornos con fecha válida.",
                        ha='center', va='center', fontsize=12, color='#888780', transform=ax.transAxes)
            else:
                _NOMBRES_CA = {'dia': 'Día (24h)', 'semana': 'Semana',
                               'mes': 'Mes', 'anio': 'Año'}
                gs = gridspec.GridSpec(2, 2, hspace=0.50, wspace=0.22)
                for i, _clave_ca in enumerate(('dia', 'semana', 'mes', 'anio')):
                    ax = fig.add_subplot(gs[i])
                    ax.set_facecolor('#111111')
                    _c_curva = curvas_ca.get(_clave_ca)
                    if _c_curva is None:
                        ax.axis('off')
                        ax.text(0.5, 0.5, "Sin periodo completo", ha='center',
                                va='center', fontsize=10, color='#888780',
                                transform=ax.transAxes)
                        continue
                    _x_ca = _c_curva['pasos']
                    _y_ca = _c_curva['y']
                    _labels_ca = _c_curva['labels']
                    if len(_y_ca) >= 2:
                        _seg_ca = np.array([_x_ca, _y_ca]).T.reshape(-1, 1, 2)
                        _segs_ca = np.concatenate([_seg_ca[:-1], _seg_ca[1:]], axis=1)
                        _col_ca = np.where(np.diff(_y_ca) >= 0, '#1D9E75', '#E24B4A')
                        ax.add_collection(LineCollection(
                            _segs_ca, colors=_col_ca, linewidth=1.4, alpha=0.95, zorder=2))
                    ax.plot(_x_ca, _y_ca, 'o', color='#58a6ff', markersize=3,
                            zorder=3, alpha=0.7)
                    ax.autoscale()
                    ax.axhline(0, color='#888780', linewidth=0.6, linestyle='--', alpha=0.6)
                    ax.plot([_x_ca[-1]], [_y_ca[-1]], 'D', markersize=7, zorder=4,
                            color='#1D9E75' if _y_ca[-1] >= 0 else '#E24B4A')
                    ax.text(_x_ca[-1], _y_ca[-1], f"{_c_curva['total']:+.2f}%",
                            fontsize=8, color='white', fontweight='bold',
                            ha='left', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='#222',
                                      alpha=0.7, edgecolor='none'))
                    _paso_ticks_ca = max(1, len(_labels_ca) // 8)
                    _ticks_ca = list(_x_ca[::_paso_ticks_ca])
                    if _x_ca[-1] not in _ticks_ca:
                        _ticks_ca.append(_x_ca[-1])
                    ax.set_xticks(_ticks_ca)
                    ax.set_xticklabels([_labels_ca[int(t)] for t in _ticks_ca],
                                       fontsize=7)
                    ax.tick_params(colors='#888780', labelsize=7)
                    ax.grid(True, alpha=0.15, color='#444')
                    ax.set_title(f"{_NOMBRES_CA[_clave_ca]} — cambio acumulado medio\n"
                                 f"({_c_curva['n']} periodos completos)",
                                 color='white', fontsize=10, pad=6)
                    ax.set_ylabel('Cambio acumulado %', color='#888780', fontsize=7)
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#333')

                if _EXPORTAR_PLOT:
                    _PLOT['cambio_acumulado'] = dict(curvas_ca)
                    _PLOT['cambio_acumulado']['dias_semana'] = dias_semana_ca

            pdf.savefig(fig, facecolor=fig.get_facecolor(), dpi=150)
            plt.close()
            _registrar_pagina('Cambio Acumulado Medio (Día/Semana/Mes/Año)')

        except Exception as e:
            print(f"ERROR EN PÁGINA M+8.5 (Cambio Acumulado Medio): {e}")
            traceback.print_exc()
            plt.close()

    # endregion
    # region ── PÁGINA M+9 — NATR, correlación Multi-TF ─────
        try:
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.patch.set_facecolor('#0f0f0f')
            gs = gridspec.GridSpec(2, 1, hspace=0.30, height_ratios=[1.5, 1])

            if _NATR_CORR is not None and not _NATR_CORR.empty and _NATR_CORR.shape[0] >= 2:
                ax_top = fig.add_subplot(gs[0])
                ax_top.set_facecolor('#111111')
                mask = np.zeros_like(_NATR_CORR, dtype=bool)
                mask[np.triu_indices_from(mask, k=1)] = True
                sns.heatmap(_NATR_CORR, mask=mask, annot=False, cmap='coolwarm',
                            center=0, vmin=-1, vmax=1, square=True, ax=ax_top,
                            linewidths=0.3, linecolor='#333',
                            cbar_kws={'label': 'Correlación de NATR', 'shrink': 0.8})
                for i in range(_NATR_CORR.shape[0]):
                    for j in range(_NATR_CORR.shape[1]):
                        if mask[i, j]:
                            continue
                        val = _NATR_CORR.iloc[i, j]
                        color = 'white' if abs(val) >= 0.5 else '#000'
                        fs = max(4.5, min(9, 36 // _NATR_CORR.shape[0]))
                        ax_top.text(j + 0.5, i + 0.5, f'{val:.2f}',
                                    ha='center', va='center', fontsize=fs, color=color,
                                    fontweight='bold' if abs(val) >= 0.5 else 'normal')
                ax_top.set_title(f"{CONFIG['nombre']} ({CONFIG['tf']}) — NATR, correlación Multi-TF",
                                 color='white', fontsize=12, pad=15)
                ax_top.set_xlabel('Timeframe', color='#888780')
                ax_top.set_ylabel('Timeframe', color='#888780')
                ax_top.tick_params(colors='#888780', labelsize=8)
                for spine in ax_top.spines.values():
                    spine.set_edgecolor('#333')
                cbar = ax_top.collections[0].colorbar
                cbar.ax.yaxis.label.set_color('#888780')
                cbar.ax.tick_params(colors='#888780')
                cbar.ax.set_yticks(cbar.get_ticks())
                cbar.ax.set_yticklabels([f'{x:.2f}' for x in cbar.get_ticks()])
            else:
                ax_top = fig.add_subplot(gs[0])
                ax_top.axis('off')
                ax_top.set_facecolor('#111111')
                ax_top.text(0.5, 0.5, "Matriz de correlación NATR no disponible.",
                            ha='center', va='center', fontsize=12, color='#888780', transform=ax_top.transAxes)

            ax_bot = fig.add_subplot(gs[1])
            ax_bot.axis('off')
            ax_bot.set_facecolor('#111111')
            all_pairs = []
            for h_name in ['General']:
                for pair_data in _NATR_PAIRS.get(h_name, []):
                    all_pairs.append((h_name, pair_data))
            if all_pairs:
                ax_bot.text(0.04, 0.92, 'Pares de volatilidad por horizonte:', fontsize=9,
                            color='#ff9900', fontweight='bold', va='center')
                headers = ['Horizonte', 'Par', 'NATR base', 'NATR target', 'Ratio', 'Lead-Lag']
                col_x = [0.04, 0.18, 0.32, 0.44, 0.56, 0.68]
                for ci, hdr in enumerate(headers):
                    ax_bot.text(col_x[ci], 0.86, hdr, fontsize=7, color='#4fc3f7',
                                fontweight='bold', va='center')
                y = 0.80
                for h_name, pair_data in all_pairs:
                    vals = [
                        h_name,
                        pair_data['pair'],
                        f'{pair_data["natr_base"]:.4f}',
                        f'{pair_data["natr_target"]:.4f}',
                        f'{pair_data["ratio"]:.4f}',
                        f'~{pair_data["lag"]} velas ({pair_data["lag_unit"]}) [max: {pair_data["max_lag"]} lags]',
                    ]
                    for ci, v in enumerate(vals):
                        ax_bot.text(col_x[ci], y, v, fontsize=6.5, color='white', va='center')
                    y -= 0.055
                    if y < 0.05:
                        break
            else:
                ax_bot.text(0.5, 0.5, "No hay pares disponibles para el TF base actual.",
                            ha='center', va='center', fontsize=10, color='#888780', transform=ax_bot.transAxes)

            if _EXPORTAR_PLOT:
                _tiene_corr_natr = (_NATR_CORR is not None and not _NATR_CORR.empty
                                    and _NATR_CORR.shape[0] >= 2)
                _PLOT['natr_multitf'] = {
                    'corr': (_NATR_CORR.values.astype(float) if _tiene_corr_natr else None),
                    'labels': ([str(c) for c in _NATR_CORR.columns] if _tiene_corr_natr else []),
                    'pares': [{
                        'horizonte': h_name,
                        'pair': str(pd_pair['pair']),
                        'natr_base': float(pd_pair['natr_base']),
                        'natr_target': float(pd_pair['natr_target']),
                        'ratio': float(pd_pair['ratio']),
                        'lag': int(pd_pair['lag']),
                        'lag_unit': str(pd_pair['lag_unit']),
                        'max_lag': int(pd_pair['max_lag']),
                    } for h_name, pd_pair in all_pairs],
                }

            plt.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.06)
            pdf.savefig(fig, facecolor=fig.get_facecolor(), dpi=150)
            plt.close()
            _registrar_pagina('NATR, correlación Multi-TF')
        except Exception as e:
            print(f"ERROR EN PÁGINA NATR: {e}")
            plt.close()
    # endregion
    # region ── PÁGINA M+10 — Correlograma (ACF/PACF) ────────
        try:
            fig = plt.figure(figsize=(11.69, 8.27))
            gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1])

            # 1. Heatmap (top-left)
            ax1 = fig.add_subplot(gs[0, 0])
            data = {p: _pacf1(stats) for p, stats in stats_temporales.items() if 'pacf_vals' in stats}
            df_heat = pd.DataFrame(data, index=['PACF Lag 1']).T
            sns.heatmap(df_heat, annot=True, cmap='coolwarm', center=0, vmin=-0.5, vmax=0.5, ax=ax1)
            ax1.set_title("Mapa de Calor: Dependencia por Escala")

            # 2. Correlograma (top-center and top-right)
            ax2 = fig.add_subplot(gs[0, 1])
            plot_acf(series_temporales['dia'], ax=ax2, lags=15, title='ACF Diario')
            ax3 = fig.add_subplot(gs[0, 2])
            plot_pacf(series_temporales['dia'], ax=ax3, lags=15, title='PACF Diario')

            # 3. Comparativa Aleatoria (middle, full width)
            ax4 = fig.add_subplot(gs[1, :])
            serie = series_temporales['dia']
            precio_inicial = float(df['close'].dropna().iloc[0])
            # exp(cumsum(retorno log)) * precio inicial reconstruye una
            # trayectoria de PRECIO (mismas unidades que el activo real), en
            # vez de un cumsum de retornos log que no es directamente un
            # precio simulado.
            precio_real_sim = precio_inicial * np.exp(serie.cumsum())

            colores_ruido = ['#e67e22', '#9b59b6', '#3498db', '#2ecc71', '#f1c40f']
            n_simulaciones = 5
            _walks_export = []
            for i in range(n_simulaciones):
                rw = np.random.normal(loc=serie.mean(), scale=serie.std(), size=len(serie))
                precio_ruido_sim = precio_inicial * np.exp(np.cumsum(rw))
                ax4.plot(precio_ruido_sim, color=colores_ruido[i % len(colores_ruido)],
                         linewidth=1.0, alpha=0.65, linestyle='--',
                         label='Random Walk (Ruido)' if i == 0 else None, zorder=2)
                # Las trayectorias son aleatorias: hay que exportar LAS MISMAS
                # que se dibujaron, no volver a simularlas en la GUI (saldrían
                # distintas y el gráfico no coincidiría con el PDF).
                if _EXPORTAR_PLOT:
                    _walks_export.append(np.asarray(precio_ruido_sim, dtype=float))

            # Línea del activo real más gruesa y por encima (zorder) del
            # resto, para que se note que es la protagonista de la comparativa.
            ax4.plot(precio_real_sim, color='#1a1a1a', linewidth=2.4, alpha=0.95,
                     label='Activo Real', zorder=5)
            ax4.legend()
            ax4.set_title("Estructura vs. Ruido Aleatorio (simulación de precio)")
            ax4.set_ylabel('Precio simulado')

            # 4. ACF de retornos al cuadrado (bottom, full width)
            ax5 = fig.add_subplot(gs[2, :])
            if not np.isnan(clustering_lag1) and len(r_sq_clean) > 30:
                lags_sq_plot = min(20, len(r_sq_clean) // 4 - 1)
                if lags_sq_plot >= 2:
                    # Usar ACF precalculado en PASO 15 (evita recalcular sobre millones de filas)
                    lags_arr = np.arange(len(acf_sq_full))
                    markerline, stemlines, baseline = ax5.stem(lags_arr, acf_sq_full)
                    plt.setp(stemlines, color='#185FA5', linewidth=1.2)
                    plt.setp(markerline, color='#185FA5', marker='o', markersize=4)
                    plt.setp(baseline, color='#444', linewidth=0.5)
                    ax5.axhline(0, color='#888780', linewidth=0.5)
                    ci = 1.96 / np.sqrt(len(r_sq_clean))
                    ax5.axhline(ci, color='#E24B4A', linewidth=0.8, linestyle='--')
                    ax5.axhline(-ci, color='#E24B4A', linewidth=0.8, linestyle='--')
                    ax5.set_title('ACF Retornos² — Volatility Clustering')
                    ax5.text(0.98, 0.95, f'LB p-valor: {clustering_lb_p:.4f}  {"✓" if clustering_presente else "✗"}',
                             transform=ax5.transAxes, ha='right', va='top', fontsize=8,
                             color='#1D9E75' if clustering_presente else '#E24B4A')
            else:
                ax5.axis('off')
                ax5.text(0.5, 0.5, "No hay suficientes datos para ACF de retornos²",
                         ha='center', va='center', fontsize=11, color='#888780', transform=ax5.transAxes)

            if _EXPORTAR_PLOT:
                _tiene_acf_sq = (not np.isnan(clustering_lag1) and len(r_sq_clean) > 30
                                 and min(20, len(r_sq_clean) // 4 - 1) >= 2)
                _serie_dia_exp = _arr(series_temporales['dia']).astype(float)
                # Valores ACF/PACF por lag para el crosshair de la GUI (mismos
                # defaults que plot_acf/plot_pacf: nlags=15, sin ajuste).
                _acf_exp = None
                _pacf_exp = None
                if len(_serie_dia_exp) > 20:
                    _acf_exp = np.asarray(acf(_serie_dia_exp, nlags=15), dtype=float)
                    _pacf_exp = np.asarray(pacf(_serie_dia_exp, nlags=15), dtype=float)
                _PLOT['dependencia'] = {
                    'escalas': {'labels': [str(i) for i in df_heat.index],
                                'valores': df_heat.values.ravel().astype(float)},
                    # plot_acf/plot_pacf aceptan un array: la GUI los llama
                    # sobre esta misma serie diaria (pequeña, ya agregada).
                    'serie_dia': _serie_dia_exp,
                    'acf': _acf_exp,
                    'pacf': _pacf_exp,
                    'precio_real': np.asarray(precio_real_sim, dtype=float),
                    'random_walks': _walks_export,
                    'acf_sq': (np.asarray(acf_sq_full, dtype=float) if _tiene_acf_sq else None),
                    'ci': (float(1.96 / np.sqrt(len(r_sq_clean))) if _tiene_acf_sq else None),
                    'lb_p': (float(clustering_lb_p) if not np.isnan(clustering_lb_p) else None),
                    'clustering': bool(clustering_presente),
                }

            plt.tight_layout(pad=3.0)
            pdf.savefig(fig)
            plt.close(fig)
            _registrar_pagina('Análisis de Dependencia (ACF/PACF)')
        except Exception as e:
            print(f"ERROR EN PÁGINA M+10 (Correlograma): {e}")
            plt.close()
    # endregion
    # region ── PÁGINAS FINALES — Dashboards NATR (uno por horizonte) ─────
        for hi, horizon_name in enumerate(HORIZON_NAMES):
            try:
                fig = plt.figure(figsize=(11.69, 8.27))
                fig.patch.set_facecolor('#0f0f0f')
                window_z_days = WINDOW_ZSCORE_DAYS.get(horizon_name, 252)
                fig.suptitle(
                    f"{CONFIG['nombre']} ({CONFIG['tf']}) — Dashboard NATR {horizon_name} — Ventana Z: {window_z_days}d",
                    color='white', fontsize=13, y=0.975
                )
                tfs_disp = sorted(_NATR_DATA.keys(), key=lambda x: _tf_to_minutes(x) or 0)
                gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.25,
                                       height_ratios=[1.0, 1.0], left=0.07, right=0.96,
                                       top=0.91, bottom=0.06)
                # Cada panel va en su propio try (uno puede fallar sin tumbar
                # la página), así que el bundle se rellena panel a panel y se
                # asigna al final con lo que haya salido bien.
                _dash = {'window_z_days': int(window_z_days), 'term': None,
                         'z': None, 'serie_z': None, 'ratio': None}

                ax_a = fig.add_subplot(gs[0, 0])
                ax_a.set_facecolor('#111111')
                try:
                    if tfs_disp and _NATR_THEORETICAL:
                        mins = [_tf_to_minutes(tf) or 1 for tf in tfs_disp]
                        natrs_actual = [float(_NATR_DATA[tf].mean()) for tf in tfs_disp]
                        natrs_teo = [_NATR_THEORETICAL.get(tf, 0) for tf in tfs_disp]
                        ax_a.plot(mins, natrs_actual, 'o-', color='#4fc3f7', linewidth=1.4,
                                  markersize=5, label='NATR actual', zorder=3)
                        ax_a.plot(mins, natrs_teo, 'x--', color='#ff9900', linewidth=1.0,
                                  markersize=4, alpha=0.7, label='Te\u00f3rico \u221aT', zorder=2)
                        ax_a.set_xscale('log')
                        ax_a.set_yscale('log')
                        ax_a.set_xticks(mins)
                        ax_a.set_xticklabels(tfs_disp, rotation=30, ha='right', fontsize=7, color='#888780')
                        ax_a.tick_params(colors='#888780', labelsize=7)
                        ax_a.legend(loc='upper left', fontsize=7, framealpha=0.3, labelcolor='#888780')
                        # Backwardacion/contango se define por la forma de la curva REAL
                        # entre el TF mas corto y el mas largo (no comparando el punto base
                        # contra si mismo, que por construccion siempre coincide con el
                        # teorico en el primer indice). Contango = NATR% crece con el TF
                        # (patron normal, coherente con la ley sqrt(T)); backwardacion =
                        # el TF mas largo tiene NATR% menor que el mas corto (invertido).
                        if len(natrs_actual) >= 2 and natrs_actual[-1] < natrs_actual[0] * 0.9:
                            ax_a.text(0.98, 0.05, 'BACKWARDACI\u00d3N', transform=ax_a.transAxes,
                                      ha='right', va='bottom', color='#e24b4a', fontsize=8, fontweight='bold')
                            _estructura = 'BACKWARDACI\u00d3N'
                        else:
                            ax_a.text(0.98, 0.05, 'CONTANGO', transform=ax_a.transAxes,
                                      ha='right', va='bottom', color='#1d9e75', fontsize=8, fontweight='bold')
                            _estructura = 'CONTANGO'
                        if _EXPORTAR_PLOT:
                            _dash['term'] = {
                                'tfs': [str(t) for t in tfs_disp],
                                'mins': [float(m) for m in mins],
                                'actual': [float(v) for v in natrs_actual],
                                'teorico': [float(v) for v in natrs_teo],
                                'estructura': _estructura,
                            }
                    else:
                        ax_a.text(0.5, 0.5, "Sin datos para term structure.", ha='center', va='center',
                                  color='#888780', transform=ax_a.transAxes)
                except Exception:
                    ax_a.text(0.5, 0.5, "Error Panel A", ha='center', va='center', color='#888780',
                              transform=ax_a.transAxes)
                ax_a.set_title('A \u2500 Term Structure NATR', color='white', fontsize=10, pad=6)
                ax_a.set_xlabel('Timeframe', color='#888780', fontsize=8)
                ax_a.set_ylabel('NATR (%)', color='#888780', fontsize=8)
                for spine in ax_a.spines.values():
                    spine.set_edgecolor('#333')
                ax_a.grid(True, alpha=0.15, color='#333', which='both')

                ax_b = fig.add_subplot(gs[0, 1])
                ax_b.set_facecolor('#111111')
                try:
                    z_current_h = _NATR_Z_CURRENT.get(horizon_name, {})
                    if z_current_h:
                        tfs_z = sorted(z_current_h.keys(), key=lambda x: _tf_to_minutes(x) or 0)
                        z_vals = [z_current_h.get(tf, 0) for tf in tfs_z]
                        colors_z = ['#e24b4a' if z > 2 else '#1d9e75' if z < -2 else '#f1c40f' if abs(z) > 1 else '#2a4a6a'
                                    for z in z_vals]
                        y_pos = np.arange(len(tfs_z))
                        ax_b.barh(y_pos, z_vals, color=colors_z, edgecolor='#333', height=0.6)
                        ax_b.set_yticks(y_pos)
                        ax_b.set_yticklabels(tfs_z, fontsize=7, color='#888780')
                        ax_b.axvline(2, color='#e24b4a', linestyle='--', alpha=0.6, linewidth=0.8)
                        ax_b.axvline(-2, color='#1d9e75', linestyle='--', alpha=0.6, linewidth=0.8)
                        ax_b.axvline(0, color='#888780', linestyle='-', alpha=0.4, linewidth=0.5)
                        for i, (tf, z) in enumerate(zip(tfs_z, z_vals)):
                            ax_b.text(max(z + 0.1, 0.15) if z >= 0 else min(z - 0.1, -0.15),
                                      i, f'{z:+.2f}', va='center', fontsize=6.5, color='white',
                                      ha='left' if z >= 0 else 'right')
                        if _EXPORTAR_PLOT:
                            _dash['z'] = {'tfs': [str(t) for t in tfs_z],
                                          'vals': [float(z) for z in z_vals]}
                    else:
                        ax_b.text(0.5, 0.5, "Sin Z-scores.", ha='center', va='center',
                                  color='#888780', transform=ax_b.transAxes)
                except Exception:
                    ax_b.text(0.5, 0.5, "Error Panel B", ha='center', va='center',
                              color='#888780', transform=ax_b.transAxes)
                ax_b.set_title('B \u2500 Z-score NATR por TF', color='white', fontsize=10, pad=6)
                ax_b.set_xlabel('Z-score', color='#888780', fontsize=8)
                ax_b.tick_params(colors='#888780', labelsize=7)
                for spine in ax_b.spines.values():
                    spine.set_edgecolor('#333')
                ax_b.grid(True, alpha=0.15, color='#333', axis='x')
                ax_b.set_xlim(-4, 4)

                ax_c = fig.add_subplot(gs[1, 0])
                ax_c.set_facecolor('#111111')
                par_principal = None
                try:
                    z_series_h = _NATR_Z_SERIES.get(horizon_name, {})
                    # Solo se pintan las 2 series del par principal de este
                    # horizonte (mismo par que usa el Panel D de ratio) \u2014 con
                    # las 8 temporalidades a la vez el grafico era ilegible.
                    horizon_pairs = HORIZON_PAIRS.get(horizon_name, [])
                    par_principal = horizon_pairs[0] if horizon_pairs else None
                    tfs_plot = []
                    if par_principal:
                        tf_a_key = _natr_key(par_principal[0])
                        tf_b_key = _natr_key(par_principal[1])
                        tfs_plot = [tf for tf in (tf_a_key, tf_b_key) if tf and tf in z_series_h]
                    if tfs_plot:
                        line_colors = ['#4fc3f7', '#ff9900']
                        _series_z_export = []
                        for i, tf in enumerate(tfs_plot):
                            zs = z_series_h[tf]
                            if len(zs) > 0:
                                step = max(1, len(zs) // 500)
                                ax_c.plot(zs.index[::step], zs.values[::step],
                                          color=line_colors[i % len(line_colors)],
                                          linewidth=0.9, alpha=0.85, label=tf)
                                if _EXPORTAR_PLOT:
                                    _series_z_export.append({
                                        'tf': str(tf),
                                        'x': _eje(zs.index[::step]),
                                        'y': np.asarray(zs.values[::step], dtype=float),
                                    })
                        if _EXPORTAR_PLOT and _series_z_export:
                            _dash['serie_z'] = {
                                'par': (f'{par_principal[0]}/{par_principal[1]}'
                                        if par_principal else None),
                                'series': _series_z_export,
                            }
                        ax_c.axhline(2, color='#e24b4a', linestyle='--', alpha=0.5, linewidth=0.8)
                        ax_c.axhline(-2, color='#1d9e75', linestyle='--', alpha=0.5, linewidth=0.8)
                        ax_c.legend(loc='upper right', fontsize=7, framealpha=0.3, labelcolor='#888780')
                    else:
                        ax_c.text(0.5, 0.5, "Sin series temporales.", ha='center', va='center',
                                  color='#888780', transform=ax_c.transAxes)
                except Exception:
                    ax_c.text(0.5, 0.5, "Error Panel C", ha='center', va='center',
                              color='#888780', transform=ax_c.transAxes)
                ax_c.set_title(f'C \u2500 Serie temporal Z-score ({par_principal[0]}/{par_principal[1]})'
                               if par_principal else 'C \u2500 Serie temporal Z-score',
                               color='white', fontsize=10, pad=6)
                ax_c.set_ylabel('Z-score', color='#888780', fontsize=8)
                ax_c.tick_params(colors='#888780', labelsize=6)
                for spine in ax_c.spines.values():
                    spine.set_edgecolor('#333')
                ax_c.grid(True, alpha=0.15, color='#333')

                ax_d = fig.add_subplot(gs[1, 1])
                ax_d.set_facecolor('#111111')
                try:
                    ratio_series_h = _NATR_RATIO_SERIES.get(horizon_name, {})
                    ratio_bb_h = _NATR_RATIO_BB.get(horizon_name, {})
                    if ratio_series_h:
                        ratio_colors = ['#4fc3f7', '#ff9900']
                        _ratio_export = []
                        for i, ((tf_a, tf_b), ratio_s) in enumerate(ratio_series_h.items()):
                            bb = ratio_bb_h.get((tf_a, tf_b), {})
                            step = max(1, len(ratio_s) // 500)
                            x = ratio_s.index[::step]
                            y = ratio_s.values[::step]
                            color_r = ratio_colors[i % len(ratio_colors)]
                            ax_d.plot(x, y, color=color_r, linewidth=0.9, alpha=0.85,
                                      label=f'{tf_a}/{tf_b}')
                            # clave tupla → "tf_a/tf_b" (el bundle no admite tuplas)
                            _r_item = {'label': f'{tf_a}/{tf_b}',
                                       'x': _eje(x) if _EXPORTAR_PLOT else None,
                                       'y': np.asarray(y, dtype=float) if _EXPORTAR_PLOT else None,
                                       'bb': None}
                            if bb.get('upper') is not None and len(bb['upper']) > 0:
                                mu_b = bb['mean'].dropna()
                                up = bb['upper'].dropna()
                                lo = bb['lower'].dropna()
                                step_bb = max(1, len(mu_b) // 500)
                                ax_d.fill_between(mu_b.index[::step_bb],
                                                  lo.values[::step_bb], up.values[::step_bb],
                                                  color=color_r, alpha=0.10)
                                ax_d.plot(mu_b.index[::step_bb], mu_b.values[::step_bb],
                                          color=color_r, linestyle=':', linewidth=0.6, alpha=0.5)
                                if _EXPORTAR_PLOT:
                                    _r_item['bb'] = {
                                        'x': _eje(mu_b.index[::step_bb]),
                                        'mu': np.asarray(mu_b.values[::step_bb], dtype=float),
                                        'lo': np.asarray(lo.values[::step_bb], dtype=float),
                                        'up': np.asarray(up.values[::step_bb], dtype=float),
                                    }
                            curr_val = bb.get('current', 0)
                            ax_d.scatter([ratio_s.index[-1]], [curr_val], color=color_r, s=25, zorder=5)
                            if _EXPORTAR_PLOT:
                                _r_item['x_actual'] = _eje(pd.Index([ratio_s.index[-1]]))
                                _r_item['current'] = float(curr_val)
                                _ratio_export.append(_r_item)
                        if _EXPORTAR_PLOT and _ratio_export:
                            _dash['ratio'] = _ratio_export
                        ax_d.legend(loc='upper right', fontsize=6, framealpha=0.3, labelcolor='#888780')
                    else:
                        ax_d.text(0.5, 0.5, "Sin ratios Short/Long disponibles.",
                                  ha='center', va='center', color='#888780', transform=ax_d.transAxes)
                except Exception:
                    ax_d.text(0.5, 0.5, "Error Panel D", ha='center', va='center',
                              color='#888780', transform=ax_d.transAxes)
                ax_d.set_title('D \u2500 Ratio Short/Long (Bandas Bollinger \u00b12\u03c3)', color='white', fontsize=10, pad=6)
                ax_d.set_ylabel('Ratio NATR fast/slow', color='#888780', fontsize=8)
                ax_d.tick_params(colors='#888780', labelsize=6)
                for spine in ax_d.spines.values():
                    spine.set_edgecolor('#333')
                ax_d.grid(True, alpha=0.15, color='#333')

                if _EXPORTAR_PLOT:
                    _PLOT.setdefault('dashboard_natr', {})[horizon_name] = _dash

                pdf.savefig(fig, facecolor=fig.get_facecolor(), dpi=150)
                plt.close()
                _registrar_pagina(f'Dashboard NATR \u2014 {horizon_name}', horizonte=horizon_name)
            except Exception as e:
                print(f"ERROR EN P\u00e1gina NATR {horizon_name}: {e}")
                traceback.print_exc()
                plt.close()
    # endregion
    # endregion
    # ── FINALIZACIÓN ──────────────────────────────────────────────────────────────
    # Volcado único del PDF completo (ya armado en memoria) a disco.
    with open(OUTPUT_PDF_TMP, 'wb') as _f:
        _f.write(_pdf_buffer.getvalue())
        _f.flush()
        os.fsync(_f.fileno())

    # El .tmp puede seguir bloqueado un instante por Explorer (miniatura/vista previa
    # del PDF anterior con el mismo nombre) u otro proceso, así que se reintenta unas
    # cuantas veces antes de rendirse. Si aun así falla, no debe tumbar el resto de la
    # finalización (guardado de métricas para la GUI, etc.): se deja el .tmp como
    # resultado válido en vez de perder todo el análisis ya calculado.
    for intento in range(5):
        try:
            os.replace(OUTPUT_PDF_TMP, OUTPUT_PDF)
            break
        except PermissionError as e:
            if intento == 4:
                print(f"⚠️  No se pudo renombrar el PDF final (archivo en uso): {e}")
                print(f"⚠️  El informe completo quedó guardado en: {OUTPUT_PDF_TMP}")
                OUTPUT_PDF = OUTPUT_PDF_TMP
            else:
                time.sleep(0.5)
    print(f"{'='*60}")
    print(f"✅ Analisis completado con exito.")
    print(f"📁 Guardado en: {OUTPUT_PDF}")
    print(f"{'='*60}")

    if 'GUI_METRICS_OUTPUT' in os.environ:
        metrics_serializable = {}
        for cat, items in metricas.items():
            metrics_serializable[cat] = {}
            for k, v in items.items():
                if k.strip() == '':
                    continue
                metrics_serializable[cat][k] = str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
        # Mapa página→horizonte para el visor adaptativo de la GUI (clave reservada)
        metrics_serializable['_paginas'] = _PAGE_MAP
        with open(os.environ['GUI_METRICS_OUTPUT'], 'w', encoding='utf-8') as f:
            json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)
    if 'GUI_PDF_OUTPUT' in os.environ:
        with open(os.environ['GUI_PDF_OUTPUT'], 'w', encoding='utf-8') as f:
            json.dump({'pdf_path': OUTPUT_PDF}, f)

    if _EXPORTAR_PLOT:
        # ── KPIs numéricos para las tarjetas de la GUI ──
        # Van aquí y no en el metrics.json porque ese archivo es el contrato de
        # strings ya formateados (con ANSI y barras) que consume MetricsScroll;
        # las tarjetas necesitan el número crudo para colorear por umbral.
        def _f(v):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return None
            return v if math.isfinite(v) else None

        _kpi_por_horizonte = {}
        for _hz in HORIZON_NAMES:
            _eh = _ER_H.get(_hz)
            # HORIZON_HURST_RANGO no define 'General': ahí el Hurst es la serie
            # base df['hurst'], que es la que muestra la categoría 7. Sin este
            # respaldo la tarjeta quedaría vacía justo en la Ventana por defecto.
            _hs = _HURST_H[_hz]['serie'] if _hz in _HURST_H else df['hurst']
            _kpi_por_horizonte[_hz] = {
                'er_medio': _f(_eh['serie'].mean()) if _eh else None,
                'pct_tendencia': _f(_eh.get('pct_tendencia')) if _eh else None,
                'pct_ruido': _f(_eh.get('pct_ruido')) if _eh else None,
                'hurst_medio': _f(_hs.mean()),
                # Contrapeso de pct_tendencia: mismo criterio que la fila
                # '% Tiempo en mean reversion' de la categoría 7.
                'pct_reversion': _f((_hs < 0.52).mean()),
            }

        # Dependencia (ACF/PACF) por escala de calendario — mismo criterio y
        # mismas funciones (_pacf1, estado_adn) que ya arman la categoría 9 del
        # informe, para que la tarjeta y la fila de esa categoría coincidan
        # siempre. Claves PLANAS (no un dict anidado): TarjetasKPI._refrescar
        # lee kpi['global'] con origen.get(spec['clave']) igual que el resto.
        _dependencia_kpi = {}
        for _escala in ('dia', 'semanal', 'mensual', 'trimestral'):
            _st_dep = stats_temporales.get(_escala, {})
            _pv = _f(_pacf1(_st_dep)) if _st_dep else None
            _um = _f(_st_dep.get('umbral')) if _st_dep else None
            _dependencia_kpi[f'dependencia_{_escala}_pacf1'] = _pv
            _dependencia_kpi[f'dependencia_{_escala}_umbral'] = _um
            _dependencia_kpi[f'dependencia_{_escala}_estado'] = (
                estado_adn(_pv, _um) if (_pv is not None and _um is not None) else None)

        _PLOT['kpi'] = {
            'global': {
                'cagr': _f(ret_anual),
                'sharpe': _f(sharpe),
                'sortino': _f(sortino),
                'calmar': _f(calmar_ratio),
                'max_dd': _f(mdd),
                'dd_medio': _f(drawdown_medio),
                'vol_anual': _f(vol_anual),
                'vol_diaria': _f(vol_diaria),
                'var95': _f(p05_r),
                'var99': _f(p01_r),
                'cvar95': _f(p05_cola_r),
                'total_velas': int(len(df)),
                # ── forma de la distribución ──
                'kurtosis': _f(r.kurtosis()),
                'skew': _f(r.skew()),
                'es_normal': bool(p_jb >= 0.05) if _f(p_jb) is not None else None,
                'jb_p_value': _f(p_jb),
                # ── estructura y régimen ──
                'half_life_velas': _f(half_life_velas),
                'half_life_dias': _f(half_life_dias),
                'estacionariedad_precio': veredicto_precio,
                'estacionariedad_retornos': veredicto_retornos,
                'clustering_presente': bool(clustering_presente),
                'clustering_lb_p': _f(clustering_lb_p),
                'clustering_lag1': _f(clustering_lag1),
                # ── recuperación del Max Drawdown (tarjeta de la GUI) ──
                'dd_recuperacion_texto': duracion_str,
                'dd_recuperacion_rango': rango_fechas_str,
                'dd_recuperacion_velas': int(recovery_velas_max),
                'dd_recuperado': dd_recuperado,
                **_dependencia_kpi,
            },
            'por_horizonte': _kpi_por_horizonte,
        }

        # Mismo patrón .tmp + os.replace que el PDF: el archivo lo lee la GUI en
        # cuanto el proceso termina, y no debe encontrarse una escritura a medias.
        _plot_path = os.environ['GUI_PLOTDATA_OUTPUT']
        _plot_tmp = _plot_path + '.tmp'
        try:
            with open(_plot_tmp, 'wb') as f:
                pickle.dump(_PLOT, f, protocol=4)
                f.flush()
                os.fsync(f.fileno())
            for intento in range(5):
                try:
                    os.replace(_plot_tmp, _plot_path)
                    break
                except PermissionError:
                    if intento == 4:
                        raise
                    time.sleep(0.3)
            print(f"📊 Datos de graficos exportados: {_plot_path}")
        except Exception as e:
            # Un fallo aquí no debe invalidar el análisis: la GUI simplemente
            # caerá al visor de PDF como con los informes antiguos.
            print(f"⚠️  No se pudieron exportar los datos de graficos: {e}")
            traceback.print_exc()
    
    if 'GUI_METRICS_OUTPUT' not in os.environ:
        carpeta_contenedora = os.path.dirname(OUTPUT_PDF)
        subprocess.Popen(f'explorer "{carpeta_contenedora}"')

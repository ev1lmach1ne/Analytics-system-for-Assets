import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from scipy import stats
import os
import subprocess

# ── 1. CONFIGURACIÓN — solo cambiar estos valores ────────────────────────────
CONFIG = {
    'activo':     'CRYPTO',
    'nombre':     'BTC/USDT',
    'tf':         '1h',
    'input_path': r"D:\DATOS\Activos\Crypto\Limpiados\btc_1h_limpio.csv"
}

FACTORES = {
    'CRYPTO': {
        '1min':  {'anual': 525600, 'dia': 1440}, #24/7 abierto
        '5min':  {'anual': 105120, 'dia': 288},
        '15min': {'anual': 35040,  'dia': 96},
        '30min': {'anual': 17520,  'dia': 48},
        '1h':    {'anual': 8760,   'dia': 24},
        '4h':    {'anual': 2190,   'dia': 6},
        '1d':    {'anual': 365,    'dia': 1}
    },
    'FUTURO': {
        '1min':  {'anual': 362880, 'dia': 1440}, # Ajusta según mercado
        '1h':    {'anual': 6048,   'dia': 24},
        '1d':    {'anual': 365,    'dia': 1}
    },
    'STOCK': {
        '1min':  {'anual': 98280,  'dia': 390}, # 6.5 horas trading
        '1h':    {'anual': 1638,   'dia': 6.5},
        '1d':    {'anual': 252,    'dia': 1}
    }
}



# ── 2. VALIDACIÓN ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"ANÁLISIS DESCRIPTIVO — {CONFIG['nombre']} ({CONFIG['tf']})")
print(f"{'='*60}")
print("[1/5] Validando configuración...")
OUTPUT_PDF = CONFIG['input_path'].replace('.csv', f"_informe_{CONFIG['activo']}_{CONFIG['tf']}.pdf")

# ── 3. CARGA ──────────────────────────────────────────────────────────────────
print("[2/5] Cargando archivo CSV...")
df = pd.read_csv(CONFIG['input_path'], parse_dates=['timestamp']).set_index('timestamp')

# ── DETECCIÓN TEMPORAL UNIFICADA ─────────────────────────────────────────────
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
        # CV bajo  -> duración entre velas casi constante -> barras de tiempo fijo
        # CV alto  -> duración muy irregular -> barras de evento (TICKS/volumen/rango)
        tipo_muestreo = 'tiempo_fijo' if cv < 0.15 else 'evento'

        rango_total_seg  = (df.index[-1] - df.index[0]).total_seconds()
        rango_total_dias = rango_total_seg / 86400

        if rango_total_dias > 0:
            velas_por_dia  = len(df) / rango_total_dias
            velas_por_anio = velas_por_dia * 365

print(f"  Tipo de muestreo detectado: {tipo_muestreo}")
if velas_por_dia is not None:
    print(f"  Velas/día (real): {velas_por_dia:.2f} | Velas/año (real): {velas_por_anio:.1f}")
else:
    print("  ⚠️ Archivo basado en TICKS (sin temporalidad fiable): las métricas "
          "anualizadas y de recuperación temporal mostrarán 'N/A (TICKS)'.")

# ── 4. INTEGRIDAD ─────────────────────────────────────────────────────────────
print("[3/5] Verificando integridad de datos...")
if 'interpolado' not in df.columns: df['interpolado'] = 0
if 'anomalia'    not in df.columns: df['anomalia']    = 0
if 'ER'          not in df.columns: df['ER']          = 0.0

# ── 5. CÁLCULO DE RETORNOS ────────────────────────────────────────────────────
print("[4/5] Calculando métricas estadísticas...")
df['retorno'] = np.log(df['close'] / df['close'].shift(1))
df = df.dropna(subset=['retorno'])
r  = df['retorno']

# ── [AÑADIDO] CONFIGURACIÓN DE BLOQUES DIARIOS MULTI-TIMEFRAME ────────────────
# ==============================================================================
# Nos aseguramos de tener una serie limpia de retornos (quitando infinitos y NaNs)
r_limp = r[np.isfinite(r)].copy()

# Determinamos el tamaño del bloque para un día completo usando tus factores detectados.
# Si tus variables de detección dinámica fallan o dan None, usamos por defecto 24.
bloques_diarios = int(velas_por_dia) if (velas_por_dia is not None and velas_por_dia > 0) else 24

# Agrupamos los retornos en bloques de "un día real" y los sumamos.
# Al ser retornos logarítmicos, la suma matemática consolida perfectamente el periodo diario.
r_diario_real = r_limp.groupby(np.arange(len(r_limp)) // bloques_diarios).sum()
total_dias_muestra = len(r_diario_real)

print(f"   [CONFIG] Mapeo Multi-Timeframe: Agrupando en bloques de {bloques_diarios} velas por día.")
print(f"   [CONFIG] Total de días reales detectados en el histórico: {total_dias_muestra} días.")

# ── 6. MÉTRICAS ───────────────────────────────────────────────────────────────
# [MODIFICADO] Volatilidad y retornos anualizados/diarios ya no usan la tabla
# manual FACTORES; usan velas_por_dia / velas_por_anio calculados arriba a
# partir de los timestamps reales. Si no hay temporalidad fiable, quedan en None.
if velas_por_anio is not None:
    vol_anual  = r.std() * np.sqrt(velas_por_anio)
    vol_diaria = r.std() * np.sqrt(velas_por_dia)
    ret_anual  = r.mean() * velas_por_anio
    ret_diario = r.mean() * velas_por_dia
    sharpe     = ret_anual / vol_anual if vol_anual != 0 else None
    calmar_disponible = True
else:
    vol_anual = vol_diaria = ret_anual = ret_diario = sharpe = None
    calmar_disponible = False

stat_jb, p_jb = stats.jarque_bera(r)

# Max Drawdown (no depende del tiempo, siempre se calcula)
cum_returns = np.exp(r.cumsum())
peak        = cum_returns.cummax()
mdd         = ((cum_returns - peak) / peak).min()

# ── [AÑADIDO] Tiempo de recuperación, Calmar Ratio y VaR ─────────────────────

# --- Tiempo de recuperación (velas siempre; tiempo real solo si hay timestamps) ---
drawdown_series = (cum_returns - peak) / peak
en_drawdown      = drawdown_series < 0

bloques          = (en_drawdown != en_drawdown.shift()).cumsum()
duraciones_velas = en_drawdown.groupby(bloques).sum()
duraciones_velas = duraciones_velas[duraciones_velas > 0]

if len(duraciones_velas) > 0:
    bloque_max_id      = duraciones_velas.idxmax()
    recovery_velas_max = int(duraciones_velas.max())

    if es_datetime_valido:
        indices_bloque = en_drawdown[bloques == bloque_max_id].index

        # [AÑADIDO] Fecha del PICO (breakeven inicial): la última vela en máximo,
        # justo antes de empezar a caer. Es la posición anterior al primer punto
        # en drawdown de este bloque.
        pos_primera_en_dd = df.index.get_loc(indices_bloque[0])
        ts_pico = df.index[pos_primera_en_dd - 1] if pos_primera_en_dd > 0 else indices_bloque[0]

        # [AÑADIDO] Fecha de RECUPERACIÓN COMPLETA: la primera vela donde el
        # precio vuelve a superar (o igualar) el pico anterior, es decir, la
        # vela siguiente a la última registrada como "en drawdown" en este bloque.
        pos_ultima_en_dd = df.index.get_loc(indices_bloque[-1])
        if pos_ultima_en_dd + 1 < len(df.index):
            ts_recuperado = df.index[pos_ultima_en_dd + 1]
        else:
            ts_recuperado = indices_bloque[-1]  # aún no recuperado al final de los datos

        recovery_timedelta = ts_recuperado - ts_pico
        dias_rec    = recovery_timedelta.days
        horas_rec   = recovery_timedelta.seconds // 3600
        minutos_rec = (recovery_timedelta.seconds % 3600) // 60

        # [AÑADIDO] Si la recuperación alcanza o supera 1 año (365 días), se
        # descompone en años + el resto en días/horas/minutos (ej. "2 años
        # 328d 11h 0m"), conservando toda la precisión. Por debajo de 1 año
        # se mantiene solo d/h/m.
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

# --- [AÑADIDO] Drawdown medio: profundidad mínima media de cada episodio ---
# A diferencia del Max Drawdown (la peor caída de toda la historia), esto
# responde a "cuando el activo entra en drawdown, ¿hasta dónde suele caer
# típicamente antes de recuperarse?". Reutiliza los mismos bloques que el
# tiempo de recuperación: cada bloque es un episodio continuo en drawdown,
# delimitado por dos máximos históricos.
#
# Importante: NO se promedia drawdown_series completa (eso incluiría los
# tramos en máximo histórico, donde vale 0, y diluiría artificialmente el
# resultado). Se promedia solo el PUNTO MÁS PROFUNDO de cada episodio.
if en_drawdown.any():
    profundidad_por_episodio = drawdown_series[en_drawdown].groupby(bloques[en_drawdown]).min()
    drawdown_medio = profundidad_por_episodio.mean()
    num_episodios_dd = len(profundidad_por_episodio)
else:
    drawdown_medio = None
    num_episodios_dd = 0

# --- Calmar Ratio ---
if calmar_disponible and mdd != 0:
    calmar_ratio = ret_anual / abs(mdd)
else:
    calmar_ratio = None

# --- Value at Risk (VaR) — no depende del tiempo, siempre se calcula ---
var_95_hist = np.percentile(r, 5)
var_99_hist = np.percentile(r, 1)
z_95 = stats.norm.ppf(0.05)
z_99 = stats.norm.ppf(0.01)
var_95_param = r.mean() + z_95 * r.std()
var_99_param = r.mean() + z_99 * r.std()

# [AÑADIDO] Helpers de formato: si la métrica es None (no disponible por
# falta de temporalidad), se muestra un texto explicativo en vez de romper
# el script o mostrar un "N/A" ambiguo.
def fmt_pct(valor, decimales=2):
    return f"{valor*100:.{decimales}f}%" if valor is not None else "N/A (TICKS)"

def fmt_num(valor, decimales=4):
    return f"{valor:.{decimales}f}" if valor is not None else "N/A (TICKS)"

# ── CÁLCULO DE UMBRALES ADAPTATIVOS PARA EL DICTIONARY ────────────────────────
er_medio         = df['ER'].mean()
er_std           = df['ER'].std()

umbral_tendencia = min(0.95, er_medio + er_std)
umbral_ruido     = max(0.05, er_medio - er_std)

total_tendencia  = (df['ER'] > umbral_tendencia).sum()
total_ruido      = (df['ER'] < umbral_ruido).sum()
# ── CÁLCULO DE HURST ─────────────────────────────────────────────────────────────
total_tendencia_h = ((df['hurst'] > 0.58)).sum()
total_aleatorio_h  = ((df['hurst'] >= 0.52) & (df['hurst'] <= 0.58)).sum()
total_reversion_h  = ((df['hurst'] < 0.52)).sum()

# CÁLCULO DE VOLATILIDAD POR VENTANAS Y VOLATILIDAD HISTÓRICA (Rolling Volatility)
# USANDO "FACTORES", PARA DETERMINAR EL TIEMPO DE SESION DE CADA PRODUCTO

tipo_activo = CONFIG['activo']  
tf_actual = CONFIG['tf']

velas_dia = FACTORES[tipo_activo][tf_actual]['dia']
velas_anual = FACTORES[tipo_activo][tf_actual]['anual']

dias_ano = 365 if tipo_activo == 'CRYPTO' else 252
dias_trimestre = 90 if tipo_activo == 'CRYPTO' else 63

vol_historica_total = r.std() * np.sqrt(velas_anual)

# --- Cálculo de Volatividades Móviles ---
hv_7d  = r.rolling(window=int(7 * velas_dia)).std() * np.sqrt(velas_anual)
hv_30d = r.rolling(window=int(30 * velas_dia)).std() * np.sqrt(velas_anual)
hv_90d = r.rolling(window=int(dias_trimestre * velas_dia)).std() * np.sqrt(velas_anual)
hv_365d = r.rolling(window=int(dias_ano * velas_dia)).std() * np.sqrt(velas_anual)

val_hv_7d   = hv_7d.iloc[-1] if len(hv_7d) >= (7 * velas_dia) else None
val_hv_30d  = hv_30d.iloc[-1] if len(hv_30d) >= (30 * velas_dia) else None
val_hv_90d  = hv_90d.iloc[-1] if len(hv_90d) >= (dias_trimestre * velas_dia) else None
val_hv_365d = hv_365d.iloc[-1] if len(hv_365d) >= (dias_ano * velas_dia) else None

# --- Comparativa de Regímenes ---
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
    
    
# 📊 ESTRUCTURACIÓN Y PRESENTACIÓN DE MÉTRICAS FINANCIERAS

metricas = {
    '1. Información General y tipo de muestreo': {
        'Periodo': f"{df.index.min()} → {df.index.max()}" if es_datetime_valido else f"{len(df):,} ticks (archivo sin temporalidad)",
        'Tipo de muestreo': tipo_muestreo,
        'Total velas': f"{len(df):,}"
    },
    '2. Rendimiento y Retornos': {
        'Retorno anualizado': fmt_pct(ret_anual),
        'Media retorno': f"{r.mean()*100:.6f}%",
        'Mediana retorno': f"{r.median()*100:.6f}%",
        'Retornos positivos': f"{(r > 0).sum() / len(r) * 100:.2f}%",
        'Retornos negativos': f"{(r < 0).sum() / len(r) * 100:.2f}%"
    },
    '3b. Avanzada & Volatilidad Histórica': {
        'Volatilidad Histórica Total': f"{vol_historica_total*100:.2f}%",
        'HV 7d': f"{val_hv_7d*100:.2f}%" if val_hv_7d is not None else "N/A",
        'HV 30d': f"{val_hv_30d*100:.2f}%" if val_hv_30d is not None else "N/A",
        f'HV {dias_trimestre}d': f"{val_hv_90d*100:.2f}%" if val_hv_90d is not None else "N/A",
        f'HV {dias_ano}d': f"{val_hv_365d*100:.2f}%" if val_hv_365d is not None else "N/A",
        ' ': '',
        'Régimen Bull Market (SMA200)': str_bull,
        'Régimen Bear Market (SMA200)': str_bear
    },
    '3. Riesgo y Volatilidad': {
        'Volatilidad anualizada': fmt_pct(vol_anual),
        'Volatilidad diaria': fmt_pct(vol_diaria),
        'Desv. estandar': f"{r.std()*100:.6f}%",
        'Ratio Sharpe (Rf=0)': fmt_num(sharpe),
        'Calmar Ratio': fmt_num(calmar_ratio)
    },
    '4. Drawdown Analysis': {
        'Max Drawdown': f"{mdd*100:.2f}%",
        'Drawdown medio': f"{drawdown_medio*100:.2f}%" if drawdown_medio is not None else "N/A",
        'Episodios de drawdown': f"{num_episodios_dd:,}",
        'Tiempo recuperación (velas)': f"{recovery_velas_max:,}",
        'Tiempo recuperación (real)': recovery_str
    },
    '5. VaR, Campana and Risks': {
        'VaR 95% (histórico)': f"{var_95_hist*100:.4f}%",
        'VaR 99% (histórico)': f"{var_99_hist*100:.4f}%",
        'VaR 95% (paramétrico)': f"{var_95_param*100:.4f}%",
        'VaR 99% (paramétrico)': f"{var_99_param*100:.4f}%",
        'Minimo': f"{r.min()*100:.4f}%",
        'Maximo': f"{r.max()*100:.4f}%",
        'Skewness': f"{r.skew():.4f}",
        'Kurtosis': f"{r.kurtosis():.4f}",
        'Jarque-Bera stat': f"{stat_jb:.2f}",
        'Jarque-Bera p-value': f"{p_jb:.6f}",
        'Distribucion normal': f"{'NO (fat tails)' if p_jb < 0.05 else 'SI'}"
    },
'6. Ratio Eficiencia (ER) y Exponente de Hurst': {
        'ER medio': f"{df['ER'].mean():.4f}",
        'ER maximo': f"{df['ER'].max():.4f}",
        'ER minimo': f"{df['ER'].min():.4f}",
        'Periodos tendencia (ER>0.5)': f"{(df['ER'] > 0.5).sum():,}",
        'Paseo aleatorio (ER 0.3-0.5)': f"{((df['ER'] >= 0.3) & (df['ER'] <= 0.5)).sum():,} (Random Walk)",
        'Periodos ruido (ER<0.3)': f"{(df['ER'] < 0.3).sum():,}",
        
        # Un espacio en blanco estratégico para separar bloques
        ' ': '',
        
        'Hurst medio': f"{df['hurst'].mean():.4f}",
        'Hurst maximo': f"{df['hurst'].max():.4f}",
        'Hurst minimo': f"{df['hurst'].min():.4f}",
        'Periodos tendencia (H>0.58)': f"{total_tendencia_h:,}",
        'Paseo aleatorio (H 0.52-0.58)': f"{total_aleatorio_h:,}",
        'Periodos mean reversion (H<0.52)': f"{total_reversion_h:,}"
    },
}
# Output formateado y elegante en la terminal
print(f"\n{'═'*70}")
print(f"{'SISTEMA DE ANALÍTICA DE ACTIVOS - REPORTE DESCRIPTIVO':^70}")
print(f"{'═'*70}")

for categoria, items in metricas.items():
    print(f"\n ▶ {categoria}")
    print(f"  {'─' * (len(categoria) + 2)}")
    for metrica, valor in items.items():
        print(f"    {metrica:<35} : {valor}")

print(f"\n{'═'*70}\n")

# ── 7. PREPARAR MÁSCARAS DE RÉGIMEN ──────────────────────────────────────────
mask_tend  = df['ER'] > 0.45
mask_trans = (df['ER'] >= 0.30) & (df['ER'] <= 0.45)
mask_ruido = df['ER'] < 0.30

def color_regimen(er_val):
    if er_val > 0.45:   return '#1D9E75'
    elif er_val > 0.30: return '#888888'
    else:               return '#E24B4A'

# ── 8. GENERACIÓN PDF ─────────────────────────────────────────────────────────
print("Generando PDF...")

# ── CÁLCULO DE RENDIMIENTOS ESTACIONALES (Capitalización Compuesta) ──
df['factor_crecimiento'] = np.exp(df['retorno'])

mapeo_meses = {'January': 'Enero', 'February': 'Febrero', 'March': 'Marzo', 'April': 'Abril', 'May': 'Mayo', 'June': 'Junio', 'July': 'Julio', 'August': 'Agosto', 'September': 'Septiembre', 'October': 'Octubre', 'November': 'Noviembre', 'December': 'Diciembre'}
mapeo_dias = {'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles', 'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}

orden_meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
orden_dias  = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

# [MODIFICADO] La estacionalidad por mes/día solo tiene sentido con fechas
# reales del calendario. Si el archivo es de TICKS sin temporalidad fiable,
# se omite el cálculo (mes_stats / dia_stats quedan vacíos) en vez de fallar.
if es_datetime_valido:
    df['mes_nombre'] = df.index.strftime('%B').map(mapeo_meses)
    df['dia_nombre'] = df.index.strftime('%A').map(mapeo_dias)

    mes_stats = (df.groupby('mes_nombre')['factor_crecimiento'].prod() - 1) * 100
    dia_stats = (df.groupby('dia_nombre')['factor_crecimiento'].prod() - 1) * 100
    mes_stats = mes_stats.reindex(orden_meses)
    dia_stats = dia_stats.reindex(orden_dias)
else:
    mes_stats = pd.Series(dtype=float)
    dia_stats = pd.Series(dtype=float)

with PdfPages(OUTPUT_PDF) as pdf:

# ── PÁGINA 1 — Métricas Estructuradas  ──
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    fig.patch.set_facecolor('#0f0f0f')

    # Encabezados principales
    fig.text(0.5, 0.94,
             f"{CONFIG['nombre']} ({CONFIG['tf']}) — Informe de Analisis Descriptivo",
             ha='center', va='top', fontsize=16, fontweight='bold', color='white')
    fig.text(0.5, 0.89,
             f"Activo: {CONFIG['activo']} | Archivo: {os.path.basename(CONFIG['input_path'])}",
             ha='center', va='top', fontsize=9, color='#888780')

    # Vinculación exacta con tus títulos de categorías
    bloque_col1 = [
        '1. Información General y tipo de muestreo',
        '2. Rendimiento y Retornos',
        '3. Riesgo y Volatilidad'
    ]
    bloque_col2 = [
        '4. Drawdown Analysis',
        '5. VaR, Campana and Risks',
        '6. Efficiency Ratio (ER)'
    ]

    dy = 0.031  # Altura de cada línea de métrica

    # Definición de las categorías asignadas a la columna izquierda
    bloque_col1 = [
        '1. Información General y tipo de muestreo',
        '2. Rendimiento y Retornos',
        '3. Riesgo y Volatilidad',
        '3b. Avanzada & Volatilidad Histórica'
    ]

    dy = 0.025  # Factor de espaciado optimizado para que quepan todos los bloques

    # --- COLUMNA 1 (IZQUIERDA) ---
    y_current = 0.84
    for categoria in bloque_col1:
        if categoria in metricas:
            # Imprimir título de la categoría
            fig.text(0.04, y_current, categoria, fontsize=10, color='#ff9900', fontweight='bold', va='center')
            y_current -= 0.022
            
            # Imprimir sus métricas
            for idx, (k, v) in enumerate(metricas[categoria].items()):
                bg = '#161616' if idx % 2 == 0 else '#0d0d0d'
                fig.patches.append(plt.Rectangle((0.03, y_current - 0.010), 0.44, dy,
                                           transform=fig.transFigure, facecolor=bg, zorder=0))
                
                # Validación: Si la clave no está vacía, pintamos el texto y su valor
                if k.strip() != '':
                    fig.text(0.05, y_current, k, fontsize=9, color='#a0a0a0', va='center')
                    
                    # Ajuste dinámico de tamaño de fuente según longitud de cadena
                    f_size = 8 if k == 'Periodo' or 'Régimen' in k else 9
                    fig.text(0.21, y_current, v, fontsize=f_size, color='white', va='center', fontweight='bold')
                
                y_current -= dy
            y_current -= 0.018  # Espacio de separación entre bloques

    # --- COLUMNA 2 (DERECHA) ---
    y_current = 0.83
    for categoria in bloque_col2:
        if categoria in metricas:
            # Imprimir título de la categoría
            fig.text(0.53, y_current, categoria, fontsize=10, color='#ff9900', fontweight='bold', va='center')
            y_current -= 0.025
            
            # Imprimir sus métricas
            for idx, (k, v) in enumerate(metricas[categoria].items()):
                bg = '#161616' if idx % 2 == 0 else '#0d0d0d'
                fig.patches.append(plt.Rectangle((0.52, y_current - 0.011), 0.44, dy,
                                           transform=fig.transFigure, facecolor=bg, zorder=0))
                fig.text(0.54, y_current, k, fontsize=9, color='#a0a0a0', va='center')
                
                # AJUSTE: Si es el tiempo de recuperación real, bajamos la fuente a 7.5 debido a la fecha final
                f_size = 7.5 if k == 'Tiempo recuperación (real)' else 9
                # AJUSTE: Movido el valor a X=0.70 (antes 0.77) para expandirse sin salir del recuadro
                fig.text(0.70, y_current, v, fontsize=f_size, color='white', va='center', fontweight='bold')
                y_current -= dy
            y_current -= 0.025  # Espacio de separación entre bloques

    # Guardar y cerrar la página del PDF
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close()
    print("Generando página 1/5 — Métricas...")

    # ── PÁGINA 2 — Precio y Retornos ─────────────────────
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('#0f0f0f')
    gs  = gridspec.GridSpec(2, 1, hspace=0.35)

    eje_x = df.index if es_datetime_valido else np.arange(len(df))

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(eje_x, df['close'], color='#1D9E75', linewidth=0.5)
    ax1.set_facecolor('#111111')
    ax1.set_title(f"{CONFIG['nombre']} — Precio de Cierre ({CONFIG['tf']})",
                  color='white', fontsize=11)
    ax1.set_ylabel('Precio', color='#888780')
    ax1.set_xlabel('Tiempo' if es_datetime_valido else 'Nº de vela (TICKS)', color='#888780')
    ax1.tick_params(colors='#888780')
    ax1.grid(True, alpha=0.2, color='#444')
    for spine in ax1.spines.values(): spine.set_edgecolor('#333')

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(eje_x, r, color='#185FA5', linewidth=0.3, alpha=0.8)
    ax2.axhline(0, color='#E24B4A', linewidth=0.8, linestyle='--')
    ax2.set_facecolor('#111111')
    ax2.set_title(f"{CONFIG['nombre']} — Retornos Logaritmicos ({CONFIG['tf']})",
                  color='white', fontsize=11)
    ax2.set_ylabel('Retorno log', color='#888780')
    ax2.set_xlabel('Tiempo' if es_datetime_valido else 'Nº de vela (TICKS)', color='#888780')
    ax2.tick_params(colors='#888780')
    ax2.grid(True, alpha=0.2, color='#444')
    for spine in ax2.spines.values(): spine.set_edgecolor('#333')

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close()
    print("Generando página 2/5 — Precio y Retornos... ")

    # ── PÁGINA 3 — Análisis de Estacionalidad ────────
    # [MODIFICADO] Si el archivo no tiene temporalidad real (TICKS), esta
    # página muestra un aviso explicativo en vez de gráficos vacíos o un error.
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('#0f0f0f')

    if es_datetime_valido and len(mes_stats.dropna()) > 0:
        gs = gridspec.GridSpec(2, 1, hspace=0.4)

        ax1 = fig.add_subplot(gs[0])
        cols_m = ['#1D9E75' if x >= 0 else '#E24B4A' for x in mes_stats]
        bars1 = ax1.bar(mes_stats.index, mes_stats, color=cols_m, alpha=0.8)
        ax1.set_title('Retorno Acumulado por Mes (%)', color='white', fontsize=12)
        ax1.tick_params(colors='#888780'); ax1.grid(True, axis='y', alpha=0.2)
        ax1.set_facecolor('#111111')
        for bar in bars1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h, f'{h:.1f}%', ha='center', va='bottom' if h>0 else 'top', color='white', fontsize=8)

        ax2 = fig.add_subplot(gs[1])
        cols_d = ['#1D9E75' if x >= 0 else '#E24B4A' for x in dia_stats]
        bars2 = ax2.bar(dia_stats.index, dia_stats, color=cols_d, alpha=0.8)
        ax2.set_title('Retorno Acumulado por Día de la Semana (%)', color='white', fontsize=12)
        ax2.tick_params(colors='#888780'); ax2.grid(True, axis='y', alpha=0.2)
        ax2.set_facecolor('#111111')
        for bar in bars2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h, f'{h:.1f}%', ha='center', va='bottom' if h>0 else 'top', color='white', fontsize=8)
    else:
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.5,
                "Análisis de estacionalidad no disponible.\n\n"
                "El archivo está basado en TICKS (sin temporalidad de calendario fija),\n"
                "por lo que no es posible agrupar los retornos por mes o día de la semana.",
                ha='center', va='center', fontsize=12, color='#888780')

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close()
    print("Generando página 3/5 — Análisis de Estacionalidad...")

    # ── PÁGINA 4 — Precio coloreado por régimen ER ────────
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('#0f0f0f')
    gs  = gridspec.GridSpec(2, 1, hspace=0.4, height_ratios=[3, 1])

    ax4 = fig.add_subplot(gs[0])
    ax4.set_facecolor('#111111')
    ax4.set_title(f"{CONFIG['activo']} — Precio coloreado por Regimen ER ({CONFIG['tf']})", color='white', fontsize=11)
    ax4.set_ylabel('Precio', color='#888780')
    ax4.set_xlabel('Tiempo' if es_datetime_valido else 'Nº de vela (TICKS)', color='#888780')
    ax4.tick_params(colors='#888780')
    ax4.grid(True, alpha=0.2, color='#444')
    for spine in ax4.spines.values(): spine.set_edgecolor('#333')

    # [MODIFICADO] El eje X usa timestamp real si existe; si no, índice secuencial
    eje_x_num = df.index.astype(np.int64) if es_datetime_valido else np.arange(len(df))
    points = np.array([eje_x_num, df['close'].values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = [color_regimen(er) for er in df['ER'].values]
    lc = LineCollection(segments, colors=colors, linewidth=0.6, alpha=0.8)
    ax4.add_collection(lc)
    ax4.autoscale()

    legend_elements = [
        Line2D([0], [0], color='#1D9E75', linewidth=2, label='Tendencia (ER>0.45)'),
        Line2D([0], [0], color='#888888', linewidth=2, label='Transicion (0.30-0.45)'),
        Line2D([0], [0], color='#E24B4A', linewidth=2, label='Ruido (ER<0.30)'),
    ]
    ax4.legend(handles=legend_elements, facecolor='#222', labelcolor='white', fontsize=8, loc='upper right')

    ax5 = fig.add_subplot(gs[1])
    ax5.set_title("Histórico ER Acumulado", color='white', fontsize=10, pad=10)
    ax5.fill_between(eje_x, df['ER'], color='#BA7517', alpha=0.2, linewidth=0)
    ax5.plot(eje_x, df['ER'], color='#BA7517', linewidth=0.3, alpha=0.5)

    er_suavizado = df['ER'].rolling(200).mean()
    ax5.plot(eje_x, er_suavizado, color='white', linewidth=1.0, label='Tendencia (SMA 200)')
    media_er = df['ER'].mean()
    ax5.axhline(media_er, color='#185FA5', linewidth=1.2, linestyle='-', label=f'Media: {media_er:.2f}')
    ax5.axhline(0.45, color='#1D9E75', linewidth=0.8, linestyle='--', label='Ruido Bajo (ER > 0.45)')
    ax5.axhline(0.30, color='#E24B4A', linewidth=0.8, linestyle='--', label='Ruido Alto (ER < 0.30)')

    ax5.set_facecolor('#111111')
    ax5.set_ylabel('ER', color='#888780')
    ax5.set_ylim(0, 1)
    ax5.tick_params(colors='#888780')
    ax5.grid(True, alpha=0.2, color='#444')
    ax5.legend(loc='upper right', facecolor='#222', labelcolor='white', fontsize=7, ncol=2)

    for spine in ax5.spines.values(): spine.set_edgecolor('#333')

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close()
    print("Generando página 4/5 — Precio por régimen ER...")

# ── PÁGINA 5 — Análisis de Riesgo y Value at Risk (VaR) ────────
    try:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('#0f0f0f')
        # Cambiamos a 3 filas con ratios iguales y un hspace ajustado para que queden perfectos
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.65)

        sigmas = [1, 2, 3]
        colores_sigmas = ['#1D9E75', '#BA7517', '#E24B4A'] # Verde, Naranja, Rojo

        def dibujar_campana_extendida(ax, media, std, titulo, es_anual=False):
            ax.set_facecolor('#111111')
            
            # Generamos la curva de distribución teórica proyectada suave
            x = np.linspace(media - 4*std, media + 4*std, 200)
            y = stats.norm.pdf(x, media, std)
            ax.plot(x, y, color='#185FA5', linewidth=2, label='Distribución Proyectada')
            
            # Dibujar y etiquetar niveles sigma
            for s in sigmas:
                ax.fill_between(x, y, where=(x >= media-s*std) & (x <= media+s*std), 
                                color=colores_sigmas[s-1], alpha=0.15)
                
                for lado in [-1, 1]:
                    val = media + (lado * s * std)
                    ax.axvline(val, color=colores_sigmas[s-1], linestyle=':', alpha=0.5)
                    
                    if not es_anual:
                        # --- GRÁFICO 1: FRECUENCIA EN DÍAS REALES ---
                        if lado == -1:
                            prob_evento = np.mean(r_diario_real <= val)
                        else:
                            prob_evento = np.mean(r_diario_real >= val)
                            
                        if prob_evento > 0:
                            inverso = 1 / prob_evento
                            texto_frecuencia = f"1 c/{inverso:.1f}d" if inverso >= 1 else f"{prob_evento:.1f}x/d"
                        else:
                            texto_frecuencia = "No reg."
                    else:
                        # --- GRÁFICO 2: FRECUENCIA EN AÑOS REALES ---
                        factor_regimen_dias = 365 if CONFIG['activo'] == 'CRYPTO' else 252
                        umbral_corte_desescalado = val / np.sqrt(factor_regimen_dias)
                        
                        if lado == -1:
                            prob_evento = np.mean(r_diario_real <= umbral_corte_desescalado)
                        else:
                            prob_evento = np.mean(r_diario_real >= umbral_corte_desescalado)
                            
                        if prob_evento > 0:
                            velas_espera_dias = 1 / prob_evento
                            unidades_anos = velas_espera_dias / factor_regimen_dias
                            texto_frecuencia = f"1 c/{unidades_anos:.1f}añ" if unidades_anos >= 1 else f"{int(1/unidades_anos)}x/añ"
                        else:
                            texto_frecuencia = "No reg."

                    val_mostrado = max(val, -0.999) if lado == -1 else val
                    
                    nivel_escalon = 0.05 if s % 2 != 0 else 0.14
                    etiqueta = f"{lado*s}σ: {val_mostrado:.1%}\n({texto_frecuencia})" 
                    
                    ax.text(val, -max(y) * nivel_escalon, etiqueta, color=colores_sigmas[s-1], fontsize=6, 
                            ha='center', fontweight='bold', bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=1))

            ax.axvline(media, color='white', linestyle='--', linewidth=1.2, alpha=0.5)
            ax.text(media, -max(y)*0.05, f"μ: {media:.2%}", color='white', fontsize=7,
                    ha='center', fontweight='bold', bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=1))

            ax.set_ylim(-max(y)*0.24, max(y)*1.15)
            ax.set_title(titulo, color='white', fontsize=10)
            ax.tick_params(colors='#888780', labelsize=8)
            ax.grid(True, alpha=0.03)
            
            skew_local = stats.skew(r_diario_real)
            kurt_local = stats.kurtosis(r_diario_real, fisher=True)
            texto_stats = f"Skewness (Muestra Diaria): {skew_local:.2f} | Kurtosis (Muestra Diaria): {kurt_local:.2f}"
            ax.text(0.015, 0.93, texto_stats, transform=ax.transAxes, color='#888780', fontsize=7, verticalalignment='top')
            ax.legend(facecolor='#222', labelcolor='white', fontsize=7, loc='upper right')

        # 1. Gráfico Diario
        ax1 = fig.add_subplot(gs[0])
        dibujar_campana_extendida(ax1, ret_diario, vol_diaria, f"Proyección de Retorno Diario (Vol Proyectada: {vol_diaria:.2%})", es_anual=False)

        # 2. Gráfico Anual
        ax2 = fig.add_subplot(gs[1])
        dibujar_campana_extendida(ax2, ret_anual, vol_anual, f"Proyección de Retorno Anual (Vol Proyectada: {vol_anual:.2%})", es_anual=True)

        # 3. [REINCORPORADO] Gráfico de Análisis de Muestra y Value at Risk (VaR)
        ax3 = fig.add_subplot(gs[2])
        ax3.set_facecolor('#111111')
        
        # Para la curva base del VaR usamos los parámetros de la muestra diaria agrupada
        mu_var = r_diario_real.mean()
        std_var = r_diario_real.std()
        x_v = np.linspace(mu_var - 4*std_var, mu_var + 4*std_var, 200)
        y_v = stats.norm.pdf(x_v, mu_var, std_var)
        
        ax3.plot(x_v, y_v, color='#555555', linewidth=1.5, linestyle='--', label='Perfil de Riesgo Diario')
        
        # Mapeo de configuraciones del VaR (usando las variables ya calculadas en tu script principal)
        # Nota: Multiplicamos por la escala diaria si tus variables originales venían a nivel de vela corta, 
        # o las dejamos nativas si usas los percentiles de la muestra.
        lista_vars = [
            {'val': var_95_hist,  'col': '#BA7517', 'ls': '-',  'lbl': 'VaR 95% Hist', 'esc': 0.05},
            {'val': var_99_hist,  'col': '#E24B4A', 'ls': '-',  'lbl': 'VaR 99% Hist', 'esc': 0.14},
            {'val': var_95_param, 'col': '#1D9E75', 'ls': ':',  'lbl': 'VaR 95% Par',  'esc': 0.05},
            {'val': var_99_param, 'col': '#A04BE2', 'ls': ':',  'lbl': 'VaR 99% Par',  'esc': 0.14}
        ]
        
        # Dibujamos las zonas de estrés y las líneas de corte del VaR limpias
        ax3.fill_between(x_v, y_v, where=(x_v <= mu_var), color='#E24B4A', alpha=0.08)
        
        for v in lista_vars:
            if v['val'] is not None:
                # Línea vertical totalmente nítida
                ax3.axvline(v['val'], color=v['col'], linestyle=v['ls'], linewidth=1.2, alpha=0.7)
                
                # Rotulado exclusivo en la base del gráfico con su color correspondiente
                etiqueta_var = f"{v['lbl']}\n{v['val']:.2%}"
                ax3.text(v['val'], -max(y_v) * v['esc'], etiqueta_var, color=v['col'], fontsize=6,
                         ha='center', fontweight='bold', bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=1))

        # Media de referencia en el gráfico de VaR
        ax3.axvline(mu_var, color='white', linestyle='--', linewidth=1, alpha=0.4)
        ax3.text(mu_var, -max(y_v)*0.05, f"μ: {mu_var:.2%}", color='white', fontsize=6.5,
                 ha='center', fontweight='bold', bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=1))

        ax3.set_ylim(-max(y_v)*0.24, max(y_v)*1.15)
        ax3.set_title("Análisis Estocástico: Modelado del Value at Risk (VaR) Diario", color='white', fontsize=10)
        ax3.tick_params(colors='#888780', labelsize=8)
        ax3.grid(True, alpha=0.03)
        ax3.legend(facecolor='#222', labelcolor='white', fontsize=7, loc='upper right')

        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close()
        print("Generando página 5/5 — Análisis de Riesgos Estructura Completa (3 Campanas)")
    except Exception as e:
        print(f"ERROR EN PÁGINA 5: {e}")

# ── FINALIZACIÓN ──────────────────────────────────────────────────────────────
print(f"{'='*60}")
print(f"✅ Analisis completado con exito.")
print(f"📁 Guardado en: {OUTPUT_PDF}")
print(f"{'='*60}")

carpeta_contenedora = os.path.dirname(OUTPUT_PDF)
subprocess.Popen(f'explorer "{carpeta_contenedora}"')
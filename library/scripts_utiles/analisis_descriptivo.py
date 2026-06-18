import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
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
try:
    FACTOR_ANUAL = FACTORES[CONFIG['activo']][CONFIG['tf']]
    OUTPUT_PDF   = CONFIG['input_path'].replace('.csv', f'_informe_{CONFIG["activo"]}_{CONFIG["tf"]}.pdf')
except KeyError:
    raise ValueError(f"Combinacion {CONFIG['activo']} + {CONFIG['tf']} no existe en FACTORES.")

# ── 3. CARGA ──────────────────────────────────────────────────────────────────
print("[2/5] Cargando archivo CSV...")
df = pd.read_csv(CONFIG['input_path'], parse_dates=['timestamp']).set_index('timestamp')

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

# ── 6. MÉTRICAS ───────────────────────────────────────────────────────────────
# Obtenemos los factores del nuevo diccionario
f_anual = FACTORES[CONFIG['activo']][CONFIG['tf']]['anual']
f_dia   = FACTORES[CONFIG['activo']][CONFIG['tf']]['dia']

# Volatilidad (anual y diaria)
vol_anual  = r.std() * np.sqrt(f_anual)
vol_diaria = r.std() * np.sqrt(f_dia)

# Retornos
ret_anual  = r.mean() * f_anual
ret_diario = r.mean() * f_dia

# Ratio Sharpe y Normalidad
sharpe        = ret_anual / vol_anual
stat_jb, p_jb = stats.jarque_bera(r) 

# Max Drawdown
cum_returns = np.exp(r.cumsum())
peak        = cum_returns.cummax()
mdd         = ((cum_returns - peak) / peak).min()

metricas = {
    'Periodo':                f"{df.index.min().date()} → {df.index.max().date()}",
    'Total velas':            f"{len(df):,}",
    'Media retorno':          f"{r.mean()*100:.6f}%",
    'Mediana retorno':        f"{r.median()*100:.6f}%",
    'Desv. estandar':         f"{r.std()*100:.6f}%",
    'Max Drawdown':           f"{mdd*100:.2f}%",
    'Minimo':                 f"{r.min()*100:.4f}%",
    'Maximo':                 f"{r.max()*100:.4f}%",
    'Skewness':               f"{r.skew():.4f}",
    'Kurtosis':               f"{r.kurtosis():.4f}",
    'Retornos positivos':     f"{(r > 0).sum() / len(r) * 100:.2f}%",
    'Retornos negativos':     f"{(r < 0).sum() / len(r) * 100:.2f}%",
    'Retorno anualizado':     f"{ret_anual*100:.2f}%",
    'Volatilidad anualizada': f"{vol_anual*100:.2f}%",
    'Volatilidad diaria':     f"{vol_diaria*100:.2f}%",
    'Ratio Sharpe (Rf=0)':    f"{sharpe:.4f}",
    'ER medio':               f"{df['ER'].mean():.4f}",
    'ER maximo':              f"{df['ER'].max():.4f}",
    'ER minimo':              f"{df['ER'].min():.4f}",
    'Periodos tendencia (ER>0.5)': f"{(df['ER'] > 0.5).sum():,}",
    'Periodos ruido (ER<0.3)':     f"{(df['ER'] < 0.3).sum():,}",
    'Jarque-Bera stat':       f"{stat_jb:.2f}",
    'Jarque-Bera p-value':    f"{p_jb:.6f}",
    'Distribucion normal':    f"{'NO (fat tails)' if p_jb < 0.05 else 'SI'}",
}

# Output terminal
print(f"\n{'─'*60}")
for k, v in metricas.items():
    print(f"  {k:<35} {v}")
print(f"{'─'*60}")

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

# ── CÁLCULO DE RENDIMIENTOS ESTACIONALES (Acumulado) ──────────────────────────
# Convertir retornos log a factores de crecimiento porcentual real
df['retorno_pct'] = np.exp(df['retorno']) - 1

# Mapeo de nombres al español
mapeo_meses = {
    'January': 'Enero', 'February': 'Febrero', 'March': 'Marzo', 'April': 'Abril',
    'May': 'Mayo', 'June': 'Junio', 'July': 'Julio', 'August': 'Agosto',
    'September': 'Septiembre', 'October': 'Octubre', 'November': 'Noviembre', 'December': 'Diciembre'
}
mapeo_dias = {
    'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
    'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
}

# Creación de columnas temporales
df['mes_nombre'] = df.index.strftime('%B').map(mapeo_meses)
df['dia_nombre'] = df.index.strftime('%A').map(mapeo_dias)

# Agrupación y suma acumulativa (transformada a %)
orden_meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
orden_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

# ── CÁLCULO DE RENDIMIENTOS ESTACIONALES (Capitalización Compuesta) ──
# Convertir retornos log a factores de crecimiento (1 + r%)
df['factor_crecimiento'] = np.exp(df['retorno'])

# Mapeo y columnas (igual que antes)
mapeo_meses = {'January': 'Enero', 'February': 'Febrero', 'March': 'Marzo', 'April': 'Abril', 'May': 'Mayo', 'June': 'Junio', 'July': 'Julio', 'August': 'Agosto', 'September': 'Septiembre', 'October': 'Octubre', 'November': 'Noviembre', 'December': 'Diciembre'}
mapeo_dias = {'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles', 'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}

df['mes_nombre'] = df.index.strftime('%B').map(mapeo_meses)
df['dia_nombre'] = df.index.strftime('%A').map(mapeo_dias)

# Agrupar mediante el producto de los factores y restar 1 para obtener el % acumulado real
mes_stats = (df.groupby('mes_nombre')['factor_crecimiento'].prod() - 1) * 100
dia_stats = (df.groupby('dia_nombre')['factor_crecimiento'].prod() - 1) * 100

# Orden cronológico
orden_meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
orden_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

mes_stats = mes_stats.reindex(orden_meses)
dia_stats = dia_stats.reindex(orden_dias)

with PdfPages(OUTPUT_PDF) as pdf:

    # ── PÁGINA 1 — Métricas ───────────────────────────────
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    fig.patch.set_facecolor('#0f0f0f')
    fig.text(0.5, 0.95,
             f"{CONFIG['nombre']} ({CONFIG['tf']}) — Informe de Analisis Descriptivo",
             ha='center', va='top', fontsize=16, fontweight='bold', color='white')
    fig.text(0.5, 0.90,
             f"Activo: {CONFIG['activo']} | Archivo: {os.path.basename(CONFIG['input_path'])}",
             ha='center', va='top', fontsize=9, color='#888780')

    items   = list(metricas.items())
    mitad   = len(items) // 2
    col1    = items[:mitad]
    col2    = items[mitad:]
    y_start = 0.83
    dy      = 0.034

    for i, (k, v) in enumerate(col1):
        y  = y_start - i * dy
        bg = '#1a1a1a' if i % 2 == 0 else '#111111'
        fig.patches.append(plt.Rectangle((0.03, y-0.012), 0.44, dy,
                           transform=fig.transFigure, facecolor=bg, zorder=0))
        fig.text(0.05, y, k, fontsize=9, color='#888780', va='center')
        fig.text(0.28, y, v, fontsize=9, color='white', va='center', fontweight='bold')

    for i, (k, v) in enumerate(col2):
        y  = y_start - i * dy
        bg = '#1a1a1a' if i % 2 == 0 else '#111111'
        fig.patches.append(plt.Rectangle((0.52, y-0.012), 0.44, dy,
                           transform=fig.transFigure, facecolor=bg, zorder=0))
        fig.text(0.54, y, k, fontsize=9, color='#888780', va='center')
        fig.text(0.77, y, v, fontsize=9, color='white', va='center', fontweight='bold')

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close()
    print("Generando página 1/5 — Métricas...")

    # ── PÁGINA 2 — Precio y Retornos ─────────────────────
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('#0f0f0f')
    gs  = gridspec.GridSpec(2, 1, hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df['close'], color='#1D9E75', linewidth=0.5)
    ax1.set_facecolor('#111111')
    ax1.set_title(f"{CONFIG['nombre']} — Precio de Cierre ({CONFIG['tf']})",
                  color='white', fontsize=11)
    ax1.set_ylabel('Precio', color='#888780')
    ax1.tick_params(colors='#888780')
    ax1.grid(True, alpha=0.2, color='#444')
    for spine in ax1.spines.values(): spine.set_edgecolor('#333')

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(df.index, r, color='#185FA5', linewidth=0.3, alpha=0.8)
    ax2.axhline(0, color='#E24B4A', linewidth=0.8, linestyle='--')
    ax2.set_facecolor('#111111')
    ax2.set_title(f"{CONFIG['nombre']} — Retornos Logaritmicos ({CONFIG['tf']})",
                  color='white', fontsize=11)
    ax2.set_ylabel('Retorno log', color='#888780')
    ax2.tick_params(colors='#888780')
    ax2.grid(True, alpha=0.2, color='#444')
    for spine in ax2.spines.values(): spine.set_edgecolor('#333')

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close()
    print("Generando página 2/5 — Precio y Retornos... ")

    # ── PÁGINA 3 — Análisis de Estacionalidad ────────
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('#0f0f0f')
    gs = gridspec.GridSpec(2, 1, hspace=0.4)

    # Gráfico Mensual Acumulado
    ax1 = fig.add_subplot(gs[0])
    cols_m = ['#1D9E75' if x >= 0 else '#E24B4A' for x in mes_stats]
    bars1 = ax1.bar(mes_stats.index, mes_stats, color=cols_m, alpha=0.8)
    ax1.set_title('Retorno Acumulado por Mes (%)', color='white', fontsize=12)
    ax1.tick_params(colors='#888780'); ax1.grid(True, axis='y', alpha=0.2)
    ax1.set_facecolor('#111111')
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h, f'{h:.1f}%', ha='center', va='bottom' if h>0 else 'top', color='white', fontsize=8)

    # Gráfico Semanal Acumulado
    ax2 = fig.add_subplot(gs[1])
    cols_d = ['#1D9E75' if x >= 0 else '#E24B4A' for x in dia_stats]
    bars2 = ax2.bar(dia_stats.index, dia_stats, color=cols_d, alpha=0.8)
    ax2.set_title('Retorno Acumulado por Día de la Semana (%)', color='white', fontsize=12)
    ax2.tick_params(colors='#888780'); ax2.grid(True, axis='y', alpha=0.2)
    ax2.set_facecolor('#111111')
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h, f'{h:.1f}%', ha='center', va='bottom' if h>0 else 'top', color='white', fontsize=8)

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close()
    print("Generando página 3/5 — Análisis de Estacionalidad...")

    # ── PÁGINA 4 — Precio coloreado por régimen ER (OPTIMIZADO) ────────
    from matplotlib.collections import LineCollection # Asegúrate de que esto esté al inicio del archivo

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('#0f0f0f')
    gs  = gridspec.GridSpec(2, 1, hspace=0.4, height_ratios=[3, 1])

    # Gráfico superior: Precio
    ax4 = fig.add_subplot(gs[0])
    ax4.set_facecolor('#111111')
    ax4.set_title(f"{CONFIG['activo']} — Precio coloreado por Regimen ER ({CONFIG['tf']})", color='white', fontsize=11)
    ax4.set_ylabel('Precio', color='#888780')
    ax4.tick_params(colors='#888780')
    ax4.grid(True, alpha=0.2, color='#444')
    for spine in ax4.spines.values(): spine.set_edgecolor('#333')

    # OPTIMIZACIÓN: LineCollection en lugar de un bucle for
    points = np.array([df.index.astype(np.int64), df['close'].values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = [color_regimen(er) for er in df['ER'].values]
    lc = LineCollection(segments, colors=colors, linewidth=0.6, alpha=0.8)
    ax4.add_collection(lc)
    ax4.autoscale() # Ajusta automáticamente los límites

    legend_elements = [
        Line2D([0], [0], color='#1D9E75', linewidth=2, label='Tendencia (ER>0.45)'),
        Line2D([0], [0], color='#888888', linewidth=2, label='Transicion (0.30-0.45)'),
        Line2D([0], [0], color='#E24B4A', linewidth=2, label='Ruido (ER<0.30)'),
    ]
    # LE FALTABA loc='upper right' para no ralentizar el proceso
    ax4.legend(handles=legend_elements, facecolor='#222', labelcolor='white', fontsize=8, loc='upper right')

    # Gráfico inferior: Histórico ER (Mismo que tenías, optimizado)
    ax5 = fig.add_subplot(gs[1])
    ax5.set_title("Histórico ER Acumulado", color='white', fontsize=10, pad=10)
    ax5.fill_between(df.index, df['ER'], color='#BA7517', alpha=0.2, linewidth=0)
    ax5.plot(df.index, df['ER'], color='#BA7517', linewidth=0.3, alpha=0.5)
    
    er_suavizado = df['ER'].rolling(200).mean()
    ax5.plot(df.index, er_suavizado, color='white', linewidth=1.0, label='Tendencia (SMA 200)')
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
    print("Generando página 4/5 — Precio por régimen ER... ✅")

    # ── PÁGINA 5 — Análisis de Riesgo: Diario vs Anual ────────
    try:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('#0f0f0f')
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.4)

        # Configuraciones de desviaciones: 1σ, 2σ, 3σ
        sigmas = [1, 2, 3]
        colores_sigmas = ['#1D9E75', '#BA7517', '#E24B4A'] # Verde, Naranja, Rojo

        def dibujar_campana_extendida(ax, media, std, titulo):
            ax.set_facecolor('#111111')
            # Ajustamos el rango de x
            x = np.linspace(media - 4*std, media + 4*std, 200)
            y = stats.norm.pdf(x, media, std)
            ax.plot(x, y, color='#185FA5', linewidth=2, label='Distribución Teórica')
            
            # Dibujar y etiquetar niveles sigma
            for s in sigmas:
                # Sombreado de las zonas
                ax.fill_between(x, y, where=(x >= media-s*std) & (x <= media+s*std), 
                                color=colores_sigmas[s-1], alpha=0.15)
                
                # Etiquetas
                for lado in [-1, 1]:
                    val = media + (lado * s * std)
                    ax.axvline(val, color=colores_sigmas[s-1], linestyle=':', alpha=0.6)
                    
                    # Limitamos visualmente el retorno negativo a -99.9%
                    val_mostrado = max(val, -0.999) if lado == -1 else val
                    
                    etiqueta = f"{lado*s}σ\n({val_mostrado:.2%})" 
                    ax.text(val, max(y)*0.75, etiqueta, color='white', fontsize=7, 
                            ha='center', fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

            ax.axvline(media, color='white', linestyle='--', linewidth=1.2, label=f'Media: {media:.2%}')
            ax.set_title(titulo, color='white', fontsize=12)
            ax.tick_params(colors='#888780')
            ax.grid(True, alpha=0.1)
            ax.legend(facecolor='#222', labelcolor='white', fontsize=8, loc='upper right')

        # 1. Gráfico Diario
        ax1 = fig.add_subplot(gs[0])
        dibujar_campana_extendida(ax1, ret_diario, vol_diaria, f"Retorno Diario (Vol: {vol_diaria:.2%})")

        # 2. Gráfico Anual
        ax2 = fig.add_subplot(gs[1])
        dibujar_campana_extendida(ax2, ret_anual, vol_anual, f"Retorno Anual (Vol: {vol_anual:.2%})")

        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close()
        print("Generando página 5/5 — Análisis de Riesgos ")
    except Exception as e:
        print(f"ERROR EN PÁGINA 5: {e}")

# ── FINALIZACIÓN ──────────────────────────────────────────────────────────────
print(f"{'='*60}")
print(f"✅ Analisis completado con exito.")
print(f"📁 Guardado en: {OUTPUT_PDF}")
print(f"{'='*60}")

carpeta_contenedora = os.path.dirname(OUTPUT_PDF)
subprocess.Popen(f'explorer "{carpeta_contenedora}"')
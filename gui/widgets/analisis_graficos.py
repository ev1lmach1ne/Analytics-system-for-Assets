"""
Sub-pestaña «Gráficos» del Analizador, dibujada nativamente en la GUI.

Sustituye al visor de páginas de PDF: cada figura que antes era una hoja A4 del
informe pasa a ser un canvas de matplotlib embebido, con el tema de la
aplicación y su barra de zoom/pan/guardar.

Los datos NO se recalculan aquí. El script de análisis
(library/scripts_utiles/analisis_descriptivo.py) exporta, junto al PDF, un
bundle con los arrays exactos que dibujó en cada página; este módulo solo los
pinta. Así ambas vistas muestran siempre lo mismo y abrir la pestaña es
instantáneo.

Las secciones se agrupan en pestañas temáticas y se dibujan de forma perezosa:
solo se pintan los canvas de la pestaña que el usuario está mirando.
"""
import numpy as np

from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
                             QTabWidget, QScrollArea, QGroupBox, QTableWidget,
                             QTableWidgetItem, QHeaderView, QAbstractItemView,
                             QApplication)

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.dates import date2num, AutoDateLocator, ConciseDateFormatter

from gui.widgets.plot_common import (
    make_canvas, style_ax, ax_placeholder, leyenda, icono_ayuda, no_crash,
    BarraGrafico, fmt, agregar_crosshair, eje_fechas, FIG_BG, AX_FG, GRID_C,
    VERDE, ROJO, GRIS, AMBAR, AZUL, NARANJA, TEXTO_TENUE, ER_TENDENCIA,
    ER_TRANSICION, ER_RUIDO)

ESTILO = """
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QGroupBox {
    background-color: #141e30; border: 1px solid #253a60;
    border-radius: 6px; margin-top: 0px; padding: 10px;
}
QLabel#titulo { color: #4fc3f7; font-size: 12px; font-weight: bold; }
QLabel#aviso { color: #5a7a9a; font-size: 11px; }
QTabWidget::pane { background-color: #0d1424; border: 1px solid #253a60; border-top: none; }
QTabBar { background-color: #141e30; border: none; }
QTabBar::tab {
    background-color: #141e30; color: #5a7a9a; padding: 6px 16px;
    border: none; border-right: 1px solid #253a60; font-size: 11px;
}
QTabBar::tab:selected { background-color: #0d1424; color: #4fc3f7; font-weight: bold; }
QTableWidget {
    background-color: #0d1424; color: #c8d6e5; gridline-color: #253a60;
    border: 1px solid #253a60; font-size: 11px;
}
QHeaderView::section {
    background-color: #1a2a45; color: #8fb3d9; border: none;
    border-right: 1px solid #253a60; padding: 4px; font-size: 10px; font-weight: bold;
}
"""

_SIGMA_COLORES = {1: VERDE, 2: '#BA7517', 3: ROJO}
_DIAS = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
_SESIONES = [(0, 'Tokio', ROJO), (8, 'Londres', AZUL), (13, 'NY', '#d29922')]


# ══════════════════════════════════════════════════════════════════
#  helpers de dibujo
# ══════════════════════════════════════════════════════════════════
def _x(bundle_x):
    """El eje X llega como datetime64[ns] o como enteros; matplotlib dibuja
    ambos, pero date2num hace falta cuando se mezcla con LineCollection."""
    arr = np.asarray(bundle_x)
    return arr


# el helper vive en plot_common (lo comparten este módulo y las tarjetas de
# tab_patrones); se mantiene el nombre local para no tocar sus ~20 llamadas
_eje_fechas = eje_fechas


def _campana(ax, media, std, sigmas, sufijo_freq, titulo, vars_=None):
    """Campana normal con bandas ±1/2/3σ (o áreas de VaR) y la frecuencia
    empírica de cada nivel — la misma figura que dibuja el PDF."""
    from scipy import stats as _st
    style_ax(ax)
    x = np.linspace(media - 4 * std, media + 4 * std, 200)
    y = _st.norm.pdf(x, media, std)
    ax.plot(x, y, color='#185FA5', linewidth=1.6, label='Distribución proyectada')
    ax.axvline(media, color='#e6edf3', linestyle='--', linewidth=1, alpha=0.5)
    ymax = float(np.max(y)) if len(y) else 1.0

    if vars_ is None:
        for s in (1, 2, 3):
            color = _SIGMA_COLORES[s]
            ax.fill_between(x, y, where=(x >= media - s * std) & (x <= media + s * std),
                            color=color, alpha=0.10)
            for lado in (-1, 1):
                info = sigmas.get(str(lado * s))
                if info is None:
                    continue
                val = info['val']
                ax.axvline(val, color=color, linestyle=':', alpha=0.45)
                prob = info['prob']
                txt = f"1 c/{1/prob:.1f}{sufijo_freq}" if prob > 0 else "No reg."
                ax.text(val, -ymax * (0.06 if s % 2 else 0.16),
                        f"{lado*s}σ: {val:.2%}\n({txt})",
                        color=color, fontsize=5.5, ha='center', fontweight='bold',
                        bbox=dict(facecolor='#0d1424', alpha=0.85, edgecolor='none', pad=0.6))
    else:
        for i, (clave, color) in enumerate((('0.95', '#BA7517'), ('0.99', ROJO))):
            info = vars_.get(clave)
            if info is None:
                continue
            val = info['val']
            ax.fill_between(x, y, where=(x <= val), color=color, alpha=0.18)
            ax.axvline(val, color=color, linewidth=1.2, alpha=0.7)
            prob = info['prob']
            txt = f"1 c/{1/prob:.1f}{sufijo_freq}" if prob > 0 else "No reg."
            ax.text(val, -ymax * (0.07 if i == 0 else 0.18),
                    f"VaR ({clave[2:]}%): {val:.2%}\n({txt})",
                    color=color, fontsize=5.5, ha='center', fontweight='bold',
                    bbox=dict(facecolor='#0d1424', alpha=0.85, edgecolor='none', pad=0.6))

    ax.set_ylim(-ymax * 0.26, ymax * 1.15)
    ax.set_title(titulo, fontsize=9, pad=6)
    leyenda(ax, loc='upper right', fontsize=6)


def _heatmap(fig, ax, m, xlabels, ylabels, cmap, vmin, vmax, titulo,
             etiqueta_cbar, anotar=True, fmt_celda='{:.2f}', triangular=False,
             fs_anot=5.5):
    """Matriz como imagen + colorbar, con el valor anotado en cada celda."""
    m = np.asarray(m, dtype=float)
    datos = m.copy()
    if triangular:
        mask = np.triu(np.ones_like(datos, dtype=bool), k=1)
        datos = np.where(mask, np.nan, datos)
    im = ax.imshow(datos, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto',
                   interpolation='nearest')
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=6, color=AX_FG)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=6, color=AX_FG)
    ax.tick_params(colors=AX_FG, labelsize=6)
    ax.set_title(titulo, color='#e6edf3', fontsize=9, pad=8)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)

    if anotar and datos.size <= 24 * 24:
        for i in range(datos.shape[0]):
            for j in range(datos.shape[1]):
                val = datos[i, j]
                if not np.isfinite(val):
                    continue
                # Color de texto según la luminancia REAL de la celda (no una
                # fracción del rango de valores): con colormaps divergentes
                # como 'coolwarm', un valor a mitad de camino del rango ya
                # puede seguir siendo una celda clara (rosa/celeste pálido),
                # así que aproximar "intenso" por distancia a vmin/vmax pone
                # texto blanco sobre fondos todavía claros y se vuelve
                # ilegible. Usando el RGBA que de verdad pintó imshow para
                # ese valor, el contraste es correcto en cualquier colormap.
                r, g, b, _ = im.cmap(im.norm(val))
                luminancia = 0.299 * r + 0.587 * g + 0.114 * b
                color_txt = '#ffffff' if luminancia < 0.5 else '#101820'
                ax.text(j, i, fmt_celda.format(val), ha='center', va='center',
                        fontsize=fs_anot, color=color_txt)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(etiqueta_cbar, color=AX_FG, fontsize=7)
    cbar.ax.tick_params(colors=AX_FG, labelsize=6)
    cbar.outline.set_edgecolor(GRID_C)


def _marcar_sesiones(ax, y_top):
    for hora, nombre, color in _SESIONES:
        ax.axvline(hora, color=color, linewidth=0.9, linestyle=':', alpha=0.6)
        ax.text(hora + 0.3, y_top * 0.97, nombre, fontsize=6, color=color,
                ha='left', va='top', alpha=0.9)


# ══════════════════════════════════════════════════════════════════
#  dibujantes — uno por sección, transcripción de la página del PDF
# ══════════════════════════════════════════════════════════════════
def _dib_precio_equity(fig, b, h):
    d = b.get('precio_equity')
    if not d:
        ax_placeholder(fig.add_subplot(111), "Sin datos de precio/equity.")
        return
    # Nota sobre los hspace/wspace de todas las secciones: el motor de layout es
    # 'constrained' (ver _Seccion.pintar), que ya reserva sitio para títulos y
    # ejes. Los valores del PDF (0.4-0.6, pensados para tight_layout) dejarían
    # aquí huecos enormes, así que se usan fracciones mucho menores.
    gs = fig.add_gridspec(3, 1, hspace=0.10, height_ratios=[1.2, 1, 1])
    x = _x(d['x'])

    ax1 = fig.add_subplot(gs[0])
    style_ax(ax1)
    close = d['close']
    ax1.fill_between(x, float(np.nanmin(close)), close, color=VERDE, alpha=0.08)
    ax1.plot(x, close, color=VERDE, linewidth=1.3)
    if d.get('sma') is not None:
        ax1.plot(x, d['sma'], color='#d29922', linewidth=0.7, alpha=0.7,
                 linestyle='--', label='SMA (200pp)')
        leyenda(ax1, loc='upper left')
    ax1.set_title('Precio de cierre', fontsize=9)
    ax1.set_ylabel('Precio', fontsize=7)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    style_ax(ax2)
    eq = d['equity']
    ax2.fill_between(x, 1, eq, where=(eq >= 1), color=VERDE, alpha=0.15, step='pre')
    ax2.fill_between(x, 1, eq, where=(eq < 1), color=ROJO, alpha=0.15, step='pre')
    ax2.plot(x, eq, color='#58a6ff', linewidth=0.9)
    ax2.axhline(1, color=GRIS, linewidth=0.5, linestyle='--')
    ax2.set_title('Curva de equity (base 1.0)', fontsize=9)
    ax2.set_ylabel('Capital (×)', fontsize=7)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    style_ax(ax3)
    dd = d['dd_pct']
    ax3.fill_between(x, 0, dd, color=ROJO, alpha=0.4, step='pre')
    ax3.plot(x, dd, color='#f85149', linewidth=0.6)
    ax3.axhline(0, color=GRIS, linewidth=0.5, linestyle='--')
    ax3.set_title('Underwater drawdown (%)', fontsize=9)
    ax3.set_ylabel('Drawdown %', fontsize=7)
    _eje_fechas(ax3, x)

    close_arr = np.asarray(close, dtype=float)
    eq_arr = np.asarray(eq, dtype=float)
    dd_arr = np.asarray(dd, dtype=float)

    def _texto_hover(idx):
        precio = f"Precio: {fmt(close_arr[idx], 4)}"
        return [
            f"{precio}\nRetorno: {(eq_arr[idx] - 1) * 100:+.2f}%",
            f"{precio}\nEquity: {fmt(eq_arr[idx], 3, '×')}",
            f"{precio}\nDrawdown: {fmt(dd_arr[idx], 2, '%')}",
        ]

    agregar_crosshair(fig, [ax1, ax2, ax3], x, [close_arr, eq_arr, dd_arr],
                      _texto_hover, nombre='precio_equity',
                      colores=[VERDE, '#58a6ff', '#f85149'])


def _dib_estacionalidad(fig, b, h):
    d = b.get('estacionalidad')
    if not d or not d.get('disponible'):
        ax_placeholder(fig.add_subplot(111),
                       "Estacionalidad no disponible: el archivo está basado en TICKS,\n"
                       "sin temporalidad de calendario fija.")
        return
    gs = fig.add_gridspec(2, 1, hspace=0.14)
    for i, (clave, labels_key, titulo) in enumerate(
            (('meses', 'meses_labels', 'Retorno acumulado por mes (%)'),
             ('dias', 'dias_labels', 'Retorno acumulado por día de la semana (%)'))):
        ax = fig.add_subplot(gs[i])
        style_ax(ax)
        vals = np.nan_to_num(np.asarray(d[clave], dtype=float))
        labels = d[labels_key]
        colores = [VERDE if v >= 0 else ROJO for v in vals]
        ax.bar(range(len(vals)), vals, color=colores, alpha=0.85)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha='right')
        ax.set_title(titulo, fontsize=9)
        ax.axhline(0, color=GRIS, linewidth=0.5)
        for j, v in enumerate(vals):
            if v != 0:
                ax.text(j, v, f'{v:.1f}%', ha='center',
                        va='bottom' if v > 0 else 'top', color=AX_FG, fontsize=6.5)


def _regimen_suavizado(er, ventana, umbral_tendencia=0.5, umbral_ruido=0.3):
    """Clasifica cada punto por el régimen MAYORITARIO en una ventana centrada
    a su alrededor, no por su valor de ER instantáneo.

    Colorear bar a bar (como se hacía antes) produce un "confeti" ilegible:
    el ER cruza los umbrales constantemente de una vela a la siguiente,
    así que el color cambiaba en casi cada segmento y no se podía distinguir
    un periodo de tendencia/ruido del siguiente. Un filtro de MODA (no de
    media — las 3 categorías no son promediables) hace que el color solo
    cambie cuando el régimen predominante alrededor de ese punto cambia de
    verdad, y las rachas se ven como bloques contiguos reconocibles.

    umbral_tendencia/umbral_ruido: por defecto 0.5/0.3 (fijos), pero el
    llamador debe pasar los TERCILES (percentiles 33/67) de la serie `er` de
    ESE horizonte — ver `_dib_regimen_er`. Dos sesgos distintos si no:
    - Con el umbral fijo, el ER decae hacia 0 con lookbacks largos incluso en
      un random walk puro (periodo=30 → ~80% de las velas ya caen bajo 0.3
      sin que haya "ruido" real de más), así que el filtro de moda arrastra
      ese sesgo a bloques enteros del gráfico en rojo.
    - Con el umbral "adaptativo" ingenuo de media ± 1σ (calcular_umbrales_er,
      pensado para las líneas de referencia del panel de abajo, no para
      clasificar) pasa lo contrario: media ± 1σ atrapa por construcción
      ~68% de cualquier distribución unimodal en la banda central, así que
      "transición" arranca con una ventaja aplastante y el filtro de moda
      pinta casi todo el gráfico gris.
    Los terciles no tienen ninguno de los dos sesgos: por construcción cada
    categoría parte de ~33% de los puntos sea cual sea la forma real de la
    distribución de ER de ese horizonte, así que el resultado del filtro de
    moda refleja de verdad dónde hay más tendencia/ruido de lo habitual en
    ESE activo, no un artefacto de la elección del umbral.

    Vectorizado con sumas acumuladas por categoría: O(n) por categoría, cada
    una trivial para los como mucho ~20k puntos que trae el bundle.
    """
    n = len(er)
    ventana = max(3, min(ventana, n))
    mitad = ventana // 2

    def _conteo_ventana(mask):
        cs = np.concatenate([[0], np.cumsum(mask)])
        idx = np.arange(n)
        izq = np.clip(idx - mitad, 0, None)
        der = np.clip(idx + mitad + 1, None, n)
        return cs[der] - cs[izq]

    tendencia = er > umbral_tendencia
    ruido = er < umbral_ruido
    transicion = ~tendencia & ~ruido

    conteos = np.vstack([_conteo_ventana(tendencia),
                         _conteo_ventana(transicion),
                         _conteo_ventana(ruido)])
    colores_por_categoria = np.array([ER_TENDENCIA, ER_TRANSICION, ER_RUIDO])
    return colores_por_categoria[np.argmax(conteos, axis=0)]


def _dib_regimen_er(fig, b, h):
    todos = b.get('regimen_er') or {}
    d = todos.get(h)
    if not d:
        ax_placeholder(fig.add_subplot(111),
                       f"Sin datos de régimen ER para la ventana «{h}».")
        return
    gs = fig.add_gridspec(2, 1, hspace=0.10, height_ratios=[3, 1])

    ax_p = fig.add_subplot(gs[0])
    style_ax(ax_p)
    x_raw = np.asarray(d['x'])
    es_fecha = np.issubdtype(x_raw.dtype, np.datetime64)
    # Mismo motivo que en el PDF: el LineCollection y el KAMA deben compartir
    # unidades numéricas en el eje X, así que las fechas van por date2num.
    # Se le pasa el datetime64 tal cual — convertirlo antes a objeto Python
    # devuelve enteros de nanosegundos (datetime no llega a esa resolución) y
    # date2num los interpretaría como ordinales, disparando las fechas al 51867.
    ex = date2num(x_raw) if es_fecha else x_raw.astype(float)
    close = np.asarray(d['close'], dtype=float)
    er = np.asarray(d['er'], dtype=float)

    # Terciles de la propia serie ER de este horizonte (no los umbrales
    # media±1σ del bundle, que son para las líneas de referencia del panel de
    # abajo): por construcción reparten ~33% de los puntos a cada categoría
    # sea cual sea la forma de la distribución, así que ni el ER decayendo
    # hacia 0 en horizontes de periodo largo sesga el filtro de moda hacia
    # rojo, ni una distribución concentrada en el centro lo sesga hacia gris
    # (los dos problemas reales que se probaron antes — ver docstring de
    # _regimen_suavizado).
    if len(er) > 0:
        umbral_ruido, umbral_tendencia = (float(v) for v in np.percentile(er, [33, 67]))
    else:
        umbral_ruido, umbral_tendencia = 0.3, 0.5

    puntos = np.array([ex, close]).T.reshape(-1, 1, 2)
    segmentos = np.concatenate([puntos[:-1], puntos[1:]], axis=1)
    ventana_regimen = max(5, len(er) // 150)
    colores = _regimen_suavizado(er, ventana_regimen,
                                 umbral_tendencia, umbral_ruido)[:-1]
    ax_p.add_collection(LineCollection(segmentos, colors=colores, linewidth=0.8,
                                       alpha=0.95, zorder=2))
    ax_p.autoscale()
    if es_fecha:
        loc = AutoDateLocator()
        ax_p.xaxis.set_major_locator(loc)
        ax_p.xaxis.set_major_formatter(ConciseDateFormatter(loc))

    handles = [Line2D([0], [0], color=ER_TENDENCIA, linewidth=2,
                      label=f'Tendencia (ER>{umbral_tendencia:.2f}, régimen local)'),
               Line2D([0], [0], color=ER_TRANSICION, linewidth=2,
                      label=f'Transición ({umbral_ruido:.2f}-{umbral_tendencia:.2f}, régimen local)'),
               Line2D([0], [0], color=ER_RUIDO, linewidth=2,
                      label=f'Ruido (ER<{umbral_ruido:.2f}, régimen local)')]
    if d.get('kama') is not None:
        ax_p.plot(ex, d['kama'], color='#58a6ff', linewidth=0.8, alpha=0.6, zorder=1)
        handles.append(Line2D([0], [0], color='#58a6ff', linewidth=2,
                              label=f"KAMA {d.get('kama_fast')}/{d.get('kama_slow')}"))
    ax_p.legend(handles=handles, facecolor='#141e30', edgecolor=GRID_C,
                labelcolor=AX_FG, fontsize=7, loc='upper right', framealpha=0.5)
    ax_p.set_title(f"Precio por régimen ER — {h} (periodo ER {d.get('periodo')})", fontsize=9)
    ax_p.set_ylabel('Precio', fontsize=7)

    ax_e = fig.add_subplot(gs[1])
    style_ax(ax_e)
    xe = _x(d['x_er'])
    hist = np.asarray(d['er_hist'], dtype=float)
    n = min(len(xe), len(hist))
    ax_e.fill_between(xe[:n], hist[:n], color='#d29922', alpha=0.15, linewidth=0)
    ax_e.plot(xe[:n], hist[:n], color='#d29922', linewidth=0.4, alpha=0.6)
    sma = np.asarray(d['er_sma'], dtype=float)
    ax_e.plot(xe[:min(n, len(sma))], sma[:min(n, len(sma))], color='#e6edf3',
              linewidth=1.0, label='Tendencia (SMA 200)')
    u = d.get('umbrales', {})
    ax_e.axhline(0.5, color=ER_TENDENCIA, linewidth=0.8, linestyle='--', label='Direccionalidad (0.5)')
    ax_e.axhline(0.3, color=ER_RUIDO, linewidth=0.8, linestyle='--', label='Alto ruido (0.3)')
    if u:
        ax_e.axhline(u['er_medio'], color='#58a6ff', linewidth=1.0, alpha=0.7,
                     label=f"Media ({u['er_medio']:.2f})")
        ax_e.axhline(u['umbral_tendencia'], color='#58a6ff', linewidth=0.7, linestyle=':',
                     alpha=0.7, label=f"+1σ ({u['umbral_tendencia']:.2f})")
        ax_e.axhline(u['umbral_ruido'], color='#58a6ff', linewidth=0.7, linestyle=':',
                     alpha=0.7, label=f"-1σ ({u['umbral_ruido']:.2f})")
    sub = (f" · Hurst v{d['hurst_ventana']}: {d['hurst_medio']:.3f}"
           if d.get('hurst_medio') is not None else "")
    _eje_fechas(ax_e, xe)
    ax_e.set_title(f"Histórico ER {h}{sub}", fontsize=9)
    ax_e.set_ylim(0, 1)
    ax_e.set_ylabel('ER', fontsize=7)
    leyenda(ax_e, loc='upper right', ncol=2, fontsize=6)


def _dib_riesgo_dia_anual(fig, b, h):
    d = b.get('riesgo_dia_anual')
    if not d:
        ax_placeholder(fig.add_subplot(111), "Sin datos de riesgo diario/anual.")
        return
    gs = fig.add_gridspec(2, 1, hspace=0.16)

    dia = d['diario']
    _campana(fig.add_subplot(gs[0]), dia['media'], dia['std'], dia['sigmas'],
             'd', 'Retorno diario (proyección)')
    anual = d['anual']
    _campana(fig.add_subplot(gs[1]), anual['media'], anual['std'], anual['sigmas'],
             'añ', 'Retorno anual (proyección)')


def _dib_qqplot(fig, b, h):
    # Mismo bundle que _dib_riesgo_dia_anual (clave 'qq' dentro de
    # 'riesgo_dia_anual'): se separó en su propia sección solo para poder
    # tener una explicación "?" dedicada, no porque el dato cambie.
    d = b.get('riesgo_dia_anual')
    if not d:
        ax_placeholder(fig.add_subplot(111), "Sin datos de riesgo diario/anual.")
        return
    ax = fig.add_subplot(111)
    style_ax(ax)
    qq = d['qq']
    osm = np.asarray(qq['osm'], dtype=float)
    ax.scatter(osm, np.asarray(qq['osr'], dtype=float), color='#58a6ff', s=6, alpha=0.4)
    ax.plot(osm, qq['slope'] * osm + qq['intercept'], color=ROJO,
            linewidth=1.1, linestyle='--')
    ax.set_title('QQ-Plot: retornos vs Normal', fontsize=9)
    ax.set_xlabel('Cuantiles teóricos', fontsize=7)
    ax.set_ylabel('Cuantiles observados', fontsize=7)
    ax.text(0.98, 0.05, f"R² ajuste: {qq['r_sq']:.4f}", transform=ax.transAxes,
            ha='right', va='bottom', fontsize=7, color=TEXTO_TENUE, fontweight='bold')


def _dib_riesgo_intradia(fig, b, h):
    d = b.get('riesgo_intradia')
    if not d:
        ax_placeholder(fig.add_subplot(111), "Sin datos de riesgo intradiario.")
        return
    gs = fig.add_gridspec(3, 1, hspace=0.14, height_ratios=[1, 1, 0.9])
    tf = b.get('_meta', {}).get('tf', '')

    _campana(fig.add_subplot(gs[0]), d['media'], d['std'], d['sigmas'],
             'vel', f'Retorno por vela {tf}')
    _campana(fig.add_subplot(gs[1]), d['media'], d['std'], {}, 'vel',
             f"VaR {'paramétrico' if d.get('es_normal') else 'histórico'} ({tf})",
             vars_=d.get('var'))

    ax = fig.add_subplot(gs[2])
    style_ax(ax)
    rol = d.get('rolling')
    if rol:
        x = _x(rol['x'])
        et = rol.get('etiqueta_ventana', '')
        ax.plot(x, rol['var95_pct'], color='#BA7517', linewidth=0.7, label=f'VaR 95% ({et})')
        ax.plot(x, rol['var99_pct'], color=ROJO, linewidth=0.7, label=f'VaR 99% ({et})')
        ax.fill_between(x, rol['var95_pct'], alpha=0.08, color='#BA7517')
        ax.axhline(d['p05_pct'], color='#BA7517', linewidth=0.8, linestyle='--', alpha=0.6,
                   label=f"P5 todo el periodo: {d['p05_pct']:.2f}%")
        ax.axhline(d['p01_pct'], color=ROJO, linewidth=0.8, linestyle='--', alpha=0.6,
                   label=f"P1 todo el periodo: {d['p01_pct']:.2f}%")
        ax.set_title(f'Rolling VaR (ventana {et})', fontsize=9)
        _eje_fechas(ax, x)
        leyenda(ax, loc='lower left', fontsize=6)
    else:
        ax_placeholder(ax, "Datos insuficientes para Rolling VaR.")
    ax.set_ylabel('Pérdida %', fontsize=7)


def _dib_perfil_horario(fig, b, h):
    d = b.get('perfil_horario')
    if not d or not d.get('disponible'):
        ax_placeholder(fig.add_subplot(111),
                       "Análisis intradía no disponible: el timeframe no tiene\n"
                       "distribución horaria (todas las velas a la misma hora).")
        return
    gs = fig.add_gridspec(2, 1, hspace=0.12, height_ratios=[1.5, 1])

    ax1 = fig.add_subplot(gs[0])
    style_ax(ax1)
    # bxp acepta los mismos dicts de estadísticos que calculó el script.
    stats_bxp = [dict(s) for s in d['bxp']]
    ax1.bxp(stats_bxp, positions=range(24), widths=0.6, patch_artist=True, showfliers=True,
            boxprops=dict(facecolor='#185FA5', alpha=0.6, edgecolor='#4A8BC2', linewidth=0.8),
            whiskerprops=dict(color='#4A8BC2', linewidth=0.8),
            capprops=dict(color='#4A8BC2', linewidth=0.8),
            medianprops=dict(color=ROJO, linewidth=1.4),
            flierprops=dict(marker='o', markerfacecolor=ROJO, markersize=2.5, alpha=0.4,
                            markeredgecolor='none'))
    medias = np.asarray(d['medias'], dtype=float)
    ax1.plot(range(24), medias, 'D', color=ROJO, markersize=4, zorder=5, alpha=0.9,
             linestyle='none')

    los = [s['whislo'] for s in stats_bxp if s['n'] > 0]
    his = [s['whishi'] for s in stats_bxp if s['n'] > 0]
    if los and his:
        rango = max(his) - min(los)
        margen = rango * 0.15 if rango > 0 else (max(his) * 0.1 or 1.0)
        ax1.set_ylim(min(0, min(los) - margen * 0.2), max(his) + margen)

    vm = d['vol_media_total']
    ax1.axhline(vm, color='#BA7517', linewidth=1.1, linestyle='--', alpha=0.85)
    _marcar_sesiones(ax1, ax1.get_ylim()[1])
    ax1.set_xticks(range(24))
    ax1.set_xticklabels(range(24), fontsize=6)
    ax1.set_title('Desviación estándar de retornos por hora UTC', fontsize=9)
    ax1.set_xlabel('Hora UTC', fontsize=7)
    ax1.set_ylabel('Desv. est. (%)', fontsize=7)
    ax1.legend(handles=[
        Line2D([0], [0], color=ROJO, linewidth=1.5, label='Mediana'),
        Line2D([0], [0], marker='D', color='none', markerfacecolor=ROJO, markersize=5,
               label='Media'),
        Line2D([0], [0], color='#BA7517', linewidth=1.1, linestyle='--',
               label=f'Vol. media total: {vm:.4f}%'),
    ], facecolor='#141e30', edgecolor=GRID_C, labelcolor=AX_FG, fontsize=7,
        loc='upper right', framealpha=0.5)

    ax2 = fig.add_subplot(gs[1])
    style_ax(ax2)
    vals = np.asarray(d['ret_por_hora'], dtype=float)
    ax2.bar(range(24), vals, color=[VERDE if v >= 0 else ROJO for v in vals],
            alpha=0.75, width=0.8)
    ax2.axhline(0, color=GRIS, linewidth=0.5, linestyle='--')
    media_h = float(np.mean(vals))
    ax2.axhline(media_h, color='#BA7517', linewidth=1.0, linestyle=':', alpha=0.8,
                label=f'Retorno medio: {media_h:.2f}%')
    ax2.set_xticks(range(24))
    ax2.set_xticklabels(range(24), fontsize=6)
    ax2.set_xlim(-0.5, 23.5)
    ax2.set_title('Retorno acumulado por hora (%)', fontsize=9)
    ax2.set_xlabel('Hora UTC', fontsize=7)
    ax2.set_ylabel('Retorno acumulado %', fontsize=7)
    top3 = set(np.argsort(np.abs(vals))[-3:])
    y0, y1 = ax2.get_ylim()
    off = abs(y1 - y0) * 0.02
    for i, v in enumerate(vals):
        if abs(v) > 0.01:
            ax2.text(i, v + (off if v > 0 else -off), f'{v:.2f}%', ha='center',
                     va='bottom' if v > 0 else 'top',
                     fontsize=6.5 if i in top3 else 6,
                     color='#d29922' if i in top3 else AX_FG,
                     fontweight='bold' if i in top3 else 'normal')
    _marcar_sesiones(ax2, y1)
    leyenda(ax2, loc='upper right')


def _dib_corr24(fig, b, h):
    d = b.get('corr_24')
    if not d:
        ax_placeholder(fig.add_subplot(111),
                       "Matriz de correlación intra-diaria no disponible.")
        return
    ax = fig.add_subplot(111)
    style_ax(ax)
    _heatmap(fig, ax, d['m'], d['labels'], d['labels'], 'coolwarm', -1, 1,
             'Correlación intra-diaria entre horas UTC (Spearman)',
             'Correlación', triangular=True, fs_anot=5.5)
    ax.set_xlabel('Hora UTC', fontsize=7)
    ax.set_ylabel('Hora UTC', fontsize=7)


def _dib_heatmaps(fig, b, h):
    d = b.get('heatmaps') or {}
    week, month = d.get('week'), d.get('month')
    if not week and not month:
        ax_placeholder(fig.add_subplot(111),
                       "Heatmaps no disponibles: TF diario o sin datos horarios.")
        return
    gs = fig.add_gridspec(2, 1, hspace=0.12)
    for i, (bloque, titulo, ylab) in enumerate(
            ((week, 'Estacionalidad semanal: volatilidad por hora y día', 'Día'),
             (month, 'Estacionalidad mensual: volatilidad por hora y mes', 'Mes'))):
        ax = fig.add_subplot(gs[i])
        if not bloque:
            ax_placeholder(ax, "No disponible")
            continue
        style_ax(ax)
        _heatmap(fig, ax, bloque['m'], bloque['cols'], bloque['filas'], 'YlOrRd',
                 0, bloque['vmax'], titulo, 'Volatilidad (%)',
                 fmt_celda='{:.3f}', fs_anot=5.8)
        ax.set_xlabel('Hora UTC', fontsize=7)
        ax.set_ylabel(ylab, fontsize=7)


def _dib_natr_corr(fig, b, h):
    d = b.get('natr_multitf')
    if not d or d.get('corr') is None:
        ax_placeholder(fig.add_subplot(111), "Matriz de correlación NATR no disponible.")
        return
    ax = fig.add_subplot(111)
    style_ax(ax)
    _heatmap(fig, ax, d['corr'], d['labels'], d['labels'], 'coolwarm', -1, 1,
             'Correlación de NATR entre timeframes', 'Correlación',
             triangular=True, fs_anot=6.5)
    ax.set_xlabel('Timeframe', fontsize=7)
    ax.set_ylabel('Timeframe', fontsize=7)


def _dib_dependencia(fig, b, h):
    d = b.get('dependencia')
    if not d:
        ax_placeholder(fig.add_subplot(111), "Análisis de dependencia no disponible.")
        return
    gs = fig.add_gridspec(3, 3, hspace=0.22, wspace=0.14)

    ax1 = fig.add_subplot(gs[0, 0])
    esc = d.get('escalas') or {}
    if esc.get('labels'):
        vals = np.asarray(esc['valores'], dtype=float).reshape(-1, 1)
        _heatmap(fig, ax1, vals, ['PACF lag 1'], esc['labels'], 'coolwarm',
                 -0.5, 0.5, 'Dependencia por escala', '', fs_anot=6)
    else:
        ax_placeholder(ax1, "Sin escalas")

    serie = np.asarray(d.get('serie_dia', []), dtype=float)
    serie = serie[np.isfinite(serie)]
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    if len(serie) > 20:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        plot_acf(serie, ax=ax2, lags=15, title='ACF diario', color=AZUL, vlines_kwargs={'colors': AZUL})
        plot_pacf(serie, ax=ax3, lags=15, title='PACF diario', color=AZUL, vlines_kwargs={'colors': AZUL})
        for a in (ax2, ax3):
            style_ax(a)
            a.set_title(a.get_title(), fontsize=9)
    else:
        ax_placeholder(ax2, "Serie corta")
        ax_placeholder(ax3, "Serie corta")

    ax4 = fig.add_subplot(gs[1, :])
    style_ax(ax4)
    colores_ruido = ['#e67e22', '#9b59b6', '#3498db', VERDE, AMBAR]
    for i, walk in enumerate(d.get('random_walks') or []):
        ax4.plot(walk, color=colores_ruido[i % len(colores_ruido)], linewidth=0.9,
                 alpha=0.6, linestyle='--',
                 label='Random walk (ruido)' if i == 0 else None)
    if d.get('precio_real') is not None:
        ax4.plot(d['precio_real'], color='#e6edf3', linewidth=2.2, alpha=0.95,
                 label='Activo real', zorder=5)
    ax4.set_title('Estructura vs. ruido aleatorio (simulación de precio)', fontsize=9)
    ax4.set_ylabel('Precio simulado', fontsize=7)
    leyenda(ax4, loc='upper left')

    ax5 = fig.add_subplot(gs[2, :])
    style_ax(ax5)
    acf_sq = d.get('acf_sq')
    if acf_sq is not None and len(acf_sq):
        lags = np.arange(len(acf_sq))
        ml, sl, bl = ax5.stem(lags, np.asarray(acf_sq, dtype=float))
        ml.set_color('#185FA5')
        ml.set_markersize(3.5)
        sl.set_color('#185FA5')
        sl.set_linewidth(1.1)
        bl.set_color(GRID_C)
        ci = d.get('ci')
        if ci:
            ax5.axhline(ci, color=ROJO, linewidth=0.8, linestyle='--')
            ax5.axhline(-ci, color=ROJO, linewidth=0.8, linestyle='--')
        ax5.axhline(0, color=GRIS, linewidth=0.5)
        ax5.set_title('ACF retornos² — volatility clustering', fontsize=9)
        if d.get('lb_p') is not None:
            ok = d.get('clustering')
            ax5.text(0.98, 0.92, f"LB p-valor: {d['lb_p']:.4f}  {'✓' if ok else '✗'}",
                     transform=ax5.transAxes, ha='right', va='top', fontsize=8,
                     color=VERDE if ok else ROJO)
    else:
        ax_placeholder(ax5, "No hay suficientes datos para el ACF de retornos².")


def _dib_dashboard_natr(fig, b, h):
    todos = b.get('dashboard_natr') or {}
    d = todos.get(h)
    if not d:
        ax_placeholder(fig.add_subplot(111),
                       f"Dashboard NATR no disponible para la ventana «{h}».")
        return
    gs = fig.add_gridspec(2, 2, hspace=0.16, wspace=0.10)

    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a)
    term = d.get('term')
    if term:
        ax_a.plot(term['mins'], term['actual'], 'o-', color=AZUL, linewidth=1.3,
                  markersize=4, label='NATR actual', zorder=3)
        ax_a.plot(term['mins'], term['teorico'], 'x--', color=NARANJA, linewidth=1.0,
                  markersize=4, alpha=0.75, label='Teórico √T', zorder=2)
        ax_a.set_xscale('log')
        ax_a.set_yscale('log')
        ax_a.set_xticks(term['mins'])
        ax_a.set_xticklabels(term['tfs'], rotation=30, ha='right', fontsize=6)
        est = term.get('estructura', '')
        ax_a.text(0.98, 0.05, est, transform=ax_a.transAxes, ha='right', va='bottom',
                  color=ROJO if est.startswith('BACK') else VERDE,
                  fontsize=8, fontweight='bold')
        leyenda(ax_a, loc='upper left', fontsize=6)
    else:
        ax_placeholder(ax_a, "Sin datos para term structure.")
    ax_a.set_title('A — Term structure NATR', fontsize=9)
    ax_a.set_ylabel('NATR (%)', fontsize=7)

    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b)
    z = d.get('z')
    if z:
        vals = z['vals']
        colores = [ROJO if v > 2 else VERDE if v < -2 else AMBAR if abs(v) > 1 else '#2a4a6a'
                   for v in vals]
        y = np.arange(len(vals))
        ax_b.barh(y, vals, color=colores, edgecolor=GRID_C, height=0.6)
        ax_b.set_yticks(y)
        ax_b.set_yticklabels(z['tfs'], fontsize=6.5)
        ax_b.axvline(2, color=ROJO, linestyle='--', alpha=0.6, linewidth=0.8)
        ax_b.axvline(-2, color=VERDE, linestyle='--', alpha=0.6, linewidth=0.8)
        ax_b.axvline(0, color=GRIS, alpha=0.5, linewidth=0.5)
        ax_b.set_xlim(-4, 4)
        for i, v in enumerate(vals):
            ax_b.text(max(v + 0.1, 0.15) if v >= 0 else min(v - 0.1, -0.15), i,
                      f'{v:+.2f}', va='center', fontsize=6, color=AX_FG,
                      ha='left' if v >= 0 else 'right')
    else:
        ax_placeholder(ax_b, "Sin Z-scores.")
    ax_b.set_title('B — Z-score NATR por TF', fontsize=9)
    ax_b.set_xlabel('Z-score', fontsize=7)

    ax_c = fig.add_subplot(gs[1, 0])
    style_ax(ax_c)
    sz = d.get('serie_z')
    if sz and sz.get('series'):
        for i, s in enumerate(sz['series']):
            ax_c.plot(_x(s['x']), s['y'], color=(AZUL, NARANJA)[i % 2],
                      linewidth=0.9, alpha=0.9, label=s['tf'])
        ax_c.axhline(2, color=ROJO, linestyle='--', alpha=0.5, linewidth=0.8)
        ax_c.axhline(-2, color=VERDE, linestyle='--', alpha=0.5, linewidth=0.8)
        _eje_fechas(ax_c, sz['series'][0]['x'])
        leyenda(ax_c, loc='upper right', fontsize=6)
        titulo_c = f"C — Serie temporal Z-score ({sz.get('par')})" if sz.get('par') else 'C — Serie temporal Z-score'
    else:
        ax_placeholder(ax_c, "Sin series temporales.")
        titulo_c = 'C — Serie temporal Z-score'
    ax_c.set_title(titulo_c, fontsize=9)
    ax_c.set_ylabel('Z-score', fontsize=7)

    ax_d = fig.add_subplot(gs[1, 1])
    style_ax(ax_d)
    ratios = d.get('ratio')
    if ratios:
        for i, r in enumerate(ratios):
            color = (AZUL, NARANJA)[i % 2]
            ax_d.plot(_x(r['x']), r['y'], color=color, linewidth=0.9, alpha=0.9,
                      label=r['label'])
            bb = r.get('bb')
            if bb:
                xb = _x(bb['x'])
                ax_d.fill_between(xb, bb['lo'], bb['up'], color=color, alpha=0.10)
                ax_d.plot(xb, bb['mu'], color=color, linestyle=':', linewidth=0.6, alpha=0.6)
            if r.get('x_actual') is not None:
                ax_d.scatter(_x(r['x_actual']), [r['current']], color=color, s=22, zorder=5)
        _eje_fechas(ax_d, ratios[0]['x'])
        leyenda(ax_d, loc='upper right', fontsize=6)
    else:
        ax_placeholder(ax_d, "Sin ratios Short/Long disponibles.")
    ax_d.set_title('D — Ratio Short/Long (Bollinger ±2σ)', fontsize=9)
    ax_d.set_ylabel('Ratio NATR fast/slow', fontsize=7)


# ══════════════════════════════════════════════════════════════════
#  widgets
# ══════════════════════════════════════════════════════════════════
class _Seccion(QGroupBox):
    """Un gráfico: cabecera (título + ayuda + barra de herramientas) y canvas.

    El dibujo es perezoso — `marcar_sucio()` invalida el contenido y solo se
    repinta cuando la sección se hace visible.
    """

    def __init__(self, titulo, ayuda, dibujante, alto=320, depende_ventana=False,
                 parent=None):
        """`ayuda`: tupla (logica, significado, uso, resultados) — las 4
        pestañas del popup que abre el icono_ayuda."""
        super().__init__(parent)
        self._dibujante = dibujante
        self._sucio = True
        self.depende_ventana = depende_ventana
        self._area = None   # lo asigna _PestanaScroll.add_seccion

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(6)

        self.fig, self.canvas = make_canvas(alto_min=alto)
        self.canvas.installEventFilter(self)
        # Que la rueda no le robe el foco al canvas: si lo hiciera, girar la
        # rueda sobre un gráfico dejaría de permitir cambiar de pestaña con las
        # flechas hasta volver a hacer clic fuera.
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        cabecera = QHBoxLayout()
        cabecera.setSpacing(6)
        lbl = QLabel(titulo)
        lbl.setObjectName("titulo")
        cabecera.addWidget(lbl)
        cabecera.addWidget(icono_ayuda(*ayuda))
        cabecera.addStretch()
        self.barra = BarraGrafico(self.canvas, self, mostrar_coords=False)
        # La barra también se filtra: sus QToolButton pueden quedarse la rueda y
        # dejar muerta esa franja de la sección.
        self.barra.installEventFilter(self)
        cabecera.addWidget(self.barra)
        lay.addLayout(cabecera)
        lay.addWidget(self.canvas, 1)

    def eventFilter(self, obj, event):
        # FigureCanvasQTAgg acepta los eventos de rueda (los traduce a
        # scroll_event de matplotlib), así que se los come y la página deja de
        # desplazarse justo cuando el cursor está sobre un gráfico — que es la
        # mayor parte de la sección.
        #
        # Se mueve la barra del QScrollArea a mano en vez de reenviarle el
        # evento: los eventos que llegan del sistema van marcados como
        # spontaneous, y volver a despacharlos con sendEvent no los propaga
        # igual que a uno sintético — de ahí que la rueda no hiciera nada en la
        # aplicación real aunque en una prueba con eventos fabricados sí
        # funcionara.
        if event.type() == QEvent.Type.Wheel and self._area is not None:
            barra = self._area.verticalScrollBar()
            barra.setValue(barra.value() + self._desplazamiento_rueda(event, barra))
            return True
        return super().eventFilter(obj, event)

    @staticmethod
    def _desplazamiento_rueda(event, barra):
        """Píxeles a desplazar, replicando el comportamiento nativo de Qt:
        `wheelScrollLines` líneas por muesca de rueda."""
        pixeles = event.pixelDelta().y()
        if pixeles:
            return -pixeles                      # touchpads: delta ya en píxeles
        muescas = event.angleDelta().y() / 120.0
        lineas = QApplication.wheelScrollLines() or 3
        return int(round(-muescas * lineas * max(barra.singleStep(), 1)))

    def marcar_sucio(self):
        self._sucio = True

    @no_crash
    def pintar(self, bundle, horizonte, forzar=False):
        if bundle is None or (not self._sucio and not forzar):
            return
        self.fig.clear()
        # Motor 'constrained' en vez de tight_layout: varias secciones llevan
        # colorbar, y tight_layout no sabe colocar esos ejes (avisa y descuadra
        # la figura).
        self.fig.set_layout_engine('constrained')
        try:
            self._dibujante(self.fig, bundle, horizonte)
        except Exception:
            import traceback as _tb
            _tb.print_exc()
            self.fig.clear()
            ax_placeholder(self.fig.add_subplot(111),
                           "No se pudo dibujar esta sección.\nRevisa la consola.")
        self.fig.set_facecolor(FIG_BG)
        self.canvas.draw_idle()
        self._sucio = False


class _PestanaScroll(QScrollArea):
    """Columna scrolleable de secciones."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.secciones = []
        self._cont = QWidget()
        self._lay = QVBoxLayout(self._cont)
        self._lay.setContentsMargins(10, 10, 10, 10)
        self._lay.setSpacing(12)
        self.setWidget(self._cont)

    def add_seccion(self, seccion):
        self.secciones.append(seccion)
        # La sección necesita saber a qué área pertenece para reenviarle la
        # rueda del ratón (ver _Seccion.eventFilter).
        seccion._area = self
        self._lay.addWidget(seccion)
        return seccion

    def add_widget(self, w):
        self._lay.addWidget(w)
        return w

    def cerrar(self):
        self._lay.addStretch()


class GraficosAnalisis(QWidget):
    """Sub-pestaña «Gráficos»: todas las figuras del informe, nativas."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(ESTILO)
        self._bundle = None
        self._horizonte = 'General'

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        lay.addWidget(self.tabs, 1)

        self._paginas = []
        self._construir()

        # ← / → cambian de pestaña temática desde cualquier punto de la vista.
        # Se usan atajos con contexto WidgetWithChildren en vez de keyPressEvent
        # porque el foco casi siempre está en un hijo (un canvas, la barra de
        # herramientas o el área de scroll), y ahí keyPressEvent nunca llegaría.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        for tecla, paso in ((Qt.Key.Key_Left, -1), (Qt.Key.Key_Right, 1)):
            atajo = QShortcut(QKeySequence(tecla), self)
            atajo.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            atajo.activated.connect(lambda p=paso: self._mover_pestana(p))

    # ── construcción ──
    def _construir(self):
        resumen = _PestanaScroll()
        resumen.add_seccion(_Seccion(
            "Precio, equity y underwater",
            ("El precio de cierre se muestra tal cual, con su media móvil de régimen "
             "superpuesta; la equity se calcula normalizando el precio a un valor inicial "
             "de 1.0 (equity = precio / precio_inicial), simulando comprar el activo al "
             "principio y mantenerlo; el drawdown bajo agua se obtiene, vela a vela, como "
             "la distancia porcentual entre esa equity y su máximo acumulado hasta ese "
             "momento.",
             "La equity en 1.0 significa \"sin cambios desde el inicio\"; por encima es "
             "ganancia acumulada, por debajo pérdida. El drawdown en 0% significa que el "
             "activo está en máximos históricos; cuanto más negativo, más lejos está del "
             "último pico.",
             "Sirve para calibrar expectativas de un buy & hold puro sobre este activo "
             "—cuánto tiempo y cuánta caída hay que aguantar— y como referencia contra la "
             "que comparar la equity de cualquier sistema que se backtestee sobre él.",
             "Una curva de equity que sube de forma sostenida con un underwater que rara "
             "vez baja de -10/-15% indica un activo \"cómodo\" de mantener; rachas largas "
             "y profundas en el panel inferior (por ejemplo -40% o más durante meses) "
             "señalan que aguantar el activo exige tolerancia alta a la pérdida en papel, "
             "aunque el retorno final sea positivo."),
            _dib_precio_equity, alto=460))
        resumen.add_seccion(_Seccion(
            "Estacionalidad de calendario",
            ("Se agrupa el retorno de cada vela por el mes calendario al que pertenece y, "
             "por separado, por el día de la semana, sumando el retorno acumulado de "
             "todos los años/semanas disponibles en cada categoría.",
             "Cada barra es el retorno total que ha aportado ese mes o ese día concreto a "
             "lo largo de TODO el histórico, no un promedio por ocurrencia — un mes que "
             "aparece muchas veces en el histórico pesa más que uno con pocas "
             "repeticiones.",
             "Sirve para detectar sesgos de calendario que podrían usarse como filtro "
             "adicional (evitar operar ciertos días, reforzar la exposición en los meses "
             "fuertes).",
             "Barras verdes marcadas y consistentes en varios meses seguidos sugieren un "
             "sesgo real; una única barra muy alta rodeada de barras pequeñas suele "
             "deberse a uno o dos eventos puntuales (crash o rally aislado) y no a un "
             "patrón repetible — conviene contrastarlo con cuántas veces se repite ese "
             "mes/día en el histórico antes de confiar en él."),
            _dib_estacionalidad, alto=380))
        resumen.cerrar()

        riesgo = _PestanaScroll()
        riesgo.add_seccion(_Seccion(
            "Riesgo diario y anual",
            ("Se ajusta una distribución normal a la media y desviación típica de los "
             "retornos diarios y, por separado, de los anuales, dibujando las bandas "
             "±1σ/±2σ/±3σ de esa campana teórica junto a la frecuencia REAL con la que "
             "se ha superado cada nivel en el histórico.",
             "Si la frecuencia real de superar una banda coincide con la teórica (por "
             "ejemplo, ±2σ debería superarse ~1 vez cada 20-22 observaciones), la "
             "Normal es un modelo razonable para ese horizonte; si los eventos reales "
             "superan la banda con más frecuencia de la esperada, hay colas más gruesas "
             "de lo normal.",
             "Sirve para decidir si conviene fiarse de métricas que asumen normalidad "
             "(VaR paramétrico, Sharpe sin corregir, dimensionamiento de riesgo por σ) o "
             "si hay que tratar este activo como propenso a sorpresas de cola gorda, "
             "tanto a nivel diario como anual.",
             "Frecuencias reales próximas a las teóricas en ambas campanas (diaria y "
             "anual) es un activo \"bien comportado\" estadísticamente en los dos "
             "horizontes; si la campana diaria se ajusta bien pero la anual no (o al "
             "revés), el comportamiento de cola gorda solo se manifiesta a una de las "
             "dos escalas temporales, algo a tener en cuenta según el horizonte de la "
             "estrategia."),
            _dib_riesgo_dia_anual, alto=380))
        riesgo.add_seccion(_Seccion(
            "QQ-Plot: retornos vs Normal",
            ("Se ordenan los retornos diarios reales de menor a mayor y se enfrentan, "
             "uno a uno, a los cuantiles que tendría una distribución Normal con la "
             "misma media y desviación; sobre esos puntos se ajusta una recta de "
             "regresión y se calcula su R².",
             "Si los puntos siguen la recta roja, la distribución es aproximadamente "
             "normal; cuanto más se curvan hacia arriba o abajo en los extremos (las "
             "colas del gráfico), más gruesas son las colas reales frente a las que "
             "predice una Normal —eventos extremos más frecuentes de lo que la campana "
             "sugeriría—. El R² resume numéricamente ese ajuste: cerca de 1 es buen "
             "ajuste a la Normal.",
             "Sirve como diagnóstico visual rápido y complementario al de las bandas "
             "±σ: en vez de mirar niveles concretos, muestra de un vistazo en qué tramo "
             "de la distribución (centro o colas) se desvía más el activo de un "
             "comportamiento normal.",
             "Puntos alineados con la recta en el centro pero que se separan "
             "claramente hacia arriba en la cola derecha y hacia abajo en la izquierda "
             "indican colas simétricas más gruesas de lo normal (eventos extremos "
             "posibles en ambas direcciones); si la desviación es solo en un extremo, "
             "el riesgo de cola es asimétrico (más propenso a caídas bruscas que a "
             "subidas bruscas, o viceversa). Un R² bajo junto con puntos muy dispersos "
             "de la recta confirma que fiarse de un VaR paramétrico aquí sería "
             "arriesgado."),
            _dib_qqplot, alto=320))
        riesgo.add_seccion(_Seccion(
            "Riesgo intradiario + Rolling VaR",
            ("Repite el ajuste de campana pero sobre el retorno de cada vela individual, "
             "y calcula el VaR al 95% y al 99% —la pérdida que solo se supera el 5%/1% de "
             "las veces— de forma paramétrica (percentil de la Normal ajustada) si los "
             "retornos pasan el test de normalidad, o histórica (percentil real de la "
             "muestra) si no lo pasan. El panel inferior recalcula ese mismo VaR en una "
             "ventana móvil.",
             "El VaR 95% es la pérdida que, en una vela cualquiera, solo debería "
             "superarse el 5% de las veces; el 99% es el equivalente más extremo. La "
             "línea rolling muestra cómo ese umbral ha cambiado en el tiempo, frente a "
             "los percentiles P5/P1 fijos de todo el periodo.",
             "Sirve para dimensionar el riesgo esperado por operación a nivel de vela "
             "(tamaño de posición) y para detectar en qué tramos temporales el riesgo se "
             "disparó por encima de lo habitual.",
             "Si la línea de VaR rolling se mantiene cerca de las líneas de referencia "
             "P5/P1 todo el periodo, el riesgo por vela es estable; picos donde el "
             "rolling se dispara muy por encima de esas referencias señalan episodios de "
             "volatilidad anómala en los que operar con el tamaño de posición \"normal\" "
             "habría sido más arriesgado de lo habitual."),
            _dib_riesgo_intradia, alto=560))
        riesgo.cerrar()

        intradia = _PestanaScroll()
        intradia.add_seccion(_Seccion(
            "Perfil horario",
            ("Para cada una de las 24 horas UTC se calcula la desviación típica de los "
             "retornos de esa hora en cada semana del histórico; esa colección de valores "
             "semanales dibuja la caja (mediana, cuartiles, outliers) y el rombo de la "
             "media. Abajo, el retorno se acumula sumando el retorno medio de esa hora a "
             "lo largo de todo el periodo.",
             "Cajas altas o con bigotes largos indican horas con volatilidad alta y "
             "variable de semana en semana; el panel inferior indica si esa hora, en "
             "conjunto, tendió a mover el precio al alza o a la baja. Las líneas "
             "verticales marcan la apertura de las sesiones de Tokio, Londres y Nueva "
             "York.",
             "Sirve para decidir en qué horas conviene operar (mayor movimiento "
             "aprovechable) y en cuáles conviene evitar exposición (volatilidad alta sin "
             "retorno direccional que la compense).",
             "Una hora con caja alta arriba pero barra casi nula abajo es \"ruidosa pero "
             "sin dirección\" (ida y vuelta); una hora con caja moderada pero barra "
             "grande y consistente abajo es más aprovechable direccionalmente. Si los "
             "picos de volatilidad coinciden con las líneas de apertura de sesión, el "
             "movimiento está ligado a la actividad institucional de esa sesión, no es "
             "aleatorio."),
            _dib_perfil_horario, alto=520))
        intradia.add_seccion(_Seccion(
            "Correlación intra-diaria 24×24",
            ("Se calcula la correlación de Spearman (basada en el orden de los valores, "
             "no en su magnitud) entre los retornos de cada par de horas UTC a lo largo "
             "de todo el histórico, formando una matriz 24×24.",
             "Un valor cercano a +1 indica que el movimiento de una hora tiende a "
             "repetirse en la otra (continuación); cercano a -1, que tiende a "
             "revertirse; cercano a 0, que son independientes.",
             "Sirve para detectar dependencias horarias explotables —por ejemplo, si la "
             "apertura de Londres anticipa sistemáticamente el movimiento de la apertura "
             "de Nueva York— que la vela aislada no revelaría.",
             "Una franja de celdas con color intenso alrededor de la diagonal (horas "
             "consecutivas) es esperable y refleja inercia de corto plazo; celdas con "
             "color intenso lejos de la diagonal (horas muy separadas, como "
             "Tokio-Londres) son las que de verdad aportan información nueva y valen la "
             "pena investigar como filtro de entrada."),
            _dib_corr24, alto=520))
        intradia.add_seccion(_Seccion(
            "Heatmaps de volatilidad",
            ("Se calcula la volatilidad media (desviación típica de retornos) de cada "
             "hora UTC cruzada con el día de la semana, y por separado cruzada con el "
             "mes del año; la escala de color se recorta al percentil 95 de cada matriz.",
             "El color de cada celda indica cuánto se mueve típicamente el activo en esa "
             "combinación concreta de hora y día/mes; colores más intensos son mayor "
             "volatilidad.",
             "Sirve para afinar el perfil horario con una segunda dimensión: si la "
             "volatilidad de una hora es alta solo ciertos días o solo en ciertos meses, "
             "en vez de asumir que su comportamiento es constante todo el año.",
             "Una franja vertical intensa (misma hora, todos los días o meses) confirma "
             "que esa hora es volátil de forma estructural, no por casualidad de un día "
             "concreto; manchas aisladas en una sola celda suelen deberse a un evento "
             "puntual más que a un patrón repetible."),
            _dib_heatmaps, alto=560))
        intradia.cerrar()

        regimen = _PestanaScroll()
        regimen.add_seccion(_Seccion(
            "Precio por régimen ER",
            ("El Efficiency Ratio (ER) mide, para cada vela, qué proporción del "
             "recorrido total del precio en la ventana fue desplazamiento neto "
             "direccional frente a idas y vueltas. Como el ER cruza los umbrales "
             "constantemente vela a vela, el precio se colorea por el régimen que "
             "predomina en una ventana local a su alrededor (filtro de moda), usando "
             "como umbrales los terciles (percentiles 33/67) de la propia serie de ER de "
             "este horizonte.",
             "Verde = tramo donde domina la tendencia (movimiento direccional limpio); "
             "gris = transición; rojo = domina el ruido (idas y vueltas que se "
             "cancelan). El panel inferior es el ER real sin suavizar, con su media, "
             "bandas ±1σ y los umbrales de referencia fijos 0.5/0.3.",
             "Sirve para ver en qué tramos el activo se movió en tendencia limpia (donde "
             "un sistema de seguimiento de tendencia, como la KAMA superpuesta, tiene "
             "ventaja) frente a tramos de ruido puro (donde ese mismo sistema generaría "
             "señales falsas); depende de la Ventana seleccionada porque el ER se "
             "recalcula con el periodo de esa ventana.",
             "Bloques verdes largos y contiguos indican periodos de tendencia sostenida "
             "donde seguir al precio (o a la KAMA) habría funcionado; predominio de rojo "
             "en el histórico de este activo/ventana es una señal de que las estrategias "
             "de tendencia probablemente sufran más que las de reversión a la media en "
             "él."),
            _dib_regimen_er, alto=560, depende_ventana=True))
        regimen.cerrar()

        dependencia = _PestanaScroll()
        dependencia.add_seccion(_Seccion(
            "Dependencia y volatility clustering",
            ("Combina varias pruebas de dependencia temporal: la PACF a lag 1 mide si el "
             "retorno de un periodo predice el signo/tamaño del siguiente en distintas "
             "escalas de calendario; el ACF/PACF diario extiende esa medida a 15 "
             "rezagos; el panel de caminos aleatorios simula series de ruido puro con la "
             "misma volatilidad del activo; el último panel calcula el ACF de los "
             "retornos AL CUADRADO (no de los retornos en sí), con el test de "
             "Ljung-Box.",
             "PACF positiva = inercia (el retorno tiende a repetir signo/tamaño); "
             "negativa = reversión; cercana a cero = ruido sin dependencia. Si la línea "
             "de precio real se confunde con las simulaciones aleatorias, el movimiento "
             "no se distingue de un paseo aleatorio. Dependencia en el ACF de retornos² "
             "aunque no la haya en los retornos es volatility clustering: la volatilidad "
             "alta sigue a volatilidad alta, aunque el signo del retorno siga siendo "
             "impredecible.",
             "Sirve para saber si técnicas que asumen retornos independientes son "
             "aplicables aquí, y si ajustar el tamaño de posición por volatilidad "
             "reciente tiene sentido (lo tiene si hay clustering claro).",
             "Un p-valor de Ljung-Box bajo (marcado con ✓) confirma clustering de "
             "volatilidad estadísticamente significativo —rachas de alta/baja "
             "volatilidad reales, no percepción—; si además la línea de precio real se "
             "separa claramente de los caminos aleatorios simulados, el activo tiene "
             "estructura temporal aprovechable más allá del azar puro."),
            _dib_dependencia, alto=700))
        dependencia.cerrar()

        natr = _PestanaScroll()
        natr.add_seccion(_Seccion(
            "Correlación de NATR multi-timeframe",
            ("El NATR (ATR dividido por el precio) de cada temporalidad del activo se "
             "correlaciona (Pearson) con el de las demás temporalidades, formando la "
             "matriz de calor.",
             "Una correlación alta entre dos timeframes indica que sus picos y valles de "
             "volatilidad tienden a coincidir en el tiempo; una baja, que se mueven de "
             "forma relativamente independiente.",
             "Sirve para decidir si vigilar la volatilidad de un timeframe superior "
             "aporta información nueva sobre el que se opera, o si es redundante porque "
             "ambos se mueven prácticamente al unísono.",
             "Correlaciones cercanas a 1 en toda la matriz indican que la volatilidad "
             "del activo es \"una sola cosa\" vista a distintas resoluciones (basta "
             "vigilar un timeframe); bloques con correlación baja entre timeframes muy "
             "separados (p. ej. M1 vs D1) señalan que la volatilidad intradía y la de "
             "swing responden a dinámicas distintas."),
            _dib_natr_corr, alto=460))
        self.tabla_pares = self._crear_tabla_pares()
        natr.add_widget(self._envolver_tabla(self.tabla_pares))
        natr.add_seccion(_Seccion(
            "Dashboard NATR",
            ("A compara el NATR real de cada timeframe contra el que predeciría la ley "
             "√T en escala log-log; B estandariza el NATR actual de cada timeframe "
             "frente a su propia media y desviación histórica (Z-score); C es la serie "
             "histórica de ese Z-score para el par de timeframes elegido; D es el ratio "
             "entre el NATR corto y el largo, con bandas de Bollinger ±2σ.",
             "En A, el NATR real por encima de la curva teórica en los timeframes largos "
             "indica más volatilidad acumulada de la que la independencia explicaría. En "
             "B, Z-score por encima de +2 es volatilidad anormalmente alta para ese "
             "timeframe, por debajo de -2 anormalmente baja. En D, el ratio saliéndose "
             "de las bandas indica que la volatilidad de corto plazo se dispara o se "
             "contrae en relación a la de fondo.",
             "Sirve para saber si el timeframe en el que se opera vive un episodio de "
             "volatilidad extremo respecto a su propia historia, algo que el NATR bruto "
             "de una sola temporalidad no revela. Depende de la Ventana seleccionada.",
             "Barras de Z-score (B) en rojo/verde intenso señalan qué timeframes están "
             "hoy en un extremo estadístico; si el ratio (D) está tocando la banda "
             "superior, la temporalidad corta se está volviendo desproporcionadamente "
             "más volátil que la larga —momento típico de expansión de rango que puede "
             "preceder una vuelta a la normalidad."),
            _dib_dashboard_natr, alto=560, depende_ventana=True))
        natr.cerrar()

        for titulo, pagina in (("  Resumen  ", resumen), ("  Riesgo  ", riesgo),
                               ("  Intradía  ", intradia), ("  Régimen ER  ", regimen),
                               ("  Dependencia  ", dependencia), ("  NATR  ", natr)):
            self._paginas.append(pagina)
            self.tabs.addTab(pagina, titulo)

    def _crear_tabla_pares(self):
        cols = ['Horizonte', 'Par', 'NATR base', 'NATR target', 'Ratio', 'Lead-Lag']
        t = QTableWidget(0, len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setAlternatingRowColors(False)
        t.setSortingEnabled(True)
        # Es una tabla de lectura, no un control: sin foco de teclado las
        # flechas siguen sirviendo para cambiar de pestaña también aquí.
        t.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        t.setMaximumHeight(220)
        return t

    def _envolver_tabla(self, tabla):
        caja = QGroupBox()
        lay = QVBoxLayout(caja)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(6)
        cab = QHBoxLayout()
        lbl = QLabel("Pares de volatilidad")
        lbl.setObjectName("titulo")
        cab.addWidget(lbl)
        cab.addWidget(icono_ayuda(
            "Para cada combinación de dos temporalidades se calcula el NATR de cada "
            "una, el ratio entre ambas y el desfase (lead-lag): cuántos periodos tarda "
            "el NATR de una temporalidad en repetir el movimiento que ya hizo la otra, "
            "estimado por correlación cruzada entre ambas series desplazadas en el "
            "tiempo.",
            "Un lead-lag distinto de cero indica que una temporalidad anticipa "
            "sistemáticamente los cambios de volatilidad de la otra; un lead-lag de "
            "cero indica que se mueven a la vez.",
            "Sirve para saber si conviene vigilar la volatilidad de un timeframe vecino "
            "como aviso temprano antes de que se traslade al timeframe en el que se "
            "opera.",
            "Cada fila es un par ya resuelto: si el \"Lead-Lag\" muestra que un "
            "timeframe mayor anticipa a uno menor en varias velas, un salto de NATR en "
            "el mayor puede leerse como aviso temprano de que la volatilidad en el "
            "menor está a punto de subir."))
        cab.addStretch()
        lay.addLayout(cab)
        lay.addWidget(tabla)
        return caja

    # ── API ──
    def tiene_datos(self):
        return self._bundle is not None

    def cargar(self, bundle, horizonte=None):
        self._bundle = bundle
        if horizonte:
            self._horizonte = horizonte
        for pagina in self._paginas:
            for s in pagina.secciones:
                s.marcar_sucio()
        self._llenar_tabla_pares()
        self._pintar_todo()

    def set_horizonte(self, horizonte):
        if horizonte == self._horizonte:
            return
        self._horizonte = horizonte
        # Solo las secciones que dependen de la Ventana necesitan repintarse;
        # _pintar_todo ya se salta las que sigan limpias (pintar() comprueba
        # _sucio), así que esto no repite trabajo en las demás 9 secciones.
        for pagina in self._paginas:
            for s in pagina.secciones:
                if s.depende_ventana:
                    s.marcar_sucio()
        self._pintar_todo()

    # ── interno ──
    @no_crash
    def _mover_pestana(self, paso):
        """Salta a la pestaña anterior/siguiente, sin dar la vuelta."""
        destino = self.tabs.currentIndex() + paso
        if 0 <= destino < self.tabs.count():
            self.tabs.setCurrentIndex(destino)

    def showEvent(self, event):
        super().showEvent(event)
        # Sin foco dentro del widget los atajos de flecha no se disparan; al
        # entrar en «Gráficos» se le da a la barra de pestañas, que además es
        # donde el usuario espera que esté.
        if not self.isAncestorOf(QApplication.focusWidget() or self):
            self.tabs.setFocus(Qt.FocusReason.OtherFocusReason)

    @no_crash
    def _on_tab_changed(self, idx):
        # Red de seguridad: con _pintar_todo() ya no debería quedar ninguna
        # sección sucia al cambiar de pestaña (por eso el cambio se nota
        # instantáneo), pero si alguna quedó pendiente por lo que sea, aquí
        # se termina de pintar en vez de dejarla en blanco.
        self._pintar_pagina(idx)

    def _pintar_pagina(self, idx):
        if self._bundle is None or not (0 <= idx < len(self._paginas)):
            return
        for s in self._paginas[idx].secciones:
            s.pintar(self._bundle, self._horizonte)

    def _asegurar_cadena_visible(self):
        """Sube por TODOS los QTabWidget antecesores (el `inner_tabs` de
        TabAnalisis, la barra de pestañas exterior de MainWindow, o cualquier
        otro nivel que se añada en el futuro) y selecciona en cada uno la
        página que lleva hasta aquí, si no lo era ya.

        Sin esto, `_pintar_todo()` puede dibujar todo lo que quiera que Qt no
        le va a dar a los canvas su ancho final mientras algún antecesor siga
        mostrando otra pestaña — el árbol entero sigue "oculto" a ojos de Qt
        aunque el propio `self.tabs` esté en el índice correcto. No hay forma
        de saber de antemano cuántos niveles de pestañas hay por encima (hoy
        son dos: `inner_tabs` y la barra de MainWindow), así que se sube sin
        límite comprobando, en cada `QTabWidget` que aparezca, cuál de sus
        páginas contiene a `self` (`isAncestorOf`, válido pase lo que pase de
        anidamiento interno de Qt entre medias).

        Devuelve la lista de `(tabwidget, índice_anterior)` a restaurar
        después, en orden inverso al que se fueron cambiando.
        """
        restaurar = []
        padre = self.parentWidget()
        while padre is not None:
            if isinstance(padre, QTabWidget):
                for i in range(padre.count()):
                    if padre.widget(i).isAncestorOf(self):
                        if padre.currentIndex() != i:
                            restaurar.append((padre, padre.currentIndex()))
                            padre.setCurrentIndex(i)
                            QApplication.processEvents()
                        break
            padre = padre.parentWidget()
        return restaurar

    @no_crash
    def _pintar_todo(self):
        """Pinta TODAS las secciones de golpe (no solo la pestaña visible).

        No basta con llamar a `pintar()` en segundo plano: mientras una
        pestaña sigue oculta, Qt no le ha dado su ancho final todavía (arrastra
        un tamaño por defecto, p. ej. 900px de canvas en vez de los ~1124px
        reales), así que dibujar ahí y mostrar la pestaña después provoca un
        SEGUNDO redibujado a tamaño correcto en cuanto se hace visible — el
        mismo coste que se quería evitar, solo que disfrazado. La única forma
        fiable de que Qt calcule el tamaño real es mostrar la pestaña de
        verdad, así que este método primero asegura que TODO el árbol de
        antecesores esté realmente visible (`_asegurar_cadena_visible`) y
        luego recorre las 6 pestañas propias una a una (`setCurrentIndex` ya
        dispara `_on_tab_changed` → `_pintar_pagina`, que es quien realmente
        dibuja). Al final restaura tanto sus propias pestañas como las de los
        antecesores a como estaban. El usuario ve un parpadeo breve entre
        pestañas mientras el cursor está en modo espera, pero paga el coste
        una sola vez.
        """
        if self._bundle is None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        idx_original = self.tabs.currentIndex()
        try:
            restaurar_antecesores = self._asegurar_cadena_visible()
            for i in range(self.tabs.count()):
                self.tabs.setCurrentIndex(i)
                # Deja respirar al bucle de eventos entre pestaña y pestaña:
                # es el que de verdad hace que Qt calcule el tamaño real de la
                # página y que matplotlib dibuje sus ~11 secciones (estadística
                # no trivial: ACF/PACF, heatmaps...); sin esto Windows podría
                # marcar la ventana como "no responde" durante la carga.
                QApplication.processEvents()
            self.tabs.setCurrentIndex(idx_original)
            QApplication.processEvents()
            for tabwidget, idx_previo in reversed(restaurar_antecesores):
                tabwidget.setCurrentIndex(idx_previo)
                QApplication.processEvents()
        finally:
            QApplication.restoreOverrideCursor()

    @no_crash
    def _llenar_tabla_pares(self):
        t = self.tabla_pares
        t.setSortingEnabled(False)
        t.setRowCount(0)
        pares = ((self._bundle or {}).get('natr_multitf') or {}).get('pares') or []
        t.setRowCount(len(pares))
        for i, p in enumerate(pares):
            celdas = [
                p.get('horizonte', ''),
                p.get('pair', ''),
                f"{p.get('natr_base', 0):.4f}",
                f"{p.get('natr_target', 0):.4f}",
                f"{p.get('ratio', 0):.4f}",
                f"~{p.get('lag', 0)} velas ({p.get('lag_unit', '')}) "
                f"[máx: {p.get('max_lag', 0)} lags]",
            ]
            for j, texto in enumerate(celdas):
                it = QTableWidgetItem(texto)
                if j >= 2:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                        Qt.AlignmentFlag.AlignVCenter)
                t.setItem(i, j, it)
        t.setSortingEnabled(True)

import os, json, re, glob, pickle
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QComboBox, QFrame, QFileDialog, QTextBrowser,
                             QTabWidget, QProgressBar, QScrollArea, QSizePolicy,
                             QMessageBox, QStackedWidget)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
from gui.widgets.pdf_viewer import PdfViewer
from gui.widgets.tab_patrones import TabPatrones
from gui.widgets.analisis_graficos import GraficosAnalisis
from gui.widgets.analisis_tarjetas import TarjetasKPI
from gui.widgets.plot_common import icono_ayuda
from gui.widgets.bombear import bombear_eventos
from core.config import tf_to_minutes

_CHEVRON_SVG = os.path.join(os.path.dirname(__file__), '..', 'assets',
                            'chevron-down.svg').replace('\\', '/')

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
QPushButton#periodo_del {
    background-color: #3a1a1a; color: #e74c3c; padding: 6px 10px; font-size: 12px; min-width: 0;
}
QPushButton#periodo_del:hover { background-color: #4a2525; }
QComboBox {
    background-color: #111828; font-size: 12px; min-width: 140px;
    combobox-popup: 0;
}
QComboBox::drop-down { border: none; background: transparent; width: 22px; }
QComboBox::down-arrow {
    image: url("__CHEVRON__");
    width: 12px; height: 8px;
}
QComboBox::down-arrow:on { }
QComboBox QAbstractItemView {
    background-color: #1a2a45; color: #c8d6e5;
    selection-background-color: #2a4a6a; selection-color: #4fc3f7;
    border: 1px solid #253a60; outline: none; margin: 0px;
}
QComboBox QAbstractItemView::item {
    padding: 6px 10px; border-bottom: 1px solid #182030;
}
QComboBox QAbstractItemView::item:selected {
    background-color: #2a4a6a; color: #4fc3f7;
}
QComboBox QAbstractItemView::item:hover {
    background-color: #223755;
}
QComboBox QAbstractItemView::item:disabled {
    background-color: #0d1424; color: #3a5a7a;
}
QComboBox QAbstractItemView::item:disabled:hover {
    background-color: #0d1424;
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
QGroupBox {
    background-color: #141e30; border: 1px solid #253a60;
    border-radius: 6px; margin-top: 0px; padding: 10px;
}
QLabel#titulo { color: #4fc3f7; font-size: 12px; font-weight: bold; }
QLabel#avisoLegacy {
    background-color: #2a2416; color: #d4a03a;
    border: 1px solid #4a3a1a; border-radius: 4px;
    padding: 6px 10px; font-size: 11px;
}
QTableWidget {
    background-color: #0d1424; color: #c8d6e5; gridline-color: #253a60;
    border: 1px solid #253a60; font-size: 11px;
}
QToolTip {
    background-color: #101a2e; color: #c8d6e5;
    border: 1px solid #253a60; border-radius: 4px;
    padding: 6px; font-size: 11px;
}
QHeaderView::section {
    background-color: #1a2a45; color: #8fb3d9; border: none;
    border-right: 1px solid #253a60; padding: 4px; font-size: 10px; font-weight: bold;
}
"""

_ANSI_RE = re.compile(r'\033\[([\d;]+)m')

ANSI_COLORS = {
    91:  '#e74c3c',
    92:  '#27ae60',
    93:  '#f1c40f',
    94:  '#3498db',
    95:  '#9b59b6',
    96:  '#1abc9c',
    97:  '#e0e0e0',
}

BAR_RE = re.compile(r'^([█░]{12})\s+(.*)$')

RANGO_RE = re.compile(r'\.analysis\.(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})\.pdf$')

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


# ══════════════════════════════════════════════════════════════════
#  Ayuda por categoría de métricas
# ══════════════════════════════════════════════════════════════════
# Cada entrada son las 4 pestañas del popup de `icono_ayuda`:
# (Lógica, Significado, Uso, Resultados). La clave es el prefijo numérico
# del título — no el título entero — porque la GUI le añade el sufijo
# " — <Ventana>" cuando la categoría tiene filas por horizonte
# (ver _render_metrics). Los títulos se generan en
# library/scripts_utiles/analisis_descriptivo.py.
AYUDA_CATEGORIAS = {
    '1': (
        "Lee la primera y la última marca de tiempo del archivo, cuenta las velas "
        "y deduce el tipo de muestreo: temporal (velas de intervalo fijo) o por "
        "ticks (una fila por operación, sin reloj regular).",
        "Define el suelo estadístico del resto del informe. Un histórico corto o "
        "con huecos hace que todo lo que venga después —Hurst, ADF, VaR— se "
        "calcule sobre pocas observaciones y sea inestable.",
        "Comprueba esto antes de fiarte de ninguna otra categoría. Por debajo de "
        "unas 500 velas, los tests estadísticos no tienen potencia suficiente y "
        "sus veredictos son ruido.",
        "«Periodo» marca el rango real analizado, que puede ser menor que el "
        "archivo si recortaste el rango al analizar. «Tipo de muestreo» avisa si "
        "la serie es de ticks: en ese caso las métricas anualizadas y el Calmar "
        "salen N/A porque no hay periodicidad que anualizar.",
    ),
    '2': (
        "El CAGR se obtiene componiendo el retorno total sobre los años que cubre "
        "el histórico. La media y la mediana son de los retornos logarítmicos vela "
        "a vela. La alineación de marcos temporales puntúa de 0 a 3 cuántas "
        "temporalidades superiores apuntan en la misma dirección.",
        "Separa cuánto sube el activo de cómo lo sube. Una media positiva con "
        "mediana negativa significa que el rendimiento vive en pocas velas "
        "extremas: la mayoría de las velas pierden y unas pocas lo compensan.",
        "Fija la expectativa de una estrategia larga pura. Si el buy & hold ya da "
        "el CAGR que buscas, cualquier sistema tiene que batirlo después de "
        "comisiones para justificar el riesgo operativo.",
        "Compara «Media retorno» con «Mediana retorno»: cuanto más se separen, "
        "más dependes de colas. «Retornos positivos» cerca del 50% es lo normal —"
        " lo que genera el rendimiento es el tamaño de las velas, no el reparto. "
        "«Alineación de marcos temporales» en 3/3 indica tendencia limpia en todas "
        "las escalas.",
    ),
    '3': (
        "La volatilidad histórica es la desviación típica de los retornos "
        "logarítmicos, anualizada. Se recalcula en ventanas móviles de 7, 30, 90 y "
        "365 días y cada una se divide por la volatilidad total, así que el valor "
        "que ves es un múltiplo: 1.00x es volatilidad normal para este activo.",
        "Sitúa el momento actual dentro de su propia historia. Por encima de 1.30x "
        "el activo está más nervioso de lo habitual; por debajo de 0.70x está "
        "comprimido y suele preceder a una expansión.",
        "Ajusta el tamaño de posición y la distancia del stop. Con la HV corta muy "
        "por encima de la larga, los stops calculados con medias largas se quedan "
        "cortos y saltan por ruido.",
        "Lee las cuatro HV como una escalera temporal: si HV 7d > HV 30d > HV 365d, "
        "la volatilidad está expandiéndose ahora mismo; el orden inverso indica "
        "calma reciente. «Desviación Escalado Fractal» mide cuánto se aleja la "
        "serie de la ley √T; un valor alto avisa de que estimar la volatilidad de "
        "un timeframe escalando otro te dará un número equivocado.",
    ),
    '4': (
        "Los tres ratios dividen exceso de retorno por una medida de riesgo "
        "distinta: Sharpe por la volatilidad total, Sortino solo por la "
        "volatilidad de las caídas, Calmar por el máximo drawdown. La tasa libre "
        "de riesgo es el T-Bill a 3 meses, indicado en la propia fila.",
        "Cada uno castiga algo diferente. Sharpe penaliza también las subidas "
        "bruscas; Sortino no. Sortino muy por encima de Sharpe significa que la "
        "volatilidad del activo es mayoritariamente al alza.",
        "Es la vara de medir contra la que compararás cualquier estrategia del "
        "Backtester. Un sistema con Sharpe por debajo del que ya da el activo "
        "comprado y mantenido no aporta nada.",
        "Los valores vienen coloreados: verde por encima de 1, amarillo entre 0.5 "
        "y 1, rojo por debajo. Mira el Calmar con más peso si operas apalancado —"
        " es el único de los tres que mide el riesgo con la peor pérdida real y no "
        "con una desviación típica.",
    ),
    '5': (
        "El drawdown es la caída desde cada máximo histórico. El máximo histórico "
        "recorre toda la serie; los internos se calculan reiniciando el máximo "
        "cada día y cada semana. Un episodio es un tramo completo desde que se "
        "pierde un máximo hasta que se recupera.",
        "Es la pérdida que habrías tenido que aguantar sin vender. El máximo "
        "histórico dice cuánto llegó a caer; el tiempo de recuperación dice cuánto "
        "tiempo estuviste en pérdidas, que es lo que de verdad rompe la disciplina.",
        "Determina el capital mínimo y el apalancamiento máximo tolerable. Con un "
        "drawdown histórico del 40%, un apalancamiento x3 liquida la cuenta antes "
        "de llegar al suelo.",
        "«Max Drawdown Interno (Diario)» trae la fecha del peor día entre "
        "paréntesis: úsala para ir al gráfico y ver qué pasó. Compara «Drawdown "
        "medio» con el máximo — si están cerca, las caídas son consistentes; si el "
        "máximo lo dobla, hubo un evento aislado que no representa el "
        "comportamiento normal.",
    ),
    '6': (
        "El VaR se calcula por percentiles empíricos de los retornos, no asumiendo "
        "normalidad: el VaR 95% es el percentil 5 de la distribución real. "
        "Skewness y kurtosis describen la forma de esa distribución, y el test de "
        "Jarque-Bera contrasta si es compatible con una normal.",
        "Marca la pérdida esperada en el peor 5% y el peor 1% de las velas. La "
        "kurtosis alta —por encima de 3— significa colas gordas: los eventos "
        "extremos son mucho más frecuentes de lo que predice la campana de Gauss.",
        "Dimensiona el riesgo por operación. Si tu stop está dentro del VaR 95%, "
        "saltará de forma rutinaria solo por el movimiento normal del activo, sin "
        "que la tesis haya fallado.",
        "«Distribucion normal: NO (fat tails)» es el resultado habitual en "
        "mercados y no es un error: confirma que no puedes usar fórmulas basadas "
        "en la normal para calcular riesgo. Skewness negativa indica que las "
        "caídas son más bruscas que las subidas.",
    ),
    '7': (
        "El Efficiency Ratio divide el desplazamiento neto del precio entre la "
        "suma de los recorridos vela a vela: mide cuánto camino se aprovecha. El "
        "exponente de Hurst mide si la serie tiene memoria, estimado sobre "
        "ventanas móviles. Ambos se recalculan con los periodos propios de cada "
        "ventana temporal cuando seleccionas un horizonte.",
        "ER por encima de 0.5 es movimiento direccional; por debajo de 0.3 es "
        "ruido lateral. Hurst por encima de 0.58 indica persistencia (lo que sube "
        "tiende a seguir subiendo); por debajo de 0.52, reversión a la media. "
        "Cerca de 0.5 la serie es un paseo aleatorio y no hay señal que extraer.",
        "Es la categoría que decide qué familia de estrategia usar. Con dominio de "
        "régimen tendencial, funcionan rupturas y seguimiento; con dominio de "
        "reversión, funcionan bandas y sobreventa. Aplicar la equivocada pierde "
        "dinero aunque la ejecución sea perfecta.",
        "Mira las barras de reparto de periodos antes que las medias: te dicen "
        "cuánto tiempo pasa el activo en cada régimen. «Mejora Sharpe (Tend vs "
        "Rev)» resuelve la elección de un vistazo — positiva favorece seguir "
        "tendencia, negativa favorece revertir. «Duración media racha tendencial» "
        "te da el horizonte de mantenimiento realista en velas.",
    ),
    '8': (
        "Correlaciona el retorno con la volatilidad en ventana móvil de unos 7 "
        "días, y calcula la volatilidad media por franja horaria agrupando las "
        "velas en las sesiones de Tokio, Londres y Nueva York en hora local de "
        "cada plaza.",
        "Una correlación retorno-volatilidad negativa es el patrón clásico de "
        "pánico: el precio cae y la volatilidad se dispara a la vez. Positiva "
        "indica euforia, con la volatilidad creciendo en las subidas. La "
        "volatilidad por sesión, expresada como múltiplo, señala en qué horas se "
        "mueve de verdad el activo.",
        "Define el horario operativo y el sesgo direccional. Operar rupturas en la "
        "sesión de menor volatilidad relativa produce falsos rompimientos "
        "sistemáticos.",
        "«Tiempo corr. negativa (%)» va coloreado: por encima del 50% en rojo, "
        "porque significa que el activo pasa la mayor parte del tiempo en régimen "
        "de miedo. Las tres barras de sesión se leen en múltiplos: 1.50x en "
        "Londres significa que allí se mueve un 50% más que en su promedio "
        "general.",
    ),
    '9': (
        "La autocorrelación parcial (PACF) en lag 1 mide cuánto explica el retorno "
        "de un periodo el retorno del periodo siguiente, aislando el efecto de los "
        "lags intermedios. Se calcula reagregando la serie a escala diaria, "
        "semanal, mensual y trimestral, y se compara con el umbral de "
        "significancia estadística que aparece en la propia categoría.",
        "Por encima del umbral hay memoria real: el pasado inmediato contiene "
        "información sobre el futuro inmediato. Por debajo, la serie es ruido "
        "blanco en esa escala y ningún modelo autorregresivo va a funcionar ahí.",
        "Elige la temporalidad en la que operar. Opera en la escala donde la "
        "dependencia supera el umbral e ignora las escalas marcadas como "
        "ruido, por muy limpio que se vea el gráfico.",
        "Compara cada «Dependencia» con «Significancia (Umbral)»: solo cuentan las "
        "que lo superan en valor absoluto. «Memoria Estructural: Débil/Ruido» "
        "significa que no hay dependencia lineal explotable a escala diaria — no "
        "que no haya nada, pero sí que un modelo lineal no lo capturará.",
    ),
    '10': (
        "Aplica autocorrelación a los retornos al cuadrado en lag 1 y confirma con "
        "el test de Ljung-Box, cuya hipótesis nula es que no hay agrupación. Se "
        "usan los retornos al cuadrado porque eliminan el signo y dejan solo la "
        "magnitud del movimiento.",
        "Detecta si la volatilidad se agrupa: días agitados seguidos de días "
        "agitados, calma seguida de calma. Un p-valor por debajo de 0.05 rechaza "
        "la nula y confirma el efecto ARCH.",
        "Cuando hay clustering, la volatilidad es predecible aunque la dirección no "
        "lo sea. Eso permite ajustar el tamaño de posición de forma anticipada: "
        "reducir exposición tras una vela extrema, porque vienen más.",
        "«Clustering detectado: SÍ» es lo habitual y valida usar stops adaptativos "
        "por ATR en lugar de stops de distancia fija. Con un «NO», la volatilidad "
        "es impredecible y conviene dimensionar por riesgo fijo.",
    ),
    '11': (
        "Parkinson usa el rango máximo-mínimo; Garman-Klass añade apertura y "
        "cierre; Rogers-Satchell corrige además la tendencia del periodo. Los tres "
        "se comparan contra el Close-to-Close, que solo mira cierres y descarta "
        "todo lo que pasó dentro de la vela.",
        "La eficiencia indica cuántas veces menos datos necesita ese estimador "
        "para la misma precisión que el Close-to-Close. Si los estimadores OHLC "
        "dan mucho más que el CtC, el precio se mueve mucho dentro de la vela y "
        "vuelve al cierre: hay recorrido que el cierre no ve.",
        "Elige con qué volatilidad calculas stops y objetivos. Para scalping "
        "importa el rango intravela, no la variación entre cierres — usar CtC "
        "subestima el riesgo real de que te salte el stop.",
        "«Estimador recomendado» ya resuelve la elección: Rogers-Satchell cuando "
        "hay tendencia clara, porque es el único que no confunde deriva con "
        "volatilidad. Una gran diferencia entre Parkinson y CtC señala mechas "
        "largas y por tanto stops que necesitan más holgura.",
    ),
    '12': (
        "ADF y KPSS se aplican dos veces: sobre el precio y sobre los retornos. "
        "ADF plantea H0 = raíz unitaria, es decir serie no estacionaria; KPSS "
        "plantea la hipótesis contraria, H0 = serie estacionaria. Se contrastan "
        "los dos porque cada uno falla justo en el caso que el otro detecta.",
        "El precio casi siempre sale no estacionario y los retornos estacionarios: "
        "eso es lo normal y confirma que los datos son sanos. Un precio "
        "estacionario indica un activo que revierte a un nivel; unos retornos no "
        "estacionarios indican cambio de régimen dentro del periodo analizado.",
        "Decide sobre qué serie se opera. Con precio no estacionario, las "
        "estrategias de reversión sobre el precio crudo no tienen anclaje y hay "
        "que trabajar con spreads, ratios o retornos. Con precio estacionario, la "
        "reversión a la media es explotable directamente.",
        "Lee primero los dos «Veredicto»: resumen ADF y KPSS ya combinados. Los "
        "p-valores están debajo por si el veredicto queda en el límite — ADF "
        "p < 0.05 rechaza la raíz unitaria; KPSS p < 0.05 rechaza la "
        "estacionariedad.",
    ),
    '13': (
        "Ajusta un modelo de Ornstein-Uhlenbeck sobre el logaritmo del precio: "
        "regresa la variación de cada periodo contra el nivel anterior. El "
        "coeficiente beta es la velocidad de reversión, y la vida media sale de "
        "ln(2) dividido por esa velocidad.",
        "La vida media es cuántas velas tarda una desviación en corregirse a la "
        "mitad. Beta positiva significa que la serie se aleja en vez de volver: no "
        "hay reversión y el dato sale N/A.",
        "Fija la duración de las operaciones de reversión y el vencimiento de las "
        "órdenes. Mantener una posición de reversión mucho más allá de la vida "
        "media convierte una operación estadística en una apuesta direccional.",
        "Usa «Half-Life (velas)» directamente como tiempo máximo de mantenimiento. "
        "«Sin reversión detectada» no invalida el activo: significa que hay que "
        "operarlo con seguimiento de tendencia, no con reversión.",
    ),
    '14': (
        "Calcula el ATR normalizado —ATR dividido por el precio, en porcentaje— en "
        "todas las temporalidades disponibles, para que sean comparables entre sí. "
        "Después empareja timeframes y mide el ratio entre sus NATR y el desfase "
        "lead-lag por correlación cruzada.",
        "Al estar normalizado, el NATR permite comparar volatilidad entre "
        "temporalidades y entre activos de precio muy distinto. El lead-lag "
        "distinto de cero indica que una temporalidad anticipa los cambios de "
        "volatilidad de la otra.",
        "Elige el timeframe de trabajo por volatilidad disponible: por debajo de "
        "cierto NATR el recorrido no cubre spread y comisiones. El lead-lag te da "
        "un aviso temprano desde un timeframe vecino antes de que la volatilidad "
        "llegue al tuyo.",
        "Compara los NATR entre timeframes: deberían crecer con la raíz del "
        "tiempo. Un par cuyo ratio se aleja mucho de esa proporción señala que una "
        "escala está anormalmente agitada o anormalmente plana respecto a la otra.",
    ),
    '14.5': (
        "Estandariza el NATR de cada temporalidad en un Z-score frente a su propia "
        "media histórica, y encierra el ratio entre pares de timeframes en bandas "
        "de Bollinger de ±2 desviaciones típicas. Todo se recalcula por horizonte, "
        "con los pares propios de cada ventana.",
        "El Z-score dice cuán extrema es la volatilidad de ahora comparada con lo "
        "normal en ese timeframe: por encima de +2 es un extremo alto, por debajo "
        "de -2 es compresión. Un ratio fuera de sus bandas indica que la relación "
        "de volatilidad entre las dos escalas se ha roto.",
        "Sirve para cronometrar la entrada. La compresión extrema precede a las "
        "expansiones, así que un Z-score muy negativo favorece estrategias de "
        "ruptura; uno muy positivo avisa de que llegas tarde y con stops caros.",
        "Lee el signo y la magnitud del Z-score, no el valor del NATR en bruto. En "
        "el ratio, compara «Actual» con «BB Sup» y «BB Inf»: dentro de las bandas "
        "la relación entre timeframes es la habitual; fuera, uno de los dos está "
        "en un régimen distinto y las señales cruzadas entre ellos dejan de ser "
        "fiables.",
    ),
}

# Ayuda del selector «Ventana» de la barra superior. Mismo formato de 4
# pestañas: es el control que reinterpreta todo el informe, así que se explica
# con el mismo detalle que una categoría.
AYUDA_VENTANA = (
    "Selecciona el horizonte de operativa con el que se lee el análisis. No "
    "recorta los datos: reetiqueta el informe usando los periodos de indicador "
    "propios de esa ventana —ER, KAMA, Hurst, ventanas de correlación y pares "
    "de NATR— que ya se calcularon al analizar el activo. En «General» se "
    "muestran los valores con los parámetros genéricos.",
    "Cada ventana es un perfil temporal: Scalping trabaja con TF de 15m o "
    "menos, Daytrading hasta 1h, Swingtrading cubre todas las temporalidades y "
    "Position parte de 4h. Las ventanas incompatibles con el timeframe del "
    "activo cargado salen en gris y no se pueden elegir — no tiene sentido "
    "pedir métricas de scalping sobre velas diarias.",
    "Cámbiala para comprobar si el activo se comporta igual en distintas "
    "escalas antes de montar una estrategia. Un activo puede ser tendencial en "
    "Position y puro ruido en Scalping: esa diferencia decide en qué "
    "temporalidad operas.",
    "Los gráficos y las tarjetas se actualizan al instante, pero la pestaña "
    "Métricas exige pulsar «Aplicar». Cuando la ventana no es General, las "
    "categorías afectadas añaden su nombre al título («… — Scalping») para que "
    "veas cuáles cambiaron de verdad; las categorías 11, 12 y 13 solo aparecen "
    "en General, Scalping y Daytrading, porque necesitan más muestras de las "
    "que dejan las ventanas largas.",
)

_CLAVE_CAT_RE = re.compile(r'^(\d+(?:\.\d+)?)\.')


def _ayuda_categoria(titulo):
    """Ayuda de una categoría a partir de su título, tolerando el sufijo
    " — <Ventana>" que añade _render_metrics. Devuelve None si la categoría no
    tiene texto asociado (categorías nuevas: mejor sin icono que con un popup
    vacío)."""
    m = _CLAVE_CAT_RE.match(titulo.strip())
    return AYUDA_CATEGORIAS.get(m.group(1)) if m else None


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

        # El icono va pegado al texto (no al borde del panel), igual que en las
        # secciones de graficos: el stretch va detras.
        fila_cab = QHBoxLayout()
        fila_cab.setContentsMargins(0, 0, 0, 0)
        fila_cab.setSpacing(6)
        fila_cab.addWidget(header)
        ayuda = _ayuda_categoria(title)
        if ayuda:
            fila_cab.addWidget(icono_ayuda(*ayuda))
        fila_cab.addStretch()
        layout.addLayout(fila_cab)

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
        self.setStyleSheet(STYLE_ANALISIS.replace('__CHEVRON__', _CHEVRON_SVG))
        self._pdf_path = None
        self._page_map = None
        self._metrics_path = None
        self._all_metrics = None
        self._bundle = None
        self._ticker = ''
        self._tf = ''
        self._csv_path = None

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
        self.horizon.currentIndexChanged.connect(self._on_horizon_changed)
        toolbar.addWidget(self.horizon)

        # Detras del combo, no entre la etiqueta y el, para no separar el
        # control de su nombre.
        self.icono_ventana = icono_ayuda(*AYUDA_VENTANA)
        toolbar.addWidget(self.icono_ventana)

        lbl_periodo_combo = QLabel("Periodo")
        lbl_periodo_combo.setStyleSheet("color: #aabbcc; font-size: 11px; font-weight: bold; padding-right: 4px; padding-left: 12px;")
        toolbar.addWidget(lbl_periodo_combo)

        self.periodo_combo = QComboBox()
        self.periodo_combo.setToolTip("Periodo analizado disponible para este activo")
        self.periodo_combo.currentIndexChanged.connect(self._on_periodo_combo_changed)
        toolbar.addWidget(self.periodo_combo)

        self.btn_periodo_del = QPushButton("✕")
        self.btn_periodo_del.setObjectName("periodo_del")
        self.btn_periodo_del.setFixedWidth(28)
        self.btn_periodo_del.setToolTip("Eliminar este periodo analizado")
        self.btn_periodo_del.clicked.connect(self._eliminar_periodo_actual)
        toolbar.addWidget(self.btn_periodo_del)

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

        # ── Metricas: tarjetas KPI arriba, categorias completas debajo ──
        pagina_metricas = QWidget()
        lay_metricas = QVBoxLayout(pagina_metricas)
        lay_metricas.setContentsMargins(20, 14, 20, 0)
        lay_metricas.setSpacing(8)
        self.kpi_cards = TarjetasKPI()
        self.kpi_cards.setVisible(False)
        lay_metricas.addWidget(self.kpi_cards)
        self.metrics_scroll = MetricsScroll()
        lay_metricas.addWidget(self.metrics_scroll, 1)
        self.inner_tabs.addTab(pagina_metricas, "  Metricas  ")
        bombear_eventos()

        # ── Graficos: nativos si el analisis trae datos, visor PDF si no ──
        # Los informes generados antes de esta version solo tienen el PDF; en
        # ese caso se sigue mostrando el visor de siempre con un aviso, en vez
        # de dejar la pestana vacia.
        self.graphs_stack = QStackedWidget()

        self.graphs_native = GraficosAnalisis()
        self.graphs_stack.addWidget(self.graphs_native)          # indice 0
        bombear_eventos()

        pagina_pdf = QWidget()
        lay_pdf = QVBoxLayout(pagina_pdf)
        lay_pdf.setContentsMargins(0, 0, 0, 0)
        lay_pdf.setSpacing(6)
        self.lbl_legacy = QLabel(
            "Este analisis se genero con una version anterior: se muestran las "
            "paginas del PDF. Vuelve a analizar el activo para ver los graficos "
            "interactivos.")
        self.lbl_legacy.setObjectName("avisoLegacy")
        self.lbl_legacy.setWordWrap(True)
        lay_pdf.addWidget(self.lbl_legacy)
        self.graphs_viewer = PdfViewer()
        lay_pdf.addWidget(self.graphs_viewer, 1)
        self.graphs_stack.addWidget(pagina_pdf)                   # indice 1

        self.graphs_stack.setCurrentIndex(1)
        self.inner_tabs.addTab(self.graphs_stack, "  Graficos  ")

        self.patterns_tab = TabPatrones()
        self.inner_tabs.addTab(self.patterns_tab, "  Patrones de velas  ")
        bombear_eventos()

        layout.addWidget(self.inner_tabs, 1)

    @property
    def current_horizon(self):
        return self.horizon.currentText()

    def _update_horizon_items(self, tf):
        print(f"[DEBUG tab_analisis] _update_horizon_items(tf={tf!r})", flush=True)
        for i in range(self.horizon.count()):
            item = self.horizon.model().item(i)
            if item is None:
                continue
            item.setEnabled(True)
            item.setData(QColor('#c8d6e5'), Qt.ItemDataRole.ForegroundRole)
        if not tf:
            print("[DEBUG tab_analisis] tf vacio, sin deshabilitar", flush=True)
            return

        try:
            self._aplicar_umbrales_horizon(tf)
        except Exception:
            import traceback
            print(f"[ERROR tab_analisis] fallo al aplicar umbrales de horizonte para tf={tf!r}:", flush=True)
            traceback.print_exc()

    def _deshabilitar_horizon_item(self, idx):
        item = self.horizon.model().item(idx)
        if item is None:
            print(f"[DEBUG tab_analisis] idx={idx} sin item en el modelo, se ignora", flush=True)
            return
        item.setEnabled(False)
        item.setData(QColor('#3a5a7a'), Qt.ItemDataRole.ForegroundRole)
        if self.horizon.currentIndex() == idx:
            self.horizon.setCurrentIndex(0)
            print(f"[DEBUG tab_analisis] fallback a General (idx 0)", flush=True)

    def _aplicar_umbrales_horizon(self, tf):
        tf_minutos = tf_to_minutes(tf)
        if tf_minutos is None:
            try:
                tf_minutos = float(tf)
            except (ValueError, TypeError):
                print(f"[DEBUG tab_analisis] tf={tf!r} no convertible a minutos, sin deshabilitar", flush=True)
                return
        print(f"[DEBUG tab_analisis] tf_minutos={tf_minutos}", flush=True)

        umbrales = {1: 6, 2: 120, 3: 300}
        excepciones = {3: {1440}}  # 1d sigue disponible en Swingtrading
        for idx, umbral in umbrales.items():
            if tf_minutos >= umbral and tf_minutos not in excepciones.get(idx, ()):
                print(f"[DEBUG tab_analisis] deshabilitado idx={idx} (>= {umbral} min)", flush=True)
                self._deshabilitar_horizon_item(idx)

        UMBRAL_POSITION_MIN = 43200  # 1 mes
        if tf_minutos > UMBRAL_POSITION_MIN:
            print(f"[DEBUG tab_analisis] deshabilitado idx=4 (> {UMBRAL_POSITION_MIN} min)", flush=True)
            self._deshabilitar_horizon_item(4)

    def _on_horizon_changed(self, idx):
        # El visor es adaptativo: al cambiar la Ventana solo se muestran las
        # páginas generales + las etiquetadas con el horizonte seleccionado.
        self._aplicar_filtro_paginas()
        horizon = self.horizon.currentText() if self.horizon else 'General'
        self.graphs_native.set_horizonte(horizon)
        self.kpi_cards.set_horizonte(horizon)

    def _aplicar_filtro_paginas(self):
        if not self._pdf_path:
            return
        if not self._page_map:
            # Resultado antiguo sin mapa de páginas: mostrar todo
            self.graphs_viewer.set_visible_pages(None)
            return
        horizon = self.horizon.currentText() if self.horizon else 'General'
        visibles = [p['pagina'] for p in self._page_map
                    if p.get('horizonte') in (None, horizon)]
        self.graphs_viewer.set_visible_pages(visibles)

    def preview_horizon_for(self, nombre, tf):
        print(f"[DEBUG tab_analisis] preview_horizon_for(nombre={nombre!r} tf={tf!r})", flush=True)
        self._ticker = nombre or ''
        self._tf = tf or ''
        self.lbl_asset.setText(f"{nombre} {tf}" if nombre and tf else "Sin activo")
        self._update_horizon_items(tf)

    def load_results(self, pdf_path, metrics_path, ticker, tf, csv_path=None):
        print(f"[DEBUG tab_analisis] load_results(ticker={ticker!r} tf={tf!r})", flush=True)
        self._ticker = ticker or ''
        self._tf = tf or ''
        self._csv_path = csv_path or None
        self.patterns_tab.set_source(self._csv_path, ticker, tf)

        self.lbl_asset.setText(f"{ticker} {tf}" if ticker and tf else "Sin activo")
        self._update_horizon_items(tf)

        self._poblar_periodo_combo(pdf_path)
        self._aplicar_resultado(pdf_path, metrics_path)

    def _poblar_periodo_combo(self, pdf_path_actual):
        self.periodo_combo.blockSignals(True)
        self.periodo_combo.clear()

        entries = []
        if self._csv_path:
            for pdf_cand in sorted(glob.glob(self._csv_path + '.analysis.*_to_*.pdf')):
                m = RANGO_RE.search(pdf_cand)
                if not m:
                    continue
                inicio, fin = m.group(1), m.group(2)
                metrics_cand = self._csv_path + f'.analysis.{inicio}_to_{fin}.metrics.json'
                label = f"{inicio} → {fin}"
                entries.append((label, pdf_cand, metrics_cand if os.path.exists(metrics_cand) else None))

        if not entries and pdf_path_actual:
            entries.append(("Actual", pdf_path_actual, self._metrics_path))

        selected_idx = 0
        for i, (label, pdf_cand, metrics_cand) in enumerate(entries):
            self.periodo_combo.addItem(label, (pdf_cand, metrics_cand))
            if pdf_cand == pdf_path_actual:
                selected_idx = i

        if entries:
            self.periodo_combo.setCurrentIndex(selected_idx)
        self.periodo_combo.blockSignals(False)
        self._actualizar_estado_btn_periodo_del()

    def _on_periodo_combo_changed(self, idx):
        data = self.periodo_combo.itemData(idx)
        if not data:
            return
        pdf_cand, metrics_cand = data
        self._aplicar_resultado(pdf_cand, metrics_cand)
        self._actualizar_estado_btn_periodo_del()

    def _actualizar_estado_btn_periodo_del(self):
        data = self.periodo_combo.currentData()
        puede_eliminar = bool(data and data[0] and RANGO_RE.search(data[0]))
        self.btn_periodo_del.setEnabled(puede_eliminar)

    def _eliminar_periodo_actual(self):
        idx = self.periodo_combo.currentIndex()
        if idx < 0:
            return
        data = self.periodo_combo.itemData(idx)
        if not data:
            return
        pdf_cand, metrics_cand = data
        if not pdf_cand or not RANGO_RE.search(pdf_cand):
            return

        label = self.periodo_combo.itemText(idx)
        reply = QMessageBox.question(
            self, "Confirmar eliminación",
            f"¿Eliminar el análisis del periodo {label}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Liberar el PDF actual del visor ANTES de borrarlo: en Windows el
        # archivo queda bloqueado mientras el visor lo tiene abierto.
        self.graphs_viewer.load(None)

        for f in (pdf_cand, metrics_cand, self._ruta_bundle(pdf_cand)):
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

        self._poblar_periodo_combo(None)
        siguiente = self.periodo_combo.currentData()
        if siguiente:
            self._aplicar_resultado(*siguiente)
        else:
            self._aplicar_resultado(None, None)

    def _aplicar_resultado(self, pdf_path, metrics_path):
        self._pdf_path = pdf_path if pdf_path and os.path.exists(pdf_path) else None
        self._metrics_path = metrics_path if metrics_path and os.path.exists(metrics_path) else None

        self.btn_export.setEnabled(self._pdf_path is not None)

        try:
            if self._metrics_path:
                with open(self._metrics_path, 'r', encoding='utf-8') as f:
                    self._all_metrics = json.load(f)
            else:
                self._all_metrics = None
        except Exception:
            self._all_metrics = None

        # Mapa página→horizonte (clave reservada del script); pop para que no
        # se pinte como categoría. Ausente en resultados antiguos → sin filtro.
        self._page_map = None
        if isinstance(self._all_metrics, dict):
            self._page_map = self._all_metrics.pop('_paginas', None)

        self._bundle = self._cargar_bundle(self._pdf_path)
        horizon = self.horizon.currentText() if self.horizon else 'General'
        self.kpi_cards.cargar(self._bundle, horizon)

        self._render_metrics()

        if self._bundle is not None:
            self.graphs_stack.setCurrentIndex(0)
            # graphs_native.cargar() dibuja TODAS sus secciones de golpe y se
            # asegura de que toda la cadena de pestañas antecesoras (esta
            # misma, la de MainWindow...) esté realmente visible mientras
            # dibuja, para que Qt le dé a cada canvas su ancho final — ver
            # GraficosAnalisis._pintar_todo / _asegurar_cadena_visible.
            self.graphs_native.cargar(self._bundle, horizon)
            # Liberar el PDF del visor: mientras QPdfDocument lo tenga abierto,
            # Windows no deja borrarlo ni sobrescribirlo en el próximo análisis.
            self.graphs_viewer.load(None)
        else:
            self.graphs_stack.setCurrentIndex(1)
            if self._pdf_path:
                self.graphs_viewer.load(self._pdf_path)
                self._aplicar_filtro_paginas()

        if self._pdf_path:
            self.inner_tabs.setCurrentIndex(0)

    def _ruta_bundle(self, pdf_path):
        """Sidecar de datos de gráficos que acompaña a un PDF de análisis."""
        if not pdf_path:
            return None
        if pdf_path.endswith('.pdf'):
            return pdf_path[:-4] + '.plotdata.pkl'
        return None

    def _cargar_bundle(self, pdf_path):
        ruta = self._ruta_bundle(pdf_path)
        if not ruta or not os.path.exists(ruta):
            return None
        try:
            with open(ruta, 'rb') as f:
                bundle = pickle.load(f)
        except Exception:
            import traceback
            print(f"[ERROR tab_analisis] no se pudo leer {ruta}:", flush=True)
            traceback.print_exc()
            return None
        if not isinstance(bundle, dict) or bundle.get('_version') != 1:
            print(f"[DEBUG tab_analisis] bundle ignorado (version incompatible): {ruta}",
                  flush=True)
            return None
        return bundle

    def _render_metrics(self):
        if not self._all_metrics:
            self.metrics_scroll.clear()
            self.lbl_period.setText("")
            return

        horizon = self.horizon.currentText() if self.horizon else 'General'
        metricas = {}

        for cat, items in self._all_metrics.items():
            filtered = {}
            tiene_horizonte = False
            for k, v in items.items():
                if not k.startswith('['):
                    filtered[k] = v
                elif k.startswith(f'[{horizon}]'):
                    clean_k = k[len(f'[{horizon}]'):].lstrip()
                    suffix = f'({horizon})'
                    if clean_k.endswith(suffix):
                        clean_k = clean_k[:-len(suffix)].rstrip()
                    filtered[clean_k] = v
                    tiene_horizonte = True
            if filtered:
                titulo = f'{cat} — {horizon}' if (tiene_horizonte and horizon != 'General') else cat
                metricas[titulo] = filtered

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
        # Incluir el rango analizado en el nombre sugerido, extrayéndolo del
        # nombre del PDF de origen (p.ej. informe_xauusd_1h_2020-03-13_to_2025-06-12.pdf
        # o caché sidecar ...csv.analysis.2020-03-13_to_2025-06-12.pdf).
        m = re.search(r'(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})',
                      os.path.basename(self._pdf_path))
        rango = f"_{m.group(1)}_to_{m.group(2)}" if m else ""
        horizon = self.horizon.currentText() if self.horizon else 'General'
        sufijo_h = f"_{horizon}" if (self._page_map and horizon != 'General') else ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar PDF",
            f"informe_{self._ticker}_{self._tf}{rango}{sufijo_h}.pdf",
            "PDF (*.pdf)"
        )
        if not path:
            return
        if self._page_map:
            # Exportar solo las páginas de la ventana seleccionada (+ generales)
            try:
                from pypdf import PdfReader, PdfWriter
                visibles = [p['pagina'] for p in self._page_map
                            if p.get('horizonte') in (None, horizon)]
                reader = PdfReader(self._pdf_path)
                writer = PdfWriter()
                for i in visibles:
                    if 0 <= i < len(reader.pages):
                        writer.add_page(reader.pages[i])
                with open(path, 'wb') as f:
                    writer.write(f)
                return
            except Exception:
                import traceback
                traceback.print_exc()
        import shutil
        shutil.copy2(self._pdf_path, path)

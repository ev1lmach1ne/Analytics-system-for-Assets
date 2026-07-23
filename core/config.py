import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config.json')

# === QuestDB Config ===
QUESTDB_HOST     = os.getenv('QUESTDB_HOST', 'localhost')
QUESTDB_PG_PORT  = int(os.getenv('QUESTDB_PG_PORT', '18812'))
QUESTDB_HTTP_PORT = int(os.getenv('QUESTDB_HTTP_PORT', '19000'))
QUESTDB_DATABASE = os.getenv('QUESTDB_DATABASE', 'qdb')
QUESTDB_USER     = os.getenv('QUESTDB_USER', 'admin')
QUESTDB_PASSWORD = os.getenv('QUESTDB_PASSWORD', 'quest')

DB_CONFIG = {
    'host':     QUESTDB_HOST,
    'port':     QUESTDB_PG_PORT,
    'database': QUESTDB_DATABASE,
    'user':     QUESTDB_USER,
    'password': QUESTDB_PASSWORD,
}

# === Paths (mutable) ===
BASE_DATA     = os.getenv('BASE_DATA', os.path.join(PROJECT_ROOT, "data"))
LIMPIADOS_DIR = os.path.join(BASE_DATA, "Limpiados")
INFORMES_DIR  = os.path.join(LIMPIADOS_DIR, "Informes")
CONFIG_PATH   = os.path.join(BASE_DATA, "sesion_config.json")
SCRIPTS_DIR   = os.path.join(PROJECT_ROOT, "library", "scripts_utiles")
SISTEMAS_DIR  = os.path.join(PROJECT_ROOT, "Sistemas")


_APP_CONFIG_CACHE = {}


def get_base_data():
    return BASE_DATA


def set_base_data(path):
    """Actualiza BASE_DATA y todas las rutas derivadas globalmente."""
    global BASE_DATA, LIMPIADOS_DIR, INFORMES_DIR, CONFIG_PATH
    BASE_DATA = path
    LIMPIADOS_DIR = os.path.join(BASE_DATA, "Limpiados")
    INFORMES_DIR = os.path.join(LIMPIADOS_DIR, "Informes")
    CONFIG_PATH = os.path.join(BASE_DATA, "sesion_config.json")
    _save_app_config()


def get_tutorial_visto():
    return _APP_CONFIG_CACHE.get('tutorial_visto', False)


def set_tutorial_visto(valor=True):
    _APP_CONFIG_CACHE['tutorial_visto'] = valor
    _save_app_config()


def _save_app_config():
    try:
        _APP_CONFIG_CACHE['base_data'] = BASE_DATA
        with open(APP_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(_APP_CONFIG_CACHE, f, indent=2)
    except Exception:
        pass


def load_app_config():
    """Lee config.json del PROJECT_ROOT y aplica BASE_DATA si existe."""
    global BASE_DATA, LIMPIADOS_DIR, INFORMES_DIR, CONFIG_PATH
    if not os.path.exists(APP_CONFIG_PATH):
        return False
    try:
        with open(APP_CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        _APP_CONFIG_CACHE.update(cfg)
        path = cfg.get('base_data')
        if path and os.path.isdir(path):
            BASE_DATA = path
            LIMPIADOS_DIR = os.path.join(BASE_DATA, "Limpiados")
            INFORMES_DIR = os.path.join(LIMPIADOS_DIR, "Informes")
            CONFIG_PATH = os.path.join(BASE_DATA, "sesion_config.json")
            return True
    except Exception:
        pass
    return False


load_app_config()

TF_LABELS  = ['30s', '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '1d']
TIPO_LABELS = ['Futuro/Cfd', 'Forex', 'Stock', 'Crypto']
TIPO_MAP    = {'Futuro/Cfd': 'FUTURO', 'Forex': 'FOREX', 'Stock': 'STOCK', 'Crypto': 'CRYPTO'}

TF_PATTERN = re.compile(r'_(\d+[a-z]+)_limpiado|_(\d+[a-z]+)_limpio')

# Fricción aproximada POR LADO (en %, como los muestran los spinboxes del
# backtest) según la clase de activo — dentro de cada clase hay más y menos
# líquidos, esto solo prellena un punto de partida razonable que el usuario
# puede ajustar a su broker.
PRESETS_FRICCION = {
    'CRYPTO': {'slippage_pct': 0.10, 'comision_pct': 0.03},
    'STOCK':  {'slippage_pct': 0.07, 'comision_pct': 0.07},
    'FUTURO': {'slippage_pct': 0.02, 'comision_pct': 0.03},
    'FOREX':  {'slippage_pct': 0.02, 'comision_pct': 0.03},
}

# tokens (minúsculas) para clasificar archivos legacy sin meta.json
_TOKENS_CRYPTO = ('btc', 'eth', 'sol', 'xrp', 'doge', 'ada', 'bnb', 'usdt',
                  'crypto', 'cripto')
_TOKENS_FOREX = ('eurusd', 'gbpusd', 'usdjpy', 'audusd', 'usdcad', 'usdchf',
                 'nzdusd', 'xauusd', 'xagusd')
_TOKENS_FUTURO = ('us500', 'spx', 'nas100', 'us30', 'ger40', 'dax', 'oil',
                  'wti', 'brent', 'ng')


def tipo_activo_de_csv(csv_path):
    """Clase de activo (FUTURO/FOREX/STOCK/CRYPTO) de un CSV limpiado, o
    None si no se puede determinar. Primero el sidecar <csv>.meta.json
    (escrito por la pestaña Importar); si no existe o no trae 'activo',
    heurística por tokens del nombre del archivo y su carpeta."""
    try:
        with open(str(csv_path) + '.meta.json', encoding='utf-8') as f:
            activo = (json.load(f) or {}).get('activo')
        if activo in PRESETS_FRICCION:
            return activo
    except (OSError, json.JSONDecodeError, AttributeError):
        pass

    nombre = os.path.basename(str(csv_path)).lower()
    carpeta = os.path.basename(os.path.dirname(str(csv_path))).lower()
    texto = f"{carpeta}_{nombre}"
    for tokens, tipo in ((_TOKENS_CRYPTO, 'CRYPTO'),
                         (_TOKENS_FOREX, 'FOREX'),
                         (_TOKENS_FUTURO, 'FUTURO')):
        if any(t in texto for t in tokens):
            return tipo
    return None

FACTORES = {
    'CRYPTO': {
        '1min':  {'anual': 525600, 'trimestral': 129600, 'mensual': 43200, 'semanal': 10080, 'dia': 1440},
        '5min':  {'anual': 105120, 'trimestral': 25920,  'mensual': 8640,  'semanal': 2016,  'dia': 288},
        '15min': {'anual': 35040,  'trimestral': 8640,   'mensual': 2880,  'semanal': 672,   'dia': 96},
        '30min': {'anual': 17520,  'trimestral': 4320,   'mensual': 1440,  'semanal': 336,   'dia': 48},
        '1h':    {'anual': 8760,   'trimestral': 2160,   'mensual': 720,   'semanal': 168,   'dia': 24},
        '4h':    {'anual': 2190,   'trimestral': 540,    'mensual': 180,   'semanal': 42,    'dia': 6},
        '1d':    {'anual': 365,    'trimestral': 90,     'mensual': 30,    'semanal': 7,     'dia': 1},
    },
    'FUTURO': {
        '1min':  {'anual': 362880, 'trimestral': 90720, 'mensual': 30240, 'semanal': 7056, 'dia': 1440, 'horaria': 60},
        '5min':  {'anual': 72576,  'trimestral': 18144, 'mensual': 6048,  'semanal': 1411, 'dia': 288,  'horaria': 12},
        '15min': {'anual': 24192,  'trimestral': 6048,  'mensual': 2016,  'semanal': 470,  'dia': 96,   'horaria': 4},
        '30min': {'anual': 12096,  'trimestral': 3024,  'mensual': 1008,  'semanal': 235,  'dia': 48,   'horaria': 2},
        '1h':    {'anual': 6048,   'trimestral': 1512,  'mensual': 504,   'semanal': 118,  'dia': 24,   'horaria': 1},
        '4h':    {'anual': 1512,   'trimestral': 378,   'mensual': 126,   'semanal': 29,   'dia': 6,    'horaria': 0.25},
    },
    'STOCK': {
        '1min':  {'anual': 98280,  'trimestral': 24570, 'mensual': 8190,  'semanal': 1950, 'dia': 390,  'horaria': 60},
        '5min':  {'anual': 19656,  'trimestral': 4914,  'mensual': 1638,  'semanal': 390,  'dia': 78,   'horaria': 12},
        '15min': {'anual': 6552,   'trimestral': 1638,  'mensual': 546,   'semanal': 130,  'dia': 26,   'horaria': 4},
        '30min': {'anual': 3276,   'trimestral': 819,   'mensual': 273,   'semanal': 65,   'dia': 13,   'horaria': 2},
        '1h':    {'anual': 1638,   'trimestral': 409,   'mensual': 136,   'semanal': 32.5, 'dia': 6.5,  'horaria': 1},
        '4h':    {'anual': 409,    'trimestral': 102,   'mensual': 34,    'semanal': 8,    'dia': 1.6,  'horaria': 0.25},
    },
}

def tf_to_minutes(tf, activo=None):
    m = re.match(r'(\d+)(min|mo|m|s|h|d|w)', str(tf), re.IGNORECASE)
    if not m:
        return None
    num = int(m.group(1))
    unit = m.group(2).lower()
    if unit in ('min', 'm', 's'):
        return num
    if unit == 'h':
        return num * 60
    if unit == 'd':
        return num * 1440
    if unit == 'w':
        return num * 10080
    if unit == 'mo':
        return num * 43200
    return num


# minutos de sesión/día de trading y días de trading al año por clase de
# activo, usados para anualizar métricas de backtest (Sharpe/CAGR) según la
# clase real del activo en vez de asumir siempre 24/7/365. Misma convención
# que get_factores() en library/scripts_utiles/analisis_descriptivo.py.
_MINUTOS_DIA_ANUALIZACION = {'CRYPTO': 1440, 'FUTURO': 1440, 'STOCK': 390, 'FOREX': 1440}
_DIAS_ANIO_ANUALIZACION = {'CRYPTO': 365, 'FUTURO': 252, 'STOCK': 252, 'FOREX': 252}


def velas_por_anio(tipo_activo, minutos_vela):
    """Nº de velas/año para anualizar Sharpe/CAGR, según la sesión real de
    `tipo_activo` ('CRYPTO'/'FUTURO'/'STOCK'/'FOREX') y el tamaño de vela en
    minutos. Si `tipo_activo` es None o desconocido, usa el supuesto 24/7/365
    (comportamiento previo) como fallback seguro.

    Dos regímenes (misma idea que get_factores() en
    library/scripts_utiles/analisis_descriptivo.py, pero a partir de minutos
    de vela en vez de la etiqueta de TF):
    - vela >= 1 día de calendario (diaria o más lenta): 1 vela por día de
      trading, escalado por cuántos días de calendario abarca la vela.
    - vela intradía: cuántas velas caben en la sesión de trading real
      (`minutos_dia`), multiplicado por los días de trading al año.
    Mezclar la fórmula intradía con velas diarias subestimaría brutalmente
    las velas/año en activos con sesión corta (ej. STOCK 1d daría ~68/año
    en vez de 252) porque `minutos_dia` ya no representa la sesión de esa
    vela sino un día de calendario completo.
    """
    minutos_dia = _MINUTOS_DIA_ANUALIZACION.get(tipo_activo, 1440)
    dias_anio = _DIAS_ANIO_ANUALIZACION.get(tipo_activo, 365)
    minutos_vela = max(float(minutos_vela), 1e-9)
    if minutos_vela >= 1440.0:
        dias_por_vela = minutos_vela / 1440.0
        return dias_anio / dias_por_vela
    return (minutos_dia / minutos_vela) * dias_anio

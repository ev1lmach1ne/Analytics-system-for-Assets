import os
import re
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

# === Paths ===
BASE_DATA     = os.getenv('BASE_DATA', r"D:\DATOS\Activos")
LIMPIADOS_DIR = os.path.join(BASE_DATA, "Limpiados")
INFORMES_DIR  = os.path.join(LIMPIADOS_DIR, "Informes")
CONFIG_PATH   = os.path.join(BASE_DATA, "sesion_config.json")
SCRIPTS_DIR   = os.path.join(PROJECT_ROOT, "library", "scripts_utiles")

TF_LABELS  = ['30s', '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '1d']
TIPO_LABELS = ['Futuro/Cfd', 'Forex', 'Stock', 'Crypto']
TIPO_MAP    = {'Futuro/Cfd': 'FUTURO', 'Forex': 'FOREX', 'Stock': 'STOCK', 'Crypto': 'CRYPTO'}

TF_PATTERN = re.compile(r'_(\d+[a-z]+)_limpiado|_(\d+[a-z]+)_limpio')

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

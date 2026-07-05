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

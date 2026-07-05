import psycopg2
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.config import DB_CONFIG

try:
    conn = psycopg2.connect(**DB_CONFIG)
    print("Conexion exitosa a QuestDB")
    conn.close()
except Exception as e:
    print(f"Error: {e}")
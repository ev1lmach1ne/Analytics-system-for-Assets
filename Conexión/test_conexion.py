import psycopg2

try:
    conn = psycopg2.connect(
        host='localhost',
        port=18812,
        database='qdb',
        user='admin',
        password='quest'
    )
    print("✅ Conexión exitosa a QuestDB")
    conn.close()
except Exception as e:
    print(f"❌ Error: {e}")
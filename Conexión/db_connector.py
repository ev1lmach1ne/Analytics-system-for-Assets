import psycopg2
import pandas as pd

def get_data(table_name):
    try:
        conn = psycopg2.connect(
            dbname="crypto_trading",
            user="postgres",
            password="1368",
            host="localhost"
        )
        # Forzamos la comunicacion a UTF8
        conn.set_client_encoding('UTF8')
        
        query = "SELECT * FROM " + table_name + " ORDER BY timestamp ASC"
        
        # Leemos los datos
        df = pd.read_sql(query, conn)
        
        conn.close()
        return df
    except Exception as e:
        return "Error en conexion: " + str(e)
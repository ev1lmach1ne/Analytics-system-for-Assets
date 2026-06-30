import pandas as pd

# Asegúrate de poner la ruta exacta donde se guarda tu archivo limpio
ruta = r'D:\DATOS\Activos\Crypto\Limpiados\btc_1h_limpio.csv'
df = pd.read_csv(ruta)

# Filtramos y mostramos solo las anomalías
anomalias = df[df['anomalia'] == 1]

if not anomalias.empty:
    print(f"Se han encontrado {len(anomalias)} anomalías:")
    print(anomalias[['close', 'volume']])
else:
    print("No se encontraron anomalías.")
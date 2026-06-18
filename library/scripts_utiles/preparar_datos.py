import pandas as pd
from pathlib import Path

# ── CONFIGURACIÓN — solo cambia esto ─────────────────────
CSV_INPUT  = r'D:\DATOS\raw\btc_raw.csv'
SEPARADOR  = ','     # cámbialo a ';' si tu CSV usa punto y coma
# ─────────────────────────────────────────────────────────

# Output automático: mismo nombre + '_preparado', misma carpeta
_p = Path(CSV_INPUT)
CSV_OUTPUT = str(_p.parent / f"{_p.stem}_preparado{_p.suffix}")

# ── MAPA DE ALIAS — añade aquí cualquier nombre alternativo ──
ALIAS_COLUMNAS = {
    'timestamp': ['time', 'date', 'datetime', 'ts', 'fecha',
                  'Date', 'Time', 'Datetime', 'Timestamp', 'index'],
    'open':      ['Open', 'OPEN', 'o', 'precio_apertura', 'open_price'],
    'high':      ['High', 'HIGH', 'h', 'max', 'maximo', 'precio_maximo'],
    'low':       ['Low',  'LOW',  'l', 'min', 'minimo', 'precio_minimo'],
    'close':     ['Close','CLOSE','c', 'price', 'precio', 'last', 'ultimo',
                  'settle', 'settlement'],
    'volume':    ['Volume','VOLUME','vol','Vol','VOL','qty','quantity',
                  'amount', 'volumen'],
}

print("="*50)
print("PREPARACIÓN DE CSV PARA QUESTDB")
print("="*50)

# [1/4] Lectura
print(f"\n[1/4] Leyendo CSV...")
df = pd.read_csv(CSV_INPUT, sep=SEPARADOR)
print(f"      Filas:    {len(df)}")
print(f"      Columnas: {list(df.columns)}")

# [2/4] Normalización de columnas
print(f"\n[2/4] Normalizando columnas...")

# Mapa inverso: alias_lower -> nombre canónico
mapa_inverso = {}
for canonico, aliases in ALIAS_COLUMNAS.items():
    mapa_inverso[canonico.lower()] = canonico
    for alias in aliases:
        mapa_inverso[alias.lower()] = canonico

renombrado = {}
descartadas = []
for col in df.columns:
    clave = col.strip().lower()
    if clave in mapa_inverso:
        renombrado[col] = mapa_inverso[clave]
    else:
        descartadas.append(col)

if descartadas:
    print(f"      ⚠️  Columnas no reconocidas y descartadas: {descartadas}")

df = df.rename(columns=renombrado)

# [3/4] Validación de obligatorias
print(f"\n[3/4] Validando columnas obligatorias...")
obligatorias = ['timestamp', 'open', 'high', 'low', 'close']
faltantes = [c for c in obligatorias if c not in df.columns]

if faltantes:
    print(f"\n❌ ERROR: Faltan columnas obligatorias: {faltantes}")
    print(f"   Columnas disponibles: {list(df.columns)}")
    print(f"   Añade el alias correspondiente en ALIAS_COLUMNAS y vuelve a ejecutar.")
    exit(1)

if 'volume' not in df.columns:
    df['volume'] = 0
    print("      ℹ️  'volume' no encontrado — rellenado con 0.")

print("      ✅ Todas las columnas obligatorias presentes.")

# Reordenar al esquema canónico
df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
print(f"      Esquema final: {list(df.columns)}")

# [4/4] Guardar
print(f"\n[4/4] Guardando CSV preparado...")
df.to_csv(CSV_OUTPUT, index=False)
print(f"\n✅ Listo. Sube este archivo a QuestDB:")
print(f"   {CSV_OUTPUT}")
print("="*50)
"""
Calendario económico histórico (TradingEconomics) para backtesting
event-driven.

Proveedor gratuito: https://tradingeconomics.com/api — plan free con ~100
peticiones/mes (cada mes descargado = 1 petición; la caché mensual en disco
lo hace viable). Devuelve eventos históricos y futuros con impacto
High/Medium/Low. No hay filtro por país en el endpoint, así que se descarga
todo y el filtrado por moneda/impacto ocurre más arriba
(core/strategies.py), no aquí: este módulo solo sabe "traer eventos crudos
de un rango de fechas", nada de lógica de setups.

Caché en disco: un fichero JSON por mes calendario bajo BASE_DATA/Noticias/
(economico_YYYY-MM.json). Un mes ya cerrado es inmutable y nunca se vuelve a
pedir; el mes en curso siempre se re-descarga porque puede tener eventos
futuros/no confirmados.
"""

import json
import os
import time
from datetime import datetime, timezone, date
from calendar import monthrange

import pandas as pd
import requests

from core.config import get_base_data

TE_URL = "https://api.tradingeconomics.com/calendar"
REQUEST_TIMEOUT = 30
_SLEEP_ENTRE_LLAMADAS = 1.1  # margen frente a límites de la API

IMPACTO_RANK = {'bajo': 0, 'medio': 1, 'alto': 2}
_TE_IMPACTO_MAP = {'High': 'alto', 'Medium': 'medio', 'Low': 'bajo'}

DESCRIPCION_IMPACTO = {
    'alto': (
        "Alto impacto: los eventos que más suelen mover el mercado — "
        "Nóminas no agrícolas (NFP), IPC/inflación, decisiones de tipos de "
        "interés de bancos centrales (Fed, BCE, BoE...), PIB trimestral, "
        "discursos relevantes de banqueros centrales."
    ),
    'medio': (
        "Impacto medio: indicadores que mueven el mercado de forma más "
        "moderada — PMI manufacturero/servicios, ventas minoristas, "
        "solicitudes semanales de desempleo, balanza comercial, confianza "
        "del consumidor."
    ),
    'bajo': (
        "Bajo impacto: indicadores secundarios con efecto habitualmente "
        "leve — inventarios, encuestas regionales, subastas de bonos "
        "rutinarias y similares."
    ),
}

# TradingEconomics devuelve el país por NOMBRE ("United States"); el filtro
# por moneda (core/strategies.py) compara códigos ISO, así que se mapean los
# nombres de los países con divisa relevante. Si no se reconoce el nombre se
# devuelve tal cual (el filtro por moneda simplemente no restringe).
_PAISES_TE_A_ISO = {
    'United States': 'US', 'United Kingdom': 'GB', 'Germany': 'DE',
    'France': 'FR', 'Italy': 'IT', 'Spain': 'ES', 'Japan': 'JP',
    'China': 'CN', 'Australia': 'AU', 'Canada': 'CA', 'Switzerland': 'CH',
    'New Zealand': 'NZ', 'Euro Area': 'EU', 'Netherlands': 'NL',
    'Belgium': 'BE', 'Sweden': 'SE', 'Norway': 'NO', 'Denmark': 'DK',
    'Austria': 'AT', 'Portugal': 'PT', 'Ireland': 'IE', 'Poland': 'PL',
    'Russia': 'RU', 'India': 'IN', 'Brazil': 'BR', 'Mexico': 'MX',
    'South Korea': 'KR', 'Singapore': 'SG', 'Hong Kong': 'HK',
    'South Africa': 'ZA', 'Turkey': 'TR', 'Indonesia': 'ID',
    'Thailand': 'TH', 'Malaysia': 'MY', 'Israel': 'IL', 'Argentina': 'AR',
    'Chile': 'CL', 'Colombia': 'CO', 'Egypt': 'EG', 'Saudi Arabia': 'SA',
    'Nigeria': 'NG', 'Greece': 'GR', 'Czech Republic': 'CZ',
    'Hungary': 'HU', 'Romania': 'RO', 'Finland': 'FI', 'Luxembourg': 'LU',
    'Ukraine': 'UA', 'Taiwan': 'TW', 'Philippines': 'PH',
}


def _pais_a_iso(nombre):
    if not nombre:
        return ''
    return _PAISES_TE_A_ISO.get(str(nombre).strip(), str(nombre).strip())


# pares/instrumentos -> monedas relevantes (heurística por nombre; None si no
# se reconoce nada, en cuyo caso el filtro de noticias solo actúa por
# impacto, sin restringir moneda)
_MONEDAS_ISO = {'EUR', 'USD', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD'}
_METALES = {'XAU': ['USD'], 'XAG': ['USD']}
_INDICES = {
    'US30': ['USD'], 'US500': ['USD'], 'SPX': ['USD'], 'NAS100': ['USD'],
    'GER40': ['EUR'], 'DAX': ['EUR'], 'UK100': ['GBP'], 'FTSE': ['GBP'],
    'JP225': ['JPY'], 'NIKKEI': ['JPY'],
}


def _mes_cache_path(anio, mes):
    carpeta = os.path.join(get_base_data(), "Noticias")
    return os.path.join(carpeta, f"economico_{anio:04d}-{mes:02d}.json")


def _mes_es_pasado(anio, mes):
    _, ultimo_dia = monthrange(anio, mes)
    fin_mes = date(anio, mes, ultimo_dia)
    return fin_mes < datetime.now(timezone.utc).date()


def _cargar_mes_cache(anio, mes):
    """DataFrame cacheado del mes, o None si no hay caché válida (mes en
    curso, o fichero inexistente/corrupto)."""
    if not _mes_es_pasado(anio, mes):
        return None
    path = _mes_cache_path(anio, mes)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            filas = json.load(f)
        return _filas_a_df(filas)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _guardar_mes_cache(anio, mes, df):
    path = _mes_cache_path(anio, mes)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    filas = [
        {'timestamp': ts.isoformat(), 'pais': pais, 'evento': evento, 'impacto': impacto}
        for ts, pais, evento, impacto in zip(
            df['timestamp'], df['pais'], df['evento'], df['impacto'])
    ]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(filas, f, ensure_ascii=False, indent=0)


def _filas_a_df(filas):
    if not filas:
        return _df_vacio()
    df = pd.DataFrame(filas)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df[['timestamp', 'pais', 'evento', 'impacto']]


def _df_vacio():
    return pd.DataFrame({
        'timestamp': pd.Series(dtype='datetime64[ns, UTC]'),
        'pais': pd.Series(dtype=str),
        'evento': pd.Series(dtype=str),
        'impacto': pd.Series(dtype=str),
    })


def _fetch_te_mes(anio, mes, api_key):
    """Una petición al endpoint de TradingEconomics cubriendo el mes completo.

    TradingEconomics devuelve JSON con cada evento: Country (nombre), Event
    (o Category), Date (ISO) e Importance ('High'/'Medium'/'Low'). Las
    fechas llegan sin zona horaria explícita: se asumen UTC (aproximado,
    coherente con el nivel del filtro de noticias).
    """
    _, ultimo_dia = monthrange(anio, mes)
    desde = date(anio, mes, 1).isoformat()
    hasta = date(anio, mes, ultimo_dia).isoformat()
    try:
        resp = requests.get(
            TE_URL,
            params={'c': api_key, 'd1': desde, 'd2': hasta},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"No se pudo conectar con TradingEconomics: {e.__class__.__name__}") from e
    if resp.status_code in (401, 403):
        raise RuntimeError(
            "La API key de TradingEconomics no es válida (HTTP "
            f"{resp.status_code}): revísala en Ajustes → Calendario "
            "económico y copia la key de tu panel de tradingeconomics.com.")
    if resp.status_code == 429:
        raise RuntimeError(
            "Límite de peticiones de TradingEconomics alcanzado (HTTP 429): "
            "el plan gratuito permite ~100 al mes — espera al mes siguiente "
            "o usa los meses ya cacheados.")
    resp.raise_for_status()
    try:
        datos = resp.json()
    except ValueError as e:
        raise RuntimeError(
            f"TradingEconomics devolvió una respuesta no válida: {e}") from e
    if not isinstance(datos, list) or not datos:
        return _df_vacio()
    filas = []
    for ev in datos:
        if not isinstance(ev, dict):
            continue
        ts_raw = ev.get('Date') or ev.get('date')
        if not ts_raw:
            continue
        try:
            ts = pd.to_datetime(ts_raw, utc=True)
        except (ValueError, TypeError):
            continue
        impacto = _TE_IMPACTO_MAP.get(
            str(ev.get('Importance') or '').strip().capitalize(), 'bajo')
        filas.append({
            'timestamp': ts,
            'pais': _pais_a_iso(ev.get('Country')),
            'evento': ev.get('Event') or ev.get('Category') or '',
            'impacto': impacto,
        })
    return _filas_a_df(filas) if filas else _df_vacio()


def _meses_en_rango(fecha_inicio, fecha_fin):
    ts_ini = pd.Timestamp(fecha_inicio)
    ts_fin = pd.Timestamp(fecha_fin)
    cursor = date(ts_ini.year, ts_ini.month, 1)
    fin = date(ts_fin.year, ts_fin.month, 1)
    meses = []
    while cursor <= fin:
        meses.append((cursor.year, cursor.month))
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return meses


def obtener_eventos(fecha_inicio, fecha_fin, api_key=None, impacto_minimo=None,
                    progress_callback=None):
    """Eventos económicos históricos en [fecha_inicio, fecha_fin] (UTC).

    Devuelve un DataFrame con columnas timestamp/pais/evento/impacto. Usa
    caché mensual en disco y solo llama a TradingEconomics para meses sin
    caché válida (el mes en curso siempre se refresca).

    `progress_callback(i, total)` se invoca por cada mes procesado (i
    empieza en 1), para que el llamador pueda dar feedback de progreso —
    con rangos largos la descarga de años de eventos puede tardar
    minutos y sin esto parece que el proceso está congelado.
    """
    if not api_key:
        raise RuntimeError(
            "API key de TradingEconomics no configurada. Añádela en "
            "Ajustes → Calendario económico.")

    meses = _meses_en_rango(fecha_inicio, fecha_fin)
    total_meses = len(meses)
    partes = []
    primero = True
    for i, (anio, mes) in enumerate(meses):
        if progress_callback:
            progress_callback(i + 1, total_meses)
        df_mes = _cargar_mes_cache(anio, mes)
        if df_mes is None:
            if not primero:
                time.sleep(_SLEEP_ENTRE_LLAMADAS)
            df_mes = _fetch_te_mes(anio, mes, api_key)
            primero = False
            if _mes_es_pasado(anio, mes):
                _guardar_mes_cache(anio, mes, df_mes)
        partes.append(df_mes)

    df = pd.concat(partes, ignore_index=True) if partes else _df_vacio()
    ts_ini = pd.Timestamp(fecha_inicio)
    if ts_ini.tzinfo is None:
        ts_ini = ts_ini.tz_localize('UTC')
    ts_fin = pd.Timestamp(fecha_fin)
    if ts_fin.tzinfo is None:
        ts_fin = ts_fin.tz_localize('UTC')
    df = df[(df['timestamp'] >= ts_ini) & (df['timestamp'] <= ts_fin)]
    if impacto_minimo:
        umbral = IMPACTO_RANK.get(impacto_minimo, 0)
        df = df[df['impacto'].map(IMPACTO_RANK).fillna(0) >= umbral]
    return df.sort_values('timestamp').reset_index(drop=True)


def monedas_de_instrumento(nombre_activo):
    """Monedas/países relevantes para un instrumento, o None si no se
    reconoce nada (el filtro de noticias entonces no restringe por moneda)."""
    if not nombre_activo:
        return None
    limpio = ''.join(c for c in str(nombre_activo).upper() if c.isalnum())

    for prefijo, monedas in _INDICES.items():
        if prefijo in limpio:
            return monedas
    for prefijo, monedas in _METALES.items():
        if prefijo in limpio:
            return monedas

    encontradas = [m for m in _MONEDAS_ISO if m in limpio]
    if len(encontradas) >= 2 and len(limpio) <= 10:
        # par FX de 6 letras tipo EURUSD: solo las dos que forman el par
        for base in _MONEDAS_ISO:
            for cotiz in _MONEDAS_ISO:
                if base != cotiz and f"{base}{cotiz}" in limpio:
                    return [base, cotiz]
        return encontradas
    if len(encontradas) == 1:
        return encontradas
    return None

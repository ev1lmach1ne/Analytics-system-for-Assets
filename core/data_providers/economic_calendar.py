"""
Calendario económico histórico (Finnhub) para backtesting event-driven.

Proveedor gratuito: https://finnhub.io/docs/api/economic-calendar — 60
peticiones/minuto sin tarjeta de crédito, con eventos históricos y futuros
(actual/estimado/previo, impacto bajo/medio/alto). No hay filtro por país en
el endpoint, así que se descarga todo y el filtrado por moneda/impacto ocurre
más arriba (core/strategies.py), no aquí: este módulo solo sabe "traer
eventos crudos de un rango de fechas", nada de lógica de setups.

Caché en disco: un fichero JSON por mes calendario bajo BASE_DATA/Noticias/.
Un mes ya cerrado (su último día es anterior a hoy) es inmutable y nunca se
vuelve a pedir; el mes en curso siempre se re-descarga porque puede tener
eventos futuros/no confirmados.
"""

import json
import os
import time
from datetime import datetime, timezone, date
from calendar import monthrange

import pandas as pd
import requests

from core.config import get_base_data

FINNHUB_URL = "https://finnhub.io/api/v1/calendar/economic"
REQUEST_TIMEOUT = 20
_SLEEP_ENTRE_LLAMADAS = 1.1  # margen holgado frente al límite de 60/min

IMPACTO_RANK = {'bajo': 0, 'medio': 1, 'alto': 2}
_FINNHUB_IMPACTO_MAP = {'': 'bajo', 'low': 'bajo', 'medium': 'medio', 'high': 'alto'}

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


def _fetch_finnhub_mes(anio, mes, api_key):
    """Una petición al endpoint de Finnhub cubriendo el mes completo."""
    _, ultimo_dia = monthrange(anio, mes)
    desde = date(anio, mes, 1).isoformat()
    hasta = date(anio, mes, ultimo_dia).isoformat()
    resp = requests.get(
        FINNHUB_URL,
        params={'from': desde, 'to': hasta, 'token': api_key},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    datos = (resp.json() or {}).get('economicCalendar') or []
    if not datos:
        return _df_vacio()
    filas = []
    for ev in datos:
        ts_raw = ev.get('time') or ev.get('date')
        if not ts_raw:
            continue
        filas.append({
            'timestamp': pd.to_datetime(ts_raw, utc=True),
            'pais': (ev.get('country') or '').upper(),
            'evento': ev.get('event') or '',
            'impacto': _FINNHUB_IMPACTO_MAP.get((ev.get('impact') or '').lower(), 'bajo'),
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


def obtener_eventos(fecha_inicio, fecha_fin, api_key=None, impacto_minimo=None):
    """Eventos económicos históricos en [fecha_inicio, fecha_fin] (UTC).

    Devuelve un DataFrame con columnas timestamp/pais/evento/impacto. Usa
    caché mensual en disco y solo llama a Finnhub para meses sin caché
    válida (el mes en curso siempre se refresca).
    """
    if not api_key:
        raise RuntimeError(
            "Finnhub API key no configurada. Añádela en Ajustes → "
            "Proveedor de calendario económico.")

    partes = []
    primero = True
    for anio, mes in _meses_en_rango(fecha_inicio, fecha_fin):
        df_mes = _cargar_mes_cache(anio, mes)
        if df_mes is None:
            if not primero:
                time.sleep(_SLEEP_ENTRE_LLAMADAS)
            df_mes = _fetch_finnhub_mes(anio, mes, api_key)
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

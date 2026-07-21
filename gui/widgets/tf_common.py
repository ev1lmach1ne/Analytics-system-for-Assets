"""Utilidades de temporalidad compartidas entre pestañas que permiten
recalcular sobre una temporalidad más gruesa que la nativa del CSV
importado (Patrones, Optimizador): lista de TFs ofrecidas, regla de
resample de pandas para cada una y parseo de TF personalizada. Solo se
puede subir de granularidad respecto a la nativa del archivo, nunca bajar
(cada pestaña valida ese mínimo con core.config.tf_to_minutes)."""
import re

TF_LABELS = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
REGLAS_RESAMPLE = {'1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min',
                   '30m': '30min', '1h': '1h', '4h': '4h', '1d': '1D', '1w': '1W'}


def parsear_tf_custom(texto):
    """'20m'/'2 h'/'3d'/'2w' → etiqueta normalizada ('20m') o None si no es
    una temporalidad válida."""
    m = re.match(r'^\s*(\d+)\s*(m|h|d|w)\s*$', str(texto), re.IGNORECASE)
    if not m or int(m.group(1)) <= 0:
        return None
    return f"{int(m.group(1))}{m.group(2).lower()}"


def regla_de_tf(tf_label):
    """Regla de resample de pandas para una TF (preset o custom)."""
    if tf_label in REGLAS_RESAMPLE:
        return REGLAS_RESAMPLE[tf_label]
    m = re.match(r'^(\d+)(m|h|d|w)$', tf_label)
    if not m:
        return None
    unidad = {'m': 'min', 'h': 'h', 'd': 'D', 'w': 'W'}[m.group(2)]
    return f"{m.group(1)}{unidad}"

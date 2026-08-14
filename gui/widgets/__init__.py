"""Estilo compartido para etiquetas "sin caja": fondo y borde forzados a
transparente/none para que nunca se dibuje un recuadro alrededor del texto
(p. ej. Nombre:, TF:, Rf(%): en Importar, Rf: en Limpiador y Ventana/
Periodo en Analizador). Color azul grisáceo suave, medio grueso.
"""

STYLE_ETIQUETA_SIN_CAJA = """
QLabel {
    background: transparent;
    border: none;
    color: #8fb3d9;
    font-size: 12px;
    font-weight: bold;
    padding-right: 6px;
}
"""

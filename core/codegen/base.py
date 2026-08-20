"""
core/codegen/base.py
Esqueleto común de los emisores.

Un emisor traduce el IR (core/codegen/ir.py) al lenguaje de una plataforma.
Todo lo que no depende del lenguaje vive aquí: recorrer el árbol de nodos,
poner nombre a cada serie, formatear números y montar la cabecera de
fidelidad. Cada subclase solo rellena cómo se escribe cada cosa en su idioma.

LA REGLA DURA: NADA SE PIERDE EN SILENCIO
─────────────────────────────────────────
`expr()` lanza ValueError ante cualquier nodo que no sepa traducir, y nunca
devuelve cadena vacía. Un nodo que se pierde por el camino produce un robot
que opera de menos —o que no opera— sin que nadie se entere leyendo el
archivo, que es exactamente el fallo que este paquete entero existe para
evitar.

Lo que NO se soporta a propósito se declara en fidelidad.CAPACIDADES y sale
por el canal de avisos (diálogo, cabecera, notas, aviso en ejecución). Lo que
no se soporta por descuido tiene que romper un test.

CADA SERIE SE DECLARA UNA VEZ
─────────────────────────────
Los emisores no incrustan la fórmula del indicador allí donde se usa: primero
declaran una variable por cada serie de ir_setup['series'] y luego las
condiciones se limitan a nombrarlas. Así el indicador se calcula una sola vez
por vela (lo que ambas plataformas esperan) y el código generado se puede
leer. El nombre lo pone nombre_serie(), que es determinista: la misma serie da
siempre el mismo identificador, en Pine y en MQL.
"""
from core.codegen import fidelidad
from core.codegen.ir import LADOS, OPS_COMPARACION

# Series que no se declaran como variable: o son el propio precio (cada
# lenguaje ya tiene su palabra para ellas) o son una constante.
SERIES_INLINE = ('valor', 'close', 'open', 'high', 'low')


class Emisor:
    """Base de todos los emisores. Las subclases definen los atributos de
    identidad y los ganchos de lenguaje marcados más abajo."""

    clave = ''            # 'mt5', 'tradingview', ...
    nombre = ''           # 'MetaTrader 5'
    lenguaje = ''         # 'MQL5'
    extension = ''        # '.mq5'
    comentario = '//'     # prefijo de comentario de línea

    # ══════════════ API que usa exportar_sistema() ══════════════

    def archivos_setup(self, ir_setup, ir_sistema, avisos):
        """{ruta relativa: contenido} de los archivos propios de un setup."""
        raise NotImplementedError

    def archivos_comunes(self, ir_sistema, avisos):
        """{ruta relativa: contenido} de lo que se comparte entre los setups
        del sistema (librería de runtime, instrucciones de instalación). Se
        escribe una sola vez aunque haya varios setups."""
        return {}

    # ══════════════ despacho de nodos ══════════════

    def expr(self, nodo):
        """Traduce un nodo booleano o de serie a una expresión del lenguaje.

        Lanza si no lo reconoce: ver la regla dura del docstring del módulo."""
        if nodo is None:
            raise ValueError(
                "expr() ha recibido None. Un lado sin señal se comprueba con "
                "`if senales[lado] is not None` ANTES de traducirlo; llegar "
                "hasta aquí significa que se iba a emitir una condición vacía.")
        if 'tipo' in nodo:
            return self.serie(nodo)
        op = nodo.get('op')
        if op == 'Y':
            return self.op_y([self.expr(p) for p in nodo['partes']])
        if op == 'O':
            return self.op_o([self.expr(p) for p in nodo['partes']])
        if op == 'NO':
            return self.op_no(self.expr(nodo['partes'][0]))
        if op in OPS_COMPARACION:
            return self.comparacion(nodo)
        if op == 'giro_sar':
            return self.giro_sar(nodo)
        if op == 'giro_supertrend':
            return self.giro_supertrend(nodo)
        if op == 'patron':
            return self.patron(nodo)
        raise ValueError(
            f"{self.nombre}: nodo sin traducción — op={op!r}. Si es una "
            f"característica que esta plataforma no soporta, declárala en "
            f"fidelidad.CAPACIDADES para que avise; si no, es un descuido.")

    def comparacion(self, nodo):
        izq, der = self.serie(nodo['izq']), self.serie(nodo['der'])
        op = nodo['op']
        if op == '>':
            return self.op_mayor(izq, der)
        if op == '<':
            return self.op_menor(izq, der)
        if op == '>=':
            return self.op_mayor_igual(izq, der)
        if op == '<=':
            return self.op_menor_igual(izq, der)
        if op == 'cruza arriba':
            return self.op_cruza_arriba(izq, der)
        if op == 'cruza abajo':
            return self.op_cruza_abajo(izq, der)
        raise ValueError(f"{self.nombre}: operador sin traducción: {op!r}")

    def serie(self, nodo):
        """Expresión que da el valor de una serie en la vela actual: el
        literal si es una constante, la palabra del lenguaje si es el precio,
        y si no el nombre de la variable ya declarada."""
        tipo = nodo['tipo']
        if tipo == 'valor':
            # un umbral que viene de un parámetro de la plantilla se emite
            # como el input correspondiente, para que se pueda tocar desde la
            # plataforma; los umbrales fijos (el 50 del RSI) van literales
            if nodo.get('origen'):
                return self.nombre_param(nodo['origen'])
            return self.num(nodo['valor'])
        if tipo in ('close', 'open', 'high', 'low'):
            return self.precio(tipo)
        return self.nombre_serie(nodo)

    def arg_serie(self, nodo, campo):
        """Valor de un campo de una serie al declararla: el input si ese campo
        salió de un parámetro de la plantilla, el literal si no."""
        origen = (nodo.get('origen') or {}).get(campo)
        if origen:
            return self.nombre_param(origen)
        return self.num(nodo[campo])

    def nombre_param(self, clave):
        """Identificador del input que corresponde a un parámetro de la
        plantilla. Se prefija para no chocar con nada del lenguaje."""
        return f"p_{clave}"

    def params_usados(self, ir_setup):
        """Parámetros de la plantilla que el código generado usa de verdad, en
        orden de aparición. Solo esos salen como input: emitir uno que no se
        lee en ninguna parte sería prometer un ajuste que no hace nada."""
        fuera = []
        for nodo in ir_setup['series']:
            origen = nodo.get('origen')
            if not origen:
                continue
            claves = [origen] if isinstance(origen, str) else origen.values()
            for clave in claves:
                if clave not in fuera:
                    fuera.append(clave)
        return fuera

    def giro_sar(self, nodo):
        raise ValueError(
            f"{self.nombre}: no sabe emitir el giro del Parabolic SAR.")

    def giro_supertrend(self, nodo):
        raise ValueError(
            f"{self.nombre}: no sabe emitir el giro del Supertrend.")

    def patron(self, nodo):
        raise ValueError(
            f"{self.nombre}: no sabe emitir el patrón de vela "
            f"«{nodo.get('nombre')}».")

    # ══════════════ ganchos de lenguaje (los rellena cada subclase) ══════════════

    def op_y(self, partes):
        raise NotImplementedError

    def op_o(self, partes):
        raise NotImplementedError

    def op_no(self, parte):
        raise NotImplementedError

    def op_mayor(self, izq, der):
        return f"{izq} > {der}"

    def op_menor(self, izq, der):
        return f"{izq} < {der}"

    def op_mayor_igual(self, izq, der):
        return f"{izq} >= {der}"

    def op_menor_igual(self, izq, der):
        return f"{izq} <= {der}"

    def op_cruza_arriba(self, izq, der):
        raise NotImplementedError

    def op_cruza_abajo(self, izq, der):
        raise NotImplementedError

    def precio(self, campo):
        raise NotImplementedError

    def declarar_serie(self, nombre, nodo):
        """Línea(s) que calculan la serie y la dejan en `nombre`."""
        raise NotImplementedError

    # ══════════════ nombres y números ══════════════

    def nombre_serie(self, nodo):
        """Identificador determinista de una serie, válido tanto en Pine como
        en MQL: minúsculas, sin acentos, con los parámetros pegados detrás
        para que dos configuraciones distintas del mismo indicador no colisionen
        (bb_inf_20_2p0 frente a bb_inf_20_3p0)."""
        tipo = nodo['tipo'].lower()
        partes = [tipo]
        for clave in ('periodo', 'desv', 'rapido', 'lento', 'senal',
                      'multiplicador', 'tenkan', 'kijun', 'senkou',
                      'mult_bb', 'mult_kc', 'periodo_k',
                      'suavizado_k', 'periodo_d', 'af_inicial', 'af_paso',
                      'af_max', 'ventana', 'periodo_base', 'desviacion',
                      'piernas', 'anclaje', 'k', 'modo'):
            if clave in nodo:
                partes.append(_sufijo(nodo[clave]))
        if nodo.get('fuente') == 'close':
            partes.append('c')
        return "_".join(partes)

    def series_declarables(self, ir_setup):
        """Las series del setup que hay que declarar, sin repetir y en orden
        estable. Se excluyen las inline (precio y constantes)."""
        vistas, fuera = set(), []
        for nodo in ir_setup['series']:
            if nodo['tipo'] in SERIES_INLINE:
                continue
            nombre = self.nombre_serie(nodo)
            if nombre in vistas:
                continue
            vistas.add(nombre)
            fuera.append((nombre, nodo))
        return fuera

    def num(self, valor):
        """Literal numérico. Los enteros de Python salen sin punto y los
        float siempre con él: en Pine el tipo de un literal es parte de la
        firma de las funciones, y `2` no vale donde se espera `2.0`."""
        if isinstance(valor, bool):
            return 'true' if valor else 'false'
        if isinstance(valor, int):
            return str(valor)
        texto = f"{float(valor):.10g}"
        if '.' not in texto and 'e' not in texto and 'n' not in texto:
            texto += '.0'
        return texto

    def texto(self, valor):
        """Literal de cadena, con las comillas escapadas. Igual en Pine y en
        MQL, así que vive aquí."""
        limpio = str(valor).replace('\\', '\\\\').replace('"', '\\"')
        limpio = limpio.replace('\n', ' ').replace('\r', ' ')
        return f'"{limpio}"'

    # ══════════════ cabecera ══════════════

    def cabecera(self, ir_setup, ir_sistema, avisos, meta):
        """Bloque de comentario que abre el archivo: qué sistema es, con qué
        activo y temporalidad se backtesteó, y las notas de fidelidad.

        Lo de la fidelidad no es decorativo: es la única forma de que alguien
        que abra este archivo dentro de seis meses sepa en qué se diferencia
        de lo que se probó."""
        cuenta = ir_sistema['cuenta']
        lineas = [
            f"{ir_setup['nombre']} — setup {ir_setup['indice']} "
            f"de «{meta.get('sistema', 'sistema')}»",
            "",
            f"Generado por Analytics System a partir de un backtest.",
            f"Plantilla   : {ir_setup['plantilla']}",
            f"Activo      : {meta.get('activo', '?')}",
            f"Temporalidad: {meta.get('tf', '?')}",
            f"Cuenta      : capital {cuenta['capital_inicial']:,.0f} · "
            f"comisión {cuenta['comision_pct'] * 100:g}% por lado · "
            f"slippage {cuenta['slippage_pct'] * 100:g}%",
            "",
            "Los parámetros están ajustados a ESE activo y ESA temporalidad.",
            "En otro mercado o marco temporal no significan lo mismo.",
            "",
        ]
        lineas += fidelidad.bloque_notas(avisos, self.nombre)
        return self.comentar(lineas)

    def comentar(self, lineas):
        """Convierte líneas sueltas en un bloque de comentario del lenguaje."""
        marco = self.comentario + " " + "─" * 68
        cuerpo = [f"{self.comentario} {linea}".rstrip() for linea in lineas]
        return "\n".join([marco] + cuerpo + [marco])

    # ══════════════ utilidades para las subclases ══════════════

    def lados_activos(self, ir_setup):
        """Los lados con señal, en orden fijo. Un lado apagado por el filtro
        de dirección es None en el IR y no genera código."""
        return [lado for lado in LADOS if ir_setup['senales'][lado] is not None]

    def identificador(self, texto):
        """Nombre de archivo/estrategia seguro a partir del nombre del setup."""
        limpio = "".join(c if c.isalnum() else '_' for c in _sin_acentos(texto))
        limpio = "_".join(p for p in limpio.split('_') if p)
        return limpio or 'setup'


# ══════════════ helpers de módulo ══════════════

def _sufijo(valor):
    """Trozo de identificador a partir de un número: el punto decimal pasa a
    'p' y el signo a 'm', porque ni Pine ni MQL admiten '.' ni '-' dentro de
    un nombre de variable. Las cadenas (p. ej. 'anclaje'/'modo' del VWAP) se
    normalizan en minúsculas sin acentos."""
    if isinstance(valor, str):
        return _sin_acentos(valor).lower().replace('.', 'p').replace('-', 'm')
    if isinstance(valor, float) and valor != int(valor):
        return f"{valor:g}".replace('.', 'p').replace('-', 'm')
    return str(int(valor)).replace('-', 'm')


def _sin_acentos(texto):
    import unicodedata
    descompuesto = unicodedata.normalize('NFKD', str(texto))
    return "".join(c for c in descompuesto if not unicodedata.combining(c))

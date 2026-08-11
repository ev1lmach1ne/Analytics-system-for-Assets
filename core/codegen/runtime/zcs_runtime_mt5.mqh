//+------------------------------------------------------------------+
//| zcs_runtime.mqh                                                   |
//| Runtime del Analytics System para MetaTrader 5 (MQL5).            |
//|                                                                   |
//| Contiene dos cosas:                                               |
//|   1. Los indicadores cuya fórmula NO coincide con la del          |
//|      indicador nativo de MetaTrader. Lo que sí coincide (RSI,     |
//|      CCI, %R, estocástico, medias, máximos/mínimos) se deja en    |
//|      manos de iRSI/iCCI/iWPR/... y no aparece aquí.               |
//|   2. La capa de órdenes: convertir riesgo en lotes, abrir, mover  |
//|      el stop y cerrar.                                            |
//|                                                                   |
//| NO EDITAR: se regenera en cada exportación.                       |
//+------------------------------------------------------------------+
#property copyright "Analytics System"
#property strict

#include <Trade\Trade.mqh>

// Velas de calentamiento de los indicadores recursivos (KAMA, SAR). Se
// recalculan enteros en cada vela nueva sobre esta ventana en vez de
// mantenerse en un buffer: es más lento pero no puede desincronizarse con el
// histórico, y a una vela por barra el coste es irrelevante.
#define ZCS_CALENTAMIENTO 600

CTrade zcsTrade;

//+------------------------------------------------------------------+
//| Lectura de precios                                                |
//+------------------------------------------------------------------+
// Series en orden CRONOLÓGICO (el índice 0 es la vela más antigua de la
// ventana pedida): así los bucles recursivos se leen igual que en el motor.
bool zcsCopiarCierres(int desde, int cuantos, double &fuera[])
{
   ArraySetAsSeries(fuera, false);
   return(CopyClose(_Symbol, _Period, desde, cuantos, fuera) == cuantos);
}

bool zcsCopiarExtremos(int desde, int cuantos, double &altos[], double &bajos[])
{
   ArraySetAsSeries(altos, false);
   ArraySetAsSeries(bajos, false);
   if(CopyHigh(_Symbol, _Period, desde, cuantos, altos) != cuantos) return(false);
   if(CopyLow(_Symbol, _Period, desde, cuantos, bajos) != cuantos) return(false);
   return(true);
}

//+------------------------------------------------------------------+
//| ATR — media SIMPLE del True Range                                 |
//+------------------------------------------------------------------+
// iATR usa el suavizado de Wilder; el motor promedia el TR con una media
// simple. La diferencia no es cosmética: el ATR fija la distancia al stop y
// de ella sale el número de lotes de CADA operación.
double zcsAtr(int periodo, int shift)
{
   int n = periodo + 1;
   double a[], b[], c[];
   if(!zcsCopiarExtremos(shift, n, a, b)) return(0.0);
   if(!zcsCopiarCierres(shift, n, c))     return(0.0);
   double suma = 0.0;
   for(int i = 1; i < n; i++)
   {
      double tr = MathMax(a[i] - b[i],
                  MathMax(MathAbs(a[i] - c[i - 1]), MathAbs(b[i] - c[i - 1])));
      suma += tr;
   }
   return(suma / periodo);
}

//+------------------------------------------------------------------+
//| Bollinger — desviación MUESTRAL (ddof = 1)                        |
//+------------------------------------------------------------------+
// pandas .std() divide por n-1; iBands divide por n. Con periodos cortos la
// diferencia mueve las bandas lo bastante como para cambiar qué velas
// disparan la señal.
double zcsBbMedia(int periodo, int shift)
{
   double c[];
   if(!zcsCopiarCierres(shift, periodo, c)) return(0.0);
   double suma = 0.0;
   for(int i = 0; i < periodo; i++) suma += c[i];
   return(suma / periodo);
}

double zcsBbDesv(int periodo, int shift)
{
   if(periodo < 2) return(0.0);
   double c[];
   if(!zcsCopiarCierres(shift, periodo, c)) return(0.0);
   double media = 0.0;
   for(int i = 0; i < periodo; i++) media += c[i];
   media /= periodo;
   double s = 0.0;
   for(int i = 0; i < periodo; i++) s += (c[i] - media) * (c[i] - media);
   return(MathSqrt(s / (periodo - 1)));
}

double zcsBbSup(int periodo, double desv, int shift)
{
   return(zcsBbMedia(periodo, shift) + desv * zcsBbDesv(periodo, shift));
}

double zcsBbInf(int periodo, double desv, int shift)
{
   return(zcsBbMedia(periodo, shift) - desv * zcsBbDesv(periodo, shift));
}

//+------------------------------------------------------------------+
//| Efficiency Ratio de Kaufman                                       |
//+------------------------------------------------------------------+
// |movimiento neto| / movimiento total sobre retornos LOG. El warm-up vale
// 0.0, igual que en el motor: con el filtro 'er_rango' (ER < 0.3) las velas
// de calentamiento SÍ pasan el filtro, y descartarlas daría menos
// operaciones que el backtest.
double zcsEr(int periodo, int shift)
{
   int n = periodo + 1;
   double c[];
   if(!zcsCopiarCierres(shift, n, c)) return(0.0);
   if(c[0] <= 0.0) return(0.0);
   double neto = MathAbs(MathLog(c[n - 1] / c[0]));
   double total = 0.0;
   for(int i = 1; i < n; i++)
   {
      if(c[i - 1] <= 0.0) return(0.0);
      total += MathAbs(MathLog(c[i] / c[i - 1]));
   }
   return(total > 0.0 ? neto / total : 0.0);
}

//+------------------------------------------------------------------+
//| KAMA                                                              |
//+------------------------------------------------------------------+
// Media adaptativa: la constante de suavizado se mueve con el ER, así que
// acelera en tendencia y se aplana en rango. Arranca en el propio precio,
// igual que calcular_kama_numba.
double zcsKama(int periodoEr, int rapido, int lento, int shift)
{
   int n = ZCS_CALENTAMIENTO + periodoEr + 1;
   double c[];
   if(!zcsCopiarCierres(shift, n, c)) return(0.0);

   double scRapido = 2.0 / (rapido + 1.0);
   double scLento  = 2.0 / (lento + 1.0);
   double kama = c[periodoEr];

   for(int i = periodoEr + 1; i < n; i++)
   {
      double neto = 0.0, total = 0.0;
      if(c[i - periodoEr] > 0.0) neto = MathAbs(MathLog(c[i] / c[i - periodoEr]));
      for(int k = i - periodoEr + 1; k <= i; k++)
         if(c[k - 1] > 0.0) total += MathAbs(MathLog(c[k] / c[k - 1]));
      double er = (total > 0.0 ? neto / total : 0.0);
      double sc = MathPow(er * (scRapido - scLento) + scLento, 2);
      kama = kama + sc * (c[i] - kama);
   }
   return(kama);
}

//+------------------------------------------------------------------+
//| Parabolic SAR con tendencia explícita                             |
//+------------------------------------------------------------------+
// La estrategia no entra por el NIVEL del SAR sino por su GIRO, y para saber
// que ha girado hace falta la tendencia vigente. Se reimplementa en vez de
// usar iSAR porque ese handle no expone la tendencia, y deducirla de
// «sar < close» falla justo en la vela del giro, que es la única que importa.
//
// La tendencia inicial se asume alcista: misma suposición arbitraria e
// inevitable que hace el motor (no hay historia antes de la primera vela).
void zcsSar(double afIni, double afPaso, double afMax, int shift,
            double &sarFuera, int &tendFuera)
{
   sarFuera = 0.0;
   tendFuera = 0;
   int n = ZCS_CALENTAMIENTO;
   double a[], b[];
   if(!zcsCopiarExtremos(shift, n, a, b)) return;

   double sar = b[0];
   int    tend = 1;
   double af = afIni;
   double ep = a[0];

   for(int i = 1; i < n; i++)
   {
      double nuevo = sar + af * (ep - sar);
      if(tend > 0)
      {
         // el SAR alcista no puede meterse dentro del rango de las dos velas
         // previas: si lo hiciera, el giro saltaría por un movimiento que ya
         // había ocurrido
         double limite = b[i - 1];
         if(i >= 2 && b[i - 2] < limite) limite = b[i - 2];
         if(nuevo > limite) nuevo = limite;
         if(b[i] < nuevo)
         {
            tend = -1;
            sar  = ep;          // al girar, el SAR salta al extremo alcanzado
            ep   = b[i];
            af   = afIni;
         }
         else
         {
            sar = nuevo;
            if(a[i] > ep) { ep = a[i]; af = MathMin(af + afPaso, afMax); }
         }
      }
      else
      {
         double limite = a[i - 1];
         if(i >= 2 && a[i - 2] > limite) limite = a[i - 2];
         if(nuevo < limite) nuevo = limite;
         if(a[i] > nuevo)
         {
            tend = 1;
            sar  = ep;
            ep   = a[i];
            af   = afIni;
         }
         else
         {
            sar = nuevo;
            if(b[i] < ep) { ep = b[i]; af = MathMin(af + afPaso, afMax); }
         }
      }
   }
   sarFuera  = sar;
   tendFuera = tend;
}

//+------------------------------------------------------------------+
//| Canal de Donchian                                                 |
//+------------------------------------------------------------------+
// Desplazado una vela a propósito: si la vela actual formara parte del canal,
// su propio máximo sería el máximo del canal y jamás podría superarlo.
double zcsDonchianSup(bool usaCierres, int periodo, int shift)
{
   double v[];
   int desde = shift + 1;
   if(usaCierres) { if(!zcsCopiarCierres(desde, periodo, v)) return(0.0); }
   else
   {
      ArraySetAsSeries(v, false);
      if(CopyHigh(_Symbol, _Period, desde, periodo, v) != periodo) return(0.0);
   }
   double m = v[0];
   for(int i = 1; i < periodo; i++) if(v[i] > m) m = v[i];
   return(m);
}

double zcsDonchianInf(bool usaCierres, int periodo, int shift)
{
   double v[];
   int desde = shift + 1;
   if(usaCierres) { if(!zcsCopiarCierres(desde, periodo, v)) return(0.0); }
   else
   {
      ArraySetAsSeries(v, false);
      if(CopyLow(_Symbol, _Period, desde, periodo, v) != periodo) return(0.0);
   }
   double m = v[0];
   for(int i = 1; i < periodo; i++) if(v[i] < m) m = v[i];
   return(m);
}

//+------------------------------------------------------------------+
//| Percentil rodante (empates a mitad de rango)                      |
//+------------------------------------------------------------------+
// Sin el reparto de empates, una serie constante daría percentil 100 en todas
// sus velas —cada valor "supera" a todos sus iguales— y un tramo de
// volatilidad plana se leería como volatilidad extrema. Con el rango medio da
// 50. Devuelve -1 si no hay ventana completa, para que no pase ni el corte
// alto ni el bajo.
double zcsPercentilDe(const double &serie[], int ventana)
{
   double actual = serie[ventana - 1];
   int menores = 0, iguales = 0, total = 0;
   for(int i = 0; i < ventana; i++)
   {
      double v = serie[i];
      total++;
      if(v < actual)       menores++;
      else if(v == actual) iguales++;
   }
   if(total <= 0) return(-1.0);
   return(100.0 * (menores + 0.5 * iguales) / total);
}

double zcsPercentilAtr(int periodoBase, int ventana, int shift)
{
   double serie[];
   ArrayResize(serie, ventana);
   for(int i = 0; i < ventana; i++)
      serie[ventana - 1 - i] = zcsAtr(periodoBase, shift + i);
   return(zcsPercentilDe(serie, ventana));
}

// Desviación estándar MUESTRAL de los retornos log, base del filtro de
// volatilidad cuando el método es 'stdev'.
double zcsStdevRet(int periodo, int shift)
{
   int n = periodo + 1;
   double c[];
   if(!zcsCopiarCierres(shift, n, c)) return(0.0);
   double r[];
   ArrayResize(r, periodo);
   double media = 0.0;
   for(int i = 1; i < n; i++)
   {
      r[i - 1] = (c[i - 1] > 0.0 ? MathLog(c[i] / c[i - 1]) : 0.0);
      media += r[i - 1];
   }
   media /= periodo;
   if(periodo < 2) return(0.0);
   double s = 0.0;
   for(int i = 0; i < periodo; i++) s += (r[i] - media) * (r[i] - media);
   return(MathSqrt(s / (periodo - 1)));
}

double zcsPercentilStdev(int periodoBase, int ventana, int shift)
{
   double serie[];
   ArrayResize(serie, ventana);
   for(int i = 0; i < ventana; i++)
      serie[ventana - 1 - i] = zcsStdevRet(periodoBase, shift + i);
   return(zcsPercentilDe(serie, ventana));
}

//+------------------------------------------------------------------+
//| Lectura de un indicador nativo                                    |
//+------------------------------------------------------------------+
double zcsValor(int handle, int buffer, int shift)
{
   if(handle == INVALID_HANDLE) return(0.0);
   double b[];
   ArraySetAsSeries(b, true);
   if(CopyBuffer(handle, buffer, shift, 1, b) <= 0) return(0.0);
   return(b[0]);
}

//+------------------------------------------------------------------+
//| Cruces                                                            |
//+------------------------------------------------------------------+
// Un cruce exige que la vela ANTERIOR estuviera al otro lado: comparar solo
// el estado actual convertiría una condición de cruce en una de posición y
// dispararía en cada vela mientras se mantuviera.
bool zcsCruzaArriba(double a1, double a2, double b1, double b2)
{
   return(a2 <= b2 && a1 > b1);
}

bool zcsCruzaAbajo(double a1, double a2, double b1, double b2)
{
   return(a2 >= b2 && a1 < b1);
}

//+------------------------------------------------------------------+
//| Vela nueva                                                        |
//+------------------------------------------------------------------+
// El motor decide con la vela cerrada y ejecuta al open de la siguiente. El
// EA hace lo mismo actuando solo en el primer tick de cada vela nueva: si
// gestionara en cada tick, el break-even y el trailing saltarían en momentos
// que el backtest nunca vio.
datetime zcsUltimaVela = 0;

bool zcsVelaNueva()
{
   datetime t = iTime(_Symbol, _Period, 0);
   if(t == zcsUltimaVela) return(false);
   zcsUltimaVela = t;
   return(true);
}

//+------------------------------------------------------------------+
//| Tamaño de posición por riesgo                                     |
//+------------------------------------------------------------------+
// El motor arriesga un % del equity ACTUAL y deriva las unidades de la
// distancia al stop. Aquí hay que convertir esas unidades a LOTES, que van en
// escalones que impone el bróker: por debajo del lote mínimo se devuelve 0 y
// la operación NO se abre, en vez de abrirla con un riesgo mayor del pedido.
double zcsNormalizarLotes(double lotes)
{
   double minimo = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maximo = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double paso   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(paso <= 0.0) paso = 0.01;
   double n = MathFloor(lotes / paso) * paso;
   if(n < minimo) return(0.0);
   if(n > maximo) n = maximo;
   return(NormalizeDouble(n, 8));
}

double zcsLotesPorRiesgo(double riesgoPct, double distancia)
{
   if(distancia <= 0.0 || riesgoPct <= 0.0) return(0.0);
   double tickValor = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickTam   = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tickValor <= 0.0 || tickTam <= 0.0) return(0.0);
   double valorPorPrecio = tickValor / tickTam;   // dinero por 1.0 de precio y 1 lote
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double riesgo = equity * riesgoPct;
   return(zcsNormalizarLotes(riesgo / (distancia * valorPorPrecio)));
}

//+------------------------------------------------------------------+
//| Posición del EA (aislada por magic number)                        |
//+------------------------------------------------------------------+
// Cada setup se exporta a su propio EA con su magic. OJO: en una cuenta
// NETTING todas las órdenes del mismo símbolo se funden en UNA posición y el
// aislamiento por magic no basta — hay que usar cuenta HEDGING.
bool zcsSeleccionar(ulong magic)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != magic) continue;
      return(true);
   }
   return(false);
}

int zcsDireccion(ulong magic)
{
   if(!zcsSeleccionar(magic)) return(0);
   return(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? 1 : -1);
}

double zcsPrecioEntrada(ulong magic)
{
   if(!zcsSeleccionar(magic)) return(0.0);
   return(PositionGetDouble(POSITION_PRICE_OPEN));
}

bool zcsAbrir(int dir, double lotes, double stop, double tp, ulong magic)
{
   if(lotes <= 0.0) return(false);
   zcsTrade.SetExpertMagicNumber(magic);
   int digitos = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double s = (stop > 0.0 ? NormalizeDouble(stop, digitos) : 0.0);
   double t = (tp   > 0.0 ? NormalizeDouble(tp,   digitos) : 0.0);
   if(dir > 0) return(zcsTrade.Buy(lotes, _Symbol, 0.0, s, t));
   return(zcsTrade.Sell(lotes, _Symbol, 0.0, s, t));
}

bool zcsMoverStop(double nuevoStop, double tp, ulong magic)
{
   if(!zcsSeleccionar(magic)) return(false);
   int digitos = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double s = (nuevoStop > 0.0 ? NormalizeDouble(nuevoStop, digitos) : 0.0);
   double t = (tp        > 0.0 ? NormalizeDouble(tp,        digitos) : 0.0);
   if(MathAbs(PositionGetDouble(POSITION_SL) - s) < _Point * 0.5 &&
      MathAbs(PositionGetDouble(POSITION_TP) - t) < _Point * 0.5) return(true);
   return(zcsTrade.PositionModify(_Symbol, s, t));
}

bool zcsCerrar(ulong magic)
{
   if(!zcsSeleccionar(magic)) return(false);
   zcsTrade.SetExpertMagicNumber(magic);
   return(zcsTrade.PositionClose(_Symbol));
}

//+------------------------------------------------------------------+
//| Guarda de activo y temporalidad                                   |
//+------------------------------------------------------------------+
// Los parámetros están ajustados a un activo y una temporalidad concretos; en
// otro sitio no significan lo mismo.
//
// La TEMPORALIDAD se comprueba exacta y puede impedir el arranque. El SÍMBOLO
// solo avisa: el nombre del CSV del backtest («zcmaiz») casi nunca coincide
// con el del bróker («ZC», «ZCH2026»…), así que exigir que casen dejaría el EA
// sin arrancar nunca. Devuelve true/false para el arranque y deja en `aviso`
// lo que haya que imprimir aunque se arranque.
bool zcsActivoCorrecto(string simboloEsperado, int minutosEsperados,
                       bool permitirOtro, string &motivo, string &aviso)
{
   motivo = "";
   aviso  = "";
   int minutosActuales = PeriodSeconds(_Period) / 60;
   bool tfOk = (minutosEsperados <= 0 || minutosActuales == minutosEsperados);
   bool simOk = (StringLen(simboloEsperado) == 0 ||
                 StringFind(_Symbol, simboloEsperado) >= 0);

   if(!simOk)
      aviso = StringFormat(
         "AVISO: el backtest se hizo sobre %s y este grafico es %s. "
         "Comprueba que es el mismo activo.", simboloEsperado, _Symbol);

   if(!tfOk)
   {
      motivo = StringFormat(
         "temporalidad %d min, y el backtest se hizo en %d min",
         minutosActuales, minutosEsperados);
      return(permitirOtro);
   }
   return(true);
}
//+------------------------------------------------------------------+
//| Filtro de noticias (calendario de MetaQuotes)                     |
//+------------------------------------------------------------------+
// El backtest evita las ventanas de noticias del calendario de Finnhub; en
// vivo solo existe el calendario de MetaQuotes (CalendarValueHistory), cuya
// clasificacion de impacto y su reparto por divisa no coinciden vela a vela
// con la del proveedor del backtest: por eso este filtro es APROXIMADO (ver
// fidelidad). Devuelve true si hay algun evento con impacto >= umbral y, si
// nMonedas > 0, cuya divisa coincide con alguna de las del instrumento.
bool zcsHayEvento(datetime desde, datetime hasta,
                  ENUM_CALENDAR_EVENT_IMPACT umbral,
                  const string &monedas[], int nMonedas)
{
   MqlCalendarValue valores[];
   int n = CalendarValueHistory(valores, desde, hasta);
   for(int i = 0; i < n; i++)
   {
      if(valores[i].impact_type < umbral) continue;
      if(nMonedas > 0)
      {
         MqlCalendarEvent ev;
         if(!CalendarEventById(valores[i].event_id, ev)) continue;
         bool coincide = false;
         for(int j = 0; j < nMonedas && !coincide; j++)
            if(StringCompare(ev.currency, monedas[j]) == 0) coincide = true;
         if(!coincide) continue;
      }
      return(true);
   }
   return(false);
}
//+------------------------------------------------------------------+

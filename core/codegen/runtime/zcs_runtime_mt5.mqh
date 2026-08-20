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
//| ATR — suavizado de Wilder (RMA)                                   |
//+------------------------------------------------------------------+
// El motor usa el suavizado de Wilder del True Range (semilla SMA de las
// primeras `periodo` velas y recursión), el mismo que iATR. La ventana de
// calentamiento (ZCS_CALENTAMIENTO) siembra la recursión desde historia y
// garantiza que ya haya convergido al llegar a la vela en curso, igual que
// en KAMA/SAR. El ATR fija la distancia al stop y de ella sale el número de
// lotes de CADA operación.
double zcsAtr(int periodo, int shift)
{
   int n = ZCS_CALENTAMIENTO + periodo + 1;
   double a[], b[], c[];
   if(!zcsCopiarExtremos(shift, n, a, b)) return(0.0);
   if(!zcsCopiarCierres(shift, n, c))     return(0.0);
   // La vela más antigua de la ventana (índice 0) no tiene cierre anterior:
   // se usa alto-bajo, el mismo borde que el motor en la primera vela.
   double suma = a[0] - b[0];
   for(int i = 1; i < periodo; i++)
   {
      double tr = MathMax(a[i] - b[i],
                  MathMax(MathAbs(a[i] - c[i - 1]), MathAbs(b[i] - c[i - 1])));
      suma += tr;
   }
   double atr = suma / periodo;   // semilla de Wilder: SMA de las primeras `periodo`
   for(int i = periodo; i < n; i++)
   {
      double tr = MathMax(a[i] - b[i],
                  MathMax(MathAbs(a[i] - c[i - 1]), MathAbs(b[i] - c[i - 1])));
      atr = (atr * (periodo - 1) + tr) / periodo;
   }
   return(atr);
}

//+------------------------------------------------------------------+
//| Bollinger — desviación POBLACIONAL (ddof = 0)                     |
//+------------------------------------------------------------------+
// pandas .std(ddof=0) divide por n, igual que iBands. Con periodos cortos la
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
   return(MathSqrt(s / periodo));
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
//| Exponente de Hurst (R/S, Anis-Lloyd calibrado)                    |
//+------------------------------------------------------------------+
// Replica hurst_rs_numba de core/metrics.py: regresion de log(R/S)
// observado contra log(lag), restando el R/S teorico de Anis-Lloyd
// calibrado por el factor de ruido (0.915). La ventana son los ultimos
// `periodo` retornos log de la vela `shift`; los lags se derivan de
// `periodo` con el mismo criterio que _lags_hurst_defecto
// (core/strategies.py). Si algo no se puede calcular devuelve 0.5
// (regimen neutro), igual que el motor.
double zcsHurst(int periodo, int shift)
{
   int n = periodo + 1;
   double c[];
   if(!zcsCopiarCierres(shift, n, c)) return(0.5);
   double ret[];
   ArrayResize(ret, periodo);
   for(int i = 1; i < n; i++)
   {
      if(c[i - 1] <= 0.0) return(0.5);
      ret[i - 1] = MathLog(c[i] / c[i - 1]);
   }

   // lags derivados de `periodo`: max_lag = mayor potencia de 2 <=
   // max(periodo/4, 8), candidatos max_lag/8 ../4 ../2 y max_lag, >= 4
   int t = MathMax(periodo / 4, 8);
   int max_lag = 1;
   while(max_lag * 2 <= t) max_lag *= 2;
   if(max_lag < 8) max_lag = 8;
   int lags[4];
   int n_lags = 0;
   int cand[4];
   cand[0] = max_lag / 8;
   cand[1] = max_lag / 4;
   cand[2] = max_lag / 2;
   cand[3] = max_lag;
   for(int k = 0; k < 4; k++)
   {
      int lg = cand[k];
      if(lg < 4) continue;
      int j = n_lags;
      while(j > 0 && lags[j - 1] > lg) { lags[j] = lags[j - 1]; j--; }
      if(j > 0 && lags[j - 1] == lg) continue;
      lags[j] = lg;
      n_lags++;
   }
   if(n_lags < 2) { lags[0] = 4; lags[1] = 8; n_lags = 2; }

   double log_lags[4], log_rs[4];
   for(int k = 0; k < n_lags; k++)
   {
      int lag = lags[k];
      int n_chunks = periodo / lag;
      double rs_sum = 0.0;
      int count = 0;
      for(int ch = 0; ch < n_chunks; ch++)
      {
         double m = 0.0;
         for(int v = 0; v < lag; v++) m += ret[ch * lag + v];
         m /= lag;
         double cumsum = 0.0, c_min = 0.0, c_max = 0.0, s_sq = 0.0;
         for(int v = 0; v < lag; v++)
         {
            double d = ret[ch * lag + v] - m;
            cumsum += d;
            if(cumsum < c_min) c_min = cumsum;
            if(cumsum > c_max) c_max = cumsum;
            s_sq += d * d;
         }
         double s = (lag > 1 ? MathSqrt(s_sq / (lag - 1)) : 0.0);
         if(s > 0.0) { rs_sum += (c_max - c_min) / s; count++; }
      }
      double rs_obs = (count > 0 ? rs_sum / count : 0.0);
      // Anis-Lloyd teorico calibrado por el factor de ruido
      double rs_teo = ((lag - 0.5) / lag)
                      * MathPow(1.0 / (2.0 * M_PI * lag), -0.5);
      for(int i = 1; i < lag; i++)
         rs_teo += MathSqrt((lag - i) / (lag * i));
      double rs_teo_aj = rs_teo * 0.915;
      log_lags[k] = MathLog(lag);
      if(rs_obs > 0.0 && rs_teo_aj > 0.0)
         log_rs[k] = MathLog(rs_obs) - MathLog(rs_teo_aj) + MathLog(lag) * 0.5;
      else
         log_rs[k] = MathLog(lag) * 0.5;
   }
   // regresion lineal (OLS) de log_rs contra log_lags
   double mx = 0.0, my = 0.0;
   for(int i = 0; i < n_lags; i++) { mx += log_lags[i]; my += log_rs[i]; }
   mx /= n_lags;
   my /= n_lags;
   double num = 0.0, den = 0.0;
   for(int i = 0; i < n_lags; i++)
   {
      num += (log_lags[i] - mx) * (log_rs[i] - my);
      den += (log_lags[i] - mx) * (log_lags[i] - mx);
   }
   if(den == 0.0) return(0.5);
   double h = num / den;
   if(h < 0.0) return(0.0);
   if(h > 1.0) return(1.0);
   return(h);
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
//| ZigZag no repintante                                              |
//+------------------------------------------------------------------+
// Replica _zigzag_eventos/_zigzag_series de core/strategies.py sobre la
// ventana de calentamiento: pivotes confirmados con der = piernas/2 velas de
// retraso, sustitución solo si el pivote del mismo tipo es más extremo, y
// desviacion % mínima para abrir un tramo de tipo contrario. Devuelve el
// [precio, tipo] del pivote vigente en la vela `shift` (0/0 si aún no hay
// ningún pivote confirmado en o antes de esa vela).
void zcsZigzag(double desviacion, int piernas, int shift,
               double &precio, int &tipo)
{
   precio = 0.0;
   tipo = 0;
   int der = MathMax(1, piernas / 2);
   int izq = der;
   int n = ZCS_CALENTAMIENTO;
   double a[], b[];
   if(!zcsCopiarExtremos(shift, n, a, b)) return;

   // candidatos a pivote: (idx_piv, idx_conf, precio, tipo)
   int n_cand = 0;
   int idx_piv[], idx_conf[], tpo[];
   double prec[];
   ArrayResize(idx_piv, n);
   ArrayResize(idx_conf, n);
   ArrayResize(prec, n);
   ArrayResize(tpo, n);
   int ventana = izq + der + 1;
   for(int i = izq; i < n - der; i++)
   {
      int j = i + der;
      double mh = a[i - izq], ml = b[i - izq];
      for(int k = i - izq + 1; k <= i + der; k++)
      {
         if(a[k] > mh) mh = a[k];
         if(b[k] < ml) ml = b[k];
      }
      if(a[i] >= mh) { idx_piv[n_cand]=i; idx_conf[n_cand]=j; prec[n_cand]=a[i]; tpo[n_cand]=1; n_cand++; }
      if(b[i] <= ml) { idx_piv[n_cand]=i; idx_conf[n_cand]=j; prec[n_cand]=b[i]; tpo[n_cand]=-1; n_cand++; }
   }
   // orden por (idx_conf, idx_piv), insercion
   for(int x = 1; x < n_cand; x++)
   {
      int ci = idx_conf[x], pi = idx_piv[x];
      double pr = prec[x];
      int tt = tpo[x];
      int y = x - 1;
      while(y >= 0 && (idx_conf[y] > ci || (idx_conf[y] == ci && idx_piv[y] > pi)))
      {
         idx_conf[y+1]=idx_conf[y]; idx_piv[y+1]=idx_piv[y];
         prec[y+1]=prec[y]; tpo[y+1]=tpo[y];
         y--;
      }
      idx_conf[y+1]=ci; idx_piv[y+1]=pi; prec[y+1]=pr; tpo[y+1]=tt;
   }
   // posicion de la vela `shift` dentro de la ventana (la mas nueva)
   int pos_shift = n - 1;
   bool hay_ult = false;
   double ult_precio = 0.0;
   int ult_tipo = 0;
   for(int e = 0; e < n_cand; e++)
   {
      if(idx_conf[e] > pos_shift) continue;   // aun no confirmado: no mirar
      if(!hay_ult)
      {
         ult_precio = prec[e]; ult_tipo = tpo[e]; hay_ult = true;
      }
      else if(tpo[e] == ult_tipo)
      {
         if((tpo[e] == 1 && prec[e] > ult_precio) ||
            (tpo[e] == -1 && prec[e] < ult_precio))
            ult_precio = prec[e];
      }
      else
      {
         if(ult_precio == 0.0) continue;
         double variacion = MathAbs(prec[e] - ult_precio)
                            / MathAbs(ult_precio) * 100.0;
         if(variacion < desviacion) continue;
         ult_precio = prec[e];
         ult_tipo = tpo[e];
      }
   }
   if(hay_ult) { precio = ult_precio; tipo = ult_tipo; }
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
//
// La capa de posición AGREGADA es lo que permite la entrada escalonada: cada
// tramo abre su propia posición con el mismo magic, y todo (dirección, precio
// medio, stop, cierre) se resuelve sobre el CONJUNTO de posiciones del magic,
// igual que el motor trabaja sobre la posición fusionada. Con un solo tramo el
// conjunto es una posición y el comportamiento no cambia.
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

double zcsTotalLotes(ulong magic)
{
   double total = 0.0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != magic) continue;
      total += PositionGetDouble(POSITION_VOLUME);
   }
   return(total);
}

int zcsDireccion(ulong magic)
{
   double neto = 0.0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != magic) continue;
      if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
         neto += PositionGetDouble(POSITION_VOLUME);
      else
         neto -= PositionGetDouble(POSITION_VOLUME);
   }
   if(neto > 0.0) return(1);
   if(neto < 0.0) return(-1);
   return(0);
}

double zcsPrecioEntrada(ulong magic)
{
   double sum = 0.0, vol = 0.0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != magic) continue;
      double v = PositionGetDouble(POSITION_VOLUME);
      sum += PositionGetDouble(POSITION_PRICE_OPEN) * v;
      vol += v;
   }
   return(vol > 0.0 ? sum / vol : 0.0);
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
   int digitos = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double s = (nuevoStop > 0.0 ? NormalizeDouble(nuevoStop, digitos) : 0.0);
   double t = (tp        > 0.0 ? NormalizeDouble(tp,        digitos) : 0.0);
   bool toco = false;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != magic) continue;
      toco = true;
      if(MathAbs(PositionGetDouble(POSITION_SL) - s) < _Point * 0.5 &&
         MathAbs(PositionGetDouble(POSITION_TP) - t) < _Point * 0.5) continue;
      zcsTrade.PositionModify(ticket, s, t);
   }
   return(toco);
}

bool zcsCerrar(ulong magic)
{
   bool toco = false;
   zcsTrade.SetExpertMagicNumber(magic);
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != magic) continue;
      toco = true;
      zcsTrade.PositionClose(ticket);
   }
   return(toco);
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
//+------------------------------------------------------------------+
//| Supertrend                                                       |
//+------------------------------------------------------------------+
// Replica _supertrend_serie de core/strategies.py iterando la recursion
// sobre la ventana de calentamiento: envolvente ATR (max/min del nivel
// previo segun el cierre) y tendencia +/-1. Igual convencion que zcsSar.
void zcsSupertrend(int periodo, double mult, int shift,
                   double &nivel, int &tend)
{
   nivel = 0.0;
   tend = 0;
   int n = ZCS_CALENTAMIENTO + periodo + 1;
   double a[], b[], c[];
   if(!zcsCopiarExtremos(shift, n, a, b)) return;
   if(!zcsCopiarCierres(shift, n, c))     return;

   // ATR de Wilder sobre la ventana, vela a vela (mismo borde que zcsAtr:
   // la vela mas antigua usa alto-bajo por no tener cierre anterior)
   double trR = a[0] - b[0];
   double atrV = 0.0;
   double hl2 = 0.0, up = 0.0, dn = 0.0;
   double st = (a[0] + b[0]) / 2.0;
   int tendV = 1;

   for(int i = 1; i < n; i++)
   {
      double tr = MathMax(a[i] - b[i],
                  MathMax(MathAbs(a[i] - c[i - 1]), MathAbs(b[i] - c[i - 1])));
      if(i < periodo) { trR += tr; atrV = (i == periodo - 1) ? trR / periodo : 0.0; }
      else            { trR = (trR * (periodo - 1) + tr) / periodo; atrV = trR; }

      hl2 = (a[i] + b[i]) / 2.0;
      double upI = hl2 - mult * atrV;
      double dnI = hl2 + mult * atrV;
      double upPrev = up, dnPrev = dn;
      up = (c[i - 1] > upPrev) ? MathMax(upI, upPrev) : upI;
      dn = (c[i - 1] < dnPrev) ? MathMin(dnI, dnPrev) : dnI;

      if(tendV > 0)
      {
         if(c[i] < up) { tendV = -1; st = dn; }
         else          { st = up; }
      }
      else
      {
         if(c[i] > dn) { tendV = 1; st = up; }
         else          { st = dn; }
      }
   }
   nivel = st;
   tend = tendV;
}

//+------------------------------------------------------------------+
//| MACD (linea, senal, histograma)                                  |
//+------------------------------------------------------------------+
// Replica _macd_series de core/strategies.py: EMA ajustada (span) de la
// diferencia de EMAs. Las recursiones de la EMA convergen desde la ventana
// de calentamiento, igual que zcsKama.
void zcsMacd(int rapido, int lento, int senal, int shift,
             double &linea, double &senalV, double &hist)
{
   linea = 0.0;
   senalV = 0.0;
   hist = 0.0;
   int n = ZCS_CALENTAMIENTO + lento + senal + 1;
   double c[];
   if(!zcsCopiarCierres(shift, n, c)) return;

   double aR = 2.0 / (rapido + 1.0);
   double aL = 2.0 / (lento + 1.0);
   double aS = 2.0 / (senal + 1.0);
   double eR = c[0], eL = c[0];
   double mLinea = 0.0;      // ema de la linea: se siembra con linea[0] = 0
   double lineaV = 0.0;
   for(int i = 1; i < n; i++)
   {
      eR += aR * (c[i] - eR);
      eL += aL * (c[i] - eL);
      lineaV = eR - eL;
      mLinea += aS * (lineaV - mLinea);
   }
   linea = lineaV;
   senalV = mLinea;
   hist = linea - senalV;
}

//+------------------------------------------------------------------+
//| ADX de Wilder (+DI, -DI, ADX)                                    |
//+------------------------------------------------------------------+
// Replica _adx_series de core/strategies.py con el RMA de Wilder (semilla
// SMA de las primeras `periodo` velas y recursion). Las semillas se siembran
// desde la historia de calentamiento, igual que zcsAtr.
void zcsAdx(int periodo, int shift, double &adx, double &pdi, double &mdi)
{
   adx = 0.0;
   pdi = 0.0;
   mdi = 0.0;
   int n = ZCS_CALENTAMIENTO + 2 * periodo + 1;
   double a[], b[], c[];
   if(!zcsCopiarExtremos(shift, n, a, b)) return;
   if(!zcsCopiarCierres(shift, n, c))     return;

   double trR = a[0] - b[0];      // tr[0]: sin cierre anterior -> alto-bajo
   double pR = 0.0, mR = 0.0;
   double dxSum = 0.0;
   int dxN = 0;
   double adxV = 0.0, pdiV = 0.0, mdiV = 0.0;

   for(int i = 1; i < n; i++)
   {
      double up = a[i] - a[i - 1];
      double dn = -(b[i] - b[i - 1]);
      double plusDm = (up > dn && up > 0.0) ? up : 0.0;
      double minusDm = (dn > up && dn > 0.0) ? dn : 0.0;
      double tr = MathMax(a[i] - b[i],
                  MathMax(MathAbs(a[i] - c[i - 1]), MathAbs(b[i] - c[i - 1])));
      if(i < periodo)
      {
         trR += tr; pR += plusDm; mR += minusDm;
         if(i == periodo - 1) { trR /= periodo; pR /= periodo; mR /= periodo; }
      }
      else
      {
         trR = (trR * (periodo - 1) + tr) / periodo;
         pR = (pR * (periodo - 1) + plusDm) / periodo;
         mR = (mR * (periodo - 1) + minusDm) / periodo;
      }
      if(i >= periodo - 1)
      {
         pdiV = (trR > 0.0) ? 100.0 * pR / trR : 0.0;
         mdiV = (trR > 0.0) ? 100.0 * mR / trR : 0.0;
         double dx = 100.0 * MathAbs(pdiV - mdiV) / MathMax(pdiV + mdiV, 1e-12);
         if(dxN < periodo) { dxSum += dx; dxN++; adxV = dxSum / dxN; }
         else adxV = (adxV * (periodo - 1) + dx) / periodo;
      }
   }
   adx = adxV;
   pdi = pdiV;
   mdi = mdiV;
}

//+------------------------------------------------------------------+
//| Aroon Up/Down                                                    |
//+------------------------------------------------------------------+
// Replica _aroon_series de core/strategies.py: % del recorrido desde el
// ultimo extremo de la ventana de `periodo` velas (incluida la actual);
// 100 cuando el extremo es la vela actual. Empates -> extremo mas reciente.
//+------------------------------------------------------------------+
//| Aroon Up/Down                                                    |
//+------------------------------------------------------------------+
// Replica _aroon_series de core/strategies.py: % del recorrido desde el
// ultimo extremo de la ventana de `periodo` velas (incluida la actual, que
// es el indice 0 de los arrays copiados); 100 cuando el extremo es la vela
// actual. Empates -> extremo mas reciente (primer indice de la ventana).
void zcsAroonUp(int periodo, int shift, double &valor)
{
   valor = 0.0;
   int n = ZCS_CALENTAMIENTO + periodo + 1;
   double a[], b[];
   if(!zcsCopiarExtremos(shift, n, a, b)) return;
   double maxH = a[0];
   int pos = 0;
   for(int i = 0; i < periodo; i++)
      if(a[i] >= maxH) { maxH = a[i]; pos = i; }
   valor = 100.0 * (periodo - pos) / periodo;
}

void zcsAroonDown(int periodo, int shift, double &valor)
{
   valor = 0.0;
   int n = ZCS_CALENTAMIENTO + periodo + 1;
   double a[], b[];
   if(!zcsCopiarExtremos(shift, n, a, b)) return;
   double minL = b[0];
   int pos = 0;
   for(int i = 0; i < periodo; i++)
      if(b[i] <= minL) { minL = b[i]; pos = i; }
   valor = 100.0 * (periodo - pos) / periodo;
}

//+------------------------------------------------------------------+
//| CMO (Chande Momentum Oscillator)                                 |
//+------------------------------------------------------------------+
// Replica _cmo_serie de core/strategies.py: 100·(SU−SD)/(SU+SD) con SU/SD
// la suma de los cambios al alza/baja de `periodo` velas.
void zcsCmo(int periodo, int shift, double &valor)
{
   valor = 0.0;
   int n = ZCS_CALENTAMIENTO + periodo + 1;
   double c[];
   if(!zcsCopiarCierres(shift, n, c)) return;
   double su = 0.0, sd = 0.0;
   for(int i = 0; i < periodo; i++)
   {
      double chg = c[i] - c[i + 1];
      if(chg > 0.0) su += chg;
      else          sd += -chg;
   }
   if(su + sd > 0.0) valor = 100.0 * (su - sd) / (su + sd);
}

//+------------------------------------------------------------------+
//| TRIX                                                             |
//+------------------------------------------------------------------+
// Replica _trix_serie de core/strategies.py: % de cambio de la EMA triple.
// Las EMAs se recorren desde la vela mas antigua (indice n-1) hacia la
// actual (indice 0), terminando con el valor de la vela anterior (indice 1)
// y el de la actual para el % de cambio.
void zcsTrix(int periodo, int shift, double &valor)
{
   valor = 0.0;
   int n = ZCS_CALENTAMIENTO + periodo + 1;
   double c[];
   if(!zcsCopiarCierres(shift, n, c)) return;
   double a = 2.0 / (periodo + 1.0);
   double e1 = c[n - 1], e2 = c[n - 1], e3 = c[n - 1];
   for(int i = n - 2; i >= 1; i--)
   {
      e1 += a * (c[i] - e1);
      e2 += a * (e1 - e2);
      e3 += a * (e2 - e3);
   }
   double e3Prev = e3;
   e1 += a * (c[0] - e1);
   e2 += a * (e1 - e2);
   e3 += a * (e2 - e3);
   if(e3Prev != 0.0) valor = (e3 - e3Prev) / e3Prev * 100.0;
}

//+------------------------------------------------------------------+
//| StochRSI (%K y %D)                                               |
//+------------------------------------------------------------------+
// Replica _stochrsi_series de core/strategies.py: estocastico sobre el RSI
// (Wilder con semilla SMA, como ta.rsi) suavizado %K=3 y %D=3. El RSI se
// recorre de la vela mas antigua (n-1) a la actual (0).
void zcsStochRsiK(int periodo, int shift, double &valor)
{
   valor = 0.0;
   int n = ZCS_CALENTAMIENTO + periodo + 8;
   double c[];
   if(!zcsCopiarCierres(shift, n, c)) return;
   double r[];
   ArrayResize(r, n);
   double up = 0.0, dn = 0.0;
   r[n - 1] = 50.0;
   for(int i = n - 2; i >= 0; i--)
   {
      double chg = c[i] - c[i + 1];
      double u = chg > 0.0 ? chg : 0.0;
      double d = chg < 0.0 ? -chg : 0.0;
      int pos = (n - 1) - i;
      if(pos < periodo)
      {
         up += u; dn += d;
         if(pos == periodo - 1) { up /= periodo; dn /= periodo; }
      }
      else
      {
         up = (up * (periodo - 1) + u) / periodo;
         dn = (dn * (periodo - 1) + d) / periodo;
      }
      r[i] = dn > 0.0 ? 100.0 - 100.0 / (1.0 + up / dn)
                      : (up > 0.0 ? 100.0 : 50.0);
   }
   double raw[], k[];
   ArrayResize(raw, n); ArrayResize(k, n);
   for(int i = 0; i < n - periodo; i++)
   {
      double mn = r[i], mx = r[i];
      for(int j = i; j < i + periodo; j++)
      {
         if(r[j] < mn) mn = r[j];
         if(r[j] > mx) mx = r[j];
      }
      raw[i] = (mx - mn > 0.0) ? (r[i] - mn) / (mx - mn) : 0.0;
   }
   for(int i = 0; i < n - periodo - 2; i++)
      k[i] = (raw[i] + raw[i + 1] + raw[i + 2]) / 3.0;
   if(n - periodo - 2 > 0) valor = k[0];
}

void zcsStochRsiD(int periodo, int shift, double &valor)
{
   valor = 0.0;
   int n = ZCS_CALENTAMIENTO + periodo + 8;
   double c[];
   if(!zcsCopiarCierres(shift, n, c)) return;
   double r[];
   ArrayResize(r, n);
   double up = 0.0, dn = 0.0;
   r[n - 1] = 50.0;
   for(int i = n - 2; i >= 0; i--)
   {
      double chg = c[i] - c[i + 1];
      double u = chg > 0.0 ? chg : 0.0;
      double d = chg < 0.0 ? -chg : 0.0;
      int pos = (n - 1) - i;
      if(pos < periodo)
      {
         up += u; dn += d;
         if(pos == periodo - 1) { up /= periodo; dn /= periodo; }
      }
      else
      {
         up = (up * (periodo - 1) + u) / periodo;
         dn = (dn * (periodo - 1) + d) / periodo;
      }
      r[i] = dn > 0.0 ? 100.0 - 100.0 / (1.0 + up / dn)
                      : (up > 0.0 ? 100.0 : 50.0);
   }
   double raw[], k[];
   ArrayResize(raw, n); ArrayResize(k, n);
   for(int i = 0; i < n - periodo; i++)
   {
      double mn = r[i], mx = r[i];
      for(int j = i; j < i + periodo; j++)
      {
         if(r[j] < mn) mn = r[j];
         if(r[j] > mx) mx = r[j];
      }
      raw[i] = (mx - mn > 0.0) ? (r[i] - mn) / (mx - mn) : 0.0;
   }
   for(int i = 0; i < n - periodo - 2; i++)
      k[i] = (raw[i] + raw[i + 1] + raw[i + 2]) / 3.0;
   if(n - periodo - 4 > 0)
      valor = (k[0] + k[1] + k[2]) / 3.0;
}

//+------------------------------------------------------------------+
//| Ichimoku (alineado)                                              |
//+------------------------------------------------------------------+
// Replica _ichimoku_series de core/strategies.py. Las Senkou se devuelven
// ALINEADAS (sin el desplazamiento +26 del grafico: seria lookahead) y el
// Chikou es el cierre 26 velas ATRAS (dato pasado).
void zcsIchimokuTenkan(int t, int k, int s, int shift, double &valor)
{
   valor = 0.0;
   int n = ZCS_CALENTAMIENTO + t + 1;
   double a[], b[];
   if(!zcsCopiarExtremos(shift, n, a, b)) return;
   double suma = 0.0;
   for(int i = 0; i < t; i++) suma += (a[i] + b[i]) / 2.0;
   valor = suma / t;
}

void zcsIchimokuKijun(int t, int k, int s, int shift, double &valor)
{
   valor = 0.0;
   int n = ZCS_CALENTAMIENTO + k + 1;
   double a[], b[];
   if(!zcsCopiarExtremos(shift, n, a, b)) return;
   double suma = 0.0;
   for(int i = 0; i < k; i++) suma += (a[i] + b[i]) / 2.0;
   valor = suma / k;
}

void zcsIchimokuSenkouA(int t, int k, int s, int shift, double &valor)
{
   double tenkan, kijun;
   zcsIchimokuTenkan(t, k, s, shift, tenkan);
   zcsIchimokuKijun(t, k, s, shift, kijun);
   valor = (tenkan + kijun) / 2.0;
}

void zcsIchimokuSenkouB(int t, int k, int s, int shift, double &valor)
{
   valor = 0.0;
   int n = ZCS_CALENTAMIENTO + s + 1;
   double a[], b[];
   if(!zcsCopiarExtremos(shift, n, a, b)) return;
   double suma = 0.0;
   for(int i = 0; i < s; i++) suma += (a[i] + b[i]) / 2.0;
   valor = suma / s;
}

void zcsIchimokuChikou(int t, int k, int s, int shift, double &valor)
{
   valor = 0.0;
   int n = ZCS_CALENTAMIENTO + 27;
   double c[];
   if(!zcsCopiarCierres(shift, n, c)) return;
   valor = c[26];   // cierre 26 velas atras (indice 26 = vela actual - 26)
}

//+------------------------------------------------------------------+
//| Keltner                                                          |
//+------------------------------------------------------------------+
// Replica _keltner_series de core/strategies.py: EMA ± mult×ATR.
void zcsKeltnerMedia(int periodo, double mult, int shift, double &valor)
{
   valor = 0.0;
   int n = ZCS_CALENTAMIENTO + periodo + 1;
   double c[];
   if(!zcsCopiarCierres(shift, n, c)) return;
   double a = 2.0 / (periodo + 1.0);
   double e = c[n - 1];
   for(int i = n - 2; i >= 0; i--) e += a * (c[i] - e);
   valor = e;
}

void zcsKeltnerSup(int periodo, double mult, int shift, double &valor)
{
   double media, atrV;
   zcsKeltnerMedia(periodo, mult, shift, media);
   atrV = zcsAtr(periodo, shift);
   valor = media + mult * atrV;
}

void zcsKeltnerInf(int periodo, double mult, int shift, double &valor)
{
   double media, atrV;
   zcsKeltnerMedia(periodo, mult, shift, media);
   atrV = zcsAtr(periodo, shift);
   valor = media - mult * atrV;
}

//+------------------------------------------------------------------+
//| TTM Squeeze                                                      |
//+------------------------------------------------------------------+
// Replica _ttm_squeeze_series de core/strategies.py: squeeze = Bollinger
// dentro de Keltner (1.0/0.0); momentum = midprice − SMA(midprice).
void zcsTtmSqueeze(int periodo, double multBb, double multKc, int shift,
                   double &valor)
{
   double sup, inf;
   sup = zcsBbSup(periodo, multBb, shift);
   inf = zcsBbInf(periodo, multBb, shift);
   double kSup, kInf;
   zcsKeltnerSup(periodo, multKc, shift, kSup);
   zcsKeltnerInf(periodo, multKc, shift, kInf);
   valor = (sup <= kSup && inf >= kInf) ? 1.0 : 0.0;
}

void zcsTtmMomentum(int periodo, double multBb, double multKc, int shift,
                    double &valor)
{
   valor = 0.0;
   int n = ZCS_CALENTAMIENTO + periodo + 1;
   double a[], b[];
   if(!zcsCopiarExtremos(shift, n, a, b)) return;
   double mid = (a[0] + b[0]) / 2.0;
   double suma = 0.0;
   for(int i = 0; i < periodo; i++) suma += (a[i] + b[i]) / 2.0;
   valor = mid - suma / periodo;
}

//+------------------------------------------------------------------+
//| VWAP (anclaje + bandas)                                          |
//+------------------------------------------------------------------+
// Replica _vwap_series de core/strategies.py (referencia TradingView):
// acumulado de hlc3·vol / vol por anclaje de calendario; banda = media ±
// (σ poblacional del anclaje o media·0.01)·k. La vela de señal es el
// indice 0 de los arrays; el anclaje se detecta con iTime. Aproximado:
// la alineacion exacta barra/volumen depende de la plataforma.
bool zcsNuevoAncla(string anclaje, datetime t)
{
   MqlDateTime dt;
   TimeToStruct(t, dt);
   if(anclaje == "D")   return(true);
   if(anclaje == "W")   return(dt.day_of_week == 0);
   if(anclaje == "M")   return(dt.day == 1);
   if(anclaje == "T")   return(dt.day == 1 && (dt.mon % 3) == 1);
   if(anclaje == "A")   return(dt.day == 1 && dt.mon == 1);
   if(anclaje == "10Y") return(dt.day == 1 && dt.mon == 1 && (dt.year % 10) == 0);
   if(anclaje == "100Y")return(dt.day == 1 && dt.mon == 1 && (dt.year % 100) == 0);
   return(true);
}

void zcsVwapMedia(string anclaje, double k, string modo, int shift,
                  double &valor)
{
   valor = 0.0;
   int n = ZCS_CALENTAMIENTO + 1;
   double h[], l[], c[];
   if(!zcsCopiarExtremos(shift, n, h, l)) return;
   if(!zcsCopiarCierres(shift, n, c)) return;
   double acT = 0.0, acV = 0.0;
   bool anclaActiva = true;
   for(int i = 0; i < n; i++)
   {
      datetime t = iTime(_Symbol, _Period, i);
      if(anclaActiva || zcsNuevoAncla(anclaje, t))
      {
         acT = 0.0;
         acV = 0.0;
         anclaActiva = false;
      }
      double src = (h[i] + l[i] + c[i]) / 3.0;
      double vol = (double)iVolume(_Symbol, _Period, i);
      acT += src * vol;
      acV += vol;
   }
   if(acV > 0.0) valor = acT / acV;
}

void zcsVwapSd(string anclaje, double k, string modo, int shift, double &valor)
{
   valor = 0.0;
   int n = ZCS_CALENTAMIENTO + 1;
   double h[], l[], c[];
   if(!zcsCopiarExtremos(shift, n, h, l)) return;
   if(!zcsCopiarCierres(shift, n, c)) return;
   double sumS = 0.0, sumS2 = 0.0;
   int cnt = 0;
   bool anclaActiva = true;
   for(int i = 0; i < n; i++)
   {
      datetime t = iTime(_Symbol, _Period, i);
      if(anclaActiva || zcsNuevoAncla(anclaje, t))
      {
         sumS = 0.0;
         sumS2 = 0.0;
         cnt = 0;
         anclaActiva = false;
      }
      double src = (h[i] + l[i] + c[i]) / 3.0;
      sumS += src;
      sumS2 += src * src;
      cnt++;
   }
   if(cnt > 0)
   {
      double media = sumS / cnt;
      valor = MathSqrt(MathMax(sumS2 / cnt - media * media, 0.0));
   }
}

void zcsVwapSup(string anclaje, double k, string modo, int shift, double &valor)
{
   double m, base;
   zcsVwapMedia(anclaje, k, modo, shift, m);
   if(modo == "pct") base = m * 0.01;
   else              zcsVwapSd(anclaje, k, modo, shift, base);
   valor = m + base * k;
}

void zcsVwapInf(string anclaje, double k, string modo, int shift, double &valor)
{
   double m, base;
   zcsVwapMedia(anclaje, k, modo, shift, m);
   if(modo == "pct") base = m * 0.01;
   else              zcsVwapSd(anclaje, k, modo, shift, base);
   valor = m - base * k;
}

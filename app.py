# =============================================================================
# ANÁLISIS DE PORTAFOLIO DE INVERSIÓN — Taller 7 EAFIT MAF
# Réplica en Python/Streamlit del modelo TALLER_7_INTEGRADO_ENTREGA_FINAL.xlsb
#
# Autor:     [Tu nombre]
# Programa:  Maestría en Administración Financiera — EAFIT
# Stack:     Python 3.11 | Streamlit | yfinance | pandas | numpy | plotly | scipy
# Deploy:    https://share.streamlit.io (gratuito)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis de Portafolio — MAF EAFIT",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# ESTILOS CSS PERSONALIZADOS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo general */
    .stApp { background-color: #0c1410; color: #eef1ec; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0f1a16; border-right: 1px solid #1e2e26; }
    
    /* Métricas */
    [data-testid="stMetric"] {
        background-color: #121d18;
        border: 1px solid #1e2e26;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] { color: #c9a24b; font-size: 1.6rem !important; }
    [data-testid="stMetricLabel"] { color: #8aa398; font-size: 0.75rem; }
    
    /* Títulos */
    h1 { color: #c9a24b !important; font-family: Georgia, serif; }
    h2 { color: #eef1ec !important; }
    h3 { color: #5bb3a0 !important; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] { color: #8aa398; background-color: #121d18; }
    .stTabs [aria-selected="true"] { color: #c9a24b !important; border-bottom-color: #c9a24b !important; }
    
    /* Inputs */
    .stTextInput > div > div > input { background-color: #121d18; color: #eef1ec; border-color: #1e2e26; }
    .stSelectbox > div > div { background-color: #121d18; color: #eef1ec; }
    
    /* Botón principal */
    .stButton > button {
        background-color: #c9a24b;
        color: #1a1408;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        width: 100%;
    }
    .stButton > button:hover { background-color: #e0b85e; }
    
    /* Expander */
    .streamlit-expanderHeader { background-color: #121d18; color: #eef1ec; border: 1px solid #1e2e26; border-radius: 8px; }
    
    /* Dataframes */
    .stDataFrame { background-color: #121d18; }
    
    /* Info boxes */
    .info-box {
        background-color: #121d18;
        border: 1px solid #1e2e26;
        border-left: 3px solid #c9a24b;
        border-radius: 0 8px 8px 0;
        padding: 14px 16px;
        margin: 10px 0;
        font-size: 0.9rem;
        color: #8aa398;
        line-height: 1.7;
    }
    .info-box b { color: #eef1ec; }
    .info-box.teal { border-left-color: #5bb3a0; }
    .info-box.red { border-left-color: #c2685a; }
    .info-box.green { border-left-color: #6aae66; }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-family: monospace;
        margin: 2px;
    }
    .badge-teal { background: #12211f; color: #5bb3a0; border: 1px solid #2a4a42; }
    .badge-gold { background: #1e1812; color: #c9a24b; border: 1px solid #4a3a12; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES CORE — EQUIVALENTES A LAS HOJAS DEL TALLER 7
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def descargar_precios(tickers: list, periodo: str) -> pd.DataFrame:
    """
    Equivale a: AP_Data + Precios_Ajustados
    Descarga precios ajustados históricos de Yahoo Finance.
    ttl=3600 significa que cachea por 1 hora (no reconsulta en cada recarga).
    """
    try:
        data = yf.download(
            tickers,
            period=periodo,
            auto_adjust=True,
            progress=False,
            threads=True
        )["Close"]
        
        # Si es un solo ticker, convertir a DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        
        # Limpiar NaN con forward fill luego backward fill
        data = data.ffill().bfill().dropna()
        return data
    except Exception as e:
        st.error(f"Error descargando datos: {e}")
        return pd.DataFrame()


def calcular_rendimientos(precios: pd.DataFrame) -> pd.DataFrame:
    """
    Equivale a: AP_Rendimientos
    Calcula retornos logarítmicos diarios: ln(P_t / P_{t-1})
    Los retornos log son el estándar en finanzas cuantitativas porque:
    - Son aditivos en el tiempo (puedes sumar retornos diarios)
    - Son simétricos (una caída de 50% se cancela exactamente con una subida de 100%)
    - Se distribuyen aproximadamente normal (supuesto BSM)
    """
    return np.log(precios / precios.shift(1)).dropna()


def calcular_base100(precios: pd.DataFrame) -> pd.DataFrame:
    """
    Equivale a: AP_ind_Base100
    Normaliza todos los activos a 100 en la primera fecha.
    Permite comparar visualmente activos con precios muy distintos
    (ej: NVDA $800 vs BTC-USD $60,000 en la misma escala).
    """
    return (precios / precios.iloc[0]) * 100


def calcular_matriz_covarianza(rendimientos: pd.DataFrame) -> pd.DataFrame:
    """
    Equivale a: AP_Covarianza
    Covarianza mide cómo se mueven dos activos JUNTOS (en magnitud).
    Fórmula: Cov(A,B) = E[(A - μ_A)(B - μ_B)]
    - Cov > 0: suben y bajan juntos
    - Cov < 0: se mueven en direcciones opuestas (diversificación!)
    - Cov = 0: movimientos independientes
    """
    return rendimientos.cov()


def calcular_matriz_correlacion(rendimientos: pd.DataFrame) -> pd.DataFrame:
    """
    Equivale a: AP_Correlacion
    Correlación = versión normalizada de la covarianza, entre -1 y 1.
    Corr(A,B) = Cov(A,B) / (σ_A × σ_B)
    Más fácil de interpretar que la covarianza.
    """
    return rendimientos.corr()


def calcular_metricas_portafolio(rendimientos: pd.DataFrame,
                                  pesos: np.ndarray,
                                  rf: float = 0.0457) -> dict:
    """
    Equivale a: AP_analisis
    Calcula las métricas principales del portafolio combinado.
    rf = tasa libre de riesgo (bono UST 10 años, ~4.57% en 2025)
    """
    # Retorno esperado diario ponderado
    ret_diarios = rendimientos.mean().values
    ret_port_diario = np.dot(pesos, ret_diarios)

    # Varianza del portafolio: w' × Σ × w
    # Esta fórmula captura no solo la volatilidad individual sino las correlaciones
    cov_matrix = rendimientos.cov().values
    var_port_diario = np.dot(pesos, np.dot(cov_matrix, pesos))
    vol_port_diario = np.sqrt(max(var_port_diario, 0))

    # Anualización (252 días de mercado por año)
    ret_anual = ret_port_diario * 252
    vol_anual = vol_port_diario * np.sqrt(252)

    # Sharpe Ratio: retorno excedente por unidad de riesgo total
    # Sharpe > 1 = bueno, > 2 = excelente, < 0 = peor que el activo libre de riesgo
    sharpe = (ret_anual - rf) / vol_anual if vol_anual > 0 else 0.0

    # Métricas individuales por activo
    metricas_ind = {}
    for i, ticker in enumerate(rendimientos.columns):
        ret_ind = rendimientos[ticker].mean() * 252
        vol_ind = rendimientos[ticker].std() * np.sqrt(252)
        sharpe_ind = (ret_ind - rf) / vol_ind if vol_ind > 0 else 0
        metricas_ind[ticker] = {
            "peso": pesos[i],
            "retorno_anual": ret_ind,
            "volatilidad_anual": vol_ind,
            "sharpe": sharpe_ind,
            "retorno_total": (rendimientos[ticker] + 1).prod() - 1
        }

    return {
        "retorno_anual": ret_anual,
        "volatilidad_anual": vol_anual,
        "sharpe": sharpe,
        "var_95": -(ret_port_diario - 1.645 * vol_port_diario) * np.sqrt(252),
        "metricas_individuales": metricas_ind
    }


def simulacion_montecarlo(rendimientos: pd.DataFrame,
                           pesos: np.ndarray,
                           n_sim: int = 1000,
                           horizonte: int = 252) -> dict:
    """
    Equivale a: AP_Montecarlo365 / AP_Montecarlo1000
    
    CONCEPTO CLAVE: Descomposición de Cholesky
    La matriz de covarianza Σ se descompone como Σ = L × L' donde L es triangular inferior.
    Esto permite generar shocks aleatorios CORRELACIONADOS entre activos.
    Sin Cholesky, simularías como si NVDA y AMD se movieran independientemente — error grave.
    
    El proceso por cada día simulado:
    1. Generar vector de números aleatorios normales z ~ N(0,1)  
    2. Correlacionarlos: shocks = L × z  → ahora respetan las correlaciones históricas
    3. Aplicar retorno: w' × (μ + shocks) para obtener el retorno del portafolio ese día
    4. Acumular: valor_{t+1} = valor_t × (1 + retorno_diario)
    """
    n_activos = len(rendimientos.columns)
    medias = rendimientos.mean().values
    cov_matrix = rendimientos.cov().values
    
    # Descomposición de Cholesky
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # Si la matriz no es definida positiva, regularizar levemente
        cov_matrix += np.eye(n_activos) * 1e-8
        L = np.linalg.cholesky(cov_matrix)
    
    # Almacenar trayectorias (solo guardamos percentiles para eficiencia de memoria)
    trayectorias = np.zeros((n_sim, horizonte + 1))
    trayectorias[:, 0] = 100  # Base 100
    
    for sim in range(n_sim):
        valor = 100.0
        for dia in range(horizonte):
            # Generar shocks correlacionados
            z = np.random.standard_normal(n_activos)
            shocks_correlacionados = L @ z  # Multiplicación matricial
            retornos_diarios = medias + shocks_correlacionados
            retorno_port = np.dot(pesos, retornos_diarios)
            valor = valor * (1 + retorno_port)
            trayectorias[sim, dia + 1] = valor
    
    # Calcular percentiles en cada día (más eficiente que guardar todas las trayectorias)
    percentiles = {}
    for p in [5, 10, 25, 50, 75, 90, 95]:
        percentiles[f"p{p}"] = np.percentile(trayectorias, p, axis=0)
    
    # Distribución de valores finales
    valores_finales = trayectorias[:, -1]
    
    return {
        "percentiles": percentiles,
        "valores_finales": valores_finales,
        "var_95": np.percentile(valores_finales, 5),      # VaR 95%
        "cvar_95": valores_finales[valores_finales <= np.percentile(valores_finales, 5)].mean(),
        "mediana": np.median(valores_finales),
        "prob_ganancia": (valores_finales > 100).mean() * 100,
        "n_sim": n_sim,
        "horizonte": horizonte,
        # Muestra de 50 trayectorias para graficar (sin saturar)
        "muestra_trayectorias": trayectorias[np.random.choice(n_sim, min(50, n_sim), replace=False)]
    }


def optimizar_portafolio(rendimientos: pd.DataFrame, rf: float = 0.0457) -> dict:
    """
    Equivale a: La optimización que faltaba en el Taller 7
    Encuentra los pesos ÓPTIMOS del portafolio maximizando el Sharpe Ratio.
    Esto es la implementación computacional de la Teoría de Markowitz (Premio Nobel 1990).
    
    El optimizador (scipy.minimize con método SLSQP) resuelve:
    max  Sharpe(w) = (μ_p - rf) / σ_p
    s.t. Σw_i = 1    (pesos suman 100%)
         w_i ≥ 0     (no posiciones cortas, puede relajarse)
    """
    n = len(rendimientos.columns)
    medias = rendimientos.mean().values * 252
    cov_anual = rendimientos.cov().values * 252

    def neg_sharpe(pesos):
        ret = np.dot(pesos, medias)
        vol = np.sqrt(np.dot(pesos, np.dot(cov_anual, pesos)))
        return -(ret - rf) / vol if vol > 0 else 0

    def ret_portafolio(pesos):
        return np.dot(pesos, medias)

    def vol_portafolio(pesos):
        return np.sqrt(np.dot(pesos, np.dot(cov_anual, pesos)))

    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = tuple((0.01, 0.60) for _ in range(n))  # Entre 1% y 60% por activo
    w0 = np.array([1 / n] * n)  # Pesos iniciales: igual peso

    # Portafolio de máximo Sharpe
    res_sharpe = minimize(neg_sharpe, w0, method="SLSQP",
                          bounds=bounds, constraints=restricciones,
                          options={"ftol": 1e-9, "maxiter": 1000})

    # Portafolio de mínima varianza
    res_minvar = minimize(vol_portafolio, w0, method="SLSQP",
                          bounds=bounds, constraints=restricciones,
                          options={"ftol": 1e-9, "maxiter": 1000})

    # Frontera eficiente: 50 puntos entre mínima varianza y máximo retorno
    pesos_frontera = []
    targets = np.linspace(ret_portafolio(res_minvar.x), max(medias), 50)
    for target in targets:
        restricciones_frontera = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: ret_portafolio(w) - t}
        ]
        res = minimize(vol_portafolio, w0, method="SLSQP",
                       bounds=bounds, constraints=restricciones_frontera,
                       options={"ftol": 1e-9, "maxiter": 500})
        if res.success:
            pesos_frontera.append(res.x)

    frontera = [(vol_portafolio(w) * 100, ret_portafolio(w) * 100,
                 (ret_portafolio(w) - rf) / vol_portafolio(w))
                for w in pesos_frontera]

    return {
        "pesos_sharpe": res_sharpe.x,
        "pesos_minvar": res_minvar.x,
        "frontera": frontera,
        "tickers": list(rendimientos.columns)
    }


def analisis_tecnico(precios_serie: pd.Series) -> pd.DataFrame:
    """
    Equivale a: Val Técnica
    Calcula indicadores técnicos: Medias Móviles, RSI, MACD, Bandas de Bollinger.
    
    MEDIAS MÓVILES (SMA): promedio del precio en los últimos N días.
    - SMA20 cruza SMA50 hacia arriba → señal de compra (Golden Cross)
    - SMA20 cruza SMA50 hacia abajo → señal de venta (Death Cross)
    
    RSI (Índice de Fuerza Relativa): oscila entre 0 y 100.
    - RSI > 70: sobrecomprado → posible caída
    - RSI < 30: sobrevendido → posible recuperación
    
    MACD: diferencia entre EMA12 y EMA26. Cuando cruza la línea signal → cambio de tendencia.
    """
    df = pd.DataFrame({"precio": precios_serie})

    # Medias Móviles Simples
    for n in [20, 50, 200]:
        df[f"SMA{n}"] = df["precio"].rolling(window=n).mean()

    # Medias Móviles Exponenciales (dan más peso a precios recientes)
    for n in [12, 26]:
        df[f"EMA{n}"] = df["precio"].ewm(span=n, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # RSI (Relative Strength Index)
    delta = df["precio"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bandas de Bollinger
    df["BB_Media"] = df["precio"].rolling(window=20).mean()
    df["BB_Std"] = df["precio"].rolling(window=20).std()
    df["BB_Superior"] = df["BB_Media"] + 2 * df["BB_Std"]
    df["BB_Inferior"] = df["BB_Media"] - 2 * df["BB_Std"]
    df["BB_Ancho"] = (df["BB_Superior"] - df["BB_Inferior"]) / df["BB_Media"] * 100

    return df


def regresion_lineal(precios_activo: pd.Series,
                     precios_benchmark: pd.Series) -> dict:
    """
    Equivale a: Val Estadistica
    Regresión OLS: R_activo = α + β × R_benchmark + ε
    
    β (Beta): sensibilidad del activo al mercado
       β > 1: más volátil que el mercado (amplifica movimientos)
       β = 1: se mueve igual que el mercado
       β < 1: menos volátil (defensivo)
       β < 0: se mueve en contra del mercado (cobertura natural)
    
    α (Alpha de Jensen): retorno que el activo genera MÁS ALLÁ de lo que
       predice el CAPM. α > 0 = el activo "agrega valor".
    
    R²: qué % del movimiento del activo se explica por el mercado.
       R² = 0.75 → el 75% de la variación es sistemática (de mercado),
                    el 25% es idiosincrática (específica de la empresa, diversificable).
    """
    # Retornos log diarios
    ret_activo = np.log(precios_activo / precios_activo.shift(1)).dropna()
    ret_benchmark = np.log(precios_benchmark / precios_benchmark.shift(1)).dropna()

    # Alinear fechas
    df = pd.DataFrame({"activo": ret_activo, "benchmark": ret_benchmark}).dropna()

    # Regresión OLS
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df["benchmark"], df["activo"]
    )

    return {
        "beta": slope,
        "alpha_diario": intercept,
        "alpha_anual": intercept * 252,
        "r_cuadrado": r_value ** 2,
        "p_value": p_value,
        "datos": df,
        "n_obs": len(df)
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE VISUALIZACIÓN — PLOTLY (Interactivo)
# ─────────────────────────────────────────────────────────────────────────────

COLORES = ["#c9a24b", "#5bb3a0", "#c2685a", "#7d9bc1", "#9b7ec8",
           "#8fb574", "#d49a6a", "#e87d9b", "#4db8d4", "#a0b86e"]

LAYOUT_BASE = dict(
    paper_bgcolor="#0c1410",
    plot_bgcolor="#0c1410",
    font=dict(color="#eef1ec", family="Inter, Arial, sans-serif"),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="#121d18", bordercolor="#1e2e26", borderwidth=1)
)


def fig_base100(base100: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(base100.columns):
        fig.add_trace(go.Scatter(
            x=base100.index, y=base100[col],
            name=col, mode="lines",
            line=dict(color=COLORES[i % len(COLORES)], width=2),
            hovertemplate=f"<b>{col}</b><br>Fecha: %{{x|%d-%b-%Y}}<br>Base 100: %{{y:.1f}}<extra></extra>"
        ))
    fig.update_layout(
        **LAYOUT_BASE, title="Evolución Indexada (Base 100)",
        xaxis=dict(gridcolor="#1e2e26", title="Fecha"),
        yaxis=dict(gridcolor="#1e2e26", title="Valor (Base 100)")
    )
    return fig


def fig_correlacion(corr: pd.DataFrame) -> go.Figure:
    tickers = corr.columns.tolist()
    z = corr.values
    
    # Crear texto con valores para mostrar en celdas
    texto = [[f"{z[i][j]:.2f}" for j in range(len(tickers))] for i in range(len(tickers))]
    
    fig = go.Figure(data=go.Heatmap(
        z=z, x=tickers, y=tickers, text=texto,
        texttemplate="%{text}",
        textfont={"size": 11, "color": "white"},
        colorscale=[
            [0.0, "#c2685a"], [0.5, "#0c1410"], [1.0, "#5bb3a0"]
        ],
        zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="Correlación", tickfont=dict(color="#eef1ec"))
    ))
    fig.update_layout(
        **LAYOUT_BASE, title="Matriz de Correlación (Retornos Diarios)",
        xaxis=dict(tickfont=dict(color="#c9a24b")),
        yaxis=dict(tickfont=dict(color="#c9a24b"), autorange="reversed")
    )
    return fig


def fig_montecarlo(resultado_mc: dict) -> go.Figure:
    percentiles = resultado_mc["percentiles"]
    dias = list(range(resultado_mc["horizonte"] + 1))
    
    fig = go.Figure()
    
    # Banda P5-P95 (rango de confianza 90%)
    fig.add_trace(go.Scatter(
        x=dias + dias[::-1],
        y=list(percentiles["p95"]) + list(percentiles["p5"])[::-1],
        fill="toself", fillcolor="rgba(30,46,38,0.5)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Rango P5–P95 (90%)",
        hoverinfo="skip"
    ))
    
    # Banda P25-P75 (rango intercuartil)
    fig.add_trace(go.Scatter(
        x=dias + dias[::-1],
        y=list(percentiles["p75"]) + list(percentiles["p25"])[::-1],
        fill="toself", fillcolor="rgba(91,179,160,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Rango P25–P75 (50%)",
        hoverinfo="skip"
    ))

    # Línea de la mediana (P50)
    fig.add_trace(go.Scatter(
        x=dias, y=percentiles["p50"],
        mode="lines", name="Mediana (P50)",
        line=dict(color="#c9a24b", width=2.5)
    ))

    # Línea base 100
    fig.add_hline(y=100, line_dash="dash", line_color="#8aa398",
                  annotation_text="Base 100", annotation_font_color="#8aa398")

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Simulación Montecarlo ({resultado_mc['n_sim']:,} simulaciones — {resultado_mc['horizonte']} días)",
        xaxis=dict(gridcolor="#1e2e26", title="Días desde hoy"),
        yaxis=dict(gridcolor="#1e2e26", title="Valor del Portafolio (Base 100)")
    )
    return fig


def fig_distribucion_final(valores_finales: np.ndarray) -> go.Figure:
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=valores_finales,
        nbinsx=60,
        marker_color="#5bb3a0",
        opacity=0.75,
        name="Distribución de resultados"
    ))
    
    p5 = np.percentile(valores_finales, 5)
    mediana = np.median(valores_finales)
    
    fig.add_vline(x=p5, line_dash="dash", line_color="#c2685a",
                  annotation_text=f"VaR 95%: {p5:.0f}", annotation_font_color="#c2685a")
    fig.add_vline(x=mediana, line_dash="dash", line_color="#c9a24b",
                  annotation_text=f"Mediana: {mediana:.0f}", annotation_font_color="#c9a24b")
    fig.add_vline(x=100, line_dash="solid", line_color="#8aa398",
                  annotation_text="Base", annotation_font_color="#8aa398")
    
    fig.update_layout(
        **LAYOUT_BASE, title="Distribución de Valores Finales",
        xaxis=dict(gridcolor="#1e2e26", title="Valor final (Base 100)"),
        yaxis=dict(gridcolor="#1e2e26", title="Frecuencia")
    )
    return fig


def fig_frontera_eficiente(resultado_opt: dict,
                            rendimientos: pd.DataFrame,
                            pesos_usuario: np.ndarray,
                            rf: float = 0.0457) -> go.Figure:
    frontera = resultado_opt["frontera"]
    if not frontera:
        return go.Figure()

    vols_f = [p[0] for p in frontera]
    rets_f = [p[1] for p in frontera]
    sharpes_f = [p[2] for p in frontera]

    fig = go.Figure()

    # Nube de portafolios aleatorios (Monte Carlo de pesos)
    n = len(rendimientos.columns)
    n_random = 3000
    rets_r, vols_r, shr_r = [], [], []
    cov_anual = rendimientos.cov().values * 252
    medias_anuales = rendimientos.mean().values * 252
    
    for _ in range(n_random):
        w = np.random.dirichlet(np.ones(n))
        r = np.dot(w, medias_anuales) * 100
        v = np.sqrt(np.dot(w, np.dot(cov_anual, w))) * 100
        s = (r / 100 - rf) / (v / 100) if v > 0 else 0
        rets_r.append(r); vols_r.append(v); shr_r.append(s)

    fig.add_trace(go.Scatter(
        x=vols_r, y=rets_r, mode="markers",
        marker=dict(color=shr_r, colorscale="Viridis", size=3, opacity=0.4,
                    colorbar=dict(title="Sharpe", x=1.05, tickfont=dict(color="#eef1ec"))),
        name="Portafolios aleatorios", hoverinfo="skip"
    ))

    # Frontera eficiente
    fig.add_trace(go.Scatter(
        x=vols_f, y=rets_f, mode="lines",
        line=dict(color="#c9a24b", width=3),
        name="Frontera Eficiente",
        hovertemplate="Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<extra></extra>"
    ))

    # Portafolio óptimo Sharpe
    m_sharpe = calcular_metricas_portafolio(rendimientos, resultado_opt["pesos_sharpe"], rf)
    fig.add_trace(go.Scatter(
        x=[m_sharpe["volatilidad_anual"] * 100], y=[m_sharpe["retorno_anual"] * 100],
        mode="markers", marker=dict(color="#c9a24b", size=14, symbol="star",
                                    line=dict(color="white", width=1)),
        name=f"Max Sharpe ({m_sharpe['sharpe']:.2f})",
        hovertemplate=f"Max Sharpe<br>Vol: {m_sharpe['volatilidad_anual']*100:.1f}%<br>Ret: {m_sharpe['retorno_anual']*100:.1f}%<extra></extra>"
    ))

    # Portafolio del usuario
    m_user = calcular_metricas_portafolio(rendimientos, pesos_usuario, rf)
    fig.add_trace(go.Scatter(
        x=[m_user["volatilidad_anual"] * 100], y=[m_user["retorno_anual"] * 100],
        mode="markers", marker=dict(color="#5bb3a0", size=12, symbol="diamond",
                                    line=dict(color="white", width=1)),
        name=f"Tu portafolio (Sharpe: {m_user['sharpe']:.2f})",
        hovertemplate=f"Tu portafolio<br>Vol: {m_user['volatilidad_anual']*100:.1f}%<br>Ret: {m_user['retorno_anual']*100:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        **LAYOUT_BASE, title="Frontera Eficiente de Markowitz",
        xaxis=dict(gridcolor="#1e2e26", title="Volatilidad Anual (%)"),
        yaxis=dict(gridcolor="#1e2e26", title="Retorno Esperado Anual (%)")
    )
    return fig


def fig_tecnico(df_tecnico: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        subplot_titles=[f"{ticker} — Precio y Medias Móviles", "RSI (14)", "MACD"]
    )

    # Panel 1: Precio + Medias + Bollinger
    for col in ["BB_Superior", "BB_Inferior"]:
        fig.add_trace(go.Scatter(
            x=df_tecnico.index, y=df_tecnico[col],
            mode="lines", line=dict(color="#2a4a42", width=1),
            showlegend=False, fill=None
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_tecnico.index, y=df_tecnico["BB_Superior"],
        y0=df_tecnico["BB_Inferior"],
        fill="tonexty", fillcolor="rgba(91,179,160,0.06)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=True, name="Bollinger"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_tecnico.index, y=df_tecnico["precio"],
                             name="Precio", line=dict(color="#eef1ec", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_tecnico.index, y=df_tecnico["SMA20"],
                             name="SMA20", line=dict(color="#c9a24b", width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_tecnico.index, y=df_tecnico["SMA50"],
                             name="SMA50", line=dict(color="#5bb3a0", width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_tecnico.index, y=df_tecnico["SMA200"],
                             name="SMA200", line=dict(color="#9b7ec8", width=1.2, dash="dash")), row=1, col=1)

    # Panel 2: RSI
    fig.add_trace(go.Scatter(x=df_tecnico.index, y=df_tecnico["RSI"],
                             name="RSI(14)", line=dict(color="#c9a24b", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#c2685a", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#6aae66", row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(194,104,90,0.08)", row=2, col=1, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(106,174,102,0.08)", row=2, col=1, line_width=0)

    # Panel 3: MACD
    colors_hist = ["#6aae66" if v >= 0 else "#c2685a" for v in df_tecnico["MACD_Hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df_tecnico.index, y=df_tecnico["MACD_Hist"],
                         name="MACD Histograma", marker_color=colors_hist, opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_tecnico.index, y=df_tecnico["MACD"],
                             name="MACD", line=dict(color="#5bb3a0", width=1.2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_tecnico.index, y=df_tecnico["MACD_Signal"],
                             name="Signal", line=dict(color="#c9a24b", width=1.2, dash="dot")), row=3, col=1)

    fig.update_layout(
        **LAYOUT_BASE, height=700,
        xaxis=dict(gridcolor="#1e2e26"),
        xaxis2=dict(gridcolor="#1e2e26"),
        xaxis3=dict(gridcolor="#1e2e26"),
        yaxis=dict(gridcolor="#1e2e26"),
        yaxis2=dict(gridcolor="#1e2e26", range=[0, 100]),
        yaxis3=dict(gridcolor="#1e2e26")
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# INTERFAZ PRINCIPAL — SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 Configuración del Portafolio")
    st.markdown("""
    <div class='info-box'>
    <b>Formato de tickers:</b><br>
    • Acciones: <code>NVDA</code>, <code>AAPL</code>, <code>MSFT</code><br>
    • Cripto: <code>BTC-USD</code>, <code>ETH-USD</code><br>
    • ETFs bonos: <code>TLT</code>, <code>AGG</code>, <code>IEF</code><br>
    • ETFs sector: <code>SOXX</code>, <code>XLK</code>, <code>XLE</code><br>
    • Índices: <code>^GSPC</code> (S&P500), <code>^IXIC</code> (Nasdaq)
    </div>
    """, unsafe_allow_html=True)

    # Tickers del portafolio (pre-cargados con los del Taller 7)
    tickers_default = "NVDA, AMD, INTC, QCOM, AVGO, AMAT, LRCX, MU, TSM, SOXX"
    tickers_input = st.text_area(
        "Tickers del portafolio (separados por coma)",
        value=tickers_default,
        height=100,
        help="Ingresa los mismos tickers de tu Taller 7 o cualquier otro activo"
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    st.markdown("---")
    st.markdown("### ⚖️ Pesos del Portafolio")
    
    # Opción de pesos
    modo_pesos = st.radio(
        "Método de ponderación:",
        ["Igual peso (1/N)", "Pesos manuales", "Optimizado (Máx. Sharpe)"],
        help="Igual peso: distribuye equitativamente. Optimizado: busca el Sharpe máximo automáticamente."
    )

    pesos_finales = None
    if modo_pesos == "Pesos manuales" and tickers:
        cols_peso = st.columns(2)
        pesos_raw = {}
        for i, t in enumerate(tickers):
            with cols_peso[i % 2]:
                pesos_raw[t] = st.number_input(f"{t} (%)", 0.0, 100.0,
                                                value=round(100 / len(tickers), 1),
                                                step=0.5, key=f"peso_{t}")
        total_pesos = sum(pesos_raw.values())
        if abs(total_pesos - 100) > 0.5:
            st.error(f"⚠️ Suma de pesos: {total_pesos:.1f}% (debe ser 100%)")
        else:
            pesos_finales = np.array([pesos_raw[t] / 100 for t in tickers])

    st.markdown("---")
    st.markdown("### ⚙️ Parámetros de Análisis")

    periodo = st.selectbox(
        "Periodo histórico", ["1y", "2y", "3y", "5y"],
        index=1,
        help="Cuántos años de datos históricos usar para los cálculos"
    )
    rf = st.number_input(
        "Tasa libre de riesgo (% anual)",
        value=4.57, min_value=0.0, max_value=20.0, step=0.01,
        help="Bono del Tesoro USA 10 años. Actualiza según el mercado actual."
    ) / 100

    ticker_benchmark = st.selectbox(
        "Benchmark para regresión",
        ["^GSPC", "^IXIC", "SOXX", "^DJI"],
        help="S&P500 (^GSPC) es el más común. SOXX si tu portafolio es de semiconductores."
    )

    st.markdown("### 🎲 Montecarlo")
    n_simulaciones = st.select_slider(
        "Número de simulaciones",
        options=[200, 365, 500, 1000],
        value=365,
        help="Más simulaciones = más preciso pero más lento. 365 es un buen balance."
    )
    horizonte_dias = st.number_input(
        "Horizonte de proyección (días)",
        value=252, min_value=30, max_value=1260, step=30,
        help="252 = 1 año de mercado. 504 = 2 años."
    )

    st.markdown("---")
    calcular = st.button("🚀 Ejecutar Análisis Completo", type="primary")


# ─────────────────────────────────────────────────────────────────────────────
# CABECERA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

st.title("📊 Análisis de Portafolio de Inversión")
st.markdown("""
<div class='info-box teal'>
<b>Réplica Python del modelo TALLER_7_INTEGRADO_ENTREGA_FINAL.xlsb</b> — MAF EAFIT<br>
Datos en vivo de Yahoo Finance · Correlación · Covarianza · Montecarlo · Frontera Eficiente · Análisis Técnico · Regresión Estadística
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if calcular:
    if len(tickers) < 2:
        st.error("⚠️ Ingresa al menos 2 tickers para el análisis de portafolio.")
        st.stop()

    # ── DESCARGA DE DATOS ──────────────────────────────────────────────────
    with st.spinner(f"📡 Descargando datos de {', '.join(tickers)} desde Yahoo Finance..."):
        # Descargar portafolio + benchmark
        todos_tickers = list(set(tickers + [ticker_benchmark]))
        precios = descargar_precios(todos_tickers, periodo)

        if precios.empty:
            st.error("No se pudieron descargar datos. Verifica los tickers.")
            st.stop()

        # Separar benchmark del portafolio
        precios_port = precios[[t for t in tickers if t in precios.columns]].dropna()
        precios_bench = precios[ticker_benchmark] if ticker_benchmark in precios.columns else None
        tickers_validos = list(precios_port.columns)

        if len(tickers_validos) < 2:
            st.error("No hay suficientes tickers válidos. Algunos pueden no estar disponibles en Yahoo Finance.")
            st.stop()

    # ── CÁLCULOS ──────────────────────────────────────────────────────────
    with st.spinner("🧮 Calculando rendimientos, correlación, covarianza..."):
        rendimientos = calcular_rendimientos(precios_port)
        base100 = calcular_base100(precios_port)
        corr_matrix = calcular_matriz_correlacion(rendimientos)
        cov_matrix = calcular_matriz_covarianza(rendimientos)

        # Determinar pesos finales
        if modo_pesos == "Igual peso (1/N)" or pesos_finales is None:
            pesos_finales = np.array([1 / len(tickers_validos)] * len(tickers_validos))
            n_pesos = len(pesos_finales)
            if n_pesos != len(tickers_validos):
                pesos_finales = np.array([1 / len(tickers_validos)] * len(tickers_validos))

        # Recortar pesos si hay tickers inválidos
        if len(pesos_finales) != len(tickers_validos):
            pesos_finales = np.array([1 / len(tickers_validos)] * len(tickers_validos))

        metricas = calcular_metricas_portafolio(rendimientos, pesos_finales, rf)

    with st.spinner(f"🎲 Corriendo Montecarlo ({n_simulaciones} simulaciones × {horizonte_dias} días)..."):
        mc = simulacion_montecarlo(rendimientos, pesos_finales, n_simulaciones, horizonte_dias)

    with st.spinner("📐 Optimizando portafolio (Frontera de Markowitz)..."):
        try:
            resultado_opt = optimizar_portafolio(rendimientos, rf)
            optimizacion_ok = True
        except Exception as e:
            optimizacion_ok = False
            st.warning(f"Optimización omitida: {e}")

    # ── PANEL DE MÉTRICAS PRINCIPALES ────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Métricas del Portafolio")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Retorno Anual Esp.", f"{metricas['retorno_anual']*100:.2f}%",
                delta="Anualizado (×252)")
    col2.metric("Volatilidad Anual", f"{metricas['volatilidad_anual']*100:.2f}%",
                delta="Desv. Estándar")
    col3.metric("Sharpe Ratio", f"{metricas['sharpe']:.3f}",
                delta="≥1 bueno · ≥2 excelente")
    col4.metric("VaR 95% (paramétrico)", f"{metricas['var_95']*100:.2f}%",
                delta="Pérdida máx. probable")
    col5.metric("Montecarlo — Mediana", f"{mc['mediana']:.0f}",
                delta=f"Base 100 → {horizonte_dias}d")
    col6.metric("Prob. Ganancia", f"{mc['prob_ganancia']:.1f}%",
                delta=f"{n_simulaciones} simulaciones")

    # Modo de pesos activos
    if modo_pesos == "Optimizado (Máx. Sharpe)" and optimizacion_ok:
        pesos_finales = resultado_opt["pesos_sharpe"]
        st.success(f"✅ Pesos optimizados aplicados (Sharpe máximo: {metricas['sharpe']:.3f})")
        pesos_df = pd.DataFrame({
            "Ticker": tickers_validos,
            "Peso Óptimo": [f"{w*100:.1f}%" for w in pesos_finales]
        }).set_index("Ticker")
        st.dataframe(pesos_df.T, use_container_width=True)

    # ── TABS DE ANÁLISIS ─────────────────────────────────────────────────
    st.markdown("---")
    tabs = st.tabs([
        "📊 Base 100",
        "🔥 Correlación",
        "🎲 Montecarlo",
        "📐 Frontera Eficiente",
        "📉 Análisis Técnico",
        "📊 Regresión Estadística",
        "🏆 Detalle por Activo",
        "📚 Guía Educativa"
    ])

    # ── TAB 1: BASE 100 ───────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("""
        <div class='info-box'>
        <b>¿Por qué Base 100?</b> Permite comparar activos con precios muy distintos en la misma escala.
        Si NVDA vale $800 y AMD vale $120, no puedes compararlos directamente. Base 100 normaliza ambos
        al mismo punto de partida y muestra cuánto ha crecido CADA PESO invertido en cada activo.
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig_base100(base100), use_container_width=True)

        # Tabla de rendimientos totales
        ret_totales = ((precios_port.iloc[-1] / precios_port.iloc[0]) - 1) * 100
        ret_df = pd.DataFrame({
            "Activo": ret_totales.index,
            "Retorno Total (%)": ret_totales.values.round(2),
            "Precio Inicial": precios_port.iloc[0].values.round(2),
            "Precio Final": precios_port.iloc[-1].values.round(2)
        }).sort_values("Retorno Total (%)", ascending=False)
        st.dataframe(ret_df, use_container_width=True, hide_index=True)

    # ── TAB 2: CORRELACIÓN ────────────────────────────────────────────────
    with tabs[1]:
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            st.markdown("""
            <div class='info-box'>
            <b>Leyendo la matriz:</b> valores cercanos a +1 (verde) = se mueven juntos.
            Cercanos a -1 (rojo) = se mueven en direcciones opuestas (¡ideal para diversificar!).
            Si todos tus activos tienen correlación >0.8, tu portafolio NO está diversificado —
            es como tener un solo activo.
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(fig_correlacion(corr_matrix), use_container_width=True)

        with col_c2:
            st.markdown("#### Correlación Promedio")
            corr_vals = corr_matrix.values
            np.fill_diagonal(corr_vals, np.nan)
            corr_promedio = np.nanmean(corr_vals)
            
            color_corr = "🟢" if corr_promedio < 0.5 else "🟡" if corr_promedio < 0.75 else "🔴"
            st.metric(f"{color_corr} Correlación Promedio", f"{corr_promedio:.3f}")
            
            if corr_promedio < 0.5:
                st.success("✅ Portafolio bien diversificado")
            elif corr_promedio < 0.75:
                st.warning("⚠️ Diversificación moderada")
            else:
                st.error("❌ Alta concentración sectorial")

            st.markdown("#### Pares más correlacionados")
            pairs = []
            for i in range(len(tickers_validos)):
                for j in range(i + 1, len(tickers_validos)):
                    pairs.append({
                        "Par": f"{tickers_validos[i]} / {tickers_validos[j]}",
                        "ρ": round(corr_matrix.iloc[i, j], 3)
                    })
            pairs_df = pd.DataFrame(pairs).sort_values("ρ", ascending=False)
            st.dataframe(pairs_df, use_container_width=True, hide_index=True)

            # Matriz covarianza
            with st.expander("Ver Matriz de Covarianza"):
                st.dataframe(
                    (cov_matrix * 252).round(6),
                    use_container_width=True
                )
                st.caption("Covarianza anualizada (×252 días)")

    # ── TAB 3: MONTECARLO ─────────────────────────────────────────────────
    with tabs[2]:
        st.markdown(f"""
        <div class='info-box'>
        <b>Cómo leer el gráfico:</b> Cada simulación es un posible futuro del portafolio.
        La banda más oscura (P5–P95) abarca el 90% de los escenarios.
        La línea dorada es la mediana (el resultado "más probable").
        El VaR 95% = {mc['var_95']:.0f} significa que solo el 5% de los escenarios termina
        por debajo de este valor — tu "peor caso probable" a {horizonte_dias} días.
        </div>
        """, unsafe_allow_html=True)

        col_m1, col_m2 = st.columns([3, 1])
        with col_m1:
            st.plotly_chart(fig_montecarlo(mc), use_container_width=True)
        with col_m2:
            st.markdown("#### Resumen Montecarlo")
            st.metric("Mediana (P50)", f"{mc['mediana']:.1f}")
            st.metric("VaR 95% (P5)", f"{mc['var_95']:.1f}", delta="Peor caso 5%")
            st.metric("CVaR 95%", f"{mc['cvar_95']:.1f}", delta="Pérdida esperada en cola")
            st.metric("P25 – P75", f"{mc['percentiles']['p25'][-1]:.0f} – {mc['percentiles']['p75'][-1]:.0f}")
            st.metric("P95", f"{mc['percentiles']['p95'][-1]:.0f}")
            st.metric("Prob. Ganancia", f"{mc['prob_ganancia']:.1f}%")

        st.plotly_chart(fig_distribucion_final(mc["valores_finales"]), use_container_width=True)

    # ── TAB 4: FRONTERA EFICIENTE ─────────────────────────────────────────
    with tabs[3]:
        if optimizacion_ok:
            st.markdown("""
            <div class='info-box'>
            <b>La Frontera Eficiente (Markowitz, 1952):</b> cada punto es un portafolio posible.
            Los puntos en la curva superior izquierda son "eficientes" — no puedes obtener más retorno
            sin asumir más riesgo. La estrella dorada es el portafolio de <b>máximo Sharpe ratio</b>
            (el mejor retorno por unidad de riesgo). El diamante azul es TU portafolio actual.
            </div>
            """, unsafe_allow_html=True)

            col_f1, col_f2 = st.columns([3, 1])
            with col_f1:
                st.plotly_chart(
                    fig_frontera_eficiente(resultado_opt, rendimientos, pesos_finales, rf),
                    use_container_width=True
                )
            with col_f2:
                st.markdown("#### Portafolio Óptimo (Max Sharpe)")
                opt_df = pd.DataFrame({
                    "Ticker": tickers_validos,
                    "Peso Óptimo": [f"{w*100:.1f}%" for w in resultado_opt["pesos_sharpe"]]
                })
                st.dataframe(opt_df, use_container_width=True, hide_index=True)

                st.markdown("#### Portafolio Mín. Varianza")
                minvar_df = pd.DataFrame({
                    "Ticker": tickers_validos,
                    "Peso Mín. Var.": [f"{w*100:.1f}%" for w in resultado_opt["pesos_minvar"]]
                })
                st.dataframe(minvar_df, use_container_width=True, hide_index=True)
        else:
            st.warning("La optimización no fue posible con estos activos. Intenta con menos tickers o un período más largo.")

    # ── TAB 5: ANÁLISIS TÉCNICO ───────────────────────────────────────────
    with tabs[4]:
        st.markdown("""
        <div class='info-box'>
        <b>Análisis Técnico:</b> Busca patrones en los precios para predecir movimientos futuros.
        No predice el futuro con certeza — identifica probabilidades y momentum.
        Úsalo como complemento al análisis fundamental, no como sustituto.
        </div>
        """, unsafe_allow_html=True)

        ticker_tec = st.selectbox("Selecciona el activo a analizar:", tickers_validos)

        if ticker_tec and ticker_tec in precios_port.columns:
            df_tec = analisis_tecnico(precios_port[ticker_tec])
            st.plotly_chart(fig_tecnico(df_tec, ticker_tec), use_container_width=True)

            # Señales actuales
            ultimo = df_tec.dropna().iloc[-1]
            col_s1, col_s2, col_s3 = st.columns(3)
            
            rsi_val = ultimo["RSI"]
            rsi_señal = "🔴 Sobrecomprado" if rsi_val > 70 else ("🟢 Sobrevendido" if rsi_val < 30 else "🟡 Neutral")
            col_s1.metric("RSI Actual", f"{rsi_val:.1f}", delta=rsi_señal)

            macd_señal = "🟢 Alcista" if ultimo["MACD"] > ultimo["MACD_Signal"] else "🔴 Bajista"
            col_s2.metric("MACD vs Signal", f"{ultimo['MACD']:.3f}", delta=macd_señal)

            tendencia = "🟢 Alcista (>SMA50)" if ultimo["precio"] > ultimo["SMA50"] else "🔴 Bajista (<SMA50)"
            col_s3.metric("Tendencia Precio", f"${ultimo['precio']:.2f}", delta=tendencia)

    # ── TAB 6: REGRESIÓN ESTADÍSTICA ──────────────────────────────────────
    with tabs[5]:
        st.markdown("""
        <div class='info-box'>
        <b>Regresión OLS (Mínimos Cuadrados Ordinarios):</b> mide qué tan bien explica el mercado
        los movimientos de tu activo. R² alto = el activo "sigue al mercado". Alpha positivo =
        el activo genera retorno adicional incluso descontando el riesgo del mercado.
        </div>
        """, unsafe_allow_html=True)

        ticker_reg = st.selectbox("Activo a analizar:", tickers_validos, key="reg_ticker")

        if ticker_reg and ticker_reg in precios_port.columns and precios_bench is not None:
            reg = regresion_lineal(precios_port[ticker_reg], precios_bench)

            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            col_r1.metric("Beta (β)", f"{reg['beta']:.3f}",
                          delta=("Alta sensibilidad al mercado" if reg['beta'] > 1.2
                                 else "Defensivo" if reg['beta'] < 0.8 else "Moderado"))
            col_r2.metric("Alpha Anual (α)", f"{reg['alpha_anual']*100:.2f}%",
                          delta="Retorno sobre CAPM")
            col_r3.metric("R²", f"{reg['r_cuadrado']:.3f}",
                          delta=f"{reg['r_cuadrado']*100:.0f}% explicado por el mercado")
            col_r4.metric("Observaciones", f"{reg['n_obs']:,}")

            # Gráfico de dispersión
            datos_reg = reg["datos"]
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(
                x=datos_reg["benchmark"] * 100, y=datos_reg["activo"] * 100,
                mode="markers",
                marker=dict(color="#5bb3a0", size=3, opacity=0.4),
                name="Retornos diarios"
            ))
            x_line = np.linspace(datos_reg["benchmark"].min(), datos_reg["benchmark"].max(), 100)
            y_line = reg["alpha_diario"] + reg["beta"] * x_line
            fig_reg.add_trace(go.Scatter(
                x=x_line * 100, y=y_line * 100,
                mode="lines", name=f"Recta OLS (β={reg['beta']:.3f})",
                line=dict(color="#c9a24b", width=2)
            ))
            fig_reg.update_layout(
                **LAYOUT_BASE,
                title=f"Regresión: {ticker_reg} vs {ticker_benchmark}",
                xaxis=dict(gridcolor="#1e2e26", title=f"Retorno {ticker_benchmark} (%)"),
                yaxis=dict(gridcolor="#1e2e26", title=f"Retorno {ticker_reg} (%)")
            )
            st.plotly_chart(fig_reg, use_container_width=True)

            # Distribución de retornos + estadísticos
            ret_serie = datos_reg["activo"]
            col_dist1, col_dist2 = st.columns(2)
            with col_dist1:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=ret_serie * 100, nbinsx=50,
                    marker_color="#5bb3a0", opacity=0.75,
                    name=f"Retornos {ticker_reg}"
                ))
                fig_hist.update_layout(**LAYOUT_BASE,
                                       title="Distribución de Retornos Diarios",
                                       xaxis=dict(title="Retorno Diario (%)"),
                                       yaxis=dict(title="Frecuencia"))
                st.plotly_chart(fig_hist, use_container_width=True)
            with col_dist2:
                st.markdown("#### Estadísticos Descriptivos")
                stats_data = {
                    "Métrica": ["Media diaria", "Media anual", "Desv. Estándar diaria",
                                "Volatilidad anual", "Mínimo", "Máximo", "Asimetría", "Curtosis"],
                    "Valor": [
                        f"{ret_serie.mean()*100:.4f}%",
                        f"{ret_serie.mean()*252*100:.2f}%",
                        f"{ret_serie.std()*100:.4f}%",
                        f"{ret_serie.std()*np.sqrt(252)*100:.2f}%",
                        f"{ret_serie.min()*100:.2f}%",
                        f"{ret_serie.max()*100:.2f}%",
                        f"{ret_serie.skew():.3f}",
                        f"{ret_serie.kurtosis():.3f}"
                    ]
                }
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
                
                skew = ret_serie.skew()
                kurt = ret_serie.kurtosis()
                if kurt > 3:
                    st.warning(f"⚠️ Curtosis={kurt:.2f} → Colas pesadas (fat tails). El VaR normal subestima el riesgo real.")
                if skew < -0.5:
                    st.warning(f"⚠️ Asimetría={skew:.2f} → Distribución sesgada negativamente. Más probabilidad de caídas extremas.")

    # ── TAB 7: DETALLE POR ACTIVO ─────────────────────────────────────────
    with tabs[6]:
        st.markdown("### Métricas Individuales por Activo")
        
        filas = []
        for ticker_i, m in metricas["metricas_individuales"].items():
            filas.append({
                "Activo": ticker_i,
                "Peso (%)": f"{m['peso']*100:.1f}%",
                "Retorno Anual": f"{m['retorno_anual']*100:.2f}%",
                "Volatilidad Anual": f"{m['volatilidad_anual']*100:.2f}%",
                "Sharpe Individual": f"{m['sharpe']:.3f}",
                "Retorno Total Periodo": f"{m['retorno_total']*100:.2f}%"
            })
        
        tabla_df = pd.DataFrame(filas)
        st.dataframe(tabla_df, use_container_width=True, hide_index=True)

        # Gráfico de pesos actuales
        fig_pesos = go.Figure(go.Pie(
            labels=tickers_validos,
            values=[m["peso"] * 100 for m in metricas["metricas_individuales"].values()],
            marker_colors=COLORES[:len(tickers_validos)],
            textinfo="label+percent",
            hole=0.4
        ))
        fig_pesos.update_layout(**LAYOUT_BASE, title="Distribución de Pesos del Portafolio")
        st.plotly_chart(fig_pesos, use_container_width=True)

        # Retorno vs Volatilidad individual
        rets_ind = [m["retorno_anual"] * 100 for m in metricas["metricas_individuales"].values()]
        vols_ind = [m["volatilidad_anual"] * 100 for m in metricas["metricas_individuales"].values()]
        fig_rv = go.Figure()
        for i, (ticker_i, m) in enumerate(metricas["metricas_individuales"].items()):
            fig_rv.add_trace(go.Scatter(
                x=[m["volatilidad_anual"] * 100], y=[m["retorno_anual"] * 100],
                mode="markers+text", text=[ticker_i],
                textposition="top center",
                marker=dict(color=COLORES[i % len(COLORES)], size=12),
                name=ticker_i
            ))
        fig_rv.update_layout(**LAYOUT_BASE, title="Retorno vs Volatilidad por Activo",
                             xaxis=dict(gridcolor="#1e2e26", title="Volatilidad Anual (%)"),
                             yaxis=dict(gridcolor="#1e2e26", title="Retorno Anual (%)"))
        st.plotly_chart(fig_rv, use_container_width=True)

    # ── TAB 8: GUÍA EDUCATIVA ─────────────────────────────────────────────
    with tabs[7]:
        st.markdown("## 📚 Guía Educativa — Cómo leer cada resultado")

        conceptos = {
            "Retorno Logarítmico": {
                "formula": "r_t = ln(P_t / P_{t-1})",
                "explicacion": "El estándar académico en finanzas cuantitativas. A diferencia del retorno aritmético, los retornos log son aditivos en el tiempo: puedes sumar retornos diarios para obtener el retorno del período. También se distribuyen aproximadamente normal, lo que permite usar modelos estadísticos.",
                "ejemplo": "Si NVDA sube de $800 a $840: ln(840/800) = 4.88% (retorno log) vs 5.00% (aritmético). La diferencia parece pequeña en un día, pero se acumula significativamente en el tiempo."
            },
            "Sharpe Ratio": {
                "formula": "Sharpe = (R_p - R_f) / σ_p",
                "explicacion": "Mide cuánto retorno obtienes por cada unidad de riesgo. Si Sharpe = 1.5 significa que por cada 1% de riesgo (volatilidad) que asumes, obtienes 1.5% de retorno por encima de la tasa libre de riesgo. Sharpe > 1 = bueno, > 2 = excelente, < 0 = peor que no hacer nada.",
                "ejemplo": "Portafolio A: retorno 12%, vol 8% → Sharpe = (12-4.57)/8 = 0.93. Portafolio B: retorno 18%, vol 20% → Sharpe = (18-4.57)/20 = 0.67. A es mejor ajustado por riesgo aunque B tiene más retorno absoluto."
            },
            "VaR (Value at Risk)": {
                "formula": "VaR_95% = -(μ - 1.645σ) × √T",
                "explicacion": "La pérdida máxima probable en un período dado con un nivel de confianza dado. VaR 95% a 1 año = 15% significa que el 95% del tiempo no perderás más del 15% anual. El 5% restante puede ser peor — ahí entra el CVaR.",
                "ejemplo": "Si tu Montecarlo muestra P5 = 85 (base 100), el VaR 95% es 15% de caída en el horizonte analizado. Solo 5 de cada 100 trayectorias terminan por debajo de 85."
            },
            "Beta (β)": {
                "formula": "β = Cov(R_i, R_m) / Var(R_m) = SLOPE(R_activo, R_mercado)",
                "explicacion": "Mide la sensibilidad del activo a los movimientos del mercado. Es la PENDIENTE de la recta de regresión entre los retornos del activo y el benchmark. β=1.5 en NVDA significa que cuando el S&P500 sube 1%, NVDA tiende a subir 1.5% — y cuando cae 1%, NVDA cae 1.5%.",
                "ejemplo": "Semiconductores tipicamente tienen β > 1.2 (activos cíclicos, amplificadores del mercado). Utilities o bonos suelen tener β < 0.5 (defensivos)."
            },
            "Descomposición de Cholesky": {
                "formula": "Σ = L × L'  →  shocks_correlacionados = L × z",
                "explicacion": "El corazón matemático de tu Montecarlo multivariado. Permite generar escenarios donde NVDA y AMD se mueven juntos (correlación 0.85) en lugar de independientemente. Sin esto, subestimarías el riesgo del portafolio porque ignorarías que cuando uno cae, el otro también tiende a caer.",
                "ejemplo": "Si z = [1.2, -0.8] (shocks independientes) y L captura que AMD y NVDA correlacionan 0.85, el resultado L×z te da shocks que respetan esa correlación: [1.2, 0.22] — AMD no cae tanto porque NVDA subió."
            }
        }

        for titulo, datos in conceptos.items():
            with st.expander(f"📐 {titulo}"):
                col_e1, col_e2 = st.columns([1, 2])
                with col_e1:
                    st.code(datos["formula"], language="text")
                with col_e2:
                    st.markdown(f"**Explicación:** {datos['explicacion']}")
                    st.info(f"💡 **Ejemplo práctico:** {datos['ejemplo']}")

        st.markdown("""
        ---
        ### 🎓 Recursos para profundizar

        | Tipo | Recurso | Tema |
        |------|---------|------|
        | 📺 YouTube | **Ben Felix** | Teoría de portafolios basada en evidencia |
        | 📺 YouTube | **Damodaran Online (NYU)** | Valoración y CAPM (curso completo gratis) |
        | 📺 YouTube | **QuantPy** | Black-Scholes y opciones en Python |
        | 🎙️ Podcast | **Flirting with Models** (Corey Hoffstein) | Quant finance avanzado |
        | 🎙️ Podcast | **Invest Like the Best** | Gestión de portafolios real |
        | 📚 Libro | **Hull — Options, Futures & Other Derivatives** | La biblia de derivados |
        | 📚 Libro | **Damodaran — Investment Valuation** | Gratis en damodaran.com |
        | 🔧 Herramienta | **Portfolio Visualizer** (portfoliovisualizer.com) | Análisis sin código |
        | 🔧 Herramienta | **Thinkorswim** (TD Ameritrade) | Greeks y opciones en tiempo real |
        | 📊 Curso | **Coursera — Financial Engineering** (Columbia) | Derivados con Python |
        """)

else:
    # ── PANTALLA DE BIENVENIDA ────────────────────────────────────────────
    st.markdown("""
    <div class='info-box teal'>
    👈 <b>Configura tu portafolio en la barra lateral y presiona "Ejecutar Análisis Completo"</b><br>
    Los tickers del Taller 7 (sector semiconductores) ya vienen precargados como ejemplo.
    </div>
    """, unsafe_allow_html=True)

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        st.markdown("""
        ### 📊 Análisis de Portafolio
        - Base 100 (AP_ind_Base100)
        - Rendimientos logarítmicos (AP_Rendimientos)
        - Matriz de correlación (AP_Correlacion)
        - Matriz de covarianza (AP_Covarianza)
        - Sharpe, VaR, CVaR (AP_analisis)
        """)
    with col_b2:
        st.markdown("""
        ### 🎲 Montecarlo
        - Cholesky para shocks correlacionados
        - 200 a 1000 simulaciones
        - Percentiles P5, P25, P50, P75, P95
        - Distribución de valores finales
        - Probabilidad de ganancia
        """)
    with col_b3:
        st.markdown("""
        ### 📐 Optimización (nuevo vs Taller 7)
        - Frontera Eficiente de Markowitz
        - Portafolio de Máximo Sharpe
        - Portafolio de Mínima Varianza
        - Análisis Técnico (RSI, MACD, Bollinger)
        - Regresión OLS (Beta, Alpha, R²)
        """)

    st.markdown("---")
    st.markdown("""
    ### ⚡ Stack Tecnológico
    `Python 3.11` · `Streamlit` · `yfinance` · `pandas` · `numpy` · `plotly` · `scipy`
    
    **Datos:** Yahoo Finance (en vivo, actualizado automáticamente en cada ejecución)  
    **Deploy:** Streamlit Community Cloud (gratuito) — [streamlit.io/cloud](https://streamlit.io/cloud)
    """)

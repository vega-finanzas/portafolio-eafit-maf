# =============================================================================
# ANÁLISIS DE PORTAFOLIO DE INVERSIÓN — Taller 7 EAFIT MAF
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

st.set_page_config(
    page_title="Análisis de Portafolio — MAF EAFIT",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0c1410; color: #eef1ec; }
    [data-testid="stSidebar"] { background-color: #0f1a16; border-right: 1px solid #1e2e26; }
    [data-testid="stMetric"] { background-color: #121d18; border: 1px solid #1e2e26; border-radius: 10px; padding: 12px 16px; }
    [data-testid="stMetricValue"] { color: #c9a24b; font-size: 1.6rem !important; }
    [data-testid="stMetricLabel"] { color: #8aa398; font-size: 0.75rem; }
    h1 { color: #c9a24b !important; font-family: Georgia, serif; }
    h2 { color: #eef1ec !important; }
    h3 { color: #5bb3a0 !important; }
    .stTabs [data-baseweb="tab"] { color: #8aa398; background-color: #121d18; }
    .stTabs [aria-selected="true"] { color: #c9a24b !important; border-bottom-color: #c9a24b !important; }
    .stButton > button { background-color: #c9a24b; color: #1a1408; font-weight: 700; border: none; border-radius: 8px; padding: 10px 24px; width: 100%; }
    .stButton > button:hover { background-color: #e0b85e; }
    .info-box { background-color: #121d18; border: 1px solid #1e2e26; border-left: 3px solid #c9a24b; border-radius: 0 8px 8px 0; padding: 14px 16px; margin: 10px 0; font-size: 0.9rem; color: #8aa398; line-height: 1.7; }
    .info-box b { color: #eef1ec; }
    .info-box.teal { border-left-color: #5bb3a0; }
</style>
""", unsafe_allow_html=True)

COLORES = ["#c9a24b","#5bb3a0","#c2685a","#7d9bc1","#9b7ec8","#8fb574","#d49a6a","#e87d9b","#4db8d4","#a0b86e"]
LAYOUT_BASE = dict(
    paper_bgcolor="#0c1410", plot_bgcolor="#0c1410",
    font=dict(color="#eef1ec", family="Inter, Arial, sans-serif"),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="#121d18", bordercolor="#1e2e26", borderwidth=1)
)
# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES CORE
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def descargar_precios(tickers, periodo):
    try:
        data = yf.download(tickers, period=periodo, auto_adjust=True, progress=False, threads=True)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        return data.ffill().bfill().dropna()
    except Exception as e:
        st.error(f"Error descargando datos: {e}")
        return pd.DataFrame()

def calcular_rendimientos(precios):
    return np.log(precios / precios.shift(1)).dropna()

def calcular_base100(precios):
    return (precios / precios.iloc[0]) * 100

def calcular_metricas_portafolio(rendimientos, pesos, rf=0.0457):
    ret_diarios = rendimientos.mean().values
    ret_port_diario = np.dot(pesos, ret_diarios)
    cov_matrix = rendimientos.cov().values
    var_port_diario = np.dot(pesos, np.dot(cov_matrix, pesos))
    vol_port_diario = np.sqrt(max(var_port_diario, 0))
    ret_anual = ret_port_diario * 252
    vol_anual = vol_port_diario * np.sqrt(252)
    sharpe = (ret_anual - rf) / vol_anual if vol_anual > 0 else 0.0
    metricas_ind = {}
    for i, ticker in enumerate(rendimientos.columns):
        ret_ind = rendimientos[ticker].mean() * 252
        vol_ind = rendimientos[ticker].std() * np.sqrt(252)
        sharpe_ind = (ret_ind - rf) / vol_ind if vol_ind > 0 else 0
        metricas_ind[ticker] = {
            "peso": pesos[i], "retorno_anual": ret_ind,
            "volatilidad_anual": vol_ind, "sharpe": sharpe_ind,
            "retorno_total": (rendimientos[ticker] + 1).prod() - 1
        }
    return {
        "retorno_anual": ret_anual, "volatilidad_anual": vol_anual,
        "sharpe": sharpe, "var_95": -(ret_port_diario - 1.645 * vol_port_diario) * np.sqrt(252),
        "metricas_individuales": metricas_ind
    }

def simulacion_montecarlo(rendimientos, pesos, n_sim=1000, horizonte=252):
    n_activos = len(rendimientos.columns)
    medias = rendimientos.mean().values
    cov_matrix = rendimientos.cov().values
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        cov_matrix += np.eye(n_activos) * 1e-8
        L = np.linalg.cholesky(cov_matrix)
    trayectorias = np.zeros((n_sim, horizonte + 1))
    trayectorias[:, 0] = 100
    for sim in range(n_sim):
        valor = 100.0
        for dia in range(horizonte):
            z = np.random.standard_normal(n_activos)
            shocks = L @ z
            retornos_diarios = medias + shocks
            retorno_port = np.dot(pesos, retornos_diarios)
            valor = valor * (1 + retorno_port)
            trayectorias[sim, dia + 1] = valor
    percentiles = {}
    for p in [5, 10, 25, 50, 75, 90, 95]:
        percentiles[f"p{p}"] = np.percentile(trayectorias, p, axis=0)
    valores_finales = trayectorias[:, -1]
    return {
        "percentiles": percentiles,
        "valores_finales": valores_finales,
        "var_95": np.percentile(valores_finales, 5),
        "cvar_95": valores_finales[valores_finales <= np.percentile(valores_finales, 5)].mean(),
        "mediana": np.median(valores_finales),
        "prob_ganancia": (valores_finales > 100).mean() * 100,
        "n_sim": n_sim, "horizonte": horizonte
    }

def optimizar_portafolio(rendimientos, rf=0.0457):
    n = len(rendimientos.columns)
    medias = rendimientos.mean().values * 252
    cov_anual = rendimientos.cov().values * 252
    def neg_sharpe(pesos):
        ret = np.dot(pesos, medias)
        vol = np.sqrt(np.dot(pesos, np.dot(cov_anual, pesos)))
        return -(ret - rf) / vol if vol > 0 else 0
    def vol_portafolio(pesos):
        return np.sqrt(np.dot(pesos, np.dot(cov_anual, pesos)))
    def ret_portafolio(pesos):
        return np.dot(pesos, medias)
    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = tuple((0.01, 0.60) for _ in range(n))
    w0 = np.array([1 / n] * n)
    res_sharpe = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=restricciones)
    res_minvar = minimize(vol_portafolio, w0, method="SLSQP", bounds=bounds, constraints=restricciones)
    pesos_frontera = []
    targets = np.linspace(ret_portafolio(res_minvar.x), max(medias), 40)
    for target in targets:
        res = minimize(vol_portafolio, w0, method="SLSQP", bounds=bounds,
                       constraints=[{"type":"eq","fun":lambda w: np.sum(w)-1},
                                    {"type":"eq","fun":lambda w,t=target: ret_portafolio(w)-t}])
        if res.success:
            pesos_frontera.append(res.x)
    frontera = [(vol_portafolio(w)*100, ret_portafolio(w)*100,
                 (ret_portafolio(w)-rf)/vol_portafolio(w)) for w in pesos_frontera]
    return {"pesos_sharpe": res_sharpe.x, "pesos_minvar": res_minvar.x,
            "frontera": frontera, "tickers": list(rendimientos.columns)}

def analisis_tecnico(precios_serie):
    df = pd.DataFrame({"precio": precios_serie})
    for n in [20, 50, 200]:
        df[f"SMA{n}"] = df["precio"].rolling(window=n).mean()
    for n in [12, 26]:
        df[f"EMA{n}"] = df["precio"].ewm(span=n, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    delta = df["precio"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["BB_Media"] = df["precio"].rolling(window=20).mean()
    df["BB_Std"] = df["precio"].rolling(window=20).std()
    df["BB_Superior"] = df["BB_Media"] + 2 * df["BB_Std"]
    df["BB_Inferior"] = df["BB_Media"] - 2 * df["BB_Std"]
    return df

def regresion_lineal(precios_activo, precios_benchmark):
    ret_activo = np.log(precios_activo / precios_activo.shift(1)).dropna()
    ret_benchmark = np.log(precios_benchmark / precios_benchmark.shift(1)).dropna()
    df = pd.DataFrame({"activo": ret_activo, "benchmark": ret_benchmark}).dropna()
    slope, intercept, r_value, p_value, std_err = stats.linregress(df["benchmark"], df["activo"])
    return {
        "beta": slope, "alpha_diario": intercept,
        "alpha_anual": intercept * 252, "r_cuadrado": r_value ** 2,
        "p_value": p_value, "datos": df, "n_obs": len(df)
    }
# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────

def fig_base100(base100):
    fig = go.Figure()
    for i, col in enumerate(base100.columns):
        fig.add_trace(go.Scatter(x=base100.index, y=base100[col], name=col, mode="lines",
            line=dict(color=COLORES[i % len(COLORES)], width=2),
            hovertemplate=f"<b>{col}</b><br>Fecha: %{{x|%d-%b-%Y}}<br>Base 100: %{{y:.1f}}<extra></extra>"))
    fig.update_layout(**LAYOUT_BASE, title="Evolución Indexada (Base 100)",
        xaxis=dict(gridcolor="#1e2e26", title="Fecha"),
        yaxis=dict(gridcolor="#1e2e26", title="Valor (Base 100)"))
    return fig

def fig_correlacion(corr):
    tickers = corr.columns.tolist()
    z = corr.values
    texto = [[f"{z[i][j]:.2f}" for j in range(len(tickers))] for i in range(len(tickers))]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=tickers, y=tickers, text=texto, texttemplate="%{text}",
        textfont={"size": 11, "color": "white"},
        colorscale=[[0.0,"#c2685a"],[0.5,"#0c1410"],[1.0,"#5bb3a0"]],
        zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="Correlación", tickfont=dict(color="#eef1ec"))))
    fig.update_layout(**LAYOUT_BASE, title="Matriz de Correlación",
        xaxis=dict(tickfont=dict(color="#c9a24b")),
        yaxis=dict(tickfont=dict(color="#c9a24b"), autorange="reversed"))
    return fig

def fig_montecarlo(resultado_mc):
    percentiles = resultado_mc["percentiles"]
    dias = list(range(resultado_mc["horizonte"] + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dias + dias[::-1],
        y=list(percentiles["p95"]) + list(percentiles["p5"])[::-1],
        fill="toself", fillcolor="rgba(30,46,38,0.5)",
        line=dict(color="rgba(0,0,0,0)"), name="Rango P5-P95 (90%)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=dias + dias[::-1],
        y=list(percentiles["p75"]) + list(percentiles["p25"])[::-1],
        fill="toself", fillcolor="rgba(91,179,160,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="Rango P25-P75 (50%)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=dias, y=percentiles["p50"], mode="lines",
        name="Mediana (P50)", line=dict(color="#c9a24b", width=2.5)))
    fig.add_hline(y=100, line_dash="dash", line_color="#8aa398",
        annotation_text="Base 100", annotation_font_color="#8aa398")
    fig.update_layout(**LAYOUT_BASE,
        title=f"Simulación Montecarlo ({resultado_mc['n_sim']:,} sim — {resultado_mc['horizonte']} días)",
        xaxis=dict(gridcolor="#1e2e26", title="Días desde hoy"),
        yaxis=dict(gridcolor="#1e2e26", title="Valor del Portafolio (Base 100)"))
    return fig

def fig_distribucion_final(valores_finales):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=valores_finales, nbinsx=60,
        marker_color="#5bb3a0", opacity=0.75, name="Distribución"))
    p5 = np.percentile(valores_finales, 5)
    mediana = np.median(valores_finales)
    fig.add_vline(x=p5, line_dash="dash", line_color="#c2685a",
        annotation_text=f"VaR 95%: {p5:.0f}", annotation_font_color="#c2685a")
    fig.add_vline(x=mediana, line_dash="dash", line_color="#c9a24b",
        annotation_text=f"Mediana: {mediana:.0f}", annotation_font_color="#c9a24b")
    fig.add_vline(x=100, line_dash="solid", line_color="#8aa398",
        annotation_text="Base", annotation_font_color="#8aa398")
    fig.update_layout(**LAYOUT_BASE, title="Distribución de Valores Finales",
        xaxis=dict(gridcolor="#1e2e26", title="Valor final (Base 100)"),
        yaxis=dict(gridcolor="#1e2e26", title="Frecuencia"))
    return fig

def fig_frontera(resultado_opt, rendimientos, pesos_usuario, rf=0.0457):
    frontera = resultado_opt["frontera"]
    if not frontera:
        return go.Figure()
    vols_f = [p[0] for p in frontera]
    rets_f = [p[1] for p in frontera]
    fig = go.Figure()
    n = len(rendimientos.columns)
    cov_anual = rendimientos.cov().values * 252
    medias_anuales = rendimientos.mean().values * 252
    rets_r, vols_r, shr_r = [], [], []
    for _ in range(2000):
        w = np.random.dirichlet(np.ones(n))
        r = np.dot(w, medias_anuales) * 100
        v = np.sqrt(np.dot(w, np.dot(cov_anual, w))) * 100
        s = (r/100 - rf) / (v/100) if v > 0 else 0
        rets_r.append(r); vols_r.append(v); shr_r.append(s)
    fig.add_trace(go.Scatter(x=vols_r, y=rets_r, mode="markers",
        marker=dict(color=shr_r, colorscale="Viridis", size=3, opacity=0.4,
        colorbar=dict(title="Sharpe", x=1.05, tickfont=dict(color="#eef1ec"))),
        name="Portafolios aleatorios", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=vols_f, y=rets_f, mode="lines",
        line=dict(color="#c9a24b", width=3), name="Frontera Eficiente"))
    m_sharpe = calcular_metricas_portafolio(rendimientos, resultado_opt["pesos_sharpe"], rf)
    fig.add_trace(go.Scatter(
        x=[m_sharpe["volatilidad_anual"]*100], y=[m_sharpe["retorno_anual"]*100],
        mode="markers", marker=dict(color="#c9a24b", size=14, symbol="star",
        line=dict(color="white", width=1)), name=f"Max Sharpe ({m_sharpe['sharpe']:.2f})"))
    m_user = calcular_metricas_portafolio(rendimientos, pesos_usuario, rf)
    fig.add_trace(go.Scatter(
        x=[m_user["volatilidad_anual"]*100], y=[m_user["retorno_anual"]*100],
        mode="markers", marker=dict(color="#5bb3a0", size=12, symbol="diamond",
        line=dict(color="white", width=1)), name=f"Tu portafolio (Sharpe: {m_user['sharpe']:.2f})"))
    fig.update_layout(**LAYOUT_BASE, title="Frontera Eficiente de Markowitz",
        xaxis=dict(gridcolor="#1e2e26", title="Volatilidad Anual (%)"),
        yaxis=dict(gridcolor="#1e2e26", title="Retorno Esperado Anual (%)"))
    return fig

def fig_tecnico(df_tecnico, ticker):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20], vertical_spacing=0.03,
        subplot_titles=[f"{ticker} — Precio y Medias Móviles", "RSI (14)", "MACD"])
    fig.add_trace(go.Scatter(x=df_tecnico.index, y=df_tecnico["precio"],
        name="Precio", line=dict(color="#eef1ec", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_tecnico.index, y=df_tecnico["SMA20"],
        name="SMA20", line=dict(color="#c9a24b", width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_tecnico.index, y=df_tecnico["SMA50"],
        name="SMA50", line=dict(color="#5bb3a0", width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_tecnico.index, y=df_tecnico["SMA200"],
        name="SMA200", line=dict(color="#9b7ec8", width=1.2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_tecnico.index, y=df_tecnico["BB_Superior"],
        line=dict(color="#2a4a42", width=1), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_tecnico.index, y=df_tecnico["BB_Inferior"],
        line=dict(color="#2a4a42", width=1), fill="tonexty",
        fillcolor="rgba(91,179,160,0.06)", name="Bollinger"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_tecnico.index, y=df_tecnico["RSI"],
        name="RSI(14)", line=dict(color="#c9a24b", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#c2685a", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#6aae66", row=2, col=1)
    colors_hist = ["#6aae66" if v >= 0 else "#c2685a" for v in df_tecnico["MACD_Hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df_tecnico.index, y=df_tecnico["MACD_Hist"],
        name="Histograma", marker_color=colors_hist, opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_tecnico.index, y=df_tecnico["MACD"],
        name="MACD", line=dict(color="#5bb3a0", width=1.2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_tecnico.index, y=df_tecnico["MACD_Signal"],
        name="Signal", line=dict(color="#c9a24b", width=1.2, dash="dot")), row=3, col=1)
    fig.update_layout(**LAYOUT_BASE, height=700,
        xaxis=dict(gridcolor="#1e2e26"), xaxis2=dict(gridcolor="#1e2e26"),
        xaxis3=dict(gridcolor="#1e2e26"), yaxis=dict(gridcolor="#1e2e26"),
        yaxis2=dict(gridcolor="#1e2e26", range=[0,100]), yaxis3=dict(gridcolor="#1e2e26"))
    return fig
# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 Configuración del Portafolio")
    st.markdown("""
    <div class='info-box'>
    <b>Formato de tickers:</b><br>
    • Acciones: <code>NVDA</code>, <code>AAPL</code><br>
    • Cripto: <code>BTC-USD</code>, <code>ETH-USD</code><br>
    • ETFs: <code>TLT</code>, <code>SOXX</code>, <code>AGG</code><br>
    • Índices: <code>^GSPC</code> (S&P500)
    </div>
    """, unsafe_allow_html=True)

    tickers_input = st.text_area(
        "Tickers del portafolio (separados por coma)",
        value="NVDA, AMD, INTC, QCOM, AVGO, AMAT, LRCX, MU, TSM, SOXX",
        height=100)
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    st.markdown("---")
    st.markdown("### ⚖️ Pesos")
    modo_pesos = st.radio("Método de ponderación:",
        ["Igual peso (1/N)", "Pesos manuales", "Optimizado (Máx. Sharpe)"])

    pesos_finales = None
    if modo_pesos == "Pesos manuales" and tickers:
        cols_peso = st.columns(2)
        pesos_raw = {}
        for i, t in enumerate(tickers):
            with cols_peso[i % 2]:
                pesos_raw[t] = st.number_input(f"{t} (%)", 0.0, 100.0,
                    value=round(100/len(tickers), 1), step=0.5, key=f"peso_{t}")
        total = sum(pesos_raw.values())
        if abs(total - 100) > 0.5:
            st.error(f"⚠️ Suma: {total:.1f}% (debe ser 100%)")
        else:
            pesos_finales = np.array([pesos_raw[t]/100 for t in tickers])

    st.markdown("---")
    st.markdown("### ⚙️ Parámetros")
    periodo = st.selectbox("Periodo histórico", ["1y","2y","3y","5y"], index=1)
    rf = st.number_input("Tasa libre de riesgo (%)", value=4.57, min_value=0.0,
        max_value=20.0, step=0.01) / 100
    ticker_benchmark = st.selectbox("Benchmark", ["^GSPC","^IXIC","SOXX","^DJI"])

    st.markdown("### 🎲 Montecarlo")
    n_simulaciones = st.select_slider("Simulaciones", options=[200,365,500,1000], value=365)
    horizonte_dias = st.number_input("Horizonte (días)", value=252, min_value=30,
        max_value=1260, step=30)

    st.markdown("---")
    calcular = st.button("🚀 Ejecutar Análisis Completo", type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# CABECERA
# ─────────────────────────────────────────────────────────────────────────────

st.title("📊 Análisis de Portafolio de Inversión")
st.markdown("""
<div class='info-box teal'>
<b>Réplica Python del modelo TALLER_7_INTEGRADO_ENTREGA_FINAL.xlsb — MAF EAFIT</b><br>
Datos en vivo · Correlación · Covarianza · Montecarlo · Frontera Eficiente · Análisis Técnico
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if calcular:
    if len(tickers) < 2:
        st.error("⚠️ Ingresa al menos 2 tickers.")
        st.stop()

    with st.spinner(f"📡 Descargando datos de {', '.join(tickers)}..."):
        todos = list(set(tickers + [ticker_benchmark]))
        precios = descargar_precios(todos, periodo)
        if precios.empty:
            st.error("No se pudieron descargar datos.")
            st.stop()
        precios_port = precios[[t for t in tickers if t in precios.columns]].dropna()
        precios_bench = precios[ticker_benchmark] if ticker_benchmark in precios.columns else None
        tickers_validos = list(precios_port.columns)
        if len(tickers_validos) < 2:
            st.error("No hay suficientes tickers válidos.")
            st.stop()

    with st.spinner("🧮 Calculando métricas..."):
        rendimientos = calcular_rendimientos(precios_port)
        base100 = calcular_base100(precios_port)
        corr_matrix = rendimientos.corr()

        if pesos_finales is None or len(pesos_finales) != len(tickers_validos):
            pesos_finales = np.array([1/len(tickers_validos)]*len(tickers_validos))

        metricas = calcular_metricas_portafolio(rendimientos, pesos_finales, rf)

    with st.spinner(f"🎲 Corriendo Montecarlo ({n_simulaciones} simulaciones)..."):
        mc = simulacion_montecarlo(rendimientos, pesos_finales, n_simulaciones, horizonte_dias)

    with st.spinner("📐 Optimizando portafolio..."):
        try:
            resultado_opt = optimizar_portafolio(rendimientos, rf)
            opt_ok = True
        except:
            opt_ok = False

    if modo_pesos == "Optimizado (Máx. Sharpe)" and opt_ok:
        pesos_finales = resultado_opt["pesos_sharpe"]
        metricas = calcular_metricas_portafolio(rendimientos, pesos_finales, rf)

    # MÉTRICAS PRINCIPALES
    st.markdown("---")
    st.markdown("### 📈 Métricas del Portafolio")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Retorno Anual", f"{metricas['retorno_anual']*100:.2f}%")
    c2.metric("Volatilidad Anual", f"{metricas['volatilidad_anual']*100:.2f}%")
    c3.metric("Sharpe Ratio", f"{metricas['sharpe']:.3f}")
    c4.metric("VaR 95%", f"{metricas['var_95']*100:.2f}%")
    c5.metric("Montecarlo Mediana", f"{mc['mediana']:.0f}")
    c6.metric("Prob. Ganancia", f"{mc['prob_ganancia']:.1f}%")

    # TABS
    st.markdown("---")
    tabs = st.tabs(["📊 Base 100","🔥 Correlación","🎲 Montecarlo",
                    "📐 Frontera Eficiente","📉 Análisis Técnico",
                    "📊 Regresión","🏆 Detalle Activos"])

    with tabs[0]:
        st.plotly_chart(fig_base100(base100), use_container_width=True)
        ret_tot = ((precios_port.iloc[-1]/precios_port.iloc[0])-1)*100
        df_ret = pd.DataFrame({"Activo":ret_tot.index,
            "Retorno Total (%)":ret_tot.values.round(2),
            "Precio Inicial":precios_port.iloc[0].values.round(2),
            "Precio Final":precios_port.iloc[-1].values.round(2)
        }).sort_values("Retorno Total (%)", ascending=False)
        st.dataframe(df_ret, use_container_width=True, hide_index=True)

    with tabs[1]:
        col1, col2 = st.columns([2,1])
        with col1:
            st.plotly_chart(fig_correlacion(corr_matrix), use_container_width=True)
        with col2:
            vals = corr_matrix.values.copy()
            np.fill_diagonal(vals, np.nan)
            prom = np.nanmean(vals)
            icono = "🟢" if prom < 0.5 else "🟡" if prom < 0.75 else "🔴"
            st.metric(f"{icono} Correlación Promedio", f"{prom:.3f}")
            if prom < 0.5: st.success("✅ Bien diversificado")
            elif prom < 0.75: st.warning("⚠️ Diversificación moderada")
            else: st.error("❌ Alta concentración sectorial")
            pairs = []
            for i in range(len(tickers_validos)):
                for j in range(i+1, len(tickers_validos)):
                    pairs.append({"Par":f"{tickers_validos[i]}/{tickers_validos[j]}",
                                  "ρ":round(corr_matrix.iloc[i,j],3)})
            st.dataframe(pd.DataFrame(pairs).sort_values("ρ", ascending=False),
                use_container_width=True, hide_index=True)

    with tabs[2]:
        c1, c2 = st.columns([3,1])
        with c1:
            st.plotly_chart(fig_montecarlo(mc), use_container_width=True)
        with c2:
            st.metric("Mediana (P50)", f"{mc['mediana']:.1f}")
            st.metric("VaR 95% (P5)", f"{mc['var_95']:.1f}")
            st.metric("CVaR 95%", f"{mc['cvar_95']:.1f}")
            st.metric("Prob. Ganancia", f"{mc['prob_ganancia']:.1f}%")
        st.plotly_chart(fig_distribucion_final(mc["valores_finales"]), use_container_width=True)

    with tabs[3]:
        if opt_ok:
            c1, c2 = st.columns([3,1])
            with c1:
                st.plotly_chart(fig_frontera(resultado_opt, rendimientos, pesos_finales, rf),
                    use_container_width=True)
            with c2:
                st.markdown("#### Pesos Óptimos (Max Sharpe)")
                df_opt = pd.DataFrame({"Ticker":tickers_validos,
                    "Peso":[ f"{w*100:.1f}%" for w in resultado_opt["pesos_sharpe"]]})
                st.dataframe(df_opt, use_container_width=True, hide_index=True)
                st.markdown("#### Mínima Varianza")
                df_mv = pd.DataFrame({"Ticker":tickers_validos,
                    "Peso":[f"{w*100:.1f}%" for w in resultado_opt["pesos_minvar"]]})
                st.dataframe(df_mv, use_container_width=True, hide_index=True)
        else:
            st.warning("Optimización no disponible. Intenta con menos tickers o periodo más largo.")

    with tabs[4]:
        ticker_tec = st.selectbox("Activo a analizar:", tickers_validos)
        if ticker_tec in precios_port.columns:
            df_tec = analisis_tecnico(precios_port[ticker_tec])
            st.plotly_chart(fig_tecnico(df_tec, ticker_tec), use_container_width=True)
            ultimo = df_tec.dropna().iloc[-1]
            c1,c2,c3 = st.columns(3)
            rsi = ultimo["RSI"]
            c1.metric("RSI", f"{rsi:.1f}",
                delta="🔴 Sobrecomprado" if rsi>70 else "🟢 Sobrevendido" if rsi<30 else "🟡 Neutral")
            macd_s = "🟢 Alcista" if ultimo["MACD"] > ultimo["MACD_Signal"] else "🔴 Bajista"
            c2.metric("MACD", f"{ultimo['MACD']:.3f}", delta=macd_s)
            tend = "🟢 >SMA50" if ultimo["precio"] > ultimo["SMA50"] else "🔴 <SMA50"
            c3.metric("Precio", f"${ultimo['precio']:.2f}", delta=tend)

    with tabs[5]:
        ticker_reg = st.selectbox("Activo:", tickers_validos, key="reg")
        if ticker_reg in precios_port.columns and precios_bench is not None:
            reg = regresion_lineal(precios_port[ticker_reg], precios_bench)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Beta (β)", f"{reg['beta']:.3f}")
            c2.metric("Alpha Anual", f"{reg['alpha_anual']*100:.2f}%")
            c3.metric("R²", f"{reg['r_cuadrado']:.3f}")
            c4.metric("Observaciones", f"{reg['n_obs']:,}")
            datos = reg["datos"]
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(x=datos["benchmark"]*100, y=datos["activo"]*100,
                mode="markers", marker=dict(color="#5bb3a0", size=3, opacity=0.4), name="Datos"))
            x_l = np.linspace(datos["benchmark"].min(), datos["benchmark"].max(), 100)
            y_l = reg["alpha_diario"] + reg["beta"] * x_l
            fig_reg.add_trace(go.Scatter(x=x_l*100, y=y_l*100, mode="lines",
                name=f"OLS β={reg['beta']:.3f}", line=dict(color="#c9a24b", width=2)))
            fig_reg.update_layout(**LAYOUT_BASE,
                title=f"Regresión: {ticker_reg} vs {ticker_benchmark}",
                xaxis=dict(gridcolor="#1e2e26", title=f"Retorno {ticker_benchmark} (%)"),
                yaxis=dict(gridcolor="#1e2e26", title=f"Retorno {ticker_reg} (%)"))
            st.plotly_chart(fig_reg, use_container_width=True)

    with tabs[6]:
        filas = []
        for t, m in metricas["metricas_individuales"].items():
            filas.append({"Activo":t, "Peso":f"{m['peso']*100:.1f}%",
                "Retorno Anual":f"{m['retorno_anual']*100:.2f}%",
                "Volatilidad":f"{m['volatilidad_anual']*100:.2f}%",
                "Sharpe":f"{m['sharpe']:.3f}",
                "Retorno Total":f"{m['retorno_total']*100:.2f}%"})
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
        fig_pie = go.Figure(go.Pie(
            labels=tickers_validos,
            values=[m["peso"]*100 for m in metricas["metricas_individuales"].values()],
            marker_colors=COLORES[:len(tickers_validos)],
            textinfo="label+percent", hole=0.4))
        fig_pie.update_layout(**LAYOUT_BASE, title="Distribución de Pesos")
        st.plotly_chart(fig_pie, use_container_width=True)

else:
    st.markdown("""
    <div class='info-box teal'>
    👈 <b>Configura tu portafolio en la barra lateral y presiona "Ejecutar Análisis Completo"</b><br>
    Los tickers del Taller 7 (semiconductores) ya vienen precargados como ejemplo.
    </div>
    """, unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("### 📊 Portafolio\n- Base 100\n- Rendimientos log\n- Correlación\n- Covarianza\n- Sharpe / VaR")
    with c2:
        st.markdown("### 🎲 Montecarlo\n- Cholesky correlacionado\n- 200–1000 simulaciones\n- P5 / P50 / P95\n- CVaR / Prob. ganancia")
    with c3:
        st.markdown("### 📐 Optimización\n- Frontera Eficiente\n- Máx. Sharpe\n- Mín. Varianza\n- RSI / MACD / Beta")

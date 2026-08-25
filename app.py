# -*- coding: utf-8 -*-
"""
Trade Marketing AI Suite
=========================
Prototipo en Streamlit que ilustra un framework "Sense -> Predict -> Act"
para resolver las problematicas mas grandes de trade marketing usando IA:

  1) Ejecucion en punto de venta   -> Vision por computadora (deteccion de quiebres / share of shelf)
  2) ROI de promociones            -> Simulador + modelo ML de incrementalidad vs canibalizacion
  3) Estrategia de canales         -> Segmentacion de tiendas con clustering
  4) Forecast de demanda           -> Prediccion de ventas por canal
  5) Negociacion con retailers     -> Simulador de escenarios "que pasa si"
  6) Silos entre areas             -> Dashboard de alertas cross-funcionales

Todos los datos son SINTETICOS (generados con numpy) para que la app corra
de inmediato sin depender de fuentes externas. Reemplaza los generadores
`get_*_data()` por tus propias fuentes (POS, ERP, TPM, CRM) cuando lo
conectes a datos reales.

Correr con:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ----------------------------------------------------------------------------
# Config general
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Trade Marketing AI Suite",
    page_icon="📊",
    layout="wide",
)

RNG = np.random.default_rng(42)


# ----------------------------------------------------------------------------
# Utilidades de datos sinteticos (sustituir por conexiones reales)
# ----------------------------------------------------------------------------
@st.cache_data
def get_promo_training_data(n=400):
    """Historico sintetico de promociones para entrenar el modelo de uplift."""
    discount = RNG.uniform(0, 40, n)          # % de descuento
    duration = RNG.integers(3, 21, n)         # dias de duracion
    display = RNG.integers(0, 2, n)           # 1 = exhibicion adicional
    season_index = RNG.uniform(0.8, 1.3, n)   # estacionalidad de la categoria
    noise = RNG.normal(0, 4, n)

    # relacion no lineal simulada: rendimientos decrecientes despues de ~25% desc.
    incremental_units = (
        2.2 * discount
        - 0.02 * discount ** 2
        + 1.3 * duration
        + 18 * display
        + 15 * (season_index - 1) * 100
        + noise
    )
    incremental_units = np.clip(incremental_units, 0, None)

    return pd.DataFrame(
        {
            "descuento_pct": discount,
            "duracion_dias": duration,
            "exhibicion_extra": display,
            "indice_estacional": season_index,
            "unidades_incrementales": incremental_units,
        }
    )


# Mapa canal -> cadenas reales del mercado mexicano, con rangos de comportamiento
# TIPICOS de cada tipo de canal (orden de magnitud ilustrativo, no cifras
# confidenciales ni oficiales de ninguna cadena). Ajusta libremente si tienes
# benchmarks propios.
CANAL_CONFIG = {
    "Autoservicio": {
        "cadenas": ["Walmart", "Soriana", "Chedraui", "HEB", "La Comer"],
        "peso": 0.28,
        "volumen_shape_scale": (2.6, 6000),   # gamma(shape, scale)
        "margen": (20, 5),                     # normal(mu, sigma)
        "distancia_cd_km": (10, 150),
        "rotacion_dias": (7, 3),
        "quiebre_rate": (0.06, 0.03),
    },
    "Bodega/Descuento": {
        "cadenas": ["Bodega Aurrera", "Waldo's"],
        "peso": 0.12,
        "volumen_shape_scale": (2.3, 5000),
        "margen": (14, 4),
        "distancia_cd_km": (10, 200),
        "rotacion_dias": (6, 3),
        "quiebre_rate": (0.07, 0.04),
    },
    "Conveniencia": {
        "cadenas": ["OXXO", "7-Eleven", "Circle K", "Extra"],
        "peso": 0.2,
        "volumen_shape_scale": (2.0, 1200),
        "margen": (32, 6),
        "distancia_cd_km": (5, 80),
        "rotacion_dias": (3, 2),
        "quiebre_rate": (0.05, 0.03),
    },
    "Club de precio": {
        "cadenas": ["Sam's Club", "Costco"],
        "peso": 0.05,
        "volumen_shape_scale": (3.2, 9000),
        "margen": (11, 3),
        "distancia_cd_km": (15, 250),
        "rotacion_dias": (10, 4),
        "quiebre_rate": (0.04, 0.02),
    },
    "Tradicional": {
        "cadenas": ["Abarrotes independientes"],
        "peso": 0.25,
        "volumen_shape_scale": (1.6, 700),
        "margen": (26, 8),
        "distancia_cd_km": (5, 400),
        "rotacion_dias": (14, 6),
        "quiebre_rate": (0.18, 0.08),
    },
    "E-commerce": {
        "cadenas": ["Walmart.com.mx", "Mercado Libre", "Amazon MX"],
        "peso": 0.1,
        "volumen_shape_scale": (2.4, 4000),
        "margen": (18, 5),
        "distancia_cd_km": (20, 300),
        "rotacion_dias": (2, 1),
        "quiebre_rate": (0.05, 0.03),
    },
}


# Presets ILUSTRATIVOS por cadena para el simulador de negociación: reflejan
# patrones generales y de conocimiento público sobre cómo suele negociar cada
# tipo de canal (autoservicio vs. conveniencia vs. club de precio), NO cifras
# reales ni confidenciales de ninguna cadena. Ajusta con tus propios términos
# reales de negociación cuando los tengas.
RETAILER_PRESETS = {
    "Walmart":                 {"descuento": 22, "fee": 20000, "display": True,  "uplift_display": 0.15},
    "Soriana":                 {"descuento": 20, "fee": 25000, "display": True,  "uplift_display": 0.14},
    "Chedraui":                {"descuento": 18, "fee": 20000, "display": True,  "uplift_display": 0.13},
    "HEB":                     {"descuento": 15, "fee": 15000, "display": False, "uplift_display": 0.10},
    "La Comer":                {"descuento": 12, "fee": 10000, "display": False, "uplift_display": 0.10},
    "Bodega Aurrera":          {"descuento": 25, "fee": 10000, "display": True,  "uplift_display": 0.16},
    "Waldo's":                 {"descuento": 25, "fee": 8000,  "display": True,  "uplift_display": 0.15},
    "OXXO":                    {"descuento": 10, "fee": 60000, "display": True,  "uplift_display": 0.20},
    "7-Eleven":                {"descuento": 10, "fee": 40000, "display": True,  "uplift_display": 0.18},
    "Circle K":                {"descuento": 10, "fee": 30000, "display": True,  "uplift_display": 0.16},
    "Extra":                   {"descuento": 10, "fee": 25000, "display": True,  "uplift_display": 0.15},
    "Sam's Club":              {"descuento": 15, "fee": 15000, "display": False, "uplift_display": 0.08},
    "Costco":                  {"descuento": 12, "fee": 10000, "display": False, "uplift_display": 0.08},
    "Abarrotes independientes": {"descuento": 5,  "fee": 0,     "display": False, "uplift_display": 0.05},
    "Walmart.com.mx":          {"descuento": 15, "fee": 5000,  "display": True,  "uplift_display": 0.12},
    "Mercado Libre":           {"descuento": 15, "fee": 5000,  "display": True,  "uplift_display": 0.12},
    "Amazon MX":               {"descuento": 15, "fee": 5000,  "display": True,  "uplift_display": 0.12},
}


@st.cache_data
def get_store_data(n=150):
    """
    Universo de tiendas con nombres de cadenas reales del mercado mexicano.
    Los VALORES (volumen, margen, quiebre, etc.) siguen siendo sintéticos:
    representan órdenes de magnitud típicos de cada tipo de canal, no datos
    confidenciales de ninguna empresa en particular.
    """
    canales = list(CANAL_CONFIG.keys())
    pesos = [CANAL_CONFIG[c]["peso"] for c in canales]
    pesos = np.array(pesos) / sum(pesos)
    canal_asignado = RNG.choice(canales, size=n, p=pesos)

    cadena, volumen_promedio, margen_pct = [], [], []
    distancia_cd_km, rotacion_dias, quiebre_rate = [], [], []

    for c in canal_asignado:
        cfg = CANAL_CONFIG[c]
        cadena.append(RNG.choice(cfg["cadenas"]))
        shape, scale = cfg["volumen_shape_scale"]
        volumen_promedio.append(RNG.gamma(shape=shape, scale=scale))
        mu, sigma = cfg["margen"]
        margen_pct.append(np.clip(RNG.normal(mu, sigma), 5, 50))
        lo, hi = cfg["distancia_cd_km"]
        distancia_cd_km.append(RNG.uniform(lo, hi))
        mu_r, sigma_r = cfg["rotacion_dias"]
        rotacion_dias.append(np.clip(RNG.normal(mu_r, sigma_r), 1, 45))
        mu_q, sigma_q = cfg["quiebre_rate"]
        quiebre_rate.append(np.clip(RNG.normal(mu_q, sigma_q), 0, 0.5))

    return pd.DataFrame(
        {
            "tienda_id": [f"T-{i:04d}" for i in range(n)],
            "canal": canal_asignado,
            "cadena": cadena,
            "volumen_promedio": volumen_promedio,
            "margen_pct": margen_pct,
            "distancia_cd_km": distancia_cd_km,
            "rotacion_dias": rotacion_dias,
            "quiebre_rate": quiebre_rate,
        }
    )


@st.cache_data
def get_demand_series(days=730):
    """Serie de tiempo sintetica de ventas diarias por canal."""
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    t = np.arange(days)
    trend = 500 + t * 0.35
    weekly = 60 * np.sin(2 * np.pi * t / 7)
    yearly = 150 * np.sin(2 * np.pi * t / 365 + 1.2)
    noise = RNG.normal(0, 25, days)
    sales = np.clip(trend + weekly + yearly + noise, 0, None)
    return pd.Series(sales, index=idx, name="ventas")


@st.cache_data
def get_alerts_data():
    return pd.DataFrame(
        [
            {"area": "Ventas", "kpi": "Cumplimiento de plan mensual", "actual": 91, "meta": 95, "unidad": "%"},
            {"area": "Ventas", "kpi": "Quiebres de stock en Walmart y Soriana", "actual": 9.4, "meta": 5, "unidad": "%"},
            {"area": "Trade Marketing", "kpi": "ROI de promociones activas", "actual": 1.6, "meta": 2.0, "unidad": "x"},
            {"area": "Trade Marketing", "kpi": "Cumplimiento de planograma en OXXO", "actual": 78, "meta": 90, "unidad": "%"},
            {"area": "Marketing", "kpi": "Share of shelf vs. objetivo (Chedraui)", "actual": 24, "meta": 30, "unidad": "%"},
            {"area": "Marketing", "kpi": "Inversión en retail media utilizada", "actual": 61, "meta": 85, "unidad": "%"},
        ]
    )


def synthetic_shelf_image(gap_ratio=0.25, width=640, height=360):
    """Genera una imagen de anaquel simulada (rectangulos = productos, hueco = quiebre)."""
    img = Image.new("RGB", (width, height), (235, 230, 220))  # fondo tipo anaquel
    draw = ImageDraw.Draw(img)
    n_slots = 10
    slot_w = width // n_slots
    colors = [(200, 60, 60), (60, 120, 200), (60, 170, 90), (230, 180, 40)]
    gap_slots = RNG.choice(n_slots, size=max(1, int(n_slots * gap_ratio)), replace=False)
    for i in range(n_slots):
        x0, x1 = i * slot_w + 6, (i + 1) * slot_w - 6
        if i in gap_slots:
            continue  # deja el hueco (fondo visible = quiebre)
        color = colors[i % len(colors)]
        draw.rectangle([x0, 40, x1, height - 30], fill=color)
    return img


# ----------------------------------------------------------------------------
# Modelos "IA" ligeros (entrenados en caliente sobre datos sinteticos/reales)
# ----------------------------------------------------------------------------
def train_uplift_model(df):
    X = df[["descuento_pct", "duracion_dias", "exhibicion_extra", "indice_estacional"]]
    y = df["unidades_incrementales"]
    model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    model.fit(X, y)
    return model


def analyze_shelf_image(pil_image, bg_color=(235, 230, 220), tolerance=18):
    """
    Heuristica de vision por computadora (prototipo, no un modelo entrenado):
    estima el % de la imagen que corresponde al color de fondo del anaquel
    (proxy de espacio vacio / quiebre) y el share of shelf por color de marca.
    """
    arr = np.array(pil_image.convert("RGB")).astype(int)
    bg = np.array(bg_color)
    dist = np.sqrt(((arr - bg) ** 2).sum(axis=-1))
    is_bg = dist < tolerance
    gap_pct = is_bg.mean() * 100

    # Agrupa pixeles de "producto" (no fondo) por color dominante (K-Means simple)
    product_pixels = arr[~is_bg]
    share_of_shelf = {}
    if len(product_pixels) > 0:
        sample = product_pixels[RNG.choice(len(product_pixels), size=min(3000, len(product_pixels)), replace=False)]
        k = min(4, max(1, len(np.unique(sample, axis=0))))
        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(sample)
        labels, counts = np.unique(km.labels_, return_counts=True)
        total = counts.sum()
        for lab, cnt in zip(labels, counts):
            color = tuple(km.cluster_centers_[lab].astype(int))
            share_of_shelf[color] = round(cnt / total * 100, 1)

    return gap_pct, share_of_shelf


# ----------------------------------------------------------------------------
# Carga de dataset real (reemplaza los generadores sinteticos si se sube un
# archivo). Acepta el mismo .xlsx que genera `generar_dataset.py` (hojas
# Tiendas, Historico_Promociones, Ventas_Diarias, Alertas_KPI), o cualquier
# archivo con nombres de hoja/columnas equivalentes.
# ----------------------------------------------------------------------------
def _first_matching_col(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None


def normalize_tiendas(df):
    d = df.copy()
    col_vol = _first_matching_col(d, "volumen_promedio", "volumen_promedio_mensual_unidades", "volumen")
    col_quiebre = _first_matching_col(d, "quiebre_rate", "quiebre_rate_pct", "quiebre_pct")
    if col_vol and col_vol != "volumen_promedio":
        d["volumen_promedio"] = d[col_vol]
    if col_quiebre:
        vals = pd.to_numeric(d[col_quiebre], errors="coerce")
        # si viene en escala 0-100 (porcentaje) lo pasamos a fraccion 0-1
        d["quiebre_rate"] = vals / 100.0 if vals.max() > 1.5 else vals
    if "cadena" not in d.columns and "canal" in d.columns:
        d["cadena"] = d["canal"]
    required = ["tienda_id", "canal", "cadena", "volumen_promedio", "margen_pct", "distancia_cd_km", "rotacion_dias", "quiebre_rate"]
    faltantes = [c for c in required if c not in d.columns]
    return d, faltantes


def normalize_promos(df):
    d = df.copy()
    if "exhibicion_extra" in d.columns and not pd.api.types.is_numeric_dtype(d["exhibicion_extra"]):
        d["exhibicion_extra"] = (
            d["exhibicion_extra"].astype(str).str.strip().str.lower()
            .map({"si": 1, "sí": 1, "yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0})
            .fillna(0).astype(int)
        )
    required = ["descuento_pct", "duracion_dias", "exhibicion_extra", "indice_estacional", "unidades_incrementales"]
    faltantes = [c for c in required if c not in d.columns]
    return d, faltantes


def normalize_ventas(df):
    d = df.copy()
    col_fecha = _first_matching_col(d, "fecha", "date")
    col_ventas = _first_matching_col(d, "ventas_unidades", "ventas", "unidades", "sales")
    if col_fecha is None or col_ventas is None:
        faltantes = [n for n, c in [("fecha", col_fecha), ("ventas_unidades", col_ventas)] if c is None]
        return d, faltantes
    d = d.rename(columns={col_fecha: "fecha", col_ventas: "ventas_unidades"})
    col_canal = _first_matching_col(d, "canal", "channel")
    if col_canal and col_canal != "canal":
        d = d.rename(columns={col_canal: "canal"})
    if "canal" not in d.columns:
        d["canal"] = "Total"
    d["fecha"] = pd.to_datetime(d["fecha"])
    d["ventas_unidades"] = pd.to_numeric(d["ventas_unidades"], errors="coerce")
    return d, []


def normalize_alertas(df):
    d = df.copy()
    required = ["area", "kpi", "actual", "meta"]
    faltantes = [c for c in required if c not in d.columns]
    if "unidad" not in d.columns:
        d["unidad"] = "%"
    return d, faltantes


def get_effective_store_data():
    return st.session_state["df_tiendas"] if "df_tiendas" in st.session_state else get_store_data()


def get_effective_promo_data():
    return st.session_state["df_promos"] if "df_promos" in st.session_state else get_promo_training_data()


def get_effective_alerts_data():
    return st.session_state["df_alertas"] if "df_alertas" in st.session_state else get_alerts_data()


def get_effective_demand_series(canal_filter=None):
    if "df_ventas" in st.session_state:
        d = st.session_state["df_ventas"]
        if canal_filter and canal_filter != "Todos":
            d = d[d["canal"] == canal_filter]
        s = d.groupby("fecha")["ventas_unidades"].sum().sort_index()
        s = s.asfreq("D").interpolate().bfill().ffill()
        s.name = "ventas"
        return s
    return get_demand_series()


# ----------------------------------------------------------------------------
# UI: Sidebar / navegacion
# ----------------------------------------------------------------------------
st.sidebar.title("📊 Trade Marketing AI Suite")
st.sidebar.caption("Framework Sense → Predict → Act")

st.sidebar.divider()
st.sidebar.subheader("📂 Tus datos")
uploaded_dataset = st.sidebar.file_uploader(
    "Cargar dataset (.xlsx)",
    type=["xlsx"],
    help=(
        "Sube el archivo trade_marketing_dataset.xlsx que generamos (o uno propio con "
        "hojas Tiendas, Historico_Promociones, Ventas_Diarias, Alertas_KPI) para que la "
        "app use esos datos en vez de los sintéticos."
    ),
)

if uploaded_dataset is not None:
    try:
        hojas = pd.read_excel(uploaded_dataset, sheet_name=None)
        cargadas = []

        hoja = next((s for s in hojas if s.strip().lower().startswith("tienda")), None)
        if hoja:
            d, faltantes = normalize_tiendas(hojas[hoja])
            if not faltantes:
                st.session_state["df_tiendas"] = d
                cargadas.append("Tiendas")
            else:
                st.sidebar.warning(f"Hoja '{hoja}': faltan columnas {faltantes}")

        hoja = next((s for s in hojas if "promo" in s.strip().lower()), None)
        if hoja:
            d, faltantes = normalize_promos(hojas[hoja])
            if not faltantes:
                st.session_state["df_promos"] = d
                cargadas.append("Historico_Promociones")
            else:
                st.sidebar.warning(f"Hoja '{hoja}': faltan columnas {faltantes}")

        hoja = next((s for s in hojas if "venta" in s.strip().lower()), None)
        if hoja:
            d, faltantes = normalize_ventas(hojas[hoja])
            if not faltantes:
                st.session_state["df_ventas"] = d
                cargadas.append("Ventas_Diarias")
            else:
                st.sidebar.warning(f"Hoja '{hoja}': faltan columnas {faltantes}")

        hoja = next((s for s in hojas if "alerta" in s.strip().lower() or "kpi" in s.strip().lower()), None)
        if hoja:
            d, faltantes = normalize_alertas(hojas[hoja])
            if not faltantes:
                st.session_state["df_alertas"] = d
                cargadas.append("Alertas_KPI")
            else:
                st.sidebar.warning(f"Hoja '{hoja}': faltan columnas {faltantes}")

        if cargadas:
            st.sidebar.success(f"✅ Cargado: {', '.join(cargadas)}")
        else:
            st.sidebar.error("No reconocí hojas compatibles en el archivo.")
    except Exception as e:
        st.sidebar.error(f"No pude leer el archivo: {e}")
else:
    for k in ["df_tiendas", "df_promos", "df_ventas", "df_alertas"]:
        st.session_state.pop(k, None)

st.sidebar.caption("**Estado de los datos por módulo:**")
_estado = {
    "Tiendas / Segmentación": "df_tiendas",
    "Promociones / ROI": "df_promos",
    "Ventas / Forecast": "df_ventas",
    "Alertas KPI": "df_alertas",
}
for _label, _key in _estado.items():
    _marca = "🟢 Tu dataset" if _key in st.session_state else "⚪ Sintético"
    st.sidebar.caption(f"{_marca} — {_label}")

st.sidebar.divider()

page = st.sidebar.radio(
    "Modulo",
    [
        "🏠 Resumen del framework",
        "📷 Ejecución en PDV (Computer Vision)",
        "💰 ROI de Promociones (ML)",
        "🗺️ Segmentación de Tiendas/Canales",
        "📈 Forecast de Demanda",
        "🤝 Simulador de Negociación",
        "🚨 Alertas Cross-funcionales",
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    "⚠️ Esta demo usa **nombres reales de cadenas mexicanas** (Walmart, Soriana, "
    "OXXO, etc.) para que se sienta cercana a tu mercado. Sin un dataset cargado, los "
    "**valores** (volumen, margen, quiebre, fees) son sintéticos — órdenes de magnitud "
    "típicos por tipo de canal, no cifras confidenciales reales."
)

# ----------------------------------------------------------------------------
# PAGINA: Resumen
# ----------------------------------------------------------------------------
if page == "🏠 Resumen del framework":
    st.title("Framework de soluciones de IA para Trade Marketing")
    st.markdown(
        """
Este prototipo traduce las problemáticas más comunes de trade marketing en
soluciones concretas basadas en IA, organizadas en tres capas:

**Sense** — capturar la realidad del punto de venta y del mercado en tiempo real.
**Predict** — anticipar el resultado de decisiones antes de tomarlas.
**Act** — recomendar o automatizar la acción óptima.
        """
    )

    cols = st.columns(3)
    items = [
        ("📷 Ejecución en PDV", "Visión por computadora sobre fotos de anaquel: quiebres de stock, share of shelf, cumplimiento de planograma.", "Sense"),
        ("💰 ROI de promociones", "Modelo de ML que separa incrementalidad de canibalización y recomienda el descuento óptimo.", "Predict"),
        ("🗺️ Canales y retailers", "Clustering de tiendas para priorizar inversión de trade por comportamiento real, no solo tamaño.", "Predict"),
        ("📈 Forecast de demanda", "Predicción de ventas por canal para planear inventario y negociar mejor.", "Predict"),
        ("🤝 Negociación", "Simulador \"qué pasa si\" para argumentar condiciones comerciales con datos.", "Act"),
        ("🚨 Alertas cross-funcionales", "Un solo tablero de KPIs para ventas, trade y marketing, evitando silos.", "Act"),
    ]
    for (title, desc, layer), col in zip(items, cols * 2):
        with col:
            st.subheader(title)
            st.caption(f"Capa: {layer}")
            st.write(desc)

    st.info(
        "Usa el menú de la izquierda para explorar cada módulo de forma interactiva.",
        icon="👈",
    )

# ----------------------------------------------------------------------------
# PAGINA: Vision por computadora en PDV
# ----------------------------------------------------------------------------
elif page == "📷 Ejecución en PDV (Computer Vision)":
    st.title("Ejecución en punto de venta con visión por computadora")
    st.write(
        "Sube una foto de anaquel (o usa la imagen de ejemplo) para estimar "
        "quiebres de stock y share of shelf por color de producto. "
        "Esta es una heurística de demostración; en producción se reemplaza "
        "por un modelo de detección de objetos entrenado con fotos reales del anaquel."
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        uploaded = st.file_uploader("Foto de anaquel", type=["png", "jpg", "jpeg"])
        gap_demo = st.slider("Quiebre simulado en imagen de ejemplo (%)", 0, 60, 25, 5)
        use_demo = st.button("Generar imagen de ejemplo")

    if uploaded is not None:
        image = Image.open(uploaded)
    elif use_demo or "demo_img" not in st.session_state:
        image = synthetic_shelf_image(gap_ratio=gap_demo / 100)
        st.session_state["demo_img"] = image
    else:
        image = st.session_state["demo_img"]

    with col_b:
        st.image(image, caption="Imagen analizada", width='stretch')

    gap_pct, shares = analyze_shelf_image(image)

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Quiebre estimado (espacio vacío en anaquel)", f"{gap_pct:.1f}%")
    status = "🔴 Crítico" if gap_pct > 20 else ("🟡 Atención" if gap_pct > 8 else "🟢 OK")
    m2.metric("Estatus de cumplimiento", status)

    if shares:
        st.subheader("Share of shelf estimado por color dominante")
        share_df = pd.DataFrame(
            {"color_rgb": [str(c) for c in shares.keys()], "participación_%": list(shares.values())}
        ).sort_values("participación_%", ascending=False)
        fig = px.bar(share_df, x="color_rgb", y="participación_%", color="color_rgb")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

    st.caption(
        "En una implementación real: (1) la foto se sube desde la app del promotor/vendedor, "
        "(2) un modelo de detección de objetos (p. ej. YOLO afinado con SKUs propios) identifica "
        "cada producto y su posición, (3) se compara contra el planograma acordado y (4) se dispara "
        "una alerta automática al supervisor o al equipo de reposición si hay incumplimiento."
    )

# ----------------------------------------------------------------------------
# PAGINA: ROI de promociones
# ----------------------------------------------------------------------------
elif page == "💰 ROI de Promociones (ML)":
    st.title("Simulador de ROI de promociones")
    st.write(
        "Un modelo de Random Forest, entrenado sobre histórico de promociones, "
        "estima las unidades incrementales esperadas para una combinación de "
        "condiciones. Ajusta los controles para simular un escenario."
    )

    if "df_promos" in st.session_state:
        st.caption("🟢 Modelo entrenado con **tu dataset** de promociones cargado.")
    model = train_uplift_model(get_effective_promo_data())

    c1, c2, c3, c4 = st.columns(4)
    descuento = c1.slider("Descuento (%)", 0, 40, 15)
    duracion = c2.slider("Duración (días)", 3, 21, 10)
    exhibicion = c3.selectbox("Exhibición adicional", ["No", "Sí"]) == "Sí"
    estacionalidad = c4.slider("Índice estacional de la categoría", 0.8, 1.3, 1.0, 0.05)

    precio_regular = st.number_input("Precio regular (MXN)", value=45.0, step=1.0)
    costo_unitario = st.number_input("Costo unitario (MXN)", value=28.0, step=1.0)
    volumen_base = st.number_input("Volumen base sin promoción (unidades)", value=1000, step=50)
    canibalizacion_pct = st.slider(
        "% de la incrementalidad que en realidad es pull-forward / canibalización",
        0, 80, 30,
        help="Parte de las ventas 'incrementales' que hubieran ocurrido de todas formas (compra anticipada, sustitución entre SKUs propios).",
    )

    X_input = pd.DataFrame(
        [[descuento, duracion, int(exhibicion), estacionalidad]],
        columns=["descuento_pct", "duracion_dias", "exhibicion_extra", "indice_estacional"],
    )
    unidades_incrementales_bruto = model.predict(X_input)[0]
    unidades_incrementales_neto = unidades_incrementales_bruto * (1 - canibalizacion_pct / 100)

    precio_promo = precio_regular * (1 - descuento / 100)
    margen_unitario_promo = precio_promo - costo_unitario
    margen_unitario_regular = precio_regular - costo_unitario

    volumen_total_promo = volumen_base + unidades_incrementales_neto
    margen_promo = volumen_total_promo * margen_unitario_promo
    margen_baseline = volumen_base * margen_unitario_regular
    delta_margen = margen_promo - margen_baseline
    inversion_promo = volumen_base * (precio_regular - precio_promo)  # costo del descuento sobre venta base
    roi = delta_margen / inversion_promo if inversion_promo > 0 else np.nan

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Unidades incrementales netas", f"{unidades_incrementales_neto:,.0f}")
    m2.metric("Margen incremental", f"${delta_margen:,.0f} MXN")
    m3.metric("Inversión en descuento", f"${inversion_promo:,.0f} MXN")
    m4.metric("ROI de la promoción", f"{roi:.2f}x" if not np.isnan(roi) else "N/A")

    st.subheader("Curva de optimización: margen incremental por nivel de descuento")
    rango_descuento = np.linspace(0, 40, 41)
    resultados = []
    for d in rango_descuento:
        X_d = pd.DataFrame(
            [[d, duracion, int(exhibicion), estacionalidad]],
            columns=["descuento_pct", "duracion_dias", "exhibicion_extra", "indice_estacional"],
        )
        u_bruto = model.predict(X_d)[0]
        u_neto = u_bruto * (1 - canibalizacion_pct / 100)
        p_promo = precio_regular * (1 - d / 100)
        m_unit = p_promo - costo_unitario
        vol_total = volumen_base + u_neto
        margen_total = vol_total * m_unit
        delta = margen_total - margen_baseline
        resultados.append(delta)

    curve_df = pd.DataFrame({"descuento_pct": rango_descuento, "margen_incremental": resultados})
    optimo = curve_df.loc[curve_df["margen_incremental"].idxmax()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve_df["descuento_pct"], y=curve_df["margen_incremental"], mode="lines", name="Margen incremental"))
    fig.add_trace(
        go.Scatter(
            x=[optimo["descuento_pct"]], y=[optimo["margen_incremental"]],
            mode="markers+text", text=["Óptimo"], textposition="top center",
            marker=dict(size=12, color="green"), name="Descuento óptimo",
        )
    )
    fig.update_layout(xaxis_title="Descuento (%)", yaxis_title="Margen incremental (MXN)")
    st.plotly_chart(fig, width='stretch')

    st.success(
        f"Con la duración, exhibición y estacionalidad seleccionadas, el descuento que "
        f"maximiza el margen incremental es **{optimo['descuento_pct']:.0f}%**, "
        f"vs. el {descuento}% simulado arriba."
    )

    with st.expander("Importancia de variables del modelo"):
        importances = pd.DataFrame(
            {"variable": X_input.columns, "importancia": model.feature_importances_}
        ).sort_values("importancia", ascending=False)
        st.bar_chart(importances.set_index("variable"))

# ----------------------------------------------------------------------------
# PAGINA: Segmentacion de tiendas
# ----------------------------------------------------------------------------
elif page == "🗺️ Segmentación de Tiendas/Canales":
    st.title("Segmentación de tiendas para priorizar inversión de trade")
    st.write(
        "Clustering (K-Means) sobre variables de comportamiento real "
        "(volumen, margen, rotación, quiebres, distancia al CD) para agrupar "
        "tiendas en perfiles estratégicos, en vez de priorizar solo por tamaño."
    )

    df = get_effective_store_data()
    if "df_tiendas" in st.session_state:
        st.caption("🟢 Usando **tu dataset** de tiendas cargado.")
    n_clusters = st.slider("Número de clusters", 2, 6, 4)

    features = ["volumen_promedio", "margen_pct", "distancia_cd_km", "rotacion_dias", "quiebre_rate"]
    X = StandardScaler().fit_transform(df[features])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(X)
    df["cluster"] = km.labels_.astype(str)

    fig = px.scatter(
        df, x="volumen_promedio", y="margen_pct", color="cluster",
        symbol="canal", hover_data=["tienda_id", "canal", "cadena", "quiebre_rate", "rotacion_dias"],
        title="Tiendas por volumen y margen, coloreadas por cluster",
    )
    st.plotly_chart(fig, width='stretch')

    st.subheader("Perfil promedio por cluster")
    perfil = df.groupby("cluster")[features].mean().round(1)
    perfil["n_tiendas"] = df.groupby("cluster").size()
    st.dataframe(perfil, width='stretch')

    with st.expander("Ver detalle de tiendas y cadenas por cluster"):
        st.dataframe(
            df[["tienda_id", "canal", "cadena", "cluster", "volumen_promedio", "margen_pct", "quiebre_rate"]]
            .sort_values(["cluster", "canal"]),
            width='stretch',
            hide_index=True,
        )
        st.caption(
            "Las cadenas mostradas son nombres reales de retailers mexicanos usados como "
            "referencia de canal; los valores de volumen, margen y quiebre son sintéticos "
            "(órdenes de magnitud típicos por tipo de canal, no cifras confidenciales de ninguna empresa)."
        )

    st.subheader("Recomendación de acción por cluster")
    for c in sorted(df["cluster"].unique()):
        row = perfil.loc[c]
        if row["margen_pct"] >= perfil["margen_pct"].median() and row["volumen_promedio"] >= perfil["volumen_promedio"].median():
            rec = "🌟 Tiendas ancla: proteger con inversión premium en exhibición y disponibilidad garantizada."
        elif row["quiebre_rate"] >= perfil["quiebre_rate"].median():
            rec = "🚨 Alto quiebre: priorizar reabasto y frecuencia de visita antes de invertir en promoción."
        elif row["volumen_promedio"] < perfil["volumen_promedio"].median():
            rec = "🔍 Bajo volumen: evaluar costo-beneficio de trade dedicado vs. estrategia de canal masiva."
        else:
            rec = "⚖️ Perfil balanceado: mantener inversión estándar y monitorear evolución trimestral."
        st.markdown(f"**Cluster {c}** ({int(row['n_tiendas'])} tiendas): {rec}")

# ----------------------------------------------------------------------------
# PAGINA: Forecast de demanda
# ----------------------------------------------------------------------------
elif page == "📈 Forecast de Demanda":
    st.title("Forecast de demanda por canal")
    st.write(
        "Modelo de suavizamiento exponencial (Holt-Winters) sobre la serie "
        "histórica de ventas, con estacionalidad semanal y anual, para anticipar "
        "necesidades de inventario y argumentar negociaciones de trade con datos."
    )

    canal_sel = None
    if "df_ventas" in st.session_state:
        st.caption("🟢 Usando **tu dataset** de ventas cargado.")
        canales_disponibles = ["Todos"] + sorted(st.session_state["df_ventas"]["canal"].dropna().unique().tolist())
        canal_sel = st.selectbox("Canal", canales_disponibles)

    serie = get_effective_demand_series(canal_sel)
    horizonte = st.slider("Días a pronosticar", 30, 180, 90, 15)

    if len(serie) < 14 or serie.dropna().empty:
        st.warning(
            "La serie cargada tiene muy pocos días de historia para pronosticar "
            "(se necesitan al menos ~14 días). Sube más historia o revisa el filtro de canal."
        )
        st.stop()

    modelo = ExponentialSmoothing(
        serie, trend="add", seasonal="add", seasonal_periods=7, initialization_method="estimated"
    ).fit()
    forecast = modelo.forecast(horizonte)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie.index[-180:], y=serie.values[-180:], name="Histórico (últimos 180 días)"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Forecast", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="Fecha", yaxis_title="Unidades vendidas")
    st.plotly_chart(fig, width='stretch')

    c1, c2 = st.columns(2)
    c1.metric("Venta promedio últimos 30 días", f"{serie[-30:].mean():,.0f} u/día")
    c2.metric("Venta promedio pronosticada", f"{forecast.mean():,.0f} u/día")

    st.caption(
        "En producción, esta serie se alimenta directamente del POS/ERP por canal y SKU, "
        "y el forecast retroalimenta tanto el plan de reabasto como el calendario de promociones "
        "(evitando promocionar cuando ya hay presión de demanda natural, por ejemplo)."
    )

# ----------------------------------------------------------------------------
# PAGINA: Simulador de negociacion
# ----------------------------------------------------------------------------
elif page == "🤝 Simulador de Negociación":
    st.title("Simulador de negociación con retailers")
    st.write(
        "Compara la propuesta del retailer contra una contrapropuesta antes de "
        "sentarte a negociar, con impacto estimado en volumen y margen."
    )

    cadena_sel = st.selectbox("Cadena con la que vas a negociar", list(RETAILER_PRESETS.keys()))
    preset = RETAILER_PRESETS[cadena_sel]
    st.caption(
        f"Valores iniciales cargados a partir de un perfil típico de **{cadena_sel}** "
        "(referencia ilustrativa, no cifras confidenciales reales) — ajústalos libremente."
    )

    volumen_base = st.number_input("Volumen base mensual (unidades)", value=20000, step=500)
    precio_regular = st.number_input("Precio regular (MXN)", value=45.0, step=1.0, key="neg_precio")
    costo_unitario = st.number_input("Costo unitario (MXN)", value=28.0, step=1.0, key="neg_costo")

    st.subheader(f"Propuesta de {cadena_sel}")
    c1, c2, c3 = st.columns(3)
    desc_retailer = c1.slider("Descuento solicitado (%)", 0, 40, preset["descuento"], key=f"dr_{cadena_sel}")
    fee_retailer = c2.number_input(
        "Listing fee / cuota fija (MXN)", value=preset["fee"], step=5000, key=f"fr_{cadena_sel}"
    )
    display_retailer = (
        c3.selectbox("Exhibición adicional incluida", ["No", "Sí"], index=1 if preset["display"] else 0, key=f"disr_{cadena_sel}")
        == "Sí"
    )

    st.subheader("Tu contrapropuesta")
    c4, c5, c6 = st.columns(3)
    desc_propio = c4.slider("Descuento ofrecido (%)", 0, 40, max(preset["descuento"] - 8, 0), key=f"dp_{cadena_sel}")
    fee_propio = c5.number_input(
        "Listing fee / cuota fija (MXN) ", value=max(preset["fee"] - 10000, 0), step=5000, key=f"fp_{cadena_sel}"
    )
    display_propio = (
        c6.selectbox("Exhibición adicional ofrecida ", ["No", "Sí"], index=1 if preset["display"] else 0, key=f"disp_{cadena_sel}")
        == "Sí"
    )

    def escenario(descuento, fee, display, uplift_por_display=None, uplift_por_desc=0.9):
        uplift_por_display = preset["uplift_display"] if uplift_por_display is None else uplift_por_display
        precio_promo = precio_regular * (1 - descuento / 100)
        margen_unit = precio_promo - costo_unitario
        uplift = volumen_base * (uplift_por_desc * descuento / 100 + (uplift_por_display if display else 0))
        volumen_total = volumen_base + uplift
        margen_total = volumen_total * margen_unit - fee
        return volumen_total, margen_total

    vol_r, marg_r = escenario(desc_retailer, fee_retailer, display_retailer)
    vol_p, marg_p = escenario(desc_propio, fee_propio, display_propio)

    comp_df = pd.DataFrame(
        {
            "Escenario": [f"Propuesta de {cadena_sel}", "Tu contrapropuesta"],
            "Volumen estimado": [vol_r, vol_p],
            "Margen neto estimado (MXN)": [marg_r, marg_p],
        }
    )
    st.dataframe(comp_df.style.format({"Volumen estimado": "{:,.0f}", "Margen neto estimado (MXN)": "${:,.0f}"}), width='stretch')

    fig = px.bar(comp_df, x="Escenario", y="Margen neto estimado (MXN)", color="Escenario", text_auto=".2s")
    st.plotly_chart(fig, width='stretch')

    diferencia = marg_p - marg_r
    if diferencia > 0:
        st.success(f"Tu contrapropuesta genera **${diferencia:,.0f} MXN** más de margen neto que la propuesta de {cadena_sel}.")
    else:
        st.warning(f"La propuesta de {cadena_sel} genera **${-diferencia:,.0f} MXN** más de margen neto que tu contrapropuesta actual.")

# ----------------------------------------------------------------------------
# PAGINA: Alertas cross-funcionales
# ----------------------------------------------------------------------------
elif page == "🚨 Alertas Cross-funcionales":
    st.title("Dashboard de alertas cross-funcionales")
    st.write(
        "Una sola fuente de verdad para Ventas, Trade Marketing y Marketing, "
        "para romper silos y alinear prioridades en la misma reunión."
    )

    df = get_effective_alerts_data()
    if "df_alertas" in st.session_state:
        st.caption("🟢 Usando **tu dataset** de alertas/KPI cargado.")

    def status(row):
        ratio = row["actual"] / row["meta"]
        if row["kpi"].lower().startswith("quiebres"):  # métrica donde menor es mejor
            ratio = row["meta"] / row["actual"]
        if ratio >= 0.97:
            return "🟢 En línea"
        elif ratio >= 0.85:
            return "🟡 Atención"
        return "🔴 Crítico"

    df["estatus"] = df.apply(status, axis=1)

    for area in df["area"].unique():
        st.subheader(area)
        sub = df[df["area"] == area]
        cols = st.columns(len(sub))
        for col, (_, row) in zip(cols, sub.iterrows()):
            col.metric(row["kpi"], f"{row['actual']}{row['unidad']}", delta=f"Meta: {row['meta']}{row['unidad']}")
            col.markdown(row["estatus"])

    st.divider()
    st.dataframe(df, width='stretch', hide_index=True)
    st.caption(
        "En producción, estas métricas se calculan en tiempo real desde POS, TPM y el CRM, "
        "y se pueden enviar automáticamente por Slack/Teams cuando un KPI cae en zona crítica."
    )


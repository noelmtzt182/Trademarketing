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


@st.cache_data
def get_store_data(n=120):
    """Universo sintetico de tiendas para segmentacion."""
    canal = RNG.choice(
        ["Autoservicio", "Conveniencia", "Tradicional", "Mayoreo", "E-commerce"],
        size=n,
        p=[0.25, 0.2, 0.3, 0.15, 0.1],
    )
    volumen_promedio = RNG.gamma(shape=2.2, scale=4000, size=n)
    margen_pct = np.clip(RNG.normal(22, 6, n), 5, 45)
    distancia_cd_km = RNG.uniform(5, 400, n)
    rotacion_dias = np.clip(RNG.normal(12, 5, n), 2, 40)
    quiebre_rate = np.clip(RNG.normal(0.08, 0.05, n), 0, 0.4)

    return pd.DataFrame(
        {
            "tienda_id": [f"T-{i:04d}" for i in range(n)],
            "canal": canal,
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
            {"area": "Ventas", "kpi": "Quiebres de stock (tiendas clave)", "actual": 9.4, "meta": 5, "unidad": "%"},
            {"area": "Trade Marketing", "kpi": "ROI de promociones activas", "actual": 1.6, "meta": 2.0, "unidad": "x"},
            {"area": "Trade Marketing", "kpi": "Cumplimiento de planograma", "actual": 78, "meta": 90, "unidad": "%"},
            {"area": "Marketing", "kpi": "Share of shelf vs objetivo", "actual": 24, "meta": 30, "unidad": "%"},
            {"area": "Marketing", "kpi": "Inversion en retail media utilizada", "actual": 61, "meta": 85, "unidad": "%"},
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
@st.cache_resource
def train_uplift_model():
    df = get_promo_training_data()
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
# UI: Sidebar / navegacion
# ----------------------------------------------------------------------------
st.sidebar.title("📊 Trade Marketing AI Suite")
st.sidebar.caption("Framework Sense → Predict → Act")

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
    "⚠️ Todos los datos de esta demo son sintéticos. "
    "Sustituye los generadores `get_*_data()` por tus fuentes reales "
    "(POS, ERP, TPM, CRM) para producción."
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

    model = train_uplift_model()

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

    df = get_store_data()
    n_clusters = st.slider("Número de clusters", 2, 6, 4)

    features = ["volumen_promedio", "margen_pct", "distancia_cd_km", "rotacion_dias", "quiebre_rate"]
    X = StandardScaler().fit_transform(df[features])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(X)
    df["cluster"] = km.labels_.astype(str)

    fig = px.scatter(
        df, x="volumen_promedio", y="margen_pct", color="cluster",
        symbol="canal", hover_data=["tienda_id", "canal", "quiebre_rate", "rotacion_dias"],
        title="Tiendas por volumen y margen, coloreadas por cluster",
    )
    st.plotly_chart(fig, width='stretch')

    st.subheader("Perfil promedio por cluster")
    perfil = df.groupby("cluster")[features].mean().round(1)
    perfil["n_tiendas"] = df.groupby("cluster").size()
    st.dataframe(perfil, width='stretch')

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

    serie = get_demand_series()
    horizonte = st.slider("Días a pronosticar", 30, 180, 90, 15)

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

    volumen_base = st.number_input("Volumen base mensual (unidades)", value=20000, step=500)
    precio_regular = st.number_input("Precio regular (MXN)", value=45.0, step=1.0, key="neg_precio")
    costo_unitario = st.number_input("Costo unitario (MXN)", value=28.0, step=1.0, key="neg_costo")

    st.subheader("Propuesta del retailer")
    c1, c2, c3 = st.columns(3)
    desc_retailer = c1.slider("Descuento solicitado (%)", 0, 40, 20, key="dr")
    fee_retailer = c2.number_input("Listing fee / cuota fija (MXN)", value=50000, step=5000, key="fr")
    display_retailer = c3.selectbox("Exhibición adicional incluida", ["No", "Sí"], key="disr") == "Sí"

    st.subheader("Tu contrapropuesta")
    c4, c5, c6 = st.columns(3)
    desc_propio = c4.slider("Descuento ofrecido (%)", 0, 40, 12, key="dp")
    fee_propio = c5.number_input("Listing fee / cuota fija (MXN) ", value=20000, step=5000, key="fp")
    display_propio = c6.selectbox("Exhibición adicional ofrecida ", ["No", "Sí"], key="disp") == "Sí"

    def escenario(descuento, fee, display, uplift_por_display=0.12, uplift_por_desc=0.9):
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
            "Escenario": ["Propuesta del retailer", "Tu contrapropuesta"],
            "Volumen estimado": [vol_r, vol_p],
            "Margen neto estimado (MXN)": [marg_r, marg_p],
        }
    )
    st.dataframe(comp_df.style.format({"Volumen estimado": "{:,.0f}", "Margen neto estimado (MXN)": "${:,.0f}"}), width='stretch')

    fig = px.bar(comp_df, x="Escenario", y="Margen neto estimado (MXN)", color="Escenario", text_auto=".2s")
    st.plotly_chart(fig, width='stretch')

    diferencia = marg_p - marg_r
    if diferencia > 0:
        st.success(f"Tu contrapropuesta genera **${diferencia:,.0f} MXN** más de margen neto que la propuesta del retailer.")
    else:
        st.warning(f"La propuesta del retailer genera **${-diferencia:,.0f} MXN** más de margen neto que tu contrapropuesta actual.")

# ----------------------------------------------------------------------------
# PAGINA: Alertas cross-funcionales
# ----------------------------------------------------------------------------
elif page == "🚨 Alertas Cross-funcionales":
    st.title("Dashboard de alertas cross-funcionales")
    st.write(
        "Una sola fuente de verdad para Ventas, Trade Marketing y Marketing, "
        "para romper silos y alinear prioridades en la misma reunión."
    )

    df = get_alerts_data()

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

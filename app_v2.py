import html
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="CLV Insights · Inteligencia Comercial",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "Datos_Complementarios_CLV.xlsx"
CLV_FILE = BASE_DIR / "Datos_CLV_2026_Tec_v2.xlsx"
# Hojas que se toman del archivo Datos_CLV además de todas las de Datos_Complementarios.
CLV_SHEETS = ["Ventas_CLV_2022_2026", "Negocios_CRM", "Histórico_Compras"]

PALETTE = ["#1B6F8A", "#F28F3B", "#2A9D8F", "#C8553D", "#6C5B7B", "#E9C46A", "#457B9D", "#8AB17D"]
px.defaults.color_discrete_sequence = PALETTE
SEED = 42
PLOT_SAMPLE = 5000
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]

# ---------------------------------------------------------------------------
# Estilo
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
.block-container {padding-top: 2.6rem; max-width: 1400px;}
.hero {background: linear-gradient(120deg, #0E3B4F 0%, #1B6F8A 55%, #2A9D8F 100%); border-radius: 18px;
  padding: 1.4rem 1.8rem; color: #fff; margin-bottom: 1.1rem; box-shadow: 0 10px 28px rgba(14,59,79,.20);}
.hero .eyebrow {font-size: .76rem; letter-spacing: .14em; text-transform: uppercase; opacity: .8;}
.hero .htitle {font-size: 1.95rem; font-weight: 750; margin: .15rem 0 .3rem; line-height: 1.2;}
.hero .hsub {opacity: .92; font-size: 1rem;}
.view-help {background: #fff; border: 1px solid #DCE6EC; border-radius: 12px; padding: .75rem 1rem;
  font-size: .92rem; color: #244B5A; margin-bottom: 1rem;}
.view-help b {color: #0E3B4F;}
.kpi {background: #fff; border: 1px solid #E1E8EE; border-radius: 14px; padding: .9rem 1.1rem;
  box-shadow: 0 2px 8px rgba(16,24,40,.05); margin-bottom: .6rem;}
.kpi .label {font-size: .72rem; text-transform: uppercase; letter-spacing: .07em; color: #5B6B7B; font-weight: 650;}
.kpi .value {font-size: 1.6rem; font-weight: 750; color: #0E3B4F; line-height: 1.25; margin-top: .2rem;}
.kpi .hint {font-size: .8rem; color: #6B7A89; margin-top: .1rem;}
.insight {background: #fff; border: 1px solid #E1E8EE; border-left: 6px solid var(--accent); border-radius: 14px;
  padding: .95rem 1.15rem; box-shadow: 0 2px 8px rgba(16,24,40,.05); margin-bottom: .8rem;}
.insight .tag {display: inline-block; font-size: .68rem; font-weight: 750; letter-spacing: .08em;
  text-transform: uppercase; color: var(--accent); background: var(--soft); padding: .16rem .55rem; border-radius: 999px;}
.insight .ititle {font-size: 1.04rem; font-weight: 700; margin: .5rem 0 .3rem; color: #1F2A37; line-height: 1.3;}
.insight .body {font-size: .91rem; color: #3D4B59; line-height: 1.45;}
.insight .mini {font-size: .8rem; color: #6B7A89;}
.insight .reco {margin-top: .65rem; padding: .55rem .75rem; background: #F4F7F9; border-radius: 10px;
  font-size: .87rem; color: #1F2A37; line-height: 1.4;}
.insight .reco b {color: var(--accent);}
.section-title {font-size: 1.12rem; font-weight: 720; color: #0E3B4F; margin: 1.2rem 0 .55rem;
  padding-bottom: .3rem; border-bottom: 2px solid #E1E8EE;}
.footer {font-size: .78rem; color: #8091A0; text-align: center; margin-top: 2rem;}
</style>
""",
    unsafe_allow_html=True,
)

INSIGHT_KINDS = {
    "hallazgo": ("Hallazgo", "#1B6F8A", "#E3F1F5"),
    "oportunidad": ("Oportunidad", "#2A9D8F", "#E2F4F1"),
    "alerta": ("Atención", "#C8553D", "#FBE9E5"),
    "accion": ("Acción", "#E07A1F", "#FDEEDF"),
    "info": ("Contexto", "#6C5B7B", "#EEEAF2"),
}

LABELS = {
    "total_neto": "Venta neta", "total_unidades": "Unidades vendidas", "precio_unitario": "Precio unitario",
    "nombre_cliente": "Cliente", "marc": "Marca", "marca": "Marca", "MARCA": "Marca", "estado": "Estado",
    "ciudad": "Ciudad", "vendedor": "Vendedor", "existencias": "Existencias",
    "valor_inventario": "Valor de inventario", "SALIDAS": "Salidas (piezas)", "ANO": "Año", "MES": "Mes",
    "NOMBRE": "Artículo", "SKU": "SKU", "ALMACEN": "Almacén", "almacen": "Almacén",
    "nombre_articulo": "Artículo", "valor_negocio": "Valor del negocio", "valor_previsto": "Valor previsto",
    "numero_actividades": "Número de actividades", "nombre_etapa": "Etapa", "estatus": "Estatus",
    "tipo_negocio": "Tipo de negocio", "nombre_zona": "Zona", "nombre_pipeline": "Pipeline",
    "oportunidad_ruta": "Oportunidad en ruta", "motivo_cierre_perdido": "Motivo de pérdida",
    "precio_total_neto": "Monto de compra", "unidades": "Unidades", "fecha_documento": "Fecha",
    "fecha": "Fecha", "fecha_creacion": "Fecha de creación", "fecha_cierre": "Fecha de cierre",
    "nombre_negocio": "Negocio", "ultima_modificacion": "Última modificación",
    "recencia_dias": "Días desde la última compra", "antiguedad_dias": "Días como cliente",
    "frecuencia": "Número de compras", "ticket_promedio": "Ticket promedio",
    "marcas_distintas": "Marcas distintas", "unidades_por_compra": "Piezas por compra",
    "venta_total": "Venta total", "CLV_historico": "Valor histórico (CLV)", "compras": "Compras",
    "ultima_compra": "Última compra", "segmento": "Segmento", "categoria": "Categoría",
    "anio_registro": "Año (de la fecha)", "mes_registro": "Mes (de la fecha)",
}
MONEY_HINTS = ("neto", "valor", "precio", "meta", "monto", "clv", "ticket", "venta")
ID_PATTERN = re.compile(r"(^|_)(id|cp|sku|folio|clave)($|_)", re.IGNORECASE)
METRIC_PRIORITY = ["total_neto", "valor_negocio", "precio_total_neto", "valor_inventario", "SALIDAS",
                   "existencias", "meta"]
CATEGORY_PRIORITY = ["marc", "marca", "MARCA", "nombre_zona", "estado", "vendedor", "tipo_negocio",
                     "nombre_etapa", "almacen"]


# ---------------------------------------------------------------------------
# Utilidades de presentación
# ---------------------------------------------------------------------------
def esc(value):
    return html.escape(str(value))


def label(column):
    return LABELS.get(column, str(column).replace("_", " ").strip().capitalize())


def is_money(column):
    return any(hint in str(column).lower() for hint in MONEY_HINTS)


def is_id_like(column):
    name = str(column).lower()
    return bool(ID_PATTERN.search(name)) or name in {"articulo", "folio_abreviado"}


def fmt_num(value, money=False):
    """Compact human format: 1.2 k, 3.4 M; money adds the $ sign."""
    if value is None or not np.isfinite(value):
        return "—"
    sign = "-" if value < 0 else ""
    magnitude = abs(float(value))
    if magnitude >= 1e9:
        text = f"{magnitude / 1e9:,.2f} mil M"
    elif magnitude >= 1e6:
        text = f"{magnitude / 1e6:,.2f} M"
    elif magnitude >= 1e4:
        text = f"{magnitude / 1e3:,.1f} k"
    elif magnitude.is_integer() or magnitude >= 100:
        text = f"{magnitude:,.0f}"
    else:
        text = f"{magnitude:,.2f}"
    return f"{sign}${text}" if money else f"{sign}{text}"


def fmt_col(column, value):
    return fmt_num(value, money=is_money(column))


def bold(value):
    return f"<b>{esc(value)}</b>"


def hero(eyebrow, title, subtitle):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{esc(eyebrow)}</div>'
        f'<div class="htitle">{esc(title)}</div><div class="hsub">{esc(subtitle)}</div></div>',
        unsafe_allow_html=True,
    )


def view_help(question, how):
    st.markdown(
        f'<div class="view-help"><b>¿Qué responde?</b> {esc(question)}<br><b>Cómo usarlo:</b> {esc(how)}</div>',
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(f'<div class="section-title">{esc(title)}</div>', unsafe_allow_html=True)


def insight(kind, title, body, reco=None, tag=None):
    """Card dict. title/body/reco are trusted HTML: dynamic values must go through esc()/bold()."""
    return {"kind": kind, "title": title, "body": body, "reco": reco, "tag": tag}


def render_cards(cards, per_row=3):
    for start in range(0, len(cards), per_row):
        columns = st.columns(per_row, gap="medium")
        for column, card in zip(columns, cards[start:start + per_row]):
            tag, accent, soft = INSIGHT_KINDS[card["kind"]]
            reco = f'<div class="reco"><b>Recomendación:</b> {card["reco"]}</div>' if card["reco"] else ""
            column.markdown(
                f'<div class="insight" style="--accent:{accent};--soft:{soft}">'
                f'<span class="tag">{esc(card["tag"] or tag)}</span>'
                f'<div class="ititle">{card["title"]}</div><div class="body">{card["body"]}</div>{reco}</div>',
                unsafe_allow_html=True,
            )


def kpis(items):
    columns = st.columns(len(items), gap="medium")
    for column, (name, value, hint) in zip(columns, items):
        column.markdown(
            f'<div class="kpi"><div class="label">{esc(name)}</div><div class="value">{esc(value)}</div>'
            f'<div class="hint">{esc(hint)}</div></div>',
            unsafe_allow_html=True,
        )


def technical(title="Ver métricas estadísticas"):
    return st.expander(title)


def show_chart(figure, height=None):
    figure.update_layout(margin=dict(l=10, r=10, t=55, b=10), legend_title_text="",
                         font=dict(family="Segoe UI, Inter, sans-serif"))
    if height:
        figure.update_layout(height=height)
    st.plotly_chart(figure, config={"displaylogo": False})


def show_table(table, height=None):
    config = {}
    for column in table.columns:
        if pd.api.types.is_numeric_dtype(table[column]) and not pd.api.types.is_bool_dtype(table[column]):
            config[column] = st.column_config.NumberColumn(
                label(column), format="dollar" if is_money(column) else "localized"
            )
        else:
            config[column] = st.column_config.Column(label(column))
    st.dataframe(table, width="stretch", hide_index=True, column_config=config, height=height or "auto")


def download(table, file_name, text="Descargar tabla (CSV)"):
    st.download_button(text, table.to_csv(index=False).encode("utf-8-sig"), file_name=file_name,
                       mime="text/csv")


def plot_sample(data, size=PLOT_SAMPLE):
    return data.sample(size, random_state=SEED) if len(data) > size else data


def trim_extremes(data, columns, share=0.01):
    """Drop rows outside the central (1 - share) of every column."""
    mask = pd.Series(True, index=data.index)
    for column in columns:
        low, high = data[column].quantile([share / 2, 1 - share / 2])
        mask &= data[column].between(low, high)
    return data[mask]


def strength_word(r_squared):
    if r_squared >= 0.7:
        return "muy fuerte"
    if r_squared >= 0.4:
        return "fuerte"
    if r_squared >= 0.1:
        return "moderada"
    return "débil"


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
def normalize_name(name):
    return str(name).strip().casefold()


def unify_text(series):
    """Trim text and merge spellings that differ only in case (CIUDAD DE MÉXICO -> Ciudad de México)."""
    text = series.map(lambda v: re.sub(r"\s+", " ", str(v)).strip() if pd.notna(v) else np.nan)
    keys = text.str.casefold()
    canonical = (
        pd.DataFrame({"key": keys, "value": text}).dropna()
        .value_counts().reset_index().drop_duplicates("key").set_index("key")["value"]
    )
    return keys.map(canonical).astype(object)


@st.cache_data(show_spinner="Cargando datos…")
def load_data(file_path, sheet_name):
    """Read a worksheet and normalize its date, numeric, inventory-value and text columns."""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.dropna(how="all").copy()

    for column in df.columns:
        name = column.lower()
        if ("fecha" in name or "date" in name or "modificacion" in name) and not name.endswith("_id"):
            parsed = pd.to_datetime(df[column], errors="coerce", dayfirst=True, utc=True)
            df[column] = parsed.dt.tz_localize(None)
        elif "valor" in name and df[column].dtype == "object":
            values = df[column].astype("string").str.replace(r"[$,\s]", "", regex=True)
            df[column] = pd.to_numeric(values, errors="coerce").astype(float)
        elif df[column].dtype == "object" and name.startswith("numero_"):
            df[column] = pd.to_numeric(df[column], errors="coerce")
        elif df[column].dtype == "object":
            # También resuelve columnas con texto y números mezclados (p. ej. clave_articulo).
            df[column] = unify_text(df[column])

    return df


def list_sheets():
    """Map 'label' -> (file, sheet) for every sheet of Complementarios plus the CLV extras."""
    sources = {}
    for sheet in pd.ExcelFile(EXCEL_FILE).sheet_names:
        sources[sheet] = (EXCEL_FILE, sheet)
    if CLV_FILE.exists():
        clv_available = {normalize_name(s): s for s in pd.ExcelFile(CLV_FILE).sheet_names}
        for wanted in CLV_SHEETS:
            actual = clv_available.get(normalize_name(wanted))
            if actual:
                sources[f"{actual} (Datos CLV)"] = (CLV_FILE, actual)
    return sources


def sheet_description(sheet_label):
    name = sheet_label.lower()
    if "existencias" in name:
        return "Inventario disponible por artículo y almacén"
    if "entradas" in name:
        return "Salidas de almacén por SKU, mes y año"
    if "ventas_clv" in name:
        return "Historial completo de ventas 2022–2026 por línea de factura (base de la segmentación de clientes)"
    if "ventas" in name:
        return "Ventas por línea de factura: cliente, artículo, marca y vendedor"
    if "negocios" in name:
        return "Oportunidades comerciales registradas en el CRM"
    if "compras" in name:
        return "Compras a proveedores por artículo"
    return "Datos de la hoja seleccionada"


# ---------------------------------------------------------------------------
# Cálculos cacheados
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def customer_features(df):
    """Six purchase-behaviour attributes per customer (same as Actividad 3.2) plus total sales."""
    sales = df.dropna(subset=["nombre_cliente", "total_neto"]).copy()
    sales["nombre_cliente"] = sales["nombre_cliente"].astype(str).str.strip().str.upper()
    sales = sales.drop_duplicates()
    grouped = sales.groupby("nombre_cliente")
    invoice = next((c for c in ("folio", "folio_abreviado") if c in sales.columns), None)
    frequency = grouped[invoice].nunique() if invoice else grouped.size()

    features = pd.DataFrame(index=frequency.index)
    date_column = next(
        (c for c in sales.columns if pd.api.types.is_datetime64_any_dtype(sales[c])), None
    )
    if date_column:
        reference = sales[date_column].max()
        features["recencia_dias"] = (reference - grouped[date_column].max()).dt.days
        features["antiguedad_dias"] = (reference - grouped[date_column].min()).dt.days
    features["frecuencia"] = frequency
    features["ticket_promedio"] = grouped["total_neto"].sum() / frequency
    brand = next((c for c in ("marca", "marc", "MARCA") if c in sales.columns), None)
    if brand:
        features["marcas_distintas"] = grouped[brand].nunique()
    if "total_unidades" in sales.columns:
        features["unidades_por_compra"] = grouped["total_unidades"].sum() / frequency
    features["venta_total"] = grouped["total_neto"].sum()
    return features


@st.cache_data(show_spinner=False)
def evaluate_k(X, max_k=8):
    rows = []
    for k in range(2, min(max_k, len(X) - 1) + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        rows.append({
            "k": k,
            "inercia": float(model.inertia_),
            "silhouette": float(silhouette_score(X, model.labels_, sample_size=min(len(X), 5000),
                                                 random_state=SEED)),
            "grupo_mas_chico": int(np.bincount(model.labels_).min()),
        })
    return pd.DataFrame(rows)


def recommend_k(evaluation, n_rows):
    """Best silhouette among the k whose smallest group is a real segment (>= 2 % of the rows)."""
    usable = evaluation[evaluation["grupo_mas_chico"] >= max(10, 0.02 * n_rows)]
    if usable.empty:
        usable = evaluation
    return int(usable.loc[usable["silhouette"].idxmax(), "k"])


@st.cache_data(show_spinner=False)
def fit_clusters(X, k, method, linkage):
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)
    if len(X) <= 5000:
        return AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X)
    # El jerárquico es O(n²): se ajusta en una muestra y el resto va al centro más cercano.
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(X), 5000, replace=False)
    sample_labels = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X[sample])
    centers = np.vstack([X[sample][sample_labels == g].mean(axis=0) for g in range(k)])
    return np.argmin(((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)


@st.cache_data(show_spinner=False)
def compare_linkages(X, k):
    if len(X) > 3000:
        X = X[np.random.default_rng(SEED).choice(len(X), 3000, replace=False)]
    kmeans_labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)
    rows = []
    for method in ["ward", "complete", "average", "single"]:
        labels = AgglomerativeClustering(n_clusters=k, linkage=method).fit_predict(X)
        sizes = np.bincount(labels)
        rows.append({
            "linkage": method,
            "silhouette": float(silhouette_score(X, labels)) if len(set(labels)) > 1 else np.nan,
            "grupo_mas_chico": int(sizes.min()),
            "grupo_mas_grande": int(sizes.max()),
            "coincidencia_con_kmeans_ARI": float(adjusted_rand_score(kmeans_labels, labels)),
        })
    return pd.DataFrame(rows)


ARCHETYPES = [
    {
        "name": "Leales de alto valor", "kind": "oportunidad",
        "score": lambda z: z["frecuencia"] + z["marcas_distintas"] + 0.5 * z["unidades_por_compra"] - z["recencia_dias"],
        "desc": "Compran seguido, de varias marcas y siguen activos: son el núcleo del negocio.",
        "action": "Protégelos con atención personalizada, condiciones preferenciales y un vendedor asignado.",
    },
    {
        "name": "Inactivos / en riesgo de fuga", "kind": "alerta",
        "score": lambda z: z["recencia_dias"] - 0.5 * z["frecuencia"],
        "desc": "Llevan mucho tiempo sin comprar.",
        "action": "Lanza una campaña de reactivación (llamada del vendedor, promoción de regreso) "
                  "empezando por quienes tienen mayor venta histórica.",
    },
    {
        "name": "Nuevos / en desarrollo", "kind": "hallazgo",
        "score": lambda z: -z["antiguedad_dias"] - 0.5 * z["recencia_dias"],
        "desc": "Clientes recientes que todavía compran poco y de pocas marcas.",
        "action": "Impulsa la venta cruzada de otras marcas y un seguimiento de bienvenida para "
                  "convertirlos en clientes leales.",
    },
    {
        "name": "Compras grandes puntuales", "kind": "accion",
        "score": lambda z: z["ticket_promedio"] + z["unidades_por_compra"] - 0.5 * z["frecuencia"],
        "desc": "Pocas compras, pero de alto monto o volumen.",
        "action": "Ofrece pedidos programados o acuerdos anuales para volver recurrente su volumen.",
    },
    {
        "name": "Compradores ocasionales", "kind": "hallazgo",
        "score": lambda z: -z["frecuencia"] - z["marcas_distintas"],
        "desc": "Compran pocas veces y de pocas marcas.",
        "action": "Activa la recompra con recordatorios del vendedor, paquetes de productos complementarios "
                  "y un beneficio por la siguiente compra.",
    },
]
# Puntaje mínimo para que un segmento reciba el nombre de un perfil; debajo se describe por sus atributos.
ARCHETYPE_MIN_SCORE = 0.3


def describe_by_features(z_row, top=2):
    strongest = z_row.abs().sort_values(ascending=False).head(top).index
    parts = [f"{'alto' if z_row[c] > 0 else 'bajo'} en {label(c).lower()}" for c in strongest]
    return "Se distingue por ser " + " y ".join(parts) + "."


def name_segments(z_centers, customer_mode):
    """Assign a business name to each segment from its standardized centroid."""
    names, kinds, descriptions, actions = {}, {}, {}, {}
    if customer_mode:
        pairs = []
        for segment, row in z_centers.iterrows():
            z = defaultdict(float, row.to_dict())
            for index, archetype in enumerate(ARCHETYPES):
                pairs.append((archetype["score"](z), segment, index))
        used_segments, used_archetypes = set(), set()
        for score, segment, index in sorted(pairs, reverse=True):
            if score < ARCHETYPE_MIN_SCORE or segment in used_segments or index in used_archetypes:
                continue
            archetype = ARCHETYPES[index]
            names[segment], kinds[segment] = archetype["name"], archetype["kind"]
            descriptions[segment], actions[segment] = archetype["desc"], archetype["action"]
            used_segments.add(segment)
            used_archetypes.add(index)
    for segment, row in z_centers.iterrows():
        if segment not in names:
            names[segment], kinds[segment] = f"Segmento {segment}", "info"
            descriptions[segment] = describe_by_features(row)
            main = row.abs().idxmax()
            actions[segment] = (
                f"Diseña una oferta específica alrededor de {esc(label(main).lower())}, "
                "que es lo que más distingue a este grupo."
            )
    return names, kinds, descriptions, actions


@st.cache_data(show_spinner=False)
def build_binary_version(df):
    """Second version of the sheet where every usable column has exactly two categories.

    Numeric columns are split at their mean; categorical columns with more than two
    categories are split by frequency into two groups of similar size.
    """
    binary = pd.DataFrame(index=df.index)
    rules = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series) or is_id_like(column):
            continue
        valid = series.dropna()
        n_unique = valid.nunique()
        if n_unique < 2:
            continue
        is_number = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        if is_number and n_unique > 2:
            mean = float(valid.mean())
            high, low = f"Alto (≥ {fmt_col(column, mean)})", f"Bajo (< {fmt_col(column, mean)})"
            binary[column] = np.where(series.isna(), None, np.where(series >= mean, high, low))
            rule, first, second = "Numérica: se divide por la media", high, low
        elif n_unique == 2:
            values = series.map(lambda v: (f"{v:g}" if isinstance(v, (int, float, np.number)) else str(v))
                                if pd.notna(v) else None)
            binary[column] = values
            counts = values.value_counts()
            rule, first, second = "Ya tenía dos categorías", counts.index[0], counts.index[1]
        elif n_unique > 200:
            rules.append({"variable": column, "regla": f"No se incluye: {n_unique:,} categorías (identificador o texto libre)",
                          "categoria_1": "", "categoria_2": "", "pct_categoria_1": np.nan})
            continue
        else:
            text = series.astype(str).where(series.notna())
            counts = text.value_counts()
            cumulative = counts.cumsum() / counts.sum()
            top = list(counts.index[: int((cumulative < 0.5).sum()) + 1])
            first = top[0] if len(top) == 1 else f"{top[0]} y {len(top) - 1} más"
            second = "Otros" if len(top) == 1 else f"Resto ({len(counts) - len(top)} categorías)"
            binary[column] = np.where(text.isna(), None, np.where(text.isin(top), first, second))
            rule = (f"Categórica con {n_unique} categorías: se agrupa por frecuencia en dos grupos de "
                    f"tamaño similar. Grupo 1 = {', '.join(top[:6])}{'…' if len(top) > 6 else ''}")
        column_values = binary[column].dropna()
        rules.append({"variable": column, "regla": rule, "categoria_1": first, "categoria_2": second,
                      "pct_categoria_1": float((column_values == first).mean())})
    return binary, pd.DataFrame(rules)


@st.cache_data(show_spinner=False)
def q_critical_table(k, alpha):
    grid = np.array([2, 3, 5, 10, 20, 40, 100, 1000, 100000], dtype=float)
    return grid, np.array([stats.studentized_range.ppf(1 - alpha, k, d) for d in grid])


def q_critical(k, alpha, degrees):
    """Studentized-range critical value, interpolated in 1/df (ppf is slow to evaluate per pair)."""
    grid, values = q_critical_table(k, alpha)
    inverse = 1 / max(float(degrees), 2.0)
    return float(np.interp(inverse, (1 / grid)[::-1], values[::-1]))


@st.cache_data(show_spinner=False)
def run_anova(data, group_column, value_column, alpha, max_posthoc_groups=12):
    grouped = list(data.groupby(group_column, observed=True)[value_column])
    valid_groups = [(lab, values.to_numpy(dtype=float)) for lab, values in grouped if len(values) >= 2]
    singleton_groups = sum(len(values) == 1 for _, values in grouped)
    if len(valid_groups) < 2:
        return None

    raw_labels = [lab for lab, _ in valid_groups]
    labels = [str(lab) for lab in raw_labels]
    groups = [values for _, values in valid_groups]
    t_values = {len(v): stats.t.ppf(1 - alpha / 2, len(v) - 1) for v in groups}
    summary = pd.DataFrame([
        {
            "grupo": lab,
            "n": len(values),
            "media": float(np.mean(values)),
            "desviacion_estandar": float(np.std(values, ddof=1)),
            "mediana": float(np.median(values)),
            "minimo": float(np.min(values)),
            "Q1": float(np.percentile(values, 25)),
            "Q3": float(np.percentile(values, 75)),
            "maximo": float(np.max(values)),
            "error_estandar": float(stats.sem(values)),
            "IC_media_inferior": float(np.mean(values) - t_values[len(values)] * stats.sem(values)),
            "IC_media_superior": float(np.mean(values) + t_values[len(values)] * stats.sem(values)),
        }
        for lab, values in zip(labels, groups)
    ])

    n_total = sum(len(values) for values in groups)
    k = len(groups)
    grand_mean = float(np.mean(np.concatenate(groups)))
    ss_between = float(sum(len(values) * (np.mean(values) - grand_mean) ** 2 for values in groups))
    ss_within = float(sum(np.sum((values - np.mean(values)) ** 2) for values in groups))
    ss_total = ss_between + ss_within
    df_between = k - 1
    df_within = n_total - k
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    if ms_within == 0:
        f_classic = float("inf") if ms_between > 0 else float("nan")
        p_classic = 0.0 if np.isinf(f_classic) else float("nan")
    else:
        f_classic = ms_between / ms_within
        p_classic = float(stats.f.sf(f_classic, df_between, df_within))

    residuals = np.concatenate([values - np.mean(values) for values in groups])
    if 3 <= len(residuals) <= 5000 and np.unique(residuals).size > 1:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
    else:
        shapiro_stat, shapiro_p = float("nan"), float("nan")
    levene_stat, levene_p = stats.levene(*groups, center="median")
    welch_result = stats.f_oneway(*groups, equal_var=False)
    welch_f, welch_p = float(welch_result.statistic), float(welch_result.pvalue)

    equal_variances = bool(np.isfinite(levene_p) and levene_p >= alpha)
    eta_squared = ss_between / ss_total if ss_total > 0 else float("nan")
    omega_squared = (
        max(0.0, (ss_between - df_between * ms_within) / (ss_total + ms_within))
        if ss_total + ms_within > 0 else float("nan")
    )

    anova_table = pd.DataFrame([
        {"fuente": "Entre grupos", "suma_cuadrados": ss_between, "gl": df_between,
         "cuadrado_medio": ms_between, "F": f_classic, "p_valor": p_classic},
        {"fuente": "Dentro de grupos (error)", "suma_cuadrados": ss_within, "gl": df_within,
         "cuadrado_medio": ms_within, "F": np.nan, "p_valor": np.nan},
        {"fuente": "Total", "suma_cuadrados": ss_total, "gl": n_total - 1,
         "cuadrado_medio": np.nan, "F": np.nan, "p_valor": np.nan},
    ])
    diagnostics = pd.DataFrame([
        {
            "supuesto": "Normalidad de residuos (Shapiro-Wilk)", "estadistico": shapiro_stat, "p_valor": shapiro_p,
            "resultado": (
                "Compatible con normalidad" if np.isfinite(shapiro_p) and shapiro_p >= alpha
                else "Posible desviación de normalidad" if np.isfinite(shapiro_p)
                else "No calculada (requiere 3–5000 residuos no constantes)"
            ),
        },
        {
            "supuesto": "Homogeneidad de varianzas (Levene)", "estadistico": float(levene_stat),
            "p_valor": float(levene_p),
            "resultado": "Compatible con varianzas iguales" if equal_variances else "Posible heterogeneidad; se usa Welch",
        },
        {
            "supuesto": "Independencia de las observaciones", "estadistico": np.nan, "p_valor": np.nan,
            "resultado": "No verificable estadísticamente; confirmar según el diseño de muestreo.",
        },
    ])

    # Post hoc: Tukey HSD con varianzas iguales, Games-Howell si no.
    posthoc_index = sorted(range(k), key=lambda i: len(groups[i]), reverse=True)[:max_posthoc_groups]
    k_post = len(posthoc_index)
    pairwise_rows = []
    for a_pos, first in enumerate(posthoc_index):
        for second in posthoc_index[a_pos + 1:]:
            values_a, values_b = groups[first], groups[second]
            mean_difference = float(np.mean(values_a) - np.mean(values_b))
            var_a = float(np.var(values_a, ddof=1) / len(values_a))
            var_b = float(np.var(values_b, ddof=1) / len(values_b))
            variance_term = ms_within * (1 / len(values_a) + 1 / len(values_b)) if equal_variances else var_a + var_b
            row = {"grupo_1": labels[first], "grupo_2": labels[second], "diferencia_medias": mean_difference,
                   "IC_inferior": np.nan, "IC_superior": np.nan, "p_valor_ajustado": np.nan, "significativo": False}
            if variance_term > 0:
                standard_error = np.sqrt(variance_term / 2)
                if equal_variances:
                    pair_df = df_within
                else:
                    pair_df = (var_a + var_b) ** 2 / (
                        var_a ** 2 / (len(values_a) - 1) + var_b ** 2 / (len(values_b) - 1)
                    )
                q_stat = abs(mean_difference) / standard_error
                adjusted_p = float(stats.studentized_range.sf(q_stat, k_post, pair_df))
                margin = q_critical(k_post, alpha, pair_df) * standard_error
                row.update({"IC_inferior": mean_difference - margin, "IC_superior": mean_difference + margin,
                            "p_valor_ajustado": adjusted_p, "significativo": adjusted_p < alpha})
            pairwise_rows.append(row)
    pairwise = pd.DataFrame(pairwise_rows).sort_values("p_valor_ajustado", na_position="last")

    return {
        "raw_labels": raw_labels, "summary": summary, "anova_table": anova_table, "diagnostics": diagnostics,
        "pairwise": pairwise, "singleton_groups": singleton_groups, "equal_variances": equal_variances,
        "f_classic": f_classic, "p_classic": p_classic, "welch_f": welch_f, "welch_p": welch_p,
        "eta_squared": eta_squared, "omega_squared": omega_squared, "shapiro_p": shapiro_p,
        "k": k, "k_post": k_post,
    }


# ---------------------------------------------------------------------------
# Selección de fuente y vista
# ---------------------------------------------------------------------------
for required_file in (EXCEL_FILE, CLV_FILE):
    if not required_file.exists():
        st.error(f"No se encontró el archivo de datos: {required_file}")
        st.stop()

sheet_sources = list_sheets()
sheet_names = list(sheet_sources)
if not sheet_names:
    st.error("Los archivos Excel no contienen hojas para analizar.")
    st.stop()

# Nombre en el menú (técnico), título simplificado de la sección, pregunta que responde y cómo usarlo.
VIEW_INFO = {
    "resumen": ("Resumen ejecutivo", "Panorama del negocio",
                "¿Cómo se comporta el negocio en esta fuente de datos?",
                "Lee los indicadores y las tarjetas de hallazgos; después elige un análisis en el menú lateral."),
    "clv": ("Valor del cliente (CLV)", "¿Quiénes son los clientes más valiosos?",
            "¿Qué clientes generan más valor y cuánto depende el negocio de ellos?",
            "Revisa la curva de concentración y descarga la lista de clientes para el equipo comercial."),
    "segmentos": ("Agrupamiento (k-means y jerárquico)", "Segmentación de clientes",
                  "¿Qué tipos de clientes o registros existen y qué acción conviene con cada uno?",
                  "Elige qué agrupar y cuántos segmentos quieres (el tablero sugiere el número ideal). "
                  "Explora cada segmento y descarga su lista."),
    "perfil": ("Extracción de características", "Perfil de variables",
               "¿Cómo se distribuye cada variable y dónde se concentra el valor?",
               "Elige una variable en el panel lateral; en las categóricas puedes medir por número de registros o por valor."),
    "lineal": ("Regresión lineal", "Relación entre variables",
               "¿Cuánto cambia una métrica cuando cambia otra?",
               "Elige la métrica a explicar (Y) y la variable que crees que la impulsa (X)."),
    "nolineal": ("Regresión no lineal", "Relaciones curvas",
                 "¿La relación se acelera o se satura en algún punto?",
                 "Compara la curva contra la recta para detectar saturación o crecimiento acelerado."),
    "logistica": ("Regresión logística", "Probabilidad de un resultado",
                  "¿Qué factores hacen más probable un resultado, como una venta alta o un cierre ganado?",
                  "Elige el resultado a predecir: el tablero convierte cada variable a dos categorías (base binaria). "
                  "Prueba escenarios en el simulador."),
    "anova": ("ANOVA", "Comparación entre grupos",
              "¿Hay diferencias reales en una métrica entre estados, marcas, vendedores u otros grupos?",
              "Elige la métrica y el grupo a comparar; el tablero indica si las diferencias son reales o casualidad."),
}

with st.sidebar:
    st.markdown("## CLV Insights")
    st.caption("Tablero de inteligencia comercial")
    with st.expander("Cómo usar este tablero"):
        st.markdown(
            "1. **Elige la fuente de datos** (hoja del Excel).\n"
            "2. **Elige el análisis** que quieres consultar.\n"
            "3. **Ajusta las opciones** que aparecen debajo.\n\n"
            "Las **tarjetas** resumen lo que significa cada resultado y qué hacer con él. "
            "Las métricas estadísticas están en el botón *Ver métricas estadísticas* al final de cada análisis."
        )
    preferred_sheet = "Histórico_Ventas"
    default_sheet_index = sheet_names.index(preferred_sheet) if preferred_sheet in sheet_names else 0
    selected_sheet = st.selectbox("Fuente de datos", sheet_names, index=default_sheet_index,
                                  help="Hojas de Datos_Complementarios_CLV y, marcadas con (Datos CLV), "
                                       "las hojas Ventas_CLV_2022_2026, Negocios_CRM e Histórico_Compras.")

df = load_data(*sheet_sources[selected_sheet])
if df.empty:
    st.error(f"La hoja '{selected_sheet}' no contiene registros.")
    st.stop()

date_columns = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
# Año y mes derivados de la fecha: sirven como grupos de comparación cuando la hoja no los trae.
derived_columns = []
if date_columns and not {"ANO", "MES"}.issubset(df.columns):
    df = df.copy()
    df["anio_registro"] = df[date_columns[0]].dt.year
    df["mes_registro"] = df[date_columns[0]].dt.month
    derived_columns = ["anio_registro", "mes_registro"]
numeric_columns = [
    c for c in df.select_dtypes(include=np.number).columns if not pd.api.types.is_bool_dtype(df[c])
]
analysis_numeric = [c for c in numeric_columns if not is_id_like(c) and c not in derived_columns] or numeric_columns
text_columns = [c for c in df.columns if pd.api.types.is_object_dtype(df[c])]
categorical_columns = [
    c for c in df.columns
    if c not in date_columns
    and (c in text_columns or pd.api.types.is_bool_dtype(df[c])
         or isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].nunique(dropna=True) <= 30)
]
group_columns = [
    c for c in categorical_columns if not is_id_like(c) and 2 <= df[c].nunique(dropna=True) <= 40
]
main_metric = next((c for c in METRIC_PRIORITY if c in analysis_numeric), analysis_numeric[0] if analysis_numeric else None)
main_category = next(
    (c for c in CATEGORY_PRIORITY if c in group_columns),
    next((c for c in group_columns if c not in derived_columns),
         text_columns[0] if text_columns else (group_columns[0] if group_columns else None)),
)
main_date = date_columns[0] if date_columns else None
has_clv_data = {"nombre_cliente", "total_neto"}.issubset(df.columns)

views = ["resumen"] + (["clv"] if has_clv_data else []) + [
    "segmentos", "perfil", "lineal", "nolineal", "logistica", "anova"
]
with st.sidebar:
    View = st.radio("Análisis", views, format_func=lambda key: VIEW_INFO[key][0])
    st.caption(f"{len(df):,} registros · {len(df.columns)} columnas")
    st.divider()
    st.markdown("**Ajustes del análisis**")

view_name, view_title, view_question, view_how = VIEW_INFO[View]
hero(f"{view_name} · {selected_sheet}", view_title,
     f"{sheet_description(selected_sheet)} · {len(df):,} registros" + (
         f" · del {df[main_date].min():%d/%m/%Y} al {df[main_date].max():%d/%m/%Y}"
         if main_date and df[main_date].notna().any() else ""
     ))
view_help(view_question, view_how)


def monthly_series(data, metric):
    """Monthly totals of a metric, using a date column or ANO/MES columns."""
    if main_date:
        dated = data.dropna(subset=[main_date])
        series = dated.set_index(main_date)[metric].resample("MS").sum()
        counts = dated.set_index(main_date)[metric].resample("MS").size()
        partial = len(dated) > 0 and dated[main_date].max().day < 25
    elif {"ANO", "MES"}.issubset(data.columns):
        periods = pd.to_datetime(dict(year=data["ANO"], month=data["MES"], day=1), errors="coerce")
        series = data[metric].groupby(periods).sum().sort_index()
        counts = data[metric].groupby(periods).size().sort_index()
        partial = False
    else:
        return None, False
    if len(counts) >= 4 and counts.iloc[-1] < 0.6 * counts.iloc[-4:-1].mean():
        partial = True
    return series, partial


def month_name(timestamp):
    return f"{MESES[timestamp.month - 1]} {timestamp.year}"


# ---------------------------------------------------------------------------
# Vistas
# ---------------------------------------------------------------------------
if View == "resumen":
    cards = []
    kpi_items = [("Registros", f"{len(df):,}", f"{len(df.columns)} columnas disponibles")]
    if main_metric:
        total = df[main_metric].sum()
        kpi_items.append((f"Total {label(main_metric).lower()}", fmt_col(main_metric, total),
                          f"Promedio por registro: {fmt_col(main_metric, df[main_metric].mean())}"))
    if main_category:
        kpi_items.append((f"Categorías de {label(main_category).lower()}", f"{df[main_category].nunique():,}",
                          f"Más frecuente: {df[main_category].mode().iat[0]}"))
    completeness = 1 - df.isna().to_numpy().mean()
    kpi_items.append(("Calidad de datos", f"{completeness:.1%}", "Celdas con información"))
    kpis(kpi_items)

    if main_metric and main_category:
        by_category = df.groupby(main_category)[main_metric].sum().sort_values(ascending=False)
        total = by_category.sum()
        if total > 0 and len(by_category) > 1:
            top_name, top_share = by_category.index[0], by_category.iloc[0] / total
            top3_share = by_category.head(3).sum() / total
            concentrated = top_share >= 0.4
            cards.append(insight(
                "alerta" if concentrated else "hallazgo",
                f"{bold(top_name)} concentra el {top_share:.0%} de {esc(label(main_metric).lower())}",
                f"Los 3 principales ({esc(label(main_category).lower())}) suman el {top3_share:.0%} del total "
                f"de {len(by_category):,}.",
                (f"Alta dependencia de {bold(top_name)}: asegura su continuidad (abasto, condiciones, relación) "
                 "y desarrolla un segundo y tercer motor de venta.")
                if concentrated else
                (f"Distribución sana. Usa a {bold(top_name)} como referencia de buenas prácticas para "
                 "impulsar a los siguientes en el ranking."),
            ))

    series, partial = monthly_series(df, main_metric) if main_metric else (None, False)
    if series is not None and len(series) >= 2:
        comparable = series.iloc[:-1] if partial else series
        if len(comparable) >= 4:
            last, previous = comparable.iloc[-1], comparable.iloc[-4:-1].mean()
            change = last / previous - 1 if previous else np.nan
            if np.isfinite(change):
                up = change >= 0
                cards.append(insight(
                    "oportunidad" if up else "alerta",
                    f"{esc(month_name(comparable.index[-1]).capitalize())} cerró {abs(change):.0%} "
                    f"{'arriba' if up else 'abajo'} del promedio reciente",
                    f"{esc(label(main_metric))} del mes: {bold(fmt_col(main_metric, last))} contra "
                    f"{bold(fmt_col(main_metric, previous))} promedio de los 3 meses anteriores.",
                    "Identifica qué impulsó el crecimiento (clientes, marcas, vendedores) para repetirlo."
                    if up else
                    "Revisa en el análisis de comparación entre grupos qué estados, marcas o vendedores "
                    "explican la caída.",
                ))
        best_month = series.idxmax()
        cards.append(insight(
            "info",
            f"Mes pico: {esc(month_name(best_month))}",
            f"Con {bold(fmt_col(main_metric, series.max()))}, "
            f"{(series.max() / series.mean() - 1):.0%} por encima del mes promedio."
            + (" El último mes aún está en curso y no se usa en las comparaciones." if partial else ""),
            "Planea inventario y fuerza de ventas con anticipación para los meses de mayor demanda.",
        ))
    elif main_metric:
        values = df[main_metric].dropna()
        if len(values) and values.sum() > 0:
            top_decile = values.sort_values(ascending=False).head(max(1, len(values) // 10)).sum() / values.sum()
            cards.append(insight(
                "hallazgo",
                f"El 10% de los registros reúne el {top_decile:.0%} de {esc(label(main_metric).lower())}",
                f"El valor típico (mediana) es {bold(fmt_col(main_metric, values.median()))} y el promedio "
                f"{bold(fmt_col(main_metric, values.mean()))}.",
                "Gestiona con prioridad ese 10%: cualquier cambio en él mueve el total.",
            ))
    if cards:
        section("Hallazgos clave")
        render_cards(cards)

    chart_a, chart_b = st.columns(2, gap="medium")
    if series is not None and len(series) >= 2:
        trend = series.rename("valor").reset_index()
        trend.columns = ["periodo", "valor"]
        figure = px.area(trend, x="periodo", y="valor", markers=True,
                         title=f"Evolución mensual de {label(main_metric).lower()}",
                         labels={"periodo": "", "valor": label(main_metric)})
        with chart_a:
            show_chart(figure, height=380)
    elif main_metric:
        with chart_a:
            show_chart(px.histogram(plot_sample(df), x=main_metric, nbins=40,
                                    title=f"Distribución de {label(main_metric).lower()}",
                                    labels={main_metric: label(main_metric)}), height=380)
    if main_category:
        if main_metric:
            ranking = df.groupby(main_category)[main_metric].sum().sort_values(ascending=False).head(10)
            axis_title = label(main_metric)
        else:
            ranking = df[main_category].value_counts().head(10)
            axis_title = "Registros"
        ranking = ranking.sort_values().rename("valor").reset_index()
        figure = px.bar(ranking, x="valor", y=main_category, orientation="h",
                        title=f"Top 10 por {label(main_category).lower()}",
                        labels={"valor": axis_title, main_category: ""})
        with chart_b:
            show_chart(figure, height=380)

    section("¿Qué más puedes consultar?")
    render_cards([
        insight("info", esc(VIEW_INFO[key][1]), esc(VIEW_INFO[key][2]), tag=VIEW_INFO[key][0])
        for key in views if key != "resumen"
    ], per_row=4)

elif View == "clv":
    sales = df.dropna(subset=["nombre_cliente", "total_neto"]).copy()
    invoice_column = next((c for c in ("folio", "folio_abreviado") if c in sales.columns), None)
    grouped = sales.groupby("nombre_cliente")
    summary = pd.DataFrame({"CLV_historico": grouped["total_neto"].sum()})
    summary["compras"] = grouped[invoice_column].nunique() if invoice_column else grouped.size()
    if "total_unidades" in sales.columns:
        summary["unidades"] = grouped["total_unidades"].sum()
    summary["ticket_promedio"] = summary["CLV_historico"] / summary["compras"]
    if main_date:
        reference = sales[main_date].max()
        summary["ultima_compra"] = grouped[main_date].max()
        summary["recencia_dias"] = (reference - summary["ultima_compra"]).dt.days
    summary = summary.sort_values("CLV_historico", ascending=False).reset_index()

    total = summary["CLV_historico"].sum()
    n_clients = len(summary)
    top_n = max(1, int(np.ceil(0.2 * n_clients)))
    top_share = summary["CLV_historico"].head(top_n).sum() / total if total else np.nan
    cumulative = summary["CLV_historico"].cumsum() / total
    clients_80 = int((cumulative < 0.8).sum()) + 1
    single_share = (summary["compras"] == 1).mean()

    kpis([
        ("Clientes", f"{n_clients:,}", "con al menos una compra"),
        ("Venta acumulada", fmt_num(total, money=True), "suma de la venta neta"),
        ("CLV promedio", fmt_num(summary["CLV_historico"].mean(), money=True),
         f"Cliente típico (mediana): {fmt_num(summary['CLV_historico'].median(), money=True)}"),
        ("Compras por cliente", f"{summary['compras'].mean():.1f}", "promedio en el periodo"),
    ])

    cards = [insight(
        "hallazgo",
        f"El 20% de los clientes genera el {top_share:.0%} de la venta",
        f"Solo {bold(f'{clients_80:,}')} clientes ({clients_80 / n_clients:.0%}) explican el 80% de la venta.",
        f"Crea un programa de cuentas clave para los {top_n:,} clientes top: perder uno de ellos pesa "
        "mucho más que perder a un cliente promedio." if top_share >= 0.6 else
        "La venta está repartida: crecer la base de clientes medianos es una palanca real.",
    )]
    cards.append(insight(
        "alerta" if single_share >= 0.4 else "oportunidad",
        f"El {single_share:.0%} de los clientes compró solo una vez",
        "Son clientes que ya confiaron en la empresa pero no regresaron.",
        "Diseña una segunda compra guiada: seguimiento del vendedor a los 30 días y una oferta de recompra.",
    ))
    if "recencia_dias" in summary.columns:
        valuable = summary.head(top_n)
        at_risk = valuable[valuable["recencia_dias"] > 90]
        if len(at_risk):
            cards.append(insight(
                "alerta",
                f"{len(at_risk):,} clientes top llevan más de 90 días sin comprar",
                f"Representan {bold(fmt_num(at_risk['CLV_historico'].sum(), money=True))} de venta histórica "
                f"({at_risk['CLV_historico'].sum() / total:.0%} del total).",
                "Contáctalos esta semana; están en la tabla de abajo, ordénala por días desde la última compra.",
            ))
        else:
            cards.append(insight("oportunidad", "Tus clientes top siguen activos",
                                 "Ninguno de los clientes del 20% superior lleva más de 90 días sin comprar.",
                                 "Mantén la cadencia de contacto y ofrece marcas que aún no compran."))
    section("Hallazgos clave")
    render_cards(cards)

    chart_a, chart_b = st.columns(2, gap="medium")
    pareto = pd.DataFrame({"pct_clientes": np.arange(1, n_clients + 1) / n_clients, "pct_venta": cumulative})
    figure = px.area(pareto, x="pct_clientes", y="pct_venta", title="Curva de concentración de la venta",
                     labels={"pct_clientes": "% de clientes (de mayor a menor valor)", "pct_venta": "% de la venta acumulada"})
    figure.add_hline(y=0.8, line_dash="dash", line_color="#C8553D", annotation_text="80% de la venta")
    figure.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    with chart_a:
        show_chart(figure, height=400)
    top_customers = summary.head(15).sort_values("CLV_historico")
    figure = px.bar(top_customers, x="CLV_historico", y="nombre_cliente", orientation="h",
                    title="15 clientes con mayor valor", labels={"CLV_historico": "Venta acumulada", "nombre_cliente": ""})
    with chart_b:
        show_chart(figure, height=400)

    section("Detalle por cliente")
    show_table(summary, height=380)
    download(summary, "clv_por_cliente.csv")
    st.caption("El CLV histórico es la suma de la venta neta por cliente en el periodo de la hoja; "
               "no es una predicción de compras futuras.")

elif View == "segmentos":
    modes = (["Clientes"] if has_clv_data else []) + (["Registros de la hoja"] if len(analysis_numeric) >= 2 else [])
    if not modes:
        st.info("Esta hoja no tiene suficientes variables numéricas para formar segmentos.")
        st.stop()
    mode = st.sidebar.radio(
        "¿Qué quieres agrupar?", modes,
        help="Clientes: se resume la compra de cada cliente en 6 atributos (recencia, antigüedad, frecuencia, "
             "ticket, marcas y piezas por compra), como en la Actividad 3.2. Registros: se agrupan las filas de la hoja.",
    )
    customer_mode = mode == "Clientes"
    if customer_mode:
        base = customer_features(df)
        attribute_options = [c for c in base.columns if c != "venta_total"]
        unit = "clientes"
    else:
        base = df[analysis_numeric]
        attribute_options = analysis_numeric
        unit = "registros"
    attributes = st.sidebar.multiselect("Atributos para agrupar", attribute_options,
                                        default=attribute_options[:6], format_func=label)
    method_label = st.sidebar.radio("Algoritmo", ["k-means", "Jerárquico (aglomerativo)"],
                                    help="k-means es el recomendado para segmentar; el jerárquico permite "
                                         "comparar tipos de enlace.")
    method = "kmeans" if method_label == "k-means" else "hierarchical"
    linkage_method = (
        st.sidebar.selectbox("Tipo de enlace (linkage)", ["ward", "complete", "average", "single"],
                             help="ward forma grupos de tamaño comparable; los demás tienden a aislar atípicos.")
        if method == "hierarchical" else "ward"
    )
    use_log = st.sidebar.checkbox("Suavizar valores extremos (escala logarítmica)", value=False,
                                  help="Reduce el peso de clientes o registros muy atípicos al formar los grupos.")
    if len(attributes) < 2:
        st.info("Selecciona al menos dos atributos para formar segmentos.")
        st.stop()

    data = base[attributes].replace([np.inf, -np.inf], np.nan).dropna()
    sampled = False
    if len(data) > 20000:
        data = data.sample(20000, random_state=SEED)
        sampled = True
    if len(data) < 20:
        st.info("No hay suficientes registros completos para formar segmentos.")
        st.stop()
    values = data.to_numpy(dtype=float)
    if use_log:
        values = np.sign(values) * np.log1p(np.abs(values))
    X = StandardScaler().fit_transform(values)

    with st.spinner("Buscando el número ideal de segmentos…"):
        evaluation = evaluate_k(X)
    suggested_k = recommend_k(evaluation, len(X))
    k = st.sidebar.slider("Número de segmentos", 2, int(evaluation["k"].max()), suggested_k,
                          help=f"Sugerido por el tablero: {suggested_k} (mejor separación sin grupos diminutos).")
    with st.spinner("Formando segmentos…"):
        raw_labels = fit_clusters(X, k, method, linkage_method)
    order = pd.Series(raw_labels).value_counts().index
    relabel = {old: new for new, old in enumerate(order, start=1)}
    segments = pd.Series([relabel[v] for v in raw_labels], index=data.index, name="segmento")

    if customer_mode:
        reference = base.loc[data.index, attribute_options]
    else:
        reference = data
    z_reference = (reference - reference.mean()) / reference.std(ddof=0).replace(0, 1)
    z_centers = z_reference.groupby(segments).mean()
    names, kinds, descriptions, actions = name_segments(z_centers, customer_mode)

    profile = data.join(segments)
    if customer_mode:
        profile["venta_total"] = base.loc[data.index, "venta_total"]
    profile["nombre_segmento"] = profile["segmento"].map(names)
    sizes = profile["segmento"].value_counts().sort_index()
    total_sales = profile["venta_total"].sum() if customer_mode else np.nan

    kpi_items = [
        ("Segmentos", f"{k}", f"Sugeridos: {suggested_k}"),
        (unit.capitalize(), f"{len(profile):,}", "con datos completos"),
        ("Segmento más grande", f"{sizes.max() / len(profile):.0%}", names[sizes.idxmax()]),
    ]
    if customer_mode:
        sales_by_segment = profile.groupby("segmento")["venta_total"].sum()
        best = sales_by_segment.idxmax()
        kpi_items.append(("Segmento que más vende", f"{sales_by_segment.max() / total_sales:.0%}", names[best]))
    kpis(kpi_items)
    full_history = next((s for s in sheet_names if s.startswith("Ventas_CLV")), None)
    if customer_mode and full_history and selected_sheet != full_history:
        st.caption(f"Esta hoja cubre un periodo corto. Para la segmentación con el historial completo "
                   f"2022–2026 elige la fuente «{full_history}».")
    if sampled:
        st.caption("La hoja tiene más de 20,000 registros: se usa una muestra aleatoria de 20,000 para formar los segmentos.")

    section("Perfil y acción recomendada por segmento")
    cards = []
    for segment in sizes.index:
        members = profile[profile["segmento"] == segment]
        share = len(members) / len(profile)
        body = f"{bold(f'{len(members):,}')} {unit} ({share:.0%})"
        if customer_mode:
            sales_share = members["venta_total"].sum() / total_sales
            body += f" · generan {bold(f'{sales_share:.0%}')} de la venta"
        body += f"<br>{esc(descriptions[segment])}"
        key_attributes = z_centers.loc[segment].abs().sort_values(ascending=False).head(3).index
        medians = " · ".join(f"{esc(label(c))}: {esc(fmt_col(c, members[c].median()))}"
                             for c in key_attributes if c in members.columns)
        if medians:
            body += f'<br><span class="mini">Valores típicos (mediana) — {medians}</span>'
        cards.append(insight(kinds[segment], esc(names[segment]), body, actions[segment],
                             tag=f"Segmento {segment} · {share:.0%}"))
    render_cards(cards)

    chart_a, chart_b = st.columns(2, gap="medium")
    with chart_a:
        if customer_mode:
            weights = pd.DataFrame({
                "segmento": [names[s] for s in sizes.index],
                f"% de {unit}": (sizes / len(profile)).to_numpy(),
                "% de la venta": [profile.loc[profile["segmento"] == s, "venta_total"].sum() / total_sales
                                  for s in sizes.index],
            }).melt(id_vars="segmento", var_name="medida", value_name="porcentaje")
            figure = px.bar(weights, x="segmento", y="porcentaje", color="medida", barmode="group",
                            title="Peso de cada segmento: clientes vs. venta", labels={"segmento": "", "porcentaje": ""})
            figure.update_layout(yaxis_tickformat=".0%")
        else:
            counts = pd.DataFrame({"segmento": [names[s] for s in sizes.index], "registros": sizes.to_numpy()})
            figure = px.bar(counts, x="segmento", y="registros", title="Tamaño de cada segmento",
                            labels={"segmento": ""})
        show_chart(figure, height=420)
    with chart_b:
        heat = z_centers.copy()
        heat.index = [names[s] for s in heat.index]
        heat.columns = [label(c) for c in heat.columns]
        figure = go.Figure(go.Heatmap(z=heat.to_numpy(), x=heat.columns, y=heat.index, zmid=0,
                                      colorscale="RdBu_r", text=np.round(heat.to_numpy(), 2),
                                      texttemplate="%{text}", colorbar=dict(title="vs. promedio")))
        figure.update_layout(title="ADN de cada segmento (rojo = arriba del promedio, azul = abajo)")
        show_chart(figure, height=420)

    section("Explora los segmentos")
    plot_options = list(data.columns) + (["venta_total"] if customer_mode else [])
    col_x, col_y, col_log = st.columns([2, 2, 1])
    x_axis = col_x.selectbox("Eje X", plot_options, index=0, format_func=label)
    y_axis = col_y.selectbox("Eje Y", plot_options, index=min(2, len(plot_options) - 1), format_func=label)
    log_axes = col_log.checkbox("Escala logarítmica", value=customer_mode)
    scatter_data = plot_sample(profile.reset_index().rename(columns={"index": "registro"}), 4000)
    hover = "nombre_cliente" if customer_mode else None
    figure = px.scatter(scatter_data, x=x_axis, y=y_axis, color="nombre_segmento", hover_name=hover,
                        opacity=0.65, labels={x_axis: label(x_axis), y_axis: label(y_axis)},
                        title=f"{label(y_axis)} vs. {label(x_axis)} por segmento",
                        category_orders={"nombre_segmento": [names[s] for s in sizes.index]})
    if log_axes:
        figure.update_xaxes(type="log" if (scatter_data[x_axis] > 0).all() else "linear")
        figure.update_yaxes(type="log" if (scatter_data[y_axis] > 0).all() else "linear")
    show_chart(figure, height=480)

    chosen = st.selectbox("Ver integrantes del segmento", list(sizes.index), format_func=lambda s: names[s])
    if customer_mode:
        members = profile[profile["segmento"] == chosen].drop(columns=["segmento"]).sort_values(
            "venta_total", ascending=False).reset_index()
    else:
        members = df.loc[profile.index[profile["segmento"] == chosen]].reset_index(drop=True)
    show_table(members.head(1000), height=360)
    download(members, f"segmento_{chosen}.csv", f"Descargar lista del segmento ({len(members):,} {unit})")

    with technical():
        current_silhouette = silhouette_score(X, segments.to_numpy(), sample_size=min(len(X), 5000),
                                              random_state=SEED)
        st.markdown(
            f"**Algoritmo:** {method_label}" + (f" · linkage **{linkage_method}**" if method == "hierarchical" else "")
            + f" · **k = {k}** · **silhouette = {current_silhouette:.3f}** "
            "(de -1 a 1; mayor = segmentos mejor separados). Los atributos se estandarizan "
            "(media 0, desviación 1) antes de agrupar."
        )
        st.markdown(
            "**Cómo se sugiere k:** se prueba k = 2 a 8 con k-means y se elige el silhouette más alto entre los k "
            "cuyo grupo más chico tiene al menos el 2% de los datos; así se evita que el silhouette suba solo por "
            "aislar a uno o dos atípicos."
        )
        eval_a, eval_b = st.columns(2)
        figure = px.line(evaluation, x="k", y="inercia", markers=True, title="Método del codo (inercia)")
        figure.add_vline(x=k, line_dash="dash", line_color="#C8553D")
        with eval_a:
            show_chart(figure, height=300)
        figure = px.line(evaluation, x="k", y="silhouette", markers=True, title="Coeficiente de silhouette")
        figure.add_vline(x=k, line_dash="dash", line_color="#C8553D")
        with eval_b:
            show_chart(figure, height=300)
        st.dataframe(evaluation, width="stretch", hide_index=True)
        st.markdown("**Comparación de tipos de enlace del jerárquico** (mismo k; muestra de hasta 3,000 datos):")
        st.dataframe(compare_linkages(X, k), width="stretch", hide_index=True)
        st.caption("Un silhouette alto con un grupo de 1–2 datos y otro con casi todos indica que el método "
                   "solo aisló atípicos, no que encontró segmentos útiles.")
        centers_table = profile.groupby("nombre_segmento")[list(data.columns)].agg(["mean", "median"]).round(2)
        st.markdown("**Promedio y mediana de cada atributo por segmento (unidades originales):**")
        st.dataframe(centers_table, width="stretch")

elif View == "perfil":
    kind = st.sidebar.radio("Tipo de variable", ["Categórica", "Numérica"] if analysis_numeric else ["Categórica"])
    if kind == "Categórica":
        if not categorical_columns:
            st.info("La hoja no contiene variables categóricas para graficar.")
            st.stop()
        default = categorical_columns.index(main_category) if main_category in categorical_columns else 0
        variable = st.sidebar.selectbox("Variable", categorical_columns, index=default, format_func=label)
        measures = ["Número de registros"] + ([f"Suma de {label(main_metric).lower()}"] if main_metric else [])
        measure = st.sidebar.radio("Medir por", measures)
        grouped_values = df[variable].astype("string").fillna("Sin dato")
        if measure == "Número de registros":
            totals = grouped_values.value_counts()
            value_name, money = "registros", False
        else:
            totals = df[main_metric].groupby(grouped_values).sum().sort_values(ascending=False)
            value_name, money = label(main_metric), is_money(main_metric)
        grand_total = totals.sum()
        shares = totals / grand_total if grand_total else totals * 0
        to_80 = int((shares.cumsum() < 0.8).sum()) + 1
        top_name, top_share = shares.index[0], shares.iloc[0]

        kpis([
            ("Categorías", f"{len(totals):,}", label(variable)),
            ("Principal", str(top_name)[:28], f"{top_share:.0%} del total"),
            ("Top 3", f"{shares.head(3).sum():.0%}", "del total"),
            ("Para llegar al 80%", f"{to_80:,}", f"de {len(totals):,} categorías"),
        ])
        concentrated = top_share >= 0.4 or shares.head(3).sum() >= 0.7
        cards = [
            insight(
                "alerta" if concentrated else "hallazgo",
                f"{bold(top_name)} lidera con el {top_share:.0%}",
                f"Medido por {esc(measure.lower())}. Le siguen "
                + ", ".join(f"{bold(n)} ({s:.0%})" for n, s in shares.iloc[1:3].items()) + ".",
                "La concentración es alta: cuida a los líderes y define un plan para reducir la dependencia."
                if concentrated else
                "El total está repartido: hay espacio para que las categorías medianas crezcan con foco comercial.",
            ),
            insight(
                "info",
                f"{to_80:,} de {len(totals):,} categorías explican el 80%",
                f"El resto ({len(totals) - to_80:,}) aporta solo el 20%.",
                "Concentra el esfuerzo comercial y de inventario en las categorías que forman ese 80%."
                if len(totals) > 5 else "Con pocas categorías conviene dar seguimiento a todas.",
            ),
        ]
        if "Sin dato" in shares.index and shares["Sin dato"] > 0.02:
            cards.append(insight("alerta", f"{shares['Sin dato']:.0%} de los registros no tiene {esc(label(variable).lower())}",
                                 "Los datos faltantes pueden sesgar las conclusiones.",
                                 "Pide al equipo que capture este campo de forma obligatoria."))
        section("Hallazgos clave")
        render_cards(cards)

        frequencies = totals.head(25).rename_axis("categoria").reset_index(name="valor")
        chart_a, chart_b = st.columns([3, 2], gap="medium")
        figure = px.bar(frequencies.sort_values("valor"), x="valor", y="categoria", orientation="h",
                        title=f"{label(variable)}: top {len(frequencies)}", labels={"valor": value_name, "categoria": ""})
        with chart_a:
            show_chart(figure, height=max(380, 22 * len(frequencies)))
        donut = frequencies.head(8).copy()
        if len(totals) > 8:
            donut = pd.concat([donut, pd.DataFrame([{"categoria": "Otros", "valor": totals.iloc[8:].sum()}])])
        with chart_b:
            show_chart(px.pie(donut, names="categoria", values="valor", hole=0.55, title="Participación"),
                       height=max(380, 22 * len(frequencies)))
        with technical("Ver tabla completa"):
            table = totals.rename_axis("categoria").reset_index(name="valor")
            table["participacion"] = table["valor"] / grand_total if grand_total else 0.0
            st.dataframe(table, width="stretch", hide_index=True,
                         column_config={"participacion": st.column_config.NumberColumn(format="percent")})
    else:
        default = analysis_numeric.index(main_metric) if main_metric in analysis_numeric else 0
        variable = st.sidebar.selectbox("Variable", analysis_numeric, index=default, format_func=label)
        log_scale = st.sidebar.checkbox("Escala logarítmica en la gráfica", value=False)
        values = df[variable].replace([np.inf, -np.inf], np.nan).dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        outliers = ((values < q1 - 1.5 * (q3 - q1)) | (values > q3 + 1.5 * (q3 - q1))).mean()
        kpis([
            ("Valor típico (mediana)", fmt_col(variable, values.median()), label(variable)),
            ("Promedio", fmt_col(variable, values.mean()), "se mueve con los extremos"),
            ("Rango central (50%)", f"{fmt_col(variable, q1)} – {fmt_col(variable, q3)}", "entre Q1 y Q3"),
            ("Valores atípicos", f"{outliers:.1%}", "fuera de 1.5 veces el rango central"),
        ])
        skewed = values.mean() > 1.5 * values.median() if values.median() > 0 else False
        cards = [insight(
            "hallazgo" if skewed else "info",
            "Unos pocos valores muy altos jalan el promedio" if skewed else "Distribución relativamente equilibrada",
            f"El promedio ({bold(fmt_col(variable, values.mean()))}) "
            f"{'supera por mucho' if skewed else 'es parecido a'} el valor típico ({bold(fmt_col(variable, values.median()))}).",
            "Para metas y presupuestos usa la mediana; el promedio sobrestima el caso común." if skewed
            else "El promedio es una buena referencia para metas y presupuestos.",
        )]
        if (values >= 0).all() and values.sum() > 0:
            top_decile = values.sort_values(ascending=False).head(max(1, len(values) // 10)).sum() / values.sum()
            cards.append(insight("oportunidad", f"El 10% superior concentra el {top_decile:.0%} del total",
                                 f"De {len(values):,} registros, {max(1, len(values) // 10):,} reúnen esa proporción.",
                                 "Identifica qué tienen en común esos registros (cliente, marca, vendedor) para replicarlo."))
        section("Hallazgos clave")
        render_cards(cards)
        figure = px.histogram(plot_sample(values.to_frame(), 20000), x=variable, nbins=50, log_x=False,
                              title=f"Distribución de {label(variable).lower()}", labels={variable: label(variable)})
        if log_scale:
            figure.update_yaxes(type="log")
        show_chart(figure, height=400)
        with technical():
            st.dataframe(values.describe().to_frame(label(variable)).T, width="stretch")

elif View in ("lineal", "nolineal"):
    if len(analysis_numeric) < 2:
        st.info("Se necesitan al menos dos columnas numéricas para ajustar una regresión.")
        st.stop()
    default_y = main_metric if main_metric in analysis_numeric else analysis_numeric[0]
    Variable_y = st.sidebar.selectbox("Métrica a explicar (Y)", analysis_numeric,
                                      index=analysis_numeric.index(default_y), format_func=label)
    predictor_candidates = [c for c in analysis_numeric if c != Variable_y]
    preferred_x = next((c for c in ("total_unidades", "numero_actividades", "unidades", "existencias")
                        if c in predictor_candidates), predictor_candidates[0])
    Variable_x = st.sidebar.selectbox("Variable que la impulsa (X)", predictor_candidates,
                                      index=predictor_candidates.index(preferred_x), format_func=label)
    exclude_extremes = st.sidebar.checkbox("Excluir el 1% de valores extremos", value=False,
                                           help="Útil cuando unas pocas operaciones enormes dominan la relación.")
    data = df[[Variable_x, Variable_y]].replace([np.inf, -np.inf], np.nan).dropna()
    if exclude_extremes:
        data = trim_extremes(data, [Variable_x, Variable_y])
    if len(data) < 3 or data[Variable_x].nunique() < 3 or data[Variable_y].nunique() < 2:
        st.info("La selección no tiene suficientes valores distintos para ajustar el modelo.")
        st.stop()
    x = data[Variable_x].to_numpy(dtype=float)
    y = data[Variable_y].to_numpy(dtype=float)
    regression = stats.linregress(x, y)
    r_squared = regression.rvalue ** 2
    x_name, y_name = label(Variable_x), label(Variable_y)

    if View == "lineal":
        positive = regression.slope >= 0
        significant = regression.pvalue < 0.05
        kpis([
            ("Fuerza de la relación", strength_word(r_squared).capitalize(), f"Explica el {r_squared:.0%} de las variaciones"),
            ("Dirección", "Positiva ↗" if positive else "Negativa ↘", f"A más {x_name.lower()}, {'más' if positive else 'menos'} {y_name.lower()}"),
            ("Efecto por unidad", fmt_col(Variable_y, regression.slope), f"de {y_name.lower()} por cada 1 de {x_name.lower()}"),
            ("Datos analizados", f"{len(data):,}", "registros completos"),
        ])
        cards = [insight(
            "hallazgo" if significant else "alerta",
            f"{esc(x_name)} está {'directamente' if positive else 'inversamente'} relacionada con {esc(y_name.lower())}"
            if significant else f"No hay una relación confiable entre {esc(x_name.lower())} y {esc(y_name.lower())}",
            f"En promedio, cada 1 de {esc(x_name.lower())} adicional se asocia con {bold(fmt_col(Variable_y, regression.slope))} "
            f"{'más' if positive else 'menos'} de {esc(y_name.lower())}. La relación es {bold(strength_word(r_squared))}.",
            (f"{esc(x_name)} es una palanca clara: incrementarla es una vía directa para mover {esc(y_name.lower())}."
             if positive else f"Vigila {esc(x_name.lower())}: cuando sube, {esc(y_name.lower())} tiende a bajar.")
            if significant and r_squared >= 0.1 else
            f"{esc(x_name)} por sí sola no explica {esc(y_name.lower())}: combínala con otras variables en el modelo múltiple.",
        )]
        section("Hallazgos clave")
        render_cards_slot = st.container()

        chart_a, chart_b = st.columns([3, 2], gap="medium")
        shown = plot_sample(data)
        figure = px.scatter(shown, x=Variable_x, y=Variable_y, opacity=0.55, title=f"{y_name} vs. {x_name}",
                            labels={Variable_x: x_name, Variable_y: y_name})
        line_x = np.linspace(x.min(), x.max(), 100)
        figure.add_trace(go.Scatter(x=line_x, y=regression.intercept + regression.slope * line_x, mode="lines",
                                    name="Tendencia", line=dict(color="#C8553D", width=3)))
        with chart_a:
            show_chart(figure, height=430)

        multiple_defaults = [c for c in predictor_candidates if c != Variable_x][:3]
        Variables_x = st.sidebar.multiselect("Variables del modelo múltiple", predictor_candidates,
                                             default=[Variable_x] + multiple_defaults, format_func=label)
        model_info = None
        if Variables_x:
            data_multiple = df[Variables_x + [Variable_y]].replace([np.inf, -np.inf], np.nan).dropna()
            if exclude_extremes:
                data_multiple = trim_extremes(data_multiple, Variables_x + [Variable_y])
            if len(data_multiple) > len(Variables_x) + 1 and data_multiple[Variable_y].nunique() > 1:
                model_multiple = LinearRegression().fit(data_multiple[Variables_x], data_multiple[Variable_y])
                r2_multiple = model_multiple.score(data_multiple[Variables_x], data_multiple[Variable_y])
                weights = pd.DataFrame({
                    "variable": Variables_x,
                    "coeficiente": model_multiple.coef_,
                    "peso_relativo": model_multiple.coef_ * data_multiple[Variables_x].std(ddof=0).to_numpy()
                    / data_multiple[Variable_y].std(ddof=0),
                })
                weights["nombre"] = weights["variable"].map(label)
                leader = weights.loc[weights["peso_relativo"].abs().idxmax()]
                model_info = (r2_multiple, weights, len(data_multiple))
                cards.append(insight(
                    "oportunidad",
                    f"Combinando {len(Variables_x)} variables se explica el {r2_multiple:.0%} de {esc(y_name.lower())}",
                    f"La de mayor peso es {bold(leader['nombre'])} "
                    f"({'aumenta' if leader['peso_relativo'] > 0 else 'disminuye'} {esc(y_name.lower())}).",
                    f"Usa {bold(leader['nombre'])} como indicador principal para planear y dar seguimiento a "
                    f"{esc(y_name.lower())}.",
                ))
                figure = px.bar(weights.sort_values("peso_relativo"), x="peso_relativo", y="nombre", orientation="h",
                                color=np.where(weights.sort_values("peso_relativo")["peso_relativo"] > 0, "Aumenta", "Disminuye"),
                                color_discrete_map={"Aumenta": "#2A9D8F", "Disminuye": "#C8553D"},
                                title="Peso de cada variable (modelo múltiple)",
                                labels={"peso_relativo": "Impacto relativo (estandarizado)", "nombre": ""})
                with chart_b:
                    show_chart(figure, height=430)
        with render_cards_slot:
            render_cards(cards)

        with technical():
            st.markdown("**Regresión lineal simple**")
            st.dataframe(pd.DataFrame([{
                "R²": r_squared, "r (correlación)": regression.rvalue, "pendiente": regression.slope,
                "intercepto": regression.intercept, "p_valor": regression.pvalue,
                "error_estandar_pendiente": regression.stderr, "n": len(data),
            }]), width="stretch", hide_index=True)
            if model_info:
                st.markdown(f"**Regresión lineal múltiple** · R² = {model_info[0]:.4f} · n = {model_info[2]:,}")
                st.dataframe(model_info[1][["variable", "coeficiente", "peso_relativo"]], width="stretch", hide_index=True)
                st.caption("El peso relativo es el coeficiente estandarizado (beta): permite comparar variables "
                           "con unidades distintas.")
    else:
        model_name = st.sidebar.selectbox("Forma de la curva", ["Función cuadrática", "Función exponencial"])
        x_scale = x.std()
        if x_scale == 0:
            st.info("La variable X no presenta variación suficiente para ajustar el modelo.")
            st.stop()
        if model_name == "Función exponencial" and (y <= 0).any():
            st.info("El modelo exponencial requiere que todos los valores de Y sean positivos. "
                    "Prueba la función cuadrática.")
            st.stop()
        x_scaled = (x - x.mean()) / x_scale
        if model_name == "Función cuadrática":
            coefficients = np.polyfit(x_scaled, y, 2)
            predictions = np.polyval(coefficients, x_scaled)
        else:
            coefficients = np.polyfit(x_scaled, np.log(y), 1)
            predictions = np.exp(np.polyval(coefficients, x_scaled))
        r2_curve = r2_score(y, predictions)
        gain = r2_curve - r_squared

        kpis([
            ("Ajuste de la curva", strength_word(max(r2_curve, 0)).capitalize(), f"Explica el {max(r2_curve, 0):.0%}"),
            ("Ajuste de la recta", strength_word(r_squared).capitalize(), f"Explica el {r_squared:.0%}"),
            ("Mejora de la curva", f"{gain:+.1%}".replace("%", " pts"), "vs. relación lineal"),
            ("Datos analizados", f"{len(data):,}", "registros completos"),
        ])
        cards = []
        if model_name == "Función cuadrática":
            a, b, _ = coefficients
            vertex = x.mean() + (-b / (2 * a)) * x_scale if a != 0 else np.nan
            inside = np.isfinite(vertex) and x.min() < vertex < x.max()
            if a < 0 and inside:
                cards.append(insight(
                    "alerta", f"Punto de saturación cerca de {esc(fmt_col(Variable_x, vertex))} de {esc(x_name.lower())}",
                    f"{esc(y_name)} crece con {esc(x_name.lower())} hasta ese punto y después deja de mejorar.",
                    f"Más allá de {bold(fmt_col(Variable_x, vertex))} el esfuerzo adicional en {esc(x_name.lower())} "
                    "ya no se traduce en resultados: redirige recursos.",
                ))
            elif a > 0:
                cards.append(insight(
                    "oportunidad", "Efecto acelerado",
                    f"Cada incremento de {esc(x_name.lower())} produce un aumento cada vez mayor de {esc(y_name.lower())}"
                    + (f" a partir de {bold(fmt_col(Variable_x, vertex))}." if inside else "."),
                    f"Los casos con valores altos de {esc(x_name.lower())} son los más rentables: priorízalos.",
                ))
            else:
                cards.append(insight("info", "Rendimientos decrecientes",
                                     f"{esc(y_name)} sigue creciendo con {esc(x_name.lower())}, pero cada vez más despacio.",
                                     "Busca el nivel de inversión en el que el beneficio adicional ya no compensa el costo."))
        else:
            growth = np.exp(coefficients[0] / x_scale) - 1
            cards.append(insight(
                "hallazgo", f"Cada 1 de {esc(x_name.lower())} cambia {esc(y_name.lower())} en {growth:+.2%}",
                "En una relación exponencial el efecto es porcentual: se multiplica, no se suma.",
                "Pequeñas mejoras sostenidas en la variable X se acumulan en cambios grandes de Y.",
            ))
        cards.append(insight(
            "oportunidad" if gain > 0.02 else "info",
            "La curva describe mejor la relación" if gain > 0.02 else "La recta es suficiente",
            f"La curva explica el {max(r2_curve, 0):.0%} contra {r_squared:.0%} de la recta.",
            "Usa la curva para proyectar escenarios: la relación no es proporcional." if gain > 0.02
            else "Para decisiones del día a día basta con la relación lineal (más fácil de comunicar).",
        ))
        section("Hallazgos clave")
        render_cards(cards)
        order = np.argsort(x)
        shown = plot_sample(data)
        figure = px.scatter(shown, x=Variable_x, y=Variable_y, opacity=0.5, title=f"{y_name} vs. {x_name}",
                            labels={Variable_x: x_name, Variable_y: y_name})
        step = max(1, len(order) // 2000)
        figure.add_trace(go.Scatter(x=x[order][::step], y=predictions[order][::step], mode="lines", name="Curva",
                                    line=dict(color="#C8553D", width=3)))
        line_x = np.linspace(x.min(), x.max(), 100)
        figure.add_trace(go.Scatter(x=line_x, y=regression.intercept + regression.slope * line_x, mode="lines",
                                    name="Recta", line=dict(color="#1B6F8A", width=2, dash="dash")))
        show_chart(figure, height=460)
        with technical():
            st.dataframe(pd.DataFrame([{
                "modelo": model_name, "R² curva": r2_curve, "R² lineal": r_squared,
                "coeficientes (X estandarizada)": ", ".join(f"{c:.4g}" for c in coefficients),
                "media_X": x.mean(), "desv_X": x_scale, "n": len(data),
            }]), width="stretch", hide_index=True)
            st.caption("Los coeficientes se estiman con X estandarizada (media 0, desviación 1) para estabilidad "
                       "numérica; en el exponencial se ajusta log(Y).")

elif View == "logistica":
    binary, rules = build_binary_version(df)
    tab_model, tab_base = st.tabs(["Modelo y simulador", "Base binaria (versión 2)"])

    with tab_base:
        st.markdown(
            "Para la regresión logística se genera una **segunda versión de la hoja** donde cada variable tiene "
            "exactamente **dos categorías**:\n"
            "- **Numéricas:** se dividen por la **media** → *Alto (≥ media)* / *Bajo (< media)*.\n"
            "- **Categóricas con más de 2 categorías:** se ordenan por frecuencia y se agrupan en **dos grupos de "
            "tamaño similar** (las más frecuentes hasta cubrir ~50% de los registros vs. el resto).\n"
            "- Identificadores, fechas y textos con más de 200 categorías no se incluyen."
        )
        rules_view = rules.copy()
        st.dataframe(rules_view, width="stretch", hide_index=True,
                     column_config={"pct_categoria_1": st.column_config.NumberColumn("% categoría 1", format="percent")})
        st.markdown("**Vista previa de la base binaria**")
        st.dataframe(binary.head(200), width="stretch")
        st.download_button("Descargar base binaria completa (CSV)", binary.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"base_binaria_{normalize_name(selected_sheet).replace(' ', '_')}.csv",
                           mime="text/csv")

    with tab_model:
        targets = list(binary.columns)
        if not targets:
            st.info("La hoja no tiene variables que puedan convertirse a dos categorías.")
            st.stop()
        default_target = next((c for c in ("nombre_etapa", main_metric) if c in targets), targets[0])
        target = st.sidebar.selectbox("Resultado a predecir (Y)", targets, index=targets.index(default_target),
                                      format_func=label)
        classes = binary[target].dropna().value_counts().index.tolist()
        default_class = next((i for i, c in enumerate(classes) if str(c).startswith("Alto") or "ganado" in str(c).lower()), 0)
        positive = st.sidebar.selectbox("Categoría que quieres predecir", classes, index=default_class)

        predictor_keys = [f"num::{c}" for c in analysis_numeric if c != target] + [
            f"cat::{c}" for c in binary.columns if c != target and c not in numeric_columns
        ]
        cat_first = dict(zip(rules["variable"], rules["categoria_1"])) if not rules.empty else {}

        def predictor_label(key):
            kind_, column = key.split("::", 1)
            return label(column) if kind_ == "num" else f"{label(column)} = {cat_first.get(column, '')}"

        defaults = [k for k in predictor_keys if k.startswith("num::")][:4] + [
            k for k in predictor_keys if k.startswith("cat::")][:2]
        chosen = st.sidebar.multiselect("Factores (X)", predictor_keys, default=defaults, format_func=predictor_label)
        if not chosen:
            st.info("Selecciona al menos un factor en el panel lateral.")
            st.stop()

        features = pd.DataFrame(index=df.index)
        for key in chosen:
            kind_, column = key.split("::", 1)
            if kind_ == "num":
                features[key] = df[column]
            else:
                features[key] = np.where(binary[column].isna(), np.nan, (binary[column] == cat_first[column]).astype(float))
        model_data = features.join(binary[target].rename("__y")).replace([np.inf, -np.inf], np.nan).dropna()
        if len(model_data) > 50000:
            model_data = model_data.sample(50000, random_state=SEED)
        y = (model_data["__y"] == positive).astype(int).to_numpy()
        X = model_data[chosen].astype(float)
        if len(model_data) < 20 or np.bincount(y, minlength=2).min() < 5:
            st.info("Cada categoría del resultado necesita al menos 5 registros completos; prueba con otros factores.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions, zero_division=0)
        precision = precision_score(y_test, predictions, zero_division=0)
        auc = roc_auc_score(y_test, probabilities) if len(set(y_test)) == 2 else np.nan
        base_rate = y.mean()
        baseline = max(base_rate, 1 - base_rate)

        scaler = model.named_steps["standardscaler"]
        logit = model.named_steps["logisticregression"]
        drivers = pd.DataFrame({"key": chosen, "coef_std": logit.coef_[0], "std": scaler.scale_})
        drivers["factor"] = drivers["key"].map(predictor_label)
        drivers["odds_std"] = np.exp(drivers["coef_std"])
        drivers["odds_unidad"] = np.exp(drivers["coef_std"] / drivers["std"].replace(0, np.nan))
        drivers = drivers.sort_values("coef_std")

        quality = ("excelente" if auc >= 0.9 else "buena" if auc >= 0.75 else "moderada" if auc >= 0.65 else "baja")
        outcome = f"{label(target)}: {positive}"
        kpis([
            ("Aciertos del modelo", f"{accuracy * 10:.1f} de 10", f"Sin modelo: {baseline * 10:.1f} de 10"),
            ("Casos detectados", f"{recall:.0%}", f"de los «{positive}» reales"),
            ("Confiabilidad de alertas", f"{precision:.0%}", f"cuando predice «{positive}»"),
            ("Capacidad de distinguir", quality.capitalize(), f"AUC = {auc:.2f}" if np.isfinite(auc) else ""),
        ])

        def odds_change(odds, up):
            if up:
                return f"×{odds:.2f}" if odds < 100 else "más de 100 veces"
            return f"−{1 - odds:.0%}"

        def driver_text(row):
            kind_, column = row["key"].split("::", 1)
            up = row["coef_std"] > 0
            if kind_ == "num":
                change = odds_change(row["odds_std"], up)
                return (f"Cuando {bold(label(column).lower())} sube {bold(fmt_col(column, row['std']))}, las probabilidades "
                        f"relativas de {bold(outcome)} {'aumentan' if up else 'bajan'} ({change}).")
            change = odds_change(row["odds_unidad"], up)
            return (f"Los registros con {bold(label(column).lower())} = {bold(cat_first[column])} tienen probabilidades "
                    f"relativas de {bold(outcome)} {'mayores' if up else 'menores'} ({change}).")

        cards = [insight(
            "hallazgo" if auc >= 0.65 else "alerta",
            f"El modelo distingue «{esc(positive)}» con calidad {quality}",
            f"Acierta {bold(f'{accuracy * 10:.1f} de cada 10')} casos y detecta el {bold(f'{recall:.0%}')} de los "
            f"«{esc(positive)}» reales. El {base_rate:.0%} de los registros pertenece a esa categoría.",
            "Úsalo para priorizar: ordena los casos por probabilidad y atiende primero a los de mayor puntaje."
            if auc >= 0.65 else "Los factores elegidos no bastan para predecir el resultado; agrega otros factores.",
        )]
        top_up = drivers.iloc[-1]
        top_down = drivers.iloc[0]
        if top_up["coef_std"] > 0:
            cards.append(insight("oportunidad", f"Principal impulsor: {esc(top_up['factor'])}", driver_text(top_up),
                                 f"Enfoca recursos en los casos con este factor: son los que más probablemente "
                                 f"terminan en «{esc(positive)}»."))
        if top_down["coef_std"] < 0:
            cards.append(insight("alerta", f"Principal freno: {esc(top_down['factor'])}", driver_text(top_down),
                                 f"Trátalo como señal temprana de riesgo: cuando aparece, la probabilidad de "
                                 f"«{esc(positive)}» cae."))
        section("Hallazgos clave")
        render_cards(cards)
        if np.isfinite(auc) and auc >= 0.97:
            st.caption(f"Nota: una predicción casi perfecta suele indicar que un factor forma parte del propio resultado "
                       f"(por ejemplo, el precio unitario dentro de la venta neta). Quita «{top_up['factor']}» de los "
                       "factores para descubrir otros impulsores.")

        chart_a, chart_b = st.columns([3, 2], gap="medium")
        figure = px.bar(drivers, x="coef_std", y="factor", orientation="h",
                        color=np.where(drivers["coef_std"] > 0, "Aumenta la probabilidad", "Disminuye la probabilidad"),
                        color_discrete_map={"Aumenta la probabilidad": "#2A9D8F", "Disminuye la probabilidad": "#C8553D"},
                        title=f"¿Qué mueve la probabilidad de «{positive}»?",
                        labels={"coef_std": "Impacto (coeficiente estandarizado)", "factor": ""})
        with chart_a:
            show_chart(figure, height=max(360, 45 * len(drivers)))
        with chart_b:
            with st.container(border=True):
                st.markdown("**Simulador de escenarios**")
                st.caption("Ajusta los factores y observa la probabilidad estimada.")
                scenario = {}
                for key in chosen:
                    kind_, column = key.split("::", 1)
                    if kind_ == "num":
                        scenario[key] = st.number_input(label(column), value=float(X[key].median()),
                                                        key=f"sim_{key}", format="%.2f")
                    else:
                        option = st.selectbox(label(column), [cat_first[column], "Otro grupo"], key=f"sim_{key}")
                        scenario[key] = 1.0 if option == cat_first[column] else 0.0
                probability = float(model.predict_proba(pd.DataFrame([scenario])[chosen])[0, 1])
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=probability * 100, number={"suffix": "%"},
                    title={"text": f"Probabilidad de «{positive}»"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1B6F8A"},
                           "steps": [{"range": [0, 40], "color": "#FBE9E5"}, {"range": [40, 70], "color": "#FDF3D8"},
                                     {"range": [70, 100], "color": "#E2F4F1"}]},
                ))
                gauge.update_layout(height=260, margin=dict(l=20, r=20, t=60, b=10))
                st.plotly_chart(gauge, config={"displaylogo": False})

        with technical():
            matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
            names_cm = [f"No «{positive}»", f"«{positive}»"]
            figure = go.Figure(go.Heatmap(z=matrix, x=names_cm, y=names_cm, text=matrix, texttemplate="%{text}",
                                          colorscale="Teal", showscale=False))
            figure.update_layout(title="Matriz de confusión (datos de prueba)", xaxis_title="Predicción", yaxis_title="Valor real")
            tech_a, tech_b = st.columns(2)
            with tech_a:
                show_chart(figure, height=340)
            with tech_b:
                st.dataframe(pd.DataFrame([
                    {"métrica": "Exactitud (accuracy)", "valor": f"{accuracy:.4f}"},
                    {"métrica": f"Precisión para «{positive}»", "valor": f"{precision:.4f}"},
                    {"métrica": f"Sensibilidad (recall) para «{positive}»", "valor": f"{recall:.4f}"},
                    {"métrica": f"Precisión para «No {positive}»",
                     "valor": f"{precision_score(y_test, predictions, pos_label=0, zero_division=0):.4f}"},
                    {"métrica": "AUC ROC", "valor": f"{auc:.4f}"},
                    {"métrica": "Registros entrenamiento / prueba", "valor": f"{len(X_train):,} / {len(X_test):,}"},
                ]), width="stretch", hide_index=True)
            st.dataframe(drivers[["factor", "coef_std", "odds_std", "odds_unidad", "std"]].rename(columns={
                "coef_std": "coeficiente (X estandarizada)", "odds_std": "razón de momios por 1 desv.",
                "odds_unidad": "razón de momios por unidad", "std": "desviación estándar de X"}),
                width="stretch", hide_index=True)
            st.caption("Modelo: StandardScaler + LogisticRegression con class_weight='balanced' (compensa clases "
                       "desbalanceadas). División 70/30 estratificada. Los factores categóricos entran como 1 si "
                       "el registro pertenece a la categoría 1 de la base binaria.")

elif View == "anova":
    if not analysis_numeric or not group_columns:
        st.info("Se necesita una variable numérica y una variable de grupos (entre 2 y 40 categorías).")
        st.stop()
    target_columns = [c for c in analysis_numeric if any(g != c for g in group_columns)]
    default_y = main_metric if main_metric in target_columns else target_columns[0]
    Variable_y = st.sidebar.selectbox("Métrica a comparar (Y)", target_columns,
                                      index=target_columns.index(default_y), format_func=label)
    available_groups = [c for c in group_columns if c != Variable_y]
    preferred_groups = ["estado", "marc", "marca", "MARCA", "nombre_zona", "vendedor", "tipo_negocio", "ANO", "MES",
                        "almacen", "anio_registro", "mes_registro"]
    default_group = next((c for c in preferred_groups if c in available_groups), available_groups[0])
    Variable_x = st.sidebar.selectbox("Grupos a comparar (X)", available_groups,
                                      index=available_groups.index(default_group), format_func=label)
    alpha = st.sidebar.selectbox("Nivel de exigencia (α)", [0.01, 0.05, 0.10], index=1,
                                 format_func=lambda a: f"{a:g} → {1 - a:.0%} de confianza")
    data = df[[Variable_x, Variable_y]].replace([np.inf, -np.inf], np.nan).dropna()
    with st.spinner("Comparando grupos…"):
        result = run_anova(data, Variable_x, Variable_y, alpha)
    if result is None:
        st.info("Se necesitan al menos dos grupos con dos observaciones válidas cada uno.")
        st.stop()

    summary = result["summary"]
    equal_variances = result["equal_variances"]
    selected_test = "ANOVA clásico" if equal_variances else "ANOVA de Welch"
    selected_f = result["f_classic"] if equal_variances else result["welch_f"]
    selected_p = result["p_classic"] if equal_variances else result["welch_p"]
    significant = bool(np.isfinite(selected_p) and selected_p < alpha)
    eta = result["eta_squared"]
    effect = ("grande" if eta >= 0.14 else "mediano" if eta >= 0.06 else "pequeño" if eta >= 0.01 else "mínimo")
    ranked = summary.sort_values("media", ascending=False)
    best, worst = ranked.iloc[0], ranked.iloc[-1]
    pairwise = result["pairwise"]
    n_significant = int(pairwise["significativo"].sum())
    y_name, x_name = label(Variable_y), label(Variable_x)

    kpis([
        ("¿Diferencias reales?", "Sí" if significant else "No concluyente", f"con {1 - alpha:.0%} de confianza"),
        ("Tamaño del efecto", effect.capitalize(), f"{x_name} explica el {eta:.1%} de la variación"),
        ("Mejor promedio", str(best["grupo"])[:26], fmt_col(Variable_y, best["media"])),
        ("Menor promedio", str(worst["grupo"])[:26], fmt_col(Variable_y, worst["media"])),
    ])
    ratio = best["media"] / worst["media"] if worst["media"] > 0 else np.nan
    cards = [
        insight(
            "hallazgo" if significant else "info",
            f"{'Sí hay' if significant else 'No hay evidencia de'} diferencias reales de {esc(y_name.lower())} entre "
            f"{esc(x_name.lower())}",
            (f"Las diferencias entre los {result['k']} grupos no se explican por azar; el efecto es {bold(effect)}."
             if significant else
             "Las diferencias observadas entre grupos pueden deberse a la variación natural de los datos."),
            (f"Vale la pena diseñar estrategias distintas por {esc(x_name.lower())}." if significant and eta >= 0.01
             else f"Hay diferencias, pero {esc(x_name.lower())} explica poco: combínalo con otros factores antes de decidir."
             if significant else f"No diferencies precios, metas o recursos solo por {esc(x_name.lower())}."),
        ),
        insight(
            "oportunidad" if significant else "info",
            f"{bold(best['grupo'])} lidera; {bold(worst['grupo'])} queda al final",
            f"Promedio de {bold(fmt_col(Variable_y, best['media']))} contra {bold(fmt_col(Variable_y, worst['media']))}"
            + (f" ({ratio:.1f} veces más)." if np.isfinite(ratio) and ratio >= 1.1 else ".")
            + (f" Ojo: {bold(best['grupo'])} tiene solo {int(best['n'])} registros, así que su promedio es poco estable."
               if best["n"] < 30 else ""),
            (f"Documenta qué hace distinto a {bold(best['grupo'])} (mezcla de productos, atención, precios) y "
             f"llévalo a {bold(worst['grupo'])}.") if significant else
            "Antes de actuar, confirma la tendencia con más datos o con otro periodo.",
        ),
    ]
    if len(pairwise):
        top_pairs = pairwise[pairwise["significativo"]].head(3)
        pairs_text = "; ".join(f"{esc(r.grupo_1)} vs. {esc(r.grupo_2)}" for r in top_pairs.itertuples())
        cards.append(insight(
            "accion" if n_significant and significant else "info",
            f"{n_significant} de {len(pairwise)} pares de grupos difieren de forma confiable"
            if significant or not n_significant else
            f"{n_significant} de {len(pairwise)} pares muestran indicios de diferencia",
            (f"Las diferencias más claras: {pairs_text}." if n_significant else
             "Ningún par de grupos se distingue de forma confiable.")
            + (f" Se comparan los {result['k_post']} grupos con más datos." if result["k_post"] < result["k"] else ""),
            ("Concentra la atención en esos pares: ahí las diferencias son reales." if significant else
             "Como la prueba global no es concluyente, tómalo como indicio a vigilar, no como conclusión.")
            if n_significant else "Trata a estos grupos como equivalentes para esta métrica.",
        ))
    section("Hallazgos clave")
    render_cards(cards)

    tab_means, tab_box = st.tabs(["Promedio por grupo", "Distribución completa"])
    with tab_means:
        colors = ["#2A9D8F" if g == best["grupo"] else "#C8553D" if g == worst["grupo"] else "#9FB3C2"
                  for g in ranked["grupo"]]
        figure = go.Figure(go.Bar(
            x=ranked["grupo"], y=ranked["media"], marker_color=colors,
            error_y=dict(type="data", symmetric=False, array=ranked["IC_media_superior"] - ranked["media"],
                         arrayminus=ranked["media"] - ranked["IC_media_inferior"], color="#5B6B7B"),
            hovertemplate="%{x}<br>Promedio: %{y:,.2f}<extra></extra>",
        ))
        figure.update_layout(title=f"Promedio de {y_name.lower()} por {x_name.lower()} "
                                   f"(barras = intervalo de confianza del {1 - alpha:.0%})",
                             xaxis_title="", yaxis_title=y_name)
        show_chart(figure, height=440)
    with tab_box:
        analysis_data = data[data[Variable_x].isin(result["raw_labels"])]
        figure = px.box(plot_sample(analysis_data, 20000), x=Variable_x, y=Variable_y, points="outliers",
                        title=f"Distribución de {y_name.lower()} por {x_name.lower()}",
                        labels={Variable_x: "", Variable_y: y_name})
        show_chart(figure, height=440)

    with technical():
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric(f"p-valor ({selected_test})", f"{selected_p:.4g}")
        metric_b.metric("Estadístico F", f"{selected_f:.4f}")
        metric_c.metric("η² (eta cuadrado)", f"{eta:.4f}")
        st.write(
            f"**H₀:** todas las medias de los grupos son iguales. **H₁:** al menos una media es diferente. "
            f"Con α = {alpha:g}, "
            + ("se rechaza H₀." if significant else "no se rechaza H₀." if np.isfinite(selected_p)
               else "no se pudo determinar el resultado porque la varianza residual es cero.")
            + f" Tamaño del efecto: ω² = {result['omega_squared']:.4f}."
        )
        st.markdown("**Estadística descriptiva por grupo**")
        st.dataframe(summary, width="stretch", hide_index=True)
        st.markdown("**Tabla ANOVA (clásico)**")
        st.dataframe(result["anova_table"], width="stretch", hide_index=True)
        st.markdown("**Diagnóstico de supuestos**")
        st.dataframe(result["diagnostics"], width="stretch", hide_index=True)
        if equal_variances:
            st.caption(f"p-valor ANOVA de Welch (referencia): {result['welch_p']:.4g}")
        if not np.isfinite(result["shapiro_p"]):
            st.caption("Shapiro-Wilk no se ejecuta con más de 5000 residuos; por eso se reporta como no calculada.")
        elif result["shapiro_p"] < alpha:
            st.warning("Shapiro-Wilk detecta una posible desviación de normalidad en los residuos; "
                       "interpreta los resultados considerando este supuesto.")
        posthoc_method = "Tukey HSD" if equal_variances else "Games-Howell"
        st.markdown(f"**Comparaciones post hoc** · {posthoc_method} con ajuste por comparaciones múltiples"
                    + (f" (los {result['k_post']} grupos con más datos)" if result["k_post"] < result["k"] else ""))
        st.dataframe(pairwise, width="stretch", hide_index=True)
        st.caption(
            f"Observaciones analizadas: {len(data):,}. Registros sin grupo o sin valor numérico excluidos: "
            f"{len(df) - len(data):,}."
            + (f" {result['singleton_groups']} grupo(s) con una sola observación se excluyeron."
               if result["singleton_groups"] else "")
        )

with st.expander("Ver datos de la hoja"):
    st.write(f"Filas: {len(df):,} | Columnas: {len(df.columns)}")
    st.dataframe(df.head(200), width="stretch")
st.markdown('<div class="footer">CLV Insights · Analítica comercial con datos de ventas, inventario, compras y CRM</div>',
            unsafe_allow_html=True)

from pathlib import Path
import re
import unicodedata
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

EXCEL_FILE = Path(__file__).resolve().parent / "Datos_Complementarios_CLV.xlsx"


def normalize_name(value):
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]", "", normalized.casefold())


def build_factorial_anova(data, response, factors):
    values = data[response].to_numpy(dtype=float)
    factor_levels = {factor: list(pd.unique(data[factor])) for factor in factors}
    factor_codes = {
        factor: pd.Categorical(data[factor], categories=factor_levels[factor]).codes
        for factor in factors
    }

    term_matrices = {}
    for term_size in range(1, len(factors) + 1):
        for term in combinations(factors, term_size):
            term_matrix = np.ones((len(data), 1))
            for factor in term:
                level_count = len(factor_levels[factor])
                contrasts = np.zeros((level_count, level_count - 1))
                contrasts[: level_count - 1] = np.eye(level_count - 1)
                contrasts[level_count - 1] = -1
                factor_matrix = contrasts[factor_codes[factor]]
                term_matrix = (
                    term_matrix[:, :, np.newaxis] * factor_matrix[:, np.newaxis, :]
                ).reshape(len(data), -1)
            term_matrices[term] = term_matrix

    design = np.column_stack([np.ones(len(data))] + list(term_matrices.values()))
    term_results = []

    def residual_sum_squares(matrix):
        coefficients, _, rank, _ = np.linalg.lstsq(matrix, values, rcond=None)
        residuals = values - matrix @ coefficients
        return float(residuals @ residuals), rank, residuals

    ss_error, model_rank, residuals = residual_sum_squares(design)
    df_error = len(values) - model_rank
    if df_error <= 0:
        return None, None, "No hay grados de libertad residuales; reduce el número de factores o categorías."

    for term, term_matrix in term_matrices.items():
        reduced_terms = [matrix for current_term, matrix in term_matrices.items() if current_term != term]
        reduced_design = np.column_stack([np.ones(len(data))] + reduced_terms)
        reduced_ss, reduced_rank, _ = residual_sum_squares(reduced_design)
        term_df = model_rank - reduced_rank
        term_ss = max(0.0, reduced_ss - ss_error)
        mean_square = term_ss / term_df if term_df else np.nan
        mean_square_error = ss_error / df_error
        if term_df == 0:
            statistic = p_value = np.nan
        elif mean_square_error == 0:
            statistic = float("inf") if term_ss > 0 else np.nan
            p_value = 0.0 if np.isinf(statistic) else np.nan
        else:
            statistic = mean_square / mean_square_error
            p_value = float(stats.f.sf(statistic, term_df, df_error))
        term_results.append(
            {
                "fuente": " × ".join(term),
                "suma_cuadrados_parcial": term_ss,
                "gl": term_df,
                "cuadrado_medio": mean_square,
                "F": statistic,
                "p_valor": p_value,
                "eta_cuadrado_parcial": (
                    term_ss / (term_ss + ss_error) if term_ss + ss_error > 0 else np.nan
                ),
            }
        )

    grand_mean = float(np.mean(values))
    ss_total = float(np.sum((values - grand_mean) ** 2))
    term_results.extend(
        [
            {
                "fuente": "Error",
                "suma_cuadrados_parcial": ss_error,
                "gl": df_error,
                "cuadrado_medio": ss_error / df_error,
                "F": np.nan,
                "p_valor": np.nan,
                "eta_cuadrado_parcial": np.nan,
            },
            {
                "fuente": "Total",
                "suma_cuadrados_parcial": ss_total,
                "gl": len(values) - 1,
                "cuadrado_medio": np.nan,
                "F": np.nan,
                "p_valor": np.nan,
                "eta_cuadrado_parcial": np.nan,
            },
        ]
    )
    return pd.DataFrame(term_results), residuals, None


@st.cache_data
def load_data(sheet_name, file_signature):
    """Read a worksheet and normalize its date and inventory-value columns."""
    df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)
    current_stat = EXCEL_FILE.stat()
    if file_signature != (current_stat.st_mtime_ns, current_stat.st_size):
        raise RuntimeError("El archivo Excel cambió durante la lectura; vuelve a actualizar los datos.")
    df.columns = df.columns.astype(str).str.strip()
    df = df.dropna(how="all").copy()

    for column in df.columns:
        if "fecha" in column.lower() or "date" in column.lower():
            df[column] = pd.to_datetime(df[column], errors="coerce", dayfirst=True)
        elif "valor" in column.lower() and df[column].dtype == "object":
            values = df[column].astype("string").str.replace(r"[$,\s]", "", regex=True)
            df[column] = pd.to_numeric(values, errors="coerce")
        elif column.lower() == "sku":
            df[column] = df[column].astype("string")

    return df


if not EXCEL_FILE.exists():
    st.error(f"No se encontró el archivo de datos: {EXCEL_FILE}")
    st.stop()

if st.sidebar.button("Actualizar datos del Excel"):
    load_data.clear()
    st.rerun()

try:
    sheet_names = pd.ExcelFile(EXCEL_FILE).sheet_names
except PermissionError:
    st.error(
        "No se puede leer el archivo Excel porque otro proceso lo mantiene bloqueado. "
        "Cierra el archivo en Excel (incluida cualquier ventana de vista previa) "
        "y vuelve a pulsar «Actualizar datos del Excel»."
    )
    st.stop()

if not sheet_names:
    st.error("El archivo Excel no contiene hojas para analizar.")
    st.stop()

preferred_sheet = normalize_name("Historico_Ventas")
default_sheet_index = next(
    (index for index, name in enumerate(sheet_names) if normalize_name(name) == preferred_sheet),
    0,
)
selected_sheet = st.sidebar.selectbox("Hoja del archivo Excel", sheet_names, index=default_sheet_index)
workbook_stat = EXCEL_FILE.stat()
try:
    df = load_data(selected_sheet, (workbook_stat.st_mtime_ns, workbook_stat.st_size))
except PermissionError:
    st.error(
        "No se puede cargar la hoja porque otro proceso mantiene bloqueado el archivo Excel. "
        "Cierra el archivo en Excel (incluida cualquier ventana de vista previa) "
        "y vuelve a pulsar «Actualizar datos del Excel»."
    )
    st.stop()

if df.empty:
    st.error(f"La hoja '{selected_sheet}' no contiene registros.")
    st.stop()

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
categorical_columns = [
    column
    for column in df.columns
    if not pd.api.types.is_datetime64_any_dtype(df[column])
    and (
        pd.api.types.is_object_dtype(df[column])
        or pd.api.types.is_categorical_dtype(df[column])
        or pd.api.types.is_bool_dtype(df[column])
        or df[column].nunique(dropna=True) <= 30
    )
]

st.sidebar.title("ANÁLISIS CLV")
st.sidebar.caption(f"{len(df):,} registros en {selected_sheet}")

views = [
    "Extracción de Características",
    "Regresión Lineal",
    "Regresión No Lineal",
    "Regresión Logística",
    "ANOVA de un factor",
    "ANOVA factorial",
]
has_clv_data = {"nombre_cliente", "total_neto"}.issubset(df.columns)
if has_clv_data:
    views.insert(0, "CLV histórico por cliente")

View = st.sidebar.selectbox("Tipo de análisis", options=views)
st.title("Análisis de datos complementarios CLV")
st.caption(f"Hoja seleccionada: {selected_sheet}")

if View == "CLV histórico por cliente":
    sales = df.dropna(subset=["nombre_cliente", "total_neto"]).copy()
    invoice_column = (
        "folio" if "folio" in sales.columns
        else "folio_abreviado" if "folio_abreviado" in sales.columns
        else None
    )

    aggregations = {"CLV_historico": ("total_neto", "sum")}
    if invoice_column:
        aggregations["compras"] = (invoice_column, "nunique")
    else:
        aggregations["compras"] = ("total_neto", "size")
    if "total_unidades" in sales.columns:
        aggregations["unidades_compradas"] = ("total_unidades", "sum")

    customer_summary = sales.groupby("nombre_cliente", as_index=False).agg(**aggregations)
    customer_summary["ticket_promedio"] = customer_summary["CLV_historico"] / customer_summary["compras"]
    customer_summary = customer_summary.sort_values("CLV_historico", ascending=False)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Clientes", f"{len(customer_summary):,}")
    col_b.metric("Ventas históricas", f"${customer_summary['CLV_historico'].sum():,.2f}")
    col_c.metric("CLV histórico promedio", f"${customer_summary['CLV_historico'].mean():,.2f}")

    top_customers = customer_summary.head(20).sort_values("CLV_historico")
    figure = px.bar(
        top_customers,
        x="CLV_historico",
        y="nombre_cliente",
        orientation="h",
        title="20 clientes con mayor venta histórica",
        labels={"CLV_historico": "Venta histórica acumulada", "nombre_cliente": "Cliente"},
    )
    st.plotly_chart(figure, use_container_width=True)
    st.subheader("Resumen por cliente")
    st.dataframe(customer_summary, use_container_width=True)
    st.caption("CLV histórico es la suma de total_neto por cliente; no es una predicción de compras futuras.")

elif View == "Extracción de Características":
    if not categorical_columns:
        st.info("La hoja no contiene variables categóricas o de baja cardinalidad para graficar.")
    else:
        Variable_Cat = st.sidebar.selectbox("Variable", options=categorical_columns)
        frequencies = (
            df[Variable_Cat]
            .astype("string")
            .fillna("Sin dato")
            .value_counts()
            .head(25)
            .rename_axis("categoria")
            .reset_index(name="frecuencia")
        )
        st.subheader(f"Frecuencia de {Variable_Cat} (máximo 25 categorías)")
        col_a, col_b = st.columns(2)
        col_a.plotly_chart(
            px.bar(frequencies, x="categoria", y="frecuencia", title="Frecuencia por categoría"),
            use_container_width=True,
        )
        col_b.plotly_chart(
            px.pie(frequencies, names="categoria", values="frecuencia", hole=0.4, title="Distribución"),
            use_container_width=True,
        )
        st.dataframe(frequencies, use_container_width=True)

elif View == "Regresión Lineal":
    if len(numeric_columns) < 2:
        st.info("Se necesitan al menos dos columnas numéricas para ajustar una regresión lineal.")
    else:
        default_y = "total_neto" if "total_neto" in numeric_columns else numeric_columns[0]
        Variable_y = st.sidebar.selectbox(
            "Variable objetivo (Y)", numeric_columns, index=numeric_columns.index(default_y)
        )
        predictor_candidates = [column for column in numeric_columns if column != Variable_y]
        default_x = "total_unidades" if "total_unidades" in predictor_candidates else predictor_candidates[0]
        Variable_x = st.sidebar.selectbox(
            "Variable independiente simple (X)",
            predictor_candidates,
            index=predictor_candidates.index(default_x),
        )

        data_simple = df[[Variable_x, Variable_y]].replace([np.inf, -np.inf], np.nan).dropna()
        st.subheader("Regresión lineal simple")
        if len(data_simple) < 2 or data_simple[Variable_x].nunique() < 2 or data_simple[Variable_y].nunique() < 2:
            st.info("La selección no tiene suficientes valores distintos para ajustar el modelo.")
        else:
            model = LinearRegression().fit(data_simple[[Variable_x]], data_simple[Variable_y])
            st.metric("R²", f"{model.score(data_simple[[Variable_x]], data_simple[Variable_y]):.4f}")
            figure = px.scatter(data_simple, x=Variable_x, y=Variable_y, title="Modelo lineal simple")
            line_x = np.linspace(data_simple[Variable_x].min(), data_simple[Variable_x].max(), 100)
            line_y = model.predict(pd.DataFrame({Variable_x: line_x}))
            figure.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", name="Ajuste lineal"))
            st.plotly_chart(figure, use_container_width=True)

        st.subheader("Regresión lineal múltiple")
        multiple_defaults = [column for column in predictor_candidates if column != Variable_x][:3]
        Variables_x = st.sidebar.multiselect(
            "Variables independientes del modelo múltiple (X)",
            options=predictor_candidates,
            default=multiple_defaults or [Variable_x],
        )
        if not Variables_x:
            st.info("Selecciona al menos una variable independiente para el modelo múltiple.")
        else:
            data_multiple = df[Variables_x + [Variable_y]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(data_multiple) < 2 or data_multiple[Variable_y].nunique() < 2:
                st.info("La selección no tiene suficientes datos para ajustar el modelo múltiple.")
            else:
                model_multiple = LinearRegression().fit(data_multiple[Variables_x], data_multiple[Variable_y])
                st.metric(
                    "R² múltiple",
                    f"{model_multiple.score(data_multiple[Variables_x], data_multiple[Variable_y]):.4f}",
                )
                coefficients = pd.DataFrame({"variable": Variables_x, "coeficiente": model_multiple.coef_})
                st.dataframe(coefficients, use_container_width=True)

elif View == "Regresión No Lineal":
    if len(numeric_columns) < 2:
        st.info("Se necesitan al menos dos columnas numéricas para ajustar una regresión no lineal.")
    else:
        default_y = "total_neto" if "total_neto" in numeric_columns else numeric_columns[0]
        Variable_y = st.sidebar.selectbox(
            "Variable objetivo (Y)", numeric_columns, index=numeric_columns.index(default_y)
        )
        predictor_candidates = [column for column in numeric_columns if column != Variable_y]
        default_x = "total_unidades" if "total_unidades" in predictor_candidates else predictor_candidates[0]
        Variable_x = st.sidebar.selectbox(
            "Variable independiente (X)",
            predictor_candidates,
            index=predictor_candidates.index(default_x),
        )
        model_name = st.sidebar.selectbox("Modelo no lineal", ["Función cuadrática", "Función exponencial"])

        data = df[[Variable_x, Variable_y]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 3 or data[Variable_x].nunique() < 3:
            st.info("Se requieren al menos tres observaciones y tres valores distintos de X.")
        else:
            x = data[Variable_x].to_numpy(dtype=float)
            y = data[Variable_y].to_numpy(dtype=float)
            x_scale = x.std()
            if x_scale == 0:
                st.info("La variable X no presenta variación suficiente para ajustar el modelo.")
            elif model_name == "Función exponencial" and (y <= 0).any():
                st.info("El modelo exponencial requiere que todos los valores de Y sean positivos.")
            else:
                x_scaled = (x - x.mean()) / x_scale
                if model_name == "Función cuadrática":
                    coefficients = np.polyfit(x_scaled, y, 2)
                    predictions = np.polyval(coefficients, x_scaled)
                else:
                    coefficients = np.polyfit(x_scaled, np.log(y), 1)
                    predictions = np.exp(np.polyval(coefficients, x_scaled))

                st.metric("R²", f"{r2_score(y, predictions):.4f}")
                order = np.argsort(x)
                figure = px.scatter(data, x=Variable_x, y=Variable_y, title=model_name)
                figure.add_trace(go.Scatter(x=x[order], y=predictions[order], mode="lines", name="Ajuste"))
                st.plotly_chart(figure, use_container_width=True)

elif View == "Regresión Logística":
    binary_targets = [
        column
        for column in df.columns
        if not pd.api.types.is_datetime64_any_dtype(df[column])
        and df[column].nunique(dropna=True) == 2
    ]
    if not binary_targets:
        st.info("Esta hoja no tiene una variable con exactamente dos clases para usar como objetivo logístico.")
    elif not numeric_columns:
        st.info("Se necesita al menos una variable numérica independiente para la regresión logística.")
    else:
        default_target = "estado" if "estado" in binary_targets else binary_targets[0]
        Variable_y = st.sidebar.selectbox(
            "Variable dependiente (Y)", binary_targets, index=binary_targets.index(default_target)
        )
        predictor_candidates = [column for column in numeric_columns if column != Variable_y]
        if not predictor_candidates:
            st.info("No hay variables numéricas independientes disponibles para el objetivo seleccionado.")
        else:
            recommended = [
                column
                for column in predictor_candidates
                if column.lower() not in {"articulo", "articulo_id", "cp"}
            ]
            Variables_x = st.sidebar.multiselect(
                "Variables independientes (X)",
                options=predictor_candidates,
                default=(recommended or predictor_candidates)[:4],
            )
            if not Variables_x:
                st.info("Selecciona al menos una variable independiente.")
            else:
                data = df[Variables_x + [Variable_y]].replace([np.inf, -np.inf], np.nan).dropna()
                encoder = LabelEncoder()
                y = encoder.fit_transform(data[Variable_y].astype(str))
                class_counts = pd.Series(y).value_counts()
                if len(data) < 4 or len(class_counts) != 2 or class_counts.min() < 2:
                    st.info("Cada clase debe tener al menos dos registros para dividir los datos en entrenamiento y prueba.")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        data[Variables_x], y, test_size=0.3, random_state=42, stratify=y
                    )
                    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
                    labels = encoder.classes_.astype(str).tolist()
                    figure = go.Figure(
                        data=go.Heatmap(
                            z=matrix,
                            x=labels,
                            y=labels,
                            text=matrix,
                            texttemplate="%{text}",
                            colorscale="Oranges",
                        )
                    )
                    figure.update_layout(
                        title="Matriz de confusión",
                        xaxis_title="Predicción",
                        yaxis_title="Valor real",
                    )
                    st.plotly_chart(figure, use_container_width=True)
                    st.metric("Exactitud", f"{accuracy_score(y_test, predictions):.4f}")
                    for class_index, label in enumerate(labels):
                        precision = precision_score(
                            y_test, predictions, pos_label=class_index, zero_division=0
                        )
                        st.write(f"Precisión para {label}: {precision:.4f}")

elif View == "ANOVA factorial":
    group_columns = [
        column for column in categorical_columns if 2 <= df[column].nunique(dropna=True) <= 30
    ]
    if not numeric_columns:
        st.info("Se necesita al menos una variable numérica para ANOVA.")
    elif len(group_columns) < 2:
        st.info("Se necesitan al menos dos variables categóricas con entre 2 y 30 niveles.")
    else:
        target_columns = [
            column
            for column in numeric_columns
            if sum(group_column != column for group_column in group_columns) >= 2
        ]
        if not target_columns:
            st.info("No hay una variable numérica separada de al menos dos factores.")
        else:
            default_y = "total_neto" if "total_neto" in target_columns else target_columns[0]
            Variable_y = st.sidebar.selectbox(
                "Variable numérica (Y)", target_columns, index=target_columns.index(default_y)
            )
            factor_options = [column for column in group_columns if column != Variable_y]
            preferred_groups = ["ano", "mes", "estado", "marca", "almacen", "vendedor", "ciudad"]
            defaults = []
            for preferred in preferred_groups:
                match = next(
                    (column for column in factor_options if normalize_name(column) == preferred),
                    None,
                )
                if match and match not in defaults:
                    defaults.append(match)
                if len(defaults) == 2:
                    break
            if len(defaults) < 2:
                defaults.extend(column for column in factor_options if column not in defaults)
            Variables_factor = st.sidebar.multiselect(
                "Factores (2 o 3; se incluyen todas sus interacciones)",
                options=factor_options,
                default=defaults[:2],
            )
            alpha = st.sidebar.selectbox("Nivel de significancia (α)", [0.01, 0.05, 0.10], index=1)

            if len(Variables_factor) < 2:
                st.info("Selecciona al menos dos factores para el ANOVA factorial.")
            elif len(Variables_factor) > 3:
                st.info("Selecciona como máximo tres factores para mantener interpretable el modelo factorial.")
            else:
                data = (
                    df[Variables_factor + [Variable_y]]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                level_counts = [data[factor].nunique(dropna=True) for factor in Variables_factor]
                if data.empty or any(level_count < 2 for level_count in level_counts):
                    st.info("Los factores seleccionados necesitan al menos dos niveles con datos válidos.")
                else:
                    model_columns = int(np.prod(level_counts))
                    if model_columns > 500:
                        st.info(
                            "El diseño genera más de 500 parámetros potenciales. "
                            "Reduce los niveles o el número de factores."
                        )
                    elif len(data) <= model_columns:
                        st.info(
                            "No hay suficientes observaciones para estimar el modelo completo. "
                            "Se requiere más observaciones que parámetros potenciales."
                        )
                    else:
                        grouped = list(data.groupby(Variables_factor, observed=True)[Variable_y])
                        valid_groups = [
                            (labels, values.to_numpy(dtype=float))
                            for labels, values in grouped
                            if len(values) >= 2
                        ]
                        if len(valid_groups) < 2:
                            st.info("Se necesitan al menos dos combinaciones de factores con dos observaciones.")
                        else:
                            anova_table, residuals, error = build_factorial_anova(
                                data, Variable_y, Variables_factor
                            )
                            if error:
                                st.info(error)
                            else:
                                group_labels = [
                                    ", ".join(
                                        f"{factor}={label}"
                                        for factor, label in zip(Variables_factor, labels)
                                    )
                                    for labels, _ in valid_groups
                                ]
                                group_values = [values for _, values in valid_groups]
                                summary = pd.DataFrame(
                                    [
                                        {
                                            "combinacion": label,
                                            "n": len(values),
                                            "media": float(np.mean(values)),
                                            "desviacion_estandar": float(np.std(values, ddof=1)),
                                            "mediana": float(np.median(values)),
                                            "minimo": float(np.min(values)),
                                            "Q1": float(np.percentile(values, 25)),
                                            "Q3": float(np.percentile(values, 75)),
                                            "maximo": float(np.max(values)),
                                        }
                                        for label, values in zip(group_labels, group_values)
                                    ]
                                )

                                st.subheader("Estadística descriptiva por combinación de factores")
                                st.dataframe(summary, use_container_width=True)
                                st.caption(
                                    f"Observaciones analizadas: {len(data):,}. "
                                    f"Filas excluidas por valores ausentes o infinitos: {len(df) - len(data):,}."
                                )
                                singleton_cells = len(grouped) - len(valid_groups)
                                if singleton_cells:
                                    st.caption(
                                        f"{singleton_cells} combinación(es) con una sola observación "
                                        "se conservaron en el modelo, pero se excluyeron de Levene y del post hoc."
                                    )

                                st.subheader("Gráficas de interacción")
                                interaction_columns = st.columns(2)
                                interaction_x = interaction_columns[0].selectbox(
                                    "Factor del eje X",
                                    Variables_factor,
                                    index=0,
                                    key="anova_interaction_x",
                                )
                                interaction_color_options = [
                                    factor for factor in Variables_factor if factor != interaction_x
                                ]
                                interaction_color = interaction_columns[1].selectbox(
                                    "Factor de interacción (líneas)",
                                    interaction_color_options,
                                    index=0,
                                    key="anova_interaction_color",
                                )
                                remaining_factors = [
                                    factor
                                    for factor in Variables_factor
                                    if factor not in {interaction_x, interaction_color}
                                ]
                                interaction_data = data
                                if remaining_factors:
                                    filter_factor = remaining_factors[0]
                                    filter_levels = list(pd.unique(data[filter_factor]))
                                    selected_level = st.selectbox(
                                        f"Nivel de {filter_factor} para la interacción",
                                        filter_levels,
                                        key="anova_interaction_filter",
                                    )
                                    interaction_data = data.loc[
                                        data[filter_factor] == selected_level
                                    ]

                                interaction_summary = (
                                    interaction_data.groupby(
                                        [interaction_x, interaction_color], observed=True
                                    )[Variable_y]
                                    .agg(media="mean", n="count", desviacion="std")
                                    .reset_index()
                                )
                                interaction_summary["error_estandar"] = (
                                    interaction_summary["desviacion"]
                                    / np.sqrt(interaction_summary["n"])
                                )
                                heatmap_data = interaction_summary.pivot(
                                    index=interaction_color,
                                    columns=interaction_x,
                                    values="media",
                                )
                                interaction_charts = st.columns(2)
                                interaction_figure = px.line(
                                    interaction_summary,
                                    x=interaction_x,
                                    y="media",
                                    color=interaction_color,
                                    markers=True,
                                    error_y="error_estandar",
                                    title=f"Interacción: {interaction_x} × {interaction_color}",
                                    labels={
                                        interaction_x: interaction_x,
                                        "media": f"Media de {Variable_y}",
                                        interaction_color: interaction_color,
                                        "error_estandar": "Error estándar",
                                    },
                                )
                                interaction_figure.update_layout(
                                    legend_title_text=interaction_color,
                                    template="plotly_white",
                                )
                                interaction_charts[0].plotly_chart(
                                    interaction_figure, use_container_width=True
                                )
                                heatmap_figure = go.Figure(
                                    data=go.Heatmap(
                                        z=heatmap_data.to_numpy(dtype=float),
                                        x=heatmap_data.columns.astype(str).tolist(),
                                        y=heatmap_data.index.astype(str).tolist(),
                                        text=np.round(heatmap_data.to_numpy(dtype=float), 2),
                                        texttemplate="%{text}",
                                        colorscale="RdBu",
                                        reversescale=True,
                                        colorbar={"title": f"Media de {Variable_y}"},
                                        hovertemplate=(
                                            f"{interaction_color}: %{{y}}<br>"
                                            f"{interaction_x}: %{{x}}<br>"
                                            "Media: %{z:.3f}<extra></extra>"
                                        ),
                                    )
                                )
                                heatmap_figure.update_layout(
                                    title=f"Heatmap de medias: {interaction_color} × {interaction_x}",
                                    xaxis_title=interaction_x,
                                    yaxis_title=interaction_color,
                                    template="plotly_white",
                                )
                                interaction_charts[1].plotly_chart(
                                    heatmap_figure, use_container_width=True
                                )
                                st.caption(
                                    "Líneas no paralelas sugieren una interacción; confirma su evidencia "
                                    "en el término de interacción de la tabla ANOVA. "
                                    "El heatmap muestra las medias observadas por combinación."
                                )

                                error_row = anova_table.loc[anova_table["fuente"] == "Error"].iloc[0]
                                error_mean_square = float(error_row["cuadrado_medio"])
                                error_df = int(error_row["gl"])
                                anova_table["significativo"] = anova_table["p_valor"].apply(
                                    lambda value: bool(np.isfinite(value) and value < alpha)
                                )
                                st.subheader("ANOVA factorial (pruebas parciales tipo III)")
                                st.dataframe(anova_table, use_container_width=True)
                                st.caption(
                                    "Se usa codificación de contrastes suma-cero y se prueban los efectos "
                                    "principales e interacciones ajustados por los demás términos. "
                                    "Las sumas de cuadrados parciales no tienen por qué sumar el total."
                                )

                                if len(residuals) >= 3 and len(residuals) <= 5000 and np.unique(residuals).size > 1:
                                    shapiro_stat, shapiro_p = stats.shapiro(residuals)
                                    normality_result = (
                                        "Compatible con normalidad"
                                        if shapiro_p >= alpha
                                        else "Posible desviación de normalidad"
                                    )
                                else:
                                    shapiro_stat, shapiro_p = np.nan, np.nan
                                    normality_result = "No calculada (requiere 3–5000 residuos no constantes)"
                                if len(group_values) >= 2:
                                    levene_stat, levene_p = stats.levene(*group_values, center="median")
                                else:
                                    levene_stat, levene_p = np.nan, np.nan
                                equal_variances = np.isfinite(levene_p) and levene_p >= alpha
                                diagnostics = pd.DataFrame(
                                    [
                                        {
                                            "supuesto": "Normalidad de residuos (Shapiro-Wilk)",
                                            "estadistico": shapiro_stat,
                                            "p_valor": shapiro_p,
                                            "resultado": normality_result,
                                        },
                                        {
                                            "supuesto": "Homogeneidad entre combinaciones (Levene)",
                                            "estadistico": float(levene_stat),
                                            "p_valor": float(levene_p),
                                            "resultado": (
                                                "Compatible con varianzas iguales"
                                                if equal_variances
                                                else "Posible heterogeneidad de varianzas"
                                                if np.isfinite(levene_p)
                                                else "No calculada"
                                            ),
                                        },
                                        {
                                            "supuesto": "Independencia de observaciones",
                                            "estadistico": np.nan,
                                            "p_valor": np.nan,
                                            "resultado": "No verificable estadísticamente; confirmar según el diseño de muestreo.",
                                        },
                                    ]
                                )
                                st.subheader("Diagnóstico de supuestos")
                                st.dataframe(diagnostics, use_container_width=True)
                                if not equal_variances:
                                    st.warning(
                                        "La prueba clásica F puede ser sensible a la heterogeneidad de varianzas. "
                                        "Interpreta los efectos factoriales con cautela."
                                    )
                                if not np.isfinite(shapiro_p):
                                    st.caption(
                                        "Shapiro-Wilk no se calculó porque los residuos no cumplen "
                                        "el rango de tamaño o variación requerido."
                                    )
                                elif shapiro_p < alpha:
                                    st.warning(
                                        "Shapiro-Wilk detecta una posible desviación de normalidad en los residuos."
                                    )

                                st.subheader("Comparaciones post hoc entre combinaciones")
                                if len(valid_groups) > 100:
                                    st.info(
                                        "Se omitieron las comparaciones: hay más de 100 combinaciones de factores."
                                    )
                                else:
                                    method = "Tukey HSD" if equal_variances else "Games-Howell"
                                    st.caption(
                                        f"Método: {method}, con ajuste por comparaciones múltiples "
                                        f"(α = {alpha:g})."
                                    )
                                    pairwise_rows = []
                                    group_count = len(valid_groups)
                                    for first, second in combinations(range(group_count), 2):
                                        values_a = group_values[first]
                                        values_b = group_values[second]
                                        difference = float(np.mean(values_a) - np.mean(values_b))
                                        if equal_variances:
                                            variance_term = error_mean_square * (
                                                1 / len(values_a) + 1 / len(values_b)
                                            )
                                            degrees_freedom = error_df
                                        else:
                                            variance_a = float(np.var(values_a, ddof=1) / len(values_a))
                                            variance_b = float(np.var(values_b, ddof=1) / len(values_b))
                                            variance_term = variance_a + variance_b
                                            degrees_freedom = (
                                                variance_term**2
                                                / (
                                                    variance_a**2 / (len(values_a) - 1)
                                                    + variance_b**2 / (len(values_b) - 1)
                                                )
                                                if variance_term > 0
                                                else np.nan
                                            )
                                        if variance_term <= 0 or not np.isfinite(degrees_freedom):
                                            adjusted_p = margin = np.nan
                                        else:
                                            standard_error = np.sqrt(variance_term / 2)
                                            q_statistic = abs(difference) / standard_error
                                            adjusted_p = float(
                                                stats.studentized_range.sf(
                                                    q_statistic, group_count, degrees_freedom
                                                )
                                            )
                                            critical_q = float(
                                                stats.studentized_range.ppf(
                                                    1 - alpha, group_count, degrees_freedom
                                                )
                                            )
                                            margin = critical_q * standard_error
                                        pairwise_rows.append(
                                            {
                                                "combinacion_1": group_labels[first],
                                                "combinacion_2": group_labels[second],
                                                "diferencia_medias": difference,
                                                "IC_inferior": difference - margin,
                                                "IC_superior": difference + margin,
                                                "p_valor_ajustado": adjusted_p,
                                                "significativo": bool(
                                                    np.isfinite(adjusted_p) and adjusted_p < alpha
                                                ),
                                            }
                                        )
                                    pairwise = pd.DataFrame(pairwise_rows).sort_values(
                                        "p_valor_ajustado", na_position="last"
                                    )
                                    st.dataframe(pairwise, use_container_width=True)

elif View == "ANOVA de un factor":
    group_columns = [
        column for column in categorical_columns if 2 <= df[column].nunique(dropna=True) <= 30
    ]
    if not numeric_columns:
        st.info("Se necesita al menos una variable numérica para ANOVA.")
    elif not group_columns:
        st.info("No se encontraron variables de agrupación con entre 2 y 30 categorías.")
    elif not any(
        group_column != numeric_column
        for group_column in group_columns
        for numeric_column in numeric_columns
    ):
        st.info("ANOVA requiere una variable numérica y otra variable de agrupación distintas.")
    else:
        target_columns = [
            column
            for column in numeric_columns
            if any(group_column != column for group_column in group_columns)
        ]
        default_y = "total_neto" if "total_neto" in target_columns else next(
            (column for column in target_columns if column not in group_columns),
            target_columns[0],
        )
        Variable_y = st.sidebar.selectbox(
            "Variable numérica (Y)", target_columns, index=target_columns.index(default_y)
        )
        available_group_columns = [column for column in group_columns if column != Variable_y]
        preferred_groups = ["ANO", "estado", "marca", "almacen", "marc", "vendedor", "ciudad", "MES"]
        default_group = next(
            (column for column in preferred_groups if column in available_group_columns),
            available_group_columns[0],
        )
        Variable_x = st.sidebar.selectbox(
            "Variable de agrupación (X)",
            available_group_columns,
            index=available_group_columns.index(default_group),
        )
        alpha = st.sidebar.selectbox("Nivel de significancia (α)", [0.01, 0.05, 0.10], index=1)
        data = df[[Variable_x, Variable_y]].replace([np.inf, -np.inf], np.nan).dropna()
        grouped_data = list(data.groupby(Variable_x, observed=True)[Variable_y])
        valid_groups = [(label, values.to_numpy(dtype=float)) for label, values in grouped_data if len(values) >= 2]
        singleton_groups = sum(len(values) == 1 for _, values in grouped_data)

        if len(valid_groups) < 2:
            st.info("Se necesitan al menos dos grupos con dos observaciones válidas cada uno.")
        else:
            labels = [label for label, _ in valid_groups]
            groups = [values for _, values in valid_groups]
            analysis_data = data[data[Variable_x].isin(labels)].copy()
            group_stats = [
                {
                    "grupo": label,
                    "n": len(values),
                    "media": float(np.mean(values)),
                    "desviacion_estandar": float(np.std(values, ddof=1)),
                    "mediana": float(np.median(values)),
                    "minimo": float(np.min(values)),
                    "Q1": float(np.percentile(values, 25)),
                    "Q3": float(np.percentile(values, 75)),
                    "maximo": float(np.max(values)),
                    "error_estandar": float(stats.sem(values)),
                    "IC_media_inferior": float(
                        np.mean(values) - stats.t.ppf(1 - alpha / 2, len(values) - 1) * stats.sem(values)
                    ),
                    "IC_media_superior": float(
                        np.mean(values) + stats.t.ppf(1 - alpha / 2, len(values) - 1) * stats.sem(values)
                    ),
                }
                for label, values in valid_groups
            ]
            summary = pd.DataFrame(group_stats)

            st.subheader("Distribución y estadística descriptiva")
            boxplot = px.box(
                analysis_data,
                x=Variable_x,
                y=Variable_y,
                points="outliers",
                title=f"Distribución de {Variable_y} por {Variable_x}",
                labels={Variable_x: "Grupo", Variable_y: Variable_y},
            )
            st.plotly_chart(boxplot, use_container_width=True)
            st.dataframe(summary, use_container_width=True)

            if singleton_groups:
                st.caption(
                    f"{singleton_groups} grupo(s) con una sola observación se excluyeron de las pruebas "
                    "y de la gráfica; se requieren al menos dos datos por grupo."
                )
            st.caption(
                f"Observaciones analizadas: {len(analysis_data):,}. "
                f"Registros sin grupo o sin valor numérico excluidos: {len(df) - len(data):,}."
            )

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
            if len(residuals) >= 3 and len(residuals) <= 5000 and np.unique(residuals).size > 1:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
            else:
                shapiro_stat, shapiro_p = float("nan"), float("nan")
            levene_stat, levene_p = stats.levene(*groups, center="median")
            welch_result = stats.f_oneway(*groups, equal_var=False)
            welch_f, welch_p = float(welch_result.statistic), float(welch_result.pvalue)

            equal_variances = np.isfinite(levene_p) and levene_p >= alpha
            selected_test = "ANOVA clásico" if equal_variances else "ANOVA de Welch"
            selected_f = f_classic if equal_variances else welch_f
            selected_p = p_classic if equal_variances else welch_p

            st.subheader("Prueba de hipótesis")
            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric(f"p-valor ({selected_test})", f"{selected_p:.4g}")
            metric_b.metric("Estadístico F", f"{selected_f:.4f}")
            eta_squared = ss_between / ss_total if ss_total > 0 else float("nan")
            omega_squared = (
                max(0.0, (ss_between - df_between * ms_within) / (ss_total + ms_within))
                if ss_total + ms_within > 0
                else float("nan")
            )
            metric_c.metric("η² (eta cuadrado)", f"{eta_squared:.4f}")
            st.write(
                f"**H₀:** todas las medias de los grupos son iguales. "
                f"**H₁:** al menos una media es diferente. Con α = {alpha:g}, "
                + (
                    "se rechaza H₀: hay evidencia de diferencias entre al menos dos medias."
                    if np.isfinite(selected_p) and selected_p < alpha
                    else "no se rechaza H₀: no hay evidencia suficiente de diferencias entre las medias."
                    if np.isfinite(selected_p)
                    else "no se pudo determinar el resultado porque la varianza residual es cero."
                )
            )
            st.caption(
                f"Tamaño del efecto: ω² = {omega_squared:.4f}. "
                "η² y ω² estiman la proporción de variabilidad asociada al factor."
            )

            anova_table = pd.DataFrame(
                [
                    {
                        "fuente": "Entre grupos",
                        "suma_cuadrados": ss_between,
                        "gl": df_between,
                        "cuadrado_medio": ms_between,
                        "F": f_classic,
                        "p_valor": p_classic,
                    },
                    {
                        "fuente": "Dentro de grupos (error)",
                        "suma_cuadrados": ss_within,
                        "gl": df_within,
                        "cuadrado_medio": ms_within,
                        "F": np.nan,
                        "p_valor": np.nan,
                    },
                    {
                        "fuente": "Total",
                        "suma_cuadrados": ss_total,
                        "gl": n_total - 1,
                        "cuadrado_medio": np.nan,
                        "F": np.nan,
                        "p_valor": np.nan,
                    },
                ]
            )
            st.subheader("Tabla ANOVA (clásico)")
            st.dataframe(anova_table, use_container_width=True)

            st.subheader("Diagnóstico de supuestos")
            diagnostics = pd.DataFrame(
                [
                    {
                        "supuesto": "Normalidad de residuos (Shapiro-Wilk)",
                        "estadistico": shapiro_stat,
                        "p_valor": shapiro_p,
                        "resultado": (
                            "Compatible con normalidad"
                            if np.isfinite(shapiro_p) and shapiro_p >= alpha
                            else "Posible desviación de normalidad"
                            if np.isfinite(shapiro_p)
                            else "No calculada (requiere 3–5000 residuos no constantes)"
                        ),
                    },
                    {
                        "supuesto": "Homogeneidad de varianzas (Levene)",
                        "estadistico": float(levene_stat),
                        "p_valor": float(levene_p),
                        "resultado": (
                            "Compatible con varianzas iguales"
                            if np.isfinite(levene_p) and levene_p >= alpha
                            else "Posible heterogeneidad; se usa Welch"
                        ),
                    },
                    {
                        "supuesto": "Independencia de las observaciones",
                        "estadistico": np.nan,
                        "p_valor": np.nan,
                        "resultado": "No verificable estadísticamente; confirmar según el diseño de muestreo.",
                    },
                ]
            )
            st.dataframe(diagnostics, use_container_width=True)
            if equal_variances:
                st.metric("p-valor ANOVA de Welch (referencia)", f"{welch_p:.4g}")
            if not np.isfinite(shapiro_p):
                st.caption("Shapiro-Wilk no se ejecuta con más de 5000 residuos; por eso se reporta como no calculada.")
            elif shapiro_p < alpha:
                st.warning(
                    "Shapiro-Wilk detecta una posible desviación de normalidad en los residuos; "
                    "interpreta los resultados considerando este supuesto."
                )

            st.subheader("Comparaciones post hoc")
            if equal_variances:
                posthoc_method = "Tukey HSD"
            else:
                posthoc_method = "Games-Howell"
            st.caption(f"Método aplicado: {posthoc_method} con ajuste por comparaciones múltiples.")
            pairwise_rows = []
            for first in range(k):
                for second in range(first + 1, k):
                    values_a, values_b = groups[first], groups[second]
                    mean_difference = float(np.mean(values_a) - np.mean(values_b))
                    variance_term = (
                        ms_within * (1 / len(values_a) + 1 / len(values_b))
                        if equal_variances
                        else float(np.var(values_a, ddof=1) / len(values_a) + np.var(values_b, ddof=1) / len(values_b))
                    )
                    if variance_term <= 0:
                        pairwise_rows.append(
                            {
                                "grupo_1": labels[first],
                                "grupo_2": labels[second],
                                "diferencia_medias": mean_difference,
                                "IC_inferior": np.nan,
                                "IC_superior": np.nan,
                                "p_valor_ajustado": np.nan,
                                "significativo": False,
                            }
                        )
                        continue

                    standard_error = np.sqrt(variance_term / 2)
                    if equal_variances:
                        pair_df = df_within
                    else:
                        var_a = float(np.var(values_a, ddof=1) / len(values_a))
                        var_b = float(np.var(values_b, ddof=1) / len(values_b))
                        pair_df = (var_a + var_b) ** 2 / (
                            var_a**2 / (len(values_a) - 1) + var_b**2 / (len(values_b) - 1)
                        )
                    q_stat = abs(mean_difference) / standard_error
                    adjusted_p = float(stats.studentized_range.sf(q_stat, k, pair_df))
                    q_critical = float(stats.studentized_range.ppf(1 - alpha, k, pair_df))
                    margin = q_critical * standard_error
                    pairwise_rows.append(
                        {
                            "grupo_1": labels[first],
                            "grupo_2": labels[second],
                            "diferencia_medias": mean_difference,
                            "IC_inferior": mean_difference - margin,
                            "IC_superior": mean_difference + margin,
                            "p_valor_ajustado": adjusted_p,
                            "significativo": adjusted_p < alpha,
                        }
                    )
            pairwise = pd.DataFrame(pairwise_rows).sort_values(
                "p_valor_ajustado", na_position="last"
            )
            st.dataframe(pairwise, use_container_width=True)

with st.expander("Vista previa de los datos"):
    st.write(f"Filas: {len(df):,} | Columnas: {len(df.columns)}")
    st.dataframe(df.head(100), use_container_width=True)

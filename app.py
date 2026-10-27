from pathlib import Path

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


@st.cache_data
def load_data(sheet_name):
    """Read a worksheet and normalize its date and inventory-value columns."""
    df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)
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

sheet_names = pd.ExcelFile(EXCEL_FILE).sheet_names
if not sheet_names:
    st.error("El archivo Excel no contiene hojas para analizar.")
    st.stop()

preferred_sheet = "Histórico_Ventas"
default_sheet_index = sheet_names.index(preferred_sheet) if preferred_sheet in sheet_names else 0
selected_sheet = st.sidebar.selectbox("Hoja del archivo Excel", sheet_names, index=default_sheet_index)
df = load_data(selected_sheet)

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
    "ANOVA",
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

elif View == "ANOVA":
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

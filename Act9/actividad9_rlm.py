"""
Actividad 9 - Regresión Logística Multinomial
Socio formador: CLV

Archivos de entrada (en la misma carpeta que este script):
    - Datos_CLV_2026_Tec_v2.xlsx
    - Datos_Complementarios_CLV.xlsx

NO se usan las hojas de Ventas ni de Existencias.
Hojas utilizadas: Negocios_CRM, Histórico_Compras, Clasificación_Articulos,
Salidas_Almacen y Metas.

Modelos (5 variables categóricas objetivo):
    1. Resultado del negocio CRM      -> Ganado / Perdido / En proceso
    2. Valor del negocio CRM          -> Bajo / Medio / Alto
    3. Precio unitario de compra      -> Bajo / Medio / Alto
    4. Rotación de salidas de almacén -> Baja / Media / Alta
    5. Meta mensual por estado        -> Baja / Media / Alta

Para cada modelo: matriz de confusión, exactitud, precisión y sensibilidad.

Uso:
    python3 actividad9_rlm.py
Los resultados se guardan en la carpeta resultados/.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_PRINCIPAL = BASE_DIR / "Datos_CLV_2026_Tec_v2.xlsx"
ARCHIVO_COMPLEMENTARIO = BASE_DIR / "Datos_Complementarios_CLV.xlsx"
RESULTADOS_DIR = BASE_DIR / "resultados"

TEST_SIZE = 0.30
RANDOM_STATE = 42
UMBRAL_NULOS = 0.50      # columnas con más de 50 % de nulos se eliminan
MAX_CATEGORIAS = 40      # categorías máximas por variable de texto (el resto -> "Otros")

LOG = []


def registrar(texto=""):
    print(texto)
    LOG.append(str(texto))


# =============================================================================
# 1. CARGA
# =============================================================================
def cargar_hojas():
    if not ARCHIVO_PRINCIPAL.exists():
        raise FileNotFoundError(f"No se encontró {ARCHIVO_PRINCIPAL.name} junto al script")

    hojas = ["Negocios_CRM", "Histórico_Compras", "Clasificación_Articulos",
             "Salidas_Almacen", "Metas"]
    registrar("Leyendo Excel (puede tardar unos segundos)...")
    datos = pd.read_excel(ARCHIVO_PRINCIPAL, sheet_name=hojas)

    # Las hojas de Ventas y Existencias quedan fuera a propósito
    for nombre, df in datos.items():
        df.columns = [str(c).strip() for c in df.columns]
        df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")],
                inplace=True)
        registrar(f"  {nombre}: {df.shape[0]} filas x {df.shape[1]} columnas")
    return datos


# =============================================================================
# 2. PREPROCESAMIENTO GENERAL
# =============================================================================
def a_numero(serie):
    return pd.to_numeric(serie.astype(str).str.replace(r"[\$,\s]", "", regex=True),
                         errors="coerce")


def a_fecha(serie):
    return pd.to_datetime(serie, errors="coerce", utc=True)


def limpiar_texto(serie):
    s = serie.astype(str).str.strip().str.title()
    return s.replace({"Nan": np.nan, "None": np.nan, "": np.nan, "0": np.nan})


def agrupar_raras(serie, max_cat=MAX_CATEGORIAS):
    top = serie.value_counts().index[:max_cat]
    return serie.where(serie.isin(top), "Otros")


def tratar_nulos(df, nombre):
    nulos = df.isna().mean()
    if (nulos > 0).any():
        registrar(f"  Nulos (%):\n{(nulos[nulos > 0] * 100).round(2).to_string()}")
    else:
        registrar("  Sin valores nulos")
    eliminar = nulos[nulos > UMBRAL_NULOS].index.tolist()
    if eliminar:
        df = df.drop(columns=eliminar)
        registrar(f"  Columnas eliminadas (>{UMBRAL_NULOS:.0%} nulos): {eliminar}")
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            moda = df[c].mode()
            df[c] = df[c].fillna(moda.iloc[0] if not moda.empty else "Desconocido")
    registrar("  Imputación: mediana (numéricas) y moda (categóricas)")
    return df


def tratar_outliers(df, columnas):
    """Recorta valores fuera de [Q1 - 1.5 IQR, Q3 + 1.5 IQR]."""
    resumen = {}
    for c in columnas:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        li, ls = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((df[c] < li) | (df[c] > ls)).sum())
        if n:
            resumen[c] = n
            df[c] = df[c].clip(li, ls)
    registrar("  Outliers recortados (IQR): " + (str(resumen) if resumen else "ninguno"))
    return df


def transformar_log(df, columnas):
    """log1p para variables positivas con sesgo alto."""
    aplicadas = []
    for c in columnas:
        if df[c].min() >= 0 and abs(df[c].skew()) > 1:
            df[c] = np.log1p(df[c])
            aplicadas.append(c)
    registrar("  Transformación log1p: " + (str(aplicadas) if aplicadas else "ninguna"))
    return df


def encabezado(titulo):
    registrar("\n" + "=" * 70)
    registrar(titulo)
    registrar("=" * 70)


# =============================================================================
# 3. PREPARACIÓN DE CADA CONJUNTO + CATEGORIZACIÓN
# =============================================================================
def preparar_crm(df):
    encabezado("PREPROCESAMIENTO: Negocios_CRM")
    df = df.drop_duplicates(subset="id_negocio").copy()

    # Tipos
    for c in ["numero_actividades", "valor_negocio", "valor_previsto"]:
        df[c] = a_numero(df[c])
    fc = a_fecha(df["fecha_creacion"])
    df["mes_creacion"] = fc.dt.month
    df["dia_semana_creacion"] = fc.dt.dayofweek
    df["anio_creacion"] = fc.dt.year

    for c in ["estado", "tipo_negocio", "nombre_zona", "nombre_pipeline",
              "nombre_etapa", "vendedor"]:
        df[c] = limpiar_texto(df[c])

    # Corrección de valores mal capturados
    df.loc[~df["tipo_negocio"].isin(["Cliente Existente", "Cliente Nuevo"]),
           "tipo_negocio"] = np.nan
    df.loc[~df["nombre_zona"].astype(str).str.startswith("Zona"), "nombre_zona"] = np.nan
    df["nombre_pipeline"] = agrupar_raras(df["nombre_pipeline"].fillna("Otros"), 4)

    # Variable categórica 1: resultado del negocio
    etapa = df["nombre_etapa"].fillna("")
    df["resultado_negocio"] = np.select(
        [etapa.str.contains("Ganado"), etapa.str.contains("Perdido")],
        ["Ganado", "Perdido"], default="En Proceso")

    # Se descartan columnas que son identificadores, texto libre
    # o que revelan el resultado (fuga de información)
    df = df[["estado", "tipo_negocio", "nombre_zona", "nombre_pipeline", "vendedor",
             "numero_actividades", "valor_negocio", "mes_creacion",
             "dia_semana_creacion", "anio_creacion", "resultado_negocio"]]
    df = tratar_nulos(df, "CRM")
    df["estado"] = agrupar_raras(df["estado"], 20)
    df["vendedor"] = agrupar_raras(df["vendedor"], 15)
    df = tratar_outliers(df, ["numero_actividades", "valor_negocio"])

    # Variable categórica 2: nivel de valor del negocio (terciles)
    df["nivel_valor_negocio"] = categorizar_terciles(df["valor_negocio"],
                                                     ["Bajo", "Medio", "Alto"])
    registrar(f"  resultado_negocio: {df['resultado_negocio'].value_counts().to_dict()}")
    registrar(f"  nivel_valor_negocio: {df['nivel_valor_negocio'].value_counts().to_dict()}")
    return df


def categoria_principal(clasif):
    """Primera categoría registrada para cada artículo."""
    clasif = clasif.copy()
    clasif["categoria"] = limpiar_texto(clasif["categoria"])
    return (clasif.dropna(subset=["categoria"])
            .drop_duplicates(subset="articulo_id")[["articulo_id", "categoria"]])


def preparar_compras(df, cat):
    encabezado("PREPROCESAMIENTO: Histórico_Compras + Clasificación_Articulos")
    n0 = len(df)
    df = df.drop_duplicates().copy()
    registrar(f"  Duplicados eliminados: {n0 - len(df)}")
    df["unidades"] = a_numero(df["unidades"])
    df["precio_unitario"] = a_numero(df["precio_unitario"])
    f = a_fecha(df["fecha"])
    df["anio"] = f.dt.year
    df["mes"] = f.dt.month
    df = df.merge(cat, on="articulo_id", how="left")
    df["categoria"] = agrupar_raras(df["categoria"].fillna("Sin Clasificar"), 25)

    df = df[["unidades", "anio", "mes", "categoria", "precio_unitario"]]
    df = tratar_nulos(df, "Compras")
    # Variable categórica 3: rango de precio unitario (antes de recortar outliers)
    df["rango_precio_compra"] = categorizar_terciles(df["precio_unitario"],
                                                     ["Bajo", "Medio", "Alto"])
    df = tratar_outliers(df, ["unidades"])
    registrar(f"  rango_precio_compra: {df['rango_precio_compra'].value_counts().to_dict()}")
    return df.drop(columns="precio_unitario")


def preparar_salidas(df, cat):
    encabezado("PREPROCESAMIENTO: Salidas_Almacen + Clasificación_Articulos")
    n0 = len(df)
    df = df.drop_duplicates().copy()
    registrar(f"  Duplicados eliminados: {n0 - len(df)}")
    df["salidas"] = a_numero(df["salidas"])
    df["marca"] = agrupar_raras(limpiar_texto(df["marca"]), 20)
    df = df.merge(cat, on="articulo_id", how="left")
    df["categoria"] = agrupar_raras(df["categoria"].fillna("Sin Clasificar"), 25)
    df = df[["anio", "mes", "marca", "categoria", "salidas"]]
    df = tratar_nulos(df, "Salidas")

    # Variable categórica 4: rotación (rangos definidos por negocio)
    df["rotacion"] = pd.cut(df["salidas"], bins=[0, 2, 10, np.inf],
                            labels=["Baja", "Media", "Alta"]).astype(str)
    registrar("  Rotación: Baja (1-2 uds), Media (3-10 uds), Alta (>10 uds)")
    registrar(f"  rotacion: {df['rotacion'].value_counts().to_dict()}")
    return df.drop(columns="salidas")


def preparar_metas(df):
    encabezado("PREPROCESAMIENTO: Metas")
    df = df.drop_duplicates(subset="meta_id").copy()
    df["meta"] = a_numero(df["meta"])
    df["estado"] = limpiar_texto(df["estado"])
    df = df[["estado", "mes", "anio", "meta"]]
    df = tratar_nulos(df, "Metas")
    sin_meta = int((df["meta"] == 0).sum())
    df = df[df["meta"] > 0]
    registrar(f"  Registros con meta = 0 eliminados: {sin_meta}")
    df = tratar_outliers(df, ["meta"])
    # Variable categórica 5: nivel de meta (terciles)
    df["nivel_meta"] = categorizar_terciles(df["meta"], ["Baja", "Media", "Alta"])
    registrar(f"  nivel_meta: {df['nivel_meta'].value_counts().to_dict()}")
    return df.drop(columns="meta")


def categorizar_terciles(serie, etiquetas):
    try:
        return pd.qcut(serie, q=3, labels=etiquetas).astype(str)
    except ValueError:  # muchos valores repetidos
        return pd.qcut(serie.rank(method="first"), q=3, labels=etiquetas).astype(str)


# =============================================================================
# 4. REGRESIÓN LOGÍSTICA MULTINOMIAL
# =============================================================================
def modelo_rlm(n, df, objetivo, quitar, orden_clases, titulo):
    encabezado(f"MODELO {n}: {titulo}")
    y = df[objetivo].astype(str)
    X = df.drop(columns=[objetivo] + quitar)

    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    X = transformar_log(X.copy(), [c for c in num if X[c].nunique() > 12])

    registrar(f"  Clases: {y.value_counts().to_dict()}")
    registrar(f"  Predictores numéricos: {num}")
    registrar(f"  Predictores categóricos: {cat}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ])),
        # lbfgs con 3 clases => regresión logística multinomial (softmax)
        ("rlm", LogisticRegression(solver="lbfgs", max_iter=5000,
                                   class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    clases = [c for c in orden_clases if c in y.unique()]
    cm = confusion_matrix(y_te, y_pred, labels=clases)
    met = {
        "modelo": n,
        "variable_objetivo": objetivo,
        "descripcion": titulo,
        "exactitud": accuracy_score(y_te, y_pred),
        "precision_macro": precision_score(y_te, y_pred, average="macro", zero_division=0),
        "sensibilidad_macro": recall_score(y_te, y_pred, average="macro", zero_division=0),
        "precision_ponderada": precision_score(y_te, y_pred, average="weighted", zero_division=0),
        "sensibilidad_ponderada": recall_score(y_te, y_pred, average="weighted", zero_division=0),
    }

    registrar("\n  Matriz de confusión (filas = real, columnas = predicción):")
    registrar(pd.DataFrame(cm, index=clases, columns=clases).to_string())
    registrar(f"\n  Exactitud (accuracy):    {met['exactitud']:.4f}")
    registrar(f"  Precisión (macro):       {met['precision_macro']:.4f}")
    registrar(f"  Sensibilidad (macro):    {met['sensibilidad_macro']:.4f}")
    registrar("\n  Reporte por clase:")
    registrar(classification_report(y_te, y_pred, labels=clases, zero_division=0))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=clases, yticklabels=clases)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(f"Modelo {n}: {titulo}")
    plt.tight_layout()
    plt.savefig(RESULTADOS_DIR / f"matriz_confusion_{n}_{objetivo}.png", dpi=150)
    plt.close()
    return met


def grafica_resumen(res):
    d = res.set_index("variable_objetivo")[
        ["exactitud", "precision_macro", "sensibilidad_macro"]]
    d.columns = ["Exactitud", "Precisión", "Sensibilidad"]
    ax = d.plot(kind="bar", figsize=(10, 5), color=["#2a6fdb", "#e3893b", "#3aa66b"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Valor")
    ax.set_xlabel("")
    ax.set_title("Métricas por modelo - Regresión Logística Multinomial (CLV)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTADOS_DIR / "resumen_metricas.png", dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTADOS_DIR.mkdir(exist_ok=True)
    registrar("ACTIVIDAD 9 - REGRESIÓN LOGÍSTICA MULTINOMIAL (CLV)")
    datos = cargar_hojas()
    cat = categoria_principal(datos["Clasificación_Articulos"])

    crm = preparar_crm(datos["Negocios_CRM"])
    compras = preparar_compras(datos["Histórico_Compras"], cat)
    salidas = preparar_salidas(datos["Salidas_Almacen"], cat)
    metas = preparar_metas(datos["Metas"])

    for nombre, df in [("crm", crm), ("compras", compras),
                       ("salidas", salidas), ("metas", metas)]:
        df.to_csv(RESULTADOS_DIR / f"datos_limpios_{nombre}.csv", index=False)

    nivel3 = ["Bajo", "Medio", "Alto"]
    nivel3f = ["Baja", "Media", "Alta"]
    resultados = [
        modelo_rlm(1, crm, "resultado_negocio", ["nivel_valor_negocio"],
                   ["Ganado", "Perdido", "En Proceso"], "Resultado del negocio CRM"),
        modelo_rlm(2, crm, "nivel_valor_negocio", ["valor_negocio"],
                   nivel3, "Nivel de valor del negocio CRM"),
        modelo_rlm(3, compras, "rango_precio_compra", [],
                   nivel3, "Rango de precio unitario de compra"),
        modelo_rlm(4, salidas, "rotacion", [],
                   nivel3f, "Rotación de salidas de almacén"),
        modelo_rlm(5, metas, "nivel_meta", [],
                   nivel3f, "Nivel de meta mensual por estado"),
    ]

    res = pd.DataFrame(resultados)
    res.round(4).to_csv(RESULTADOS_DIR / "resumen_metricas.csv", index=False)
    grafica_resumen(res)

    encabezado("RESUMEN DE MÉTRICAS")
    registrar(res[["modelo", "variable_objetivo", "exactitud",
                   "precision_macro", "sensibilidad_macro"]].round(4).to_string(index=False))

    (RESULTADOS_DIR / "reporte.txt").write_text("\n".join(LOG), encoding="utf-8")
    print(f"\nResultados guardados en: {RESULTADOS_DIR}")


if __name__ == "__main__":
    main() 
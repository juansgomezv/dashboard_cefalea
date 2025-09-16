## POSIBLE ENTREGA CON RUIDO DEL 20%

# --- Configuración para reproducibilidad y control de hilos ---
import os
os.environ["PYTHONHASHSEED"] = str(42)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import io
import hashlib
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import sklearn, importlib.metadata
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import hdbscan

# ---------------- CONSTANTES ----------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

MIN_CLUSTER_SIZE = 20
MIN_SAMPLES = 1

# ---------------- FUNCIONES AUXILIARES ----------------
def file_md5(path):
    """MD5 de un archivo (bytes)."""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return "file-not-found"

def dataframe_hash(df: pd.DataFrame) -> str:
    """Hash estable del DataFrame: hash_pandas_object -> md5 sobre bytes.
       NOTA: incluye contenido y orden de filas/columnas (no reordena columnas)."""
    try:
        arr = pd.util.hash_pandas_object(df, index=True).values
        return hashlib.md5(arr.tobytes()).hexdigest()
    except Exception as e:
        return f"error_hash:{e}"

def md5_bytes(arr_bytes: bytes) -> str:
    return hashlib.md5(arr_bytes).hexdigest()

def download_button_bytes(data_bytes: bytes, filename: str, label: str):
    st.download_button(label, data=data_bytes, file_name=filename)

# Traducción mapeos (desde CSV)
def read_csv_with_fallback(path):
    """Intento robusto lectura de CSV (UTF-8, fallback latin-1)."""
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding="latin-1")

def cargar_mapeo_valores(path="valores.csv"):
    try:
        df = read_csv_with_fallback(path)
        df.columns = df.columns.str.strip().str.lower()
        if not {"columna", "valor", "traduccion"}.issubset(set(df.columns)):
            st.sidebar.warning(f"'{path}' no contiene columnas esperadas. Encontradas: {list(df.columns)}")
            return {}
        mapeo = {}
        for c in df["columna"].unique():
            sub = df[df["columna"] == c]
            mapeo[c] = list(zip(sub["valor"].astype(str), sub["traduccion"].astype(str)))
        return mapeo
    except FileNotFoundError:
        st.sidebar.warning(f"No se encontró '{path}'.")
        return {}
    except Exception as e:
        st.sidebar.warning(f"Error cargando '{path}': {e}")
        return {}

def cargar_nombres_columnas(path="columnas.csv"):
    try:
        df = read_csv_with_fallback(path)
        df.columns = df.columns.str.strip().str.lower()
        if not {"columna", "nombre_amigable"}.issubset(set(df.columns)):
            st.sidebar.warning(f"'{path}' no contiene columnas esperadas. Encontradas: {list(df.columns)}")
            return {}
        return dict(zip(df["columna"].astype(str), df["nombre_amigable"].astype(str)))
    except FileNotFoundError:
        st.sidebar.warning(f"No se encontró '{path}'.")
        return {}
    except Exception as e:
        st.sidebar.warning(f"Error cargando '{path}': {e}")
        return {}

def traducir_valor_aproximado(col, val, mapeo_dict):
    """Intento de traducción numérica o por string; retorna val si no hay mapeo."""
    if col not in mapeo_dict or pd.isna(val):
        return val
    try:
        val_float = float(val)
        valores = []
        trads = []
        for v, t in mapeo_dict[col]:
            try:
                valores.append(float(v))
                trads.append(t)
            except Exception:
                valores = None
                break
        if valores and len(valores) > 0:
            idx = (np.abs(np.array(valores) - val_float)).argmin()
            return trads[int(idx)]
    except Exception:
        pass
    # fallback por string exacta
    val_s = str(val)
    for v, t in mapeo_dict[col]:
        if str(v) == val_s:
            return t
    return val

# ---------------- CARGA MAPEOS ----------------
mapeo_valores = cargar_mapeo_valores("valores.csv")
nombre_columnas = cargar_nombres_columnas("columnas.csv")

# ---------------- LECTURA DATASET ----------------
# Ver hashes de archivos base
hash_dataset_file = file_md5("DatasetV4.csv")
hash_valores_file = file_md5("valores.csv")
hash_columnas_file = file_md5("columnas.csv")

try:
    df_orig = pd.read_csv("DatasetV4.csv", index_col=0)
except FileNotFoundError:
    st.error("No se encontró 'DatasetV4.csv'. Colócalo en el repo y vuelve a desplegar.")
    raise
except Exception as e:
    st.error(f"Error leyendo 'DatasetV4.csv': {e}")
    raise

# Guardar snapshot inicial
hash_df_initial = dataframe_hash(df_orig)

# ---------------- LIMPIEZA / PREPROCESSING (con snapshots) ----------------
df = df_orig.copy()

# snapshot antes de drop
hash_before_drop = dataframe_hash(df)

# eliminar columnas con > threshold de ceros o constantes
threshold = 0.85
cols_to_drop_zeros = (df == 0).sum() / len(df) > threshold
cols_to_drop_const = df.nunique() == 1
cols_to_drop = df.columns[cols_to_drop_zeros | cols_to_drop_const].tolist()
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
hash_after_drop = dataframe_hash(df)

# variables cefalea
cefalea_vars = [
    "AntecedentesFamiliares", "LugarDolor", "IntensidadDolor",
    "DuracionDolor", "FrecuenciaDolor", "ActividadFisica",
    "InasistenciaDolor", "IndiceDolor"
]
cefalea_vars_presentes = [v for v in cefalea_vars if v in df.columns]

# snapshot antes de filtrar filas cefalea
hash_before_filter = dataframe_hash(df)

# eliminar filas donde todas las variables de cefalea sean 0 (si existen)
if cefalea_vars_presentes:
    mask = ~(df[cefalea_vars_presentes] == 0).all(axis=1)
    df = df[mask]
hash_after_filter = dataframe_hash(df)

# crear copias y ponderar
df_original = df.copy()
df_weighted = df.copy()

# intentar multiplicar solo si columnas numéricas
for c in cefalea_vars_presentes:
    try:
        df_weighted[c] = pd.to_numeric(df_weighted[c], errors="coerce") * 3
    except Exception:
        # si falla, dejar como estaba
        pass
hash_after_weight = dataframe_hash(df_weighted)

# resetear índices para consistencia
df = df.reset_index(drop=True)
df_weighted = df_weighted.reset_index(drop=True)
df_original = df_original.reset_index(drop=True)

# exportar snapshot csv para descargar/comparar
csv_df_post_filter = df.to_csv(index=False).encode("utf-8")
csv_df_weighted = df_weighted.to_csv(index=False).encode("utf-8")
csv_df_limpio = df_original.to_csv(index=False).encode("utf-8")

# ---------------- ESCALADO Y UMAP ----------------
# convertir a numérico (coerce) y rellenar NaN con 0 para que scaler no falle
df_weighted_numeric = df_weighted.apply(pd.to_numeric, errors="coerce").fillna(0)

# snapshot pre-escalado
hash_before_scaling = dataframe_hash(df_weighted_numeric)

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_weighted_numeric), columns=df_weighted_numeric.columns)
hash_after_scaling = dataframe_hash(df_scaled)

# UMAP: ejecutar dos veces para comparar determinismo
umap_reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED)
X_umap_1 = umap_reducer.fit_transform(df_scaled)
hash_umap_1 = md5_bytes(X_umap_1.tobytes())

# ejecutar UMAP una segunda vez (misma instancia puede reutilizar estado; usamos nueva instancia)
umap_reducer2 = UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED)
X_umap_2 = umap_reducer2.fit_transform(df_scaled)
hash_umap_2 = md5_bytes(X_umap_2.tobytes())

# Exportar UMAPs a csv (x,y)
df_umap_1 = pd.DataFrame(X_umap_1, columns=["x", "y"])
df_umap_2 = pd.DataFrame(X_umap_2, columns=["x", "y"])
csv_umap_1 = df_umap_1.to_csv(index=False).encode("utf-8")
csv_umap_2 = df_umap_2.to_csv(index=False).encode("utf-8")

# ---------------- HDBSCAN ----------------
clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES)
cluster_labels = clusterer.fit_predict(X_umap_1)  # usar primera ejecución como principal
hash_labels = hashlib.md5(cluster_labels.tobytes()).hexdigest()

# añadir etiquetas a dataframes
df_scaled["grupo"] = cluster_labels
df_original["grupo"] = cluster_labels

# labels CSV
csv_labels = pd.DataFrame({"grupo": cluster_labels}).to_csv(index=False).encode("utf-8")

# ---------------- TABLAS Y MÉTRICAS ----------------
df_scaled_no_noise = df_scaled[df_scaled["grupo"] != -1]
df_original_no_noise = df_original[df_original["grupo"] != -1]

tabla_medias = df_scaled_no_noise.groupby("grupo").mean().round(2) if not df_scaled_no_noise.empty else pd.DataFrame()
tabla_real_round = df_original_no_noise.copy()
for col in tabla_real_round.select_dtypes(include=np.number).columns:
    tabla_real_round[col] = tabla_real_round[col].round()
tabla_original = tabla_real_round.groupby("grupo").mean().round(2) if not tabla_real_round.empty else pd.DataFrame()

total_estudiantes = len(df_original)
estudiantes_clasificados = np.sum(cluster_labels != -1)
estudiantes_ruido = np.sum(cluster_labels == -1)
n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Dashboard Diagnóstico", layout="wide")
st.title("CEFALEA — Dashboard (modo diagnóstico)")

# KPIs
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("👥 Total estudiantes", total_estudiantes)
col_b.metric("✅ Clasificados en Grupos", estudiantes_clasificados)
col_c.metric("❌ Detectados como ruido", estudiantes_ruido)
col_d.metric("🔢 Total de Grupos", n_clusters)

# Panel principal: UMAP + análisis (compactado)
col1, col2 = st.columns([1, 2])

with col1:
    # opciones para descargar snapshots
    st.header("📁 Exportar snapshots")
    download_button_bytes(csv_df_post_filter, "df_post_filter.csv", "Descargar df_post_filter.csv")
    download_button_bytes(csv_df_weighted, "df_weighted.csv", "Descargar df_weighted.csv")
    download_button_bytes(csv_df_limpio, "df_limpio.csv", "Descargar df_limpio.csv")
    download_button_bytes(csv_labels, "labels.csv", "Descargar labels.csv")
    download_button_bytes(csv_umap_1, "umap_1.csv", "Descargar umap_1.csv")
    download_button_bytes(csv_umap_2, "umap_2.csv", "Descargar umap_2.csv")

    st.markdown("---")
    st.header("🔍 Diagnóstico rápido")
    st.write("Hash archivo DatasetV4.csv:", hash_dataset_file)
    st.write("Hash valores.csv:", hash_valores_file)
    st.write("Hash columnas.csv:", hash_columnas_file)
    st.write("Hash inicial DataFrame (sin limpiar):", hash_df_initial)
    st.write("Hash antes drop columnas:", hash_before_drop)
    st.write("Hash después drop columnas:", hash_after_drop)
    st.write("Hash antes filtrado filas:", hash_before_filter)
    st.write("Hash después filtrado filas:", hash_after_filter)
    st.write("Hash después ponderado (weight):", hash_after_weight)
    st.write("Hash antes escalado:", hash_before_scaling)
    st.write("Hash después escalado:", hash_after_scaling)
    st.write("Hash UMAP (ejecución 1):", hash_umap_1)
    st.write("Hash UMAP (ejecución 2):", hash_umap_2)
    st.write("Hash labels (cluster):", hash_labels)

    st.markdown("---")
    st.header("🧾 Info del DataFrame limpio")
    st.write("Shape:", df.shape)
    st.write("Columnas:", list(df.columns))
    st.write("Dtypes:")
    st.write(df.dtypes)
    st.write("Valores faltantes por columna:")
    st.write(df.isna().sum())

with col2:
    st.markdown("### Proyección UMAP (usando UMAP ejecución 1)")
    mask_valid = cluster_labels != -1
    df_umap_vis = pd.DataFrame(X_umap_1[mask_valid], columns=["x", "y"])
    df_umap_vis["grupo"] = cluster_labels[mask_valid]

    unique_clusters = sorted(df_umap_vis["grupo"].unique()) if not df_umap_vis.empty else []
    palette = sns.color_palette("tab10", n_colors=max(len(unique_clusters), 1))
    cluster_to_color = {cid: palette[i % len(palette)] for i, cid in enumerate(unique_clusters)}

    fig, ax = plt.subplots(figsize=(7, 5))
    for cid in unique_clusters:
        sub = df_umap_vis[df_umap_vis["grupo"] == cid]
        color = cluster_to_color[cid] if cid in cluster_to_color else "#d3d3d3"
        ax.scatter(sub["x"], sub["y"], s=40, c=[color], alpha=0.8, label=("Ruido" if cid == -1 else f"Grupo {int(cid)+1}"))
    ax.set_title("UMAP projection (ejecución 1)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

    st.markdown("### Comparación UMAP ejecución 1 vs 2 (hashes arriba).")
    st.write("Si los hashes UMAP 1 y UMAP 2 **difieren**, UMAP no está siendo determinista en tu entorno (posibles causas: BLAS/OMP threads, versiones binarias, etc.).")

# pestañas para vista de tablas y comparaciones
tab1, tab2 = st.tabs(["Tabla medias (estandarizado)", "Valores reales (interpretados)"])
with tab1:
    if tabla_medias.empty:
        st.write("No hay tablas de medias (todos ruido o sin grupos).")
    else:
        st.dataframe(tabla_medias)

with tab2:
    if tabla_original.empty:
        st.write("No hay valores interpretados (todos ruido o sin grupos).")
    else:
        # traducir y forzar strings para evitar problemas de Arrow al mostrar
        translated = {}
        for col in tabla_original.columns:
            try:
                translated[col] = tabla_original[col].astype(str)
            except Exception:
                translated[col] = tabla_original[col].apply(lambda x: str(x))
        st.dataframe(tabla_original.astype(str))

# ---------------- DIAGNÓSTICO DETALLADO EN SIDEBAR ----------------
st.sidebar.header("🔧 Diagnóstico (detallado)")
try:
    st.sidebar.write(f"NumPy: {np.__version__}")
    st.sidebar.write(f"UMAP: {importlib.metadata.version('umap-learn') if 'umap-learn' in importlib.metadata.distributions() else 'umap (no version)'}")
except Exception:
    try:
        st.sidebar.write(f"UMAP: {umap.__version__}")
    except Exception:
        st.sidebar.write("UMAP: (no disponible)")
try:
    st.sidebar.write(f"HDBSCAN: {importlib.metadata.version('hdbscan')}")
except Exception:
    st.sidebar.write("HDBSCAN: (no disponible)")
st.sidebar.write(f"scikit-learn: {sklearn.__version__}")
st.sidebar.write(f"Pandas: {pd.__version__}")

st.sidebar.subheader("Hashes resumidos")
st.sidebar.write("Hash dataset file:", hash_dataset_file)
st.sidebar.write("Hash processed df:", dataframe_hash(df))
st.sidebar.write("Hash scaled df:", dataframe_hash(df_scaled))

st.sidebar.subheader("Primeras filas del df limpio")
st.sidebar.dataframe(df.head(10))

# ---------------- FIN ----------------
st.info("✅ Diagnóstico cargado. Usa los botones de descarga para comparar exactamente los CSV entre local y la app online.")



######################################################COMANDO PARA INICIAR STREAMLIT###############################################################################
# python -m streamlit run prueba1-4.py

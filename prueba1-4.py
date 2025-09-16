## POSIBLE ENTREGA CON RUIDO DEL 20%

# prueba1-4.py (corrigido y robustecido)

# --- Configuración para reproducibilidad ---
import os
os.environ["PYTHONHASHSEED"] = str(42)
# limitar hilos para reproducibilidad numérica entre entornos
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

from umap import UMAP
import hdbscan
import pandas as pd
import sklearn
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import hashlib
import importlib.metadata

# Parámetros de clustering / reducción
min_cluster_size = 20
min_samples = 1
clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

# ---------------- FUNCIONES DE TRADUCCIÓN Y AUXILIARES ----------------
def read_csv_with_fallback(path):
    """Intenta leer CSV con UTF-8; si falla, intenta latin-1."""
    try:
        return pd.read_csv(path, dtype=str)
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin-1", dtype=str)
        except Exception as e:
            raise

def cargar_mapeo_valores(path="valores.csv"):
    """Carga mapeo de valores desde valores.csv con columnas: columna, valor, traduccion"""
    try:
        df = read_csv_with_fallback(path)
        df.columns = df.columns.str.strip().str.lower()
        expected = {"columna", "valor", "traduccion"}
        if not expected.issubset(set(df.columns)):
            st.warning(f"Archivo '{path}' no contiene las columnas esperadas {expected}. Columnas encontradas: {list(df.columns)}")
            return {}
        mapeo = {}
        # mantener tipos: valor puede ser numérico o texto
        for columna in df["columna"].unique():
            sub = df[df["columna"] == columna]
            # conservar orden y tipos como strings
            pares = list(zip(sub["valor"].astype(str), sub["traduccion"].astype(str)))
            # intentar convertir valores numéricos a float para búsqueda aproximada en traducir_valor_aproximado
            mapeo[columna] = pares
        return mapeo
    except FileNotFoundError:
        st.warning(f"No se encontró '{path}'. Se omitirán traducciones de valores.")
        return {}
    except Exception as e:
        st.warning(f"No se pudo cargar '{path}'. Error: {e}")
        return {}

def cargar_nombres_columnas(path="columnas.csv"):
    """Carga nombres amigables desde columnas.csv con columnas: columna, nombre_amigable"""
    try:
        df = read_csv_with_fallback(path)
        df.columns = df.columns.str.strip().str.lower()
        expected = {"columna", "nombre_amigable"}
        if not expected.issubset(set(df.columns)):
            st.warning(f"Archivo '{path}' no contiene las columnas esperadas {expected}. Columnas encontradas: {list(df.columns)}")
            return {}
        return dict(zip(df["columna"].astype(str), df["nombre_amigable"].astype(str)))
    except FileNotFoundError:
        st.warning(f"No se encontró '{path}'. Se usarán nombres de columna originales.")
        return {}
    except Exception as e:
        st.warning(f"No se pudo cargar '{path}'. Error: {e}")
        return {}

def traducir_valor_aproximado(col, val, mapeo_dict):
    """
    Traduce un valor aproximado:
    - Si el mapeo para la columna existe y los valores mapeados son numéricos, busca el valor numérico más cercano.
    - Si los valores mapeados son strings, busca coincidencia exacta (comparando como strings).
    """
    if col not in mapeo_dict or pd.isna(val):
        return val
    try:
        # val puede venir como np.int/float o string; tratamos de convertir a float
        val_float = float(val)
        # intentar obtener lista de valores numéricos desde los mapeos (si todos convertibles)
        valores = []
        trads = []
        for v, t in mapeo_dict[col]:
            try:
                valores.append(float(v))
                trads.append(t)
            except Exception:
                # si alguno no es convertible, abortar y trabajar por strings
                valores = None
                break
        if valores is not None and len(valores) > 0:
            arr = np.array(valores, dtype=float)
            idx = (np.abs(arr - val_float)).argmin()
            return trads[idx]
        # si no podemos trabajar numéricamente, buscar por string exacta
        val_str = str(val)
        for v, t in mapeo_dict[col]:
            if str(v) == val_str:
                return t
        return val
    except Exception:
        # fallback: comparar como string
        val_str = str(val)
        for v, t in mapeo_dict[col]:
            if str(v) == val_str:
                return t
        return val

def dataframe_hash(df):
    """Hash estable del DataFrame (incluye índices y orden)"""
    try:
        # normalizar: ordenar columnas para que el hash sea consistente si el orden de columnas es distinto
        df_copy = df.copy()
        # convertimos a bytes de hash_pandas_object
        arr = pd.util.hash_pandas_object(df_copy, index=True).values
        return hashlib.md5(arr.tobytes()).hexdigest()
    except Exception as e:
        return f"error_hash:{e}"

# ---------------- CARGA DE MAPEOS (CSV) ----------------
mapeo_valores = cargar_mapeo_valores("valores.csv")
nombre_columnas = cargar_nombres_columnas("columnas.csv")

# ---------------- CARGAR Y PREPARAR DATOS ----------------
# Leer Dataset principal - se asume DatasetV4.csv en repo
try:
    df = pd.read_csv("DatasetV4.csv", index_col=0)
except FileNotFoundError:
    st.error("No se encontró 'DatasetV4.csv' en el directorio del app. Coloca el archivo en el repo.")
    raise
except Exception as e:
    st.error(f"Error leyendo 'DatasetV4.csv': {e}")
    raise

# eliminar columnas con demasiados ceros o constantes
threshold = 0.85
cols_to_drop_zeros = (df == 0).sum() / len(df) > threshold
cols_to_drop_const = df.nunique() == 1
cols_to_drop = df.columns[cols_to_drop_zeros | cols_to_drop_const]
if len(cols_to_drop) > 0:
    df = df.drop(columns=cols_to_drop)

# variables de cefalea
cefalea_vars = [
    "AntecedentesFamiliares", "LugarDolor", "IntensidadDolor",
    "DuracionDolor", "FrecuenciaDolor", "ActividadFisica",
    "InasistenciaDolor", "IndiceDolor"
]
cefalea_vars_presentes = [v for v in cefalea_vars if v in df.columns]
if len(cefalea_vars_presentes) > 0:
    # eliminar filas donde todas las variables de cefalea sean 0 (si existen)
    mask = ~(df[cefalea_vars_presentes] == 0).all(axis=1)
    df = df[mask]

# copias
df_original = df.copy()
df_weighted = df.copy()
if len(cefalea_vars_presentes) > 0:
    # multiplicar por 3 (si columnas presentes)
    # proteger si columnas no numéricas: forzar a numeric donde corresponda
    for c in cefalea_vars_presentes:
        try:
            df_weighted[c] = pd.to_numeric(df_weighted[c], errors="coerce") * 3
        except Exception:
            pass

# reset índices para consistencia
df = df.reset_index(drop=True)
df_weighted = df_weighted.reset_index(drop=True)
df_original = df_original.reset_index(drop=True)

# ---------------- PREPROCESSING, UMAP Y HDBSCAN ----------------
scaler = StandardScaler()
# Necesitamos convertir a numérico solo las columnas que son numéricas; si hay mix, coerción a NaN
numeric_cols = df_weighted.select_dtypes(include=[np.number]).columns.tolist()
# Si hay columnas no numéricas (por ejemplo, por mapeos), intentamos forzar
if len(numeric_cols) < df_weighted.shape[1]:
    # intentar convertir todas las columnas a numérico si corresponde
    df_weighted_numeric = df_weighted.apply(pd.to_numeric, errors="coerce")
else:
    df_weighted_numeric = df_weighted.copy()

# Rellenar NaNs por 0 antes de escalar (o puedes elegir otra estrategia)
df_weighted_numeric = df_weighted_numeric.fillna(0)

# Escalado
df_scaled = pd.DataFrame(scaler.fit_transform(df_weighted_numeric), columns=df_weighted_numeric.columns)

# UMAP
umap_reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED)
X_umap = umap_reducer.fit_transform(df_scaled)

# HDBSCAN clustering
cluster_labels = clusterer.fit_predict(X_umap)

df_scaled["grupo"] = cluster_labels
df_original["grupo"] = cluster_labels

# DataFrames sin ruido
df_scaled_no_noise = df_scaled[df_scaled["grupo"] != -1]
df_original_no_noise = df_original[df_original["grupo"] != -1]

# tablas de medias
if not df_scaled_no_noise.empty:
    tabla_medias = df_scaled_no_noise.groupby("grupo").mean().round(2)
else:
    tabla_medias = pd.DataFrame()

tabla_real_round = df_original_no_noise.copy()
# redondear solo columnas numéricas
for col in tabla_real_round.select_dtypes(include=np.number).columns:
    tabla_real_round[col] = tabla_real_round[col].round()
if not tabla_real_round.empty:
    tabla_original = tabla_real_round.groupby("grupo").mean().round(2)
else:
    tabla_original = pd.DataFrame()

# ---------------- ESTADÍSTICAS ----------------
total_estudiantes = len(df_original)
estudiantes_clasificados = np.sum(cluster_labels != -1)
estudiantes_ruido = np.sum(cluster_labels == -1)
n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))

# ---------------- STREAMLIT DASHBOARD ----------------
st.set_page_config(page_title="Dashboard", layout="wide")
st.title("CEFALEA EN LOS ESTUDIANTES UPB BUCARAMANGA")

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("👥 Total estudiantes", total_estudiantes)
col_b.metric("✅ Clasificados en Grupos", estudiantes_clasificados)
col_c.metric("❌ Detectados como ruido", estudiantes_ruido)
col_d.metric("🔢 Total de Grupos", n_clusters)

col1, col2 = st.columns([1, 2])

with col1:
    cluster_opciones = {}
    # Si tabla_medias está vacía (p. ej. todos ruido), no habrá grupos
    if tabla_medias.empty:
        # sólo mostrar "Ruido" si existe ruido
        if estudiantes_ruido > 0:
            cluster_opciones["Ruido"] = -1
        else:
            cluster_opciones["Sin grupos"] = -999
    else:
        for i in sorted(tabla_medias.index):
            cluster_opciones[f"Grupo {int(i) + 1}"] = i
        # añadir opción Ruido si existe
        if estudiantes_ruido > 0:
            cluster_opciones["Ruido"] = -1

    cluster_seleccionado = st.selectbox("Selecciona un Grupo", list(cluster_opciones.keys()))
    cluster_id = cluster_opciones[cluster_seleccionado]

    tamano_cluster_actual = np.sum(df_scaled["grupo"] == cluster_id)
    st.info(f"El grupo **{cluster_seleccionado}** contiene **{tamano_cluster_actual} estudiantes**.")

    st.markdown("### Proyección UMAP")
    mask_valid = cluster_labels != -1
    X_valid = X_umap[mask_valid]
    labels_valid = cluster_labels[mask_valid]

    df_umap = pd.DataFrame(X_valid, columns=["x", "y"])
    df_umap["grupo"] = labels_valid

    unique_clusters = sorted(df_umap["grupo"].unique()) if not df_umap.empty else []
    if len(unique_clusters) > 0:
        palette = sns.color_palette("tab10", n_colors=max(len(unique_clusters), 1))
        cluster_to_color = {cid: palette[i % len(palette)] for i, cid in enumerate(unique_clusters)}
    else:
        cluster_to_color = {}

    fig_umap, ax_umap = plt.subplots(figsize=(6, 5))
    for cid in unique_clusters:
        cluster_data = df_umap[df_umap["grupo"] == cid]
        color = cluster_to_color.get(cid, "#d3d3d3")
        color = color if cid == cluster_id else "#d3d3d3"
        size = 60 if cid == cluster_id else 30
        alpha = 1.0 if cid == cluster_id else 0.3
        ax_umap.scatter(cluster_data["x"], cluster_data["y"], s=size, c=[color], alpha=alpha)
        if len(cluster_data) >= 10:
            sns.kdeplot(
                x=cluster_data["x"], y=cluster_data["y"],
                ax=ax_umap, levels=3, color=color, linewidths=2, alpha=0.5
            )
    ax_umap.set_title("Proyección UMAP")
    ax_umap.set_xlabel("Componente 1")
    ax_umap.set_ylabel("Componente 2")
    st.pyplot(fig_umap)

with col2:
    tab1, tab2 = st.tabs(["Visualización Estandarizada", "Interpretación Real"])
    with tab1:
        col_izq, col_der = st.columns(2)
        with col_izq:
            st.subheader("Los 10 más Impactantes (estandarizado)")
            if tabla_medias.empty or cluster_id not in tabla_medias.index:
                st.write("No hay datos estandarizados para mostrar en este grupo.")
            else:
                sorted_vars = tabla_medias.loc[cluster_id].sort_values()
                top_10 = sorted_vars.tail(10)
                fig1, ax1 = plt.subplots(figsize=(5, 4))
                top_10.plot(kind="barh", ax=ax1)
                ax1.set_title("Variables más representativas")
                st.pyplot(fig1)

        with col_der:
            st.subheader("Variables de cefalea (estandarizadas)")
            if tabla_medias.empty or cluster_id not in tabla_medias.index:
                st.write("No hay variables de cefalea para mostrar en este grupo.")
            else:
                present = [c for c in cefalea_vars_presentes if c in tabla_medias.columns]
                if len(present) == 0:
                    st.write("No hay variables de cefalea presentes en la tabla estandarizada.")
                else:
                    cefalea_vals = tabla_medias.loc[cluster_id][present].sort_values()
                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    cefalea_vals.plot(kind='barh', ax=ax2, color='darkred')
                    ax2.set_title("Variables de cefalea")
                    st.pyplot(fig2)

        st.markdown("---")
        col_izq3, col_der3 = st.columns(2)
        with col_izq3:
            cluster_counts = df_scaled["grupo"].value_counts().sort_index()
            labels = ["Ruido" if i == -1 else f"Grupo {int(i) + 1}" for i in cluster_counts.index]
            fig3, ax3 = plt.subplots(figsize=(3, 2.5))
            ax3.pie(cluster_counts, labels=labels, autopct="%1.1f%%", colors=sns.color_palette("pastel"))
            ax3.set_title("Distribución por grupos", fontsize=10)
            st.pyplot(fig3)

        with col_der3:
            st.markdown("#### Comparar variables por Grupo")
            opciones_vars = {nombre_columnas.get(col, col): col for col in tabla_medias.columns} if not tabla_medias.empty else {}
            if not opciones_vars:
                st.write("No hay variables para comparar.")
            else:
                seleccion_amigable = st.selectbox("Variable a comparar", sorted(opciones_vars.keys()))
                seleccion_var = opciones_vars[seleccion_amigable]
                valores = tabla_medias[seleccion_var]
                etiquetas = ["Ruido" if i == -1 else f"Grupo {int(i) + 1}" for i in valores.index]
                fig4, ax4 = plt.subplots(figsize=(3.5, 2.5))
                sns.barplot(x=etiquetas, y=valores.values, ax=ax4)
                ax4.set_ylabel("Valor estandarizado")
                ax4.set_xlabel("Grupo")
                ax4.set_title(seleccion_amigable, fontsize=10)
                ax4.tick_params(axis='x', labelrotation=45)
                st.pyplot(fig4)

    with tab2:
        col_izq2, col_der2 = st.columns(2)
        with col_izq2:
            st.subheader("Los 10 más Impactantes (Interpretado)")
            if tabla_original.empty or cluster_id not in tabla_original.index:
                st.write("No hay datos interpretados para mostrar en este grupo.")
            else:
                sorted_vars = tabla_medias.loc[cluster_id].sort_values()
                top_10 = sorted_vars.tail(10)
                combined_vars = list(top_10.index)
                real_values_combined = tabla_original.loc[cluster_id][combined_vars]
                tabla_real_mostrar = real_values_combined.copy()

                # aplicar traducciones y asegurar tipo string para evitar errores de Arrow
                translated = {}
                for col in tabla_real_mostrar.index:
                    val = tabla_real_mostrar[col]
                    try:
                        trad = traducir_valor_aproximado(col, val, mapeo_valores)
                    except Exception:
                        trad = val
                    translated[col] = str(trad)
                df_to_show = pd.Series(translated).rename("Valor real").to_frame()
                df_to_show.index = [nombre_columnas.get(c, c) for c in df_to_show.index]
                st.dataframe(df_to_show)

        with col_der2:
            st.subheader("Variables de cefalea (Interpretado)")
            if tabla_original.empty or cluster_id not in tabla_original.index:
                st.write("No hay datos de cefalea interpretados para este grupo.")
            else:
                cefalea_real = tabla_original.loc[cluster_id][[c for c in cefalea_vars_presentes if c in tabla_original.columns]]
                translated = {}
                for col in cefalea_real.index:
                    try:
                        trad = traducir_valor_aproximado(col, cefalea_real[col], mapeo_valores)
                    except Exception:
                        trad = cefalea_real[col]
                    translated[col] = str(trad)
                df_to_show = pd.Series(translated).rename("Valor real").to_frame()
                df_to_show.index = [nombre_columnas.get(c, c) for c in df_to_show.index]
                st.dataframe(df_to_show)

        col_sel, col_val = st.columns([1, 1])
        with col_sel:
            st.markdown("### Explora una variable específica")
            opciones_vars = {nombre_columnas.get(col, col): col for col in tabla_original.columns} if not tabla_original.empty else {}
            if opciones_vars:
                seleccion_amigable = st.selectbox("Selecciona una variable:", sorted(opciones_vars.keys()))
                seleccion_var = opciones_vars[seleccion_amigable]
            else:
                st.write("No hay variables para seleccionar.")
                seleccion_var = None

        with col_val:
            if seleccion_var is not None and cluster_id in tabla_original.index:
                valor_original = tabla_original.loc[cluster_id][seleccion_var]
                valor_traducido = traducir_valor_aproximado(seleccion_var, valor_original, mapeo_valores)
                nombre_amigable = nombre_columnas.get(seleccion_var, seleccion_var)
                st.markdown("### Valor Real")
                st.metric(label=nombre_amigable, value=str(valor_traducido))
            else:
                st.write("Valor no disponible para mostrar.")

# ---------------- EXPORTACIÓN ----------------
st.divider()
if st.button("Generar Excel de resumen"):
    try:
        with pd.ExcelWriter("resumen_post_dashboard.xlsx") as writer:
            if not tabla_medias.empty:
                tabla_medias.to_excel(writer, sheet_name="Estandarizado")
            if not tabla_original.empty:
                tabla_original.to_excel(writer, sheet_name="Valores Reales")
        st.success("Archivo Excel generado como 'resumen_post_dashboard.xlsx'")
    except Exception as e:
        st.error(f"No se pudo generar el Excel: {e}")

# ---------------- VALIDACIÓN / DIAGNÓSTICO ----------------
st.sidebar.header("🔧 Diagnóstico")
st.sidebar.subheader("Versiones de librerías")
try:
    st.sidebar.write(f"NumPy: {np.__version__}")
    st.sidebar.write(f"UMAP: {umap.__version__}")
except Exception:
    st.sidebar.write("UMAP: (no disponible)")
try:
    st.sidebar.write(f"HDBSCAN: {importlib.metadata.version('hdbscan')}")
except Exception:
    st.sidebar.write("HDBSCAN: (no disponible)")
st.sidebar.write(f"scikit-learn: {sklearn.__version__}")
st.sidebar.write(f"Pandas: {pd.__version__}")

st.sidebar.subheader("Semilla aleatoria")
st.sidebar.write(f"SEED utilizada: {SEED}")

st.sidebar.subheader("Clústeres detectados")
unique, counts = np.unique(cluster_labels, return_counts=True)
for cluster, count in zip(unique, counts):
    st.sidebar.write(f"Clúster {cluster}: {count} puntos")

st.sidebar.subheader("Hash y estructura de datos")
try:
    st.sidebar.write(f"Hash del archivo DatasetV4.csv: {hashlib.md5(open('DatasetV4.csv','rb').read()).hexdigest()}")
except Exception:
    st.sidebar.write("Hash archivo: (no disponible)")

st.sidebar.write(f"Hash del dataframe procesado: {dataframe_hash(df_scaled)}")
st.sidebar.write("Shape:", df.shape)
st.sidebar.write("Columnas:", list(df.columns))
st.sidebar.write("Primeras filas:")
st.sidebar.write(df.head())



######################################################COMANDO PARA INICIAR STREAMLIT###############################################################################
# python -m streamlit run prueba1-4.py

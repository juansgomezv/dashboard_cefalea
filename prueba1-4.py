## POSIBLE ENTREGA CON RUIDO DEL 20%

# --- Configuración para reproducibilidad ---
import os
import numpy as np
import random
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
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

# Reducir dimensionalidad con UMAP usando semilla
umap_model = UMAP(random_state=SEED)
min_cluster_size = 20
min_samples = 1

# ✅ HDBSCAN sin random_state (no soportado)
clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

# ---------------- FUNCIONES DE TRADUCCIÓN ----------------
def cargar_mapeo_valores(path):
    try:
        df = pd.read_excel(path, sheet_name="valores")
        df.columns = df.columns.str.lower()
        mapeo = {}
        for columna in df["columna"].unique():
            sub = df[df["columna"] == columna]
            mapeo[columna] = list(zip(sub["valor"], sub["traduccion"]))
        return mapeo
    except Exception as e:
        st.warning(f"No se pudo cargar 'mapeos.xlsx'. Error: {e}")
        return {}

def cargar_nombres_columnas(path):
    try:
        df = pd.read_excel(path, sheet_name="columnas")
        df.columns = df.columns.str.lower()
        return dict(zip(df["columna"], df["nombre_amigable"]))
    except Exception as e:
        st.warning(f"No se pudo cargar nombres amigables. Error: {e}")
        return {}

def traducir_valor_aproximado(col, val, mapeo_dict):
    if col not in mapeo_dict:
        return val
    try:
        val = float(val)
        valores = np.array([v[0] for v in mapeo_dict[col]])
        idx = (np.abs(valores - val)).argmin()
        return mapeo_dict[col][idx][1]
    except:
        return val

# ---------------- CARGAR MAPEOS ----------------
mapeo_valores = cargar_mapeo_valores("mapeos.xlsx")
nombre_columnas = cargar_nombres_columnas("mapeos.xlsx")

# ---------------- CARGAR Y PREPARAR DATOS ----------------
df = pd.read_csv("DatasetV4.csv", index_col=0)
threshold = 0.85
cols_to_drop_zeros = (df == 0).sum() / len(df) > threshold
cols_to_drop_const = df.nunique() == 1
cols_to_drop = df.columns[cols_to_drop_zeros | cols_to_drop_const]
df = df.drop(columns=cols_to_drop)

cefalea_vars = [
    "AntecedentesFamiliares", "LugarDolor", "IntensidadDolor",
    "DuracionDolor", "FrecuenciaDolor", "ActividadFisica",
    "InasistenciaDolor", "IndiceDolor"
]
cefalea_vars_presentes = [v for v in cefalea_vars if v in df.columns]
df = df[~(df[cefalea_vars_presentes] == 0).all(axis=1)]

df_original = df.copy()
df_weighted = df.copy()
df_weighted[cefalea_vars_presentes] *= 3

df = df.reset_index(drop=True)
df_weighted = df_weighted.reset_index(drop=True)
df_original = df_original.reset_index(drop=True)

# ---------------- CLUSTERING Y REDUCCIÓN DIMENSIONAL ----------------
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_weighted), columns=df.columns)

umap_reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED)
X_umap = umap_reducer.fit_transform(df_scaled)

cluster_labels = clusterer.fit_predict(X_umap)

df_scaled["grupo"] = cluster_labels
df_original["grupo"] = cluster_labels

df_scaled_no_noise = df_scaled[df_scaled["grupo"] != -1]
df_original_no_noise = df_original[df_original["grupo"] != -1]

tabla_medias = df_scaled_no_noise.groupby("grupo").mean().round(2)

tabla_real_round = df_original_no_noise.copy()
for col in tabla_real_round.select_dtypes(include=np.number).columns:
    tabla_real_round[col] = tabla_real_round[col].round()
tabla_original = tabla_real_round.groupby("grupo").mean().round(2)

# ---------------- ESTADÍSTICAS GENERALES ----------------
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
    for i in sorted(tabla_medias.index):
        if i == -1:
            cluster_opciones["Ruido"] = -1
        else:
            cluster_opciones[f"Grupo {i + 1}"] = i
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

    unique_clusters = sorted(df_umap["grupo"].unique())
    palette = sns.color_palette("tab10", n_colors=len(unique_clusters))
    cluster_to_color = {cid: palette[i] for i, cid in enumerate(unique_clusters)}

    fig_umap, ax_umap = plt.subplots(figsize=(6, 5))
    for cid in unique_clusters:
        cluster_data = df_umap[df_umap["grupo"] == cid]
        color = cluster_to_color[cid] if cid == cluster_id else "#d3d3d3"
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
            sorted_vars = tabla_medias.loc[cluster_id].sort_values()
            top_5 = sorted_vars.tail(10)
            combined = top_5
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            combined.plot(kind="barh", ax=ax1, color=["teal"] * len(combined))
            ax1.set_title("Variables más representativas")
            st.pyplot(fig1)

        with col_der:
            st.subheader("Variables de cefalea (estandarizadas)")
            cefalea_vals = tabla_medias.loc[cluster_id][cefalea_vars_presentes].sort_values()
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            cefalea_vals.plot(kind='barh', ax=ax2, color='darkred')
            ax2.set_title("Variables de cefalea")
            st.pyplot(fig2)

        st.markdown("---")
        col_izq3, col_der3 = st.columns(2)
        with col_izq3:
            cluster_counts = df_scaled["grupo"].value_counts().sort_index()
            labels = ["Ruido" if i == -1 else f"Grupo {i + 1}" for i in cluster_counts.index]
            fig3, ax3 = plt.subplots(figsize=(3, 2.5))
            ax3.pie(cluster_counts, labels=labels, autopct="%1.1f%%", colors=sns.color_palette("pastel"))
            ax3.set_title("Distribución por grupos", fontsize=10)
            st.pyplot(fig3)

        with col_der3:
            st.markdown("#### Comparar variables por Grupo")
            opciones_vars = {nombre_columnas.get(col, col): col for col in tabla_medias.columns}
            seleccion_amigable = st.selectbox("Variable a comparar", sorted(opciones_vars.keys()))
            seleccion_var = opciones_vars[seleccion_amigable]
            valores = tabla_medias[seleccion_var]
            etiquetas = ["Ruido" if i == -1 else f"Grupo {i + 1}" for i in valores.index]
            fig4, ax4 = plt.subplots(figsize=(3.5, 2.5))
            sns.barplot(x=etiquetas, y=valores.values, ax=ax4, palette="Blues_d")
            ax4.set_ylabel("Valor estandarizado")
            ax4.set_xlabel("Grupo")
            ax4.set_title(seleccion_amigable, fontsize=10)
            ax4.tick_params(axis='x', labelrotation=45)
            st.pyplot(fig4)

    with tab2:
        col_izq2, col_der2 = st.columns(2)
        with col_izq2:
            st.subheader("Los 10 más Impactantes (Interpretado)")
            sorted_vars = tabla_medias.loc[cluster_id].sort_values()
            top_5 = sorted_vars.tail(10)
            combined_vars = list(top_5.index)
            real_values_combined = tabla_original.loc[cluster_id][combined_vars]
            tabla_real_mostrar = real_values_combined.copy()
            for col in tabla_real_mostrar.index:
                tabla_real_mostrar[col] = traducir_valor_aproximado(col, tabla_real_mostrar[col], mapeo_valores)
            tabla_real_mostrar.index = [nombre_columnas.get(c, c) for c in tabla_real_mostrar.index]
            st.dataframe(tabla_real_mostrar.to_frame(name="Valor real"))

        with col_der2:
            st.subheader("Variables de cefalea (Interpretado)")
            cefalea_real = tabla_original.loc[cluster_id][cefalea_vars_presentes]
            cefalea_para_mostrar = cefalea_real.copy()
            for col in cefalea_para_mostrar.index:
                cefalea_para_mostrar[col] = traducir_valor_aproximado(col, cefalea_para_mostrar[col], mapeo_valores)
            cefalea_para_mostrar.index = [nombre_columnas.get(c, c) for c in cefalea_para_mostrar.index]
            st.dataframe(cefalea_para_mostrar.to_frame(name="Valor real"))

        col_sel, col_val = st.columns([1, 1])
        with col_sel:
            st.markdown("### Explora una variable específica")
            opciones_vars = {nombre_columnas.get(col, col): col for col in tabla_original.columns}
            seleccion_amigable = st.selectbox("Selecciona una variable:", sorted(opciones_vars.keys()))
            seleccion_var = opciones_vars[seleccion_amigable]

        with col_val:
            valor_original = tabla_original.loc[cluster_id][seleccion_var]
            valor_traducido = traducir_valor_aproximado(seleccion_var, valor_original, mapeo_valores)
            nombre_amigable = nombre_columnas.get(seleccion_var, seleccion_var)
            st.markdown("### Valor Real")
            st.metric(label=nombre_amigable, value=str(valor_traducido))

# ---------------- EXPORTACIÓN ----------------
st.divider()
if st.button("Generar Excel de resumen"):
    with pd.ExcelWriter("resumen_post_dashboard.xlsx") as writer:
        tabla_medias.to_excel(writer, sheet_name="Estandarizado")
        tabla_original.to_excel(writer, sheet_name="Valores Reales")
    st.success("Archivo Excel generado como 'resumen_post_dashboard.xlsx'")

# ---------------- VALIDACIÓN ----------------
st.sidebar.header("🔧 Diagnóstico")
st.sidebar.subheader("Versiones de librerías")
st.sidebar.write(f"NumPy: {np.__version__}")
st.sidebar.write(f"UMAP: {umap.__version__}")
st.sidebar.write(f"HDBSCAN: {importlib.metadata.version('hdbscan')}")
st.sidebar.write(f"scikit-learn: {sklearn.__version__}")
st.sidebar.write(f"Pandas: {pd.__version__}")

st.sidebar.subheader("Semilla aleatoria")
st.sidebar.write(f"SEED utilizada: {SEED}")

st.sidebar.subheader("Clústeres detectados")
unique, counts = np.unique(cluster_labels, return_counts=True)
for cluster, count in zip(unique, counts):
    st.sidebar.write(f"Clúster {cluster}: {count} puntos")

def dataframe_hash(df):
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

st.sidebar.subheader("Hash de datos")
st.sidebar.write(f"Hash de datos: {dataframe_hash(df)}")
st.sidebar.write("Shape:", df.shape)
st.sidebar.write("Columnas:", list(df.columns))
st.sidebar.write("Primeras filas:")
st.sidebar.write(df.head())



######################################################COMANDO PARA INICIAR STREAMLIT###############################################################################
# python -m streamlit run prueba1-4.py

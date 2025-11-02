# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kmodes import KModes
import umap.umap_ as umap
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import ConvexHull
import plotly.graph_objects as go

# ######################################################FUNCIONES DE TRADUCCIÓN##################################################################################
def cargar_mapeo_valores(path):
    try:
        df = pd.read_csv(path, encoding="ISO-8859-1")
        df.columns = df.columns.str.lower()
        mapeo = {}
        for columna in df["columna"].unique():
            sub = df[df["columna"] == columna]
            mapeo[columna] = list(zip(sub["valor"], sub["traduccion"]))
        return mapeo
    except Exception as e:
        st.warning(f"No se pudo cargar 'valores.csv'. Error: {e}")
        return {}

def cargar_nombres_columnas(path):
    try:
        df = pd.read_csv(path, encoding="ISO-8859-1")
        df.columns = df.columns.str.lower()
        return dict(zip(df["columna"], df["nombre_amigable"]))
    except Exception as e:
        st.warning(f"No se pudo cargar 'columnas.csv'. Error: {e}")
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

# ######################################################CARGAR MAPEOS DE VALORES############################################################################################
mapeo_valores = cargar_mapeo_valores("valores.csv")
nombre_columnas = cargar_nombres_columnas("columnas.csv")

# ######################################################CARGAR Y PREPARAR DATOS##################################################################################
df = pd.read_csv("DatasetV5.csv", index_col=0)

threshold = 0.85
columnas_importantes = [
    "AntecedentesFamiliares", "LugarDolor", "IntensidadDolor", "DuracionDolor",
    "FrecuenciaDolor", "ActividadFisica", "InasistenciaDolor", "IndiceDolor"
]

cols_to_drop_zeros = (df == 0).sum() / len(df) > threshold
cols_to_drop_const = df.nunique() == 1
cols_to_drop = df.columns[(cols_to_drop_zeros | cols_to_drop_const) & ~df.columns.isin(columnas_importantes)]
df = df.drop(columns=cols_to_drop)

cefalea_vars_presentes = [v for v in columnas_importantes if v in df.columns]
df = df[~(df[cefalea_vars_presentes] == 0).all(axis=1)]

df_original = df.copy()
df_weighted = df.copy()
df_weighted[cefalea_vars_presentes] *= 1  

df = df.reset_index(drop=True)
df_weighted = df_weighted.reset_index(drop=True)
df_original = df_original.reset_index(drop=True)

# ######################################################CLUSTERING CON K-MODES ##################################################################################
X_kmodes = df_original.copy()  

km = KModes(
    n_clusters=10,     # Mejor valor encontrado
    init="Huang",      # Mejor inicialización
    n_init=10,         # Número de inicializaciones
    random_state=42
)

cluster_labels = km.fit_predict(X_kmodes)
df_original["grupo"] = cluster_labels

tabla_original = df_original.groupby("grupo").agg(lambda x: x.mode().iloc[0])
tabla_scores = pd.DataFrame(index=sorted(df_original["grupo"].unique()), 
                            columns=[c for c in df_original.columns if c not in ["grupo", "IndiceDolor", "PrediccionDolor", "PrediccionDolorCat"]])

for g in tabla_scores.index:
    subset = df_original[df_original["grupo"] == g]
    for col in tabla_scores.columns:
        mode_val = tabla_original.loc[g, col]
        proportion = (subset[col] == mode_val).mean()
        tabla_scores.loc[g, col] = proportion

tabla_medias = tabla_scores.astype(float)  

# ######################################################BOSQUE ALEATORIO##########################################################################################
if "IndiceDolor" in df_original.columns:
    X = df_original.drop(columns=["IndiceDolor", "grupo"])
    y = df_original["IndiceDolor"]
    y = y.replace({4: 3})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    smote_strategy = {3: y_train.value_counts().max()}

    rf_pipeline = Pipeline([
        ("smote", SMOTE(sampling_strategy=smote_strategy, random_state=42)),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    rf_pipeline.fit(X_train, y_train)
    predicciones_test = rf_pipeline.predict(X_test)
    df_original.loc[X_test.index, "PrediccionDolor"] = predicciones_test
    predicciones_full = rf_pipeline.predict(X)
    df_original["PrediccionDolor"] = predicciones_full

    traduccion_dolor = {0: "Muy bajo", 1: "Bajo", 2: "Medio", 3: "Alto"}
    df_original["PrediccionDolorCat"] = df_original["PrediccionDolor"].map(traduccion_dolor)
    exactitud_test = accuracy_score(y_test, predicciones_test)
else:
    rf_pipeline = None

# ######################################################UMAP SOLO PARA VISUALIZACIÓN ########################################################
try:
    umap_reducer_vis = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="hamming", random_state=42)
    X_for_umap = X.astype(int).values 
    X_umap_vis = umap_reducer_vis.fit_transform(X_for_umap)
    df_umap_vis = pd.DataFrame(X_umap_vis, columns=["x", "y"])
    df_umap_vis["grupo"] = df_original["grupo"].values
except Exception as e:
    print("UMAP visual falló:", e)
    df_umap_vis = pd.DataFrame(columns=["x", "y", "grupo"])

# ######################################################ESTADISTICAS GENERALES####################################################################################
total_estudiantes = len(df_original)
estudiantes_clasificados = np.sum(df_original["grupo"].notna())
n_clusters = len(np.unique(df_original["grupo"]))

# ######################################################STREAMLIT#################################################################################################
st.set_page_config(page_title="Dashboard", layout="wide")

col_title, col_total = st.columns([3, 1])
with col_title:
    st.title("CEFALEA EN LOS ESTUDIANTES UPB BUCARAMANGA")
with col_total:
    st.metric("👥 Total estudiantes encuestados", total_estudiantes)

tab1, tab2 = st.tabs(["Agrupaciones", "Predicciones"])

# ###################################################### TAB AGRUPACIONES ######################################################
with tab1:
    with col_title:
        st.header("Análisis de Agrupaciones")

    # --- Primera fila: UMAP + Tabla 1 + Tabla 2 ---
    col_umap, col_tab1, col_tab2 = st.columns([1.2, 1, 1])

    # ----------------- COLUMNA 1 (UMAP) -----------------
    with col_umap:
        if df_umap_vis.empty:
            st.info("Visualización UMAP no disponible.")
            cluster_id = None
        else:
            unique_clusters = sorted(df_umap_vis["grupo"].unique())
            opciones_clusters = {f"Grupo {g+1}": g for g in unique_clusters}

            # Dividir espacio: mitad para dropdown, mitad para métrica
            col_drop, col_info = st.columns([1, 1])

            with col_drop:
                cluster_seleccionado = st.selectbox("", list(opciones_clusters.keys()))
                cluster_id = opciones_clusters[cluster_seleccionado]

            with col_info:
                n_estudiantes_grupo = len(df_original[df_original["grupo"] == cluster_id])
                st.markdown(
                    f"<div style='text-align:center; line-height:1;'>"
                    f"<span style='font-size:36px; font-weight:bold;'>{n_estudiantes_grupo}</span><br>"
                    f"<span style='font-size:16px; color:gray;'>estudiantes</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # --- Gráfico UMAP ---
            fig_umap, ax_umap = plt.subplots(figsize=(4.5, 3.5))
            palette = sns.color_palette("tab10", n_colors=max(3, len(unique_clusters)))
            cluster_to_color = {cid: palette[i % len(palette)] for i, cid in enumerate(unique_clusters)}

            for cid in unique_clusters:
                cluster_data = df_umap_vis[df_umap_vis["grupo"] == cid]
                color = cluster_to_color[cid] if cid == cluster_id else "#D3D3D3"
                alpha = 0.9 if cid == cluster_id else 0.25
                ax_umap.scatter(cluster_data["x"], cluster_data["y"], s=40, c=[color], alpha=alpha)

                if cid == cluster_id and len(cluster_data) >= 3:
                    hull = ConvexHull(cluster_data[["x", "y"]].values)
                    hull_points = cluster_data.iloc[hull.vertices][["x", "y"]].values
                    ax_umap.plot(
                        np.append(hull_points[:, 0], hull_points[0, 0]),
                        np.append(hull_points[:, 1], hull_points[0, 1]),
                        c=color, linewidth=1.5
                    )
                    ax_umap.fill(hull_points[:, 0], hull_points[:, 1], color=color, alpha=0.20)

            ax_umap.set_xticks([]); ax_umap.set_yticks([])
            st.pyplot(fig_umap)

    # ----------------- COLUMNA 2 (Tabla 1) -----------------
    with col_tab1:
        st.subheader("🔹 Top 10 características")
        if (cluster_id is None) or (cluster_id not in tabla_medias.index):
            st.info("Selecciona un grupo válido.")
        else:
            sorted_vars = tabla_medias.loc[cluster_id].sort_values()
            top_10 = sorted_vars.tail(10)
            real_values_combined = tabla_original.loc[cluster_id][top_10.index]
            tabla_real_mostrar = real_values_combined.copy()
            for col in tabla_real_mostrar.index:
                tabla_real_mostrar[col] = traducir_valor_aproximado(col, tabla_real_mostrar[col], mapeo_valores)
            tabla_real_mostrar.index = [nombre_columnas.get(c, c) for c in tabla_real_mostrar.index]
            st.dataframe(tabla_real_mostrar.to_frame(name="Valor real"))

    # ----------------- COLUMNA 3 (Tabla 2) -----------------
    with col_tab2:
        st.subheader("🩺 Variables clínicas")
        if (cluster_id is None) or (cluster_id not in tabla_original.index):
            st.info("Selecciona un grupo válido.")
        else:
            cefalea_real = tabla_original.loc[cluster_id][cefalea_vars_presentes]
            cefalea_para_mostrar = cefalea_real.copy()
            for col in cefalea_para_mostrar.index:
                cefalea_para_mostrar[col] = traducir_valor_aproximado(col, cefalea_para_mostrar[col], mapeo_valores)
            cefalea_para_mostrar.index = [nombre_columnas.get(c, c) for c in cefalea_para_mostrar.index]
            st.dataframe(cefalea_para_mostrar.to_frame(name="Valor real"))

    # --- Segunda fila: Explorar variable + Perfil Multivariante ---
    st.markdown("---")
    col_explorar, col_radar = st.columns([1, 1])

    # ----------------- EXPLORAR VARIABLE -----------------
    with col_explorar:
        st.subheader("🔎 Explorar variable específica")

        opciones_vars = {nombre_columnas.get(col, col): col for col in tabla_original.columns}
        seleccion_amigable = st.selectbox("Selecciona una variable:", sorted(opciones_vars.keys()))
        seleccion_var = opciones_vars[seleccion_amigable]

        if seleccion_var not in df_original.columns:
            st.info("Selecciona una variable válida.")
        else:
            # Calcular moda (valor más común) por grupo
            valores_por_grupo = df_original.groupby("grupo")[seleccion_var].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
            )

            # Traducir los valores de la variable
            valores_traducidos = [
                traducir_valor_aproximado(seleccion_var, v, mapeo_valores)
                for v in valores_por_grupo.values
            ]

            # Crear DataFrame temporal para graficar
            df_plot = pd.DataFrame({
                "Grupo": [f"{g+1}" for g in valores_por_grupo.index],
                "Valor": valores_por_grupo.values,
                "Valor traducido": valores_traducidos
            })

            # Obtener todos los valores posibles de esa variable (desde el mapeo)
            if seleccion_var in mapeo_valores:
                posibles_valores = [v[0] for v in mapeo_valores[seleccion_var]]
                etiquetas_traducidas = [v[1] for v in mapeo_valores[seleccion_var]]
            else:
                posibles_valores = sorted(df_plot["Valor"].unique())
                etiquetas_traducidas = [traducir_valor_aproximado(seleccion_var, v, mapeo_valores) for v in posibles_valores]

            # Crear gráfico de barras
            fig_bar, ax_bar = plt.subplots(figsize=(6, 3))
            sns.barplot(
                data=df_plot,
                x="Grupo",
                y="Valor",
                palette="Blues_d",
                ax=ax_bar
            )

            # Forzar eje Y con todos los valores posibles (aunque no aparezcan)
            ax_bar.set_yticks(posibles_valores)
            ax_bar.set_yticklabels(etiquetas_traducidas)

            # Ajustes visuales
            ax_bar.set_xlabel("Grupo")
            ax_bar.set_ylabel(seleccion_amigable)
            ax_bar.set_title("Comparación entre grupos")
            st.pyplot(fig_bar)



    # ----------------- PERFIL MULTIVARIABLE -----------------
    with col_radar:
        st.subheader("📈 Perfil Multivariante por grupo")
        import plotly.graph_objects as go

        grupos_radar = st.multiselect("Selecciona grupos a comparar", sorted(tabla_scores.index), default=[0, 1])
        vars_radar = [v for v in cefalea_vars_presentes if v in tabla_scores.columns]

        fig = go.Figure()
        for g in grupos_radar:
            valores = tabla_scores.loc[g, vars_radar].astype(float).values
            fig.add_trace(go.Scatterpolar(
                r=valores,
                theta=[nombre_columnas.get(v, v) for v in vars_radar],
                fill='toself',
                name=f'Grupo {g}'
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Comparativa de perfiles clínicos entre grupos",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)




# ###################################################### TAB BOSQUE ALEATORIO ######################################################
with tab2:
    if rf_pipeline is not None:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.header("Crear un Caso Personalizado")
        with col2:
            st.markdown(
                f"<p style='text-align: right; font-size:18px;'>Exactitud del modelo: <b>{exactitud_test:.1%}</b></p>",
                unsafe_allow_html=True
            )

        st.subheader("Características más Relevantes")

        drop_vars = [
            'InasistenciaDolor', 'IntensidadDolor', 'DuracionDolor', 'FrecuenciaDolor',
            'Ruido/Luz', 'DoloresSueno', 'Nauseas', 'LugarDolor', 'Temor', 'DecaidoDeprimido'
        ]

        model_cols = X_train.columns.tolist()
        caso_df = pd.DataFrame([{col: df_original[col].mode()[0] for col in model_cols}])

        fila1_cols = st.columns(5)
        fila2_cols = st.columns(5)

        for i, var in enumerate(drop_vars):
            valores_originales = df_original[var].unique()
            traducciones = [traducir_valor_aproximado(var, v, mapeo_valores) for v in valores_originales]
            valores_display = sorted(traducciones)

            fila = fila1_cols[i] if i < 5 else fila2_cols[i - 5]
            valor_inicial = traducir_valor_aproximado(var, caso_df[var].iloc[0], mapeo_valores)
            seleccionado = fila.selectbox(
                f"{nombre_columnas.get(var, var)}",
                valores_display,
                index=valores_display.index(valor_inicial)
            )
            for val_original, val_trad in zip(valores_originales, traducciones):
                if val_trad == seleccionado:
                    caso_df[var] = val_original
                    break

        pred_caso = rf_pipeline.predict(caso_df[model_cols])[0]
        traduccion_dolor = {0: "Muy bajo", 1: "Bajo", 2: "Medio", 3: "Alto"}
        pred_caso_trad = traduccion_dolor.get(pred_caso, pred_caso)
        st.markdown(f"### 🔹 Predicción del Indice de Dolor: **{pred_caso_trad}**")



######################################################COMANDO PARA INICIAR STREAMLIT###############################################################################
# python -m streamlit run prueba2.py

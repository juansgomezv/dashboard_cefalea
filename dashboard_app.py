# ======================================================
# DASHBOARD DE CEFALEA - INTEGRADO (SVM + UMAP/HDBSCAN)
# ======================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import warnings
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ======================================================
# 1. FUNCIONES CON CACHÉ (OPTIMIZACIÓN)
# ======================================================
@st.cache_data
def cargar_diccionarios():
    """Carga y procesa los archivos CSV de mapeo y nombres de columnas."""
    mapeo_valores = {}
    nombre_columnas = {}
    try:
        df_cols = pd.read_csv("columnas.csv", encoding="ISO-8859-1")
        df_cols.columns = df_cols.columns.str.lower().str.strip()
        nombre_columnas = {k.lower(): v for k, v in zip(df_cols["columna"], df_cols["nombre_amigable"])}
    except Exception as e:
        st.warning(f"No se pudo cargar 'columnas.csv': {e}")

    try:
        df_vals = pd.read_csv("valores.csv", encoding="ISO-8859-1")
        df_vals.columns = df_vals.columns.str.lower().str.strip()
        for columna in df_vals["columna"].unique():
            sub = df_vals[df_vals["columna"] == columna]
            mapeo_valores[columna.lower()] = list(zip(sub["valor"], sub["traduccion"]))
    except Exception as e:
        st.warning(f"No se pudo cargar 'valores.csv': {e}")
        
    return nombre_columnas, mapeo_valores

@st.cache_data
def cargar_dataset_base():
    """Carga el dataset para calcular modas (valores por defecto)."""
    try:
        df = pd.read_csv("DatasetV5.csv")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error cargando el dataset base: {e}")
        return pd.DataFrame()

@st.cache_resource
def cargar_modelo_svm():
    """Carga el modelo predictivo una sola vez en la memoria."""
    try:
        # Cambiado para cargar el modelo SVM
        return joblib.load("modelo_svm_final.joblib")
    except Exception:
        return None

@st.cache_data
def cargar_resultados_clustering():
    """Carga los resultados precalculados del modelo UMAP+HDBSCAN."""
    try:
        df_clusters = pd.read_csv("dataset_con_subgrupos.csv")
        with open("resultados_finales.json", "r", encoding="utf-8") as f:
            resultados_json = json.load(f)
        return df_clusters, resultados_json
    except Exception as e:
        return pd.DataFrame(), {}

def traducir_valor(columna, valor_crudo, mapeo):
    """Busca la traducción en texto de un valor numérico."""
    col_lower = str(columna).lower()
    if col_lower in mapeo:
        for val, trad in mapeo[col_lower]:
            if str(val) == str(valor_crudo) or str(float(val)) == str(float(valor_crudo)):
                return trad
    return str(valor_crudo)

# ======================================================
# 2. INICIALIZACIÓN DE LA APLICACIÓN
# ======================================================
st.set_page_config(page_title="Dashboard Cefalea UPB", layout="wide", page_icon="🏥")

# Cargar recursos optimizados
nombre_columnas, mapeo_valores = cargar_diccionarios()
df_base = cargar_dataset_base()
svm_model = cargar_modelo_svm() # Variable actualizada
df_clusters, resultados_json = cargar_resultados_clustering()

st.markdown("<h1 style='text-align: center;'>HERRAMIENTA DE APOYO PARA LA CARACTERIZACIÓN DE CEFALEA EN COMUNIDAD UPB</h1>", unsafe_allow_html=True)
st.markdown("---")

# Crear las pestañas
tab_kmodes, tab_pred = st.tabs(["🧩 Agrupaciones", "🎯 Predicciones"])

# ======================================================
# 3. PESTAÑA AGRUPACIONES (UMAP + HDBSCAN) - INTACTA
# ======================================================
with tab_kmodes:
    if df_clusters.empty or not resultados_json:
        st.warning("⚠️ No se encontraron los archivos de clustering ('dataset_con_subgrupos.csv' y 'resultados_finales.json'). Ejecuta primero el pipeline de entrenamiento.")
    else:
        st.subheader("Selección de Grupo y Subgrupo", help="Selecciona un Índice de Dolor y un Subgrupo específico para visualizar su mapa y su perfil clínico.")
        col_f1, col_f2, col_m1, col_m2, col_m3 = st.columns([1.5, 1.5, 1, 1, 1])
        
        niveles_disp = sorted(list(resultados_json.keys()))
        if not niveles_disp:
            st.error("No hay niveles procesados en el JSON.")
            nivel_sel, subgrupo_sel = None, None
        else:
            with col_f1:
                nivel_sel = st.selectbox("Índice de Dolor:", niveles_disp)
                datos_nivel = resultados_json[nivel_sel]
            
            subgrupos_disp = [sg["id"] for sg in datos_nivel.get("subgrupos", [])]
            with col_f2:
                if not subgrupos_disp:
                    st.warning("Sin subgrupos.")
                    subgrupo_sel, subgrupo_data = None, None
                else:
                    subgrupo_sel = st.selectbox("Subgrupo:", subgrupos_disp)
                    subgrupo_data = next((sg for sg in datos_nivel["subgrupos"] if sg["id"] == subgrupo_sel), None)
            
            with col_m1: st.metric("Estudiantes por Nivel", datos_nivel["n_pacientes"])
            with col_m2: st.metric("Atípicos", datos_nivel["n_outliers"], help="Pacientes que no encajan en ningún subgrupo definido.")
            with col_m3:
                if subgrupo_data:
                    st.metric(f"Estudiantes Subgrupo {subgrupo_sel}", f"{subgrupo_data['n']} ({subgrupo_data['porcentaje']:.1f}%)")
                    
        st.markdown("---")

        if subgrupo_sel is not None and subgrupo_data is not None:
            col_umap, col_radar = st.columns(2)
            
            with col_umap:
                st.markdown("#### 🗺️ Mapa de Grupos y Subgrupos")
                df_plot = df_clusters[df_clusters['IndiceDolor'].astype(str) == nivel_sel].copy()
                
                def asignar_color(sg_id):
                    if sg_id == subgrupo_sel: return f"Subgrupo {sg_id} (Seleccionado)"
                    elif sg_id == -1: return "Outliers (Ruido)"
                    else: return "Otros Subgrupos"
                
                df_plot['Categoria_Color'] = df_plot['Subgrupo_Global'].apply(asignar_color)
                
                color_map = {
                    f"Subgrupo {subgrupo_sel} (Seleccionado)": "#1f77b4", 
                    "Otros Subgrupos": "#e0e0e0", 
                    "Outliers (Ruido)": "#ff7f0e"  
                }
                
                fig_umap = px.scatter(
                    df_plot, x="umap_x", y="umap_y", color="Categoria_Color",
                    color_discrete_map=color_map, hover_data=["IndiceDolor", "Subgrupo_Global"], opacity=0.85
                )
                
                fig_umap.update_traces(marker=dict(size=9, line=dict(width=1, color='DarkSlateGrey')))
                fig_umap.update_layout(
                    plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'),
                    legend_title_text="Clasificación", margin=dict(l=0, r=0, t=10, b=0)
                )
                fig_umap.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, title="")
                fig_umap.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, title="")
                st.plotly_chart(fig_umap, use_container_width=True, theme=None)
                
            with col_radar:
                st.markdown("#### 🕸️ Comparativa Multivariable")
                variables_radar = list(subgrupo_data["perfil"].keys())[:8]
                nombres_radar = [nombre_columnas.get(v.lower(), v) for v in variables_radar]
                
                fig_radar = go.Figure()
                for sg in datos_nivel["subgrupos"]:
                    valores_freq = [sg["perfil"].get(var, {"frecuencia": 0})["frecuencia"] for var in variables_radar]
                    valores_freq += [valores_freq[0]]
                    nombres_cerrados = nombres_radar + [nombres_radar[0]]
                    
                    if sg["id"] == subgrupo_sel:
                        line_config, fill_config, opacity = dict(color="#1f77b4", width=3), "toself", 0.8
                    else:
                        line_config, fill_config, opacity = dict(color="black", width=1.5, dash="dash"), "none", 0.6
                        
                    fig_radar.add_trace(go.Scatterpolar(
                        r=valores_freq, theta=nombres_cerrados, fill=fill_config,
                        name=f"Subgrupo {sg['id']}", line=line_config, opacity=opacity
                    ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%", showticklabels=False)),
                    showlegend=True, margin=dict(l=40, r=40, t=30, b=30)
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            st.markdown("---")
            st.subheader(f"🩺 Perfil Clínico Dominante: Subgrupo {subgrupo_sel}")
            
            dict_transpuesto = {}
            for var, datos_var in subgrupo_data["perfil"].items():
                nombre_amigable = nombre_columnas.get(var.lower(), var)
                valor_traducido = traducir_valor(var, datos_var["valor"], mapeo_valores)
                dict_transpuesto[nombre_amigable] = [valor_traducido, f"{datos_var['frecuencia'] * 100:.1f}%"]
            
            df_perfil_t = pd.DataFrame(dict_transpuesto, index=["Valor Mayoritario", "Frecuencia"])
            st.dataframe(df_perfil_t, use_container_width=True)

        st.markdown("---")
        st.subheader("Flujo Poblacional Completo")
        st.write("Distribución desde el total de encuestados hasta la asignación de subgrupos.")
        try:
            with open("flujo_pacientes_sankey.html", "r", encoding="utf-8") as f:
                html_sankey = f.read()
            components.html(html_sankey, height=450, scrolling=False)
        except Exception:
            st.info("Visualización de Sankey no disponible. Asegúrate de tener 'flujo_pacientes_sankey.html' en la carpeta.")

# ======================================================
# 4. PESTAÑA PREDICCIONES (SVM INTEGRADO)
# ======================================================
with tab_pred:
    st.subheader("Predicción del Índice de Dolor")
    st.write("Completa el perfil del estudiante para predecir su índice de dolor.")

    if svm_model is not None and not df_base.empty:
        columnas_modelo = list(svm_model.feature_names_in_)
        
        # --- LÓGICA SVM: Usar coef_ absoluto para importancia ---
        importancias = np.abs(svm_model.coef_).mean(axis=0)
        
        df_imp = pd.DataFrame({"Variable": columnas_modelo, "Importancia": importancias})
        df_imp = df_imp.sort_values(by="Importancia", ascending=False)
        top_vars = df_imp["Variable"].head(10).tolist()
        input_usuario = {}

        st.markdown("### 📋 Variables más influyentes")
        col1, col2 = st.columns(2)
        
        for i, var in enumerate(top_vars):
            var_lower = var.lower()
            nombre_mostrar = nombre_columnas.get(var_lower, var)
            
            with col1 if i % 2 == 0 else col2:
                if var_lower in mapeo_valores:
                    opciones_texto = [texto for valor, texto in mapeo_valores[var_lower]]
                    seleccion_texto = st.selectbox(label=nombre_mostrar, options=opciones_texto, key=f"ui_{var}")
                    valor_num = next(val for val, txt in mapeo_valores[var_lower] if txt == seleccion_texto)
                    input_usuario[var] = float(valor_num)
                else:
                    moda_actual = float(df_base[var].mode()[0]) if var in df_base.columns else 0.0
                    input_usuario[var] = st.number_input(label=nombre_mostrar, value=moda_actual, key=f"ui_{var}")

        st.markdown("---")
        
        if st.button("🔍 Predecir Índice de Dolor", type="primary"):
            fila_prediccion = {}
            for col in columnas_modelo:
                if col in input_usuario:
                    fila_prediccion[col] = input_usuario[col]
                else:
                    moda_val = df_base[col].mode()[0] if col in df_base.columns else 0
                    fila_prediccion[col] = float(moda_val)
            
            df_prediccion = pd.DataFrame([fila_prediccion])
            pred_num = svm_model.predict(df_prediccion)[0]
            
            traduccion_nivel = {
                0: "Bajo 🟢",
                1: "Medio 🟡",
                2: "Alto 🔴"
            }
            
            resultado_texto = traduccion_nivel.get(pred_num, f"Clase {pred_num}")
            st.success(f"### Nivel de Dolor Predicho: {resultado_texto}")
    else:
        st.error("No se pudo cargar el modelo SVM o los datos base. Asegúrate de tener 'modelo_svm_final.joblib'.")


######################################################COMANDO PARA INICIAR STREAMLIT###############################################################################
# python -m streamlit run dashboard_app.py
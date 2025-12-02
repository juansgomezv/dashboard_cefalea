# ======================================================
# DASHBOARD DE CEFALEA - STREAMLIT
# ======================================================
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
import joblib
import io
import warnings
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ======================================================
# FUNCIONES DE TRADUCCIÓN (uso de keys en minúscula)
# ======================================================
def cargar_mapeo_valores(path):
    try:
        df = pd.read_csv(path, encoding="ISO-8859-1")
        df.columns = df.columns.str.lower().str.strip()
        mapeo = {}
        if "columna" not in df.columns or "valor" not in df.columns or "traduccion" not in df.columns:
            return {}
        for columna in df["columna"].unique():
            sub = df[df["columna"] == columna]
            mapeo[columna.lower()] = list(zip(sub["valor"], sub["traduccion"]))
        return mapeo
    except Exception as e:
        st.warning(f"No se pudo cargar 'valores.csv'. Error: {e}")
        return {}

def cargar_nombres_columnas(path):
    try:
        df = pd.read_csv(path, encoding="ISO-8859-1")
        df.columns = df.columns.str.lower().str.strip()
        if "columna" not in df.columns or "nombre_amigable" not in df.columns:
            return {}
        return {k.lower(): v for k, v in zip(df["columna"], df["nombre_amigable"])}
    except Exception as e:
        st.warning(f"No se pudo cargar 'columnas.csv'. Error: {e}")
        return {}

def traducir_valor_aproximado(col, val, mapeo_dict):
    key = str(col).lower()
    if key not in mapeo_dict:
        return val
    try:
        val_num = float(val)
        valores = np.array([float(v[0]) for v in mapeo_dict[key]])
        idx = (np.abs(valores - val_num)).argmin()
        return mapeo_dict[key][idx][1]
    except Exception:
        for orig, trad in mapeo_dict[key]:
            if str(orig) == str(val):
                return trad
        return val

# ======================================================
# CARGAR MAPEOS Y DATOS
# ======================================================
mapeo_valores = cargar_mapeo_valores("valores.csv")
nombre_columnas = cargar_nombres_columnas("columnas.csv")

df = pd.read_csv("DatasetV5.csv")
df.columns = df.columns.str.strip()

# Cargar modelos (se asume que existen)
rf_model = joblib.load("modelo_rf.joblib")
km_model = joblib.load("modelo_kmodes.joblib")

df_original = df.copy()

# variables clínicas fijas (mantener mayúsculas tal como aparecen en df)
cefalea_vars_presentes = [
    "AntecedentesFamiliares", "LugarDolor", "IntensidadDolor", "DuracionDolor",
    "FrecuenciaDolor", "ActividadFisica", "InasistenciaDolor", "IndiceDolor"
]

# ======================================================
# PREDICCIONES CON KMODES (PREPROCESAMIENTO CONSISTENTE)
# ======================================================
# Usaremos las columnas guardadas en km_model.columns_ si existen; si no, usamos todas.
cols_for_km = None
if hasattr(km_model, "columns_") and km_model.columns_:
    cols_for_km = list(km_model.columns_)
else:
    cols_for_km = list(df_original.columns)

# Construir dataframe con esas columnas (llenar faltantes con "NA" o "0")
X_kmodes = df_original.copy()
for col in cols_for_km:
    if col not in X_kmodes.columns:
        X_kmodes[col] = "NA"

# Reordenar (si alguna columna de cols_for_km no está en df_original, ya la añadimos)
X_kmodes = X_kmodes[cols_for_km].fillna("NA").astype(str)

# Si el modelo fue entrenado usando Gower, debemos calcular la matriz Gower y pasarla al predict.
try:
    use_gower = getattr(km_model, "use_gower", False)
except Exception:
    use_gower = False

if use_gower:
    import gower
    # calcular matriz Gower sobre X_kmodes (mismo orden de columnas)
    D_new = gower.gower_matrix(X_kmodes)
    cluster_labels = km_model.predict(D_new)
else:
    # codificación categórica estable (cada columna -> category codes)
    X_encoded = X_kmodes.apply(lambda col: col.astype("category").cat.codes)
    cluster_labels = km_model.predict(X_encoded)

df_original["grupo"] = cluster_labels.astype(int)

# ======================================================
# PREDICCIONES CON RANDOM FOREST 
# ======================================================
if "IndiceDolor" in df_original.columns:
    X_rf = df_original.drop(columns=["IndiceDolor", "grupo"])
    y_rf = df_original["IndiceDolor"]

    # Asegurar que rf_model tenga feature_names_in_; si no, inferir de X_rf
    try:
        expected_features = list(rf_model.feature_names_in_)
    except Exception:
        expected_features = list(X_rf.columns)

    # Añadir columnas faltantes que el RF espera
    missing_cols_rf = [c for c in expected_features if c not in X_rf.columns]
    for col in missing_cols_rf:
        X_rf[col] = 0

    # Reordenar columnas al orden esperado por el modelo
    X_rf_model = X_rf[expected_features].copy()

    # Predecir (para mostrar, no usado mas que para ejemplo)
    try:
        y_pred = rf_model.predict(X_rf_model)
    except Exception:
        # si falla la predicción, rellenar con NaNs
        y_pred = np.array([np.nan] * len(X_rf_model))
else:
    X_rf_model = pd.DataFrame()  # placeholder para evitar errores posteriores

# Importancia de variables (si el modelo la tiene)
importancia = pd.DataFrame({
    "variable": expected_features,
    "importancia": getattr(rf_model, "feature_importances_", np.zeros(len(expected_features)))
})

importancia = importancia.sort_values("importancia", ascending=False)
importancia["nombre_trad"] = importancia["variable"].str.lower().map(nombre_columnas).fillna(importancia["variable"])
importancia["acumulado"] = importancia["importancia"].cumsum()
vars_recomendadas = importancia[importancia["acumulado"] <= 0.64]["variable"].tolist()

if len(vars_recomendadas) < 8:
    vars_recomendadas = importancia["variable"].head(10).tolist()

# ======================================================
# UMAP 
# ======================================================

try:
    
    X_umap_input = X_kmodes.copy()  

    X_umap_numeric = X_umap_input.apply(lambda col: col.astype("category").cat.codes)

    umap_reducer_vis = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="hamming",
        random_state=42
    )

    X_umap_vis = umap_reducer_vis.fit_transform(X_umap_numeric.values)

    df_umap_vis = pd.DataFrame(X_umap_vis, columns=["x", "y"])
    df_umap_vis["grupo"] = df_original["grupo"].values

except Exception as e:
    print("UMAP visual falló:", e)
    df_umap_vis = pd.DataFrame(columns=["x", "y", "grupo"])

# ======================================================
# CREAR TABLAS DE RESUMEN
# ======================================================
# tabla_original: moda por grupo (valores reales)
# Si no hay grupos (por algún error) devolver dataframe vacío manejable
if "grupo" in df_original.columns and len(df_original) > 0:
    try:
        tabla_original = df_original.groupby("grupo").agg(lambda x: x.mode().iloc[0])
    except Exception:
        # fallback: intentar con as_type str y modo simple
        tabla_original = df_original.groupby("grupo").agg(lambda x: x.mode().iloc[0] if len(x.mode())>0 else x.iloc[0])
else:
    tabla_original = pd.DataFrame()

# tabla_scores: para cada grupo y variable, proporción de la moda (0..1)
cols_for_scores = [c for c in df_original.columns if c not in ["grupo", "IndiceDolor"]]
grupos_sorted = sorted(df_original["grupo"].unique()) if "grupo" in df_original.columns else []

tabla_scores = pd.DataFrame(index=grupos_sorted, columns=cols_for_scores)

for g in tabla_scores.index:
    subset = df_original[df_original["grupo"] == g]
    for col in tabla_scores.columns:
        try:
            mode_val = tabla_original.loc[g, col]
            proportion = (subset[col] == mode_val).mean()
            tabla_scores.loc[g, col] = proportion
        except Exception:
            tabla_scores.loc[g, col] = 0

tabla_medias = tabla_scores.astype(float) if not tabla_scores.empty else pd.DataFrame()

# ======= Determinar Top-10 variables más distintivas globalmente =======
if not tabla_medias.empty:
    diferencia_variables = tabla_medias.apply(lambda col: col.max() - col.min(), axis=0)
    top10_vars = diferencia_variables.nlargest(10).index.tolist()
else:
    top10_vars = []

# Preparar dataframes reproducibles para UI y export
df_top10_per_group_rows = []
for g in grupos_sorted:
    for var in top10_vars:
        raw_val = tabla_original.loc[g, var] if (not tabla_original.empty and var in tabla_original.columns) else ""
        val_trad = traducir_valor_aproximado(var, raw_val, mapeo_valores)
        df_top10_per_group_rows.append({
            "Grupo": f"Grupo {g+1}",
            "Característica": nombre_columnas.get(var.lower(), var),
            "Variable": var,
            "Valor real": val_trad
        })
df_top10_per_group = pd.DataFrame(df_top10_per_group_rows)

# ======================================================
# DASHBOARD STREAMLIT
# ======================================================
st.set_page_config(page_title="Dashboard", layout="wide")
col_title, col_total = st.columns([3, 1])
with col_title:
    st.title("CEFALEA EN LOS ESTUDIANTES UPB BUCARAMANGA")
with col_total:
    st.metric("👥 Total estudiantes encuestados", len(df_original))

tab1, tab2 = st.tabs(["Agrupaciones", "Predicciones"])

# ---------------------- TAB PREDICCIONES ----------------------
with tab2:

    st.write("Completa los campos para predecir el nivel de dolor de un estudiante con ese perfil. 📋✏️")
    st.subheader("Variables más influyentes necesarias para la predicción ⬆️")

    vars_prediccion = vars_recomendadas
    cols_modelo = expected_features

    #cols_modelo = list(rf_model.feature_names_in_)     # ← las 22 columnas reales del RF
    #vars_prediccion = cols_modelo.copy()  

    caso = {}

    # === Crear matriz 5 columnas ===
    n_cols = 5
    chunks = [vars_prediccion[i:i+n_cols] for i in range(0, len(vars_prediccion), n_cols)]

    for fila in chunks:
        cols = st.columns(n_cols)
        for idx, col in enumerate(fila):
            with cols[idx]:
                col_lower = col.lower()

                # Si existe en el mapeo → Selectbox con traducciones
                if col_lower in mapeo_valores:
                    opciones_trad = [t for _, t in mapeo_valores[col_lower]]

                    seleccion = st.selectbox(
                        f"{nombre_columnas.get(col_lower, col)}",
                        opciones_trad,
                        key=f"pred_{col}"
                    )

                    # Buscar valor numérico
                    valor_numerico = None
                    for valor, traduccion in mapeo_valores[col_lower]:
                        if traduccion == seleccion:
                            valor_numerico = valor
                            break

                    caso[col] = valor_numerico if valor_numerico is not None else 0

                else:
                    # Columna numérica → pedir número
                    caso[col] = st.number_input(
                        f"{nombre_columnas.get(col_lower, col)}",
                        min_value=0,
                        step=1,
                        key=f"pred_{col}"
                    )

    # === Completar valores faltantes usando la moda del dataset ===
    caso_completo = {}
    for col in cols_modelo:
        if col in vars_prediccion:
            caso_completo[col] = caso.get(col, df_original[col].mode()[0] if col in df_original.columns else 0)
        else:
            caso_completo[col] = df_original[col].mode()[0] if col in df_original.columns else 0

    # Crear DataFrame en el orden del modelo
    caso_df = pd.DataFrame([caso_completo])[cols_modelo]

    # === Botón de predicción ===
    if st.button("🔍 Predecir Índice de Dolor"):
        try:
            pred = rf_model.predict(caso_df)[0]
            traduccion_nivel = {
                0: "Muy Bajo",
                1: "Bajo",
                2: "Medio",
                3: "Alto"
            }
            st.success(f"**☝️🤓 Nivel de Dolor Predicho:** {traduccion_nivel.get(pred, pred)}")
        except Exception as e:
            st.error(f"Error al predecir: {e}")

# ---------------------- TAB AGRUPACIONES ----------------------
with tab1:
    st.subheader("Agrupaciones de estudiantes")

    # Dropdown para seleccionar grupo (asegurar existencia)
    if df_umap_vis.empty:
        grupos_disponibles = grupos_sorted
    else:
        grupos_disponibles = sorted(df_umap_vis["grupo"].unique())

    opciones_groups = [f"Grupo {g+1}" for g in grupos_disponibles]
    if len(opciones_groups) == 0:
        opciones_groups = ["Grupo 1"]
    grupo_seleccionado = st.selectbox("Selecciona un grupo a visualizar", opciones_groups)
    grupo_id = int(grupo_seleccionado.split(" ")[1]) - 1

    # ---------------- FILA 1: UMAP 30% | Tabla1 35% | Tabla2 35% ----------------
    col_umap, col_tab1, col_tab2 = st.columns([3, 3.5, 3.5])

    # --- Grafico UMAP (solo grupo seleccionado coloreado) ---
    with col_umap:
        if df_umap_vis.empty:
            st.info("Visualización UMAP no disponible")
        else:
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            palette = sns.color_palette("tab10", n_colors=max(3, len(grupos_disponibles)))
            cluster_to_color = {g: palette[i % len(palette)] for i, g in enumerate(grupos_disponibles)}

            for g in grupos_disponibles:
                data = df_umap_vis[df_umap_vis["grupo"] == g]
                color = cluster_to_color[g] if g == grupo_id else "#D3D3D3"
                alpha = 0.9 if g == grupo_id else 0.2
                ax.scatter(data["x"], data["y"], s=30, c=[color], alpha=alpha)

            ax.set_xticks([])
            ax.set_yticks([])
            st.pyplot(fig)

    # --- Tabla1: Top 10 características más distintivas del grupo seleccionado ---
    with col_tab1:
        st.subheader("⭐ Top 10 Características ⭐")

        if grupo_id not in tabla_scores.index:
            st.info("Grupo no disponible")
        else:
            group_props = tabla_scores.loc[grupo_id].astype(float)
            others = tabla_scores.drop(grupo_id)
            others_mean = others.mean(axis=0).astype(float) if len(others) > 0 else pd.Series(0, index=tabla_scores.columns)
            diff = (group_props - others_mean).abs()
            top10_vars_group = diff.nlargest(10).index.tolist()
            valores_moda = tabla_original.loc[grupo_id, top10_vars_group]
            filas = []
            for var in top10_vars_group:
                raw = valores_moda[var]
                trad = traducir_valor_aproximado(var, raw, mapeo_valores)
                nombre = nombre_columnas.get(var.lower(), var)
                filas.append({"Característica": nombre, "Valor real": trad})
            df_display = pd.DataFrame(filas)
            st.dataframe(df_display.reset_index(drop=True), use_container_width=True)

    # --- Tabla2: Variables clínicas (valores reales traducidos) ---
    with col_tab2:
        st.subheader("Variables clínicas 🩺")
        if grupo_id not in df_original["grupo"].unique():
            st.info("Grupo no disponible")
        else:
            df_tabla2 = df_original[df_original["grupo"] == grupo_id][cefalea_vars_presentes].mode().iloc[0]
            filas_clin = []
            for col in df_tabla2.index:
                raw_val = df_tabla2[col]
                val_trad = traducir_valor_aproximado(col, raw_val, mapeo_valores)
                nombre_trad = nombre_columnas.get(col.lower(), col)
                filas_clin.append({"Variable": nombre_trad, "Valor": val_trad})
            df_tabla2_out = pd.DataFrame(filas_clin)
            st.dataframe(df_tabla2_out.reset_index(drop=True), use_container_width=True)

    # ---------------- FILA 2: Explorar variable | Perfil multivariable ----------------
    col_explorar, col_radar = st.columns([1, 1])

    # --- Explorar variable ---
    with col_explorar:
        st.subheader("Explorar variable específica 🔎")
        opciones_vars = {nombre_columnas.get(c.lower(), c): c for c in df_original.columns}
        seleccion_amigable = st.selectbox("Selecciona una variable:", sorted(opciones_vars.keys()))
        seleccion_var = opciones_vars[seleccion_amigable]
        valores_por_grupo = df_original.groupby("grupo")[seleccion_var].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
        )
        valores_traducidos = [traducir_valor_aproximado(seleccion_var, v, mapeo_valores) for v in valores_por_grupo.values]
        df_plot = pd.DataFrame({
            "Grupo": [f"{g+1}" for g in valores_por_grupo.index],
            "Valor": valores_por_grupo.values,
            "Valor traducido": valores_traducidos
        })
        fig_bar, ax_bar = plt.subplots(figsize=(6, 3))
        sns.barplot(data=df_plot, x="Grupo", y="Valor", palette="Blues_d", ax=ax_bar)
        unique_vals = sorted(df_plot["Valor"].unique())
        ax_bar.set_yticks(unique_vals)
        ax_bar.set_yticklabels([traducir_valor_aproximado(seleccion_var, v, mapeo_valores) for v in unique_vals])
        ax_bar.set_xlabel("Grupo")
        ax_bar.set_ylabel(seleccion_amigable)
        st.pyplot(fig_bar)

    # ----------------- PERFIL MULTIVARIABLE (RADAR) -----------------
    with col_radar:
        st.subheader("📈 Perfil Multivariante por grupo")
        opciones_grupos = [f"Grupo {i+1}" for i in grupos_sorted]
        default_grupos = opciones_grupos[:2] if len(opciones_grupos) >= 2 else opciones_grupos
        grupos_radar = st.multiselect("Selecciona grupos a comparar", opciones_grupos, default=default_grupos)
        vars_radar = cefalea_vars_presentes.copy()
        fig = go.Figure()
        for g_nombre in grupos_radar:
            g = int(g_nombre.split(" ")[1]) - 1
            modas_grupo = df_original[df_original["grupo"] == g][vars_radar].mode().iloc[0]
            valores_trad = [traducir_valor_aproximado(var, modas_grupo[var], mapeo_valores) for var in vars_radar]
            valores_numericos = []
            for i, var in enumerate(vars_radar):
                valor_texto = valores_trad[i]
                var_lower = var.lower()
                valor_num = 0
                if var_lower in mapeo_valores:
                    for val_num, traduccion in mapeo_valores[var_lower]:
                        if traduccion == valor_texto:
                            valor_num = val_num
                            break
                valores_numericos.append(valor_num)
            nombres_vars = [nombre_columnas.get(v.lower(), v) for v in vars_radar]
            fig.add_trace(go.Scatterpolar(
                r=valores_numericos,
                theta=nombres_vars,
                fill='toself',
                name=f"Grupo {g+1}",
                text=valores_trad,
                hovertemplate="%{theta}: %{text}<extra></extra>"
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

# ----------------------- EXPORTAR RESUMEN GLOBAL -----------------------
st.markdown("---")
st.subheader("📤 Exportar Resumen Global")

if st.button("Generar archivo Excel"):
    try:
        wb = Workbook()

        # ============== HOJA 1: TOP 10 POR GRUPO (DINÁMICOS) ============================
        ws1 = wb.active
        ws1.title = "Top10_por_grupo"

        filas_excel = []

        for g in sorted(tabla_scores.index):
            group_props = tabla_scores.loc[g].astype(float)
            others = tabla_scores.drop(g)
            others_mean = others.mean(axis=0).astype(float) if len(others) > 0 else pd.Series(0, index=tabla_scores.columns)
            diff = (group_props - others_mean).abs()
            top10_vars_group = diff.nlargest(10).index.tolist()
            for var in top10_vars_group:
                raw_val = tabla_original.loc[g, var]
                val_trad = traducir_valor_aproximado(var, raw_val, mapeo_valores)
                nombre = nombre_columnas.get(var.lower(), var)
                filas_excel.append({
                    "Grupo": f"Grupo {g+1}",
                    "Característica": nombre,
                    "Valor real": val_trad
                })

        df_top10_excel = pd.DataFrame(filas_excel)
        for row in dataframe_to_rows(df_top10_excel, index=False, header=True):
            ws1.append(row)

        # ============== HOJA 2: VARIABLES CLÍNICAS POR GRUPO ======================
        ws2 = wb.create_sheet("Clinicas_por_grupo")
        filas_clinicas = []
        for g in sorted(tabla_original.index):
            clinicas = tabla_original.loc[g, cefalea_vars_presentes]
            for var, val in clinicas.items():
                val_trad = traducir_valor_aproximado(var, val, mapeo_valores)
                nombre_trad = nombre_columnas.get(var.lower(), var)
                filas_clinicas.append({
                    "Grupo": f"Grupo {g+1}",
                    "Variable clínica": nombre_trad,
                    "Valor real": val_trad
                })
        df_clinicas = pd.DataFrame(filas_clinicas)
        for row in dataframe_to_rows(df_clinicas, index=False, header=True):
            ws2.append(row)

        # ============== EXPORTAR EXCEL ============================================
        output = io.BytesIO()
        wb.save(output)
        output.seek(0)

        st.success("✅ Archivo con evaluación generado correctamente")
        st.download_button(
            label="⬇️ Descargar Resumen_Modelos.xlsx",
            data=output,
            file_name="Resumen_Modelos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"Error al generar el archivo: {e}")




######################################################COMANDO PARA INICIAR STREAMLIT###############################################################################
# python -m streamlit run prueba2.py

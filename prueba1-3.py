##POSIBLE ENTREGA CON RUIDO DEL 20%

# IMPORTS
import pandas as pd # Para manejar datos en formato DataFrame
import numpy as np # Para operaciones numéricas y manejo de arrays
from sklearn.preprocessing import StandardScaler # Para escalar los datos
import matplotlib.pyplot as plt # Para crear gráficos
import seaborn as sns # Para visualización de datos
import hdbscan # Para clustering basado en densidad
import umap.umap_ as umap # Para reducción de dimensionalidad
import streamlit as st # Para crear aplicaciones web interactivas

######################################################FUNCIONES DE TRADUCCIÓN##################################################################################

def cargar_mapeo_valores(path): # Cargar mapeos de valores desde un archivo Excel
    try:
        df = pd.read_excel(path, sheet_name="valores") #Cargar la hoja de valores
        df.columns = df.columns.str.lower() # Convertir nombres de columnas a minúsculas
        mapeo = {} # Crear un diccionario para almacenar los mapeos
        for columna in df["columna"].unique(): # Iterar sobre cada columna única
            sub = df[df["columna"] == columna] # Filtrar el DataFrame por la columna actual
            mapeo[columna] = list(zip(sub["valor"], sub["traduccion"])) # Crear una lista de tuplas (valor, traducción)
        return mapeo # Retornar el diccionario de mapeos
    except Exception as e: 
        st.warning(f"No se pudo cargar 'mapeos.xlsx'. Las traducciones no se aplicarán. Error: {e}") # Si hay un error, mostrar una advertencia en Streamlit
        return {}

def cargar_nombres_columnas(path): # Cargar nombres amigables de columnas desde un archivo Excel
    try:
        df = pd.read_excel(path, sheet_name="columnas") # Cargar la hoja de nombres de columnas
        df.columns = df.columns.str.lower() # Convertir nombres de columnas a minúsculas
        return dict(zip(df["columna"], df["nombre_amigable"]))  # Crear un diccionario de nombres amigables
    except Exception as e:
        st.warning(f"No se pudo cargar nombres amigables desde 'mapeos.xlsx'. Error: {e}") # Si hay un error, mostrar una advertencia en Streamlit
        return {}

def traducir_valor_aproximado(col, val, mapeo_dict): # Traducir un valor aproximado usando un diccionario de mapeos
    if col not in mapeo_dict: # Si la columna no está en el diccionario, retornar el valor original
        return val
    try: 
        val = float(val) # Convertir el valor a float para comparación
        valores = np.array([v[0] for v in mapeo_dict[col]]) # Extraer los valores del mapeo
        idx = (np.abs(valores - val)).argmin() # Encontrar el índice del valor más cercano
        return mapeo_dict[col][idx][1] # Retornar la traducción correspondiente
    except:
        return val # Si hay un error, retornar el valor original

######################################################CARGAR MAPEOS DE VALORES############################################################################################

mapeo_valores = cargar_mapeo_valores("mapeos.xlsx") # Cargar mapeo de valores
nombre_columnas = cargar_nombres_columnas("mapeos.xlsx") # Cargar nombres amigables de columnas


######################################################CARGAR Y PREPARAR DATOS##################################################################################

df = pd.read_csv("DatasetV4.csv", index_col=0) # Cargar el dataset

threshold = 0.85 # Porcentaje de columnas con valores cero para eliminar
min_cluster_size = 20 # Tamaño mínimo de clúster para HDBSCAN
min_samples = 1  # Muestras mínimas para HDBSCAN
cols_to_drop_zeros = (df == 0).sum() / len(df) > threshold # Calcular columnas con más del 85% de ceros
cols_to_drop_const = df.nunique() == 1 # Calcular columnas constantes (con un solo valor)
cols_to_drop = df.columns[cols_to_drop_zeros | cols_to_drop_const] # Eliminar columnas con muchos ceros o constantes
df = df.drop(columns=cols_to_drop) # Eliminar las columnas identificadas

cefalea_vars = [  # Variables relacionadas con cefalea
    "AntecedentesFamiliares", "LugarDolor", "IntensidadDolor",
    "DuracionDolor", "FrecuenciaDolor", "ActividadFisica",
    "InasistenciaDolor", "IndiceDolor"
]
cefalea_vars_presentes = [v for v in cefalea_vars if v in df.columns]  # Filtrar las variables de cefalea que están presentes en el DataFrame
df = df[~(df[cefalea_vars_presentes] == 0).all(axis=1)] # Eliminar filas donde todas las variables de cefalea son cero

df_original = df.copy() # Copiar el DataFrame original para mantener los valores originales
df_weighted = df.copy() # Crear una copia ponderada del DataFrame original
df_weighted[cefalea_vars_presentes] *= 3  # Multiplicar las variables de cefalea por 3 para darles más peso

df = df.reset_index(drop=True) # Reiniciar el índice del DataFrame para evitar problemas con el índice al aplicar UMAP y HDBSCAN
df_weighted = df_weighted.reset_index(drop=True) # Reiniciar el índice del DataFrame ponderado
df_original = df_original.reset_index(drop=True) # Reiniciar el índice del DataFrame original

######################################################CLUSTERING Y REDUCCION DIMENSIONAL########################################################################

scaler = StandardScaler() # Estandarizar los datos
df_scaled = pd.DataFrame(scaler.fit_transform(df_weighted), columns=df.columns) # Crear un DataFrame escalado con los mismos nombres de columnas

umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42) # Reducir la dimensionalidad usando UMAP
X_umap = umap_reducer.fit_transform(df_scaled) # Aplicar UMAP para obtener una representación de menor dimensión

clusterer = hdbscan.HDBSCAN(min_cluster_size, min_samples) # Aplicar HDBSCAN para clustering
cluster_labels = clusterer.fit_predict(X_umap) # Obtener las etiquetas de clúster

df_scaled["grupo"] = cluster_labels # Añadir las etiquetas de clúster al DataFrame escalado
df_original["grupo"] = cluster_labels # Añadir las etiquetas de clúster al DataFrame original

df_scaled_no_noise = df_scaled[df_scaled["grupo"] != -1] # Eliminar el ruido del DataFrame escalado
df_original_no_noise = df_original[df_original["grupo"] != -1] # Eliminar el ruido del DataFrame original

tabla_medias = df_scaled_no_noise.groupby("grupo").mean().round(2) # Calcular las medias de cada grupo en el DataFrame escalado sin ruido

tabla_real_round = df_original_no_noise.copy() # Redondear los valores del DataFrame original sin ruido
for col in tabla_real_round.select_dtypes(include=np.number).columns: # Redondear las columnas numéricas
    tabla_real_round[col] = tabla_real_round[col].round() # Redondear los valores numéricos a enteros
tabla_original = tabla_real_round.groupby("grupo").mean().round(2) # Calcular las medias de cada grupo en el DataFrame original sin ruido

######################################################ESTADISTICAS GENERALES####################################################################################

total_estudiantes = len(df_original) # Total de estudiantes en el DataFrame original
estudiantes_clasificados = np.sum(cluster_labels != -1) # Total de estudiantes clasificados en grupos (excluyendo ruido)
estudiantes_ruido = np.sum(cluster_labels == -1) # Total de estudiantes detectados como ruido
n_clusters = len(np.unique(cluster_labels[cluster_labels != -1])) # Total de grupos (excluyendo ruido)


######################################################STREAMLIT#################################################################################################

st.set_page_config(page_title="Dashboard", layout="wide") # Configurar Streamlit
st.title("CEFALEA EN LOS ESTUDIANTES UPB BUCARAMANGA") # Título del dashboard

col_a, col_b, col_c, col_d = st.columns(4) # Mostrar estadísticas generales
col_a.metric("👥 Total estudiantes", total_estudiantes) 
col_b.metric("✅ Clasificados en Grupos", estudiantes_clasificados)
col_c.metric("❌ Detectados como ruido", estudiantes_ruido)
col_d.metric("🔢 Total de Grupos", n_clusters)

col1, col2 = st.columns([1, 2]) # Dividir la pantalla en dos columnas

with col1: # Columna izquierda para selección de grupo
    cluster_opciones = {} # Crear un diccionario para las opciones de clúster
    for i in sorted(tabla_medias.index): # Añadir cada grupo al diccionario de opciones
        if i == -1:
            cluster_opciones["Ruido"] = -1 # Si el grupo es ruido, asignar "Ruido"
        else:
            cluster_opciones[f"Grupo {i + 1}"] = i # Asignar "Grupo X" para grupos válidos
    cluster_seleccionado = st.selectbox("Selecciona un Grupo", list(cluster_opciones.keys())) # Seleccionar un grupo de clúster
    cluster_id = cluster_opciones[cluster_seleccionado] # Obtener el ID del grupo seleccionado

    tamano_cluster_actual = np.sum(df_scaled["grupo"] == cluster_id) # Calcular el tamaño del clúster seleccionado
    st.info(f"El grupo **{cluster_seleccionado}** contiene **{tamano_cluster_actual} estudiantes**.") # Mostrar la cantidad de estudiantes por grupo

    st.markdown("### Proyección UMAP") # Mostrar proyección UMAP del clúster seleccionado
    mask_valid = cluster_labels != -1 # Filtrar los datos para eliminar el ruido
    X_valid = X_umap[mask_valid] # Obtener las coordenadas UMAP de los datos válidos
    labels_valid = cluster_labels[mask_valid] # Obtener las etiquetas de clúster de los datos válidos

    df_umap = pd.DataFrame(X_valid, columns=["x", "y"]) # Crear un DataFrame con las coordenadas UMAP
    df_umap["grupo"] = labels_valid # Añadir las etiquetas de clúster al DataFrame UMAP

    unique_clusters = sorted(df_umap["grupo"].unique()) # Obtener los grupos únicos presentes en el DataFrame UMAP
    palette = sns.color_palette("tab10", n_colors=len(unique_clusters)) # Crear una paleta de colores para los grupos
    cluster_to_color = {cid: palette[i] for i, cid in enumerate(unique_clusters)} # Asignar un color a cada grupo

    fig_umap, ax_umap = plt.subplots(figsize=(6, 5)) # Crear una figura para la proyección UMAP
    for cid in unique_clusters: # Iterar sobre cada grupo único
        cluster_data = df_umap[df_umap["grupo"] == cid] # Filtrar los datos del clúster actual
        color = cluster_to_color[cid] if cid == cluster_id else "#d3d3d3" # Asignar un color específico al clúster seleccionado, gris para los demás
        size = 60 if cid == cluster_id else 30 # Tamaño del punto: más grande para el clúster seleccionado
        alpha = 1.0 if cid == cluster_id else 0.3 # Transparencia: más opaco para el clúster seleccionado
        ax_umap.scatter(cluster_data["x"], cluster_data["y"], s=size, c=[color], alpha=alpha) # Dibujar los puntos del clúster en la proyección UMAP

        if len(cluster_data) >= 10: # Si hay suficientes puntos, añadir una densidad de kernel
            sns.kdeplot( # Dibujar una densidad de kernel para el clúster
                x=cluster_data["x"], y=cluster_data["y"], # Coordenadas UMAP
                ax=ax_umap, levels=3, color=color, linewidths=2, alpha=0.5 # Añadir transparencia a la densidad
            )

    ax_umap.set_title("Proyección UMAP") # Título del gráfico UMAP
    ax_umap.set_xlabel("Componente 1") # Etiqueta del eje X
    ax_umap.set_ylabel("Componente 2") # Etiqueta del eje Y
    st.pyplot(fig_umap) # Mostrar el gráfico UMAP en Streamlit

with col2: # Columna derecha para visualización y análisis del clúster
    tab1, tab2 = st.tabs(["Visualización Estandarizada", "Interpretación Real"]) # Dividir en dos pestañas: una para la visualización estandarizada y otra para la interpretación real
    with tab1: # Pestaña de visualización estandarizada
        col_izq, col_der = st.columns(2) # Dividir la pestaña en dos columnas

        with col_izq:  # Mostrar las variables más impactantes en el clúster seleccionado
            st.subheader("Los 10 más Impactantes (estandarizado)") # Mostrar las 10 variables más impactantes en el clúster seleccionado
            sorted_vars = tabla_medias.loc[cluster_id].sort_values() # Ordenar las variables del clúster seleccionado
            top_5 = sorted_vars.tail(10) # Obtener las 10 variables más altas
            bottom_5 = sorted_vars.head(0) # Obtener las 10 variables más bajas, por ahora solamente las 10 más altas
            combined = pd.concat([bottom_5, top_5]) # Combinar las variables más altas y más bajas
            fig1, ax1 = plt.subplots(figsize=(5, 4)) # Crear una figura para las variables más impactantes
            combined.plot(kind="barh", ax=ax1, color= ["teal"] * 10) # Dibujar un gráfico de barras horizontales
            ax1.set_title("Variables más representativas") # Título del gráfico
            st.pyplot(fig1) # Mostrar el gráfico de barras en Streamlit

        with col_der: # Mostrar las variables de cefalea en el clúster seleccionado
            st.subheader("Variables de cefalea (estandarizadas)") # Mostrar las variables de cefalea en el clúster seleccionado
            cefalea_vals = tabla_medias.loc[cluster_id][cefalea_vars_presentes].sort_values() # Ordenar las variables de cefalea del clúster seleccionado
            fig2, ax2 = plt.subplots(figsize=(5, 4)) # Crear una figura para las variables de cefalea
            cefalea_vals.plot(kind='barh', ax=ax2, color='darkred') # Dibujar un gráfico de barras horizontales
            ax2.set_title("Variables de cefalea") # Título del gráfico
            st.pyplot(fig2) # Mostrar el gráfico de barras de cefalea en Streamlit

        st.markdown("---") # Línea divisoria para separar secciones

        col_izq3, col_der3 = st.columns(2) # Dividir la sección en dos columnas para análisis adicional

        with col_izq3: # Gráfico de distribución de clústeres
            cluster_counts = df_scaled["grupo"].value_counts().sort_index() # Contar la cantidad de estudiantes en cada clúster
            labels = ["Ruido" if i == -1 else f"Grupo {i + 1}" for i in cluster_counts.index] # Crear etiquetas para los clústeres

            fig3, ax3 = plt.subplots(figsize=(3, 2.5)) # Crear una figura para el gráfico de distribución
            ax3.pie(cluster_counts, labels=labels, 
                    autopct="%1.1f%%", colors=sns.color_palette("pastel")) # Dibujar un gráfico de pastel con los clústeres
            ax3.set_title("Distribución por grupos", fontsize=10) # Título del gráfico
            st.pyplot(fig3) # Mostrar el gráfico de pastel en Streamlit

        with col_der3: # Tabla de medias por grupo
            st.markdown("#### Comparar variables por Grupo") # Mostrar una tabla de medias por grupo
            opciones_vars = {
                nombre_columnas.get(col, col): col for col in tabla_medias.columns
            } # Crear un diccionario de opciones de variables para seleccionar
            seleccion_amigable = st.selectbox("Variable a comparar", sorted(opciones_vars.keys())) # Seleccionar una variable para comparar
            seleccion_var = opciones_vars[seleccion_amigable] # Obtener el nombre de la variable seleccionada

            valores = tabla_medias[seleccion_var] # Obtener los valores de la variable seleccionada
            etiquetas = ["Ruido" if i == -1 else f"Grupo {i + 1}" for i in valores.index] # Crear etiquetas para los grupos

            fig4, ax4 = plt.subplots(figsize=(3.5, 2.5)) # Crear una figura para el gráfico de barras
            sns.barplot(x=etiquetas, y=valores.values, ax=ax4, palette="Blues_d") # Dibujar un gráfico de barras con los valores de la variable seleccionada
            ax4.set_ylabel("Valor estandarizado") # Etiqueta del eje Y
            ax4.set_xlabel("Grupo") # Etiqueta del eje X
            ax4.set_title(seleccion_amigable, fontsize=10) # Título del gráfico
            ax4.tick_params(axis='x', labelrotation=45) # Rotar las etiquetas del eje X para mejor legibilidad
            st.pyplot(fig4) # Mostrar el gráfico de barras en Streamlit


    with tab2: # Pestaña de interpretación real de los datos
        col_izq2, col_der2 = st.columns(2) # Dividir la pestaña en dos columnas para mostrar información interpretada
        with col_izq2: # Mostrar las variables más impactantes en el clúster seleccionado (interpretado)
            st.subheader("Los 10 más Impactantes (Interpretado)") # Mostrar las 10 variables más impactantes en el clúster seleccionado
            sorted_vars = tabla_medias.loc[cluster_id].sort_values() # Ordenar las variables del clúster seleccionado
            top_5 = sorted_vars.tail(10) # Obtener las 10 variables más altas
            bottom_5 = sorted_vars.head(0) # Obtener las 10 variables más bajas, por ahora solamente las 10 más altas
            combined_vars = list(bottom_5.index) + list(top_5.index) # Combinar las variables más altas y más bajas
            real_values_combined = tabla_original.loc[cluster_id][combined_vars] # Crear una serie con los valores reales de las variables combinadas

            tabla_real_mostrar = real_values_combined.copy() # Traducir los valores reales de las variables combinadas
            for col in tabla_real_mostrar.index: # Traducir cada valor usando el mapeo de valores
                tabla_real_mostrar[col] = traducir_valor_aproximado(col, tabla_real_mostrar[col], mapeo_valores) # Traducir el valor aproximado
            tabla_real_mostrar.index = [nombre_columnas.get(c, c) for c in tabla_real_mostrar.index] # Renombrar las columnas usando los nombres amigables
            st.dataframe(tabla_real_mostrar.to_frame(name="Valor real")) # Mostrar la tabla de valores reales en Streamlit

        with col_der2: # Mostrar las variables de cefalea en el clúster seleccionado (interpretado)
            st.subheader("Variables de cefalea (Interpretado)") # Mostrar las variables de cefalea en el clúster seleccionado
            cefalea_real = tabla_original.loc[cluster_id][cefalea_vars_presentes] # Ordenar las variables de cefalea del clúster seleccionado
            cefalea_para_mostrar = cefalea_real.copy() # Traducir los valores de las variables de cefalea
            for col in cefalea_para_mostrar.index: 
                cefalea_para_mostrar[col] = traducir_valor_aproximado(col, cefalea_para_mostrar[col], mapeo_valores) # Traducir el valor aproximado
            cefalea_para_mostrar.index = [nombre_columnas.get(c, c) for c in cefalea_para_mostrar.index] # Renombrar las columnas usando los nombres amigables
            st.dataframe(cefalea_para_mostrar.to_frame(name="Valor real")) # Mostrar la tabla de valores reales de cefalea en Streamlit

        col_sel, col_val = st.columns([1, 1]) # Dividir la sección en dos columnas para selección de variable y valor real
        with col_sel: # Selección de variable para explorar
            st.markdown("### Explora una variable específica") # Indicativo de orden
            opciones_vars = {
                nombre_columnas.get(col, col): col for col in tabla_original.columns
            } # Crear un diccionario de opciones de variables para seleccionar
            seleccion_amigable = st.selectbox("Selecciona una variable:", sorted(opciones_vars.keys())) # Seleccionar una variable para explorar
            seleccion_var = opciones_vars[seleccion_amigable] # Obtener el nombre de la variable seleccionada

        with col_val: # Mostrar el valor real de la variable seleccionada
            valor_original = tabla_original.loc[cluster_id][seleccion_var] # Obtener el valor real de la variable seleccionada
            valor_traducido = traducir_valor_aproximado(seleccion_var, valor_original, mapeo_valores) # Traducir el valor real usando el mapeo de valores
            nombre_amigable = nombre_columnas.get(seleccion_var, seleccion_var) #
            st.markdown("### Valor Real") # Indicativo de orden
            st.metric(label=nombre_amigable, value=str(valor_traducido)) # Mostrar el valor real traducido de la variable seleccionada

######################################################EXPORTACIÓN#################################################################################################

st.divider() # Línea divisoria para separar secciones
if st.button("Generar Excel de resumen"): # Indicativo de Orden
    with pd.ExcelWriter("resumen_post_dashboard.xlsx") as writer: # Crear un archivo Excel para exportar los resultados
        tabla_medias.to_excel(writer, sheet_name="Estandarizado") # Exportar la tabla de medias estandarizadas
        tabla_original.to_excel(writer, sheet_name="Valores Reales") # Exportar la tabla de valores reales
    st.success("Archivo Excel generado como 'resumen_post_dashboard.xlsx'") # Mostrar mensaje de éxito al generar el archivo Excel

######################################################COMANDO PARA INICIAR STREAMLIT###############################################################################
# python -m streamlit run prueba1-3.py

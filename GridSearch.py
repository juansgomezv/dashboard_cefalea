import pandas as pd # Para manejar datos en formato DataFrame
import numpy as np # Para operaciones numéricas y manejo de arrays
import matplotlib.pyplot as plt # Para crear gráficos
import seaborn as sns # Para visualización de datos
from sklearn.preprocessing import StandardScaler # Para escalar los datos
import umap.umap_ as umap # Para reducción de dimensionalidad
import hdbscan # Para clustering basado en densidad
from itertools import product # Para generar combinaciones de parámetros
from scipy.stats import entropy # Para calcular la entropía

######################################################CONFIGURACIÓN############################################################################################################

DATASET_PATH = "DatasetV4.csv" # Ruta al dataset
SALIDA_EXCEL = "gridsearch_hdbscan_mejorado.xlsx" # Ruta de salida para el archivo Excel
HEATMAP_DIR = "." # Directorio para guardar los heatmaps

######################################################CARGAR DATOS############################################################################################################

df = pd.read_csv(DATASET_PATH, index_col=0, encoding='utf-8', on_bad_lines='skip') # Cargar el dataset

threshold = 0.85 # Umbral para eliminar columnas con muchos ceros
cols_to_drop_zeros = (df == 0).sum() / len(df) > threshold # Eliminar columnas con más del 85% de ceros
cols_to_drop_const = df.nunique() == 1 # Eliminar columnas constantes (con un solo valor)
cols_to_drop = df.columns[cols_to_drop_zeros | cols_to_drop_const] # Unir las columnas a eliminar
df.drop(columns=cols_to_drop, inplace=True) # Eliminar columnas innecesarias

cefalea_vars = [ # Variables relacionadas con cefalea
    "AntecedentesFamiliares", "LugarDolor", "IntensidadDolor",
    "DuracionDolor", "FrecuenciaDolor", "ActividadFisica",
    "InasistenciaDolor", "IndiceDolor"
]
cefalea_vars_presentes = [v for v in cefalea_vars if v in df.columns] # Filtrar solo las variables presentes en el DataFrame
df = df[~(df[cefalea_vars_presentes] == 0).all(axis=1)] # Eliminar filas donde todas las variables de cefalea son cero

df_weighted = df.copy() # Ponderar las variables de cefalea
df_weighted[cefalea_vars_presentes] *= 3 # Aumentar el peso de las variables de cefalea

scaler = StandardScaler() # Estandarizar los datos
df_scaled = pd.DataFrame(scaler.fit_transform(df_weighted), columns=df.columns) # Crear un DataFrame escalado con los mismos nombres de columnas

######################################################REDUCCIÓN UMAP############################################################################################################

umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42) # Reducir la dimensionalidad con UMAP
X_umap = umap_reducer.fit_transform(df_scaled) # Transformar los datos escalados a 2D

######################################################GRID SEARCH##############################################################################################################

resultados = [] # Lista para almacenar los resultados del grid search

for min_cluster_size, min_samples in product(range(0, 35), range(0, 10)): # Generar combinaciones de parámetros
    try: # Configurar el clusterer HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples) # Ajustar el modelo HDBSCAN
        labels = clusterer.fit_predict(X_umap) # Predecir los clusters

        n_total = len(labels) # Total de puntos
        n_ruido = np.sum(labels == -1) # Contar el número de puntos de ruido
        porcentaje_ruido = round((n_ruido / n_total) * 100, 2) # Calcular el porcentaje de ruido

        labels_validos = labels[labels != -1] # Filtrar etiquetas válidas (no ruido)
        n_clusters = len(np.unique(labels_validos)) # Contar el número de clusters válidos

        if n_clusters == 0: # Si no hay clusters válidos, continuar al siguiente par de parámetros
            continue  # Evitar división por cero

        counts = np.array([np.sum(labels_validos == i) for i in np.unique(labels_validos)]) # Contar el tamaño de cada cluster
        entropia = round(entropy(counts, base=2), 3) # Calcular la entropía de la distribución de tamaños de clusters
        promedio_tamano = round(np.mean(counts), 2) # Calcular el tamaño promedio de los clusters
        std_tamano = round(np.std(counts), 2) # Calcular la desviación estándar del tamaño de los clusters

        # Ponderación: bajo ruido, muchos clusters, alta entropía
        score = (100 - porcentaje_ruido) * 0.5 + n_clusters * 1.0 + entropia * 10 # Puntuación compuesta

        resultados.append({ # Almacenar los resultados
            "min_cluster_size": min_cluster_size, # Tamaño mínimo del cluster
            "min_samples": min_samples, # Mínimo de muestras para formar un cluster
            "porcentaje_ruido": porcentaje_ruido, # Porcentaje de ruido
            "n_clusters": n_clusters, # Número de clusters válidos
            "entropia": entropia, # Entropía de la distribución de tamaños de clusters
            "tamano_promedio": promedio_tamano, # Tamaño promedio de los clusters
            "std_tamano": std_tamano, # Desviación estándar del tamaño de los clusters
            "score": round(score, 2) # Puntuación compuesta
        })

    except Exception as e: # Manejo de excepciones para errores en el ajuste del modelo
        print(f"Error en min_cluster_size={min_cluster_size}, min_samples={min_samples}: {e}") # Continuar con el siguiente par de parámetros

######################################################ALMACENAMIENTO DE DATOS##################################################################################################

df_resultados = pd.DataFrame(resultados) # Crear un DataFrame con los resultados
df_resultados.sort_values(by="score", ascending=False, inplace=True) # Ordenar los resultados por la puntuación compuesta


######################################################GENERAR HEATMAPS##########################################################################################################

def guardar_heatmap(df, value_col, titulo, archivo, cmap="viridis", fmt=".2f"): # Función para guardar un heatmap
    pivot = df.pivot(index="min_cluster_size", columns="min_samples", values=value_col) # Crear una tabla dinámica para el heatmap
    plt.figure(figsize=(10, 6)) # Configurar el tamaño de la figura
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, cbar_kws={'label': value_col}) # Crear el heatmap con anotaciones
    plt.title(titulo) # Título del heatmap
    plt.xlabel("min_samples") # Etiqueta del eje X
    plt.ylabel("min_cluster_size") # Etiqueta del eje Y
    plt.tight_layout() # Ajustar el layout para evitar solapamientos
    ruta = f"{HEATMAP_DIR}/{archivo}" # Guardar la ruta del archivo
    plt.savefig(ruta) # Guardar el heatmap como imagen
    print(f"Guardado: {ruta}") # Indicativo de Orden
    plt.close() # Cerrar la figura para liberar memoria

guardar_heatmap(df_resultados, "n_clusters", "N° de Clústeres", "heatmap_num_clusters.png", cmap="YlGnBu", fmt=".0f") # Heatmap del número de clusters
guardar_heatmap(df_resultados, "porcentaje_ruido", "% de Ruido", "heatmap_porcentaje_ruido.png", cmap="Reds") # Heatmap del porcentaje de ruido
guardar_heatmap(df_resultados, "entropia", "Entropía de Distribución", "heatmap_entropia.png", cmap="Purples") # Heatmap de la entropía de distribución
guardar_heatmap(df_resultados, "score", "Puntuación Compuesta (Score)", "heatmap_score.png", cmap="Blues") # Heatmap de la puntuación compuesta

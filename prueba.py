# ======================================================
# VERIFICACIÓN Y PREDICCIÓN KMODES + CENTROIDES + COLUMNAS
# ======================================================

import pandas as pd
import joblib
import numpy as np

# ==================== CARGAR DATOS ====================
df = pd.read_csv("DatasetV5.csv")
df = df.fillna("NA").astype(str)  # Convertir a string como en entrenamiento

# ==================== CARGAR MODELO ====================
km_model = joblib.load("modelo_kmodes.joblib")

# ==================== OBTENER COLUMNAS ====================
if hasattr(km_model, "columns_"):
    columnas_modelo = km_model.columns_
else:
    print("⚠️ El modelo KModes no tiene atributo 'columns_'. Debes indicar las columnas manualmente.")
    columnas_modelo = df.columns.tolist()  # fallback temporal

# ==================== VERIFICAR COLUMNAS ====================
cols_actuales = df.columns.tolist()
cols_faltantes = [c for c in columnas_modelo if c not in cols_actuales]
cols_extra = [c for c in cols_actuales if c not in columnas_modelo]

print("Columnas que faltan respecto al modelo:")
print(cols_faltantes)
print("\nColumnas extra respecto al modelo:")
print(cols_extra)


print("\nColumnas del modelo KModes (según centroides):")
if hasattr(km_model, "cluster_centroids_"):
    n_centroid_cols = km_model.cluster_centroids_.shape[1]
    print(f"Total columnas en centroides: {n_centroid_cols}")
else:
    print("No se encontró atributo cluster_centroids_ en el modelo.")

# ==================== FILTRAR DATOS ====================
X_kmodes = df[[c for c in columnas_modelo if c in df.columns]].copy()

print(f"\nShape centroides del modelo: {km_model.cluster_centroids_.shape}")
print(f"Shape DataFrame X_kmodes: {X_kmodes.shape}")

# ==================== IMPRIMIR CENTROIDES ====================
print("\nCentroides del modelo KModes:")
for i, centroid in enumerate(km_model.cluster_centroids_):
    print(f"\nCluster {i}:")
    print(centroid)

# ==================== PREDICCIÓN ====================
try:
    cluster_labels = km_model.predict(X_kmodes)
    print("\n✅ Predicción exitosa. Etiquetas de clusters:")
    print(cluster_labels)
except ValueError as e:
    print("\n⚠️ Error al predecir con KModes:")
    print(e)

# ==================== RESUMEN ====================
print(f"\nNúmero de filas en X_kmodes: {X_kmodes.shape[0]}")
print(f"Número de columnas en X_kmodes: {X_kmodes.shape[1]}")

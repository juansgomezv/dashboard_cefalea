import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
from itertools import product
from scipy.stats import entropy
from sklearn.metrics import silhouette_score

###################################################### CONFIGURACIÓN ##########################################################################################################

DATASET_PATH = "DatasetV5.csv"  # Ruta al dataset

###################################################### CARGAR DATOS ############################################################################################################

df = pd.read_csv(DATASET_PATH, index_col=0, encoding="utf-8", on_bad_lines="skip")

# Eliminar columnas constantes
cols_to_drop_const = df.nunique() == 1
df.drop(columns=df.columns[cols_to_drop_const], inplace=True)

# Convertir todas las variables a categóricas
df = df.astype(str)

###################################################### GRID SEARCH K-MODES ######################################################################################################

resultados = []

param_grid = {
    "n_clusters": range(2, 11),   # Número de clusters a probar
    "init": ["Huang", "Cao"],     # Métodos de inicialización
    "n_init": [5, 10]             # Número de inicializaciones
}

keys, values = zip(*param_grid.items())
combinaciones = [dict(zip(keys, v)) for v in product(*values)]

for params in combinaciones:
    try:
        km = KModes(
            n_clusters=params["n_clusters"],
            init=params["init"],
            n_init=params["n_init"],
            verbose=0,
            random_state=42
        )
        labels = km.fit_predict(df)

        # Medidas básicas
        counts = np.array([np.sum(labels == i) for i in np.unique(labels)])
        entropia = round(entropy(counts, base=2), 3)

        # Silhouette con variables categóricas → OneHotEncoding
        silhouette = None
        try:
            df_onehot = pd.get_dummies(df)
            silhouette = silhouette_score(df_onehot, labels)
        except Exception:
            silhouette = np.nan

        # Score compuesto: favorece entropía alta + silhouette
        score = (entropia * 10) + (silhouette if silhouette is not None else 0)

        resultados.append({
            **params,
            "cost": km.cost_,
            "entropia": entropia,
            "silhouette": silhouette,
            "score": round(score, 3)
        })

    except Exception as e:
        print(f"Error con {params}: {e}")

###################################################### MEJOR COMBINACIÓN ######################################################################################################

df_resultados = pd.DataFrame(resultados)
df_resultados.sort_values(by="score", ascending=False, inplace=True)

print("\n========= Mejor combinación de hiperparámetros =========")
print(df_resultados.iloc[0])

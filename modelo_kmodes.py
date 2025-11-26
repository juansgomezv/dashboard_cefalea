
import pandas as pd
import joblib
import os
import numpy as np
from kmodes.kmodes import KModes
from contextlib import redirect_stdout

# ==================== FUNCIONES ====================

def load_dataset(dataset_path):
    
    df = pd.read_csv(dataset_path)
    df = df.fillna("NA").astype(str)
    return df

def entrenar_kmodes(
    dataset_path,
    k=3,
    init="Huang",
    n_init=10,
    random_state=42,
    output_model="modelo_kmodes.joblib"
):

    # Cargar dataset
    df = load_dataset(dataset_path)

    # Fijar semilla
    np.random.seed(random_state)

    # Silenciar salida interna del algoritmo
    with open(os.devnull, "w") as f, redirect_stdout(f):
        model = KModes(
            n_clusters=k,
            init=init,
            n_init=n_init,
            verbose=0,
            random_state=random_state
        )
        model.fit(df)

    # Guardar columnas usadas dentro del modelo
    model.columns_ = df.columns.tolist()

    # Guardar modelo con joblib
    joblib.dump(model, output_model)

    # Mensaje resumen
    print(f"✅ Modelo KModes entrenado y guardado en '{output_model}'")
    print(f"Hiperparámetros: k={k}, init={init}, n_init={n_init}")
    print(f"Dimensiones del dataset: {df.shape}")
    print(f"Función de coste final: {model.cost_}")
    print(f"Número de clusters: {model.n_clusters}")

# ==================== MAIN ====================

if __name__ == "__main__":
    entrenar_kmodes(
        dataset_path="DatasetV5.csv",
        k=3,
        init="Huang",
        n_init=5,
        random_state=42,
        output_model="modelo_kmodes.joblib"
    )

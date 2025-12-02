import pandas as pd
import joblib
import os
import numpy as np
from kmodes.kmodes import KModes
from contextlib import redirect_stdout
import gower
from scipy.stats import chi2_contingency


# =============================================================
# Cargar dataset
# =============================================================
def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    df = df.fillna("NA").astype(str)
    return df


# =============================================================
# Cálculo de Cramér’s V EXACTAMENTE IGUAL AL CÓDIGO 2
# =============================================================
def compute_cramers_v(col, labels):
    contingency = pd.crosstab(col, labels)
    chi2, p, _, _ = chi2_contingency(contingency)
    n = contingency.sum().sum()
    phi2 = chi2 / n
    r, k = contingency.shape

    # Corrección de sesgo
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - (r - 1)**2 / (n - 1)
    k_corr = k - (k - 1)**2 / (n - 1)

    cramers_v = np.sqrt(phi2_corr / min((k_corr - 1), (r_corr - 1)))
    return chi2, p, cramers_v


# =============================================================
# ENTRENAMIENTO KModes
# =============================================================
def entrenar_kmodes(
    dataset_path='datasetv5.csv',
    k=3,
    init="Cao",
    n_init=5,
    random_state=42,
    use_gower=True,
    output_model="modelo_kmodes.joblib"
):

    # ------------------------------------------------------
    # Cargar datos
    # ------------------------------------------------------
    df = load_dataset(dataset_path)

    # ------------------------------------------------------
    # Entrenar KModes (gower o codificado)
    # ------------------------------------------------------
    print("\nEntrenando KModes...")

    np.random.seed(random_state)

    if use_gower:
        print("Computando matriz Gower...")
        D = gower.gower_matrix(df)

        with open(os.devnull, "w") as f, redirect_stdout(f):
            model = KModes(
                n_clusters=k,
                init=init,
                n_init=n_init,
                verbose=0,
                random_state=random_state
            )
            labels = model.fit_predict(D)

        df_used = D

    else:
        # Codificación ordinal básica
        df_encoded = df.apply(lambda col: col.astype("category").cat.codes)

        with open(os.devnull, "w") as f, redirect_stdout(f):
            model = KModes(
                n_clusters=k,
                init=init,
                n_init=n_init,
                verbose=0,
                random_state=random_state
            )
            labels = model.fit_predict(df_encoded)

        df_used = df_encoded

    # Guardar columnas y tipo de distancia
    model.columns_ = df.columns.tolist()
    model.use_gower = use_gower

    # ------------------------------------------------------
    # Calcular Cramér’s V por variable exactamente igual al CÓDIGO 2
    # ------------------------------------------------------
    chi_list = []
    for col in df.columns:
        chi2, p, v = compute_cramers_v(df[col], labels)
        chi_list.append([col, chi2, p, v])

    df_chi = pd.DataFrame(chi_list, columns=["variable", "chi2", "p_value", "cramers_v"])
    df_chi_sorted = df_chi.sort_values("cramers_v", ascending=False)

    # TOP 10 copiable
    top10_vars = df_chi_sorted.head(10)["variable"].tolist()

    print("\n=== TOP 10 VARIABLES POR CRAMER'S V ===")
    print(top10_vars)
    print("==============================================================\n")

    # ------------------------------------------------------
    # Guardar modelo
    # ------------------------------------------------------
    joblib.dump(model, output_model)

    print("==============================================")
    print(f"Modelo KModes guardado en '{output_model}'")
    print(f"Hiperparámetros: k={k}, init={init}, n_init={n_init}")
    print(f"use_gower = {use_gower}")
    print(f"Función de coste final: {model.cost_}")
    print("==============================================")

    return model, labels, df_chi_sorted


# =============================================================
# Ejecutar
# =============================================================
entrenar_kmodes()

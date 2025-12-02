import pandas as pd
import numpy as np
import itertools
from kmodes.kmodes import KModes
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import chi2_contingency
import gower
from openpyxl import Workbook

# =============================================================
# Utilidades
# =============================================================

def load_dataset():
    df = pd.read_csv('datasetv5.csv')
    df = df.astype(str).fillna("NA")
    return df

def ordinal_encode(df):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    arr = enc.fit_transform(df.values)
    return arr.astype(int), enc

def compute_cramers_v(col, labels):
    contingency = pd.crosstab(col, labels)
    chi2, p, _, _ = chi2_contingency(contingency)
    n = contingency.sum().sum()
    phi2 = chi2 / n
    r, k = contingency.shape

    # Corrección de sesgo
    phi2_corr = max(0, phi2 - ((k - 1)*(r - 1))/(n - 1))
    r_corr = r - (r - 1)**2/(n - 1)
    k_corr = k - (k - 1)**2/(n - 1)

    cramers_v = np.sqrt(phi2_corr / min((k_corr - 1), (r_corr - 1)))
    return chi2, p, cramers_v


# =============================================================
# Gridsearch basado SOLO en silhouette
# =============================================================

def run_gridsearch():

    print("Cargando datos...")
    df = load_dataset()

    print("Codificando datos...")
    X_enc, enc = ordinal_encode(df)

    # Variables clínicas ponderadas
    cefalea_vars = ["AntecedentesFamiliares", "LugarDolor", "IntensidadDolor",
                    "DuracionDolor", "FrecuenciaDolor", "ActividadFisica",
                    "InasistenciaDolor", "IndiceDolor"]
    peso_clinicas = 1
    pesos = np.ones(X_enc.shape[1])

    for i, col in enumerate(df.columns):
        if col in cefalea_vars:
            pesos[i] = peso_clinicas

    X_enc_weighted = X_enc * pesos

    # Hiperparámetros
    k_list =  [3, 4, 5]
    init_list = ["Huang", "Cao"]
    n_init_list = [5, 10, 15]
    use_gower_list = [True, False]

    all_results = []
    best_sil = -999
    best_combo = None
    best_labels = None

    print("\nIniciando gridsearch...\n")

    np.random.seed(42)

    for k, init, n_init, use_gower in itertools.product(
        k_list, init_list, n_init_list, use_gower_list
    ):
        np.random.seed(42 + k*10 + n_init*100 + (1 if use_gower else 0))

        print(f"Probando: k={k}, init={init}, n_init={n_init}, gower={use_gower}")

        # ------------------------------------------------------------------
        # Clustering
        # ------------------------------------------------------------------
        if use_gower:
            D = gower.gower_matrix(df)
            model = KModes(n_clusters=k, init=init, n_init=n_init,
                           verbose=0, random_state=42)
            labels = model.fit_predict(D)

            try:
                sil = silhouette_score(D, labels, metric="precomputed")
                sil_samples = silhouette_samples(D, labels, metric="precomputed")
                sil_std = sil_samples.std()
            except:
                sil = np.nan
                sil_std = np.nan

        else:
            model = KModes(n_clusters=k, init=init, n_init=n_init,
                           verbose=0, random_state=42)
            labels = model.fit_predict(X_enc_weighted)

            try:
                sil = silhouette_score(X_enc_weighted, labels, metric="cosine")
                sil_samples = silhouette_samples(X_enc_weighted, labels, metric="cosine")
                sil_std = sil_samples.std()
            except:
                sil = np.nan
                sil_std = np.nan

        # Guardar resultados
        all_results.append({
            "k": k,
            "init": init,
            "n_init": n_init,
            "use_gower": use_gower,
            "silhouette": sil,
            "silhouette_std": sil_std
        })

        # Actualizar mejor modelo
        if sil > best_sil:
            best_sil = sil
            best_combo = (k, init, n_init, use_gower)
            best_labels = labels
            best_model = model
            best_distance_matrix = D if use_gower else X_enc_weighted


    # =============================================================
    # MÉTRICAS ADICIONALES SOLO PARA EL MEJOR MODELO
    # =============================================================
    print("\n==========================")
    print("MEJOR MODELO (por silhouette)")
    print("==========================\n")
    print(best_combo)

    k, init, n_init, use_gower = best_combo

    # Costo
    cost = best_model.cost_

    # =============================================================
    # Cramér’s V entre CADA VARIABLE y los CLUSTERS DEL MEJOR MODELO
    # =============================================================
    chi_list = []
    for col in df.columns:
        chi2, p, v = compute_cramers_v(df[col], best_labels)
        chi_list.append([col, chi2, p, v])

    df_chi = pd.DataFrame(chi_list, columns=["variable", "chi2", "p_value", "cramers_v"])

    avg_cramers = df_chi["cramers_v"].mean()

    # =============================================================
    # Excel FINAL
    # =============================================================

    df_results = pd.DataFrame(all_results)

    df_best = pd.DataFrame([{
        "k": k,
        "init": init,
        "n_init": n_init,
        "use_gower": use_gower,
        "silhouette": best_sil,
        "cost": cost,
        "avg_cramers_v": avg_cramers
    }])

    archivo_salida = "resultados_gridsearch_kmodes.xlsx"
    with pd.ExcelWriter(archivo_salida, engine="openpyxl") as writer:
        df_results.to_excel(writer, index=False, sheet_name="gridsearch_silhouette")
        df_best.to_excel(writer, index=False, sheet_name="mejor_modelo")
        df_chi.to_excel(writer, index=False, sheet_name="chi_cuadrado_cramers")

    print(f"\nArchivo Excel generado: {archivo_salida}")
    print("\nMejor modelo + importancia de variables (Cramér’s V) guardado correctamente.")


run_gridsearch()

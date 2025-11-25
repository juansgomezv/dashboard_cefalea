import pandas as pd
import numpy as np
import itertools
from kmodes.kmodes import KModes
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import silhouette_score
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

    # Corrección de bias
    phi2_corr = max(0, phi2 - ((k - 1)*(r - 1))/(n - 1))
    r_corr = r - (r - 1)**2/(n - 1)
    k_corr = k - (k - 1)**2/(n - 1)

    cramers_v = np.sqrt(phi2_corr / min((k_corr - 1), (r_corr - 1)))
    return chi2, p, cramers_v

# =============================================================
# Gridsearch principal con ponderación
# =============================================================

def run_gridsearch():

    print("Cargando datos...")
    df = load_dataset()

    print("Codificando datos...")
    X_enc, enc = ordinal_encode(df)

    # =============================================================
    # Definir variables clínicas y pesos
    # =============================================================
    cefalea_vars = ["AntecedentesFamiliares", "LugarDolor", "IntensidadDolor",
                    "DuracionDolor", "FrecuenciaDolor", "ActividadFisica",
                    "InasistenciaDolor", "IndiceDolor"]
    peso_clinicas = 5  # factor de peso mayor para variables clínicas
    pesos = np.ones(X_enc.shape[1])
    for i, col in enumerate(df.columns):
        if col in cefalea_vars:
            pesos[i] = peso_clinicas

    X_enc_weighted = X_enc * pesos  # aplicar ponderación

    # =============================================================
    # Hiperparámetros
    # =============================================================
    k_list = [3, 4, 5, 6, 7, 8]
    init_list = ["Huang", "Cao"]
    n_init_list = [5, 10, 15]
    use_gower_list = [True, False]

    all_results = []

    print("Iniciando gridsearch...\n")

    # Reproducibilidad global
    np.random.seed(42)

    for k, init, n_init, use_gower in itertools.product(
        k_list, init_list, n_init_list, use_gower_list
    ):

        # Seed única por combinación para reproducibilidad total
        np.random.seed(42 + k*10 + n_init*100 + (1 if use_gower else 0))

        print(f"Probando: k={k}, init={init}, n_init={n_init}, gower={use_gower}")

        # =============================================================
        # Distancias y clustering
        # =============================================================
        if use_gower:
            D = gower.gower_matrix(df)
            model = KModes(n_clusters=k, init=init, n_init=n_init, verbose=0, random_state=42)
            labels = model.fit_predict(D)
        else:
            model = KModes(n_clusters=k, init=init, n_init=n_init, verbose=0, random_state=42)
            labels = model.fit_predict(X_enc_weighted)  # usar ponderación

        cost = model.cost_

        # Silhouette
        try:
            if use_gower:
                sil = silhouette_score(D, labels, metric="precomputed")
            else:
                sil = silhouette_score(X_enc_weighted, labels, metric="cosine")
        except:
            sil = np.nan

        # Desviación estándar intra-cluster
        std_cluster = np.mean([
            X_enc_weighted[labels == c].std().mean() if np.sum(labels == c) > 1 else 0
            for c in np.unique(labels)
        ])

        # Cramer's V promedio
        cramers_vals = []
        for col in df.columns:
            _, _, v = compute_cramers_v(df[col], labels)
            cramers_vals.append(v)

        avg_cramers_v = np.mean(cramers_vals)

        all_results.append({
            "k": k,
            "init": init,
            "n_init": n_init,
            "use_gower": use_gower,
            "cost": cost,
            "silhouette": sil,
            "std_cluster": std_cluster,
            "avg_cramers_v": avg_cramers_v
        })

    # =============================================================
    # Convertir resultados a DataFrame
    # =============================================================
    df_summary = pd.DataFrame(all_results)

    # Normalización de métricas
    for col in ['silhouette', 'avg_cramers_v']:
        df_summary[col + '_norm'] = (
            (df_summary[col] - df_summary[col].min()) /
            (df_summary[col].max() - df_summary[col].min() + 1e-9)
        )

    for col in ['cost', 'std_cluster']:
        temp = (
            (df_summary[col] - df_summary[col].min()) /
            (df_summary[col].max() - df_summary[col].min() + 1e-9)
        )
        df_summary[col + '_norm'] = 1 - temp  # invertir para que "menor = mejor"

    # =============================================================
    # Ranking con Pesos Normalizados
    # =============================================================
    weight_sil = 0.4375
    weight_cost = 0.375
    weight_std = 0.125
    weight_cram = 0.0625

    df_summary["ranking_score"] = (
        weight_sil * df_summary["silhouette_norm"] +
        weight_cost * df_summary["cost_norm"] +
        weight_std * df_summary["std_cluster_norm"] +
        weight_cram * df_summary["avg_cramers_v_norm"]
    )

    best_row = df_summary.loc[df_summary['ranking_score'].idxmax()]

    # =============================================================
    # Guardar resultados en Excel
    # =============================================================
    archivo_salida = "resultados_gridsearch_kmodes.xlsx"
    with pd.ExcelWriter(archivo_salida, engine="openpyxl") as writer:
        df_summary.to_excel(writer, index=False, sheet_name="resultados_completos")
        best_row.to_frame().T.to_excel(writer, index=False, sheet_name="mejor_modelo")

    print(f"\nArchivo Excel generado: {archivo_salida}")
    print("\n==========================")
    print("MEJOR MODELO")
    print("==========================\n")
    print(best_row)


run_gridsearch()

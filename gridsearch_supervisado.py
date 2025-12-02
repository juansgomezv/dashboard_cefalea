import warnings
warnings.filterwarnings("ignore")

import time
import pandas as pd
import numpy as np
from itertools import product
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# -------------------------
# Funciones auxiliares
# -------------------------
def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    TP = np.diag(cm)
    FN = cm.sum(axis=1) - TP
    FP = cm.sum(axis=0) - TP
    TN = cm.sum() - (TP + FP + FN)

    sens_class = TP / (TP + FN + 1e-9)
    spec_class = TN / (TN + FP + 1e-9)

    sens_macro = np.mean(sens_class)
    spec_macro = np.mean(spec_class)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    return sens_macro, spec_macro, f1_macro, bal_acc, prec_macro

def format_seconds(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

# -------------------------
# Gridsearch principal
# -------------------------
def run_gridsearch(dataset_path, target_col, random_seed=42):
    np.random.seed(random_seed)

    df = pd.read_csv(dataset_path)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # CV: 20 repeticiones
    cv = ShuffleSplit(n_splits=20, test_size=0.2, random_state=random_seed)

    smote_options = [True, False]

    rf_params = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [None, "sqrt", "log2"]
    }

    xgb_params = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    }

    svm_params = {
        "C": [0.1, 1.0, 2.0],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"],
        "class_weight": [None, "balanced"]
    }

    # Calcular total de combinaciones
    def total_param_combinations(param_dict):
        total = 1
        for v in param_dict.values():
            total *= len(v)
        return total

    total_RF = total_param_combinations(rf_params)
    total_XGB = total_param_combinations(xgb_params)
    total_SVM = total_param_combinations(svm_params)
    TOTAL_COMBINATIONS = (total_RF + total_XGB + total_SVM) * len(smote_options)

    print(f"\nTOTAL DE COMBINACIONES A EJECUTAR: {TOTAL_COMBINATIONS}\n")

    # resultados
    all_results = []
    comb_counter = 1
    start_time = time.time()
    times = []

    # función que evalúa cada modelo/param grid
    def evaluate_model(model_name, model_cls, param_dict, smote_option):
        nonlocal comb_counter, all_results, times

        keys = list(param_dict.keys())
        values = list(param_dict.values())

        for comb in product(*values):
            params = dict(zip(keys, comb))

            # print combinación y ETA ocasionalmente
            elapsed = time.time() - start_time
            avg = np.mean(times) if len(times) > 0 else (elapsed / max(1, comb_counter))
            remaining = max(0, TOTAL_COMBINATIONS - comb_counter + 1)
            eta = remaining * avg
            # print every iteration but ETA shown every 5 iterations to reduce clutter
            if comb_counter % 5 == 0:
                print(f"[{comb_counter}/{TOTAL_COMBINATIONS}] Modelo={model_name} | SMOTE={smote_option} | Params={params} | ETA={format_seconds(eta)}")
            else:
                print(f"[{comb_counter}/{TOTAL_COMBINATIONS}] Modelo={model_name} | SMOTE={smote_option} | Params={params}")

            combo_start = time.time()

            f1_scores = []
            metrics_list = []

            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # SMOTE seguro: ajustar k_neighbors según la clase minoritaria del fold
                if smote_option:
                    # usar pandas para conteo por clase
                    class_counts = pd.Series(y_train).value_counts()
                    min_class_size = int(class_counts.min())

                    # ajustar k_neighbors
                    k_neighbors = 5
                    if min_class_size <= k_neighbors:
                        k_neighbors = max(1, min_class_size - 1)

                    if min_class_size >= 2:
                        sm = SMOTE(random_state=random_seed, k_neighbors=k_neighbors)
                        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
                    else:
                        # si hay menos de 2 muestras en la clase minoritaria, no aplicar SMOTE
                        X_train_res, y_train_res = X_train, y_train
                else:
                    X_train_res, y_train_res = X_train, y_train

                # Crear modelo (mantener todas las combinaciones SVM aunque gamma no aplique)
                if model_name == "XGB":
                    model = model_cls(**params, eval_metric="logloss", random_state=random_seed)
                elif model_name == "SVM":
                    model = model_cls(**params)  # SVC: no pasar random_state aquí
                else:
                    model = model_cls(**params)

                model.fit(X_train_res, y_train_res)
                y_pred = model.predict(X_test)

                sens_macro, spec_macro, f1_macro, bal_acc, prec_macro = compute_metrics(y_test, y_pred)

                metrics_list.append([sens_macro, spec_macro, f1_macro, bal_acc, prec_macro])
                f1_scores.append(f1_macro)

            # Si por alguna razón no se obtuvieron folds (debe ser raro), evitar crash
            if len(metrics_list) == 0:
                # saltar esta combinación sin añadir resultados
                combo_time = time.time() - combo_start
                times.append(combo_time)
                comb_counter += 1
                continue

            metrics_mean = np.mean(metrics_list, axis=0)
            f1_std = float(np.std(f1_scores))

            all_results.append({
                "modelo": model_name,
                "params": params,
                "smote": smote_option,
                "sens_macro": float(metrics_mean[0]),
                "spec_macro": float(metrics_mean[1]),
                "f1_macro": float(metrics_mean[2]),
                "bal_acc": float(metrics_mean[3]),
                "prec_macro": float(metrics_mean[4]),
                "f1_std_cv": f1_std
            })

            combo_time = time.time() - combo_start
            times.append(combo_time)
            comb_counter += 1

    # Ejecutar evaluaciones para cada SMOTE option
    for sm in smote_options:
        evaluate_model("RF", RandomForestClassifier, rf_params, sm)
        evaluate_model("XGB", XGBClassifier, xgb_params, sm)
        evaluate_model("SVM", SVC, svm_params, sm)

    # Convertir a DataFrame
    df_results = pd.DataFrame(all_results)

    # Hoja 1: solo columnas resumidas
    df_sheet1 = df_results[["modelo", "params", "smote", "f1_macro", "f1_std_cv"]].copy()

    # Hojas Top5: incluir las métricas solicitadas
    cols_top = ["modelo", "params", "smote", "f1_macro", "f1_std_cv", "sens_macro", "spec_macro", "bal_acc", "prec_macro"]

    top5_SVM = df_results[df_results["modelo"] == "SVM"].nlargest(5, "f1_macro")[cols_top]
    top5_RF  = df_results[df_results["modelo"] == "RF"].nlargest(5, "f1_macro")[cols_top]
    top5_XGB = df_results[df_results["modelo"] == "XGB"].nlargest(5, "f1_macro")[cols_top]

    # Mejor combinación global (por f1_macro)
    best_idx = df_results["f1_macro"].idxmax()
    best_row = df_results.loc[[best_idx]][cols_top]

    # Guardar Excel con las 5 hojas solicitadas
    output_file = "resultados_gridsearch_supervisado.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_sheet1.to_excel(writer, index=False, sheet_name="RESULTADOS")
        top5_SVM.to_excel(writer, index=False, sheet_name="TOP5_SVM")
        top5_RF.to_excel(writer, index=False, sheet_name="TOP5_RF")
        top5_XGB.to_excel(writer, index=False, sheet_name="TOP5_XGB")
        best_row.to_excel(writer, index=False, sheet_name="MEJOR_MODELO")

    print(f"\n✅ Archivo generado: {output_file}")
    total_elapsed = time.time() - start_time
    print(f"Tiempo total: {format_seconds(total_elapsed)}")

# MAIN
if __name__ == "__main__":
    run_gridsearch("datasetv5.csv", target_col="IndiceDolor")

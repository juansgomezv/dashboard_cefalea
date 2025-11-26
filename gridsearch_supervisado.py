
import pandas as pd
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# =============================================================
# Funciones auxiliares
# =============================================================

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # Sensibilidad y especificidad por clase
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
    std_class = np.std(sens_class)
    
    return sens_macro, spec_macro, f1_macro, bal_acc, prec_macro, std_class

# =============================================================
# Gridsearch principal
# =============================================================

def run_gridsearch(dataset_path, target_col, random_seed=42):
    np.random.seed(random_seed)
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # =============================================================
    # Hiperparámetros
    # =============================================================
    smote_options = [True, False]

    rf_params = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [None, 'sqrt', 'log2']
    }

    xgb_params = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.15],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    }

    svm_params = {
        "C": [0.1, 0.5, 1.0, 2.0],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"],
        "class_weight": [None, "balanced"]
    }

    all_results = []
    comb_counter = 1  # contador global de combinaciones

    # =============================================================
    # Iteración sobre modelos y combinaciones de hiperparámetros
    # =============================================================
    for smote_option in smote_options:

        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=random_seed
        )

        if smote_option:
            sm = SMOTE(random_state=random_seed)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train.copy(), y_train.copy()

        # ---- Random Forest ----
        for ne, md, mss, msl, mf in product(
            rf_params["n_estimators"], rf_params["max_depth"],
            rf_params["min_samples_split"], rf_params["min_samples_leaf"],
            rf_params["max_features"]
        ):
            
            print(f"[{comb_counter}] Probando: modelo=RF, params={{'n_estimators': {ne}, 'max_depth': {md}, 'min_samples_split': {mss}, 'min_samples_leaf': {msl}, 'max_features': {mf}}}, smote={smote_option}")
            
            model = RandomForestClassifier(
                n_estimators=ne,
                max_depth=md,
                min_samples_split=mss,
                min_samples_leaf=msl,
                max_features=mf,
                random_state=random_seed
            )
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)
            sens_macro, spec_macro, f1_macro, bal_acc, prec_macro, std_class = compute_metrics(y_test, y_pred)

            all_results.append({
                "modelo": "RF",
                "params": {"n_estimators": ne, "max_depth": md, "min_samples_split": mss,
                           "min_samples_leaf": msl, "max_features": mf},
                "smote": smote_option,
                "sens_macro": sens_macro,
                "spec_macro": spec_macro,
                "f1_macro": f1_macro,
                "bal_acc": bal_acc,
                "prec_macro": prec_macro,
                "std_class": std_class
            })
            comb_counter += 1

        # ---- XGBoost ----
        for ne, md, lr, ss, csb in product(
            xgb_params["n_estimators"], xgb_params["max_depth"], xgb_params["learning_rate"],
            xgb_params["subsample"], xgb_params["colsample_bytree"]
        ):
            
            print(f"[{comb_counter}] Probando: modelo=XGB, params={{'n_estimators': {ne}, 'max_depth': {md}, 'learning_rate': {lr}, 'subsample': {ss}, 'colsample_bytree': {csb}}}, smote={smote_option}")

            model = XGBClassifier(
                n_estimators=ne,
                max_depth=md,
                learning_rate=lr,
                subsample=ss,
                colsample_bytree=csb,
                eval_metric="logloss",
                random_state=random_seed
            )
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)
            sens_macro, spec_macro, f1_macro, bal_acc, prec_macro, std_class = compute_metrics(y_test, y_pred)

            all_results.append({
                "modelo": "XGB",
                "params": {"n_estimators": ne, "max_depth": md, "learning_rate": lr,
                           "subsample": ss, "colsample_bytree": csb},
                "smote": smote_option,
                "sens_macro": sens_macro,
                "spec_macro": spec_macro,
                "f1_macro": f1_macro,
                "bal_acc": bal_acc,
                "prec_macro": prec_macro,
                "std_class": std_class
            })
            comb_counter += 1

        # ---- SVM ----
        for c, k, g, cw in product(
            svm_params["C"], svm_params["kernel"], svm_params["gamma"], svm_params["class_weight"]
        ):
            
            print(f"[{comb_counter}] Probando: modelo=SVM, params={{'C': {c}, 'kernel': {k}, 'gamma': {g}, 'class_weight': {cw}}}, smote={smote_option}")

            model = SVC(C=c, kernel=k, gamma=g, class_weight=cw, random_state=random_seed)
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)
            sens_macro, spec_macro, f1_macro, bal_acc, prec_macro, std_class = compute_metrics(y_test, y_pred)

            all_results.append({
                "modelo": "SVM",
                "params": {"C": c, "kernel": k, "gamma": g, "class_weight": cw},
                "smote": smote_option,
                "sens_macro": sens_macro,
                "spec_macro": spec_macro,
                "f1_macro": f1_macro,
                "bal_acc": bal_acc,
                "prec_macro": prec_macro,
                "std_class": std_class
            })
            comb_counter += 1

    # =============================================================
    # Convertir a DataFrame y normalizar
    # =============================================================
    df_results = pd.DataFrame(all_results)
    metrics = ["sens_macro", "spec_macro", "f1_macro", "bal_acc", "prec_macro"]
    for m in metrics:
        df_results[m + "_norm"] = (df_results[m] - df_results[m].min()) / (df_results[m].max() - df_results[m].min() + 1e-9)
    df_results["std_class_norm"] = 1 - (df_results["std_class"] - df_results["std_class"].min()) / \
                                   (df_results["std_class"].max() - df_results["std_class"].min() + 1e-9)

    # =============================================================
    # Ranking ponderado (puedes ajustar pesos)
    # =============================================================
    w_sens = 0.25
    w_spec = 0.20
    w_f1 = 0.20
    w_bal = 0.15
    w_prec = 0.10
    w_std = 0.10

    df_results["ranking_score"] = (
        w_sens * df_results["sens_macro_norm"] +
        w_spec * df_results["spec_macro_norm"] +
        w_f1 * df_results["f1_macro_norm"] +
        w_bal * df_results["bal_acc_norm"] +
        w_prec * df_results["prec_macro_norm"] +
        w_std * df_results["std_class_norm"]
    )

    best_row = df_results.loc[df_results["ranking_score"].idxmax()]

    # =============================================================
    # Guardar Excel con tres hojas
    # =============================================================
    output_file = "resultados_gridsearch_supervisado.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        # Hoja 1: resultados completos
        df_results.to_excel(writer, index=False, sheet_name="resultados_completos")
        
        # Hoja 2: mejor modelo
        best_row.to_frame().T.to_excel(writer, index=False, sheet_name="mejor_modelo")
        
        # Hoja 3: promedios por modelo y smote
        metrics_all = ["sens_macro", "spec_macro", "f1_macro", "bal_acc", "prec_macro", "std_class",
                    "sens_macro_norm", "spec_macro_norm", "f1_macro_norm", "bal_acc_norm", "prec_macro_norm", "std_class_norm",
                    "ranking_score"]
        
        df_prom = df_results.groupby(["modelo", "smote"])[metrics_all].mean().reset_index()
        
        # Renombrar columnas para indicar que son promedios
        rename_dict = {col: col + "_prom" for col in metrics_all}
        df_prom = df_prom.rename(columns=rename_dict)
        
        df_prom.to_excel(writer, index=False, sheet_name="promedios_modelo")

    print(f"\nArchivo Excel generado: {output_file} con tres hojas (completos, mejor_modelo, promedios_modelo)")

# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    run_gridsearch("datasetv5.csv", target_col="IndiceDolor")

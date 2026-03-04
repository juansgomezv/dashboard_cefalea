import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from imblearn.over_sampling import SMOTE
import joblib

# ==============================
# Configuración
# ==============================
dataset_path = "datasetv5.csv"
target_col = "IndiceDolor"
random_seed = 42

best_params = {
    "n_estimators": 50,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 1,
    "max_features": None
}

use_smote = False

# ==============================
# Cargar datos
# ==============================
df = pd.read_csv(dataset_path)
X = df.drop(columns=[target_col])
y = df[target_col]

# ==============================
# División train/test
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=random_seed
)

# ==============================
# Aplicar SMOTE (si corresponde)
# ==============================
if use_smote:
    sm = SMOTE(random_state=random_seed)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
else:
    X_train_res, y_train_res = X_train.copy(), y_train.copy()

# ==============================
# Entrenar Random Forest
# ==============================
rf_model = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    min_samples_leaf=best_params["min_samples_leaf"],
    max_features=best_params["max_features"],
    random_state=random_seed
)

rf_model.fit(X_train_res, y_train_res)

# ==============================
# Evaluación global
# ==============================
y_pred = rf_model.predict(X_test)
f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

# ==============================
# Evaluación ONE-VS-REST
# ==============================
clases = sorted(y.unique())

resultados_ovr = []
matrices_confusion = {}

print("\n===== Evaluación One-vs-Rest =====\n")

for clase in clases:

    print(f"--- Clase {clase} vs Todas ---")

    # Binarización
    y_test_bin = (y_test == clase).astype(int)

    # Probabilidades para esta clase
    y_prob = rf_model.predict_proba(X_test)[:, clases.index(clase)]
    y_pred_bin = (y_prob >= 0.5).astype(int)

    # Matriz de confusión
    cm = confusion_matrix(y_test_bin, y_pred_bin)
    matrices_confusion[clase] = cm

    # Métricas
    reporte = classification_report(
        y_test_bin,
        y_pred_bin,
        output_dict=True,
        zero_division=0
    )

    try:
        auc = roc_auc_score(y_test_bin, y_prob)
    except:
        auc = np.nan

    sensibilidad = reporte["1"]["recall"]
    precision = reporte["1"]["precision"]
    f1 = reporte["1"]["f1-score"]

    print(f"Sensibilidad: {sensibilidad:.4f}")
    print(f"Precisión:    {precision:.4f}")
    print(f"F1:           {f1:.4f}")
    print(f"ROC-AUC:      {auc:.4f}")
    print("Matriz de confusión:\n", cm, "\n")

    resultados_ovr.append({
        "Clase": clase,
        "Sensibilidad": sensibilidad,
        "Precision": precision,
        "F1": f1,
        "ROC_AUC": auc
    })

# ==============================
# DataFrame de resultados
# ==============================
df_ovr = pd.DataFrame(resultados_ovr)

# ==============================
# Guardar modelo
# ==============================
joblib.dump(rf_model, "modelo_rf.joblib")

# ==============================
# Guardar resultados en Excel
# ==============================
excel_output = "resultados_1a1.xlsx"

with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:

    # Hoja 1: métricas OVR
    df_ovr.to_excel(writer, sheet_name="Metricas_OVR", index=False)

    # Hojas siguientes: matrices de confusión
    for clase, cm in matrices_confusion.items():
        df_cm = pd.DataFrame(
            cm,
            index=["Real 0", "Real 1"],
            columns=["Pred 0", "Pred 1"]
        )
        df_cm.to_excel(writer, sheet_name=f"CM_Clase_{clase}")

print("\n==============================")
print("Proceso finalizado correctamente")
print(f"F1 macro (test): {f1_macro:.5f}")
print(f"SMOTE aplicado: {use_smote}")
print(f"Archivo generado: {excel_output}")
print("==============================")

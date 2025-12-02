import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, recall_score, precision_score, balanced_accuracy_score,
    roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib

# ==============================
# Configuración
# ==============================
dataset_path = "datasetv5.csv"
target_col = "IndiceDolor"
random_seed = 42

# >> Selección manual de columnas <<
selected_features = ['InsatisfaccionFamiliar', 'EstimuladoReprimido', 'SatisfechoInsatisfecho', 'AnimadoDesanimado',
                     'ComodoIncomodo', 'PersonaEntenderSituacion', 'SerenoAgitado', 'ReconfortadoDesconsolado',
                     'PersonaSugiereManejarProblemas', 'ApoyadoCriticado',
                     "AntecedentesFamiliares", "LugarDolor", "IntensidadDolor",
                     "DuracionDolor", "FrecuenciaDolor", "ActividadFisica",
                     "InasistenciaDolor", "Ruido/Luz", "Nauseas", "EdemaOcular",
                     "Alteraciones", "EspasmosFaciales"]

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

missing_cols = [c for c in selected_features if c not in df.columns]
if missing_cols:
    raise ValueError(f"Columnas no existen en el dataset: {missing_cols}")

X = df[selected_features].copy()
y = df[target_col]

# ==============================
# División train/test
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=random_seed
)

# ==============================
# SMOTE opcional
# ==============================
if use_smote:
    sm = SMOTE(random_state=random_seed)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
else:
    X_train_res, y_train_res = X_train.copy(), y_train.copy()

# ==============================
# Entrenar RF
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
rf_model.feature_names_in_ = X_train.columns.tolist()

# ==============================
# Predicciones
# ==============================
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)

# ==============================
# Métricas globales
# ==============================
f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
sens_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
bal_acc = balanced_accuracy_score(y_test, y_pred)

# Especificidad macro (manual)
cm = confusion_matrix(y_test, y_pred)
num_classes = cm.shape[0]
spec_list = []

for i in range(num_classes):
    TN = cm.sum() - (cm[i,:].sum() + cm[:,i].sum() - cm[i,i])
    FP = cm[:,i].sum() - cm[i,i]
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0
    spec_list.append(spec)

spec_macro = np.mean(spec_list)

# ==============================
# UNO VS TODOS (1-vs-all)
# ==============================
classes = sorted(y.unique())
resultados_1a1 = []

for cls in classes:
    y_true_binary = (y_test == cls).astype(int)
    y_pred_binary = (y_pred == cls).astype(int)
    y_proba_binary = y_proba[:, cls]

    sens = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    # ROC-AUC (manejar clases sin positivos)
    try:
        roc = roc_auc_score(y_true_binary, y_proba_binary)
    except:
        roc = 0

    resultados_1a1.append([
        cls, sens, prec, f1, roc
    ])

df_1a1 = pd.DataFrame(resultados_1a1,
                      columns=["clase", "sensibilidad", "precision", "f1", "roc_auc"])

# ==============================
# Guardar Excel
# ==============================
output_excel = "resultados_1a1.xlsx"
df_1a1.to_excel(output_excel, index=False)

print("\n=== Resultados 1 vs Todos guardados ===")
print(f"Archivo: {output_excel}")

# ==============================
# Imprimir métricas globales
# ==============================
print("\n===== MÉTRICAS GENERALES =====")
print(f"f1_macro   = {f1_macro:.5f}")
print(f"sens_macro = {sens_macro:.5f}")
print(f"spec_macro = {spec_macro:.5f}")
print(f"bal_acc    = {bal_acc:.5f}")
print(f"prec_macro = {prec_macro:.5f}")

print("\nColumnas usadas en el modelo:")
for col in selected_features:
    print(" -", col)


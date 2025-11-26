

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from imblearn.over_sampling import SMOTE
import joblib

# ==============================
# Configuración
# ==============================
dataset_path = "datasetv5.csv"
target_col = "IndiceDolor"
random_seed = 42

# Parámetros seleccionados del mejor modelo segun el grid
best_params = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 1,
    "max_features": None
}
use_smote = False  # True o False según el resultado del grid

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
    X, y, test_size=0.2, stratify=y, random_state=random_seed
)

# ==============================
# Aplicar SMOTE si corresponde
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
# Evaluar desempeño en test
# ==============================
y_pred = rf_model.predict(X_test)
precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)

# ==============================
# Guardar modelo entrenado
# ==============================
output_file = "modelo_rf.joblib"
print(f"Guardando modelo en archivo: {output_file}\n")
joblib.dump(rf_model, output_file)
print("Modelo guardado correctamente.\n")

print("Hiperparámetros usados:")
for k, v in best_params.items():
    print(f"  {k} = {v}")
print(f"\nSMOTE aplicado: {use_smote}")
print(f"Precisión macro en test: {precision_macro:.5f}")

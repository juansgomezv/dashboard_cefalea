# ==========================================================
# Comparación de Modelos:
# 1. RF con enteros + SMOTE
# 2. RF con OneHotEncoding
# 3. SVM con OneHotEncoding
# ==========================================================

# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    balanced_accuracy_score, precision_recall_fscore_support
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# =======================
# Cargar y preparar datos
# =======================
df = pd.read_csv("DatasetV5.csv", index_col=0)

# Features y target
X = df.drop(columns=["IndiceDolor"])
y = df["IndiceDolor"]

# 🔹 Combinar clases 3 y 4 en una sola (clase "3")
y = y.replace({4: 3})

# =======================
# Train/Test Split fijo
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =======================
# Función de evaluación
# =======================
def evaluar_modelo(nombre, modelo, X_train, y_train, X_test, y_test):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    return {
        "Modelo": nombre,
        "Accuracy": acc,
        "Balanced Acc": bal_acc,
        "Precision (macro)": precision,
        "Recall (macro)": recall,
        "F1 (macro)": f1,
        "Precision (weighted)": precision_w,
        "Recall (weighted)": recall_w,
        "F1 (weighted)": f1_w
    }

# =======================
# 1. Random Forest con enteros + SMOTE
# =======================
smote_strategy = {3: y_train.value_counts().max()}
rf_smote = ImbPipeline([
    ("smote", SMOTE(sampling_strategy=smote_strategy, random_state=42)),
    ("rf", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# =======================
# 2. Random Forest con OHE
# =======================
ohe = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), X.columns)],
    remainder="drop"
)

rf_ohe = Pipeline([
    ("ohe", ohe),
    ("rf", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# =======================
# 3. SVM con OHE
# =======================
svm_ohe = Pipeline([
    ("ohe", ohe),
    ("svm", SVC(
        kernel="rbf",
        C=1,
        gamma="scale",
        class_weight="balanced",
        random_state=42
    ))
])

# =======================
# Entrenar y Evaluar
# =======================
resultados = []
resultados.append(evaluar_modelo("RF + SMOTE (enteros)", rf_smote, X_train, y_train, X_test, y_test))
resultados.append(evaluar_modelo("RF + OHE", rf_ohe, X_train, y_train, X_test, y_test))
resultados.append(evaluar_modelo("SVM + OHE", svm_ohe, X_train, y_train, X_test, y_test))

# =======================
# Comparación Final
# =======================
df_resultados = pd.DataFrame(resultados)
print("\n================ COMPARACIÓN DE MODELOS ================")
print(df_resultados.round(4))


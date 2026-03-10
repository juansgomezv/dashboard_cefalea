import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from imblearn.over_sampling import SMOTE
import joblib

# ==============================
# 1. Configuración General
# ==============================
dataset_path = "datasetv5.csv"
target_col = "IndiceDolor"
random_seed = 42

# Parámetros óptimos encontrados en tu GridSearch
best_params_svm = {
    "C": 2.0,
    "kernel": "linear",
    "gamma": "scale",
    "class_weight": "balanced"
}

use_smote = False

# Configuración visual para las gráficas (Times New Roman, 12pt)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

# ==============================
# 2. Cargar datos y Reducir Clases
# ==============================
print("--- Cargando datos y aplicando reducción de clases ---")
df = pd.read_csv(dataset_path)

# Convertir a string para hacer el reemplazo seguro
df[target_col] = df[target_col].astype(str)

# Agrupamos los niveles 2, 3 y 4 en el nivel '2'
df.loc[df[target_col].isin(['2', '3', '4']), target_col] = '2'

# Convertimos de nuevo a entero (quedarán clases 0, 1 y 2)
df[target_col] = df[target_col].astype(int)
print(f"Distribución de clases para entrenamiento:\n{df[target_col].value_counts().sort_index()}\n")

X = df.drop(columns=[target_col])
y = df[target_col]

# ==============================
# 3. División train/test y SMOTE
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=random_seed
)

if use_smote:
    print("--- Aplicando SMOTE ---")
    sm = SMOTE(random_state=random_seed)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
else:
    X_train_res, y_train_res = X_train.copy(), y_train.copy()

# ==============================
# 4. Entrenar el Modelo SVM
# ==============================
print("--- Entrenando modelo SVM Lineal ---")
svm_model = SVC(
    C=best_params_svm["C"],
    kernel=best_params_svm["kernel"],
    gamma=best_params_svm["gamma"],
    class_weight=best_params_svm["class_weight"],
    probability=True,  # Crucial para las Curvas ROC
    random_state=random_seed
)

svm_model.fit(X_train_res, y_train_res)

# ==============================
# 5. Evaluación ONE-VS-REST y ROC
# ==============================
y_pred = svm_model.predict(X_test)
f1_macro_global = f1_score(y_test, y_pred, average="macro", zero_division=0)

clases = sorted(y.unique())
resultados_ovr = []
matrices_confusion = {}

# Diccionarios para guardar los datos de las gráficas
fpr_dict = {}
tpr_dict = {}
auc_dict = {}

print("===== Evaluación One-vs-Rest (SVM) =====")

for clase in clases:
    print(f"\n--- Clase {clase} vs Todas ---")
    
    # Binarizar las etiquetas reales (1 si es la clase actual, 0 si es otra)
    y_test_bin = (y_test == clase).astype(int)
    
    # Obtener las probabilidades predichas para la clase actual
    y_prob = svm_model.predict_proba(X_test)[:, clases.index(clase)]
    y_pred_bin = (y_prob >= 0.5).astype(int)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test_bin, y_pred_bin)
    matrices_confusion[clase] = cm

    # Métricas de texto
    reporte = classification_report(y_test_bin, y_pred_bin, output_dict=True, zero_division=0)
    
    # Calcular y guardar datos exactos para la curva ROC
    fpr, tpr, _ = roc_curve(y_test_bin, y_prob)
    auc = roc_auc_score(y_test_bin, y_prob)
    
    fpr_dict[clase] = fpr
    tpr_dict[clase] = tpr
    auc_dict[clase] = auc

    sensibilidad = reporte["1"]["recall"]
    precision = reporte["1"]["precision"]
    f1 = reporte["1"]["f1-score"]

    print(f"Sensibilidad: {sensibilidad:.4f}")
    print(f"Precisión:    {precision:.4f}")
    print(f"F1:            {f1:.4f}")
    print(f"ROC-AUC:      {auc:.4f}")
    print("Matriz de confusión:\n", cm)

    resultados_ovr.append({
        "Clase": clase,
        "Sensibilidad": sensibilidad,
        "Precision": precision,
        "F1": f1,
        "ROC_AUC": auc
    })

# ==============================
# 6. Generar Gráfica de Curvas ROC
# ==============================
print("\n--- Generando y guardando gráficas ROC ---")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, clase in enumerate(clases):
    ax = axes[idx]
    
    # Dibujar la curva del modelo
    ax.plot(fpr_dict[clase], tpr_dict[clase], color='#1f77b4', lw=2, 
            label=f'AUC = {auc_dict[clase]:.3f}')
    
    # Dibujar la línea diagonal punteada naranja (modelo aleatorio)
    ax.plot([0, 1], [0, 1], color='#ff7f0e', lw=1.5, linestyle='--')
    
    # Configuraciones visuales idénticas a tu imagen
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Curva ROC One-vs-Rest — Clase {clase}')
    ax.legend(loc="lower right")
    ax.grid(True)

plt.tight_layout()
grafica_path = "curvas_roc_svm_3clases.png"
plt.savefig(grafica_path, dpi=300)
plt.show()

# ==============================
# 7. Guardar Modelo y Excel
# ==============================
print("\n--- Guardando resultados y modelo ---")
df_ovr = pd.DataFrame(resultados_ovr)
joblib.dump(svm_model, "modelo_svm_final.joblib")

excel_output = "resultados_svm_1a1.xlsx"
with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
    df_ovr.to_excel(writer, sheet_name="Metricas_OVR", index=False)
    for clase, cm in matrices_confusion.items():
        df_cm = pd.DataFrame(cm, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"])
        df_cm.to_excel(writer, sheet_name=f"CM_Clase_{clase}")

print("\n=======================================================")
print("✅ Proceso finalizado correctamente")
print(f"F1 macro global (test): {f1_macro_global:.5f}")
print(f"Modelo guardado en:      modelo_svm_final.joblib")
print(f"Resultados guardados en: {excel_output}")
print(f"Gráfica guardada en:     {grafica_path}")
print("=======================================================")
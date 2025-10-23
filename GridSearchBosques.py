import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score, classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import numpy as np

######################################################CONFIGURACIÓN############################################################################################################

DATASET_PATH = "DatasetV5.csv"
HEATMAP_DIR = "."

######################################################CARGAR Y PREPARAR DATOS###################################################################################################

df = pd.read_csv(DATASET_PATH, index_col=0, encoding="utf-8", on_bad_lines="skip")

X = df.drop(columns=["IndiceDolor"])
y = df["IndiceDolor"]

# Combinar clases 3 y 4 en una sola clase 3
y = y.replace({4: 3})

######################################################DEFINIR GRID SEARCH#######################################################################################################

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

resultados = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Métricas
scorers = {
    "accuracy": make_scorer(accuracy_score),
    "f1_macro": make_scorer(f1_score, average="macro"),
    "f1_weighted": make_scorer(f1_score, average="weighted"),
    "precision_macro": make_scorer(precision_score, average="macro"),
    "recall_macro": make_scorer(recall_score, average="macro")
}

# Calcular número de muestras para SMOTE (balancear clase 3 con respecto a la clase mayoritaria)
counts = y.value_counts()
max_count = counts.max()
smote_strategy = {3: max_count}  # Genera suficientes muestras para igualar la clase mayoritaria

######################################################GRID SEARCH MANUAL#####################################################################################################

for n in param_grid["n_estimators"]:
    for d in param_grid["max_depth"]:
        for split in param_grid["min_samples_split"]:
            for leaf in param_grid["min_samples_leaf"]:
                pipeline = Pipeline([
                    ("smote", SMOTE(sampling_strategy=smote_strategy, random_state=42)),
                    ("rf", RandomForestClassifier(
                        n_estimators=n,
                        max_depth=d,
                        min_samples_split=split,
                        min_samples_leaf=leaf,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1
                    ))
                ])
                
                scores = {}
                for metric_name, scorer in scorers.items():
                    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer, n_jobs=-1)
                    scores[metric_name] = cv_scores.mean()
                
                resultados.append({
                    "n_estimators": n,
                    "max_depth": d if d is not None else -1,
                    "min_samples_split": split,
                    "min_samples_leaf": leaf,
                    **scores
                })

df_resultados = pd.DataFrame(resultados)
df_resultados.sort_values(by="f1_macro", ascending=False, inplace=True)

######################################################FUNCION PARA HEATMAP######################################################################################################

def guardar_heatmap(df, value_col, titulo, archivo, cmap="viridis", fmt=".3f"):
    pivot = (
        df.groupby(["max_depth", "n_estimators"])[value_col]
        .mean()
        .reset_index()
        .pivot(index="max_depth", columns="n_estimators", values=value_col)
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, cbar_kws={"label": value_col})
    plt.title(titulo)
    plt.xlabel("n_estimators")
    plt.ylabel("max_depth")
    plt.tight_layout()
    plt.savefig(f"{HEATMAP_DIR}/{archivo}")
    plt.close()
    print(f"Guardado: {archivo}")


######################################################SELECCIONAR MEJORES HIPERPARAMETROS#################################################################################################

best_params = df_resultados.iloc[0]
rf_best = Pipeline([
    ("smote", SMOTE(sampling_strategy=smote_strategy, random_state=42)),
    ("rf", RandomForestClassifier(
        n_estimators=int(best_params["n_estimators"]),
        max_depth=None if best_params["max_depth"]==-1 else int(best_params["max_depth"]),
        min_samples_split=int(best_params["min_samples_split"]),
        min_samples_leaf=int(best_params["min_samples_leaf"]),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

######################################################RESULTADOS POR FOLD#################################################################################################

fold_results = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    rf_best.fit(X_train, y_train)
    y_pred_fold = rf_best.predict(X_test)

    fold_results.append({
        "fold": fold + 1,
        "accuracy": accuracy_score(y_test, y_pred_fold),
        "f1_macro": f1_score(y_test, y_pred_fold, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred_fold, average="weighted"),
        "precision_macro": precision_score(y_test, y_pred_fold, average="macro"),
        "recall_macro": recall_score(y_test, y_pred_fold, average="macro")
    })

df_fold_results = pd.DataFrame(fold_results)
print("\n=== Resultados por fold ===")
print(df_fold_results)

summary = df_fold_results.drop(columns="fold").agg(["mean", "std"])
print("\n=== Resumen promedio y desviación estándar ===")
print(summary)

y_pred = cross_val_predict(rf_best, X, y, cv=cv, n_jobs=-1)
print("\n=== Classification Report Global ===")
print(classification_report(y, y_pred))

print(best_params)

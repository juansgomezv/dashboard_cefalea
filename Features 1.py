import pandas as pd
import numpy as np
import json
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

# =============================================================
# Configuración Global
# =============================================================
# Opción 1: Cantidad de Clases del IndiceDolor
#   3 -> Agrupa niveles 3 y 4 en el nivel 2. (Resultante: 0, 1, 2)
#   5 -> Mantiene las clases originales (0, 1, 2, 3, 4).
NUM_CLASES = 3

# Opción 2: Estrategia de Pesos para el GridSearch
#   "ninguno"          -> K-Modes Estándar (Sin pesos).
#   "pesos_crudos"     -> K-Modes con influencia fija (Valor exacto Cramér's V).
#   "pesos_norm"       -> K-Modes con influencia fija (Cramér's V normalizado a 1).
#   "pesos_dinamicos"  -> wk-modes: El algoritmo aprende pesos por entropía.
ESTRATEGIA = "ninguno"

def load_dataset(file_path='datasetv5.csv'): 
    df = pd.read_csv(file_path)
    # Primero rellenar nulos, luego convertir a string
    return df.fillna("NA").astype(str)

# =============================================================
# Configurar Cramer's V (Robustecido)
# =============================================================

def compute_cramers_v(col, labels):
    # Si la columna es constante, no aporta información -> Cramér = 0
    if col.nunique() <= 1:
        return 0.0, 1.0, 0.0
        
    contingency = pd.crosstab(col, labels)
    chi2, p, _, _ = chi2_contingency(contingency)
    n = contingency.sum().sum()
    phi2 = chi2 / n
    r, k = contingency.shape
    
    phi2_corr = max(0, phi2 - ((k - 1)*(r - 1))/(n - 1))
    r_corr = r - (r - 1)**2/(n - 1)
    k_corr = k - (k - 1)**2/(n - 1)
    
    divisor = min((k_corr - 1), (r_corr - 1))
    
    # Protección contra división por cero o negativos
    if divisor <= 0:
        return 0.0, 1.0, 0.0
        
    cramers_v = np.sqrt(phi2_corr / divisor)
    
    # IMPORTANTE: Convertir de numpy.float64 a float nativo de Python
    return float(chi2), float(p), float(cramers_v)

# =============================================================
# Lógica Principal
# =============================================================

def run_feature_selection(target='IndiceDolor', top_n=30):
    print(f"--- INICIANDO PROCESO DE SELECCIÓN ---")
    print(f"Configuración: {NUM_CLASES} Clases | Estrategia: {ESTRATEGIA}")
    
    print("1. Cargando y procesando dataset...")
    df = load_dataset()
    
    # Procesamiento de Clases
    if NUM_CLASES == 3:
        print("   -> Aplicando reducción a 3 clases (3 y 4 se vuelven 2)...")
        df.loc[df[target].isin(['3', '4']), target] = '2'
    else:
        print("   -> Manteniendo 5 clases originales...")

    # Guardado
    df.to_csv("dataset_procesado.csv", index=False)
    print(f"   -> Dataset procesado guardado.")

    # Selección
    print("\n2. Calculando Cramér's V...")
    cramer_results = []
    for col in df.columns:
        if col == target: continue
        try:
            # Desempaquetamos los 3 valores que retorna la función
            chi2, p, v = compute_cramers_v(df[col], df[target])
            cramer_results.append([col, v])
        except Exception as e:
            print(f"Error en columna {col}: {e}")
            cramer_results.append([col, 0.0])

    cramer_df = pd.DataFrame(cramer_results, columns=['feature', 'cramers_v'])
    
    print("3. Calculando Información Mutua...")
    X = df.drop(columns=[target])
    y = df[target]
    
    # Codificación simple para MI
    X_enc = X.apply(LabelEncoder().fit_transform)
    y_enc = LabelEncoder().fit_transform(y)
    
    mi = mutual_info_classif(X_enc, y_enc, discrete_features=True, random_state=42)
    mi_df = pd.DataFrame({'feature': X.columns, 'mi': mi})

    print("4. Seleccionando Top 30 variables...")
    combined = cramer_df.merge(mi_df, on='feature', how='inner')
    combined['score'] = combined['cramers_v'].rank(ascending=False) + combined['mi'].rank(ascending=False)
    combined = combined.sort_values('score')
    
    top_30_df = combined.head(top_n).copy()
    final_features = top_30_df['feature'].tolist()

    # Cálculo de Pesos (Asegurando floats puros)
    # Convertimos explícitamente a float para evitar problemas con JSON
    raw_weights = {k: float(v) for k, v in top_30_df.set_index('feature')['cramers_v'].to_dict().items()}
    
    total_sum = top_30_df['cramers_v'].sum()
    top_30_df['norm_weight'] = top_30_df['cramers_v'] / total_sum
    norm_weights = {k: float(v) for k, v in top_30_df.set_index('feature')['norm_weight'].to_dict().items()}

    # Exportación
    print("\n5. Generando archivo de configuración...")
    
    config_data = {
        "num_clases": int(NUM_CLASES), # Aseguramos int nativo
        "estrategia": ESTRATEGIA,
        "features": final_features,
        "pesos_crudos": raw_weights,
        "pesos_norm": norm_weights,
        "target": target
    }
    
    with open("config_transfer.json", "w") as f:
        json.dump(config_data, f, indent=4)
        
    print(f"   -> Configuración guardada correctamente.")
    
    with pd.ExcelWriter(f"reporte_features_{NUM_CLASES}clases.xlsx", engine="openpyxl") as writer:
        combined.to_excel(writer, index=False, sheet_name="Ranking_Completo")
        top_30_df.to_excel(writer, index=False, sheet_name="Top_30_Pesos")

if __name__ == "__main__":
    run_feature_selection(target='IndiceDolor', top_n=30)
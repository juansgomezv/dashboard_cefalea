import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. CONFIGURACIÓN ACADÉMICA (Times New Roman 12)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

# 2. DATOS EXTRAÍDOS DE LAS MATRICES (image_f94d7c.png)
# VP = Pred 1 / Real 1 | FN = Pred 0 / Real 1 | FP = Pred 1 / Real 0
data = {
    'Clase 0': [16, 8, 3],  # Tabla 23
    'Clase 1': [42, 11, 10], # Tabla 24
    'Clase 2': [8, 10, 3],  # Tabla 25
    'Clase 3': [0, 4, 0],   # Tabla 26
    'Clase 4': [0, 1, 0]    # Tabla 27
}

df_cm = pd.DataFrame(data, index=['Verdaderos Positivos', 'Falsos Negativos', 'Falsos Positivos']).T

# 3. CREACIÓN DEL HEATMAP
plt.figure(figsize=(10, 6))
sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=True)

# 4. ETIQUETAS Y TÍTULO
plt.title('Mapa de Calor Unificado: Errores y Aciertos por Clase', fontweight='bold', pad=20)
plt.ylabel('Clase del Índice de Dolor', fontweight='bold')
plt.xlabel('Métrica de Predicción', fontweight='bold')

plt.tight_layout()
# plt.savefig('heatmap_unificado_inferencia.png', dpi=300)
plt.show()
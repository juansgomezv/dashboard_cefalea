import matplotlib.pyplot as plt

# ==========================================
# 1. Configuración de fuente y estilo
# ==========================================
# Configurar Times New Roman, tamaño 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

# ==========================================
# 2. Datos (Orden exacto que solicitaste)
# ==========================================
modelos = ['SVM', 'XGB', 'Random Forest']
valores = [86.297, 82.583, 77.980]

# Colores: Azul para el primero (SVM ganador), Gris para el resto
colores = ['#5470c6', '#a0a0a0', '#a0a0a0']

# ==========================================
# 3. Creación de la gráfica
# ==========================================
fig, ax = plt.subplots(figsize=(8, 5))

# Crear las barras
barras = ax.bar(modelos, valores, color=colores, width=0.7)

# Ajustar los límites del eje Y para que no empiece desde cero 
# (he puesto de 75 a 90 para que las barras se vean proporcionadas con los nuevos valores)
ax.set_ylim(75.0, 90.0)

# Etiqueta del eje Y
ax.set_ylabel('F1-macro (%)')

# Añadir el texto con el porcentaje encima de cada barra
for barra in barras:
    altura = barra.get_height()
    # Posicionar el texto un poquito por encima de la barra
    ax.text(
        barra.get_x() + barra.get_width() / 2, 
        altura + 0.3, 
        f"{altura:.3f}%", # Ajustado a 3 decimales para que coincida con tus datos
        ha='center', 
        va='bottom'
    )

# Ajustar márgenes para que nada quede cortado
plt.tight_layout()

# ==========================================
# 4. Mostrar y guardar
# ==========================================
# Guarda la imagen en alta resolución
plt.savefig('grafica_modelos_f1_orden_original.png', dpi=300)
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo (usando el nombre de archivo disponible)
df = pd.read_csv('comparacion_ear_20250818_123005.csv')
orig = df['EAR_Original']
mejorado = df['EAR_Mejorado']
tiempo = df['Tiempo(s)']

# Configuración del gráfico
plt.figure(figsize=(10, 5))

plt.plot(tiempo, orig, label='EAR Original', color='blue', alpha=0.7, linewidth=1)
plt.plot(tiempo, mejorado, label='EAR Mejorado', color='green', alpha=0.9, linewidth=1)

plt.title('4. Variación Temporal (Fidelidad y Estabilidad)', fontsize=14)
plt.xlabel('Tiempo (s)')
plt.ylabel('Valor EAR')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
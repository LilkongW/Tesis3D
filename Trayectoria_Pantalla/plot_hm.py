import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar datos
file_path = r'C:\Users\Victor\Documents\Tesis3D\Data\Experimento_1\Victor_data\Victor_9_puntos_intento_1_data.csv'
df = pd.read_csv(file_path)
df_valid = df[df['valid_deteccion']].copy() # Usamos .copy() para evitar advertencias de pandas

# ---------------------------------------------------------
# PASO NUEVO: Suavizado de datos (Rolling Mean)
# ---------------------------------------------------------
# 'window': Cantidad de puntos a promediar. 
#   - Valor bajo (ej. 5): Poco suavizado, mantiene detalles rápidos.
#   - Valor alto (ej. 20): Muy suave, pero puede perder movimientos rápidos.
window_size = 15 

df_valid['gaze_x_smooth'] = df_valid['gaze_x'].rolling(window=window_size, center=True).mean()
df_valid['gaze_y_smooth'] = df_valid['gaze_y'].rolling(window=window_size, center=True).mean()

# Eliminamos los valores NaN que se generan en los bordes por el suavizado
df_valid = df_valid.dropna(subset=['gaze_x_smooth', 'gaze_y_smooth'])

# 2. Configurar gráfico
plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid")

# 3. Crear Heatmap (Usando las columnas _smooth)
sns.kdeplot(
    x=df_valid['gaze_x_smooth'], 
    y=df_valid['gaze_y_smooth'], 
    cmap="rocket_r", 
    fill=True, 
    thresh=0.01, 
    levels=15, 
    alpha=0.8
)

# 4. Superponer trayectoria tenue (Usando las columnas _smooth)
plt.plot(
    df_valid['gaze_x_smooth'], 
    df_valid['gaze_y_smooth'], 
    color='black', 
    linewidth=1.5, # Aumenté un poco el grosor para que se note la curva suave
    alpha=0.4
)

# 5. Invertir los ejes
plt.gca().invert_xaxis()  
plt.gca().invert_yaxis()  

plt.title('Mapa de Calor con Trayectoria Suavizada')
plt.xlabel('Posicion X')
plt.ylabel('Posicion Y')

plt.show()
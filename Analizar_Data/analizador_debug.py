import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import sys
import os

# --- 1. CONFIGURACIÓN Y PARÁMETROS ---

# Archivo de entrada
INPUT_FILE = "/home/vit/Documentos/Tesis3D/Data/Victor_data/grabacion_experimento_ESP32CAM_1_ROI_640x480_data.csv"

# --- Archivos de Salida ---
OUTPUT_DIR = "/home/vit/Documentos/Tesis3D/Analizar_Data/Resultados/Victor_data"
OUTPUT_REPORT_FIXATIONS = os.path.join(OUTPUT_DIR, "reporte_fijaciones_IAT.csv")
OUTPUT_REPORT_SACCADES = os.path.join(OUTPUT_DIR, "reporte_sacadicos_IAT.csv")
OUTPUT_PLOT_COMBINED = os.path.join(OUTPUT_DIR, "grafico_combinado_velocidad_aceleracion_IAT.png")
OUTPUT_PLOT_FIXATIONS = os.path.join(OUTPUT_DIR, "grafico_fijaciones_IAT.png")
OUTPUT_PLOT_SACCADES = os.path.join(OUTPUT_DIR, "grafico_sacadicos_IAT.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Umbrales de Detección ---
UMBRALES_3D = {
    # Umbrales de Aceleración (I-AT) para ENCONTRAR sacádicos
    'A_PICO_MINIMO': 540.0, # Umbral de aceleración (positiva y negativa)
    
    # Umbral de Velocidad (I-VT) para ENCONTRAR fijaciones
    'V_MAXIMA_FIJACION': 35.0,
    
    # Umbrales de Duración y Búsqueda
    'T_MIN_FIJACION_MS': 200,
    'T_MAX_ENTRE_PICOS_MS': 180, # Máximo tiempo entre pico de acel. y decel.
}

# Parámetros del filtro Savitzky-Golay
SAVGOL_WINDOW = 5
SAVGOL_POLY = 2

def calcular_velocidad_y_aceleracion(df):
    """
    Calcula velocidad y aceleración, y las filtra.
    """
    print("Calculando velocidad y aceleración...")
    
    gaze_cols = ['gaze_x', 'gaze_y', 'gaze_z']
    df.loc[~df['valid_deteccion'], gaze_cols] = np.nan
    
    df['timestamp_s'] = df['timestamp_ms'] / 1000.0
    df['delta_t'] = df['timestamp_s'].diff()

    # --- Cálculo de Velocidad ---
    df['gaze_x_prev'] = df['gaze_x'].shift()
    df['gaze_y_prev'] = df['gaze_y'].shift()
    df['gaze_z_prev'] = df['gaze_z'].shift()
    df['dot_product'] = (df['gaze_x'] * df['gaze_x_prev'] +
                         df['gaze_y'] * df['gaze_y_prev'] +
                         df['gaze_z'] * df['gaze_z_prev'])
    df['dot_product_clipped'] = df['dot_product'].clip(-1.0, 1.0)
    df['angular_dist_rad'] = np.arccos(df['dot_product_clipped'])
    df['angular_dist_deg'] = np.degrees(df['angular_dist_rad'])
    df['velocity_angular_deg_s'] = df['angular_dist_deg'] / df['delta_t']

    velocity_filled = df['velocity_angular_deg_s'].fillna(0)
    df['velocity_angular_filtered'] = savgol_filter(velocity_filled, 
                                                    window_length=SAVGOL_WINDOW,
                                                    polyorder=SAVGOL_POLY)
    
    # --- Cálculo de Aceleración ---
    df['acceleration_angular'] = df['velocity_angular_filtered'].diff() / df['delta_t']
    acceleration_filled = df['acceleration_angular'].fillna(0)
    df['acceleration_angular_filtered'] = savgol_filter(acceleration_filled,
                                                        window_length=SAVGOL_WINDOW,
                                                        polyorder=SAVGOL_POLY)
    
    # Rellenar NaNs al inicio para evitar problemas
    df.fillna(0, inplace=True)
    return df

# --- LÓGICA DE DETECCIÓN I-AT (MODIFICADA) ---

def detectar_sacadicos_por_aceleracion(df, umbrales):
    """
    Implementa la lógica "sacadico-primero" (I-AT).
    Encuentra perfiles bifásicos de aceleración.
    El evento se define DESDE el pico de aceleración HASTA el pico de deceleración.
    """
    print("Detectando sacádicos por perfil de aceleración (I-AT)...")
    
    # 1. Encontrar todos los picos de aceleración y deceleración
    umbral_acel = umbrales['A_PICO_MINIMO']
    indices_picos_pos, _ = find_peaks(df['acceleration_angular_filtered'], height=umbral_acel)
    indices_picos_neg_raw, _ = find_peaks(-df['acceleration_angular_filtered'], height=umbral_acel)
    
    indices_df_picos_pos = df.index[indices_picos_pos]
    indices_df_picos_neg = df.index[indices_picos_neg_raw]
    
    sacadicos_validos = []
    indices_picos_neg_usados = set() 

    # 2. Iterar sobre los "arranques" (picos positivos)
    for idx_pico_pos in indices_df_picos_pos:
        
        # 3. Encontrar el "frenazo" (pico negativo) más cercano DESPUÉS del arranque
        candidatos_pico_neg = [
            idx for idx in indices_df_picos_neg 
            if idx > idx_pico_pos and idx not in indices_picos_neg_usados
        ]
        
        if not candidatos_pico_neg:
            continue
        
        idx_pico_neg = min(candidatos_pico_neg) 
        
        # 4. Validar Perfil: ¿Están los picos lo suficientemente cerca?
        tiempo_pico_pos_ms = df.at[idx_pico_pos, 'timestamp_ms']
        tiempo_pico_neg_ms = df.at[idx_pico_neg, 'timestamp_ms']
        
        if (tiempo_pico_neg_ms - tiempo_pico_pos_ms) > umbrales['T_MAX_ENTRE_PICOS_MS'] or (tiempo_pico_neg_ms - tiempo_pico_pos_ms <= 0):
            continue # Perfil bifásico demasiado lento o inválido
            
        # 5. ¡PERFIL DE ACELERACIÓN VÁLIDO!
        
        # 6. MODIFICADO: El evento se define por los picos de aceleración
        idx_inicio_real = idx_pico_pos
        idx_fin_real = idx_pico_neg
        
        # 7. Recopilar datos del evento
        evento_inicio = df.loc[idx_inicio_real]
        evento_fin = df.loc[idx_fin_real]
        
        # 8. Calcular velocidad promedio en esta "zona"
        segmento_vel_evento = df.loc[idx_inicio_real:idx_fin_real, 'velocity_angular_filtered']
        velocidad_promedio = 0.0
        if not segmento_vel_evento.empty:
            velocidad_promedio = segmento_vel_evento.mean()
        
        # 9. Calcular amplitud (ángulo entre vector de inicio y fin)
        v_start = evento_inicio[['gaze_x', 'gaze_y', 'gaze_z']].values
        v_end = evento_fin[['gaze_x', 'gaze_y', 'gaze_z']].values
        dot_prod_amp = np.dot(v_start, v_end)
        dot_prod_amp_clipped = np.clip(dot_prod_amp, -1.0, 1.0)
        amplitude_rad = np.arccos(dot_prod_amp_clipped)
        amplitude_deg = np.degrees(amplitude_rad)

        sacadicos_validos.append({
            'event_id': f"s_{idx_pico_pos}",
            'event_type': 'sacadico',
            'event_class': 'sacadico',
            'start_time_ms': evento_inicio['timestamp_ms'],
            'end_time_ms': evento_fin['timestamp_ms'],
            'duration_ms': evento_fin['timestamp_ms'] - evento_inicio['timestamp_ms'],
            'amplitude_deg': amplitude_deg,
            'velocidad_promedio_deg_s': velocidad_promedio 
        })
        indices_picos_neg_usados.add(idx_pico_neg) 
            
    return pd.DataFrame(sacadicos_validos)

def detectar_fijaciones_restantes(df, sacadicos_df, umbrales):
    """
    Encuentra fijaciones en los datos que NO están clasificados como sacádicos.
    """
    print("Detectando fijaciones en los segmentos restantes...")
    
    mascara_fijacion = pd.Series(True, index=df.index)
    for _, sac in sacadicos_df.iterrows():
        mascara_fijacion.loc[
            (df['timestamp_ms'] >= sac['start_time_ms']) & 
            (df['timestamp_ms'] <= sac['end_time_ms'])
        ] = False
    
    df_fijaciones = df[mascara_fijacion & (df['velocity_angular_filtered'] <= umbrales['V_MAXIMA_FIJACION'])].copy()
    
    columnas_reporte = ['event_id', 'event_type', 'event_class', 'start_time_ms', 'end_time_ms', 'duration_ms', 'amplitude_deg', 'velocidad_promedio_deg_s']
    if df_fijaciones.empty:
        return pd.DataFrame(columns=columnas_reporte)

    df_fijaciones['group_id'] = (df_fijaciones.index.to_series().diff() != 1).cumsum()
    
    fijaciones_validas = []
    for group_id, group_df in df_fijaciones.groupby('group_id'):
        duration_ms = group_df['timestamp_ms'].iloc[-1] - group_df['timestamp_ms'].iloc[0]
        
        if duration_ms >= umbrales['T_MIN_FIJACION_MS']:
            velocidad_promedio_fij = group_df['velocity_angular_filtered'].mean()
            
            fijaciones_validas.append({
                'event_id': f"f_{group_id}",
                'event_type': 'fijacion',
                'event_class': 'fijacion_valida',
                'start_time_ms': group_df['timestamp_ms'].iloc[0],
                'end_time_ms': group_df['timestamp_ms'].iloc[-1],
                'duration_ms': duration_ms,
                'amplitude_deg': 0.0,
                'velocidad_promedio_deg_s': velocidad_promedio_fij 
            })
            
    return pd.DataFrame(fijaciones_validas)

# --- Funciones de Graficación (Sin cambios) ---

def generar_visualizacion_combinada(df_raw, events_report, umbrales, output_file):
    print(f"Generando gráfico combinado en {output_file}...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    event_color_map = {'fijacion_valida': 'green', 'sacadico': 'red'}

    # Subplot 1: Posición
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_x'], label='dirección_x', alpha=0.8)
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_y'], label='dirección_y', linestyle='--', alpha=0.8)
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_z'], label='dirección_z (adelante)', linestyle=':', alpha=0.6)
    ax1.set_title('Posición del Vector de Mirada 3D')
    ax1.set_ylabel('Componente del Vector')
    ax1.legend()
    ax1.grid(True)

    # Subplot 2: Velocidad
    ax2.plot(df_raw['timestamp_s'], df_raw['velocity_angular_filtered'], label='Velocidad Angular Filtrada (°/s)', color='blue', alpha=0.7)
    ax2.axhline(y=umbrales['V_MAXIMA_FIJACION'], color='green', linestyle=':', label=f"Umbral Fijación ({umbrales['V_MAXIMA_FIJACION']} °/s)")
    ax2.set_title('Velocidad Angular 3D y Eventos Oculares Detectados')
    ax2.set_ylabel('Velocidad Angular (°/s)')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    # Subplot 3: Aceleración
    ax3.plot(df_raw['timestamp_s'], df_raw['acceleration_angular_filtered'], label='Aceleración Angular Filtrada (°/s²)', color='purple', alpha=0.7)
    ax3.axhline(y=umbrales['A_PICO_MINIMO'], color='orange', linestyle=':', label=f"Umbral Acel. Pos. ({umbrales['A_PICO_MINIMO']} °/s²)")
    ax3.axhline(y=-umbrales['A_PICO_MINIMO'], color='orange', linestyle=':', label=f"Umbral Acel. Neg. ({-umbrales['A_PICO_MINIMO']} °/s²)")
    ax3.set_title('Aceleración Angular 3D y Detección de Sacádicos (I-AT)')
    ax3.set_xlabel('Tiempo (s)')
    ax3.set_ylabel('Aceleración (°/s²)')
    ax3.legend(loc='upper left')
    ax3.grid(True)

    # Dibujar regiones de eventos
    for _, event in events_report.iterrows():
        color = event_color_map.get(event['event_class'])
        if color:
            ax2.axvspan(event['start_time_ms'] / 1000.0, event['end_time_ms'] / 1000.0, color=color, alpha=0.3)
            ax3.axvspan(event['start_time_ms'] / 1000.0, event['end_time_ms'] / 1000.0, color=color, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file)
    print("Gráfico combinado guardado.")

def generar_visualizacion_fijaciones(df_raw, events_report, umbrales, output_file):
    print(f"Generando gráfico de fijaciones en {output_file}...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    event_color_map = { 'fijacion_valida': 'green' }

    # Subplot 1: Posición
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_x'], label='dirección_x', alpha=0.8)
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_y'], label='dirección_y', linestyle='--', alpha=0.8)
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_z'], label='dirección_z (adelante)', linestyle=':', alpha=0.6)
    ax1.set_title('Posición de Mirada y Fijaciones Detectadas')
    ax1.set_ylabel('Componente del Vector')
    ax1.legend()
    ax1.grid(True)

    # Subplot 2: Velocidad
    ax2.plot(df_raw['timestamp_s'], df_raw['velocity_angular_filtered'], label='Velocidad Angular Filtrada (°/s)', color='blue', alpha=0.7)
    ax2.axhline(y=umbrales['V_MAXIMA_FIJACION'], color='green', linestyle=':', label=f"Umbral Fijación ({umbrales['V_MAXIMA_FIJACION']} °/s)")
    ax2.set_title('Velocidad y Fijaciones Detectadas')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Velocidad Angular (°/s)')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    # Dibujar regiones
    fixations_report = events_report[events_report['event_class'] == 'fijacion_valida']
    for _, event in fixations_report.iterrows():
        color = event_color_map.get(event['event_class'])
        if color:
            ax1.axvspan(event['start_time_ms'] / 1000.0, event['end_time_ms'] / 1000.0, color=color, alpha=0.3)
            ax2.axvspan(event['start_time_ms'] / 1000.0, event['end_time_ms'] / 1000.0, color=color, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file)
    print("Gráfico de fijaciones guardado.")

def generar_visualizacion_sacadicos(df_raw, events_report, umbrales, output_file):
    print(f"Generando gráfico de sacádicos en {output_file}...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    event_color_map = { 'sacadico': 'red' }

    # Subplot 1: Posición
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_x'], label='dirección_x', alpha=0.8)
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_y'], label='dirección_y', linestyle='--', alpha=0.8)
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_z'], label='dirección_z (adelante)', linestyle=':', alpha=0.6)
    ax1.set_title('Posición de Mirada y Sacádicos (Detectados por Acel.)')
    ax1.set_ylabel('Componente del Vector')
    ax1.legend()
    ax1.grid(True)

    # Subplot 2: Velocidad
    ax2.plot(df_raw['timestamp_s'], df_raw['velocity_angular_filtered'], label='Velocidad Angular Filtrada (°/s)', color='blue', alpha=0.7)
    ax2.set_title('Velocidad y Sacádicos (Detectados por Acel.)')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Velocidad Angular (°/s)')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    # Dibujar regiones
    saccades_report = events_report[events_report['event_class'] == 'sacadico']
    for _, event in saccades_report.iterrows():
        color = event_color_map.get(event['event_class'])
        if color:
            ax1.axvspan(event['start_time_ms'] / 1000.0, event['end_time_ms'] / 1000.0, color=color, alpha=0.3)
            ax2.axvspan(event['start_time_ms'] / 1000.0, event['end_time_ms'] / 1000.0, color=color, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file)
    print("Gráfico de sacádicos guardado.")

# --- FUNCIÓN PRINCIPAL ---

def main():
    """
    Función principal que ejecuta el pipeline "Sacadico-Primero" (I-AT).
    """
    print(f"Iniciando análisis 3D (I-AT) del archivo: {INPUT_FILE}")
    try:
        df_original = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo '{INPUT_FILE}'.")
        print("Por favor, asegúrate de que el archivo CSV está en la ubicación especificada.")
        sys.exit(1)
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo: {e}")
        sys.exit(1)

    # --- 1. Calcular Velocidad y Aceleración ---
    df_completo = calcular_velocidad_y_aceleracion(df_original.copy())
    
    # --- 2. Detección "Sacadico-Primero" ---
    sacadicos_df = detectar_sacadicos_por_aceleracion(df_completo, UMBRALES_3D)
    
    # --- 3. Detección de Fijaciones (en lo restante) ---
    fijaciones_df = detectar_fijaciones_restantes(df_completo, sacadicos_df, UMBRALES_3D)
    
    # --- 4. Combinar Reporte (solo para gráficos) ---
    reporte_final = pd.concat([sacadicos_df, fijaciones_df]).sort_values(by='start_time_ms')
    
    # --- 5. Guardar Reportes CSV Separados ---
    try:
        fijaciones_df.to_csv(OUTPUT_REPORT_FIXATIONS, index=False)
        sacadicos_df.to_csv(OUTPUT_REPORT_SACCADES, index=False)
        
    except Exception as e:
        print(f"ERROR: No se pudo guardar uno o más reportes CSV en '{OUTPUT_DIR}'. Verifica los permisos y la ruta.")
        print(f"Error: {e}")
        sys.exit(1)

    # --- Imprimir Resumen ---
    print("\n--- Reporte Final de Eventos (Detectado por I-AT) ---")
    if reporte_final.empty:
        print("No se detectaron eventos válidos con los umbrales actuales.")
    else:
        print(reporte_final['event_class'].value_counts())
    
    print(f"\nReporte de FIJACIONES guardado en: {OUTPUT_REPORT_FIXATIONS}")
    print(f"Reporte de SACÁDICOS guardado en: {OUTPUT_REPORT_SACCADES}")
    
    # --- Generar Gráficos ---
    try:
        generar_visualizacion_combinada(df_completo, reporte_final, UMBRALES_3D, OUTPUT_PLOT_COMBINED)
        generar_visualizacion_fijaciones(df_completo, reporte_final, UMBRALES_3D, OUTPUT_PLOT_FIXATIONS)
        generar_visualizacion_sacadicos(df_completo, reporte_final, UMBRALES_3D, OUTPUT_PLOT_SACCADES)
    except Exception as e:
        print(f"ERROR: No se pudo guardar uno o más gráficos en '{OUTPUT_DIR}'. Verifica los permisos y la ruta.")
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"\nAnálisis 3D (I-AT) completado. Se generaron 3 gráficos y 2 reportes en: {OUTPUT_DIR}")

# --- Ejecutar el script ---
if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import sys
import os
import glob 

# --- 1. CONFIGURACIÓN Y PARÁMETROS ---

# Directorio de ENTRADA
INPUT_DIR = "/home/vit/Documentos/Tesis3D/Data/Victor_data"

# Directorio de SALIDA
OUTPUT_DIR = "/home/vit/Documentos/Tesis3D/Analizar_Data/Resultados/Victor_data"

# Nombres de archivos de reporte y gráficos
OUTPUT_REPORT_AGGREGATE = os.path.join(OUTPUT_DIR, "reporte_agregado_general.csv")
# Los nombres individuales se generan dinámicamente

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Umbrales de Detección ---
UMBRALES_3D = {
    # Umbrales de Aceleración (I-AT) para ENCONTRAR sacádicos
    'A_PICO_MINIMO': 500.0, 
    
    # Umbral de Velocidad (I-VT) para ENCONTRAR fijaciones
    'V_MAXIMA_FIJACION': 30.0,
    
    # Umbrales de Duración y Búsqueda
    'T_MIN_FIJACION_MS': 250,
    'T_MAX_ENTRE_PICOS_MS': 200, # Máximo tiempo entre pico de acel. y decel.
}

# Parámetros del filtro Savitzky-Golay
SAVGOL_WINDOW = 5
SAVGOL_POLY = 2

# --- FUNCIONES DE CÁLCULO (Sin Cambios) ---

def calcular_velocidad_y_aceleracion(df):
    """
    Calcula velocidad y aceleración, y las filtra.
    """
    print("Calculando velocidad y aceleración...")
    
    gaze_cols = ['gaze_x', 'gaze_y', 'gaze_z']
    df.loc[~df['valid_deteccion'], gaze_cols] = np.nan
    
    df['timestamp_s'] = df['timestamp_ms'] / 1000.0
    df['delta_t'] = df['timestamp_s'].diff()

    # Cálculo de Velocidad
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
    
    # Cálculo de Aceleración
    df['acceleration_angular'] = df['velocity_angular_filtered'].diff() / df['delta_t']
    acceleration_filled = df['acceleration_angular'].fillna(0)
    df['acceleration_angular_filtered'] = savgol_filter(acceleration_filled,
                                                        window_length=SAVGOL_WINDOW,
                                                        polyorder=SAVGOL_POLY)
    
    df.fillna(0, inplace=True)
    return df

# --- LÓGICA DE DETECCIÓN I-AT (MODIFICADA) ---

# MODIFICADO: Eliminada la función encontrar_inicio_fin_sacadico

def detectar_sacadicos_por_aceleracion(df, umbrales):
    """
    Implementa la lógica "sacadico-primero" (I-AT).
    Encuentra perfiles bifásicos de aceleración.
    El evento se define ESTRICTAMENTE DESDE el pico de aceleración HASTA el pico de deceleración.
    """
    print("Detectando sacádicos por perfil de aceleración (I-AT estricto)...")
    
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
        
        # Asegurarse de que la duración sea positiva y dentro del límite
        duracion_picos_ms = tiempo_pico_neg_ms - tiempo_pico_pos_ms
        if not (0 < duracion_picos_ms <= umbrales['T_MAX_ENTRE_PICOS_MS']):
            continue
            
        # 5. ¡PERFIL DE ACELERACIÓN VÁLIDO!
        
        # 6. MODIFICADO: El inicio y fin REALES son los picos de aceleración
        idx_inicio_real = idx_pico_pos
        idx_fin_real = idx_pico_neg
        
        # 7. Recopilar datos del evento
        evento_inicio = df.loc[idx_inicio_real]
        evento_fin = df.loc[idx_fin_real]
        
        # 8. Calcular velocidad promedio en esta "zona" (entre picos de acel.)
        segmento_vel_evento = df.loc[idx_inicio_real:idx_fin_real, 'velocity_angular_filtered']
        velocidad_promedio = 0.0
        if not segmento_vel_evento.empty:
            velocidad_promedio = segmento_vel_evento.mean()
        
        # 9. Calcular amplitud (ángulo entre vector de inicio y fin, en los picos de acel.)
        v_start = evento_inicio[['gaze_x', 'gaze_y', 'gaze_z']].values
        v_end = evento_fin[['gaze_x', 'gaze_y', 'gaze_z']].values
        dot_prod_amp = np.dot(v_start, v_end)
        # Asegurar que el producto punto esté en [-1, 1] antes de arccos
        dot_prod_amp_clipped = np.clip(dot_prod_amp, -1.0, 1.0) 
        amplitude_rad = np.arccos(dot_prod_amp_clipped)
        amplitude_deg = np.degrees(amplitude_rad)

        sacadicos_validos.append({
            'event_id': f"s_{idx_pico_pos}",
            'event_type': 'sacadico',
            'event_class': 'sacadico',
            'start_time_ms': evento_inicio['timestamp_ms'],
            'end_time_ms': evento_fin['timestamp_ms'],
            # La duración es ahora estrictamente entre los picos
            'duration_ms': duracion_picos_ms, 
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
    # IMPORTANTE: Usamos los start/end times de los sacádicos detectados
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

# --- Funciones de Graficación (Sin Cambios) ---

def generar_visualizacion_combinada(df_raw, events_report, umbrales, output_file):
    print(f"Generando gráfico combinado en {output_file}...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    event_color_map = {'fijacion_valida': 'green', 'sacadico': 'red'}

    # Subplot 1: Posición
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_x'], label='dirección_x', alpha=0.8)
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_y'], label='dirección_y', linestyle='--', alpha=0.8)
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_z'], label='dirección_z (adelante)', linestyle=':', alpha=0.6)
    ax1.set_title(f"Posición del Vector de Mirada 3D\n(Archivo: {os.path.basename(output_file)})")
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
    print(f"Gráfico combinado guardado en: {output_file}")
    plt.close(fig) # Cerrar la figura para liberar memoria

def generar_visualizacion_fijaciones(df_raw, events_report, umbrales, output_file):
    print(f"Generando gráfico de fijaciones en {output_file}...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    event_color_map = { 'fijacion_valida': 'green' }

    # Subplot 1: Posición
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_x'], label='dirección_x', alpha=0.8)
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_y'], label='dirección_y', linestyle='--', alpha=0.8)
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_z'], label='dirección_z (adelante)', linestyle=':', alpha=0.6)
    ax1.set_title(f"Posición de Mirada y Fijaciones Detectadas\n(Archivo: {os.path.basename(output_file)})")
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
    print(f"Gráfico de fijaciones guardado en: {output_file}")
    plt.close(fig) # Cerrar la figura

def generar_visualizacion_sacadicos(df_raw, events_report, umbrales, output_file):
    print(f"Generando gráfico de sacádicos en {output_file}...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    event_color_map = { 'sacadico': 'red' }

    # Subplot 1: Posición
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_x'], label='dirección_x', alpha=0.8)
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_y'], label='dirección_y', linestyle='--', alpha=0.8)
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_z'], label='dirección_z (adelante)', linestyle=':', alpha=0.6)
    ax1.set_title(f"Posición de Mirada y Sacádicos (Detectados por Acel.)\n(Archivo: {os.path.basename(output_file)})")
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
    print(f"Gráfico de sacádicos guardado en: {output_file}")
    plt.close(fig) # Cerrar la figura

# --- FUNCIÓN PRINCIPAL (Sin Cambios) ---

def main():
    """
    Función principal que ejecuta el pipeline "Sacadico-Primero" (I-AT).
    """
    print(f"Iniciando análisis por lotes en el directorio: {INPUT_DIR}")
    
    patron_busqueda = os.path.join(INPUT_DIR, "*.csv")
    lista_archivos_csv = glob.glob(patron_busqueda)
    
    if not lista_archivos_csv:
        print(f"ERROR: No se encontraron archivos .csv en '{INPUT_DIR}'.")
        sys.exit(1)
        
    print(f"Se encontraron {len(lista_archivos_csv)} archivos CSV para analizar.")
    
    reportes_agregados = []

    # --- INICIO DEL BUCLE DE PROCESAMIENTO ---
    for input_file_path in lista_archivos_csv:
        
        base_filename = os.path.splitext(os.path.basename(input_file_path))[0]
        print(f"\n--- Procesando Archivo: {base_filename}.csv ---")
        
        # Nombres de archivo dinámicos
        output_report_fix = os.path.join(OUTPUT_DIR, f"{base_filename}_reporte_fijaciones.csv")
        output_report_sac = os.path.join(OUTPUT_DIR, f"{base_filename}_reporte_sacadicos.csv")
        output_plot_comb = os.path.join(OUTPUT_DIR, f"{base_filename}_grafico_combinado.png")
        output_plot_fix = os.path.join(OUTPUT_DIR, f"{base_filename}_grafico_fijaciones.png")
        output_plot_sac = os.path.join(OUTPUT_DIR, f"{base_filename}_grafico_sacadicos.png")
        
        try:
            df_original = pd.read_csv(input_file_path)
            if df_original.empty:
                print(f"Advertencia: El archivo {base_filename}.csv está vacío. Omitiendo.")
                continue
        except Exception as e:
            print(f"ERROR: No se pudo leer el archivo '{input_file_path}'. Error: {e}")
            continue 
            
        # 1. Calcular Velocidad y Aceleración
        df_completo = calcular_velocidad_y_aceleracion(df_original.copy())
        
        # 2. Detección "Sacadico-Primero"
        sacadicos_df = detectar_sacadicos_por_aceleracion(df_completo, UMBRALES_3D)
        
        # 3. Detección de Fijaciones
        fijaciones_df = detectar_fijaciones_restantes(df_completo, sacadicos_df, UMBRALES_3D)
        
        # 4. Combinar Reporte (para gráficos)
        reporte_final_combinado = pd.concat([sacadicos_df, fijaciones_df]).sort_values(by='start_time_ms')
        
        # 5. Guardar Reportes CSV Separados
        try:
            fijaciones_df.to_csv(output_report_fix, index=False)
            sacadicos_df.to_csv(output_report_sac, index=False)
            print(f"Reportes individuales guardados para: {base_filename}")
        except Exception as e:
            print(f"ERROR: No se pudo guardar el reporte para '{base_filename}'. Error: {e}")
            continue

        # 6. Generar Gráficos
        try:
            generar_visualizacion_combinada(df_completo, reporte_final_combinado, UMBRALES_3D, output_plot_comb)
            generar_visualizacion_fijaciones(df_completo, reporte_final_combinado, UMBRALES_3D, output_plot_fix)
            generar_visualizacion_sacadicos(df_completo, reporte_final_combinado, UMBRALES_3D, output_plot_sac)
        except Exception as e:
            print(f"ERROR: No se pudo guardar gráficos para '{base_filename}'. Error: {e}")
            continue

        # 7. Recopilar datos para el reporte agregado
        try:
            duracion_total_seg = df_completo['timestamp_s'].iloc[-1]
            conteo_fijaciones = len(fijaciones_df)
            conteo_sacadicos = len(sacadicos_df)
            
            reportes_agregados.append({
                'archivo': base_filename,
                'duracion_total_seg': duracion_total_seg,
                'conteo_fijaciones': conteo_fijaciones,
                'conteo_sacadicos': conteo_sacadicos,
                'frecuencia_fijaciones_hz': conteo_fijaciones / duracion_total_seg if duracion_total_seg > 0 else 0,
                'frecuencia_sacadicos_hz': conteo_sacadicos / duracion_total_seg if duracion_total_seg > 0 else 0,
                'duracion_media_fij_ms': fijaciones_df['duration_ms'].mean(),
                'duracion_media_sac_ms': sacadicos_df['duration_ms'].mean(),
                'amplitud_media_sac_deg': sacadicos_df['amplitude_deg'].mean(),
                'vel_media_fij_deg_s': fijaciones_df['velocidad_promedio_deg_s'].mean(),
                'vel_media_sac_deg_s': sacadicos_df['velocidad_promedio_deg_s'].mean(),
            })
        except Exception as e:
            print(f"ERROR: No se pudo calcular el resumen para '{base_filename}'. Error: {e}")

    # --- FIN DEL BUCLE ---

    # --- 8. Generar el Reporte Agregado Final ---
    if not reportes_agregados:
        print("\nNo se pudo procesar ningún archivo exitosamente. No se generará reporte agregado.")
        sys.exit(1)
        
    print("\n--- Análisis por lotes completado. Generando reporte agregado... ---")
    
    try:
        df_agregado = pd.DataFrame(reportes_agregados)
        promedios_generales = df_agregado.mean(numeric_only=True)
        promedios_generales['total_archivos_procesados'] = len(df_agregado)
        
        promedios_generales.name = 'PROMEDIO_GENERAL'
        df_agregado = pd.concat([df_agregado, promedios_generales.to_frame().T])
        
        df_agregado.to_csv(OUTPUT_REPORT_AGGREGATE, index=True)
        
        print("\n--- Resumen General (Promedio de todos los archivos) ---")
        print(promedios_generales)
        print(f"\nReporte agregado general guardado en: {OUTPUT_REPORT_AGGREGATE}")
        print(f"Análisis 3D (I-AT) completado. Se generaron reportes y gráficos en: {OUTPUT_DIR}")

    except Exception as e:
        print(f"ERROR: No se pudo guardar el reporte agregado general. Error: {e}")
        sys.exit(1)

# --- Ejecutar el script ---
if __name__ == "__main__":
    main()
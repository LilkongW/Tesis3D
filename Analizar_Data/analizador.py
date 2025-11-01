import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import sys
import os
import glob 

# --- 1. CONFIGURACIÓN Y PARÁMETROS ---
NOMBRE = "Victoria"
# Directorio de ENTRADA
EXP_NUM = 1
INPUT_DIR = fr"C:\Users\Victor\Documents\Tesis3D\Data\Experimento_{EXP_NUM}\{NOMBRE}_data"
# Directorio de SALIDA
OUTPUT_DIR = fr"C:\Users\Victor\Documents\Tesis3D\Analizar_Data\Resultados\Experimento_{EXP_NUM}\{NOMBRE}_data"

# Nombres de archivos de reporte y gráficos
OUTPUT_REPORT_AGGREGATE = os.path.join(OUTPUT_DIR, "reporte_agregado_general.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- NUEVA CONFIGURACIÓN: Offset de Sincronización ---
# Ignorar los primeros N milisegundos de los datos del ojo
OFFSET_MS = 500.0

# --- Umbrales de Detección ---
UMBRALES_3D = {
    'A_PICO_MINIMO': 500.0, 
    'V_MAXIMA_FIJACION': 30.0,
    'T_MIN_FIJACION_MS': 250,
    'T_MAX_ENTRE_PICOS_MS': 200,
    'T_MIN_PARPADEO_FRAMES': 2, # Mínimo frames en False para ser parpadeo (user: "mas de dos")
}
SAVGOL_WINDOW = 5
SAVGOL_POLY = 2

# --- NUEVA FUNCIÓN: DEFINICIÓN DE ESTÍMULOS ---
def get_stimulus_events_exp1(offset_ms=0):
    print(f"Generando marcas de tiempo de estímulos para Exp 1 (Offset: {offset_ms}ms)...")
    DURACION_PUNTO_MS = 2000.0
    N_PUNTOS = 9
    eventos_estimulo = []
    posiciones = [
        "P1 (Sup-Izq)", "P2 (Sup-Cen)", "P3 (Sup-Der)",
        "P4 (Med-Izq)", "P5 (Med-Cen)", "P6 (Med-Der)",
        "P7 (Inf-Izq)", "P8 (Inf-Cen)", "P9 (Inf-Der)"
    ]
    for i in range(N_PUNTOS):
        anim_start_time = i * DURACION_PUNTO_MS
        anim_end_time = (i + 1) * DURACION_PUNTO_MS
        eventos_estimulo.append({
            'label': posiciones[i],
            'start_time_ms': anim_start_time + offset_ms,
            'end_time_ms': anim_end_time + offset_ms
        })
    return eventos_estimulo

# --- FUNCIONES DE CÁLCULO (Robustecida) ---
def calcular_velocidad_y_aceleracion(df):
    print("Calculando velocidad y aceleración...")
    gaze_cols = ['gaze_x', 'gaze_y', 'gaze_z']
    
    if 'valid_deteccion' in df.columns:
        if df['valid_deteccion'].dtype == 'object':
             df['valid_deteccion'] = df['valid_deteccion'].map({'True': True, 'False': False}).fillna(False)
        # Aplicar NaNs donde la detección no es válida
        df.loc[~df['valid_deteccion'], gaze_cols] = np.nan
    
    df['timestamp_s'] = df['timestamp_ms'] / 1000.0
    
    df['delta_t'] = df['timestamp_s'].diff().fillna(method='bfill')
    df['delta_t'].fillna(0.0333, inplace=True)

    # Cálculo de Velocidad (los NaNs en gaze_ se propagarán)
    df['gaze_x_prev'] = df['gaze_x'].shift()
    df['gaze_y_prev'] = df['gaze_y'].shift()
    df['gaze_z_prev'] = df['gaze_z'].shift()
    df['dot_product'] = (df['gaze_x'] * df['gaze_x_prev'] +
                         df['gaze_y'] * df['gaze_y_prev'] +
                         df['gaze_z'] * df['gaze_z_prev'])
    df['dot_product_clipped'] = df['dot_product'].clip(-1.0, 1.0)
    df['angular_dist_rad'] = np.arccos(df['dot_product_clipped'])
    df['angular_dist_deg'] = np.degrees(df['angular_dist_rad'])
    
    df['velocity_angular_deg_s'] = 0.0
    mask_delta_t = df['delta_t'] > 0
    # Los NaNs en angular_dist_deg (de parpadeos) se convierten en 0 por fillna(0) más adelante
    df.loc[mask_delta_t, 'velocity_angular_deg_s'] = df['angular_dist_deg'][mask_delta_t] / df['delta_t'][mask_delta_t]

    # Rellenar NaNs (creados por shift y por parpadeos) con 0 ANTES de filtrar
    velocity_filled = df['velocity_angular_deg_s'].fillna(0)
    df['velocity_angular_filtered'] = savgol_filter(velocity_filled, 
                                                    window_length=SAVGOL_WINDOW,
                                                    polyorder=SAVGOL_POLY)
    
    # Cálculo de Aceleración
    df['acceleration_angular'] = 0.0
    df.loc[mask_delta_t, 'acceleration_angular'] = df['velocity_angular_filtered'].diff()[mask_delta_t] / df['delta_t'][mask_delta_t]

    acceleration_filled = df['acceleration_angular'].fillna(0)
    df['acceleration_angular_filtered'] = savgol_filter(acceleration_filled,
                                                        window_length=SAVGOL_WINDOW,
                                                        polyorder=SAVGOL_POLY)
    
    # Rellenar cualquier NaN restante (p.ej. al inicio del df)
    df.fillna(0, inplace=True)
    return df

# --- LÓGICA DE DETECCIÓN ---

def detectar_parpadeos(df, umbrales):
    """
    Detecta parpadeos basado en 'valid_deteccion' == False
    Un parpadeo se define como 'valid_deteccion' == False por MÁS DE N frames consecutivos.
    """
    print("Detectando parpadeos...")
    
    # Asegurarse que la columna es booleana
    if df['valid_deteccion'].dtype == 'object':
         df['valid_deteccion'] = df['valid_deteccion'].map({'True': True, 'False': False}).fillna(False)

    # Crear un 'group_id' para bloques consecutivos del MISMO valor
    df['block_id'] = (df['valid_deteccion'] != df['valid_deteccion'].shift()).cumsum()
    
    # Filtrar solo los bloques donde la detección fue FALSA
    df_invalid = df[~df['valid_deteccion']].copy()
    
    if df_invalid.empty:
        return pd.DataFrame(columns=['event_id', 'event_type', 'event_class', 'start_time_ms', 'end_time_ms', 'duration_ms'])

    parpadeos_detectados = []
    
    # Agrupar por esos bloques de 'False'
    for block_id, group in df_invalid.groupby('block_id'):
        count = len(group)
        
        # Aplicar la regla: "más de dos frames"
        if count > umbrales['T_MIN_PARPADEO_FRAMES']:
            start_time = group['timestamp_ms'].iloc[0]
            end_time = group['timestamp_ms'].iloc[-1]
            duration = end_time - start_time
            
            parpadeos_detectados.append({
                'event_id': f"b_{block_id}",
                'event_type': 'parpadeo',
                'event_class': 'parpadeo',
                'start_time_ms': start_time,
                'end_time_ms': end_time,
                'duration_ms': duration,
                'amplitude_deg': np.nan,
                'velocidad_promedio_deg_s': np.nan
            })
            
    return pd.DataFrame(parpadeos_detectados)


def detectar_sacadicos_por_aceleracion(df, umbrales):
    """
    Detecta sacádicos usando el algoritmo I-AT.
    """
    print("Detectando sacádicos por perfil de aceleración (I-AT estricto)...")
    umbral_acel = umbrales['A_PICO_MINIMO']
    # Importante: no detectar picos donde la velocidad ya es 0 (parpadeos filtrados)
    acel_data_valid = df['acceleration_angular_filtered'].copy()
    acel_data_valid[df['velocity_angular_filtered'] == 0] = 0
    
    indices_picos_pos, _ = find_peaks(acel_data_valid, height=umbral_acel)
    indices_picos_neg_raw, _ = find_peaks(-acel_data_valid, height=umbral_acel)
    
    indices_df_picos_pos = df.index[indices_picos_pos]
    indices_df_picos_neg = df.index[indices_picos_neg_raw]
    
    sacadicos_validos = []
    indices_picos_neg_usados = set() 

    for idx_pico_pos in indices_df_picos_pos:
        candidatos_pico_neg = [
            idx for idx in indices_df_picos_neg 
            if idx > idx_pico_pos and idx not in indices_picos_neg_usados
        ]
        if not candidatos_pico_neg:
            continue
        idx_pico_neg = min(candidatos_pico_neg) 
        
        tiempo_pico_pos_ms = df.at[idx_pico_pos, 'timestamp_ms']
        tiempo_pico_neg_ms = df.at[idx_pico_neg, 'timestamp_ms']
        duracion_picos_ms = tiempo_pico_neg_ms - tiempo_pico_pos_ms
        
        if not (0 < duracion_picos_ms <= umbrales['T_MAX_ENTRE_PICOS_MS']):
            continue
            
        idx_inicio_real = idx_pico_pos
        idx_fin_real = idx_pico_neg
        evento_inicio = df.loc[idx_inicio_real]
        evento_fin = df.loc[idx_fin_real]
        
        segmento_vel_evento = df.loc[idx_inicio_real:idx_fin_real, 'velocity_angular_filtered']
        velocidad_promedio = 0.0
        if not segmento_vel_evento.empty:
            velocidad_promedio = segmento_vel_evento.mean()
        
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
            'duration_ms': duracion_picos_ms, 
            'amplitude_deg': amplitude_deg,
            'velocidad_promedio_deg_s': velocidad_promedio 
        })
        indices_picos_neg_usados.add(idx_pico_neg) 
            
    return pd.DataFrame(sacadicos_validos)

def detectar_fijaciones_restantes(df, sacadicos_df, parpadeos_df, umbrales):
    """
    MODIFICADO: Acepta 'parpadeos_df' y los excluye del análisis de fijaciones.
    """
    print("Detectando fijaciones en los segmentos restantes (excluyendo sacádicos y parpadeos)...")
    
    mascara_fijacion = pd.Series(True, index=df.index)
    
    # Excluir sacádicos
    for _, sac in sacadicos_df.iterrows():
        mascara_fijacion.loc[
            (df['timestamp_ms'] >= sac['start_time_ms']) & 
            (df['timestamp_ms'] <= sac['end_time_ms'])
        ] = False
        
    # Excluir parpadeos
    for _, blink in parpadeos_df.iterrows():
        mascara_fijacion.loc[
            (df['timestamp_ms'] >= blink['start_time_ms']) & 
            (df['timestamp_ms'] <= blink['end_time_ms'])
        ] = False
    
    # Aplicar máscara Y umbral de velocidad
    df_fijaciones = df[mascara_fijacion & (df['velocity_angular_filtered'] <= umbrales['V_MAXIMA_FIJACION'])].copy()
    
    columnas_reporte = ['event_id', 'event_type', 'event_class', 'start_time_ms', 'end_time_ms', 'duration_ms', 'amplitude_deg', 'velocidad_promedio_deg_s']
    if df_fijaciones.empty:
        return pd.DataFrame(columns=columnas_reporte)

    df_fijaciones['group_id'] = (df_fijaciones.index.to_series().diff() != 1).cumsum()
    
    fijaciones_validas = []
    for group_id, group_df in df_fijaciones.groupby('group_id'):
        if group_df.empty:
            continue
            
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

# --- Funciones de Graficación (MODIFICADAS) ---
def generar_visualizacion_combinada(df_raw, events_report, umbrales, output_file, stimulus_events=None):
    print(f"Generando gráfico combinado en {output_file}...")
    # MODIFICADO: Añadido 'parpadeo' al mapa de colores
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    event_color_map = {'fijacion_valida': 'green', 'sacadico': 'red', 'parpadeo': 'gray'}

    # Subplot 1: Posición
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_x'], label='dirección_x', alpha=0.8)
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_y'], label='dirección_y', linestyle='--', alpha=0.8)
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_z'], label='dirección_z (adelante)', linestyle=':', alpha=0.6)
    ax1.set_title(f"Posición del Vector de Mirada 3D\n(Archivo: {os.path.basename(output_file)})")
    ax1.set_ylabel('Componente del Vector')
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

    # Dibujar regiones de eventos (fijacion, sacadico, parpadeo)
    for _, event in events_report.iterrows():
        color = event_color_map.get(event['event_class'])
        if color:
            # Dibujar parpadeos en todos los ejes
            if event['event_class'] == 'parpadeo':
                ax1.axvspan(event['start_time_ms'] / 1000.0, event['end_time_ms'] / 1000.0, color=color, alpha=0.4)
            ax2.axvspan(event['start_time_ms'] / 1000.0, event['end_time_ms'] / 1000.0, color=color, alpha=0.3)
            ax3.axvspan(event['start_time_ms'] / 1000.0, event['end_time_ms'] / 1000.0, color=color, alpha=0.3)

    # Dibujar marcas de estímulos (Líneas Verticales)
    if stimulus_events:
        y_lim_ax1 = ax1.get_ylim()
        for i, event in enumerate(stimulus_events):
            color = 'red' if i % 2 == 0 else 'magenta'
            start_s = event['start_time_ms'] / 1000.0
            
            ax1.axvline(x=start_s, color=color, linestyle='--', alpha=0.7, zorder=1)
            ax2.axvline(x=start_s, color=color, linestyle='--', alpha=0.7, zorder=1)
            ax3.axvline(x=start_s, color=color, linestyle='--', alpha=0.7, zorder=1)
            
            ax1.text(start_s + 0.05, y_lim_ax1[1] * 0.9, event['label'], 
                     rotation=90, verticalalignment='top', color=color, fontsize=9, weight='bold')
        ax1.set_ylim(y_lim_ax1) # Restaurar límites

    ax1.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Gráfico combinado guardado en: {output_file}")
    plt.close(fig)

def generar_visualizacion_fijaciones(df_raw, events_report, umbrales, output_file, stimulus_events=None):
    """
    MODIFICADO: Usa líneas verticales para estímulos, no regiones.
    """
    print(f"Generando gráfico de fijaciones en {output_file}...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    # MODIFICADO: Añadido 'parpadeo'
    event_color_map = { 'fijacion_valida': 'green', 'parpadeo': 'gray' }

    # Subplot 1: Posición
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_x'], label='dirección_x', alpha=0.8, zorder=1)
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_y'], label='dirección_y', linestyle='--', alpha=0.8, zorder=1)
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_z'], label='dirección_z (adelante)', linestyle=':', alpha=0.6, zorder=1)
    ax1.set_title(f"Posición de Mirada y Fijaciones Detectadas\n(Archivo: {os.path.basename(output_file)})")
    ax1.set_ylabel('Componente del Vector')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Subplot 2: Velocidad
    ax2.plot(df_raw['timestamp_s'], df_raw['velocity_angular_filtered'], label='Velocidad Angular Filtrada (°/s)', color='blue', alpha=0.7, zorder=1)
    ax2.axhline(y=umbrales['V_MAXIMA_FIJACION'], color='green', linestyle=':', label=f"Umbral Fijación ({umbrales['V_MAXIMA_FIJACION']} °/s)")
    ax2.set_title('Velocidad y Fijaciones Detectadas')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Velocidad Angular (°/s)')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    # Dibujar regiones de eventos (SOLO fijaciones y parpadeos)
    fix_blink_report = events_report[events_report['event_class'].isin(['fijacion_valida', 'parpadeo'])]
    for _, event in fix_blink_report.iterrows():
        color = event_color_map.get(event['event_class'])
        if color:
            ax1.axvspan(event['start_time_ms'] / 1000.0, event['end_time_ms'] / 1000.0, color=color, alpha=0.4, zorder=2)
            ax2.axvspan(event['start_time_ms'] / 1000.0, event['end_time_ms'] / 1000.0, color=color, alpha=0.4, zorder=2)

    # --- MODIFICADO: Dibujar marcas de estímulos (Líneas Verticales) ---
    if stimulus_events:
        y_lim_ax1 = ax1.get_ylim()
        for i, event in enumerate(stimulus_events):
            color = 'red' if i % 2 == 0 else 'magenta'
            start_s = event['start_time_ms'] / 1000.0
            
            ax1.axvline(x=start_s, color=color, linestyle='--', alpha=0.7, zorder=1)
            ax2.axvline(x=start_s, color=color, linestyle='--', alpha=0.7, zorder=1)
            
            ax1.text(start_s + 0.05, y_lim_ax1[1] * 0.9, event['label'], 
                     rotation=90, verticalalignment='top', color=color, fontsize=9, weight='bold')
        ax1.set_ylim(y_lim_ax1)

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Gráfico de fijaciones guardado en: {output_file}")
    plt.close(fig)

def generar_visualizacion_sacadicos(df_raw, events_report, umbrales, output_file, stimulus_events=None):
    """
    MODIFICADO: Usa líneas verticales para estímulos, no regiones.
    """
    print(f"Generando gráfico de sacádicos en {output_file}...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    # MODIFICADO: Añadido 'parpadeo'
    event_color_map = { 'sacadico': 'red', 'parpadeo': 'gray' }

    # Subplot 1: Posición
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_x'], label='dirección_x', alpha=0.8, zorder=1)
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_y'], label='dirección_y', linestyle='--', alpha=0.8, zorder=1)
    ax1.plot(df_raw['timestamp_s'], df_raw['gaze_z'], label='dirección_z (adelante)', linestyle=':', alpha=0.6, zorder=1)
    ax1.set_title(f"Posición de Mirada y Sacádicos\n(Archivo: {os.path.basename(output_file)})")
    ax1.set_ylabel('Componente del Vector')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Subplot 2: Velocidad
    ax2.plot(df_raw['timestamp_s'], df_raw['velocity_angular_filtered'], label='Velocidad Angular Filtrada (°/s)', color='blue', alpha=0.7, zorder=1)
    ax2.set_title('Velocidad y Sacádicos')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Velocidad Angular (°/s)')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    # Dibujar regiones de eventos (SOLO sacádicos y parpadeos)
    sac_blink_report = events_report[events_report['event_class'].isin(['sacadico', 'parpadeo'])]
    for _, event in sac_blink_report.iterrows():
        color = event_color_map.get(event['event_class'])
        if color:
            ax1.axvspan(event['start_time_ms'] / 1000.0, event['end_time_ms'] / 1000.0, color=color, alpha=0.3, zorder=2)
            ax2.axvspan(event['start_time_ms'] / 1000.0, event['end_time_ms'] / 1000.0, color=color, alpha=0.3, zorder=2)

    # --- MODIFICADO: Dibujar marcas de estímulos (Líneas Verticales) ---
    if stimulus_events:
        y_lim_ax1 = ax1.get_ylim()
        for i, event in enumerate(stimulus_events):
            color = 'red' if i % 2 == 0 else 'magenta'
            start_s = event['start_time_ms'] / 1000.0
            
            ax1.axvline(x=start_s, color=color, linestyle='--', alpha=0.7, zorder=1)
            ax2.axvline(x=start_s, color=color, linestyle='--', alpha=0.7, zorder=1)
            
            ax1.text(start_s + 0.05, y_lim_ax1[1] * 0.9, event['label'], 
                     rotation=90, verticalalignment='top', color=color, fontsize=9, weight='bold')
        ax1.set_ylim(y_lim_ax1)

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Gráfico de sacádicos guardado en: {output_file}")
    plt.close(fig)

# --- Funciones de Histograma (Sin Cambios) ---
def generar_histograma_velocidad(df_raw, umbrales, output_file):
    print(f"Generando histograma de velocidad en {output_file}...")
    vel_data = df_raw['velocity_angular_filtered']
    if vel_data.empty:
        print("No hay datos de velocidad para el histograma.")
        return
    max_val = max(500, vel_data.quantile(0.995))
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(vel_data, bins=100, alpha=0.75, color='blue', range=(0, max_val))
    ax.set_title(f"Histograma de Velocidad Angular (Escala Log)\n(Archivo: {os.path.basename(output_file)})")
    ax.set_xlabel("Velocidad Angular (°/s)")
    ax.set_ylabel("Frecuencia (Conteo de Muestras) [Escala Log]")
    ax.axvline(x=umbrales['V_MAXIMA_FIJACION'], color='green', linestyle=':', 
               label=f"Umbral Fijación ({umbrales['V_MAXIMA_FIJACION']} °/s)")
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log') 
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Histograma de velocidad guardado en: {output_file}")
    plt.close(fig)

def generar_histograma_aceleracion(df_raw, umbrales, output_file):
    print(f"Generando histograma de aceleración en {output_file}...")
    acel_data = df_raw['acceleration_angular_filtered']
    if acel_data.empty:
        print("No hay datos de aceleración para el histograma.")
        return
    max_abs_val = max(1000, acel_data.abs().quantile(0.995))
    hist_range = (-max_abs_val, max_abs_val)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(acel_data, bins=150, alpha=0.75, color='purple', range=hist_range)
    ax.set_title(f"Histograma de Aceleración Angular (Escala Log)\n(Archivo: {os.path.basename(output_file)})")
    ax.set_xlabel("Aceleración Angular (°/s²)")
    ax.set_ylabel("Frecuencia (Conteo de Muestras) [Escala Log]")
    umbral_pos = umbrales['A_PICO_MINIMO']
    umbral_neg = -umbrales['A_PICO_MINIMO']
    ax.axvline(x=umbral_pos, color='orange', linestyle=':', 
               label=f"Umbral Acel. Pos. ({umbral_pos} °/s²)")
    ax.axvline(x=umbral_neg, color='orange', linestyle=':', 
               label=f"Umbral Acel. Neg. ({umbral_neg} °/s²)")
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log') 
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Histograma de aceleración guardado en: {output_file}")
    plt.close(fig)

# --- FUNCIÓN PRINCIPAL (MODIFICADA) ---
def main():
    print(f"Iniciando análisis por lotes en el directorio: {INPUT_DIR}")
    
    try:
        eventos_estimulo_exp1 = get_stimulus_events_exp1(OFFSET_MS)
    except Exception as e:
        print(f"ERROR: No se pudo generar la lista de eventos de estímulo. Error: {e}")
        sys.exit(1)
        
    patron_busqueda = os.path.join(INPUT_DIR, f"{NOMBRE}*_data.csv")
    lista_archivos_csv = glob.glob(patron_busqueda)
    
    if not lista_archivos_csv:
        print(f"ERROR: No se encontraron archivos .csv con el patrón '{patron_busqueda}'.")
        if os.path.exists("Victor3_intento_1_data.csv"):
            lista_archivos_csv = ["Victor3_intento_1_data.csv"]
            print("ADVERTENCIA: Usando 'Victor3_intento_1_data.csv' como fallback.")
        else:
            print("Saliendo. No hay archivos para procesar.")
            sys.exit(1)
        
    print(f"Se encontraron {len(lista_archivos_csv)} archivos CSV para analizar.")
    
    reportes_agregados = []

    for input_file_path in lista_archivos_csv:
        base_filename = os.path.splitext(os.path.basename(input_file_path))[0]
        print(f"\n--- Procesando Archivo: {base_filename}.csv ---")
        
        # Nombres de reportes actualizados
        output_report_fix = os.path.join(OUTPUT_DIR, f"{base_filename}_reporte_fijaciones.csv")
        output_report_sac = os.path.join(OUTPUT_DIR, f"{base_filename}_reporte_sacadicos.csv")
        output_report_blink = os.path.join(OUTPUT_DIR, f"{base_filename}_reporte_parpadeos.csv")
        
        output_plot_comb = os.path.join(OUTPUT_DIR, f"{base_filename}_grafico_combinado.png")
        output_plot_fix = os.path.join(OUTPUT_DIR, f"{base_filename}_grafico_fijaciones.png")
        output_plot_sac = os.path.join(OUTPUT_DIR, f"{base_filename}_grafico_sacadicos.png")
        output_plot_hist_vel = os.path.join(OUTPUT_DIR, f"{base_filename}_histograma_velocidad.png")
        output_plot_hist_acel = os.path.join(OUTPUT_DIR, f"{base_filename}_histograma_aceleracion.png")
        
        try:
            df_original = pd.read_csv(input_file_path)
            
            print(f"Datos originales: {len(df_original)} filas. Recortando {OFFSET_MS}ms del inicio...")
            df_original = df_original[df_original['timestamp_ms'] >= OFFSET_MS].copy()
            
            df_original.reset_index(drop=True, inplace=True) 
            
            if df_original.empty:
                print(f"Advertencia: El archivo {base_filename}.csv está vacío después de recortar {OFFSET_MS}ms. Omitiendo.")
                continue
            print(f"Datos después del recorte: {len(df_original)} filas.")
                
        except Exception as e:
            print(f"ERROR: No se pudo leer o recortar el archivo '{input_file_path}'. Error: {e}")
            continue 
            
        # 1. Calcular Velocidad y Aceleración
        df_completo = calcular_velocidad_y_aceleracion(df_original.copy())
        
        # 2. Detección de Parpadeos (¡NUEVO!)
        parpadeos_df = detectar_parpadeos(df_completo, UMBRALES_3D)
        
        # 3. Detección "Sacadico-Primero"
        sacadicos_df = detectar_sacadicos_por_aceleracion(df_completo, UMBRALES_3D)
        
        # 4. Detección de Fijaciones (MODIFICADO: pasa parpadeos_df)
        fijaciones_df = detectar_fijaciones_restantes(df_completo, sacadicos_df, parpadeos_df, UMBRALES_3D)
        
        # 5. Combinar Reporte (MODIFICADO: incluye parpadeos)
        reports_list = []
        if not sacadicos_df.empty:
            reports_list.append(sacadicos_df)
        if not fijaciones_df.empty:
            reports_list.append(fijaciones_df)
        if not parpadeos_df.empty:
            reports_list.append(parpadeos_df)

        if reports_list:
            reporte_final_combinado = pd.concat(reports_list).sort_values(by='start_time_ms')
        else:
            print("No se detectaron eventos (fijaciones, sacádicos o parpadeos).")
            reporte_final_combinado = pd.DataFrame(columns=['start_time_ms', 'event_class'])

        # 6. Guardar Reportes CSV Separados
        try:
            fijaciones_df.to_csv(output_report_fix, index=False)
            sacadicos_df.to_csv(output_report_sac, index=False)
            parpadeos_df.to_csv(output_report_blink, index=False) # ¡NUEVO!
            print(f"Reportes individuales (fij, sac, blink) guardados para: {base_filename}")
        except Exception as e:
            print(f"ERROR: No se pudo guardar el reporte para '{base_filename}'. Error: {e}")
            continue

        # 7. Generar Gráficos
        try:
            generar_visualizacion_combinada(df_completo, reporte_final_combinado, UMBRALES_3D, output_plot_comb, stimulus_events=eventos_estimulo_exp1)
            generar_visualizacion_fijaciones(df_completo, reporte_final_combinado, UMBRALES_3D, output_plot_fix, stimulus_events=eventos_estimulo_exp1)
            generar_visualizacion_sacadicos(df_completo, reporte_final_combinado, UMBRALES_3D, output_plot_sac, stimulus_events=eventos_estimulo_exp1)
            generar_histograma_velocidad(df_completo, UMBRALES_3D, output_plot_hist_vel)
            generar_histograma_aceleracion(df_completo, UMBRALES_3D, output_plot_hist_acel)
        except Exception as e:
            print(f"ERROR: No se pudo guardar gráficos para '{base_filename}'. Error: {e}")
            import traceback
            print(traceback.format_exc())
            continue

        # 8. Recopilar datos para el reporte agregado (MODIFICADO: incluye parpadeos)
        try:
            duracion_total_seg = (df_completo['timestamp_s'].iloc[-1] - df_completo['timestamp_s'].iloc[0])
            conteo_fijaciones = len(fijaciones_df)
            conteo_sacadicos = len(sacadicos_df)
            conteo_parpadeos = len(parpadeos_df)
            
            reportes_agregados.append({
                'archivo': base_filename,
                'offset_aplicado_ms': OFFSET_MS,
                'duracion_analizada_seg': duracion_total_seg,
                'conteo_fijaciones': conteo_fijaciones,
                'conteo_sacadicos': conteo_sacadicos,
                'conteo_parpadeos': conteo_parpadeos,
                'frecuencia_fijaciones_hz': conteo_fijaciones / duracion_total_seg if duracion_total_seg > 0 else 0,
                'frecuencia_sacadicos_hz': conteo_sacadicos / duracion_total_seg if duracion_total_seg > 0 else 0,
                'duracion_media_fij_ms': fijaciones_df['duration_ms'].mean() if not fijaciones_df.empty else 0,
                'duracion_media_sac_ms': sacadicos_df['duration_ms'].mean() if not sacadicos_df.empty else 0,
                'duracion_media_parpadeo_ms': parpadeos_df['duration_ms'].mean() if not parpadeos_df.empty else 0,
                'amplitud_media_sac_deg': sacadicos_df['amplitude_deg'].mean() if not sacadicos_df.empty else 0,
                'vel_media_fij_deg_s': fijaciones_df['velocidad_promedio_deg_s'].mean() if not fijaciones_df.empty else 0,
                'vel_media_sac_deg_s': sacadicos_df['velocidad_promedio_deg_s'].mean() if not sacadicos_df.empty else 0,
            })
        except Exception as e:
            print(f"ERROR: No se pudo calcular el resumen para '{base_filename}'. Error: {e}")

    # --- FIN DEL BUCLE ---
    if not reportes_agregados:
        print("\nNo se pudo procesar ningún archivo exitosamente. No se generará reporte agregado.")
        sys.exit(1)
        
    print("\n--- Análisis por lotes completado. Generando reporte agregado... ---")
    
    try:
        df_agregado = pd.DataFrame(reportes_agregados)
        promedios_generales = df_agregado.mean(numeric_only=True)
        promedios_generales['total_archivos_procesados'] = len(df_agregado)
        
        promedios_df = pd.DataFrame(promedios_generales).T
        promedios_df.index = ['PROMEDIO_GENERAL']

        df_agregado_final = pd.concat([df_agregado.set_index('archivo'), promedios_df])
        df_agregado_final.to_csv(OUTPUT_REPORT_AGGREGATE, index=True)
        
        print("\n--- Resumen General (Promedio de todos los archivos) ---")
        print(promedios_generales.to_string())
        print(f"\nReporte agregado general guardado en: {OUTPUT_REPORT_AGGREGATE}")
        print(f"Análisis 3D (I-AT) completado. Se generaron reportes y gráficos en: {OUTPUT_DIR}")

    except Exception as e:
        print(f"ERROR: No se pudo guardar el reporte agregado general. Error: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"ERROR al ejecutar main(): {e}")
        print(traceback.format_exc())
        sys.exit(1)
import pandas as pd
import os
import glob
import numpy as np

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
BASE_PATH = r"C:\Users\Victor\Documents\Tesis3D"
STIMULUS_PATH = os.path.join(BASE_PATH, "Videos", "Experimento_1", "Venegas")
DATA_PATH = os.path.join(BASE_PATH, "Data", "Experimento_1", "Venegas_data")

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def calculate_offset_for_pair(stimulus_csv, data_csv):
    """
    Calcula el offset entre un par de archivos stimulus y data.
    Retorna el offset en ms, o None si hay error.
    """
    try:
        # Cargar ambos CSVs
        df_stim = pd.read_csv(stimulus_csv)
        df_eye = pd.read_csv(data_csv)
        
        # Verificar columnas necesarias
        if 'relative_time_s' not in df_stim.columns:
            print(f"  ⚠️  Falta 'relative_time_s' en {os.path.basename(stimulus_csv)}")
            return None
        
        if 'timestamp_ms' not in df_eye.columns:
            print(f"  ⚠️  Falta 'timestamp_ms' en {os.path.basename(data_csv)}")
            return None
        
        # Duración de cada señal
        stim_duration_s = df_stim['relative_time_s'].max()
        eye_duration_s = df_eye['timestamp_ms'].max() / 1000.0
        
        # Calcular offset
        offset_ms = (eye_duration_s - stim_duration_s) * 1000.0
        
        return offset_ms
        
    except Exception as e:
        print(f"  ❌ Error procesando archivos: {e}")
        return None


def find_matching_pairs(stimulus_path, data_path):
    """
    Encuentra pares de archivos stimulus y data que coincidan.
    Retorna lista de tuplas (stimulus_file, data_file, base_name)
    """
    pairs = []
    
    # Buscar todos los archivos stimulus
    stimulus_files = glob.glob(os.path.join(stimulus_path, "*_stimulus.csv"))
    
    if not stimulus_files:
        print(f"❌ No se encontraron archivos *_stimulus.csv en: {stimulus_path}")
        return pairs
    
    print(f"📂 Encontrados {len(stimulus_files)} archivos de estímulo\n")
    
    for stim_file in stimulus_files:
        # Extraer nombre base
        # Ejemplo: "Victor_9_puntos_intento_1_stimulus.csv" -> "Victor_9_puntos_intento_1"
        stim_basename = os.path.basename(stim_file).replace("_stimulus.csv", "")
        
        # Buscar el archivo de data correspondiente
        data_file = os.path.join(data_path, f"{stim_basename}_data.csv")
        
        if os.path.exists(data_file):
            pairs.append((stim_file, data_file, stim_basename))
        else:
            print(f"⚠️  No se encontró pareja para: {stim_basename}")
    
    return pairs


def calculate_average_offset():
    """
    Función principal que calcula el offset promedio de todos los pares.
    """
    print("="*70)
    print("   CÁLCULO DE OFFSET PROMEDIO DE SINCRONIZACIÓN")
    print("="*70)
    print(f"\n📁 Stimulus Path: {STIMULUS_PATH}")
    print(f"📁 Data Path: {DATA_PATH}\n")
    
    # Encontrar pares de archivos
    pairs = find_matching_pairs(STIMULUS_PATH, DATA_PATH)
    
    if not pairs:
        print("\n❌ No se encontraron pares de archivos para procesar.")
        return
    
    print(f"\n✅ Se encontraron {len(pairs)} pares válidos\n")
    print("-"*70)
    
    # Calcular offset para cada par
    offsets = []
    results = []
    
    for stim_file, data_file, base_name in pairs:
        print(f"\n📊 Procesando: {base_name}")
        
        offset = calculate_offset_for_pair(stim_file, data_file)
        
        if offset is not None:
            offsets.append(offset)
            results.append({
                'video': base_name,
                'offset_ms': offset
            })
            print(f"   ✅ Offset: {offset:.2f} ms")
        else:
            print(f"   ❌ No se pudo calcular offset")
    
    print("\n" + "="*70)
    
    # Calcular estadísticas
    if offsets:
        offsets_array = np.array(offsets)
        
        mean_offset = np.mean(offsets_array)
        std_offset = np.std(offsets_array)
        min_offset = np.min(offsets_array)
        max_offset = np.max(offsets_array)
        median_offset = np.median(offsets_array)
        
        print("\n📈 ESTADÍSTICAS DE OFFSET:")
        print("-"*70)
        print(f"   Promedio (Mean):     {mean_offset:.2f} ms")
        print(f"   Mediana:             {median_offset:.2f} ms")
        print(f"   Desviación Estándar: {std_offset:.2f} ms")
        print(f"   Mínimo:              {min_offset:.2f} ms")
        print(f"   Máximo:              {max_offset:.2f} ms")
        print(f"   Rango:               {max_offset - min_offset:.2f} ms")
        print("-"*70)
        
        # Detección de outliers (valores fuera de 2 desviaciones estándar)
        outliers = []
        for result in results:
            offset = result['offset_ms']
            if abs(offset - mean_offset) > 2 * std_offset:
                outliers.append(result)
        
        if outliers:
            print("\n⚠️  OUTLIERS DETECTADOS (>2σ):")
            for outlier in outliers:
                print(f"   • {outlier['video']}: {outlier['offset_ms']:.2f} ms")
        
        # Recomendación
        print("\n" + "="*70)
        print("🎯 RECOMENDACIÓN:")
        print("="*70)
        
        # Si la variabilidad es baja (std < 50ms), usar la media
        if std_offset < 50:
            recommended_offset = mean_offset
            print(f"   Offset recomendado: {recommended_offset:.0f} ms")
            print(f"   Razón: Baja variabilidad (σ={std_offset:.2f}ms)")
        else:
            recommended_offset = median_offset
            print(f"   Offset recomendado: {recommended_offset:.0f} ms")
            print(f"   Razón: Alta variabilidad (σ={std_offset:.2f}ms), usar mediana")
        
        print("\n   Actualiza en tu código:")
        print(f"   'INITIAL_OFFSET_MS': {int(round(recommended_offset))}")
        
        # Guardar resultados detallados
        output_file = os.path.join(BASE_PATH, "Analizar_Data", "offset_analysis.csv")
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False)
        print(f"\n💾 Resultados guardados en: {output_file}")
        
        # Tabla resumida en consola
        print("\n📋 TABLA DE OFFSETS POR VIDEO:")
        print("-"*70)
        for i, result in enumerate(results, 1):
            deviation = result['offset_ms'] - mean_offset
            marker = "⚠️" if abs(deviation) > std_offset else "✅"
            print(f"{marker} {i:2d}. {result['video']:<40} {result['offset_ms']:7.2f} ms  (Δ={deviation:+6.2f})")
        
        print("="*70 + "\n")
        
    else:
        print("\n❌ No se pudo calcular ningún offset válido.")


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    calculate_average_offset()
    input("\nPresiona ENTER para salir...")
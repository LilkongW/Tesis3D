import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

# ========== CONFIGURACIÓN ==========
# (Se mantendrán los valores del script original, el usuario puede cambiarlos si es necesario)
NOMBRE = "Raul"
EXP_NUM = 1  # 1 o 2
INPUT_DIR = fr"Analizar_Data/Resultados/Experimento_{EXP_NUM}/{NOMBRE}_data"
OUTPUT_DIR = fr"Analizar_Data/Resultados/Experimento_{EXP_NUM}/{NOMBRE}_data/analisis_reaccion"


# Parámetros de análisis
VENTANA_BUSQUEDA_ADELANTE_MS = 1000.0  # Buscar fijación dentro de 1000ms DESPUÉS del estímulo
VENTANA_BUSQUEDA_ATRAS_MS = 1000.0      # Buscar fijación hasta 350ms ANTES del estímulo (para anticipaciones)
OFFSET_MS = 0.0  # El mismo offset usado en el análisis original

# Umbrales de clasificación
UMBRAL_ANTICIPACION_MS = -50.0     # Menos de -50ms = anticipación clara
UMBRAL_REACCION_RAPIDA_MS = 150.0  # 0-150ms = reacción rápida
UMBRAL_REACCION_NORMAL_MS = 300.0  # 150-300ms = reacción normal
# Mayor a 300ms = reacción lenta

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== FUNCIONES DE ESTÍMULOS ==========
# (Sin cambios)
def get_stimulus_events_exp1(offset_ms=0):
    """Define los 9 eventos de estímulo para el Experimento 1."""
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

def get_stimulus_events_exp2(offset_ms=0):
    """Define los 10 eventos de estímulo para el Experimento 2."""
    DURACION_PUNTO_MS = 2000.0
    eventos_estimulo = []
    posiciones_base = [
        "P1 (Sup-Izq)", "P2 (Sup-Der)", "P3 (Inf-Izq)", 
        "P4 (Inf-Der)", "P5 (Centro)"
    ]
    posiciones = posiciones_base + [f"{p} (R)" for p in posiciones_base]
    N_PUNTOS = len(posiciones)

    for i in range(N_PUNTOS):
        anim_start_time = i * DURACION_PUNTO_MS
        anim_end_time = (i + 1) * DURACION_PUNTO_MS
        eventos_estimulo.append({
            'label': posiciones[i],
            'start_time_ms': anim_start_time + offset_ms,
            'end_time_ms': anim_end_time + offset_ms
        })
    return eventos_estimulo

# ========== ANÁLISIS DE TIEMPO DE REACCIÓN (CORREGIDO) ==========
# (Sin cambios)
def analizar_tiempo_reaccion(fijaciones_df, eventos_estimulo, ventana_adelante_ms=1000.0, ventana_atras_ms=300.0):
    """
    Calcula el tiempo de reacción y cuenta fijaciones por cada estímulo.
    Ahora considera un margen HACIA ATRÁS para detectar anticipaciones.
    
    Tiempo de reacción NEGATIVO = anticipación (fijación antes del estímulo)
    Tiempo de reacción POSITIVO = reacción normal (fijación después del estímulo)
    
    Retorna:
        DataFrame con métricas por estímulo
    """
    resultados = []
    
    # === INICIO DE LA CORRECCIÓN ===
    # Copiamos el DF y calculamos 'end_time_ms' para CADA fijación.
    # Esto es crucial para detectar el solapamiento (overlap).
    fijaciones_df_con_fin = fijaciones_df.copy()
    fijaciones_df_con_fin['end_time_ms'] = fijaciones_df_con_fin['start_time_ms'] + fijaciones_df_con_fin['duration_ms']
    # === FIN DE LA CORRECCIÓN ===

    for idx, estimulo in enumerate(eventos_estimulo):
        stim_start = estimulo['start_time_ms']
        stim_end = estimulo['end_time_ms']
        stim_label = estimulo['label']
        
        # MODIFICADO: Buscar fijaciones desde ANTES del estímulo hasta DESPUÉS
        ventana_inicio = stim_start - ventana_atras_ms  # Margen hacia atrás
        ventana_fin = stim_start + ventana_adelante_ms  # Margen hacia adelante
        
        # Buscar la fijación más cercana al inicio del estímulo (puede ser antes o después)
        # Usamos el DF con 'end_time_ms' (fijaciones_df_con_fin)
        fijaciones_ventana = fijaciones_df_con_fin[
            (fijaciones_df_con_fin['start_time_ms'] >= ventana_inicio) & 
            (fijaciones_df_con_fin['start_time_ms'] <= ventana_fin)
        ].copy()
        
        # Calcular distancia al inicio del estímulo
        fijaciones_ventana['distancia_ms'] = fijaciones_ventana['start_time_ms'] - stim_start
        
        # === INICIO DE LA CORRECCIÓN ===
        # Contar fijaciones que se SOLAPAN con el período del estímulo
        # Lógica de solapamiento: (inicio_fij < fin_estim) Y (fin_fij > inicio_estim)
        fijaciones_durante_estimulo = fijaciones_df_con_fin[
            (fijaciones_df_con_fin['start_time_ms'] < stim_end) & 
            (fijaciones_df_con_fin['end_time_ms'] > stim_start)
        ]
        # === FIN DE LA CORRECCIÓN ===
        
        num_fijaciones = len(fijaciones_durante_estimulo)
        
        # Encontrar la fijación más cercana (puede tener distancia negativa o positiva)
        if len(fijaciones_ventana) > 0:
            # Ordenar por valor absoluto de la distancia para encontrar la más cercana
            fijaciones_ventana['distancia_abs'] = fijaciones_ventana['distancia_ms'].abs()
            fijaciones_ventana = fijaciones_ventana.sort_values('distancia_abs')
            
            fijacion_relevante = fijaciones_ventana.iloc[0]
            tiempo_reaccion_ms = fijacion_relevante['distancia_ms']  # Puede ser negativo
            duracion_primera_fij_ms = fijacion_relevante['duration_ms']
            
            # Clasificación mejorada
            if tiempo_reaccion_ms < UMBRAL_ANTICIPACION_MS:
                estado = "anticipación"
            elif tiempo_reaccion_ms < 0:
                estado = "anticipación_leve"  # Entre -50ms y 0ms
            elif tiempo_reaccion_ms <= UMBRAL_REACCION_RAPIDA_MS:
                estado = "reacción_rápida"  # 0-150ms
            elif tiempo_reaccion_ms <= UMBRAL_REACCION_NORMAL_MS:
                estado = "reacción_normal"   # 150-300ms
            else:
                estado = "reacción_lenta"    # >300ms
        else:
            # Esto realmente NO debería pasar si siempre hay fijaciones
            tiempo_reaccion_ms = np.nan
            duracion_primera_fij_ms = np.nan
            estado = "sin_respuesta"
        
        # Calcular duración promedio de fijaciones en este estímulo
        if num_fijaciones > 0:
            duracion_promedio_fij = fijaciones_durante_estimulo['duration_ms'].mean()
        else:
            duracion_promedio_fij = np.nan
        
        resultados.append({
            'estimulo_num': idx + 1,
            'estimulo_label': stim_label,
            'stim_start_ms': stim_start,
            'stim_end_ms': stim_end,
            'tiempo_reaccion_ms': tiempo_reaccion_ms,
            'num_fijaciones': num_fijaciones,
            'duracion_primera_fijacion_ms': duracion_primera_fij_ms,
            'duracion_promedio_fijaciones_ms': duracion_promedio_fij,
            'estado': estado
        })
    
    return pd.DataFrame(resultados)


# ========== VISUALIZACIONES ==========

# === MODIFICADO (Request 2: Quitar subplot de fijaciones) ===
def graficar_tiempos_reaccion(resultados_df, output_file):
    """
    Gráfico de barras de tiempos de reacción por estímulo.
    MODIFICADO: Se eliminó el subplot de número de fijaciones.
    """
    # Se cambia de 2 subplots a 1 y se ajusta el tamaño
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 7))
    
    # Subplot 1: Tiempo de reacción (ahora puede ser negativo)
    valid_data = resultados_df.dropna(subset=['tiempo_reaccion_ms'])
    
    # Código de colores mejorado
    colors = []
    for estado in valid_data['estado']:
        if estado == 'anticipación':
            colors.append('darkred')
        elif estado == 'anticipación_leve':
            colors.append('orange')
        elif estado == 'reacción_rápida':
            colors.append('limegreen')
        elif estado == 'reacción_normal':
            colors.append('green')
        elif estado == 'reacción_lenta':
            colors.append('gold')
        else:
            colors.append('gray')
    
    ax1.bar(valid_data['estimulo_num'], valid_data['tiempo_reaccion_ms'], 
            color=colors, alpha=0.7, edgecolor='black')
    
    # Líneas de referencia
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1.5, label='Inicio del estímulo')
    ax1.axhline(y=UMBRAL_REACCION_RAPIDA_MS, color='limegreen', linestyle='--', 
                alpha=0.5, label=f'Reacción rápida ({UMBRAL_REACCION_RAPIDA_MS}ms)')
    ax1.axhline(y=UMBRAL_REACCION_NORMAL_MS, color='green', linestyle='--', 
                alpha=0.5, label=f'Reacción normal ({UMBRAL_REACCION_NORMAL_MS}ms)')
    ax1.axhline(y=UMBRAL_ANTICIPACION_MS, color='red', linestyle='--', 
                alpha=0.5, label=f'Anticipación ({UMBRAL_ANTICIPACION_MS}ms)')
    
    ax1.set_xlabel('Número de Estímulo', fontsize=12)
    ax1.set_ylabel('Tiempo de Reacción (ms)\n[Negativo = Anticipación]', fontsize=12)
    ax1.set_title('Tiempo de Reacción por Estímulo', fontsize=14, weight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Añadir etiquetas de estímulo
    ax1.set_xticks(valid_data['estimulo_num'])
    ax1.set_xticklabels(valid_data['estimulo_label'], rotation=45, ha='right', fontsize=9)
    
    # --- SE ELIMINÓ EL SUBPLOT 2 (ax2) ---
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✓ Gráfico de tiempos de reacción guardado: {output_file}")
    plt.close(fig)

# (Función sin cambios)
def graficar_timeline_detallado(fijaciones_df, eventos_estimulo, resultados_df, output_file):
    """Timeline detallado mostrando estímulos, fijaciones y tiempos de reacción."""
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Dibujar regiones de estímulos
    for idx, estimulo in enumerate(eventos_estimulo):
        color = 'lightcoral' if idx % 2 == 0 else 'lightblue'
        ax.axvspan(estimulo['start_time_ms'], estimulo['end_time_ms'], 
                   color=color, alpha=0.3, label='Estímulo' if idx == 0 else '')
        
        # Línea de inicio de estímulo
        ax.axvline(x=estimulo['start_time_ms'], color='red', 
                   linestyle='--', alpha=0.6, linewidth=1.5)
        
        # Etiqueta del estímulo
        ax.text(estimulo['start_time_ms'], ax.get_ylim()[1] * 0.95, 
                estimulo['label'], rotation=90, verticalalignment='top', 
                fontsize=8, weight='bold')
    
    # Dibujar fijaciones
    for _, fij in fijaciones_df.iterrows():
        ax.barh(0.5, width=fij['duration_ms'], left=fij['start_time_ms'], 
                height=0.3, color='green', alpha=0.7, edgecolor='darkgreen')
    
    # Marcar tiempos de reacción (incluyendo negativos)
    primera_linea = True
    for _, resultado in resultados_df.iterrows():
        if not pd.isna(resultado['tiempo_reaccion_ms']):
            tiempo_primera_fij = resultado['stim_start_ms'] + resultado['tiempo_reaccion_ms']
            
            # Color según tipo de reacción
            if resultado['tiempo_reaccion_ms'] < 0:
                color_linea = 'darkred'  # Anticipación
                marker_style = 'v'  # Triángulo hacia abajo
            else:
                color_linea = 'purple'  # Reacción normal
                marker_style = 'o'
            
            ax.plot([resultado['stim_start_ms'], tiempo_primera_fij], 
                    [0.8, 0.8], color=color_linea, linewidth=2, marker=marker_style, 
                    markersize=6, label='Tiempo de reacción' if primera_linea else '')
            primera_linea = False
            
            # Anotar valor del tiempo de reacción
            texto = f"{resultado['tiempo_reaccion_ms']:.0f}ms"
            if resultado['tiempo_reaccion_ms'] < 0:
                texto = f"({texto})"  # Paréntesis para negativos
            
            ax.text(tiempo_primera_fij, 0.85, texto, 
                    fontsize=7, ha='center', color=color_linea, weight='bold')
    
    ax.set_ylim(0, 1)
    ax.set_xlabel('Tiempo (ms)', fontsize=12)
    ax.set_title('Timeline de Estímulos, Fijaciones y Tiempos de Reacción\n' + 
                 '(Valores negativos = Anticipación)', 
                 fontsize=14, weight='bold')
    ax.set_yticks([])
    ax.legend(loc='upper right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✓ Timeline detallado guardado: {output_file}")
    plt.close(fig)

# === MODIFICADO (Request 1: Arreglar gráfico comprimido) ===
def graficar_distribucion_tiempos(resultados_df, output_file):
    """
    Histograma y estadísticas de tiempos de reacción (incluyendo negativos).
    MODIFICADO: Layout cambiado a 2x1 para evitar compresión y bins='auto'.
    """
    # Se cambia de 1x2 a 2x1 y se ajusta el tamaño
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    valid_data = resultados_df.dropna(subset=['tiempo_reaccion_ms'])
    
    if valid_data.empty:
        print("⚠ Advertencia: No hay datos válidos para graficar la distribución de tiempos.")
        plt.close(fig)
        return

    # Histograma (ahora con rango que incluye negativos)
    # Se cambian los bins a 'auto' para mejor ajuste
    ax1.hist(valid_data['tiempo_reaccion_ms'], bins='auto', color='skyblue', 
             edgecolor='black', alpha=0.7)
    ax1.axvline(x=valid_data['tiempo_reaccion_ms'].mean(), color='red', 
                linestyle='--', linewidth=2, label=f"Media: {valid_data['tiempo_reaccion_ms'].mean():.1f}ms")
    ax1.axvline(x=valid_data['tiempo_reaccion_ms'].median(), color='green', 
                linestyle='--', linewidth=2, label=f"Mediana: {valid_data['tiempo_reaccion_ms'].median():.1f}ms")
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1.5, 
                alpha=0.7, label='Inicio del estímulo')
    
    ax1.set_xlabel('Tiempo de Reacción (ms)\n[Negativo = Anticipación]', fontsize=12)
    ax1.set_ylabel('Frecuencia', fontsize=12)
    ax1.set_title('Distribución de Tiempos de Reacción', fontsize=13, weight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Box plot (ahora está debajo y en horizontal)
    ax2.boxplot(valid_data['tiempo_reaccion_ms'], vert=False, patch_artist=True, # Cambiado a horizontal (vert=False)
                boxprops=dict(facecolor='lightgreen', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Tiempo de Reacción (ms)\n[Negativo = Anticipación]', fontsize=12)
    ax2.set_title('Estadísticas de Tiempos de Reacción', fontsize=13, weight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_yticks([]) # Ocultar el tick de 'y'
    
    # Añadir estadísticas como texto
    num_anticipaciones = len(valid_data[valid_data['tiempo_reaccion_ms'] < 0])
    num_reacciones = len(valid_data[valid_data['tiempo_reaccion_ms'] >= 0])
    
    stats_text = f"Media: {valid_data['tiempo_reaccion_ms'].mean():.1f} ms\n"
    stats_text += f"Mediana: {valid_data['tiempo_reaccion_ms'].median():.1f} ms\n"
    stats_text += f"Desv. Est.: {valid_data['tiempo_reaccion_ms'].std():.1f} ms\n"
    stats_text += f"Mín: {valid_data['tiempo_reaccion_ms'].min():.1f} ms\n"
    stats_text += f"Máx: {valid_data['tiempo_reaccion_ms'].max():.1f} ms\n"
    stats_text += f"\n─────────────────\n"
    stats_text += f"Anticipaciones: {num_anticipaciones}\n"
    stats_text += f"Reacciones: {num_reacciones}"
    
    # Colocar el texto anclado al gráfico (funciona mejor con tight_layout)
    # Se ajusta la posición y anclaje para el nuevo layout
    ax2.text(0.95, 0.95, stats_text,
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✓ Gráfico de distribución guardado: {output_file}")
    plt.close(fig)

# ========== FUNCIÓN PRINCIPAL ==========
def main():
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE TIEMPO DE REACCIÓN - EXPERIMENTO {EXP_NUM}")
    print(f"{'='*60}\n")
    
    # Obtener eventos de estímulo
    if EXP_NUM == 1:
        eventos_estimulo = get_stimulus_events_exp1(OFFSET_MS)
    elif EXP_NUM == 2:
        eventos_estimulo = get_stimulus_events_exp2(OFFSET_MS)
    else:
        print(f"ERROR: EXP_NUM {EXP_NUM} no reconocido.")
        sys.exit(1)
    
    print(f"Estímulos configurados: {len(eventos_estimulo)} eventos")
    print(f"Buscando archivos en: {INPUT_DIR}\n")
    
    # Buscar archivos de reporte de fijaciones
    patron_fijaciones = os.path.join(INPUT_DIR, f"{NOMBRE}*_reporte_fijaciones.csv")
    archivos_fijaciones = glob.glob(patron_fijaciones)
    
    if not archivos_fijaciones:
        print(f"ERROR: No se encontraron archivos de fijaciones con patrón: {patron_fijaciones}")
        sys.exit(1)
    
    print(f"Archivos encontrados: {len(archivos_fijaciones)}\n")
    
    # Procesar cada archivo
    resultados_globales = []
    
    for archivo_fij in archivos_fijaciones:
        base_name = os.path.basename(archivo_fij).replace('_reporte_fijaciones.csv', '')
        print(f"\n{'─'*60}")
        print(f"Procesando: {base_name}")
        print(f"{'─'*60}")
        
        try:
            # Cargar fijaciones
            fijaciones_df = pd.read_csv(archivo_fij)
            
            if fijaciones_df.empty:
                print(f"⚠ Advertencia: No hay fijaciones en {base_name}")
                continue
            
            print(f"  • Fijaciones cargadas: {len(fijaciones_df)}")
            
            # Analizar tiempo de reacción
            resultados_df = analizar_tiempo_reaccion(
                fijaciones_df, eventos_estimulo, 
                VENTANA_BUSQUEDA_ADELANTE_MS, VENTANA_BUSQUEDA_ATRAS_MS
            )
            
            # Añadir información del archivo
            resultados_df['archivo'] = base_name
            resultados_globales.append(resultados_df)
            
            # Guardar CSV individual
            output_csv = os.path.join(OUTPUT_DIR, f"{base_name}_analisis_reaccion.csv")
            resultados_df.to_csv(output_csv, index=False)
            print(f"  ✓ Reporte CSV guardado: {output_csv}")
            
            # Generar gráficos
            output_barras = os.path.join(OUTPUT_DIR, f"{base_name}_tiempos_reaccion.png")
            graficar_tiempos_reaccion(resultados_df, output_barras) # Función modificada
            
            output_timeline = os.path.join(OUTPUT_DIR, f"{base_name}_timeline_detallado.png")
            graficar_timeline_detallado(fijaciones_df, eventos_estimulo, resultados_df, output_timeline)
            
            output_dist = os.path.join(OUTPUT_DIR, f"{base_name}_distribucion_tiempos.png")
            graficar_distribucion_tiempos(resultados_df, output_dist) # Función modificada
            
            # Mostrar estadísticas
            valid_rt = resultados_df.dropna(subset=['tiempo_reaccion_ms'])
            if not valid_rt.empty:
                print(f"\n  📊 ESTADÍSTICAS:")
                print(f"     Tiempo de reacción promedio: {valid_rt['tiempo_reaccion_ms'].mean():.1f} ms")
                print(f"     Mediana: {valid_rt['tiempo_reaccion_ms'].median():.1f} ms")
                print(f"     Desviación estándar: {valid_rt['tiempo_reaccion_ms'].std():.1f} ms")
                print(f"     Rango: {valid_rt['tiempo_reaccion_ms'].min():.1f} - {valid_rt['tiempo_reaccion_ms'].max():.1f} ms")
                print(f"     Respuestas válidas: {len(valid_rt)}/{len(eventos_estimulo)}")
                
                # Separar anticipaciones y reacciones
                anticipaciones = valid_rt[valid_rt['tiempo_reaccion_ms'] < 0]
                reacciones = valid_rt[valid_rt['tiempo_reaccion_ms'] >= 0]
                
                if len(anticipaciones) > 0:
                    print(f"\n  ⚡ ANTICIPACIONES:")
                    print(f"     Cantidad: {len(anticipaciones)}")
                    print(f"     Promedio: {anticipaciones['tiempo_reaccion_ms'].mean():.1f} ms")
                
                if len(reacciones) > 0:
                    print(f"\n  ✓ REACCIONES POSITIVAS:")
                    print(f"     Cantidad: {len(reacciones)}")
                    print(f"     Promedio: {reacciones['tiempo_reaccion_ms'].mean():.1f} ms")
                
                # Contar estados
                estados = resultados_df['estado'].value_counts()
                print(f"\n  📈 DISTRIBUCIÓN DE RESPUESTAS:")
                for estado, count in estados.items():
                    print(f"     {estado}: {count}")
            
        except Exception as e:
            print(f"  ✗ ERROR procesando {base_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generar reporte consolidado
    if resultados_globales:
        print(f"\n{'='*60}")
        print("GENERANDO REPORTE CONSOLIDADO")
        print(f"{'='*60}\n")
        
        df_consolidado = pd.concat(resultados_globales, ignore_index=True)
        output_consolidado = os.path.join(OUTPUT_DIR, "reporte_consolidado_tiempos_reaccion.csv")
        df_consolidado.to_csv(output_consolidado, index=False)
        print(f"✓ Reporte consolidado guardado: {output_consolidado}")
        
        # Estadísticas globales
        valid_global = df_consolidado.dropna(subset=['tiempo_reaccion_ms'])
        if not valid_global.empty:
            print(f"\n📊 ESTADÍSTICAS GLOBALES (todos los archivos):")
            print(f"   Tiempo de reacción promedio global: {valid_global['tiempo_reaccion_ms'].mean():.1f} ms")
            print(f"   Mediana global: {valid_global['tiempo_reaccion_ms'].median():.1f} ms")
            print(f"   Desviación estándar global: {valid_global['tiempo_reaccion_ms'].std():.1f} ms")
            print(f"   Total de respuestas analizadas: {len(valid_global)}")

            # === NUEVO (Request 3 y 4): Generar CSV de resumen de métricas ===
            print("\nGenerando reporte resumen de métricas...")
            try:
                resumen_metricas = {}
                reacciones = valid_global[valid_global['tiempo_reaccion_ms'] >= 0]
                anticipaciones = valid_global[valid_global['tiempo_reaccion_ms'] < 0]

                # Métricas solicitadas
                resumen_metricas['tiempo_reaccion_promedio_positivo_ms'] = reacciones['tiempo_reaccion_ms'].mean() if len(reacciones) > 0 else np.nan
                resumen_metricas['tiempo_anticipacion_promedio_ms'] = anticipaciones['tiempo_reaccion_ms'].mean() if len(anticipaciones) > 0 else np.nan

                # Otras métricas generales
                resumen_metricas['tiempo_reaccion_promedio_total_ms'] = valid_global['tiempo_reaccion_ms'].mean()
                resumen_metricas['tiempo_reaccion_mediana_total_ms'] = valid_global['tiempo_reaccion_ms'].median()
                resumen_metricas['tiempo_reaccion_std_total_ms'] = valid_global['tiempo_reaccion_ms'].std()
                resumen_metricas['tiempo_reaccion_min_ms'] = valid_global['tiempo_reaccion_ms'].min()
                resumen_metricas['tiempo_reaccion_max_ms'] = valid_global['tiempo_reaccion_ms'].max()
                
                resumen_metricas['conteo_total_respuestas'] = len(valid_global)
                resumen_metricas['conteo_reacciones_positivas'] = len(reacciones)
                resumen_metricas['conteo_anticipaciones'] = len(anticipaciones)
                resumen_metricas['tasa_anticipacion_pct'] = (len(anticipaciones) / len(valid_global)) * 100 if len(valid_global) > 0 else 0
                
                # Métricas de duración de fijación
                resumen_metricas['duracion_promedio_primera_fijacion_total_ms'] = valid_global['duracion_primera_fijacion_ms'].mean()
                resumen_metricas['duracion_promedio_primera_fijacion_reaccion_ms'] = reacciones['duracion_primera_fijacion_ms'].mean() if len(reacciones) > 0 else np.nan
                resumen_metricas['duracion_promedio_primera_fijacion_anticipacion_ms'] = anticipaciones['duracion_primera_fijacion_ms'].mean() if len(anticipaciones) > 0 else np.nan
                resumen_metricas['duracion_promedio_fijaciones_por_estimulo_ms'] = valid_global['duracion_promedio_fijaciones_ms'].mean()
                resumen_metricas['numero_promedio_fijaciones_por_estimulo'] = valid_global['num_fijaciones'].mean()
                
                # Conteo de estados
                conteo_estados = valid_global['estado'].value_counts().to_dict()
                for k, v in conteo_estados.items():
                    resumen_metricas[f'conteo_estado_{k}'] = v

                # Convertir a DataFrame y guardar
                df_resumen = pd.DataFrame(resumen_metricas.items(), columns=['Metrica', 'Valor'])
                output_resumen_csv = os.path.join(OUTPUT_DIR, "reporte_resumen_metricas.csv")
                
                df_resumen['Valor'] = df_resumen['Valor'].round(2) # Redondear valores
                
                df_resumen.to_csv(output_resumen_csv, index=False)
                print(f"✓ Reporte de resumen de métricas guardado: {output_resumen_csv}")

            except Exception as e:
                print(f"  ✗ ERROR generando el reporte de resumen de métricas: {e}")
            # === FIN DE LA MODIFICACIÓN ===
    
    print(f"\n{'='*60}")
    print("✓ ANÁLISIS COMPLETADO")
    print(f"Resultados guardados en: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
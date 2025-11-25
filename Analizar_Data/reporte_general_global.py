import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from matplotlib import rcParams 
import re 

class AnalizadorGlobalExperimento:
    def __init__(self, ruta_experimento):
        """
        Inicializa el analizador con la ruta del experimento
        """
        self.ruta_experimento = Path(ruta_experimento)
        self.datos_personas = {}
        self.reporte_global = None
        
        # Determinar el ideal de fijaciones basado en el nombre del experimento
        nombre_exp = self.ruta_experimento.name
        if 'experimento_1' in nombre_exp.lower():
            self.ideal_fijaciones = 9
        elif 'experimento_2' in nombre_exp.lower():
            self.ideal_fijaciones = 10
        else:
            self.ideal_fijaciones = None
        
    def cargar_datos(self):
        """
        Carga los reportes agregados de cada persona
        """
        print("Cargando datos de participantes...")
        
        # Buscar todas las carpetas de personas
        carpetas_personas = [d for d in self.ruta_experimento.iterdir() 
                            if d.is_dir() and d.name.endswith('_data')]
        
        for carpeta in carpetas_personas:
            nombre_persona = carpeta.name.replace('_data', '')
            archivo_reporte = carpeta / 'reporte_agregado_general.csv'
            
            if archivo_reporte.exists():
                try:
                    df = pd.read_csv(archivo_reporte)
                    # Filtrar solo la fila de PROMEDIO_GENERAL
                    fila_promedio = df[df.iloc[:, 0].str.contains('PROMEDIO_GENERAL', na=False)]
                    
                    if not fila_promedio.empty:
                        self.datos_personas[nombre_persona] = fila_promedio.iloc[0]
                        print(f"  ✓ {nombre_persona}: {len(df)-1} intentos")
                    else:
                        print(f"  ✗ {nombre_persona}: No se encontró PROMEDIO_GENERAL")
                except Exception as e:
                    print(f"  ✗ Error al cargar {nombre_persona}: {e}")
            else:
                print(f"  ✗ {nombre_persona}: Archivo no encontrado")
        
        print(f"\nTotal de participantes cargados: {len(self.datos_personas)}")
        
    def calcular_estadisticas_globales(self):
        """
        Calcula las estadísticas globales del dataset
        """
        if not self.datos_personas:
            print("No hay datos cargados")
            return
        
        # Crear DataFrame con todos los promedios
        df_global = pd.DataFrame(self.datos_personas).T
        
        # Métricas de interés
        metricas = [
            'conteo_fijaciones',
            'conteo_sacadicos',
            'conteo_parpadeos',
            'frecuencia_fijaciones_hz',
            'frecuencia_sacadicos_hz',
            'duracion_media_fij_ms',
            'duracion_media_sac_ms',
            'duracion_media_parpadeo_ms',
            'amplitud_media_sac_deg',
            'vel_media_fij_deg_s',
            'vel_media_sac_deg_s',
            'velocidad_promedio_general_deg_s',
            'aceleracion_promedio_general_deg_s2',
            'velocidad_promedio_entre_estimulos_deg_s' 
        ]
        
        # Calcular estadísticas
        estadisticas = {}
        for metrica in metricas:
            if metrica in df_global.columns:
                valores = pd.to_numeric(df_global[metrica], errors='coerce')
                valores = valores.dropna()
                if len(valores) > 0:
                    estadisticas[metrica] = {
                        'promedio': valores.mean(),
                        'mediana': valores.median(),
                        'desv_std': valores.std() if len(valores) > 1 else 0.0,
                        'min': valores.min(),
                        'max': valores.max(),
                        'n_participantes': valores.notna().sum()
                    }
        
        self.reporte_global = pd.DataFrame(estadisticas).T
        self.df_personas = df_global
        
        return self.reporte_global
    
    def guardar_reporte_global(self, archivo_salida='reporte_global_experimento1.csv'):
        """
        Guarda el reporte global en un archivo CSV
        """
        if self.reporte_global is not None:
            ruta_salida = self.ruta_experimento.parent / archivo_salida
            self.reporte_global.to_csv(ruta_salida)
            print(f"\n✓ Reporte global guardado en: {ruta_salida}")
            return ruta_salida
        
    def crear_histogramas_comparativos(self, carpeta_salida='Graficos_Globales'):
        """
        Crea histogramas comparativos para cada métrica con estilo mejorado.
        """
        if self.reporte_global is None:
            print("Primero debes calcular las estadísticas globales")
            return
        
        # Crear carpeta de salida
        ruta_salida = self.ruta_experimento.parent / carpeta_salida
        ruta_salida.mkdir(exist_ok=True)
        
        # Configuración de métricas con nombres legibles y notación LaTeX para unidades
        metricas_config = {
            'conteo_fijaciones': ('Número de Fijaciones (Total)', 'Cantidad', ''),
            'conteo_sacadicos': ('Número de Sacádicos (Total)', 'Cantidad', ''),
            'conteo_parpadeos': ('Número de Parpadeos (Total)', 'Cantidad', ''),
            'frecuencia_fijaciones_hz': ('Frecuencia de Fijaciones', 'Frecuencia', 'Hz'),
            'frecuencia_sacadicos_hz': ('Frecuencia de Sacádicos', 'Frecuencia', 'Hz'),
            'duracion_media_fij_ms': ('Duración Media de Fijaciones', 'Duración', 'ms'),
            'duracion_media_sac_ms': ('Duración Media de Sacádicos', 'Duración', 'ms'),
            'duracion_media_parpadeo_ms': ('Duración Media de Parpadeo', 'Duración', 'ms'),
            'amplitud_media_sac_deg': ('Amplitud Media de Sacádicos', 'Amplitud', r'$^\circ$'),
            'vel_media_fij_deg_s': ('Velocidad Media en Fijaciones', 'Velocidad', r'$^\circ/s$'),
            'vel_media_sac_deg_s': ('Velocidad Media en Sacádicos', 'Velocidad', r'$^\circ/s$'),
            'velocidad_promedio_general_deg_s': ('Velocidad Promedio General', 'Velocidad', r'$^\circ/s$'),
            'aceleracion_promedio_general_deg_s2': ('Aceleración Promedio General', 'Aceleración', r'$^\circ/s^2$'),
            'velocidad_promedio_entre_estimulos_deg_s': ('Velocidad Promedio entre Estímulos', 'Velocidad', r'$^\circ/s$')
        }
        
        print(f"\nGenerando histogramas con estilo profesional en: {ruta_salida}")
        
        # 🎨 ESTILO BASE: ggplot 🎨
        plt.style.use('ggplot') 

        for metrica, (titulo, ylabel, unidad) in metricas_config.items():
            if metrica not in self.df_personas.columns:
                continue
            
            valores = pd.to_numeric(self.df_personas[metrica], errors='coerce').dropna()
            
            if len(valores) == 0:
                continue
            
            promedio_global = self.reporte_global.loc[metrica, 'promedio']
            
            # Crear figura
            fig, ax = plt.subplots(figsize=(10, 6)) 
            personas = valores.index
            
            # 🎨 PALETA DE COLORES PROFESIONAL Y CONTRASTADA 🎨
            COLOR_BASE = '#1f77b4'      # Azul moderno
            COLOR_ALTO_IDEAL = '#2ca02c'  # Verde (Línea ideal)
            COLOR_ALTO_MEDIA = '#ff7f0e'  # Naranja (Línea de la media global)
            COLOR_DESVIACION = '#d62728'  # Rojo (Barras desviadas)


            # --- DIBUJAR BARRAS ---
            if metrica == 'conteo_fijaciones' and self.ideal_fijaciones is not None:
                IDEAL = self.ideal_fijaciones
                # Se marca en rojo si se desvía más de 2 unidades del ideal.
                colores = [COLOR_DESVIACION if abs(v - IDEAL) > 2 else COLOR_BASE
                           for v in valores.values]
            else:
                colores = [COLOR_BASE] * len(valores)
            
            bars = ax.bar(range(len(personas)), valores.values, color=colores, alpha=0.9)
            
            # --- LÍNEAS DE REFERENCIA ---
            
            # Línea de Media Global (Naranja)
            ax.axhline(y=promedio_global, color=COLOR_ALTO_MEDIA, linestyle='--', 
                        linewidth=2.5, label=f'Media Global: {promedio_global:.2f} {unidad}')
            
            # Línea de Ideal (Verde) - Solo para conteo_fijaciones
            if metrica == 'conteo_fijaciones' and self.ideal_fijaciones is not None:
                ax.axhline(y=self.ideal_fijaciones, color=COLOR_ALTO_IDEAL, linestyle='-', 
                            linewidth=2.5, label=f'Ideal ({self.ideal_fijaciones} estímulos)')
            
            
            # --- CONFIGURACIÓN DE EJES Y TÍTULOS ---
            ax.set_title(f'Distribución de {titulo}', fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel('Participantes', fontsize=12) 
            ax.set_ylabel(f'{ylabel} ({unidad})' if unidad else ylabel, fontsize=12)
            
            ax.set_xticks(range(len(personas)))
            ax.set_xticklabels(personas, rotation=45, ha='right', fontsize=10)
            
            # Agregar valores sobre las barras
            for i, (bar, val) in enumerate(zip(bars, valores.values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (max(valores)*0.01), 
                        f'{val:.2f}',
                        ha='center', va='bottom', fontsize=9, 
                        color='black', fontweight='bold') 
            
            # 📌 LEYENDA: Colocada fuera del gráfico a la derecha superior 📌
            ax.legend(loc='upper left',          # Punto de anclaje de la leyenda
                      bbox_to_anchor=(1.01, 1), # Coordenadas fuera del plot (1.01 = justo a la derecha, 1 = arriba)
                      fontsize=10, 
                      frameon=True, 
                      facecolor='white', 
                      edgecolor='black')
            
            plt.tight_layout()
            
            # Guardar con alta resolución (dpi=300)
            nombre_archivo = f'histograma_estilo_final_{metrica}.png'
            plt.savefig(ruta_salida / nombre_archivo, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ {nombre_archivo}")
        
        print(f"\n✓ Histogramas generados exitosamente")
    
    def generar_reporte_completo(self):
        """
        Genera el reporte completo con todas las visualizaciones
        """
        print("="*60)
        print("ANÁLISIS GLOBAL - EXPERIMENTO 1")
        print("="*60)
        
        # Cargar datos
        self.cargar_datos()
        
        if not self.datos_personas:
            print("No se pudieron cargar los datos")
            return
        
        # Calcular estadísticas
        print("\n" + "="*60)
        print("CALCULANDO ESTADÍSTICAS GLOBALES")
        print("="*60)
        reporte = self.calcular_estadisticas_globales()
        print("\nEstadísticas Globales:")
        print(reporte.to_string())
        
        # Guardar reporte
        self.guardar_reporte_global()
        
        # Crear histogramas
        print("\n" + "="*60)
        print("GENERANDO HISTOGRAMAS COMPARATIVOS")
        print("="*60)
        self.crear_histogramas_comparativos()
        
        print("\n" + "="*60)
        print("ANÁLISIS COMPLETADO")
        print("="*60)


# Uso del programa
if __name__ == "__main__":
    # Configurar la ruta al Experimento 
    ruta_experimento = r"C:\Users\Victor\Documents\Tesis3D\Analizar_Data\Resultados\Experimento_1"
    
    try:
        # Crear analizador
        analizador = AnalizadorGlobalExperimento(ruta_experimento)
        
        # Generar reporte completo
        analizador.generar_reporte_completo()
    except Exception as e:
        print(f"\n[ERROR CRÍTICO]: Fallo en el script principal: {e}")
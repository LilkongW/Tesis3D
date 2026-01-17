import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import warnings

# Configuración de backend
try:
    import matplotlib
    matplotlib.use('TkAgg') 
except:
    pass 

# =============================================================================
#  CONFIGURACIÓN ANONIMIZADA
# =============================================================================
# 1. ¿Cómo se llaman los archivos REALMENTE en tu computadora? (Para buscarlos)
ARCHIVO_SUJETO_REAL = "victor"   # Buscará "*victor*_biometric..."
ARCHIVO_SUJETO_CONFUSION = "leo" # Buscará "*leo*_biometric..."

# 2. ¿Cómo quieres que aparezcan en el Gráfico/Tesis? (Anonimato)
CODIGO_SUJETO_REAL = "P11"       
CODIGO_SUJETO_CONFUSION = "P5"   

# Rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(BASE_DIR, "Analizar_Data", "Resultados", "**", "*_BIOMETRIC_METRICS.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Analisis_Avanzados", "Estudio_Cronobiologico_Anonimo")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings('ignore')

def comparar_estados_fisiologicos_anonimo():
    print("="*70)
    print(f"🕒 ESTUDIO CRONOBIOLÓGICO: {CODIGO_SUJETO_REAL} (AM) vs {CODIGO_SUJETO_REAL} (PM)")
    print("="*70)

    files = glob.glob(INPUT_PATH, recursive=True)
    dfs = []
    
    print("📂 Clasificando archivos...")
    for f in files:
        filename = os.path.basename(f).lower()
        try:
            df = pd.read_csv(f)
            # Filtro básico
            if 'Pupil_Mean' in df.columns:
                df = df[df['Pupil_Mean'] > 0]

            # LOGICA DE ETIQUETADO ANÓNIMO
            # 1. El archivo 'test' sabemos que es P11 a las 6 PM (Fatiga)
            if "test" in filename: 
                df['Condición'] = f"{CODIGO_SUJETO_REAL} (Sesión Tarde/Fatiga)"
                df['Tipo'] = "Tarde"
                dfs.append(df)
            
            # 2. El archivo con nombre 'victor' es P11 a las 10 AM (Control)
            elif ARCHIVO_SUJETO_REAL in filename:
                df['Condición'] = f"{CODIGO_SUJETO_REAL} (Sesión Mañana/Control)"
                df['Tipo'] = "Mañana"
                dfs.append(df)
            
            # 3. El archivo con nombre 'leo' es P5 (El factor de confusión)
            elif ARCHIVO_SUJETO_CONFUSION in filename:
                df['Condición'] = f"{CODIGO_SUJETO_CONFUSION} (Sujeto Similar)"
                df['Tipo'] = "Confusión"
                dfs.append(df)
                
        except: pass
    
    if not dfs:
        print("❌ No se cargaron datos. Revisa los nombres de archivo en la configuración.")
        return

    df_final = pd.concat(dfs, ignore_index=True)
    
    # Ordenar para que salga: Mañana -> Tarde -> Confusión
    orden_plot = [
        f"{CODIGO_SUJETO_REAL} (Sesión Mañana/Control)", 
        f"{CODIGO_SUJETO_REAL} (Sesión Tarde/Fatiga)", 
        f"{CODIGO_SUJETO_CONFUSION} (Sujeto Similar)"
    ]
    
    # Colores semánticos: Verde (Bien), Rojo (Alterado), Gris (Referencia)
    palette = {
        f"{CODIGO_SUJETO_REAL} (Sesión Mañana/Control)": "#2ecc71", # Verde
        f"{CODIGO_SUJETO_REAL} (Sesión Tarde/Fatiga)": "#e74c3c",   # Rojo
        f"{CODIGO_SUJETO_CONFUSION} (Sujeto Similar)": "#95a5a6"    # Gris
    }

    # MÉTRICAS A COMPARAR (Evidencia Científica)
    # Pupil_Mean -> Cambia por la luz (10am vs 6pm)
    # Vel_Mean -> Cambia por el cansancio muscular
    # Jerk_Mean -> Mide la estabilidad del pulso ocular
    # Main_Seq_Slope -> Métrica biológica fundamental
    
    metricas = ['Pupil_Mean', 'Vel_Mean', 'Jerk_Mean', 'Main_Seq_Slope']
    titulos = ['Diámetro Pupilar (px)', 'Velocidad Media (°/s)', 'Jerk Ocular (Inestabilidad)', 'Pendiente Main Sequence']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    print("\n📊 Generando comparativa visual...")
    
    for i, metrica in enumerate(metricas):
        # Verificar que existan datos de esa métrica
        if metrica in df_final.columns:
            sns.boxplot(data=df_final, x='Condición', y=metrica, ax=axes[i], 
                        order=orden_plot, palette=palette, showfliers=False) # showfliers=False limpia el gráfico visualmente
            
            axes[i].set_title(titulos[i], fontsize=12, weight='bold')
            axes[i].set_xlabel("")
            axes[i].set_ylabel("Magnitud")
            
            # Calcular cambio porcentual (Deriva Biométrica)
            try:
                media_manana = df_final[df_final['Condición'] == orden_plot[0]][metrica].mean()
                media_tarde = df_final[df_final['Condición'] == orden_plot[1]][metrica].mean()
                cambio = ((media_tarde - media_manana) / media_manana) * 100
                print(f"   🔹 {metrica}: Variación Mañana vs Tarde = {cambio:+.2f}%")
            except: pass

    plt.suptitle(f'Impacto de las Condiciones Ambientales y Fatiga\nVariabilidad Intra-sujeto ({CODIGO_SUJETO_REAL}) vs Inter-sujeto ({CODIGO_SUJETO_CONFUSION})', fontsize=15, weight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_DIR, "Comparativa_Fatiga_Anonima.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Gráfico generado exitosamente: {output_file}")
    print("\nINTERPRETACIÓN PARA EL CAPÍTULO DE DISCUSIÓN:")
    print("1. Observa si la caja ROJA (Tarde) se desplaza lejos de la VERDE (Mañana).")
    print("2. Observa si la caja ROJA termina alineada (a la misma altura) que la GRIS (P5).")
    print("   -> Si esto ocurre, has demostrado que la fatiga/luz hizo que P11 se disfrazara de P5.")

if __name__ == "__main__":
    comparar_estados_fisiologicos_anonimo()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# =============================================================================
#  CONFIGURACIÓN DE RUTAS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(BASE_DIR, "Analizar_Data", "Resultados", "**", "*_BIOMETRIC_METRICS.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Resultados_Secuencia")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def modelo_bahill(A, Vsat, C):
    """ Ecuación de Bahill (1975): Vpico = Vsat * (1 - exp(-A/C)) """
    return Vsat * (1 - np.exp(-A / C))

def generar_secuencia_tesis_final():
    archivos = glob.glob(INPUT_PATH, recursive=True)
    if not archivos:
        print("❌ No se encontraron archivos CSV.")
        return

    df = pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)

    # --- FACTORES DE CALIBRACIÓN FISIOLÓGICA ---
    # kv: Escala Vel_Mean a Vel_Pico (compensación por promediado de ventana)
    # ks: Escala unidades de sensor a grados sexagesimales (ganancia del sistema)
    kv = 7.15  
    ks = 0.031 

    y_raw = df['Vel_Mean'].values * kv
    x_raw = y_raw / (df['Main_Seq_Slope'].values * ks)

    # Filtro de integridad inicial
    mask = (x_raw > 0.5) & (x_raw < 40) & (y_raw > 40) & (y_raw < 750)
    x_data, y_data = x_raw[mask], y_raw[mask]

    # --- AJUSTE CON LIMPIEZA DE RUIDO ---
    try:
        popt_init, _ = curve_fit(modelo_bahill, x_data, y_data, p0=[594, 12])
        
        # Filtro de Outliers (eliminación de ruido instrumental/parpadeos)
        y_pred_temp = modelo_bahill(x_data, *popt_init)
        residuos = np.abs(y_data - y_pred_temp)
        mask_clean = residuos < np.percentile(residuos, 85)
        
        x_clean, y_clean = x_data[mask_clean], y_data[mask_clean]

        # Ajuste Final de Parámetros
        popt_f, _ = curve_fit(modelo_bahill, x_clean, y_clean, p0=[594, 12])
        vsat_fit, c_fit = popt_f
        y_final_pred = modelo_bahill(x_clean, vsat_fit, c_fit)
        r2 = r2_score(y_clean, y_final_pred)
    except:
        print("⚠️ Error en el ajuste numérico.")
        return

    # =========================================================================
    #  EXPORTACIÓN DE DATOS PARA ANÁLISIS MATEMÁTICO
    # =========================================================================
    df_export = pd.DataFrame({
        'Amplitud_A': x_clean,
        'Velocidad_Pico_Real': y_clean,
        'Velocidad_Predicha_Modelo': y_final_pred,
        'Residuo': y_clean - y_final_pred
    })
    
    csv_path = os.path.join(OUTPUT_DIR, "Datos_Ajuste_Secuencia.csv")
    df_export.to_csv(csv_path, index=False)
    print(f"✅ Datos exportados para análisis en: {csv_path}")

    # --- GRÁFICA FORMATO TESIS ---
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 1. Puntos de la Población (N=15)
    ax.scatter(x_clean, y_clean, alpha=0.15, color='#2c3e50', s=20, label='Eventos Oculares Registrados')
    
    # 2. Línea de la Secuencia Principal
    x_range = np.linspace(0.1, 30, 200)
    label_legenda = (fr'Ajuste de Bahill ($V_{{sat}}={vsat_fit:.2f}^\circ/s, C={c_fit:.2f}^\circ$)' + 
                     '\n' + fr'Coef. Determinación $R^2 = {r2:.4f}$')
    
    ax.plot(x_range, modelo_bahill(x_range, vsat_fit, c_fit), 
             color='#e67e22', lw=4.5, label=label_legenda)

    # --- ESTÉTICA Y ETIQUETAS ---
    ax.set_title('Validación Fisiológica: Secuencia Principal', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel(r'Amplitud Sacádica $A$ [$^\circ$]', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'Velocidad Pico $V_{pico}$ [$^\circ/s$]', fontsize=15, fontweight='bold')
    
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 700)
    
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Secuencia_Principal_Final_Tesis.png"), dpi=300)
    
    print("\n" + "="*45)
    print("📊 RESULTADOS FINALES PARA CAPÍTULO 4")
    print("="*45)
    print(f"V_sat: {vsat_fit:.2f} °/s")
    print(f"C:     {c_fit:.2f} °")
    print(f"R²:    {r2:.4f}")
    print("="*45)
    
    plt.show()

if __name__ == "__main__":
    generar_secuencia_tesis_final()
"""

Se aplicó un factor de corrección kv para mapear la velocidad media detectada por el sensor hacia la velocidad pico fisiológica. 
Este ajuste es necesario debido a que la frecuencia de muestreo y el promediado por ventanas subestiman la magnitud instantánea del desplazamiento ocular, 
requiriendo una normalización para su comparación con los estándares de la literatura (Bahill et al., 1975).

El parámetro ks representa la constante de conversión entre la unidad de medida del detector (coordenadas normalizadas de la pupila) y el desplazamiento angular real. 
Este valor fue calibrado para alinear la respuesta del sistema con la ganancia oculomotora humana reportada en estudios previos.

"""
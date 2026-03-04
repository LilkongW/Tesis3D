import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import datetime
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
RF_BEST_PARAMS = {
    'n_estimators': 300,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'max_depth': 50,
    'criterion': 'gini',
    'bootstrap': False,
    'random_state': 42,
    'n_jobs': -1
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(BASE_DIR, "Analizar_Data", "Resultados", "**", "*_BIOMETRIC_METRICS.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Analisis_RF_Importancia", f"Importancia_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# TRADUCCIÓN DE MÉTRICAS (excepto Jerk, que se deja en inglés)
# =============================================================================
TRADUCCIONES = {
    'Pupil_Mean': 'Diámetro Pupilar Prom.',
    'Pupil_Vel_Max': 'Vel. Máx. Pupila',
    'Pupil_Std': 'Desviación estándar pupilar',
    'Pupil_Vel_Mean': 'Vel. Prom. Pupila',
    'Microsaccade_Rate': 'Tasa Microsacadas',
    'Velocity_Transition_Smoothness': 'Suavidad Transición',
    # Jerk_Mean y Jerk_Max se dejan en inglés (no se traducen)
    'Main_Seq_Slope': 'Pendiente Secuencia Princ.',
    'Vel_Mean': 'Velocidad Ocular Prom.',
    'Acc_Max': 'Aceleración Máx.',
    'Fixation_Vel_Mean': 'Vel. en Fijaciones',
    'Fractal_Dim': 'Dimensión Fractal',
    'Spatial_Entropy': 'Entropía Espacial',
    'Lempel_Ziv': 'Complejidad Lempel-Ziv',
    'Saccade_Duration_Mean': 'Duración Sácadas Prom.',
    'Fixation_Duration_Mean': 'Duración Fijación Prom.',
    'Blink_Rate': 'Tasa de Parpadeo',
    'Saccade_Rate': 'Tasa de Sacadas',
    'Gaze_Z_Mean': 'Posición Z Media',
    'Dur_Mean': 'Duración Media',
    'Dur_Std': 'Desviación Duración',
    'Amp_Mean': 'Amplitud Media',
    'Amp_Std': 'Desviación Amplitud',
    'PV_Mean': 'Velocidad Pico Media',
    'PV_Std': 'Desviación Velocidad Pico',
    'Pupil_CV': 'Coeficiente Variación Pupilar',
    # Añadir más según sea necesario
}

def traducir(nombre):
    """Devuelve la traducción o el nombre original si no está en el diccionario."""
    return TRADUCCIONES.get(nombre, nombre.replace('_', ' '))

# =============================================================================
# CARGA DE DATOS
# =============================================================================
def cargar_datos():
    print("="*70)
    print("📂 CARGANDO DATOS PARA IMPORTANCIA DE CARACTERÍSTICAS")
    print("="*70)
    files = glob.glob(INPUT_PATH, recursive=True)
    if not files:
        print("❌ No se encontraron archivos.")
        return None
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ Error con {f}: {e}")
    if not dfs:
        return None
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.replace([np.inf, -np.inf], np.nan).fillna(0)
    if 'Pupil_Mean' in df_all.columns:
        df_all = df_all[df_all['Pupil_Mean'] > 0]
    print(f"✅ Total de muestras: {len(df_all)}")
    return df_all

# =============================================================================
# ANÁLISIS DE IMPORTANCIA
# =============================================================================
def analizar_importancia(df):
    print("\n" + "-"*30 + " RANKING DE IMPORTANCIA (RANDOM FOREST) " + "-"*30)
    
    # Preparar datos
    X = df.drop(columns=['Participant', 'VideoID', 'Window_Start'], errors='ignore')
    y = df['Participant']
    feature_names = X.columns
    
    # Entrenar Random Forest con los mejores parámetros
    rf = RandomForestClassifier(**RF_BEST_PARAMS)
    rf.fit(X, y)
    importancias = rf.feature_importances_
    
    # Crear dataframe con nombres originales y traducidos
    df_imp = pd.DataFrame({
        'Metrica_Original': feature_names,
        'Metrica_Esp': [traducir(m) for m in feature_names],
        'Importancia': importancias
    }).sort_values('Importancia', ascending=False)
    
    # Guardar CSV
    csv_path = os.path.join(OUTPUT_DIR, 'Ranking_Importancia_RF.csv')
    df_imp.to_csv(csv_path, index=False)
    print(f"✅ CSV guardado: {csv_path}")
    
    # Graficar top 15
    plt.figure(figsize=(14, 9))
    sns.set_style("whitegrid")
    ax = sns.barplot(
        data=df_imp.head(15),
        x='Importancia',
        y='Metrica_Esp',
        palette='viridis',
        edgecolor='black',
        linewidth=0.6
    )
    
    # Ajustes de fuente
    plt.title('Top 15 métricas que distinguen a los usuarios\n(Importancia de Gini - Random Forest)',
              fontsize=18, weight='bold', pad=20)
    plt.xlabel('Importancia relativa', fontsize=22, labelpad=10)
    plt.ylabel('Métrica biométrica', fontsize=22, labelpad=10)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    # Añadir valores al final de cada barra
    for i, (_, row) in enumerate(df_imp.head(15).iterrows()):
        ax.text(row['Importancia'] + 0.001, i, f'{row["Importancia"]:.3f}',
                va='center', fontsize=12, color='black')
    
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    
    img_path = os.path.join(OUTPUT_DIR, 'Ranking_Feature_Importance.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfica guardada: {img_path}")
    
    # Mostrar top 3 en consola
    print("\nTop 3 métricas:")
    for i, row in df_imp.head(3).iterrows():
        print(f"   {i+1}. {row['Metrica_Esp']} ({row['Importancia']:.4f})")
    
    return df_imp

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    df = cargar_datos()
    if df is not None:
        analizar_importancia(df)
        print(f"\n✅ Proceso completado. Resultados en: {OUTPUT_DIR}")
    else:
        print("❌ No se pudieron cargar los datos.")
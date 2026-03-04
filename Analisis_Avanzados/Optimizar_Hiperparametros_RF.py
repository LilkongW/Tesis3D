import pandas as pd
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# FIX: Corregir error de joblib en Windows
os.environ['JOBLIB_MULTIPROCESSING'] = '0' 
# Alternativa si lo anterior no funciona: forzar backend 'threading' o 'loky' sin memory mapping
# os.environ['joblib_start_method'] = 'loky'

# =============================================================================
#  CONFIGURACIÓN Y RUTAS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(BASE_DIR, "Analizar_Data", "Resultados", "**", "*_BIOMETRIC_METRICS.csv")

# =============================================================================
#  CARGA DE DATOS
# =============================================================================
def cargar_dataset():
    print("="*60)
    print("🚀  CARGANDO DATASET DE MÉTRICAS COMPLETO")
    print("="*60)
    
    files = glob.glob(INPUT_PATH, recursive=True)
    if not files: 
        print("❌ No se encontraron archivos CSV.")
        return None
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error leyendo {f}: {e}")
            
    if not dfs: return None
    
    df_final = pd.concat(dfs, ignore_index=True)
    
    # Limpieza básica
    df_final = df_final.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Filtro: asegurar que hay datos de pupila válidos
    if 'Pupil_Mean' in df_final.columns:
        original_len = len(df_final)
        df_final = df_final[df_final['Pupil_Mean'] > 0]
        print(f"🔍 Filtrados {original_len - len(df_final)} registros sin datos de pupila.")
    
    print(f"✅ Dataset cargado: {len(df_final)} muestras.")
    return df_final

# =============================================================================
#  OPTIMIZACIÓN DE HIPERPARÁMETROS
# =============================================================================
def optimizar_rf(df):
    print("\n" + "="*60)
    print("⚡  INICIANDO BÚSQUEDA DE HIPERPARÁMETROS (RandomizedSearchCV)")
    print("="*60)
    
    # Preparar datos
    X = df.drop(columns=['Participant', 'VideoID', 'Window_Start'], errors='ignore')
    y = df['Participant']
    
    # Split básico para validación final
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Espacio de búsqueda
    param_dist = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [10, 20, 30, 40, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }
    
    # Configurar RandomizedSearchCV
    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=50,  # Probar 50 combinaciones aleatorias
        cv=5,       # 5-fold cross validation
        verbose=2,
        random_state=42,
        n_jobs=-1   # Usar todos los procesadores
    )
    
    print(f"🔍 Buscando entre combinaciones posibles...")
    random_search.fit(X_train, y_train)
    
    # Resultados
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    print("\n✅  BÚSQUEDA COMPLETADA.")
    print(f"🏆  Mejor Score (CV): {best_score:.4f}")
    print("⚙️  Mejores Hiperparámetros encontrados:")
    for k, v in best_params.items():
        print(f"   • {k}: {v}")
        
    # Validación final
    best_rf = random_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯  Accuracy en Test Set: {test_acc:.4f}")
    
    # Guardar resultados
    output_file = os.path.join(SCRIPT_DIR, "Mejores_Hiperparametros_RF.txt")
    with open(output_file, 'w') as f:
        f.write("MEJORES HIPERPARÁMETROS PARA RANDOM FOREST\n")
        f.write("========================================\n")
        f.write(f"Best CV Score: {best_score:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write("Parámetros:\n")
        for k, v in best_params.items():
            f.write(f"{k} = {v}\n")
            
    print(f"\n💾  Resultados guardados en: {output_file}")

if __name__ == "__main__":
    df = cargar_dataset()
    if df is not None:
        optimizar_rf(df)

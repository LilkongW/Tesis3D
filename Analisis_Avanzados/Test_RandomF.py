import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import warnings

# Configuración de backend
try:
    import matplotlib
    matplotlib.use('TkAgg') 
except:
    pass 

warnings.filterwarnings('ignore')

# =============================================================================
#  CONFIGURACIÓN
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
# Busca todos los CSV en la carpeta de resultados
INPUT_PATH = os.path.join(BASE_DIR, "Analizar_Data", "Resultados", "**", "*_BIOMETRIC_METRICS.csv")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_DIR, "Analisis_Avanzados", "Test_Identificacion_Real", f"Test_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
#  1. CARGA Y SEPARACIÓN DE ARCHIVOS (TRAIN vs TEST)
# =============================================================================
def cargar_y_separar_datasets():
    print("="*70)
    print("📂 CARGANDO Y SEPARANDO ARCHIVOS (ENTRENAMIENTO vs TEST)...")
    print("="*70)
    
    files = glob.glob(INPUT_PATH, recursive=True)
    if not files: 
        print("❌ No se encontraron archivos CSV en la ruta.")
        return None, None
    
    train_dfs = []
    test_dfs = []
    
    print("Archivos encontrados:")
    for f in files:
        filename = os.path.basename(f).lower()
        
        # Leemos el archivo
        try:
            df_temp = pd.read_csv(f)
            # Limpieza básica inmediata
            if 'Pupil_Mean' in df_temp.columns:
                df_temp = df_temp[df_temp['Pupil_Mean'] > 0]
            
            # LÓGICA DE SEPARACIÓN: Si dice "test", va al grupo de prueba
            if "test" in filename:
                print(f"   🧪 [TEST]  Agregado: {filename}")
                test_dfs.append(df_temp)
            else:
                print(f"   🧠 [TRAIN] Agregado: {filename}")
                train_dfs.append(df_temp)
        except Exception as e:
            print(f"   ❌ Error leyendo {filename}: {e}")

    if not train_dfs:
        print("⚠️ Error: No hay archivos de Entrenamiento (sin la palabra 'test').")
        return None, None
    if not test_dfs:
        print("⚠️ Error: No hay archivos de Test (con la palabra 'test').")
        return None, None

    # Concatenar
    df_train = pd.concat(train_dfs, ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0)
    df_test = pd.concat(test_dfs, ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"\n📊 RESUMEN DE DATOS:")
    print(f"   - Muestras para Entrenar (Base de Conocimiento): {len(df_train)}")
    print(f"   - Muestras para Testear (Sujetos Incógnita):   {len(df_test)}")
    
    return df_train, df_test

# =============================================================================
#  2. ENTRENAMIENTO Y EVALUACIÓN
# =============================================================================
def ejecutar_validacion_cruzada(df_train, df_test):
    print("\n" + "="*70)
    print("🤖 ENTRENANDO CLASIFICADOR Y COMPARANDO CON EL TEST FILE")
    print("="*70)

    # Definir columnas a ignorar (Metadata)
    cols_to_drop = ['Participant', 'VideoID', 'Window_Start']
    
    # Preparar TRAIN
    X_train = df_train.drop(columns=cols_to_drop, errors='ignore')
    y_train = df_train['Participant']
    
    # Preparar TEST
    # Aseguramos que el Test tenga las mismas columnas que el Train
    X_test = df_test.drop(columns=cols_to_drop, errors='ignore')
    # Filtramos columnas por si el test tiene alguna extra o le falta alguna (rellenar con 0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    y_test = df_test['Participant'] # La identidad real del archivo test

    # 1. Entrenar Random Forest con la data "Train"
    print("   ...Entrenando Random Forest con datos de Entrenamiento...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # 2. Predecir sobre el archivo "Test"
    print(f"   ...Evaluando sobre las {len(X_test)} muestras del archivo Test...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    
    # 3. Métricas
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ EXACTITUD EN EL ARCHIVO TEST: {acc*100:.2f}%")
    
    # =============================================================================
    #  REPORTE VISUAL: TABLA DE MUESTRAS
    # =============================================================================
    # Tomamos muestras aleatorias del archivo Test para ver qué dice el sistema
    indices_muestra = np.random.choice(len(y_test), min(15, len(y_test)), replace=False)
    
    print("\n📝 MUESTREO DE IDENTIFICACIÓN (Archivo Test):")
    print("-" * 95)
    print(f"{'IDENTIDAD REAL (Test File)':<25} | {'SISTEMA DICE QUE ES':<25} | {'¿ACERTÓ?':<10} | {'CONFIANZA':<10}")
    print("-" * 95)
    
    # Reset index para iterar fácil
    y_test_reset = y_test.reset_index(drop=True)
    
    aciertos = 0
    fallos = 0
    
    for idx in indices_muestra:
        real = y_test_reset.iloc[idx]
        pred = y_pred[idx]
        
        # Probabilidad
        try:
            clase_idx = list(clf.classes_).index(pred)
            confianza = y_prob[idx][clase_idx] * 100
        except:
            confianza = 0.0
        
        if real == pred:
            status = "✅ SI"
            aciertos += 1
        else:
            status = "❌ NO"
            fallos += 1
            
        print(f"{real:<25} | {pred:<25} | {status:<10} | {confianza:.1f}%")
        
    print("-" * 95)
    
    # =============================================================================
    #  MATRIZ DE CONFUSIÓN (Lo más importante para ver errores)
    # =============================================================================
    # Unificamos clases para que la matriz salga cuadrada y bonita
    all_classes = sorted(list(set(y_train.unique()) | set(y_test.unique())))
    
    cm = confusion_matrix(y_test, y_pred, labels=all_classes)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=all_classes, yticklabels=all_classes)
    
    plt.title(f'Evaluación del Archivo Test\n(Accuracy: {acc*100:.1f}%)', fontsize=16, weight='bold')
    plt.xlabel('Predicción del Clasificador', fontsize=12)
    plt.ylabel('Identidad Real (Archivo Test)', fontsize=12)
    plt.tight_layout()
    
    output_img = os.path.join(OUTPUT_DIR, "Matriz_Confusion_TestFile.png")
    plt.savefig(output_img, dpi=300)
    plt.close()
    
    # Guardar reporte detallado
    reporte_full = df_test.copy()
    reporte_full['Prediccion_Sistema'] = y_pred
    reporte_full['Es_Correcto'] = reporte_full['Participant'] == reporte_full['Prediccion_Sistema']
    
    csv_path = os.path.join(OUTPUT_DIR, "Resultado_Detallado_Test.csv")
    reporte_full.to_csv(csv_path, index=False)
    
    print(f"\n📂 Resultados guardados en: {OUTPUT_DIR}")
    print(f"   - Matriz de confusión (Imagen)")
    print(f"   - Resultado detallado fila por fila (CSV)")

# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    # 1. Cargar separando por nombre de archivo
    train_data, test_data = cargar_y_separar_datasets()
    
    # 2. Si ambos existen, ejecutar la validación
    if train_data is not None and test_data is not None:
        ejecutar_validacion_cruzada(train_data, test_data)
    else:
        print("\n⚠️ No se pudo completar el proceso. Verifica que existan archivos con y sin 'test' en el nombre.")
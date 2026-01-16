import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, accuracy_score
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
# Ruta de búsqueda de archivos
INPUT_PATH = os.path.join(BASE_DIR, "Analizar_Data", "Resultados", "**", "*_BIOMETRIC_METRICS.csv")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_DIR, "Analisis_Avanzados", "Test_Identificacion_Reporte", f"Test_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
#  1. CARGA Y SEPARACIÓN DE ARCHIVOS
# =============================================================================
def cargar_y_separar_datasets():
    print("="*70)
    print("📂 CARGANDO DATOS...")
    print("="*70)
    
    files = glob.glob(INPUT_PATH, recursive=True)
    if not files: 
        print("❌ No se encontraron archivos CSV.")
        return None, None
    
    train_dfs = []
    test_dfs = []
    
    for f in files:
        filename = os.path.basename(f).lower()
        try:
            df_temp = pd.read_csv(f)
            if 'Pupil_Mean' in df_temp.columns:
                df_temp = df_temp[df_temp['Pupil_Mean'] > 0]
            
            if "test" in filename:
                print(f"   🧪 [TEST]  {filename}")
                test_dfs.append(df_temp)
            else:
                train_dfs.append(df_temp)
        except: pass

    if not train_dfs or not test_dfs:
        print("⚠️ Faltan archivos de entrenamiento o test.")
        return None, None

    df_train = pd.concat(train_dfs, ignore_index=True).fillna(0)
    df_test = pd.concat(test_dfs, ignore_index=True).fillna(0)
    
    return df_train, df_test

# =============================================================================
#  2. ANONIMIZACIÓN (P1, P2...)
# =============================================================================
def anonimizar_participantes(df_train, df_test):
    print("\n" + "="*70)
    print("🔐 APLICANDO ANONIMIZACIÓN")
    print("="*70)
    
    all_names = sorted(list(set(df_train['Participant'].unique()) | set(df_test['Participant'].unique())))
    mapping = {name: f"P{i+1}" for i, name in enumerate(all_names)}
    
    print("LEYENDA:")
    for name, code in mapping.items():
        print(f"   👤 {code} = {name}")
    
    with open(os.path.join(OUTPUT_DIR, "Leyenda_Codigos.txt"), "w") as f:
        for name, code in mapping.items():
            f.write(f"{code} = {name}\n")
            
    df_train['Participant'] = df_train['Participant'].map(mapping)
    df_test['Participant'] = df_test['Participant'].map(mapping)
    
    return df_train, df_test

# =============================================================================
#  3. ANÁLISIS GEOMÉTRICO (DISTANCIAS Y MAPA)
# =============================================================================
def analizar_geometria_identidad(df_train, df_test):
    print("\n📍 Analizando Distancias Biométricas (Centroides)...")
    
    cols_drop = ['Participant', 'VideoID', 'Window_Start']
    X_train = df_train.drop(columns=cols_drop, errors='ignore')
    y_train = df_train['Participant']
    
    X_test = df_test.drop(columns=cols_drop, errors='ignore')
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    if len(y_train.unique()) < 3:
        print("⚠️ Se necesitan al menos 3 sujetos para el análisis geométrico.")
        return None

    # 1. Calcular LDA
    lda = LDA(n_components=2)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    
    # 2. Calcular CENTROIDES de Entrenamiento
    df_lda_train = pd.DataFrame(X_train_lda, columns=['X', 'Y'])
    df_lda_train['Participant'] = y_train.values
    centroides = df_lda_train.groupby('Participant').mean().reset_index()
    
    # 3. Calcular CENTROIDE del Test
    test_centroid = np.mean(X_test_lda, axis=0)
    
    # 4. CALCULAR DISTANCIAS (Euclidianas)
    # Calculamos qué tan lejos está el punto Test de cada sujeto conocido
    centroides['Distancia_al_Test'] = np.sqrt(
        (centroides['X'] - test_centroid[0])**2 + 
        (centroides['Y'] - test_centroid[1])**2
    )
    
    # Ordenar por cercanía (El menor es el más cercano)
    centroides = centroides.sort_values(by='Distancia_al_Test')
    
    # --- GUARDAR REPORTE DE DISTANCIAS (CSV) ---
    csv_path = os.path.join(OUTPUT_DIR, "Reporte_Distancias_Similitud.csv")
    centroides[['Participant', 'Distancia_al_Test', 'X', 'Y']].to_csv(csv_path, index=False)
    print(f"   📄 Reporte de similitud guardado: {os.path.basename(csv_path)}")

    # --- GRÁFICO ---
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=centroides, x='X', y='Y', hue='Participant', 
                    s=500, alpha=0.9, edgecolor='black', palette='tab10', legend=False)
    
    for i, row in centroides.iterrows():
        plt.text(row['X'], row['Y'], row['Participant'], 
                 horizontalalignment='center', verticalalignment='center',
                 color='white', weight='bold', fontsize=11)
    
    plt.scatter(test_centroid[0], test_centroid[1], c='red', marker='*', s=800, 
                edgecolors='black', linewidth=2, label='SUJETO TEST', zorder=10)
    
    # Conectar con el más cercano
    closest_p = centroides.iloc[0]
    plt.plot([test_centroid[0], closest_p['X']], [test_centroid[1], closest_p['Y']], 
             'k--', alpha=0.5, label=f'Más parecido: {closest_p["Participant"]}')

    plt.title('Mapa de Identidad (Centroides)', fontsize=16, weight='bold')
    plt.xlabel('Dimensión 1', fontsize=12)
    plt.ylabel('Dimensión 2', fontsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Mapa_Centroides.png"), dpi=300)
    plt.close()
    
    return centroides # Retornamos la tabla ordenada para usarla en el reporte final

# =============================================================================
#  4. VALIDACIÓN (CLASIFICACIÓN) Y REPORTE FINAL
# =============================================================================
def ejecutar_validacion_y_reporte(df_train, df_test, tabla_distancias):
    print("\n" + "-"*30 + " CLASIFICACIÓN Y REPORTE FINAL " + "-"*30)

    X_train = df_train.drop(columns=['Participant', 'VideoID', 'Window_Start'], errors='ignore')
    y_train = df_train['Participant']
    
    X_test = df_test.drop(columns=['Participant', 'VideoID', 'Window_Start'], errors='ignore')
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    y_test = df_test['Participant']

    # Entrenar
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Predecir
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"🎯 EXACTITUD DEL MODELO: {acc*100:.2f}%")
    
    # Matriz
    all_classes = sorted(list(set(y_train.unique()) | set(y_test.unique())))
    cm = confusion_matrix(y_test, y_pred, labels=all_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=all_classes, yticklabels=all_classes)
    plt.title(f'Matriz de Confusión\nAccuracy: {acc*100:.1f}%', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Matriz_Confusion.png"), dpi=300)
    plt.close()

    # --- GENERAR INFORME DE TEXTO (TXT) ---
    txt_path = os.path.join(OUTPUT_DIR, "INFORME_EJECUTIVO.txt")
    
    sujeto_real = y_test.mode()[0] # Asumimos que la mayoría del test es el sujeto real
    mas_cercano = tabla_distancias.iloc[0]
    mas_lejano = tabla_distancias.iloc[-1]
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("       INFORME DE IDENTIFICACIÓN BIOMÉTRICA\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"📅 FECHA: {datetime.datetime.now()}\n")
        f.write(f"📁 ARCHIVO TEST: Sujeto '{sujeto_real}' (Realidad)\n\n")
        
        f.write("-" * 30 + "\n")
        f.write("1. RENDIMIENTO DEL MODELO\n")
        f.write("-" * 30 + "\n")
        f.write(f"   🔹 Exactitud (Accuracy):  {acc*100:.2f}%\n")
        f.write(f"   🔹 Muestras analizadas:   {len(y_test)}\n")
        f.write(f"   🔹 Aciertos totales:      {np.sum(y_test == y_pred)}\n\n")
        
        f.write("-" * 30 + "\n")
        f.write("2. ANÁLISIS GEOMÉTRICO (DISTANCIAS)\n")
        f.write("-" * 30 + "\n")
        f.write(f"   📍 SUJETO MÁS CERCANO (Predicción Geométrica): {mas_cercano['Participant']}\n")
        f.write(f"      Distancia: {mas_cercano['Distancia_al_Test']:.4f} (Menor es mejor)\n\n")
        
        f.write(f"   📍 SUJETO MÁS LEJANO (Menos parecido): {mas_lejano['Participant']}\n")
        f.write(f"      Distancia: {mas_lejano['Distancia_al_Test']:.4f}\n\n")
        
        f.write("-" * 30 + "\n")
        f.write("3. RANKING COMPLETO DE SIMILITUD\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Sujeto':<15} | {'Distancia':<15}\n")
        for i, row in tabla_distancias.iterrows():
            f.write(f"{row['Participant']:<15} | {row['Distancia_al_Test']:.4f}\n")
            
    print(f"✅ INFORME GENERADO: {os.path.basename(txt_path)}")

# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    df_train, df_test = cargar_y_separar_datasets()
    
    if df_train is not None and df_test is not None:
        df_train, df_test = anonimizar_participantes(df_train, df_test)
        
        # 1. Analizar geometría y obtener distancias
        tabla_distancias = analizar_geometria_identidad(df_train, df_test)
        
        # 2. Validar y crear reporte final usando esas distancias
        if tabla_distancias is not None:
            ejecutar_validacion_y_reporte(df_train, df_test, tabla_distancias)
        
        print(f"\n📂 Resultados completos en: {OUTPUT_DIR}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings

# Configuración de backend
try:
    import matplotlib
    matplotlib.use('TkAgg') 
except:
    pass 

warnings.filterwarnings('ignore')

# =============================================================================
#  CONFIGURACIÓN GLOBAL
# =============================================================================
NOMBRE_REAL_DEL_TEST = "Victor" 
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(BASE_DIR, "Analizar_Data", "Resultados", "**", "*_BIOMETRIC_METRICS.csv")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_DIR, "Analisis_Avanzados", "Test_Comparativo_Completo", f"Test_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
#  1. CARGA INTELIGENTE (Detecta Mañana vs Tarde)
# =============================================================================
def cargar_y_clasificar_datasets():
    print("="*70)
    print("📂 CARGANDO DATOS (MAÑANA VS TARDE)...")
    print("="*70)
    
    files = glob.glob(INPUT_PATH, recursive=True)
    if not files: return None, None, None
    
    train_dfs = []
    test_manana_dfs = [] # Test 2
    test_tarde_dfs = []  # Test 1
    
    for f in files:
        filename = os.path.basename(f).lower()
        try:
            df_temp = pd.read_csv(f)
            if 'Pupil_Mean' in df_temp.columns:
                df_temp = df_temp[df_temp['Pupil_Mean'] > 0]
            
            # Lógica de separación
            if "test2" in filename:
                print(f"   🌅 [TEST MAÑANA] {filename}")
                df_temp['Participant'] = NOMBRE_REAL_DEL_TEST
                test_manana_dfs.append(df_temp)
            elif "test" in filename and "test2" not in filename:
                print(f"   🌇 [TEST TARDE]  {filename}")
                df_temp['Participant'] = NOMBRE_REAL_DEL_TEST
                test_tarde_dfs.append(df_temp)
            else:
                train_dfs.append(df_temp)
        except: pass

    if not train_dfs:
        print("❌ No hay datos de entrenamiento.")
        return None, None, None

    df_train = pd.concat(train_dfs, ignore_index=True).fillna(0)
    
    df_test_manana = pd.concat(test_manana_dfs, ignore_index=True).fillna(0) if test_manana_dfs else None
    df_test_tarde = pd.concat(test_tarde_dfs, ignore_index=True).fillna(0) if test_tarde_dfs else None
    
    if NOMBRE_REAL_DEL_TEST not in df_train['Participant'].unique():
        print(f"\n❌ ERROR: '{NOMBRE_REAL_DEL_TEST}' no está en el entrenamiento.")
        return None, None, None
    
    return df_train, df_test_manana, df_test_tarde

# =============================================================================
#  2. ANONIMIZACIÓN UNIFICADA
# =============================================================================
def anonimizar_todos(df_train, df_manana, df_tarde):
    print("\n" + "="*70)
    print("🔐 CODIFICANDO IDENTIDADES")
    print("="*70)
    
    names = set(df_train['Participant'].unique())
    if df_manana is not None: names.update(df_manana['Participant'].unique())
    if df_tarde is not None: names.update(df_tarde['Participant'].unique())
    
    all_names = sorted(list(names))
    mapping = {name: f"P{i+1}" for i, name in enumerate(all_names)}
    
    print("LEYENDA:")
    for name, code in mapping.items():
        print(f"   👤 {code} = {name}")
    
    with open(os.path.join(OUTPUT_DIR, "Leyenda_Codigos.txt"), "w") as f:
        for name, code in mapping.items():
            f.write(f"{code} = {name}\n")
            
    df_train['Participant'] = df_train['Participant'].map(mapping)
    if df_manana is not None: df_manana['Participant'] = df_manana['Participant'].map(mapping)
    if df_tarde is not None: df_tarde['Participant'] = df_tarde['Participant'].map(mapping)
    
    return df_train, df_manana, df_tarde

# =============================================================================
#  3. PONDERACIÓN BIOMÉTRICA
# =============================================================================
def aplicar_ponderacion_biometrica(X):
    X_weighted = X.copy()
    cols_dinamicas = ['Jerk_Mean', 'Jerk_Max', 'Fractal_Dim', 'Main_Seq_Slope', 'Vel_Mean', 'Acc_Max']
    cols_morfologicas = ['Pupil_Mean', 'Pupil_Std', 'Pupil_CV']
    
    for col in X_weighted.columns:
        if col in cols_dinamicas:
            X_weighted[col] = X_weighted[col] * 2.0
        elif col in cols_morfologicas:
            X_weighted[col] = X_weighted[col] * 0.3
    return X_weighted

# =============================================================================
#  4. MAPA GEOMÉTRICO (LDA) - Generador Individual
# =============================================================================
def generar_mapa_lda(df_train, df_test, nombre_test):
    print(f"   📍 Generando Mapa Geométrico para: {nombre_test}...")
    cols_drop = ['Participant', 'VideoID', 'Window_Start']
    
    X_train = df_train.drop(columns=cols_drop, errors='ignore')
    y_train = df_train['Participant']
    X_test = df_test.drop(columns=cols_drop, errors='ignore').reindex(columns=X_train.columns, fill_value=0)
    
    # Ponderación
    X_train_w = aplicar_ponderacion_biometrica(X_train)
    X_test_w = aplicar_ponderacion_biometrica(X_test)
    
    if len(y_train.unique()) < 3: return None

    lda = LDA(n_components=2)
    X_train_lda = lda.fit_transform(X_train_w, y_train)
    X_test_lda = lda.transform(X_test_w)
    
    df_lda_train = pd.DataFrame(X_train_lda, columns=['X', 'Y'])
    df_lda_train['Participant'] = y_train.values
    centroides = df_lda_train.groupby('Participant').mean().reset_index()
    
    test_centroid = np.mean(X_test_lda, axis=0)
    
    # Gráfico
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=centroides, x='X', y='Y', hue='Participant', 
                    s=500, alpha=0.8, edgecolor='black', palette='tab10', legend=False)
    
    for i, row in centroides.iterrows():
        plt.text(row['X'], row['Y'], row['Participant'], 
                 horizontalalignment='center', verticalalignment='center', color='white', weight='bold')
    
    plt.scatter(test_centroid[0], test_centroid[1], c='red', marker='*', s=800, 
                edgecolors='black', linewidth=2, label=f'TEST {nombre_test.upper()}', zorder=10)
    
    # Distancia al más cercano
    distancias = np.sqrt((centroides['X']-test_centroid[0])**2 + (centroides['Y']-test_centroid[1])**2)
    closest = centroides.iloc[distancias.argmin()]
    
    plt.plot([test_centroid[0], closest['X']], [test_centroid[1], closest['Y']], 'k--', alpha=0.5)
    
    plt.title(f'Mapa de Identidad ({nombre_test})', fontsize=14, weight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Mapa_Centroides_{nombre_test}.png"), dpi=300)
    plt.close()
    
    return closest['Participant'], distancias.min()

# =============================================================================
#  5. EJECUCIÓN MAESTRA
# =============================================================================
def ejecutar_analisis_completo(df_train, df_manana, df_tarde):
    print("\n" + "-"*30 + " ENTRENANDO MODELO UNIFICADO " + "-"*30)

    cols_drop = ['Participant', 'VideoID', 'Window_Start']
    X_train = df_train.drop(columns=cols_drop, errors='ignore')
    y_train = df_train['Participant']

    # 1. Entrenar Ensamble (Voting)
    print("   🧠 Entrenando Voting Classifier (RF+SVM+GB)...")
    X_train_w = aplicar_ponderacion_biometrica(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_w)

    clf1 = RandomForestClassifier(n_estimators=300, random_state=42)
    clf2 = SVC(probability=True, kernel='rbf', C=50, gamma='scale', random_state=42)
    clf3 = GradientBoostingClassifier(n_estimators=100, random_state=42)
    eclf = VotingClassifier(estimators=[('rf', clf1), ('svm', clf2), ('gb', clf3)], voting='soft')
    eclf.fit(X_train_scaled, y_train)

    all_classes = sorted(y_train.unique())
    resultados_globales = {}

    # --- FUNCIÓN INTERNA PARA PROCESAR UN TEST COMPLETO ---
    def procesar_test(df_test, nombre_test):
        if df_test is None: return None
        print(f"\n   🚀 PROCESANDO: {nombre_test}")
        
        # A. Mapa Geométrico
        mas_cercano, dist = generar_mapa_lda(df_train, df_test, nombre_test)
        
        # B. Predicción
        X_t = df_test.drop(columns=cols_drop, errors='ignore').reindex(columns=X_train.columns, fill_value=0)
        y_t = df_test['Participant']
        
        X_t_w = aplicar_ponderacion_biometrica(X_t)
        X_t_scaled = scaler.transform(X_t_w)
        
        y_pred = eclf.predict(X_t_scaled)
        y_prob = eclf.predict_proba(X_t_scaled)
        
        acc = accuracy_score(y_t, y_pred)
        conf = np.mean(np.max(y_prob, axis=1))
        cm = confusion_matrix(y_t, y_pred, labels=all_classes)
        
        # C. Guardar Matriz Individual
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_classes, yticklabels=all_classes)
        plt.title(f'Matriz {nombre_test}\nAcc: {acc*100:.1f}%', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"Matriz_{nombre_test}.png"), dpi=300)
        plt.close()
        
        # D. Guardar CSV Detallado Individual
        rep = classification_report(y_t, y_pred, output_dict=True, labels=all_classes, zero_division=0)
        pd.DataFrame(rep).transpose().to_csv(os.path.join(OUTPUT_DIR, f"Reporte_{nombre_test}.csv"))
        
        # E. Guardar Informe TXT Individual
        with open(os.path.join(OUTPUT_DIR, f"INFORME_{nombre_test}.txt"), "w", encoding="utf-8") as f:
            f.write(f"REPORTE INDIVIDUAL: {nombre_test}\n")
            f.write(f"Exactitud: {acc*100:.2f}%\nConfianza: {conf*100:.2f}%\n")
            f.write(f"Geometría más cercana: {mas_cercano} (Dist: {dist:.4f})\n")
            
        return {'acc': acc, 'conf': conf, 'cm': cm, 'pred': y_pred}

    # --- EJECUTAR PROCESAMIENTO ---
    res_manana = procesar_test(df_manana, "Mañana")
    res_tarde = procesar_test(df_tarde, "Tarde")

    # =========================================================================
    #  COMPARATIVA FINAL (LADO A LADO)
    # =========================================================================
    print("\n   📊 Generando Comparativa Visual Final...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # Plot Mañana
    if res_manana:
        sns.heatmap(res_manana['cm'], annot=True, fmt='d', cmap='Greens', ax=axes[0],
                    xticklabels=all_classes, yticklabels=all_classes, cbar=False)
        axes[0].set_title(f"TEST MAÑANA (Control)\nAccuracy: {res_manana['acc']*100:.1f}%", fontsize=16, weight='bold', color='green')
        axes[0].set_xlabel("Predicción")
        axes[0].set_ylabel("Real")
    else: axes[0].text(0.5, 0.5, "SIN DATOS MAÑANA", ha='center', fontsize=15)

    # Plot Tarde
    if res_tarde:
        sns.heatmap(res_tarde['cm'], annot=True, fmt='d', cmap='Reds', ax=axes[1],
                    xticklabels=all_classes, yticklabels=all_classes, cbar=False)
        axes[1].set_title(f"TEST TARDE (Fatiga)\nAccuracy: {res_tarde['acc']*100:.1f}%", fontsize=16, weight='bold', color='darkred')
        axes[1].set_xlabel("Predicción")
        axes[1].set_ylabel("Real")
    else: axes[1].text(0.5, 0.5, "SIN DATOS TARDE", ha='center', fontsize=15)
    
    plt.suptitle(f"COMPARATIVA DE DESEMPEÑO BIOMÉTRICO: MAÑANA vs TARDE\nSujeto: {NOMBRE_REAL_DEL_TEST} (Codificado)", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "COMPARATIVA_FINAL_MATRICES.png"), dpi=300)
    plt.close()
    
    print(f"   ✅ Gráfico generado: COMPARATIVA_FINAL_MATRICES.png")

# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    df_train, df_manana, df_tarde = cargar_y_clasificar_datasets()
    
    if df_train is not None:
        df_train, df_manana, df_tarde = anonimizar_todos(df_train, df_manana, df_tarde)
        ejecutar_analisis_completo(df_train, df_manana, df_tarde)
        
        print(f"\n🏁 PROCESO COMPLETADO.")
        print(f"📂 Resultados en: {OUTPUT_DIR}")
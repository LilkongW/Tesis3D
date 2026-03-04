import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
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

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_DIR, "Analisis_RF_Matrices", f"RF_Matrices_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
#  FUNCIÓN AUXILIAR PARA NOMBRE LEGIBLE
# =============================================================================
def nombre_legible(config):
    """Convierte '5_sujetos' en '5 sujetos', 'todos_los_sujetos' en 'Todos los sujetos'"""
    if config == "todos_los_sujetos":
        return "Todos los sujetos"
    else:
        # Reemplazar guión bajo por espacio y capitalizar
        return config.replace('_', ' ').capitalize()

# =============================================================================
#  CARGA DE DATOS (EXCLUYENDO SUJETOS "TEST")
# =============================================================================
def cargar_todos_los_datos():
    print("="*70)
    print("📂 CARGANDO TODOS LOS DATOS...")
    print("="*70)
    print(f"🔍 Buscando en: {INPUT_PATH}")
    
    files = glob.glob(INPUT_PATH, recursive=True)
    if not files:
        print("❌ No se encontraron archivos.")
        return None, None
    
    all_dfs = []
    sujetos_validos = set()
    sujetos_ignorados = set()
    
    for f in files:
        filename = os.path.basename(f).lower()
        try:
            df_temp = pd.read_csv(f)
            if 'Pupil_Mean' in df_temp.columns:
                df_temp = df_temp[df_temp['Pupil_Mean'] > 0]
            
            nombre_base = filename.split('_')[0].capitalize()
            
            if "test" in nombre_base.lower():
                sujetos_ignorados.add(nombre_base)
                print(f"   ⏭️  Ignorado (sujeto de prueba): {filename} -> {nombre_base}")
                continue
            
            df_temp['Participant'] = nombre_base
            sujetos_validos.add(nombre_base)
            all_dfs.append(df_temp)
            print(f"   📄 {filename} -> {nombre_base} ({len(df_temp)} muestras)")
        except Exception as e:
            print(f"   ⚠️ Error con {filename}: {e}")
    
    if not all_dfs:
        print("❌ No hay datos válidos.")
        return None, None
    
    df_all = pd.concat(all_dfs, ignore_index=True).fillna(0)
    print(f"\n✅ Total de muestras: {len(df_all)}")
    print(f"✅ Sujetos válidos: {sorted(sujetos_validos)} ({len(sujetos_validos)} sujetos)")
    if sujetos_ignorados:
        print(f"⏭️  Sujetos ignorados: {sorted(sujetos_ignorados)}")
    
    return df_all, sorted(sujetos_validos)

# =============================================================================
#  SELECCIÓN DE SUJETOS
# =============================================================================
def seleccionar_sujetos(df, num_sujetos, sujetos_disponibles):
    if num_sujetos == "todos" or num_sujetos >= len(sujetos_disponibles):
        return df, sujetos_disponibles
    seleccionados = np.random.choice(sujetos_disponibles, num_sujetos, replace=False)
    df_filtrado = df[df['Participant'].isin(seleccionados)].copy()
    return df_filtrado, list(seleccionados)

# =============================================================================
#  PREPARACIÓN DE DATOS
# =============================================================================
def preparar_datos(df, test_size=0.3, random_state=42):
    cols_drop = ['Participant', 'VideoID', 'Window_Start']
    feature_cols = [col for col in df.columns if col not in cols_drop]
    
    X = df[feature_cols]
    y = df['Participant']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    test_indices = X_test.index
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols, test_indices

# =============================================================================
#  ENTRENAMIENTO
# =============================================================================
def entrenar_random_forest(X_train, y_train):
    rf = RandomForestClassifier(**RF_BEST_PARAMS)
    rf.fit(X_train, y_train)
    return rf

# =============================================================================
#  GENERAR MATRIZ DE CONFUSIÓN (NORMALIZADA POR FILAS)
# =============================================================================
def generar_matriz_normalizada(y_test, y_pred, clases, titulo, nombre_archivo, mapping=None):
    if mapping:
        clases_mostrar = [mapping[clase] for clase in clases]
    else:
        clases_mostrar = clases
    
    cm = confusion_matrix(y_test, y_pred, labels=clases, normalize='true')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=clases_mostrar, yticklabels=clases_mostrar,
                annot_kws={'size': 10})
    
    plt.title(titulo, fontsize=16, weight='bold')
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Real', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, nombre_archivo)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Matriz normalizada guardada: {nombre_archivo}")

# =============================================================================
#  GUARDAR MÉTRICAS DETALLADAS
# =============================================================================
def guardar_metricas(y_test, y_pred, clases, nombre_config, mapping=None):
    reporte = classification_report(y_test, y_pred, labels=clases, output_dict=True, zero_division=0)
    df_reporte = pd.DataFrame(reporte).transpose()
    
    # Añadir código anónimo de forma segura (mapeando el índice)
    if mapping:
        df_reporte['Codigo_Anonimo'] = df_reporte.index.map(lambda x: mapping.get(x, ''))
    else:
        df_reporte['Codigo_Anonimo'] = ''
    
    csv_reporte = os.path.join(OUTPUT_DIR, f'Reporte_{nombre_config}.csv')
    df_reporte.to_csv(csv_reporte)
    print(f"   ✅ Reporte guardado: {csv_reporte}")
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    metricas_globales = {
        'configuracion': nombre_config,
        'num_clases': len(clases),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted
    }
    return metricas_globales

# =============================================================================
#  GUARDAR PREDICCIONES POR MUESTRA
# =============================================================================
def guardar_predicciones(test_indices, y_test, y_pred, probas, mapping, nombre_config):
    real_cod = [mapping[valor] for valor in y_test]
    pred_cod = [mapping[valor] for valor in y_pred]
    acierto = (y_test == y_pred).astype(int)
    confianza = np.max(probas, axis=1)
    
    df_pred = pd.DataFrame({
        'Muestra_ID': test_indices,
        'Real': real_cod,
        'Predicho': pred_cod,
        'Acierto': acierto,
        'Confianza': confianza
    })
    csv_path = os.path.join(OUTPUT_DIR, f'Predicciones_{nombre_config}.csv')
    df_pred.to_csv(csv_path, index=False)
    print(f"   ✅ Predicciones guardadas: {csv_path}")
    return df_pred

# =============================================================================
#  ANONIMIZACIÓN (P1, P2, ...)
# =============================================================================
def crear_mapping_anonimo(clases_originales):
    mapping = {clase: f"P{i+1}" for i, clase in enumerate(sorted(clases_originales))}
    print("\n🔐 CODIFICACIÓN ANÓNIMA:")
    for original, codigo in mapping.items():
        print(f"   👤 {codigo} = {original}")
    
    with open(os.path.join(OUTPUT_DIR, "Leyenda_Codigos.txt"), "a") as f:
        f.write(f"\n--- Configuración: {len(clases_originales)} sujetos ---\n")
        for original, codigo in mapping.items():
            f.write(f"{codigo} = {original}\n")
    return mapping

# =============================================================================
#  ANÁLISIS PARA N SUJETOS
# =============================================================================
def analizar_con_n_sujetos(df_all, sujetos_disponibles, num_sujetos, nombre_config):
    print(f"\n{'='*70}")
    print(f"🔍 ANALIZANDO CON {nombre_config.upper()}")
    print(f"{'='*70}")
    
    df_filtrado, sujetos_seleccionados = seleccionar_sujetos(df_all, num_sujetos, sujetos_disponibles)
    print(f"✅ Sujetos seleccionados: {sujetos_seleccionados}")
    print(f"✅ Muestras totales: {len(df_filtrado)}")
    
    mapping = crear_mapping_anonimo(sujetos_seleccionados)
    
    X_train, X_test, y_train, y_test, scaler, features, test_indices = preparar_datos(df_filtrado)
    print(f"✅ Train: {X_train.shape[0]} muestras | Test: {X_test.shape[0]} muestras")
    
    print("🔄 Entrenando Random Forest...")
    rf_model = entrenar_random_forest(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)
    
    clases_ordenadas = sorted(y_train.unique())
    metricas = guardar_metricas(y_test, y_pred, clases_ordenadas, nombre_config, mapping)
    acc = metricas['accuracy']
    
    # Guardar predicciones
    df_pred = guardar_predicciones(test_indices, y_test, y_pred, y_proba, mapping, nombre_config)
    
    # Nombre legible para el título
    titulo_legible = nombre_legible(nombre_config)
    
    # ---- MATRICES NORMALIZADAS ----
    # Real (con nombres reales, mostrando códigos)
    generar_matriz_normalizada(
        y_test, y_pred, clases_ordenadas,
        f'Matriz de confusión normalizada - {titulo_legible} (acierto: {acc*100:.2f}%)',
        f'Matriz_{nombre_config}_normalizada.png',
        mapping=mapping
    )
    
    # Anónima (directamente con códigos)
    y_test_anon = y_test.map(mapping)
    y_pred_anon = pd.Series(y_pred).map(mapping)
    clases_anon = [mapping[clase] for clase in clases_ordenadas]
    generar_matriz_normalizada(
        y_test_anon, y_pred_anon, clases_anon,
        f'Matriz de confusión normalizada anónima - {titulo_legible}',
        f'Matriz_{nombre_config}_ANON_normalizada.png'
    )
    
    # Información adicional
    with open(os.path.join(OUTPUT_DIR, f'INFO_{nombre_config}.txt'), 'w', encoding='utf-8') as f:
        f.write(f"CONFIGURACIÓN: {titulo_legible}\n")
        f.write(f"Sujetos: {sujetos_seleccionados}\n")
        f.write(f"Número de sujetos: {len(sujetos_seleccionados)}\n")
        f.write(f"Muestras totales: {len(df_filtrado)}\n")
        f.write(f"Accuracy: {acc*100:.2f}%\n")
        f.write(f"\nHiperparámetros:\n")
        for k, v in RF_BEST_PARAMS.items():
            f.write(f"  {k}: {v}\n")
    
    return {
        'config': nombre_config,
        'num_sujetos': len(sujetos_seleccionados),
        'metricas': metricas,
        'clases': clases_ordenadas,
        'y_test': y_test,
        'y_pred': y_pred,
        'df_pred': df_pred
    }

# =============================================================================
#  COMPARATIVA FINAL (MATRICES NORMALIZADAS)
# =============================================================================
def generar_comparativa_final(resultados):
    if len(resultados) != 3:
        print("⚠️ No hay 3 resultados para la comparativa")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    colores = ['Blues', 'Greens', 'Oranges']
    
    for i, (config, res) in enumerate(resultados.items()):
        y_test = res['y_test']
        y_pred = res['y_pred']
        clases = res['clases']
        cm = confusion_matrix(y_test, y_pred, labels=clases, normalize='true')
        acc = res['metricas']['accuracy']
        
        # Nombre legible para el título
        titulo_legible = nombre_legible(config)
        
        sns.heatmap(cm, annot=True, fmt='.2f', cmap=colores[i], ax=axes[i],
                    xticklabels=clases, yticklabels=clases,
                    annot_kws={'size': 8}, cbar=False)
        
        axes[i].set_title(f'{titulo_legible}\nAccuracy: {acc*100:.1f}%', fontsize=14, weight='bold')
        axes[i].set_xlabel('Predicción', fontsize=10)
        axes[i].set_ylabel('Real', fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.suptitle('COMPARATIVA DE MATRICES NORMALIZADAS - RANDOM FOREST', fontsize=16, weight='bold', y=1.02)
    plt.tight_layout()
    
    output_fig = os.path.join(OUTPUT_DIR, 'COMPARATIVA_FINAL_MATRICES_NORMALIZADAS.png')
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Comparativa guardada: {output_fig}")
    
    # CSV con métricas globales
    df_global = pd.DataFrame([res['metricas'] for res in resultados.values()])
    csv_global = os.path.join(OUTPUT_DIR, 'COMPARATIVA_METRICAS_GLOBALES.csv')
    df_global.to_csv(csv_global, index=False)
    print(f"✅ CSV comparativo guardado: {csv_global}")
    
    # CSV con todas las predicciones
    df_pred_all = pd.concat([res['df_pred'].assign(Configuracion=nombre_legible(res['config'])) for res in resultados.values()], ignore_index=True)
    csv_pred_all = os.path.join(OUTPUT_DIR, 'TODAS_PREDICCIONES.csv')
    df_pred_all.to_csv(csv_pred_all, index=False)
    print(f"✅ CSV de todas las predicciones guardado: {csv_pred_all}")

# =============================================================================
#  PROGRAMA PRINCIPAL
# =============================================================================
def main():
    try:
        print("\n" + "="*70)
        print("🎯 RANDOM FOREST - MATRICES NORMALIZADAS")
        print("   (Hiperparámetros optimizados)")
        print("="*70)
        print(f"📂 Resultados en: {OUTPUT_DIR}")
        print("\n⚙️  Hiperparámetros:")
        for k, v in RF_BEST_PARAMS.items():
            print(f"   {k}: {v}")
        
        df_all, sujetos = cargar_todos_los_datos()
        if df_all is None:
            return
        
        print(f"\n📊 Total de sujetos válidos: {len(sujetos)}")
        
        configs = [
            (5, "5_sujetos"),
            (10, "10_sujetos"),
            (len(sujetos), "todos_los_sujetos")
        ]
        
        np.random.seed(42)
        resultados = {}
        
        for num, nombre in configs:
            try:
                res = analizar_con_n_sujetos(df_all, sujetos, num, nombre)
                resultados[nombre] = res
            except KeyboardInterrupt:
                print("\n⚠️  Interrupción. Finalizando...")
                break
            except Exception as e:
                print(f"❌ Error en {nombre}: {e}")
                import traceback
                traceback.print_exc()
        
        if len(resultados) == 3:
            generar_comparativa_final(resultados)
        elif resultados:
            print("\n⚠️  No se completaron las 3 configuraciones. Resultados parciales guardados.")
        
        print(f"\n✅ Proceso finalizado. Resultados en: {OUTPUT_DIR}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Proceso interrumpido.")

if __name__ == "__main__":
    main()
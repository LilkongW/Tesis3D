import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
# Importar toolkit 3D
from mpl_toolkits.mplot3d import Axes3D 
import seaborn as sns
import os
import glob
import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, silhouette_score
import warnings

# Intentar configurar el backend interactivo para los gráficos 3D
try:
    import matplotlib
    matplotlib.use('TkAgg') 
except:
    pass 

warnings.filterwarnings('ignore')

# =============================================================================
#  CONFIGURACIÓN Y RUTAS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(BASE_DIR, "Analizar_Data", "Resultados", "**", "*_BIOMETRIC_METRICS.csv")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_DIR, "Analisis_Avanzados", "Reportes_Finales_Interactivos", f"Analisis_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Estilo visual
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# =============================================================================
#  HELPER: ELIPSES DE CONFIANZA (Solo para 2D)
# =============================================================================
def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# =============================================================================
#  1. CARGA DE DATOS
# =============================================================================
def cargar_dataset():
    print("="*70)
    print("🚀 ANÁLISIS BIOMÉTRICO: MODO EXPLORACIÓN 3D & RADARES COMPLETOS")
    print("="*70)
    
    files = glob.glob(INPUT_PATH, recursive=True)
    if not files: return None
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except: pass
            
    if not dfs: return None
    df_final = pd.concat(dfs, ignore_index=True)
    df_final = df_final.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_final = df_final[df_final['Pupil_Mean'] > 0]
    
    print(f"✅ Dataset cargado: {len(df_final)} muestras.")
    return df_final

# =============================================================================
#  2A. MAPAS 2D (EL CLÁSICO ROBUSTO)
# =============================================================================
def generar_mapa_lda_2d(df_in, filename_suffix, title_suffix):
    X = df_in.drop(columns=['Participant', 'VideoID', 'Window_Start'], errors='ignore')
    y = df_in['Participant']
    participants = y.unique()
    n_classes = len(participants)
    
    if n_classes < 2: return

    # LDA 2D
    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(X, y)
    var = lda.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    sns.scatterplot(x=X_lda[:,0], y=X_lda[:,1], hue=y, style=y, 
                    palette='tab10', s=100, alpha=0.8, edgecolor='white', ax=ax)
    
    for idx, p in enumerate(participants):
        mask = (y == p)
        if np.sum(mask) > 5:
            x_p = X_lda[mask, 0]
            y_p = X_lda[mask, 1]
            confidence_ellipse(x_p, y_p, ax, n_std=2.0, edgecolor=colors[idx % len(colors)], linestyle='--', linewidth=2)
            ax.text(np.mean(x_p), np.mean(y_p), p, fontsize=10, weight='bold', 
                    color='black', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.title(f'Clusterización LDA 2D - {title_suffix}', fontsize=16, fontweight='bold')
    plt.xlabel(f'LD1 ({var[0]*100:.1f}%)')
    plt.ylabel(f'LD2 ({var[1]*100:.1f}%)')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Mapa_2D_{filename_suffix}.png"), dpi=300)
    plt.close()

# =============================================================================
#  2B. MAPAS 3D INTERACTIVOS
# =============================================================================
def generar_mapa_lda_3d_interactivo(df_in, filename_suffix, title_suffix):
    X = df_in.drop(columns=['Participant', 'VideoID', 'Window_Start'], errors='ignore')
    y = df_in['Participant']
    participants = y.unique()
    n_classes = len(participants)

    if n_classes < 4: 
        print(f"⚠️ Saltando 3D para {title_suffix} (se necesitan mín 4 participantes).")
        return

    print(f"🧊 Abriendo Espacio 3D: {title_suffix}...")
    print("   👉 Usa el mouse para ROTAR y hacer ZOOM.")
    
    # LDA 3D
    lda = LDA(n_components=3)
    X_lda = lda.fit_transform(X, y)
    var = lda.explained_variance_ratio_

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    label_encoder = {name: i for i, name in enumerate(participants)}
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for p in participants:
        mask = (y == p)
        idx = label_encoder[p]
        ax.scatter(X_lda[mask, 0], X_lda[mask, 1], X_lda[mask, 2], 
                   label=p, s=50, alpha=0.8, color=colors[idx % len(colors)], edgecolors='w')
        
        center = np.mean(X_lda[mask], axis=0)
        ax.text(center[0], center[1], center[2], p, fontsize=9, weight='bold', color='black')

    ax.set_xlabel(f'LD1 ({var[0]*100:.1f}%)')
    ax.set_ylabel(f'LD2 ({var[1]*100:.1f}%)')
    ax.set_zlabel(f'LD3 ({var[2]*100:.1f}%)')
    ax.set_title(f'Espacio Biométrico 3D - {title_suffix}\nVarianza Total: {sum(var[:3])*100:.1f}%', fontsize=14)
    ax.legend(bbox_to_anchor=(1.1, 0.9))

    plt.savefig(os.path.join(OUTPUT_DIR, f"Captura_3D_{filename_suffix}.png"), dpi=300, bbox_inches='tight')
    plt.show() 

def ejecutar_lda_escalonado(df):
    print("\n" + "-"*30 + " MAPAS LDA (2D y 3D) " + "-"*30)
    participants = df['Participant'].unique()
    
    # 5 Personas
    if len(participants) >= 5:
        df_5 = df[df['Participant'].isin(participants[:5])]
        generar_mapa_lda_2d(df_5, "5_Personas", "5 Participantes")
        generar_mapa_lda_3d_interactivo(df_5, "5_Personas", "5 Participantes")
        
    # 10 Personas
    if len(participants) >= 10:
        df_10 = df[df['Participant'].isin(participants[:10])]
        generar_mapa_lda_2d(df_10, "10_Personas", "10 Participantes")
        generar_mapa_lda_3d_interactivo(df_10, "10_Personas", "10 Participantes")
    
    # Todos
    generar_mapa_lda_2d(df, "Completo", "Dataset Completo")
    generar_mapa_lda_3d_interactivo(df, "Completo", "Dataset Completo")

# =============================================================================
#  3. CLASIFICACIÓN ESCALONADA
# =============================================================================
def generar_matriz_svm(df_in, filename_suffix, title_suffix):
    print(f"🤖 Entrenando SVM: {title_suffix}")
    X = df_in.drop(columns=['Participant', 'VideoID', 'Window_Start'], errors='ignore')
    y = df_in['Participant']
    
    if len(y) < 20: return 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    scaler = StandardScaler()
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(scaler.fit_transform(X_train), y_train)
    
    y_pred = clf.predict(scaler.transform(X_test))
    acc = accuracy_score(y_test, y_pred)
    print(f"   🎯 Accuracy: {acc*100:.2f}%")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title(f'Matriz de Confusión - {title_suffix}\nAcc: {acc*100:.1f}%', fontsize=14, fontweight='bold')
    plt.ylabel('Realidad')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Matriz_Confusion_{filename_suffix}.png"), dpi=300)
    plt.close()
    
    pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().to_csv(
        os.path.join(OUTPUT_DIR, f"Reporte_{filename_suffix}.csv"))

def ejecutar_svm_escalonado(df):
    print("\n" + "-"*30 + " MATRICES DE CONFUSIÓN " + "-"*30)
    participants = df['Participant'].unique()
    
    if len(participants) >= 5:
        generar_matriz_svm(df[df['Participant'].isin(participants[:5])], "5_Personas", "5 Participantes")
    if len(participants) >= 10:
        generar_matriz_svm(df[df['Participant'].isin(participants[:10])], "10_Personas", "10 Participantes")
    generar_matriz_svm(df, "Completo", "Dataset Completo")

# =============================================================================
#  4. RADARES (12 MÉTRICAS)
# =============================================================================
def generar_fichas_radar(df):
    print("\n" + "-"*30 + " GENERANDO PERFILES (12 MÉTRICAS) " + "-"*30)
    participants = df['Participant'].unique()
    
    # --- SELECCIÓN DE LAS 12 MÉTRICAS MÁS IMPORTANTES ---
    # Organizadas por categorías para facilitar la lectura del gráfico
    metrics_radar = [
        # --- Pupila & Fisiología ---
        'Pupil_Mean', 
        'Pupil_Vel_Max', 
        'Pupil_Std',
        
        # --- Control Motor (Involuntario) ---
        'Microsaccade_Rate', 
        'Velocity_Transition_Smoothness', 
        'Jerk_Mean',
        
        # --- Dinámica de Movimiento (Biomecánica) ---
        'Main_Seq_Slope', 
        'Vel_Mean', 
        'Acc_Max',
        
        # --- Estrategia & Cognición ---
        'Fractal_Dim', 
        'Spatial_Entropy', 
        'Fixation_Vel_Mean'
    ]
    
    # Validar que existan en el DF
    metrics_radar = [m for m in metrics_radar if m in df.columns]
    
    if len(metrics_radar) < 3: 
        print("⚠️ No hay suficientes métricas para el radar.")
        return

    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[metrics_radar] = scaler.fit_transform(df[metrics_radar])
    global_mean = df_scaled[metrics_radar].mean()
    
    categories = metrics_radar
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(participants)))
    
    for idx, p in enumerate(participants):
        p_data = df_scaled[df_scaled['Participant'] == p][metrics_radar].mean()
        values = p_data.values.flatten().tolist()
        values += values[:1]
        g_values = global_mean.values.flatten().tolist()
        g_values += g_values[:1]
        
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        
        # Fondo (Promedio Global)
        ax.plot(angles, g_values, linewidth=1, linestyle='--', color='gray', alpha=0.5, label='Promedio Global')
        ax.fill(angles, g_values, color='gray', alpha=0.1)
        
        # Perfil del Usuario
        color_line = colors[idx % len(colors)]
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', color=color_line, label=p)
        ax.fill(angles, values, color=color_line, alpha=0.35)
        
        # Puntos destacados
        ax.scatter(angles, values, s=50, c=color_line, edgecolors='white', zorder=10)
        
        # Etiquetas
        plt.xticks(angles[:-1], categories, size=8, weight='bold', color='#333333')
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=7)
        plt.ylim(0, 1)
        
        plt.title(f"Huella Biométrica: {p}", size=18, weight='bold', color=color_line, y=1.08)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"Perfil_{p}.png"), dpi=300)
        plt.close()
        
    print(f"✅ Fichas generadas con {len(metrics_radar)} métricas.")

# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    df = cargar_dataset()
    if df is not None:
        # 1. Mapas LDA (2D y 3D Interactivo)
        ejecutar_lda_escalonado(df) 
        
        # 2. Matrices SVM
        ejecutar_svm_escalonado(df)
        
        # 3. Radares (12 Métricas)
        generar_fichas_radar(df)
        
        print("\n" + "="*70)
        print("✅ ANÁLISIS COMPLETADO")
        print(f"📂 Archivos en: {OUTPUT_DIR}")
        print("="*70)
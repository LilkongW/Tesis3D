import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_FILE = r"c:\Users\Victor\Documents\Tesis3D\Komogortsev\dataset_komogortsev.csv"

def main():
    if not os.path.exists(DATASET_FILE):
        print(f"No se encuentra el archivo {DATASET_FILE}. Ejecuta Procesar_Dataset.py primero.")
        return
        
    df = pd.read_csv(DATASET_FILE)
    df = df[df['participante'] != 'test']
    
    feature_cols = [c for c in df.columns if c not in ['participante', 'video']]
    X_full = df[feature_cols].values
    y = df['participante'].values
    
    participant_classes = np.unique(y)
    print(f"Total de muestras: {len(y)}")
    print(f"Total de sujetos: {len(participant_classes)}")
    print(f"Features en uso ({len(feature_cols)}): {', '.join(feature_cols)}\n")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        "Random Forest (1000 Arboles)": RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced'),
        "SVM (Kernel RBF - Trick)": SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, class_weight='balanced')
    }
    
    best_acc = 0.0
    best_name = ""
    
    print("--- Evaluando Modelos (StratifiedKFold) ---")
    
    for name, model in models.items():
        print(f"\n--- Evaluando {name} ---")
        y_pred = cross_val_predict(model, X_scaled, y, cv=cv)
        acc = accuracy_score(y, y_pred)
        
        print(f"PRECISION: {acc*100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_name = name
            
        # Generar matriz de confusión para cada modelo
        cm = confusion_matrix(y, y_pred, labels=participant_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=participant_classes, yticklabels=participant_classes, cmap="Blues")
        plt.title(f"Matriz de Confusion - {name} ({acc*100:.1f}%)")
        plt.xlabel("Prediccion")
        plt.ylabel("Real")
        
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
        output_img = rf"c:\Users\Victor\Documents\Tesis3D\Komogortsev\Matriz_Confusion_{safe_name}.png"
        plt.tight_layout()
        plt.savefig(output_img)
        plt.close()
        print(f"Matriz de confusion guardada en: {output_img}")
        
    print(f"\n{'*'*60}")
    print(f"  MEJOR RESULTADO GLOBAL: {best_acc*100:.2f}% ({best_name})")
    print(f"{'*'*60}")

if __name__ == "__main__":
    main()

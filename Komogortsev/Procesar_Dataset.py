import os
import glob
import pandas as pd
from Extractor_Komogortsev import extract_cemb_features

DATA_DIR = r"c:\Users\Victor\Documents\Tesis3D\Data\Experimento_1"
OUTPUT_FILE = r"c:\Users\Victor\Documents\Tesis3D\Komogortsev\dataset_komogortsev.csv"

def main():
    print("Iniciando procesamiento de CEM-B (Komogortsev)...")
    all_features = []
    
    # Recorrer todas las carpetas de participantes
    participant_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    for p_dir in participant_dirs:
        participant_name = p_dir.replace("_data", "")
        folder_path = os.path.join(DATA_DIR, p_dir)
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        for csv_file in csv_files:
            video_name = os.path.basename(csv_file).replace("_data.csv", "")
            
            try:
                df = pd.read_csv(csv_file)
                features = extract_cemb_features(df, velocity_threshold=30.0)
                
                if features is not None:
                    # Añadir metadatos
                    features['participante'] = participant_name
                    features['video'] = video_name
                    all_features.append(features)
                    
            except Exception as e:
                print(f"Error procesando {csv_file}: {e}")
                
    if len(all_features) > 0:
        final_df = pd.DataFrame(all_features)
        
        # Reordenar columnas para tener participante y video al inicio
        cols = ['participante', 'video'] + [c for c in final_df.columns if c not in ['participante', 'video']]
        final_df = final_df[cols]
        
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Dataset guardado exitosamente en: {OUTPUT_FILE}")
        print(f"Total de intentos procesados: {len(final_df)}")
    else:
        print("No se extrajeron métricas de ningún archivo.")

if __name__ == "__main__":
    main()

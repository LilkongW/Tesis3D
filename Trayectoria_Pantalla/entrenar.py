import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import glob
from datetime import datetime

class GazeModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.history = None
        self.screen_width = None
        self.screen_height = None
        
    def load_calibration_data(self, pkl_file_path):
        """Carga datos de calibración desde archivo .pkl"""
        print("\n=== Cargando datos de calibración ===")
        print(f"Archivo: {pkl_file_path}")
        
        with open(pkl_file_path, 'rb') as f:
            data_package = pickle.load(f)
        
        calibration_data = data_package['calibration_data']
        self.screen_width = data_package['screen_width']
        self.screen_height = data_package['screen_height']
        
        # Extraer vectores de mirada (X) y posiciones en pantalla (y)
        X = []
        y = []
        
        for sample in calibration_data:
            gaze_vector = sample['gaze_vector']
            screen_pos = sample['screen_pos']
            
            X.append(gaze_vector)
            y.append(screen_pos)
        
        X = np.array(X)
        y = np.array(y)
        
        print("✓ Datos cargados:")
        print(f"  - Total de muestras: {len(X)}")
        print(f"  - Forma de X (vectores de mirada): {X.shape}")
        print(f"  - Forma de y (posiciones en pantalla): {y.shape}")
        print(f"  - Resolución de pantalla: {self.screen_width}x{self.screen_height}")
        
        # Mostrar estadísticas
        print("\nEstadísticas de vectores de mirada:")
        print(f"  Gaze X: min={X[:,0].min():.3f}, max={X[:,0].max():.3f}, mean={X[:,0].mean():.3f}")
        print(f"  Gaze Y: min={X[:,1].min():.3f}, max={X[:,1].max():.3f}, mean={X[:,1].mean():.3f}")
        print(f"  Gaze Z: min={X[:,2].min():.3f}, max={X[:,2].max():.3f}, mean={X[:,2].mean():.3f}")
        
        print("\nEstadísticas de posiciones en pantalla:")
        print(f"  Screen X: min={y[:,0].min():.1f}, max={y[:,0].max():.1f}, mean={y[:,0].mean():.1f}")
        print(f"  Screen Y: min={y[:,1].min():.1f}, max={y[:,1].max():.1f}, mean={y[:,1].mean():.1f}")
        
        return X, y
    
    def load_multiple_calibrations(self, pkl_files):
        """Carga múltiples archivos de calibración para más datos"""
        print("\n=== Cargando múltiples calibraciones ===")
        
        X_all = []
        y_all = []
        
        for pkl_file in pkl_files:
            X, y = self.load_calibration_data(pkl_file)
            X_all.append(X)
            y_all.append(y)
        
        X_combined = np.vstack(X_all)
        y_combined = np.vstack(y_all)
        
        print("\n✓ Total combinado:")
        print(f"  - Muestras totales: {len(X_combined)}")
        print(f"  - De {len(pkl_files)} sesiones de calibración")
        
        return X_combined, y_combined
    
    def prepare_data(self, X, y, test_size=0.2, val_size=0.1):
        """Prepara los datos: normalización y división en conjuntos"""
        print("\n=== Preparando datos ===")
        
        # División inicial: train+val / test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # División: train / val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, random_state=42
        )
        
        # Normalizar features (vectores de mirada)
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # Normalizar targets (posiciones en pantalla)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_val_scaled = self.scaler_y.transform(y_val)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        print("✓ División de datos:")
        print(f"  - Entrenamiento: {len(X_train)} muestras ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  - Validación: {len(X_val)} muestras ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  - Prueba: {len(X_test)} muestras ({len(X_test)/len(X)*100:.1f}%)")
        
        return (X_train_scaled, y_train_scaled), (X_val_scaled, y_val_scaled), (X_test_scaled, y_test_scaled)
    
    def build_model(self, input_dim=3, output_dim=2):
        """Construye la red neuronal"""
        print("\n=== Construyendo modelo ===")
        
        model = keras.Sequential([
            # Capa de entrada
            layers.Input(shape=(input_dim,)),
            
            # Capas ocultas con Dropout para regularización
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            
            # Capa de salida
            layers.Dense(output_dim, activation='linear')
        ])
        
        # Compilar modelo
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print("✓ Arquitectura del modelo:")
        model.summary()
        
        self.model = model
        return model
    
    def train(self, train_data, val_data, epochs=200, batch_size=32):
        """Entrena el modelo"""
        print("\n=== Entrenando modelo ===")
        print(f"Épocas: {epochs}, Batch size: {batch_size}")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            verbose=1
        )
        
        # Entrenar
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("\n✓ Entrenamiento completado")
        
    def evaluate(self, test_data):
        """Evalúa el modelo en el conjunto de prueba"""
        print("\n=== Evaluando modelo ===")
        
        X_test, y_test = test_data
        
        # Predicciones escaladas
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        
        # Desescalar predicciones y targets
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_test_original = self.scaler_y.inverse_transform(y_test)
        
        # Calcular métricas
        mse = mean_squared_error(y_test_original, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, y_pred)
        
        # Errores por eje
        mae_x = mean_absolute_error(y_test_original[:, 0], y_pred[:, 0])
        mae_y = mean_absolute_error(y_test_original[:, 1], y_pred[:, 1])
        
        print("✓ Métricas de evaluación:")
        print(f"  - RMSE (píxeles): {rmse:.2f}")
        print(f"  - MAE (píxeles): {mae:.2f}")
        print(f"  - MAE en X: {mae_x:.2f} píxeles")
        print(f"  - MAE en Y: {mae_y:.2f} píxeles")
        
        # Calcular precisión en términos de pantalla
        if self.screen_width and self.screen_height:
            mae_percent_x = (mae_x / self.screen_width) * 100
            mae_percent_y = (mae_y / self.screen_height) * 100
            print(f"  - Error relativo X: {mae_percent_x:.2f}% de la pantalla")
            print(f"  - Error relativo Y: {mae_percent_y:.2f}% de la pantalla")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mae_x': mae_x,
            'mae_y': mae_y,
            'predictions': y_pred,
            'actual': y_test_original
        }
    
    def plot_training_history(self):
        """Grafica el historial de entrenamiento"""
        if self.history is None:
            print("No hay historial de entrenamiento disponible")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Train Loss')
        ax1.plot(self.history.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Pérdida durante el Entrenamiento')
        ax1.legend()
        ax1.grid(True)
        
        # MAE
        ax2.plot(self.history.history['mae'], label='Train MAE')
        ax2.plot(self.history.history['val_mae'], label='Val MAE')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('MAE')
        ax2.set_title('Error Absoluto Medio durante el Entrenamiento')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        print("\n✓ Gráfica guardada: training_history.png")
        plt.show()
    
    def plot_predictions(self, eval_results):
        """Grafica las predicciones vs valores reales"""
        predictions = eval_results['predictions']
        actual = eval_results['actual']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot X
        ax1.scatter(actual[:, 0], predictions[:, 0], alpha=0.5)
        ax1.plot([actual[:, 0].min(), actual[:, 0].max()],
                [actual[:, 0].min(), actual[:, 0].max()], 'r--', lw=2)
        ax1.set_xlabel('Posición Real X (píxeles)')
        ax1.set_ylabel('Posición Predicha X (píxeles)')
        ax1.set_title(f'Predicciones X (MAE: {eval_results["mae_x"]:.2f} px)')
        ax1.grid(True)
        
        # Scatter plot Y
        ax2.scatter(actual[:, 1], predictions[:, 1], alpha=0.5)
        ax2.plot([actual[:, 1].min(), actual[:, 1].max()],
                [actual[:, 1].min(), actual[:, 1].max()], 'r--', lw=2)
        ax2.set_xlabel('Posición Real Y (píxeles)')
        ax2.set_ylabel('Posición Predicha Y (píxeles)')
        ax2.set_title(f'Predicciones Y (MAE: {eval_results["mae_y"]:.2f} px)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('predictions_scatter.png', dpi=150)
        print("✓ Gráfica guardada: predictions_scatter.png")
        plt.show()
    
    def save_model(self, model_name='gaze_model'):
        """Guarda el modelo y los scalers"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear carpeta si no existe
        os.makedirs('models', exist_ok=True)
        
        # Guardar modelo Keras
        model_path = f'/home/vit/Documentos/Tesis3D/Trayectoria_Pantalla/models/{model_name}_{timestamp}.h5'
        self.model.save(model_path)
        
        # Guardar scalers y metadata
        metadata = {
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'timestamp': timestamp
        }
        
        metadata_path = f'models/{model_name}_{timestamp}_metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print("\n✓ Modelo guardado:")
        print(f"  - Modelo: {model_path}")
        print(f"  - Metadata: {metadata_path}")
        
        return model_path, metadata_path


def main():
    print("=" * 60)
    print("  ENTRENAMIENTO DE MODELO DE EYE TRACKING")
    print("=" * 60)
    
    # Inicializar trainer
    trainer = GazeModelTrainer()
    
    # Definir la ruta de la carpeta
    base_path = "/home/vit/Documentos/Tesis3D/Trayectoria_Pantalla/calibracion_data"

    # Usar glob para encontrar todos los archivos .pkl en esa carpeta
    calibration_files = glob.glob(os.path.join(base_path, "*.pkl"))
    
    if not calibration_files:
        print("\n✗ Error: No se encontraron archivos de calibración (.pkl)")
        print("  Ejecuta primero el script de calibración para generar datos")
        return
    
    print(f"\n✓ Se encontraron {len(calibration_files)} archivo(s) de calibración:")
    for i, file in enumerate(calibration_files, 1):
        print(f"  {i}. {file}")
    
    # Cargar datos
    if len(calibration_files) == 1:
        X, y = trainer.load_calibration_data(calibration_files[0])
    else:
        print("\n¿Usar todos los archivos? (s/n): ", end='')
        use_all = input().lower().strip()
        if use_all == 's':
            X, y = trainer.load_multiple_calibrations(calibration_files)
        else:
            X, y = trainer.load_calibration_data(calibration_files[0])
    
    # Verificar que tengamos suficientes datos
    if len(X) < 100:
        print(f"\n⚠ Advertencia: Solo hay {len(X)} muestras. Se recomienda al menos 500.")
        print("  Considera ejecutar más sesiones de calibración.")
    
    # Preparar datos
    train_data, val_data, test_data = trainer.prepare_data(X, y)
    
    # Construir modelo
    trainer.build_model()
    
    # Entrenar
    print("\nPresiona Enter para iniciar el entrenamiento...")
    input()
    
    trainer.train(train_data, val_data, epochs=300, batch_size=32)
    
    # Evaluar
    eval_results = trainer.evaluate(test_data)
    
    # Graficar
    trainer.plot_training_history()
    trainer.plot_predictions(eval_results)
    
    # Guardar modelo
    print("\n¿Guardar el modelo? (s/n): ", end='')
    save = input().lower().strip()
    if save == 's':
        trainer.save_model()
        print("\n✓ Modelo guardado exitosamente")
        print("\nAhora puedes usar el script de prueba para probar el modelo en tiempo real")
    
    print("\n" + "=" * 60)
    print("  ENTRENAMIENTO COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()
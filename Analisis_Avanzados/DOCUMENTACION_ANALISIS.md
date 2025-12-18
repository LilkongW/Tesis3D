# Documentación de Análisis Avanzado de Métricas

El script `Analizar_Metricas.py` es el módulo encargado del procesamiento estadístico avanzado, aprendizaje automático (Machine Learning) y visualización de los datos biométricos generados previamente. Su objetivo es encontrar patrones que distingan a los usuarios y generar reportes visuales interpretables.

## Flujo de Trabajo

### 1. Carga y Preprocesamiento de Datos
*   **Fuente**: Carga todos los archivos `*_BIOMETRIC_METRICS.csv` generados por el script de métricas.
*   **Limpieza**: Consolida múltiples participantes en un único DataFrame, maneja valores nulos e infinitos, y filtra datos sin información pupilar válida.
*   **Traducción**: Utiliza un diccionario interno para traducir los nombres técnicos de las métricas (ej. `Main_Seq_Slope`) a términos legibles en español (ej. `Pendiente Secuencia Princ.`) para los gráficos finales.

### 2. Selección de Características (Feature Importance)
Utiliza un algoritmo de **Random Forest** para determinar qué métricas son las más relevantes para distinguir entre los diferentes participantes.
*   **Salida**: Genera un gráfico de barras (`Ranking_Feature_Importance.png`) con el Top 15 de métricas más discriminantes.
*   **Ranking**: Guarda un archivo CSV con el peso específico de cada variable biométrica en la clasificación.

### 3. Visualización de Espacios Biométricos (LDA)
Aplica **Análisis Discriminante Lineal (LDA)** para reducir la dimensionalidad de los datos y proyectarlos en mapas visuales.
*   **Mapas 2D**: Proyecciones en 2 dimensiones con elipses de confianza (2 desviaciones estándar) para visualizar la separabilidad de los clusters de usuarios.
*   **Mapas 3D**: Genera visualizaciones tridimensionales rotables (si el backend lo permite) o capturas estáticas (`Captura_3D_*.png`) para entender mejor la distribución espacial de los perfiles biométricos.
*   **Estrategia Escalonada**: Genera gráficos tanto para subgrupos (ej. 5 participantes) como para el dataset completo, facilitando la lectura cuando hay muchos usuarios.

### 4. Clasificación y Validación (SVM)
Entrena un modelo de clasificación **Support Vector Machine (SVM)** (kernel RBF) para validar matemáticamente qué tan distinguibles son los patrones biométricos.
*   **Validación**: Divide los datos en entrenamiento (70%) y prueba (30%).
*   **Reportes**:
    *   **Matriz de Confusión**: (`Matriz_Confusion_*.png`) Mapa de calor que muestra dónde se confunde el algoritmo (qué usuario se parece a cuál).
    *   **Reporte de Métricas**: (`Reporte_*.csv`) Precisión (Accuracy), Recall y F1-Score detallado por usuario.

### 5. Perfiles de Usuario (Gráficos de Radar)
Genera una "huella digital biométrica" visual para cada participante.
*   **Normalización**: Escala todas las métricas entre 0 y 1 para compararlas en un mismo gráfico.
*   **Comparativa**: Superpone el perfil individual del usuario contra el promedio global de todos los participantes.
*   **Métricas Dinámicas**: El gráfico de radar utiliza automáticamente las métricas identificadas como "más importantes" en el paso 2, asegurando que el gráfico muestre solo la información más relevante.

## Salida de Archivos
Todos los resultados se guardan en una carpeta con *timestamp* dentro de `Analisis_Avanzados/Reportes_Finales_Interactivos/`, permitiendo mantener un historial de análisis sin sobrescribir resultados previos.

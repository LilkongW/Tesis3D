# Documentación de Métricas Biométricas

Este documento describe detalladamente las métricas generadas por el pipeline de análisis biométrico (`Generar_Metricas.py`). Las métricas se calculan procesando ventanas de tiempo (por defecto 2.5s) de los datos de *eye-tracking*.

## 1. Análisis de Pupila (Sistema Nervioso Autónomo)
Estas métricas se derivan del diámetro pupilar y reflejan la actividad del Sistema Nervioso Autónomo (SNA), a menudo asociadas con carga cognitiva, excitación emocional (arousal) y esfuerzo mental.

| Métrica | Origen / Cálculo | Propósito e Interpretación |
| :--- | :--- | :--- |
| **Pupil_Mean** | Promedio del diámetro de la pupila en la ventana. | Línea base del estado fisiológico. Pupilas más dilatadas suelen indicar mayor carga cognitiva o interés emocional. |
| **Pupil_Std** | Desviación estándar del diámetro pupilar. | Variabilidad de la señal. Una alta variabilidad puede indicar inestabilidad en el estado de atención. |
| **Pupil_CV** | Coeficiente de Variación (`Std / Mean`). | Normaliza la variabilidad respecto al tamaño del ojo, permitiendo comparar entre sujetos con tamaños de pupila basales diferentes. |
| **Pupil_Vel_Max** | Máximo valor absoluto de la derivada del diámetro pupilar (`d(diameter)/dt`). | Detecta cambios rápidos (dilatación/constricción). Altos valores indican respuestas fásicas fuertes ante estímulos repentinos (sobresalto o captura atencional). |

## 2. Control Motor y Estabilidad
Evalúan la calidad y suavidad del movimiento ocular. Son útiles para detectar fatiga, control motor fino o alteraciones neurológicas.

| Métrica | Origen / Cálculo | Propósito e Interpretación |
| :--- | :--- | :--- |
| **Jerk_Mean** | Promedio del valor absoluto del *Jerk* (derivada de la aceleración). | Mide la "suavidad" del movimiento. Valores altos indican movimientos bruscos o espasmódicos (menos control motor). |
| **Jerk_Max** | Pico máximo de Jerk en la ventana. | Identifica el momento de mayor inestabilidad motora en el periodo analizado. |
| **Velocity_Transition_Smoothness** | Calculado a partir de la relación entre la desviación del Jerk y el rango de velocidades. Escala 0-1. | Cuantifica qué tan fluidas son las transiciones de velocidad. Valores cercanos a 1.0 indican movimientos extremadamente suaves y controlados. |
| **Microsaccade_Rate** | Frecuencia de picos de velocidad pequeños que no califican como sacadas completas (filtrado por umbrales y percentiles). | Tasa de movimientos microsacádicos por segundo. Relacionado con la supresión de la visión durante la fijación y procesos atencionales finos. |

## 3. Complejidad Cognitiva (Análisis Fractal)
Métricas avanzadas basadas en la teoría del caos y dimensiones fractales para evaluar la estructura de la señal.

| Métrica | Origen / Cálculo | Propósito e Interpretación |
| :--- | :--- | :--- |
| **Fractal_Dim** | **Dimensión Fractal de Higuchi (HFD)** aplicada a la serie de tiempo de velocidad ocular. `K_max=5`. | Mide la complejidad o "rugosidad" de la señal. <br>• **Bajo (~1.0)**: Movimiento simple, predecible.<br>• **Alto (>1.5)**: Comportamiento caótico/complejo. Puede indicar un estado de búsqueda visual activa o mayor procesamiento cognitivo. |

## 4. Eventos y Biomecánica (Sacadas y Fijaciones)
Clasificación clásica de eventos oculares enriquecida con parámetros biomecánicos.

| Métrica | Origen / Cálculo | Propósito e Interpretación |
| :--- | :--- | :--- |
| **Saccade_Rate** | Número de sacadas detectadas dividido por la duración de la ventana. | Tasa de exploración. Más sacadas por segundo indican una búsqueda visual activa y rápida. |
| **Main_Seq_Slope** | Pendiente de la regresión lineal entre **Velocidad Pico** (eje Y) y **Amplitud** (eje X) de las sacadas. | "Huella digital" biomecánica (Main Sequence). Relaciona qué tan rápido se mueve el ojo para una distancia dada. Desviaciones de la pendiente normal pueden indicar fatiga muscular o patologías. |
| **Fixation_Vel_Mean** | Promedio de la velocidad *durante* los periodos clasificados como fijaciones. | Estabilidad de la fijación. Debería ser cercana a 0. Valores altos indican derivas (*drift*) o dificultad para mantener la mirada fija. |

## 5. Cinemática Básica
Parámetros físicos directos del movimiento del vector de mirada.

| Métrica | Origen / Cálculo | Propósito e Interpretación |
| :--- | :--- | :--- |
| **Vel_Mean** | Promedio de la velocidad angular (grados/segundo). | Actividad global. Cuánto se está moviendo el ojo en promedio. |
| **Acc_Max** | Máxima aceleración alcanzada en la ventana. | Fuerza explosiva del movimiento ocular. |
| **Gaze_Z_Mean** | Promedio de la coordenada Z del vector de mirada. | Dependiendo de la calibración, indica profundidad o distancia promedio de enfoque. |

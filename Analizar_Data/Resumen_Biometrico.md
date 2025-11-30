-----

# Diccionario de Datos: Métricas Oculomotoras y Biométricas

Este documento describe las variables generadas en el archivo `Victor_Resumen_Biometrico.csv` para el análisis de patrones de comportamiento visual.

## 1\. Identificación y Globales

Información general sobre la sesión de grabación.

| Columna | Unidad | Descripción | Interpretación |
| :--- | :--- | :--- | :--- |
| **`archivo`** | Texto | Nombre del archivo CSV original procesado (ej. `intento_1.csv`). | Identificador único del intento. |
| **`duracion_total_s`** | Segundos (s) | Tiempo total que duró la grabación válida. | Útil para normalizar otras métricas. |
| **`distancia_total_deg`** | Grados (°) | **Odometría Ocular**: La suma de todos los movimientos angulares realizados. | Mide el "trabajo total" realizado por el ojo. Un valor muy alto puede indicar búsqueda ineficiente o ansiedad. |
| **`velocidad_promedio_global`** | °/s | Velocidad angular promedio de toda la sesión. | Indicador general de dinamismo. Disminuye con la sedación o fatiga extrema. |

-----

## 2\. Métricas de Fijación (Atención y Estabilidad)

Estas métricas analizan los momentos en que el ojo está "quieto" procesando información. Son clave para perfiles cognitivos.

| Columna | Unidad | Descripción | Interpretación |
| :--- | :--- | :--- | :--- |
| **`num_fijaciones`** | Conteo (\#) | Cantidad total de veces que el ojo se detuvo (\>250ms). | Mide la frecuencia de muestreo de información. |
| **`fijacion_duracion_media_ms`** | Milisegundos (ms) | Promedio de cuánto dura cada fijación. | **Clave Cognitiva:** <br>• **\<200ms:** Exploración rápida (Ambiental).<br>• **\>400ms:** Procesamiento profundo (Focal). |
| **`fijacion_duracion_std`** | ms | Desviación estándar de la duración. | ¿Es rítmico (bajo) o caótico (alto)? La irregularidad puede indicar distracción. |
| **`ratio_tiempo_fijacion`** | 0.0 - 1.0 | Porcentaje del tiempo total dedicado a fijar la mirada. | **Ciclo de Trabajo:** Un ratio bajo significa que el usuario pasó mucho tiempo "buscando" (moviéndose) y poco "viendo". |

-----

## 3\. Métricas de Sacadas (Dinámica Muscular)

Estas métricas analizan los movimientos rápidos (balísticos). Son las más importantes para la **Biometría Física** (identificación de la persona) y detección de **Fatiga**.

| Columna | Unidad | Descripción | Interpretación |
| :--- | :--- | :--- | :--- |
| **`num_sacadas`** | Conteo (\#) | Total de movimientos rápidos detectados. | Actividad motora. |
| **`tasa_sacadas_hz`** | Hz (Eventos/s) | Frecuencia de movimientos por segundo. | **Nivel de Activación:** <br>• **Alta (\>2Hz):** Alerta, ansiedad o búsqueda.<br>• **Baja (\<1Hz):** Somnolencia o aburrimiento. |
| **`sacada_vel_pico_media`** | °/s | Promedio de la velocidad máxima alcanzada en los saltos. | **"Motor Ocular":** Es muy personal. Si baja respecto a la base del usuario, indica **fatiga muscular**. |
| **`sacada_vel_pico_max`** | °/s | La velocidad más alta registrada en todo el intento. | Límite físico de los músculos del sujeto. |
| **`sacada_acel_pico_media`** | °/s² | Promedio de la fuerza de arranque (aceleración máxima). | **Explosividad:** Qué tan rápido reaccionan los músculos. Muy útil para diferenciar jóvenes de mayores. |
| **`sacada_amplitud_media`** | Grados (°) | Tamaño promedio de los saltos realizados. | Estrategia de búsqueda: ¿Hace saltos cortos (locales) o largos (globales)? |
| **`sacada_duracion_media`** | ms | Tiempo promedio que tarda el ojo en moverse. | Eficiencia del movimiento. |

-----

## 4\. La "Main Sequence" (Huella Biométrica)

Métricas derivadas de la relación matemática entre Amplitud y Velocidad ($Velocidad = K \cdot Amplitud + C$).

| Columna | Unidad | Descripción | Interpretación |
| :--- | :--- | :--- | :--- |
| **`main_seq_pendiente_k`** | Adim. (Slope) | Pendiente de la recta de ajuste (Velocidad vs Amplitud). | **EL DNI BIOMÉTRICO:** Este valor es casi único por persona. Representa la "ganancia" del sistema oculomotor. |
| **`main_seq_intercepto`** | Adim. (Bias) | Punto de corte de la recta. | Ajuste secundario del modelo biométrico. |

-----

### ¿Cómo usar esto en tu Tesis?

1.  **Para Identificar Personas:**

      * Usa `main_seq_pendiente_k`, `sacada_vel_pico_media` y `sacada_acel_pico_media`. Son rasgos físicos difíciles de fingir.
      * Al aplicar PCA, estas variables deberían separar a "Víctor" de "Usuario X".

2.  **Para Detectar Fatiga:**

      * Compara los intentos iniciales (1-3) con los finales (8-10).
      * Deberías ver una **bajada** en `sacada_vel_pico_media` (músculos cansados).
      * Deberías ver un **aumento** en `fijacion_duracion_media_ms` (cerebro procesando más lento).

3.  **Para Evaluar Atención:**

      * Usa `ratio_tiempo_fijacion`. Si es muy bajo, el usuario estaba "perdido" o escaneando sin prestar atención.
# Proyecto Integrador — Diseño de un índice de señales textuales basado en preguntas de proveedores en procesos de Subasta Inversa Electrónica del Sistema Oficial de Contratación Pública del Ecuador (SOCE) mediante Machine Learning y NLP

> SOCE / Subasta Inversa Electrónica (SIE)

Este directorio contiene el código fuente de la tesis. El trabajo se construye sobre el proyecto base de Francisco Roh (modelos NB, RF y LR entrenados sobre embeddings de OpenAI `text-embedding-3-large`) y extiende su uso a preguntas SIE 2021 extraídas de la base Kapak, define una familia de índices de riesgo por proceso, y los evalúa frente a un juez humano y a un juez LLM (gpt-5-mini con prompt 15).

---

## Flujo general del pipeline (4 etapas)

Cada etapa del proyecto vive en una subcarpeta de `src/`. Se ejecutan en este orden:

### [1] `Extraccion_Datos_Kapak_y_embeddings/`
Extrae preguntas SIE 2021 desde Kapak (PostgreSQL) y genera sus embeddings con `text-embedding-3-large`.

- **Entradas:** base Kapak + `OPENAI_API_KEY`
- **Salidas:**
  - `preguntas_2021.csv`
  - `output/2021/preguntas_2021_prepared.csv`
  - `output/2021/embeddings_2021_<start>_<end>.csv` (chunks de 20k)

### [2] `Reproduccion_de_Modelos_Seleccionados/`
Reproduce los tres modelos seleccionados con sus hiperparámetros finales y valida métricas sobre el conjunto de test de Roh.

- **Entradas:** `Archivos_NB / RF / LR / RF_aug` + `test_embeddings_*.csv`
- **Salidas:** métricas en consola (no persiste archivos)

### [3] `Experimentacion_Diseno_Indice/`
Diseña y compara distintas familias de índices de riesgo por proceso a partir de los embeddings de la etapa 1. Aquí también se calculan los pesos del Brier score para `ID_prob_weighted`.

- **Entradas:** `embeddings_2021_*.csv` + `Archivos_NB / RF / LR / RF_aug`
- **Salidas:** `output/2021/indices_per_process.csv` (7 índices `ID_*`)

### [4] `Evaluacion_LLM_as_a_Judge/`
Calibra un LLM como juez sustituto del juez humano (iteración sobre 15 prompts) y compara los 7 índices `ID_*` contra el LLM-juez sobre 400 procesos aleatorios de 2021. Selecciona el mejor índice.

- **Entradas:**
  - `dataset.csv` (5005 preguntas etiquetadas humanas)
  - `output/2021/preguntas_2021_prepared.csv`
  - `output/2021/indices_per_process.csv`
- **Salidas:**
  - `llm_alignment_results_val_prompt15.csv`
  - `ID_*_final_test_results_20.csv`
  - `output/2021/llm_prompt15_random_process_questions.csv`
  - `output/2021/llm_prompt15_random_process_densities.csv`

---

Nota: la gran mayoría de los archivos de entrada y salida son bastante pesados (MB o incluso GB). Por ende, se decidió excluirlos de este repositorio

## Contenido por carpeta

### `src/Extraccion_Datos_Kapak_y_embeddings/`

**Propósito:** extraer las preguntas SIE 2021 desde la base Kapak (PostgreSQL) y convertirlas en embeddings utilizables por los modelos.

- **`kapak_extract_embeddings.py`** — Primer experimento. Extrae preguntas-respuestas de Kapak (proyecto SIE) y genera embeddings con `text-embedding-3-large` en una sola pasada.
  - Salidas: `output/kapak_sie_questions_raw.csv`, `output/kapak_sie_questions_embeddings.jsonl`, `output/kapak_sie_embeddings_matrix_text_embedding_3_large.csv`.
  - Posteriormente se descartó en favor de un flujo en dos pasos para 2021 (ver siguientes archivos), porque Roh ya había tomado preguntas de 2022 para sus datasets de entrenamiento.

- **`kapak_extract_questions_2021.py`** — Solo extracción. Trae todas las preguntas SIE de 2021 desde Kapak con metadata de proceso (presupuesto, monto, estado, `es_acusatoria` por reglas Kapak, clasificación).
  - Salida: `preguntas_2021.csv`

- **`data_transformation_2021.py`** — Transforma `preguntas_2021.csv` → embeddings `text-embedding-3-large` en chunks (20.000 filas por chunk). Soporta reanudación por `start-idx` y usa `ThreadPoolExecutor` para no exceder rate limits de OpenAI.
  - Salidas: `output/2021/preguntas_2021_prepared.csv` (limpio), `output/2021/embeddings_2021_<start>_<end>.csv` (3072 columnas)
  - Ejecución típica: cada bloque de 20k preguntas (134.294 en total).

- **`data_transformation_2021.txt`** — Notas y guía de ejecución para `data_transformation_2021.py`.

- **`resumen_extraccion_datos_kapak_y_embeddings.txt`** — Resumen narrativo de la fase de extracción y por qué se cambió del pipeline de 2022 al de 2021.

- **`.env.example`** — Plantilla de variables de entorno para esta etapa: credenciales de PostgreSQL Kapak + `OPENAI_API_KEY` + parámetros de extracción.

---

### `src/Experimentacion_Diseno_Indice/`

**Propósito:** explorar y diseñar la familia de índices de riesgo por proceso. Los notebooks evolucionan desde el primer pipeline end-to-end (NB+RF+LR con 8 índices `IC_*`) hasta la versión final con 7 índices `ID_*` y RF aumentado.

- **`modelos_probabilidades.ipynb`** — Explora estrategias de agregación de probabilidades por proceso (media, máximo, top-K, percentil 90, densidad) sobre los modelos NB, RF, LR y RF aumentado.
  - **AQUÍ SE CALCULAN LOS PESOS DEL BRIER SCORE QUE SE USAN EN `ID_prob_weighted`** (`W_LR=0.5478`, `W_RF=0.3998`, `W_RF_AUG=0.0524`).
  - No persiste archivos: solo análisis exploratorio.

- **`results_2021_v1.ipynb`** — Primera versión "limpia" del cálculo de índices por proceso. Usa solo 2 modelos (RF + LR) y produce 5 familias de índices `IC_*`:
  - `IC_M1_rf`, `IC_M1_lr` (media)
  - `IC_M2_rf`, `IC_M2_lr` (Noisy-OR)
  - `IC_M3` (media del soft vote ponderado)
  - `IC_M4` (Noisy-OR del soft vote ponderado)
  - `IC_M5_density` (% preguntas con voto binario ponderado ≥ 0.5)

  También calcula normalizaciones (z-score, percentil, robustZ) y un análisis semanal con boxplots, violines, top-K y heatmaps. Descartada porque omitía Naive Bayes y luego se decidió mantener los tres perfiles de modelos.

- **`results_2021_v2.ipynb`** — **[VERSIÓN FINAL DE LOS ÍNDICES]** Reescritura. Usa 3 modelos (RF + RF aumentado + LR) y produce 7 índices `ID_*`:
  - `ID_bin_rf`, `ID_bin_rf_aug`, `ID_bin_lr` (% acusatorias por modelo)
  - `ID_prob_rf`, `ID_prob_rf_aug`, `ID_prob_lr` (media de probabilidades)
  - `ID_prob_weighted = W_RF·ID_prob_rf + W_RF_AUG·ID_prob_rf_aug + W_LR·ID_prob_lr`

  Salida: `output/2021/indices_per_process.csv`

- **`indices_and_distributions_2021.txt`** — Resumen del notebook end-to-end original (está también anotado a propósito como referencia histórica del primer experimento).

- **`resumen_Experimentacion_Diseño_Indice.txt`** — Resumen narrativo de las decisiones de diseño entre v1 y v2.

---

### `src/Reproduccion_de_Modelos_Seleccionados/`

**Propósito:** reproducir los tres modelos seleccionados con los hiperparámetros finales y validar métricas sobre el conjunto de test, garantizando que el pipeline de la tesis es coherente con el de Roh.

- **`reproduccion_modelos_seleccionados.ipynb`** — Entrena Naive Bayes, Random Forest, Random Forest aumentado y Logistic Regression con los `Archivos_*/train_embeddings_*.csv` del repositorio de Roh; también evalúa las variantes basadas en probabilidad (no binarias). Solo imprime métricas; no escribe archivos.

- **`resumen_reproduccion_modelos_seleccionados.txt`** — Resumen narrativo de la fase. Recordatorio: los `Archivos_*` de entrenamiento y test provienen directamente del repositorio de Roh.

---

### `src/Evaluacion_LLM_as_a_Judge/`

**Propósito:** validar los índices producidos en la etapa 2 usando un LLM calibrado como juez sustituto del juez humano, y elegir el mejor índice.

- **`llm_as_a_judge.ipynb`** — Notebook con 9 secciones:
  1. Splits del dataset humano (5005 preguntas) en alignment 80% y test 20%, con sub-split dev/val.
  2. Calibración del prompt LLM-juez (prompt 15 activo). Reporta confusion matrix, F1/precision/recall vs etiqueta humana sobre `alignment_set_val`.
  3. Genera predicciones de los 7 índices `ID_*` sobre el test set 20%.
  4. Genera embeddings `text-embedding-3-large` del test set.
  5. Aplica el LLM-juez a 400 procesos aleatorios de 2021 y calcula `llm_densidad` por proceso. 
  6. Compara distribuciones LLM vs cada índice (KS, Wasserstein, Jensen-Shannon, histogramas y ECDFs).
  7. Compara ordenamientos (Spearman, Kendall, Pearson, top-K overlap `k=10, 20, 50, 100`).
  8. Tablas finales y búsqueda de mejor threshold para índices probabilísticos.

- **`Prompts/`** — Historial de los 15 prompts iterados durante la calibración del juez (`prompt1.txt` … `prompt15.txt`). El prompt 15 es el activo y vive embebido en `llm_as_a_judge.ipynb`; los `.txt` sirven como bitácora del proceso de alineación de prompt.

- **`resumen_evaluacion_llm_as_a_judge.txt`** — Resumen narrativo del notebook con descripción de cada sección.

---


## Configuración

- **`.env.example`** — Plantilla raíz para variables de entorno. Copiar a `.env` (en la raíz del proyecto o en `src/`) y completar:
  - `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` (Kapak PostgreSQL)
  - `OPENAI_API_KEY` (embeddings + LLM)
  - `START_DATE`, `END_DATE`, `PROCESS_LIMIT`, `BATCH_SIZE` (filtros extracción)
  - `OUTPUT_DIR` (carpeta de salidas)


---

## Dependencias de datos externos

- Base PostgreSQL Kapak SIE (acceso vía VPN del proyecto Kapak).
- Carpetas `Archivos_Naive_Bayes/`, `Archivos_Random_Forest/`, `Archivos_Random_Forest_1/` y `Archivos_Logistic_Regression/` en la raíz del proyecto, con los CSV de entrenamiento de Roh.
- `dataset.csv` (5.005 preguntas etiquetadas humanas) para la etapa 4.
- `OPENAI_API_KEY` con acceso a `text-embedding-3-large` y al modelo de razonamiento usado como juez.


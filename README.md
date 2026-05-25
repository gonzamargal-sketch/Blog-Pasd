# Clasificación de grietas en carreteras con DINO

Repositorio del trabajo de la asignatura PASD. Clasificación de grietas en imágenes de carretera del dataset RDD2022 usando transformers de la familia DINO como extractores de características y clasificadores clásicos (MLP, SVM, LightGBM).

---

## Requisitos

```bash
pip install torch torchvision transformers pillow tqdm numpy
pip install scikit-learn lightgbm scipy shap matplotlib nbformat
```

---


## 1. Limpieza del dataset

Elimina China_Drone, imágenes sin anotaciones, imágenes con solo D40 y aplica submuestreo estratificado por país.

```bash
# Primero prueba sin copiar archivos
python clean_rdd2022.py --input C:/RDD2022 --output C:/RDD2022_clean --dry-run

# Ejecución real
python clean_rdd2022.py --input C:/RDD2022 --output C:/RDD2022_clean

# Con ratio de submuestreo personalizado
python clean_rdd2022.py --input C:/RDD2022 --output C:/RDD2022_clean --sample-ratio 0.3
```

---

## 2. Extracción de características

Extrae features con DINOv1, DINOv2, DINOv3-ViT y DINOv3-ConvNext y guarda los tensores en disco.

```bash
# Todos los modelos a la vez
python extract_features.py --dataset C:/RDD2022_clean --output C:/features --model all --num-workers 0

# Un modelo concreto
python extract_features.py --dataset C:/RDD2022_clean --output C:/features --model dinov2 --num-workers 0

# Si hay OOM en GPU, reducir batch
python extract_features.py --dataset C:/RDD2022_clean --output C:/features --model dinov2 --batch-size 16 --num-workers 0
```

Modelos disponibles: `dinov1` · `dinov2` · `dinov3-vit` · `dinov3-convnext` · `all`

---

## 3. Análisis de viabilidad del split

Verifica que hay suficientes imágenes por clase en validación y test antes de entrenar.

Abrir y ejecutar: `análisis_dataset.ipynb`

---

## 4. Entrenamiento de clasificadores

### MLP

```bash
# Configuración recomendada (mejor resultado con DINOv1)
python train_detector.py --features C:/features/dinov2 --output C:/models/dinov2 --epochs 20 --dropout 0.5 --hidden-dim 256
```

### SVM y LightGBM

```bash
# Ambos modelos
python train_classifiers.py --features C:/features/dinov2 --output C:/models/dinov2

# Solo LightGBM guardando el modelo para SHAP
python train_classifiers.py --features C:/features/dinov2 --output C:/models/dinov2 --model lgbm --save-models

# Parámetros opcionales
python train_classifiers.py --features C:/features/dinov2 --output C:/models/dinov2 --svm-c 1.0 --lgbm-lr 0.05 --lgbm-leaves 63
```

Repetir para cada extractor cambiando la ruta de `--features`:
`dinov1` · `dinov2` · `dinov3-vit` · `dinov3-convnext`

---

## 5. Comparativa de resultados

```bash
# Comparativa experimentos DINOv1 (MLP)
# Abrir: dinov1_experiment_comparison.ipynb

# Comparativa extractores + tiempos de extracción
# Abrir: dino_comparison_with_timing.ipynb

# Comparativa SVM y LightGBM sobre los 4 extractores
# Abrir: svm_lgbm_4dinos_comparison.ipynb
```

---

## 6. Fine-tuning de DINOv2

### Fine-tuning de clasificación

```bash
# Configuración recomendada RTX 4060
python finetune_dinov2.py --dataset C:/RDD2022_clean --output C:/models/dinov2-finetuned --n-unfrozen-blocks 2 --epochs 30 --num-workers 0

# Si hay OOM reducir batch
python finetune_dinov2.py --dataset C:/RDD2022_clean --output C:/models/dinov2-finetuned --batch-size 8 --num-workers 0
```

### Evaluación del modelo fine-tuneado

Ajusta las rutas `CHECKPOINT`, `DATASET` y `SCRIPT_DIR` al principio del archivo y ejecuta:

```bash
python evaluate_finetuned.py
```

### Fine-tuning de detección

```bash
# Primero prueba con 2 épocas para verificar que no hay errores
python finetune_dinov2_detection.py --dataset C:/RDD2022_clean --output C:/PASD/models/dinov2-detection --epochs 2 --batch-size 4 --num-workers 0

# Entrenamiento completo (~3-6 horas con RTX 4060)
python finetune_dinov2_detection.py --dataset C:/RDD2022_clean --output C:/PASD/models/dinov2-detection --epochs 50 --batch-size 4 --num-workers 0
```

> ⚠️ Usar una ruta sin caracteres especiales (sin `º`, espacios, etc.) en `--output` para evitar problemas con LightGBM y scipy.

---

## 7. Explicabilidad

### Attention Rollout

```bash
# DINOv2 sin fine-tuning
python attention_rollout_fine_tunning.py --dataset C:/RDD2022_clean --model dinov2 --focus all --n-images 15 --output C:/attention_maps

# DINOv2 fine-tuneado (clasificación o detección — detecta automáticamente)
python attention_rollout_fine_tunning.py --dataset C:/RDD2022_clean --model dinov2-finetuned --checkpoint C:/models/dinov2-finetuned/best_model.pt --focus all --n-images 15 --output C:/attention_maps
```

Opciones de `--focus`: `all` · `d00` · `d10` · `d20`

### Patch Importance SHAP

Requiere haber ejecutado `train_classifiers.py` con `--save-models` previamente.

```bash
python patch_importance.py --features C:/features/dinov2 --model-dir C:/PASD/models/dinov2 --dataset C:/RDD2022_clean --dino dinov2 --focus d10 --n-images 8
```

---

## Notas

- Usar siempre `--num-workers 0` en Windows si hay errores de multiprocessing
- Para rutas con espacios, envolver entre comillas dobles: `"C:/Mi Carpeta/dataset"`
- Los modelos LightGBM se guardan como `.txt` — usar rutas sin caracteres especiales
- El test set oficial de RDD2022 no tiene anotaciones; el split de test se construye a partir del train anotado

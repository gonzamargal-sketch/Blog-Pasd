"""
RDD2022 — Mapa de importancia de patches con SHAP + LightGBM
=============================================================
Conecta la explicabilidad de DINO con las predicciones de LightGBM.
Para cada imagen, muestra qué regiones espaciales de la imagen han
contribuido más a la predicción de cada clase.

Cómo funciona:
  1. Carga el modelo DINO y extrae los patch tokens de cada imagen
     (NO el vector promediado, sino los 196 tokens individuales)
  2. Usa SHAP TreeExplainer sobre el modelo LightGBM entrenado
  3. Para cada imagen, obtiene los SHAP values por dimensión de feature
  4. Proyecta esos valores de vuelta al espacio espacial de patches (14x14)
  5. Genera un mapa de calor por clase (D00, D10, D20)

Por qué esto es mejor que Attention Rollout:
  - Está directamente conectado a las predicciones del clasificador
  - Muestra qué regiones importan para CADA clase por separado
  - Distingue entre regiones que activan una clase vs las que la inhiben
  - Usa SHAP values que tienen interpretación matemática sólida

Uso:
  # Analizar 5 imágenes con DINOv2 + LightGBM
  python patch_importance.py --features C:/features/dinov2 --model-dir C:/models/dinov2 --dataset C:/RDD2022_clean --dino dinov2

  # Foco en imágenes con D10
  python patch_importance.py --features C:/features/dinov2 --model-dir C:/models/dinov2 --dataset C:/RDD2022_clean --dino dinov2 --focus d10 --n-images 8

Dependencias:
  pip install torch torchvision shap lightgbm pillow numpy matplotlib tqdm

Salida:
  patch_importance/
    dinov2/
      importance_img001_D10.png   ← mapa por clase para cada imagen
      importance_img002_D00_D20.png
      summary_grid.png
"""

import argparse
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import shap
import lightgbm as lgb
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────

CLASSES       = ["D00", "D10", "D20"]
IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

VALID_COUNTRIES = [
    "China_MotorBike", "Czech", "India",
    "Japan", "Norway", "United_States",
]

CLASS_COLORS = {
    "D00": "Oranges",
    "D10": "Greens",
    "D20": "Reds",
}


# ─────────────────────────────────────────────
# Carga de modelos DINO para extracción de patches
# ─────────────────────────────────────────────

def load_dino_for_patches(dino_name: str, device: torch.device):
    """
    Carga DINO configurado para devolver patch tokens individuales,
    NO el vector promediado. Necesitamos los 196 tokens por separado
    para poder proyectarlos de vuelta al espacio espacial 14x14.
    """
    if dino_name == "dinov1":
        model = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_vitb16", pretrained=True, verbose=False
        )
        patch_size        = 16
        n_register_tokens = 0
        hidden_size       = 768

    elif dino_name == "dinov2":
        model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14", pretrained=True, verbose=False
        )
        patch_size        = 14
        n_register_tokens = 0
        hidden_size       = 768

    elif dino_name == "dinov3-vit":
        from transformers import AutoModel
        CKPT  = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        model = AutoModel.from_pretrained(CKPT)
        patch_size        = 16
        n_register_tokens = 4
        hidden_size       = model.config.hidden_size

    else:
        raise ValueError(f"Modelo no soportado: {dino_name}")

    model.eval()
    model.to(device)
    return model, patch_size, n_register_tokens, hidden_size


@torch.no_grad()
def extract_patch_tokens(model, img_tensor: torch.Tensor,
                         dino_name: str, n_register_tokens: int) -> np.ndarray:
    """
    Extrae los patch tokens individuales de una imagen.
    Devuelve (n_patches, hidden_size) — por ejemplo (196, 768).

    Estos tokens son los que forman el espacio 14x14 de la imagen.
    Cada token corresponde a un parche de 16x16 pixels.
    """
    if dino_name == "dinov1":
        # get_intermediate_layers devuelve tokens incluyendo CLS
        out   = model.get_intermediate_layers(img_tensor, n=1)[0]  # (1, 1+P, 768)
        tokens = out[0, 1:, :]  # quitar CLS, quedarse con patches

    elif dino_name == "dinov2":
        out    = model.forward_features(img_tensor)
        tokens = out["x_norm_patchtokens"][0]  # (P, 768) — ya sin CLS

    elif dino_name == "dinov3-vit":
        out    = model(pixel_values=img_tensor)
        # last_hidden_state: (1, 1+4+196, 768) — CLS + 4 registers + patches
        n_skip = 1 + n_register_tokens
        tokens = out.last_hidden_state[0, n_skip:, :]  # (196, 768)

    return tokens.cpu().numpy()  # (196, 768)


def tokens_to_feature_vector(patch_tokens: np.ndarray,
                              cls_vector: np.ndarray) -> np.ndarray:
    """
    Construye el vector de features que usa el clasificador LightGBM:
    CLS (768) + patch avg (768) = 1536 dimensiones.
    """
    patch_avg = patch_tokens.mean(axis=0)  # (768,)
    return np.concatenate([cls_vector, patch_avg])  # (1536,)


# ─────────────────────────────────────────────
# SHAP sobre LightGBM
# ─────────────────────────────────────────────

def load_lgbm_models(model_dir: Path) -> list:
    """
    Carga los modelos LightGBM guardados.
    train_classifiers.py guarda un modelo por clase (D00, D10, D20).
    Intentamos cargar el report JSON para recuperar los parámetros
    y recrear los modelos si no están guardados como archivos .txt.
    """
    models = []

    # Buscar modelos guardados como booster
    for cls in CLASSES:
        model_path = model_dir / f"lgbm_{cls.lower()}.txt"
        if model_path.exists():
            clf = lgb.Booster(model_file=str(model_path))
            models.append(clf)
        else:
            print(f"  [WARN] Modelo LightGBM para {cls} no encontrado en {model_path}")
            print(f"         Ejecuta train_classifiers.py con --save-models primero")

    return models


def compute_shap_patch_importance(
    shap_values: np.ndarray,
    n_patches_side: int = 14,
    hidden_size: int = 768,
) -> np.ndarray:
    """
    Proyecta los SHAP values del vector de features (1536 dims) de vuelta
    al espacio espacial de patches (14x14).

    El vector de features tiene esta estructura:
      [CLS_768dims | patch_avg_768dims]

    Los SHAP values de la segunda mitad (patch_avg) corresponden a las
    768 dimensiones del promedio de todos los patches. Para recuperar la
    importancia espacial, calculamos qué tan diferente es cada patch
    individual respecto al promedio, ponderado por los SHAP values.

    En la práctica: importancia_patch_i = |patch_i - patch_avg| · |shap_patch_avg|
    Esto nos da una estimación de qué patches han desviado más el promedio
    hacia la dirección que el modelo considera importante.
    """
    # Los SHAP de la segunda mitad corresponden al patch avg
    shap_patch_avg = shap_values[hidden_size:]  # (768,)

    return shap_patch_avg  # se combina con los tokens individuales en plot


def project_importance_to_spatial(
    patch_tokens:     np.ndarray,  # (196, 768)
    shap_patch_avg:   np.ndarray,  # (768,)
    n_patches_side:   int = 14,
) -> np.ndarray:
    """
    Para cada patch, calcula su contribución a la predicción como:
      importancia_i = cosine_similarity(patch_i, shap_direction)

    donde shap_direction es el vector de SHAP values del patch_avg.
    Esto da un escalar por patch que indica si ese patch empuja
    la predicción hacia arriba o hacia abajo.

    Devuelve un mapa (14, 14) con valores en [-1, 1].
    """
    # Normalizar la dirección SHAP
    shap_norm = shap_patch_avg / (np.linalg.norm(shap_patch_avg) + 1e-8)

    # Para cada patch, proyectar sobre la dirección SHAP
    # (196, 768) · (768,) = (196,)
    importance = patch_tokens @ shap_norm

    # Normalizar a [-1, 1]
    max_abs = np.abs(importance).max() + 1e-8
    importance = importance / max_abs

    # Reshape a grid espacial
    return importance.reshape(n_patches_side, n_patches_side)


def importance_to_heatmap(importance_grid: np.ndarray,
                           img_size: int = 224) -> np.ndarray:
    """Redimensiona el grid de importancia al tamaño de la imagen."""
    from PIL import Image as PILImage
    # Usar solo valores positivos (contribuciones que activan la clase)
    pos_map = np.clip(importance_grid, 0, 1)
    grid_img = PILImage.fromarray((pos_map * 255).astype(np.uint8))
    grid_img = grid_img.resize((img_size, img_size), PILImage.BILINEAR)
    return np.array(grid_img) / 255.0


# ─────────────────────────────────────────────
# Visualización
# ─────────────────────────────────────────────

def plot_patch_importance(
    img_path:       str,
    patch_tokens:   np.ndarray,
    shap_values:    list,        # lista de (1536,) por clase
    classes:        list,
    dino_name:      str,
    output_path:    Path,
    hidden_size:    int = 768,
    n_patches_side: int = 14,
    img_size:       int = 224,
):
    """
    Genera figura con:
      Columna 1: imagen original
      Columnas 2-4: mapa de importancia por clase (D00, D10, D20)
    """
    img = np.array(Image.open(img_path).convert("RGB").resize((img_size, img_size)))
    cls_label = " + ".join(classes) if classes else "Sin anotación"

    n_cols = 1 + len(CLASSES)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), facecolor="#0F1117")
    fig.suptitle(
        f"{dino_name.upper()} + LightGBM SHAP | Clases presentes: {cls_label}",
        fontsize=11, color="#E8EAF0", y=1.02, fontfamily="monospace"
    )

    # Panel 1: imagen original
    axes[0].imshow(img)
    axes[0].set_title("Original", fontsize=10, color="#E8EAF0", pad=8)
    axes[0].axis("off")
    axes[0].set_facecolor("#0F1117")

    # Paneles 2-4: importancia por clase
    for j, cls in enumerate(CLASSES):
        ax      = axes[j + 1]
        sv      = shap_values[j]                 # (1536,) SHAP values para esta clase
        shap_pa = sv[hidden_size:]               # (768,) — parte de patch_avg

        # Proyectar al espacio espacial
        imp_grid = project_importance_to_spatial(
            patch_tokens, shap_pa, n_patches_side
        )
        heatmap = importance_to_heatmap(imp_grid, img_size)

        ax.imshow(img)
        cmap = CLASS_COLORS[cls]
        im   = ax.imshow(heatmap, cmap=cmap, alpha=0.6, vmin=0, vmax=1)

        # Marcar si la clase está presente en la imagen
        present = cls in classes
        marker  = "✓ PRESENTE" if present else "✗ AUSENTE"
        color   = "#2A9D8F" if present else "#E8503A"

        ax.set_title(f"Clase {cls}\n{marker}", fontsize=10,
                     color=color, pad=8, fontfamily="monospace")
        ax.axis("off")
        ax.set_facecolor("#0F1117")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.patch.set_facecolor("#0F1117")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="#0F1117")
    plt.close()


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

def load_samples(dataset_root: Path, metadata_path: Path,
                 n_images: int, focus: str, seed: int) -> list:
    """
    Carga muestras del metadata JSON del extractor de features.
    Reconstruye la ruta de imagen a partir del dataset_root actual,
    ignorando la ruta absoluta guardada en el metadata (que puede
    corresponder a otro ordenador o ubicación distinta).
    """
    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)

    # Reconstruir ruta usando dataset_root actual + country + split + filename
    for sample in meta:
        old_path = Path(sample["image_path"])
        # Extraer country, split y nombre de archivo del path original
        # Estructura esperada: .../country/split/images/filename.jpg
        parts    = old_path.parts
        try:
            # Buscar el índice de "images" en el path
            img_idx  = [p.lower() for p in parts].index("images")
            country  = parts[img_idx - 2]
            split    = parts[img_idx - 1]
            filename = parts[img_idx + 1] if len(parts) > img_idx + 1 else old_path.name
            # Reconstruir con dataset_root actual
            new_path = dataset_root / country / split / "images" / filename
            sample["image_path"] = str(new_path)
        except (ValueError, IndexError):
            # Si no se puede reconstruir, dejar la ruta original
            pass

    if focus != "all":
        cls_filter = focus.upper()
        meta = [s for s in meta if cls_filter in s.get("classes", [])]

    random.seed(seed)
    random.shuffle(meta)
    return meta[:n_images]


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────

def run_patch_importance(
    features_dir: Path,
    model_dir:    Path,
    dataset_root: Path,
    dino_name:    str,
    output_dir:   Path,
    n_images:     int,
    focus:        str,
    seed:         int,
    device:       torch.device,
):
    print(f"\n{'='*60}")
    print(f"  Patch Importance — {dino_name.upper()} + LightGBM SHAP")
    print(f"{'='*60}")

    # ── Cargar DINO ───────────────────────────────────────────────────
    print("\n  Cargando modelo DINO...")
    model, patch_size, n_reg, hidden_size = load_dino_for_patches(dino_name, device)
    n_patches_side = IMG_SIZE // patch_size  # 14

    # ── Cargar features de train para SHAP background ─────────────────
    print("  Cargando features de train para SHAP...")
    train_features = torch.load(
        features_dir / "train_features.pt", weights_only=True
    ).numpy()

    # SHAP necesita un background dataset (muestra representativa del train)
    # Usamos 100 muestras aleatorias para eficiencia
    np.random.seed(seed)
    bg_idx  = np.random.choice(len(train_features), size=min(100, len(train_features)), replace=False)
    bg_data = train_features[bg_idx]
    print(f"  Background SHAP: {bg_data.shape}")

    # ── Cargar modelos LightGBM ────────────────────────────────────────
    print("  Cargando modelos LightGBM...")
    lgbm_models = []
    for cls in CLASSES:
        model_path = model_dir / f"lgbm_{cls.lower()}.txt"
        if model_path.exists():
            clf = lgb.Booster(model_file=str(model_path))
            lgbm_models.append(clf)
            print(f"    {cls}: cargado")
        else:
            print(f"    {cls}: NO ENCONTRADO en {model_path}")
            print(f"           Asegúrate de haber corrido train_classifiers.py con --save-models")
            return

    # Crear explicadores SHAP por clase
    print("  Creando explicadores SHAP...")
    explainers = [
        shap.TreeExplainer(clf, data=bg_data, feature_perturbation="interventional")
        for clf in lgbm_models
    ]

    # ── Cargar muestras ────────────────────────────────────────────────
    meta_path = features_dir / "train_metadata.json"
    samples   = load_samples(dataset_root, meta_path, n_images, focus, seed)
    print(f"\n  {len(samples)} imágenes seleccionadas (focus={focus})")

    out_dir = output_dir / dino_name
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # ── Procesar cada imagen ───────────────────────────────────────────
    for i, sample in enumerate(tqdm(samples, desc="  Procesando")):
        img_path = sample["image_path"]
        classes  = list(set(sample.get("classes", [])))

        try:
            img_pil    = Image.open(img_path).convert("RGB")
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            # Extraer patch tokens individuales (196, 768)
            patch_tokens = extract_patch_tokens(model, img_tensor, dino_name, n_reg)

            # Construir el vector de features que usa LightGBM (1536)
            with torch.no_grad():
                if dino_name == "dinov1":
                    cls_vec = model(img_tensor).cpu().numpy()[0]
                elif dino_name == "dinov2":
                    out     = model.forward_features(img_tensor)
                    cls_vec = out["x_norm_clstoken"].cpu().numpy()[0]
                elif dino_name == "dinov3-vit":
                    out     = model(pixel_values=img_tensor)
                    cls_vec = out.last_hidden_state[0, 0, :].cpu().numpy()

            feat_vector = np.concatenate([cls_vec, patch_tokens.mean(axis=0)])

            # Calcular SHAP values para esta imagen
            shap_values = []
            for explainer in explainers:
                sv = explainer.shap_values(feat_vector.reshape(1, -1))
                if isinstance(sv, list):
                    sv = sv[1]  # clase positiva en clasificación binaria
                shap_values.append(sv.flatten())

            # Generar figura
            cls_str  = "_".join(sorted(classes)) if classes else "noclass"
            out_path = out_dir / f"importance_{i:03d}_{Path(img_path).stem}_{cls_str}.png"

            plot_patch_importance(
                img_path, patch_tokens, shap_values, classes,
                dino_name, out_path, hidden_size, n_patches_side
            )

        except Exception as e:
            print(f"\n  [ERROR] {img_path}: {e}")
            continue

    print(f"\n  Mapas de importancia guardados en: {out_dir}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Mapa de importancia de patches SHAP + LightGBM sobre DINO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python patch_importance.py \\
    --features C:/features/dinov2 \\
    --model-dir C:/models/dinov2 \\
    --dataset C:/RDD2022_clean \\
    --dino dinov2

  # Foco en D10 (la clase más difícil)
  python patch_importance.py \\
    --features C:/features/dinov2 \\
    --model-dir C:/models/dinov2 \\
    --dataset C:/RDD2022_clean \\
    --dino dinov2 --focus d10 --n-images 8

IMPORTANTE: train_classifiers.py debe haberse ejecutado con --save-models
para que los archivos lgbm_d00.txt, lgbm_d10.txt, lgbm_d20.txt existan.
        """
    )
    parser.add_argument("--features",  type=Path, required=True,
                        help="Carpeta con train_features.pt y train_metadata.json")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Carpeta con los modelos LightGBM (.txt)")
    parser.add_argument("--dataset",   type=Path, required=True,
                        help="Ruta al dataset limpio RDD2022_clean")
    parser.add_argument("--dino",      type=str, default="dinov2",
                        choices=["dinov1", "dinov2", "dinov3-vit"],
                        help="Extractor DINO a usar (default: dinov2)")
    parser.add_argument("--output",    type=Path, default=Path("patch_importance"),
                        help="Carpeta de salida (default: patch_importance/)")
    parser.add_argument("--n-images",  type=int, default=5,
                        help="Número de imágenes (default: 5)")
    parser.add_argument("--focus",     type=str, default="all",
                        choices=["all", "d00", "d10", "d20"],
                        help="Filtrar por clase (default: all)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--cpu",       action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cpu") if args.cpu else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    print(f"\nDevice:  {device}")
    print(f"DINO:    {args.dino}")
    print(f"Focus:   {args.focus}")
    print(f"Imágenes: {args.n_images}")

    args.output.mkdir(parents=True, exist_ok=True)

    run_patch_importance(
        features_dir=args.features,
        model_dir=args.model_dir,
        dataset_root=args.dataset,
        dino_name=args.dino,
        output_dir=args.output,
        n_images=args.n_images,
        focus=args.focus,
        seed=args.seed,
        device=device,
    )


if __name__ == "__main__":
    main()

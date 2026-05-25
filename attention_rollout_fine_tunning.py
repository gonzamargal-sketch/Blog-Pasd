"""
RDD2022 — Attention Rollout sobre modelos DINO
===============================================
Genera mapas de calor de atención usando Attention Rollout sobre
DINOv1, DINOv2 y DINOv3-ViT. El mapa muestra qué regiones de la
imagen contribuyen más a la representación final del CLS token.

Qué es Attention Rollout:
  En un transformer cada capa tiene cabezas de atención que indican
  cuánto mira cada token a los demás. Attention Rollout combina los
  mapas de atención de TODAS las capas de forma acumulativa, propagando
  la atención desde el CLS token hacia los patch tokens.
  El resultado es un mapa 14x14 que se redimensiona a la imagen original.

Uso:
  # Sobre DINOv2, 5 imágenes aleatorias del test
  python attention_rollout.py --dataset C:/RDD2022_clean --model dinov2

  # Sobre imágenes específicas
  python attention_rollout.py --dataset C:/RDD2022_clean --model dinov2 --n-images 10

  # Solo imágenes donde falla D10 (requiere metadata con predicciones)
  python attention_rollout.py --dataset C:/RDD2022_clean --model dinov2 --focus d10

  # Guardar en carpeta específica
  python attention_rollout.py --dataset C:/RDD2022_clean --model dinov2 --output C:/attention_maps

Dependencias:
  pip install torch torchvision transformers pillow numpy matplotlib tqdm

Salida:
  attention_maps/
    dinov2/
      rollout_img001_D00_D10.png   ← imagen + mapa superpuesto + atención por cabeza
      rollout_img002_D20.png
      ...
      summary_grid.png             ← grid con todas las imágenes procesadas
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
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
    "D00": "#F4A261",
    "D10": "#2A9D8F",
    "D20": "#E76F51",
}


# ─────────────────────────────────────────────
# Carga de modelos con hooks de atención
# ─────────────────────────────────────────────

class AttentionExtractor:
    """
    Registra hooks en las capas de atención del transformer
    para capturar los mapas de atención durante el forward pass.
    Los hooks se enganchan a las capas de atención sin modificar
    el modelo, de forma que el forward pass normal no cambia.
    """

    def __init__(self, model, model_name: str, n_register_tokens: int = 0):
        self.model             = model
        self.model_name        = model_name
        self.n_register_tokens = n_register_tokens
        self.attention_maps    = []  # lista de (n_heads, seq_len, seq_len) por capa
        self._hooks            = []
        self._register_hooks()

    def _register_hooks(self):
        """Engancha hooks a todas las capas de atención del transformer."""
        for name, module in self.model.named_modules():
            if self._is_attention_layer(name, module):
                hook = module.register_forward_hook(self._attention_hook)
                self._hooks.append(hook)

    def _is_attention_layer(self, name: str, module: nn.Module) -> bool:
        """Detecta capas de atención según el modelo."""
        if self.model_name in ["dinov1", "dinov2"]:
            return "attn" in name and hasattr(module, "attn_drop")
        elif self.model_name == "dinov3-vit":
            return "attention" in name and hasattr(module, "dropout")
        return False

    def _attention_hook(self, module, input, output):
        """Hook que captura los pesos de atención tras cada capa."""
        # DINOv1/v2: el módulo de atención guarda attn_weights internamente
        # Accedemos mediante el output si tiene la forma correcta
        if hasattr(module, "attn_weights"):
            self.attention_maps.append(module.attn_weights.detach().cpu())

    def clear(self):
        self.attention_maps = []

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


def load_dino_model(model_name: str, device: torch.device):
    """Carga el modelo DINO con acceso a los pesos de atención."""
    if model_name == "dinov1":
        print(f"  Cargando DINOv1 (ViT-B/16)...")
        model = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_vitb16", pretrained=True, verbose=False
        )
        n_heads           = 12
        patch_size        = 16
        n_register_tokens = 0

    elif model_name == "dinov2":
        print(f"  Cargando DINOv2 (ViT-B/14)...")
        model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14", pretrained=True, verbose=False
        )
        n_heads           = 12
        patch_size        = 14
        n_register_tokens = 0

    elif model_name == "dinov3-vit":
        from transformers import AutoModel, AutoImageProcessor
        CKPT  = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        print(f"  Cargando DINOv3-ViT desde HuggingFace...")
        model             = AutoModel.from_pretrained(CKPT)
        n_heads           = model.config.num_attention_heads
        patch_size        = 16
        n_register_tokens = 4  # fijo según documentación oficial

    elif model_name == "dinov2-finetuned":
        # Carga DINOv2 fine-tuneado desde checkpoint
        # Se pasa via kwargs porque necesita la ruta del checkpoint
        raise ValueError(
            "dinov2-finetuned debe cargarse con load_finetuned_model(), no con load_dino_model()"
        )

    else:
        raise ValueError(f"Modelo no soportado para Attention Rollout: {model_name}")

    model.eval()
    model.to(device)
    return model, n_heads, patch_size, n_register_tokens


def load_finetuned_model(checkpoint_path: Path, script_dir: Path, device: torch.device):
    """
    Carga el backbone de DINOv2 fine-tuneado desde best_model.pt.
    Soporta tanto el modelo de clasificacion (DINOv2FineTuned)
    como el de deteccion (DINOv2Detector).
    Detecta el tipo mirando las claves del state_dict directamente.
    """
    import sys
    sys.path.insert(0, str(script_dir))

    print(f"  Cargando checkpoint desde: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    saved_args = checkpoint["args"]

    # Detectar tipo de modelo mirando las claves del state_dict
    # Si tiene "detection_head" → modelo de deteccion
    # Si tiene "head.0" → modelo de clasificacion
    state_dict_keys = list(checkpoint["model_state_dict"].keys())
    is_detection = any("detection_head" in k for k in state_dict_keys)
    task = "detection" if is_detection else "classification"
    print(f"  Tipo de modelo detectado: {task}")

    if is_detection:
        from finetune_dinov2_detection import DINOv2Detector
        full_model = DINOv2Detector(
            n_unfrozen_blocks=int(saved_args.get("n_unfrozen_blocks", 2)),
            dropout=float(saved_args.get("dropout", 0.1)),
        )
        metric_val  = checkpoint.get("val_map", 0.0)
        metric_name = "mAP"
    else:
        from finetune_dinov2 import DINOv2FineTuned
        full_model = DINOv2FineTuned(
            n_unfrozen_blocks=int(saved_args.get("n_unfrozen_blocks", 2)),
            dropout=float(saved_args.get("dropout", 0.3)),
        )
        metric_val  = checkpoint.get("val_f1", 0.0)
        metric_name = "val F1"

    full_model.load_state_dict(checkpoint["model_state_dict"])

    # Extraer solo el backbone — igual para clasificacion y deteccion
    backbone = full_model.backbone
    backbone.eval()
    backbone.to(device)

    n_heads           = 12
    patch_size        = 14
    n_register_tokens = 0

    print(f"  Checkpoint OK | epoca: {checkpoint['epoch']} | {metric_name}: {metric_val:.4f}")
    return backbone, n_heads, patch_size, n_register_tokens


# ─────────────────────────────────────────────
# Attention Rollout
# ─────────────────────────────────────────────

def get_attention_maps_dinov1(model, img_tensor: torch.Tensor, n_heads: int):
    """
    Extrae mapas de atención de DINOv1 usando get_last_selfattention,
    que devuelve la atención de la ÚLTIMA capa directamente.
    Para Attention Rollout completo usamos get_intermediate_layers.
    """
    with torch.no_grad():
        # get_last_selfattention devuelve (1, n_heads, seq_len, seq_len)
        attn = model.get_last_selfattention(img_tensor)
    return attn  # (1, 12, 197, 197)


def get_attention_maps_dinov2(model, img_tensor: torch.Tensor):
    """
    Extrae mapas de atención de DINOv2.
    DINOv2 expone attn_weights en su módulo de atención.
    """
    attention_maps = []

    def hook_fn(module, input, output):
        # DINOv2 Attention module guarda attn en output cuando return_attn=True
        if isinstance(output, tuple) and len(output) > 1:
            attention_maps.append(output[1].detach().cpu())

    hooks = []
    for module in model.modules():
        if hasattr(module, "attn") and hasattr(module.attn, "proj"):
            h = module.attn.register_forward_hook(hook_fn)
            hooks.append(h)

    with torch.no_grad():
        model(img_tensor)

    for h in hooks:
        h.remove()

    return attention_maps  # lista de (1, n_heads, seq_len, seq_len) por capa


def get_attention_maps_dinov3(model, img_tensor: torch.Tensor):
    """Extrae atención de DINOv3-ViT usando output_attentions=True."""
    with torch.no_grad():
        outputs = model(
            pixel_values=img_tensor,
            output_attentions=True
        )
    # outputs.attentions: tuple de (1, n_heads, seq_len, seq_len) por capa
    return list(outputs.attentions)


def attention_rollout(attention_list: list, n_register_tokens: int = 0) -> np.ndarray:
    """
    Calcula Attention Rollout combinando los mapas de atención de todas las capas.

    Algoritmo:
      1. Para cada capa, promediar las n_heads cabezas de atención
      2. Añadir la identidad (conexiones residuales del transformer)
      3. Normalizar por filas
      4. Multiplicar acumulativamente todas las capas
      5. Extraer la fila del CLS token (pos 0) → atención del CLS a los patches

    El resultado indica qué patches contribuyen más a la representación del CLS.
    """
    result = None

    for attn in attention_list:
        # attn: (1, n_heads, seq_len, seq_len)
        attn = attn.squeeze(0)          # (n_heads, seq_len, seq_len)
        attn = attn.mean(dim=0)         # (seq_len, seq_len) — media sobre cabezas

        # Añadir identidad para modelar las skip connections
        seq_len = attn.shape[0]
        identity = torch.eye(seq_len)
        attn = attn + identity
        attn = attn / attn.sum(dim=-1, keepdim=True)  # normalizar

        if result is None:
            result = attn
        else:
            result = torch.matmul(attn, result)  # producto acumulativo

    # Extraer la atención del CLS token (posición 0) hacia todos los tokens
    cls_attn = result[0, :]  # (seq_len,)

    # Saltar CLS (pos 0) y register tokens (pos 1..n_reg)
    # quedarse solo con los patch tokens
    n_skip     = 1 + n_register_tokens
    patch_attn = cls_attn[n_skip:]  # (n_patches,)

    # Normalizar entre 0 y 1
    patch_attn = patch_attn - patch_attn.min()
    patch_attn = patch_attn / (patch_attn.max() + 1e-8)

    return patch_attn.numpy()


def rollout_to_heatmap(patch_attn: np.ndarray, img_size: int = 224,
                       patch_size: int = 16) -> np.ndarray:
    """
    Convierte el vector de atención de patches a un mapa 2D del tamaño
    de la imagen original.
    """
    n_patches_side = img_size // patch_size              # 14 para patch_size=16
    grid = patch_attn[:n_patches_side**2].reshape(n_patches_side, n_patches_side)

    # Redimensionar al tamaño de la imagen con interpolación bilineal
    from PIL import Image as PILImage
    grid_img = PILImage.fromarray((grid * 255).astype(np.uint8))
    grid_img = grid_img.resize((img_size, img_size), PILImage.BILINEAR)
    return np.array(grid_img) / 255.0


# ─────────────────────────────────────────────
# Visualización
# ─────────────────────────────────────────────

def plot_attention(
    img_path:   str,
    heatmap:    np.ndarray,
    classes:    list,
    model_name: str,
    output_path: Path,
    alpha:      float = 0.5,
):
    """
    Genera una figura con 3 paneles:
      1. Imagen original con clases anotadas
      2. Mapa de calor de atención
      3. Superposición (imagen + mapa de calor)
    """
    img = np.array(Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))

    cls_label = " + ".join(classes) if classes else "Sin anotación"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0F1117")
    fig.suptitle(
        f"{model_name.upper()} — Attention Rollout | Clases: {cls_label}",
        fontsize=12, color="#E8EAF0", y=1.02, fontfamily="monospace"
    )

    # Panel 1: imagen original
    axes[0].imshow(img)
    axes[0].set_title("Imagen original", fontsize=10, color="#E8EAF0",
                      fontfamily="monospace", pad=8)
    axes[0].axis("off")

    # Panel 2: mapa de calor solo
    im = axes[1].imshow(heatmap, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("Mapa de atención (Rollout)", fontsize=10,
                      color="#E8EAF0", fontfamily="monospace", pad=8)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: superposición
    axes[2].imshow(img)
    axes[2].imshow(heatmap, cmap="inferno", alpha=alpha, vmin=0, vmax=1)
    axes[2].set_title(f"Superposicion (alpha={alpha})", fontsize=10,
                      color="#E8EAF0", fontfamily="monospace", pad=8)
    axes[2].axis("off")

    for ax in axes:
        ax.set_facecolor("#0F1117")
    fig.patch.set_facecolor("#0F1117")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="#0F1117")
    plt.close()


def plot_summary_grid(results: list, output_path: Path, model_name: str):
    """
    Genera un grid resumen con todas las imágenes procesadas.
    Cada fila: imagen original | superposición con mapa de atención.
    """
    n = len(results)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n), facecolor="#0F1117")
    if n == 1:
        axes = [axes]

    fig.suptitle(f"{model_name.upper()} — Attention Rollout Summary",
                 fontsize=14, color="#E8EAF0", y=1.01)

    for i, (img_path, heatmap, classes) in enumerate(results):
        img = np.array(Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
        cls_label = " + ".join(classes) if classes else "Sin anotación"

        axes[i][0].imshow(img)
        axes[i][0].set_title(f"Original | {cls_label}", fontsize=9,
                             color="#E8EAF0", pad=5)
        axes[i][0].axis("off")

        axes[i][1].imshow(img)
        axes[i][1].imshow(heatmap, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
        axes[i][1].set_title("Atención superpuesta", fontsize=9,
                             color="#E8EAF0", pad=5)
        axes[i][1].axis("off")

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("#171B26")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"  Grid guardado: {output_path}")


# ─────────────────────────────────────────────
# Carga de imágenes del dataset
# ─────────────────────────────────────────────

def load_dataset_samples(dataset_root: Path, n_images: int,
                         focus: str, seed: int = 42) -> list:
    """
    Carga N imágenes del dataset con sus anotaciones.
    focus: 'all' | 'd00' | 'd10' | 'd20' — filtra por clase presente
    """
    import xml.etree.ElementTree as ET
    samples = []

    for country in VALID_COUNTRIES:
        for split in ["train", "test"]:
            images_dir = dataset_root / country / split / "images"
            xml_dir    = dataset_root / country / split / "annotations" / "xmls"

            if not images_dir.exists():
                continue

            for img_path in sorted(images_dir.glob("*.[jJpP][pPnN][gG]*")):
                xml_path = xml_dir / (img_path.stem + ".xml")
                classes  = []

                if xml_path.exists():
                    try:
                        root = ET.parse(xml_path).getroot()
                        for obj in root.findall("object"):
                            name_el = obj.find("name")
                            if name_el is not None and name_el.text.strip() in CLASSES:
                                classes.append(name_el.text.strip())
                        classes = list(set(classes))
                    except Exception:
                        pass

                samples.append({"image_path": str(img_path), "classes": classes})

    # Filtrar por clase si se especifica
    if focus != "all":
        cls_filter = focus.upper()
        samples = [s for s in samples if cls_filter in s["classes"]]

    random.seed(seed)
    random.shuffle(samples)
    return samples[:n_images]


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────

def run_attention_rollout(
    dataset_root:    Path,
    model_name:      str,
    output_dir:      Path,
    n_images:        int,
    focus:           str,
    alpha:           float,
    device:          torch.device,
    checkpoint_path: Path = None,
    script_dir:      Path = None,
):
    print(f"\n{'='*60}")
    print(f"  Attention Rollout — {model_name.upper()}")
    print(f"{'='*60}")

    # Cargar modelo
    if model_name == "dinov2-finetuned":
        if checkpoint_path is None:
            raise ValueError("Para dinov2-finetuned debes indicar --checkpoint con la ruta al best_model.pt")
        model, n_heads, patch_size, n_register_tokens = load_finetuned_model(
            checkpoint_path, script_dir or checkpoint_path.parent.parent, device
        )
    else:
        model, n_heads, patch_size, n_register_tokens = load_dino_model(model_name, device)

    # Transform
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Cargar muestras
    print(f"\n  Cargando imágenes (focus={focus})...")
    samples = load_dataset_samples(dataset_root, n_images, focus)
    print(f"  {len(samples)} imágenes seleccionadas")

    model_out_dir = output_dir / model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)

    results = []  # para el grid resumen

    for i, sample in enumerate(tqdm(samples, desc="  Procesando")):
        img_path = sample["image_path"]
        classes  = sample["classes"]

        # Cargar y preprocesar imagen
        img_pil   = Image.open(img_path).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        # Extraer mapas de atención según el modelo
        try:
            if model_name == "dinov1":
                # DINOv1 tiene get_last_selfattention nativo
                with torch.no_grad():
                    attn = model.get_last_selfattention(img_tensor)
                # attn: (1, n_heads, seq_len, seq_len)
                # Para rollout usamos solo la última capa
                attention_list = [attn.cpu()]

            elif model_name in ["dinov2", "dinov2-finetuned"]:
                # DINOv2 y DINOv2 fine-tuneado usan la misma arquitectura
                # de atención, así que el mismo código funciona para ambos
                # Capturamos qkv para reconstruir los mapas de atención
                attn_captured = []

                def _dinov2_attn_hook(module, input, output):
                    """
                    DINOv2 Attention: el módulo tiene un atributo 'qkv' que
                    proyecta a (B, N, 3*H*D). Capturamos los scores de atención
                    accediendo al output del softmax interno.
                    Alternativa: capturar directamente qkv y reconstruir.
                    """
                    # output del módulo Attention es el tensor ya proyectado (B, N, D)
                    # Necesitamos los scores: accedemos via el forward modificado
                    # DINOv2 guarda attn internamente en algunos builds
                    if hasattr(module, "attn_map"):
                        attn_captured.append(module.attn_map.detach().cpu())

                # Enganchamos hooks a todas las capas de atención
                hooks_dinov2 = []
                for name, module in model.named_modules():
                    # En DINOv2 torch.hub las capas de atención son MemEffAttention o Attention
                    if "attn" in name and hasattr(module, "qkv"):
                        h = module.register_forward_hook(_dinov2_attn_hook)
                        hooks_dinov2.append(h)

                with torch.no_grad():
                    model(img_tensor)

                for h in hooks_dinov2:
                    h.remove()

                if attn_captured:
                    attention_list = attn_captured
                else:
                    # Fallback robusto: usar get_last_selfattention si existe
                    # o reconstruir desde qkv manualmente
                    attention_list = []
                    hooks_qkv = []
                    qkv_outputs = []

                    def _qkv_hook(module, input, output):
                        # output de qkv: (B, N, 3*H*D)
                        B, N, _ = output.shape
                        n_h = n_heads
                        head_dim = output.shape[-1] // (3 * n_h)
                        qkv = output.reshape(B, N, 3, n_h, head_dim).permute(2, 0, 3, 1, 4)
                        q, k, v = qkv.unbind(0)  # (B, H, N, D)
                        scale = head_dim ** -0.5
                        attn = (q @ k.transpose(-2, -1)) * scale
                        attn = attn.softmax(dim=-1)          # (B, H, N, N)
                        qkv_outputs.append(attn.detach().cpu())

                    for name, module in model.named_modules():
                        if "attn" in name and hasattr(module, "qkv"):
                            h = module.qkv.register_forward_hook(_qkv_hook)
                            hooks_qkv.append(h)

                    with torch.no_grad():
                        model(img_tensor)

                    for h in hooks_qkv:
                        h.remove()

                    attention_list = qkv_outputs

            elif model_name == "dinov3-vit":
                attention_list = get_attention_maps_dinov3(model, img_tensor)
                attention_list = [a.cpu() for a in attention_list]

        except Exception as e:
            print(f"  [ERROR] {img_path}: {e}")
            continue

        if not attention_list:
            continue

        # Calcular Attention Rollout
        patch_attn = attention_rollout(attention_list, n_register_tokens)
        heatmap    = rollout_to_heatmap(patch_attn, IMG_SIZE, patch_size)

        # Guardar figura individual
        cls_str  = "_".join(sorted(classes)) if classes else "noclass"
        stem     = Path(img_path).stem
        out_path = model_out_dir / f"rollout_{i:03d}_{stem}_{cls_str}.png"
        plot_attention(img_path, heatmap, classes, model_name, out_path, alpha)

        results.append((img_path, heatmap, classes))

    # Guardar grid resumen
    if results:
        grid_path = model_out_dir / "summary_grid.png"
        plot_summary_grid(results, grid_path, model_name)

    print(f"\n  Mapas guardados en: {model_out_dir}")
    return results


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Attention Rollout sobre modelos DINO para RDD2022",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # 5 imágenes aleatorias con DINOv2
  python attention_rollout.py --dataset C:/RDD2022_clean --model dinov2

  # 10 imágenes que contienen D10 (la clase más difícil)
  python attention_rollout.py --dataset C:/RDD2022_clean --model dinov2 --n-images 10 --focus d10

  # Comparar DINOv1 vs DINOv2
  python attention_rollout.py --dataset C:/RDD2022_clean --model dinov1 --focus d10
  python attention_rollout.py --dataset C:/RDD2022_clean --model dinov2 --focus d10
        """
    )
    parser.add_argument("--dataset",   type=Path, required=True,
                        help="Ruta al dataset limpio RDD2022_clean")
    parser.add_argument("--model",      type=str, default="dinov2",
                        choices=["dinov1", "dinov2", "dinov3-vit", "dinov2-finetuned"],
                        help="Modelo DINO a usar (default: dinov2)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Ruta al best_model.pt del fine-tuning (solo para dinov2-finetuned)")
    parser.add_argument("--output",    type=Path, default=Path("attention_maps"),
                        help="Carpeta de salida (default: attention_maps/)")
    parser.add_argument("--n-images",  type=int, default=5,
                        help="Número de imágenes a procesar (default: 5)")
    parser.add_argument("--focus",     type=str, default="all",
                        choices=["all", "d00", "d10", "d20"],
                        help="Filtrar por clase presente (default: all)")
    parser.add_argument("--alpha",     type=float, default=0.55,
                        help="Transparencia del mapa sobre la imagen (default: 0.55)")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Semilla para reproducibilidad (default: 42)")
    parser.add_argument("--cpu",       action="store_true",
                        help="Forzar CPU")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cpu") if args.cpu else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"\nDevice: {device}")
    print(f"Modelo: {args.model}")
    print(f"Imágenes: {args.n_images} (focus={args.focus})")

    args.output.mkdir(parents=True, exist_ok=True)

    # script_dir: carpeta donde están finetune_dinov2.py y finetune_dinov2_detection.py
    # Por defecto asumimos que están en la misma carpeta que este script
    script_dir = Path(__file__).parent if args.checkpoint else None

    run_attention_rollout(
        dataset_root=args.dataset,
        model_name=args.model,
        output_dir=args.output,
        n_images=args.n_images,
        focus=args.focus,
        alpha=args.alpha,
        device=device,
        checkpoint_path=args.checkpoint,
        script_dir=script_dir,
    )


if __name__ == "__main__":
    main()

"""
RDD2022 — Extracción de características con DINOv1, DINOv2 y DINOv3
=====================================================================
Recorre el dataset limpio, extrae features por imagen usando el modelo
elegido y las guarda en disco junto con sus metadatos.

Modelos disponibles:
  dinov1        → ViT-B/16, torch.hub, feature dim: 1536 (CLS + patch avg)
  dinov2        → ViT-B/14, torch.hub, feature dim: 1536 (CLS + patch avg)
  dinov3-vit    → ViT-B/16, HuggingFace, feature dim: 1536 (CLS + patch avg)
                  Incluye register tokens que se descartan en la extracción
  dinov3-convnext → ConvNeXt-Tiny, HuggingFace, feature dim: 768 (pooler output)
                    Feature maps espaciales, pensado para detection heads

Uso:
  # Extraer con un modelo concreto
  python extract_features.py --dataset C:/RDD2022_clean --output C:/features --model dinov3-vit
  python extract_features.py --dataset C:/RDD2022_clean --output C:/features --model dinov3-convnext

  # Extraer con todos los modelos a la vez
  python extract_features.py --dataset C:/RDD2022_clean --output C:/features --model all

  # Solo DINOv1 y DINOv2 (como antes)
  python extract_features.py --dataset C:/RDD2022_clean --output C:/features --model dinov1
  python extract_features.py --dataset C:/RDD2022_clean --output C:/features --model dinov2

  # Reducir batch si hay OOM en GPU < 8GB
  python extract_features.py --dataset C:/RDD2022_clean --output C:/features --model dinov3-vit --batch-size 16

Dependencias:
  pip install torch torchvision transformers timm pillow tqdm numpy

Salida (por modelo):
  features/
    dinov1/
      train_features.pt     ← tensor (N, 1536)
      train_metadata.json
      test_features.pt
      test_metadata.json
    dinov2/           ← igual
    dinov3-vit/       ← igual, feature dim 1536
    dinov3-convnext/  ← feature dim 768 (pooler output espacial)
"""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────

VALID_COUNTRIES = [
    "China_MotorBike", "Czech", "India",
    "Japan", "Norway", "United_States",
]

IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Todos los modelos disponibles
ALL_MODELS = ["dinov1", "dinov2", "dinov3-vit", "dinov3-convnext"]

# Checkpoint de HuggingFace para cada variante DINOv3
DINOV3_VIT_CKPT      = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINOV3_CONVNEXT_CKPT = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class RDDDataset(Dataset):
    """
    Carga imágenes del RDD2022 limpio y sus metadatos.
    El transform se aplica externamente para poder reutilizar
    el mismo dataset con distintos preprocesados.
    """

    def __init__(self, root: Path, split: str, transform):
        self.transform = transform
        self.samples   = []

        for country in VALID_COUNTRIES:
            split_path = root / country / split
            images_dir = split_path / "images"
            xml_dir    = split_path / "annotations" / "xmls"

            if not images_dir.exists():
                continue

            for img_path in sorted(images_dir.glob("*.[jJpP][pPnN][gG]*")):
                xml_path   = xml_dir / (img_path.stem + ".xml") if xml_dir.exists() else None
                annotation = self._parse_xml(xml_path)
                self.samples.append({
                    "image_path": str(img_path),
                    "country":    country,
                    "split":      split,
                    "classes":    annotation["classes"],
                    "boxes":      annotation["boxes"],
                    "n_boxes":    len(annotation["boxes"]),
                })

        print(f"  [{split}] {len(self.samples)} imágenes encontradas")

    def _parse_xml(self, xml_path) -> dict:
        result = {"classes": [], "boxes": []}
        if xml_path is None or not Path(xml_path).exists():
            return result
        try:
            root = ET.parse(xml_path).getroot()
            for obj in root.findall("object"):
                name_el = obj.find("name")
                bnd     = obj.find("bndbox")
                if name_el is None or bnd is None:
                    continue
                cls = name_el.text.strip()
                box = {
                    "class": cls,
                    "xmin":  int(float(bnd.find("xmin").text)),
                    "ymin":  int(float(bnd.find("ymin").text)),
                    "xmax":  int(float(bnd.find("xmax").text)),
                    "ymax":  int(float(bnd.find("ymax").text)),
                }
                result["classes"].append(cls)
                result["boxes"].append(box)
        except Exception:
            pass
        return result

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        return img, idx


# ─────────────────────────────────────────────
# Carga de modelos y transforms
# ─────────────────────────────────────────────

def load_model_and_transform(model_name: str, device: torch.device):
    """
    Carga el modelo y devuelve (model, transform, feature_dim).
    Cada modelo tiene su propio preprocesado y dimensión de salida.
    """

    if model_name == "dinov1":
        print("  Cargando DINOv1 (ViT-B/16) desde torch.hub...")
        model = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_vitb16",
            pretrained=True,
            verbose=False,
        )
        transform   = _imagenet_transform()
        # CLS token + patch avg de la ultima capa concatenados
        feature_dim = 768 * 2  # 1536
        print(f"  DINOv1 cargado OK  feature dim: {feature_dim}")

    elif model_name == "dinov2":
        print("  Cargando DINOv2 (ViT-B/14) desde torch.hub...")
        model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            pretrained=True,
            verbose=False,
        )
        transform   = _imagenet_transform()
        # CLS token + patch avg de la ultima capa concatenados
        feature_dim = 768 * 2  # 1536
        print(f"  DINOv2 cargado OK  feature dim: {feature_dim}")

    elif model_name == "dinov3-vit":
        from transformers import AutoImageProcessor, AutoModel
        print(f"  Cargando DINOv3-ViT desde HuggingFace ({DINOV3_VIT_CKPT})...")
        processor = AutoImageProcessor.from_pretrained(DINOV3_VIT_CKPT)
        model     = AutoModel.from_pretrained(DINOV3_VIT_CKPT)
        # Usar el processor oficial en lugar del transform manual de ImageNet
        # DINOv3 puede usar normalizacion o resolucion distinta a la estandar
        transform   = _hf_transform(processor)
        feature_dim = model.config.hidden_size * 2  # CLS + patch avg = 1536
        # Segun documentacion oficial: todos los ViT de DINOv3 tienen exactamente
        # 4 register tokens: 1 CLS + 4 registers + 196 patches = 201 tokens
        model._num_register_tokens = 4
        print(f"  DINOv3-ViT cargado OK  register tokens: 4 (fijo)  feature dim: {feature_dim}")

    elif model_name == "dinov3-convnext":
        from transformers import AutoModel
        print(f"  Cargando DINOv3-ConvNext desde HuggingFace ({DINOV3_CONVNEXT_CKPT})...")
        model     = AutoModel.from_pretrained(DINOV3_CONVNEXT_CKPT)
        transform = _imagenet_transform()
        # Feature dim = suma de todas las etapas (96+192+384+768 para ConvNext-Tiny)
        # Cada etapa aporta su hidden_size tras avg pool espacial
        feature_dim = sum(model.config.hidden_sizes)  # 1440 para ConvNext-Tiny
        print(f"  DINOv3-ConvNext cargado OK  feature dim jerarquico: {feature_dim}")
        print(f"  Etapas: {model.config.hidden_sizes} -> concatenadas tras spatial avg pool")

    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

    model.eval()
    model.to(device)
    return model, transform, feature_dim


def _imagenet_transform() -> transforms.Compose:
    """Transform estandar ImageNet para DINOv1 y DINOv2."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _hf_transform(processor):
    """
    Wrapper que convierte un HuggingFace ImageProcessor en un callable
    compatible con el Dataset (recibe PIL Image, devuelve tensor).
    Usado para DINOv3-ViT para respetar la normalizacion oficial del modelo.
    """
    def transform(img):
        return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
    return transform




# ─────────────────────────────────────────────
# Extracción de features
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    model:      nn.Module,
    dataloader: DataLoader,
    device:     torch.device,
    model_name: str,
) -> torch.Tensor:
    """
    Pasa todas las imágenes por el modelo y devuelve un tensor (N, feature_dim).

    Estrategia por modelo:
      DINOv1          → CLS token + patch avg ultima capa              → (N, 1536)
      DINOv2          → CLS token + patch avg ultima capa              → (N, 1536)
      DINOv3-ViT      → CLS + patch avg ultima capa (sin registers)    → (N, 1536)
      DINOv3-ConvNext → avg pool espacial de 4 etapas concatenadas     → (N, 1440)
    """
    all_features = []

    for images, _ in tqdm(dataloader, desc=f"  Extrayendo [{model_name}]"):
        images = images.to(device)

        if model_name == "dinov1":
            cls_feat   = model(images)                                     # (B, 768)
            patch_tok  = model.get_intermediate_layers(images, n=1)[0]    # (B, 1+P, 768)
            patch_feat = patch_tok[:, 1:, :].mean(dim=1)                  # (B, 768)
            feat       = torch.cat([cls_feat, patch_feat], dim=1)         # (B, 1536)

        elif model_name == "dinov2":
            out        = model.forward_features(images)
            cls_feat   = out["x_norm_clstoken"]                           # (B, 768)
            patch_feat = out["x_norm_patchtokens"].mean(dim=1)            # (B, 768)
            feat       = torch.cat([cls_feat, patch_feat], dim=1)         # (B, 1536)

        elif model_name == "dinov3-vit":
            out              = model(pixel_values=images)
            last_hidden      = out.last_hidden_state                      # (B, 1+R+P, 768)
            n_reg            = model._num_register_tokens
            cls_feat         = last_hidden[:, 0, :]                       # (B, 768)
            # saltar CLS (0) y register tokens (1..n_reg), quedarse con patches
            patch_feat       = last_hidden[:, 1 + n_reg:, :].mean(dim=1) # (B, 768)
            feat             = torch.cat([cls_feat, patch_feat], dim=1)   # (B, 1536)

        elif model_name == "dinov3-convnext":
            # output_hidden_states=True devuelve los mapas de las 4 etapas:
            #   etapa 1: (B,  96, 56, 56) bordes y texturas finas
            #   etapa 2: (B, 192, 28, 28) patrones locales
            #   etapa 3: (B, 384, 14, 14) estructuras medias (grietas lineales)
            #   etapa 4: (B, 768,  7,  7) semantica global
            out = model(pixel_values=images, output_hidden_states=True)
            # Avg pool espacial de cada etapa: (B, C, H, W) -> (B, C)
            stage_feats = [
                stage.mean(dim=[2, 3])          # avg pool sobre H y W
                for stage in out.hidden_states  # 4 mapas de features
            ]
            # Concatenar las 4 etapas: (B, 96+192+384+768) = (B, 1440)
            feat = torch.cat(stage_feats, dim=1)

        else:
            raise ValueError(f"Modelo desconocido en extracción: {model_name}")

        all_features.append(feat.cpu())

    return torch.cat(all_features, dim=0)


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────

def run_extraction(
    dataset_root: Path,
    output_dir:   Path,
    model_name:   str,
    batch_size:   int,
    device:       torch.device,
    num_workers:  int,
) -> dict:
    """Extrae features y devuelve un dict con tiempos y stats por split."""
    print(f"\n{'='*60}")
    print(f"  Modelo: {model_name.upper()}")
    print(f"{'='*60}")

    t_load_start = time.time()
    model, transform, feature_dim = load_model_and_transform(model_name, device)
    t_load = time.time() - t_load_start
    print(f"  Tiempo de carga del modelo: {t_load:.1f}s")

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    timing = {
        "model":        model_name,
        "feature_dim":  feature_dim,
        "load_time_s":  round(t_load, 2),
        "splits":       {},
    }

    total_images = 0
    t_total_start = time.time()

    for split in ["train", "test"]:
        print(f"\n  --- Split: {split} ---")

        dataset = RDDDataset(dataset_root, split, transform)
        if len(dataset) == 0:
            print(f"  Sin imagenes en {split}, saltando...")
            continue

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        t_split_start = time.time()
        features      = extract_features(model, loader, device, model_name)
        t_split       = time.time() - t_split_start

        n_imgs = len(dataset)
        total_images += n_imgs
        imgs_per_sec  = n_imgs / t_split if t_split > 0 else 0

        feat_path = model_dir / f"{split}_features.pt"
        torch.save(features, feat_path)
        print(f"  Features: {feat_path}  shape={tuple(features.shape)}")

        meta_path = model_dir / f"{split}_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(dataset.samples, f, indent=2, ensure_ascii=False)

        print(f"  Tiempo {split}: {t_split:.1f}s  ({imgs_per_sec:.1f} imgs/s)")

        timing["splits"][split] = {
            "n_images":    n_imgs,
            "time_s":      round(t_split, 2),
            "imgs_per_s":  round(imgs_per_sec, 1),
            "shape":       list(features.shape),
        }

    timing["total_time_s"]  = round(time.time() - t_total_start, 2)
    timing["total_images"]  = total_images
    timing["total_imgs_per_s"] = round(
        total_images / timing["total_time_s"] if timing["total_time_s"] > 0 else 0, 1
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return timing


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrae features del RDD2022 con DINOv1, DINOv2 o DINOv3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Modelos disponibles: {ALL_MODELS + ['all']}

Ejemplos:
  # Un modelo concreto
  python extract_features.py --dataset C:/RDD2022_clean --output C:/features --model dinov3-vit
  python extract_features.py --dataset C:/RDD2022_clean --output C:/features --model dinov3-convnext

  # Todos a la vez
  python extract_features.py --dataset C:/RDD2022_clean --output C:/features --model all

  # Reducir batch si hay OOM (GPU < 8GB)
  python extract_features.py --dataset C:/RDD2022_clean --output C:/features --model dinov3-vit --batch-size 16

  # Sin multiprocessing (Windows)
  python extract_features.py --dataset C:/RDD2022_clean --output C:/features --model dinov2 --num-workers 0

Dimensiones de salida:
  dinov1          → (N, 1536)  CLS + patch avg
  dinov2          → (N, 1536)  CLS + patch avg
  dinov3-vit      → (N, 1536)  CLS + patch avg (register tokens descartados)
  dinov3-convnext → (N, 1440)  avg pool espacial de 4 etapas jerarquicas
        """
    )
    parser.add_argument("--dataset",     type=Path, required=True,
                        help="Ruta al dataset limpio RDD2022_clean")
    parser.add_argument("--output",      type=Path, required=True,
                        help="Directorio donde guardar las features")
    parser.add_argument("--model",       type=str, default="dinov2",
                        choices=ALL_MODELS + ["all"],
                        help="Modelo a usar (default: dinov2)")
    parser.add_argument("--batch-size",  type=int, default=32,
                        help="Batch size (default: 32, reducir a 8-16 si hay OOM)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Workers DataLoader (default: 4, usar 0 en Windows si hay errores)")
    parser.add_argument("--cpu",         action="store_true",
                        help="Forzar CPU aunque haya GPU")
    return parser.parse_args()


def print_timing_report(all_timings: list, device: torch.device):
    """Imprime tabla comparativa de tiempos y guarda timing_report.json."""
    print(f"\n{'='*70}")
    print("  REPORTE COMPARATIVO DE TIEMPOS")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print()
    print(f"  {'Modelo':<20} {'Carga(s)':>10} {'Train(s)':>10} {'Test(s)':>10} {'Total(s)':>10} {'imgs/s':>8} {'Feat dim':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

    for t in all_timings:
        train_s = t["splits"].get("train", {}).get("time_s", "-")
        test_s  = t["splits"].get("test",  {}).get("time_s", "-")
        print(
            f"  {t['model']:<20} "
            f"{t['load_time_s']:>10.1f} "
            f"{str(train_s):>10} "
            f"{str(test_s):>10} "
            f"{t['total_time_s']:>10.1f} "
            f"{t['total_imgs_per_s']:>8.1f} "
            f"{t['feature_dim']:>10}"
        )

    # Mejor velocidad
    fastest = max(all_timings, key=lambda x: x["total_imgs_per_s"])
    slowest = min(all_timings, key=lambda x: x["total_imgs_per_s"])
    print(f"\n  Mas rapido: {fastest['model']} ({fastest['total_imgs_per_s']:.1f} imgs/s)")
    print(f"  Mas lento:  {slowest['model']} ({slowest['total_imgs_per_s']:.1f} imgs/s)")
    if slowest["total_imgs_per_s"] > 0:
        ratio = fastest["total_imgs_per_s"] / slowest["total_imgs_per_s"]
        print(f"  Diferencia: {ratio:.1f}x mas rapido")
    print(f"{'='*70}")


def main():
    args = parse_args()

    # Device
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device   = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("\nSin GPU, usando CPU (mas lento)")

    print(f"Dataset: {args.dataset}")
    print(f"Output:  {args.output}")
    print(f"Batch:   {args.batch_size}  |  Workers: {args.num_workers}")

    args.output.mkdir(parents=True, exist_ok=True)

    models_to_run = ALL_MODELS if args.model == "all" else [args.model]

    t_global_start = time.time()
    all_timings    = []

    for model_name in models_to_run:
        timing = run_extraction(
            dataset_root=args.dataset,
            output_dir=args.output,
            model_name=model_name,
            batch_size=args.batch_size,
            device=device,
            num_workers=args.num_workers,
        )
        all_timings.append(timing)

    t_global = time.time() - t_global_start

    # Reporte comparativo solo si se han corrido varios modelos
    if len(all_timings) > 1:
        print_timing_report(all_timings, device)

    # Guardar timing report en JSON
    timing_report = {
        "device":         str(device),
        "batch_size":     args.batch_size,
        "num_workers":    args.num_workers,
        "total_time_s":   round(t_global, 2),
        "models":         all_timings,
    }
    report_path = args.output / "timing_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(timing_report, f, indent=2, ensure_ascii=False)
    print(f"\nTiming report guardado: {report_path}")
    print(f"Extraccion completada en {t_global:.1f}s")


if __name__ == "__main__":
    main()

"""
RDD2022 — Fine-tuning nivel 2 de DINOv2
=========================================
Descongela las últimas 4 capas del transformer DINOv2 junto con una
cabeza de clasificación multilabel (D00, D10, D20) y los entrena
conjuntamente con learning rate diferencial.

Nivel 2 significa:
  - Capas 0-7 de DINOv2: CONGELADAS (no aprenden)
  - Capas 8-11 de DINOv2: DESCONGELADAS (aprenden con lr muy pequeño)
  - Cabeza clasificadora: DESCONGELADA (aprende con lr normal)

Por qué learning rate diferencial:
  Las capas preentrenadas de DINO contienen conocimiento visual general
  que no queremos destruir. Un lr muy pequeño (1e-5) las ajusta suavemente
  al dominio de grietas sin perder ese conocimiento. La cabeza clasificadora
  empieza desde cero y necesita un lr mayor (1e-3) para converger.

Uso:
  python finetune_dinov2.py --dataset C:/RDD2022_clean --output C:/models/dinov2-finetuned

  # Con parámetros personalizados
  python finetune_dinov2.py --dataset C:/RDD2022_clean --output C:/models/dinov2-finetuned
      --epochs 30 --batch-size 16 --lr-dino 1e-5 --lr-head 1e-3

  # Si hay OOM, reducir batch
  python finetune_dinov2.py --dataset C:/RDD2022_clean --output C:/models/dinov2-finetuned
      --batch-size 8

Dependencias:
  pip install torch torchvision pillow tqdm numpy

Salida:
  models/dinov2-finetuned/
    best_model.pt          ← mejor checkpoint según val F1
    final_model.pt         ← modelo al final del entrenamiento
    training_report.json   ← métricas por época
    frozen_layers.txt      ← qué capas están congeladas y cuáles no
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────

CLASSES       = ["D00", "D10", "D20"]
NUM_CLASSES   = len(CLASSES)
IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
SEED          = 42

VALID_COUNTRIES = [
    "China_MotorBike", "Czech", "India",
    "Japan", "Norway", "United_States",
]

VAL_SPLIT  = 0.15
TEST_SPLIT = 0.15


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class RDDFineTuneDataset(Dataset):
    """
    Dataset que carga imágenes y sus etiquetas multilabel directamente
    desde disco para el fine-tuning end-to-end.
    A diferencia de extract_features.py, aquí no guardamos features:
    cada imagen pasa por DINO en cada época.
    """

    def __init__(self, samples: list, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            img = Image.open(sample["image_path"]).convert("RGB")
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE)

        label = torch.tensor(sample["label"], dtype=torch.float32)
        return img, label


def scan_dataset(dataset_root: Path) -> list:
    """
    Recorre el dataset y devuelve lista de dicts con:
      image_path, label (vector [D00, D10, D20])
    Solo usa el split train (que tiene anotaciones).
    """
    samples = []

    for country in VALID_COUNTRIES:
        images_dir = dataset_root / country / "train" / "images"
        xml_dir    = dataset_root / country / "train" / "annotations" / "xmls"

        if not images_dir.exists():
            continue

        for img_path in sorted(images_dir.glob("*.[jJpP][pPnN][gG]*")):
            xml_path = xml_dir / (img_path.stem + ".xml")
            label    = [0.0, 0.0, 0.0]

            if xml_path.exists():
                try:
                    root = ET.parse(xml_path).getroot()
                    for obj in root.findall("object"):
                        name_el = obj.find("name")
                        if name_el is not None:
                            cls = name_el.text.strip()
                            if cls in CLASSES:
                                label[CLASSES.index(cls)] = 1.0
                except Exception:
                    pass

            # Solo incluir imágenes con al menos una clase válida
            if sum(label) > 0:
                samples.append({
                    "image_path": str(img_path),
                    "label":      label,
                    "country":    country,
                })

    return samples


def make_splits(samples: list) -> tuple:
    """Split reproducible 70/15/15."""
    torch.manual_seed(SEED)
    n       = len(samples)
    n_test  = int(n * TEST_SPLIT)
    n_val   = int(n * VAL_SPLIT)
    n_train = n - n_val - n_test

    perm   = torch.randperm(n).tolist()
    tr_idx   = perm[:n_train]
    val_idx  = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    return (
        [samples[i] for i in tr_idx],
        [samples[i] for i in val_idx],
        [samples[i] for i in test_idx],
    )


def get_transforms(augment: bool = True):
    """
    Train: augmentations para mejorar generalización.
    Val/Test: solo resize y normalización.
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ─────────────────────────────────────────────
# Modelo
# ─────────────────────────────────────────────

class DINOv2FineTuned(nn.Module):
    """
    DINOv2 con cabeza de clasificación multilabel.
    Las primeras 8 capas están congeladas.
    Las últimas 4 capas (8-11) y la cabeza están descongeladas.

    Arquitectura:
      DINOv2 ViT-B/14 (12 capas transformer)
        └── Capas 0-7:  CONGELADAS
        └── Capas 8-11: DESCONGELADAS (lr=1e-5)
      Cabeza clasificadora:
        └── LayerNorm → Linear(768, 256) → GELU → Dropout → Linear(256, 3)
        └── DESCONGELADA (lr=1e-3)
    """

    def __init__(self, n_unfrozen_blocks: int = 4, dropout: float = 0.3):
        super().__init__()

        print("  Cargando DINOv2 (ViT-B/14)...")
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            pretrained=True,
            verbose=False,
        )

        # Congelar todo el backbone primero
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Descongelar las últimas n_unfrozen_blocks capas
        n_blocks = len(self.backbone.blocks)
        for i in range(n_blocks - n_unfrozen_blocks, n_blocks):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True

        # Descongelar también la norm final del backbone
        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

        # Cabeza de clasificación multilabel
        hidden_size = self.backbone.embed_dim  # 768 para ViT-B
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, NUM_CLASSES),
            # Sin sigmoid: usamos BCEWithLogitsLoss (más estable)
        )

        self._log_trainable_params(n_unfrozen_blocks, n_blocks)

    def _log_trainable_params(self, n_unfrozen: int, n_total: int):
        frozen    = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        unfrozen  = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        head_p    = sum(p.numel() for p in self.head.parameters())

        print(f"  Capas congeladas:    bloques 0-{n_total - n_unfrozen - 1} ({frozen:,} params)")
        print(f"  Capas descongeladas: bloques {n_total - n_unfrozen}-{n_total - 1} ({unfrozen:,} params)")
        print(f"  Cabeza clasificadora: {head_p:,} params")
        print(f"  Total entrenable:    {unfrozen + head_p:,} params")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extraer CLS token del backbone
        features = self.backbone.forward_features(x)
        cls_token = features["x_norm_clstoken"]  # (B, 768)
        logits    = self.head(cls_token)          # (B, 3)
        return logits

    def get_backbone_params(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def get_head_params(self):
        return list(self.head.parameters())


# ─────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor,
                    threshold: float = 0.5) -> dict:
    preds_bin = (torch.sigmoid(preds) > threshold).float()
    metrics   = {}
    f1s       = []

    for i, cls in enumerate(CLASSES):
        tp = ((preds_bin[:, i] == 1) & (targets[:, i] == 1)).sum().item()
        fp = ((preds_bin[:, i] == 1) & (targets[:, i] == 0)).sum().item()
        fn = ((preds_bin[:, i] == 0) & (targets[:, i] == 1)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        metrics[cls] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
        }
        f1s.append(f1)

    metrics["macro_f1"]       = round(float(np.mean(f1s)), 4)
    metrics["exact_accuracy"] = round(
        (preds_bin == targets).all(dim=1).float().mean().item(), 4
    )
    return metrics


# ─────────────────────────────────────────────
# Entrenamiento
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device) -> dict:
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping para estabilidad
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        all_preds.append(logits.detach().cpu())
        all_targets.append(labels.cpu())

    avg_loss = total_loss / len(loader.dataset)
    metrics  = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
    metrics["loss"] = round(avg_loss, 4)
    return metrics


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, labels in tqdm(loader, desc="  Eval ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        all_preds.append(logits.cpu())
        all_targets.append(labels.cpu())

    avg_loss = total_loss / len(loader.dataset)
    metrics  = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
    metrics["loss"] = round(avg_loss, 4)
    return metrics


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────

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
        print("\nSin GPU detectada, usando CPU (sera lento)")

    print(f"Dataset: {args.dataset}")
    print(f"Output:  {args.output}")
    print(f"Epocas:  {args.epochs} | Batch: {args.batch_size}")

    # ── Cargar dataset ────────────────────────────────────────────────
    print("\nEscaneando dataset...")
    all_samples = scan_dataset(args.dataset)
    print(f"  {len(all_samples)} imagenes con anotaciones validas")

    tr_samples, val_samples, te_samples = make_splits(all_samples)
    print(f"  Split → train: {len(tr_samples)} | val: {len(val_samples)} | test: {len(te_samples)}")

    # ── DataLoaders ───────────────────────────────────────────────────
    tr_dataset  = RDDFineTuneDataset(tr_samples,  get_transforms(augment=True))
    val_dataset = RDDFineTuneDataset(val_samples, get_transforms(augment=False))
    te_dataset  = RDDFineTuneDataset(te_samples,  get_transforms(augment=False))

    tr_loader  = DataLoader(tr_dataset,  batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type=="cuda"))
    te_loader  = DataLoader(te_dataset,  batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type=="cuda"))

    # ── Modelo ────────────────────────────────────────────────────────
    print("\nCreando modelo...")
    model = DINOv2FineTuned(
        n_unfrozen_blocks=args.n_unfrozen_blocks,
        dropout=args.dropout,
    ).to(device)

    # ── Loss con pos_weight para desbalance ───────────────────────────
    labels_matrix = torch.tensor([s["label"] for s in tr_samples])
    pos_counts    = labels_matrix.sum(dim=0).clamp(min=1)
    neg_counts    = len(tr_samples) - pos_counts
    pos_weight    = (neg_counts / pos_counts).to(device)
    criterion     = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"\nPos weights: " + " | ".join(
        f"{cls}={w:.2f}" for cls, w in zip(CLASSES, pos_weight.tolist())
    ))

    # ── Optimizador con learning rate diferencial ─────────────────────
    optimizer = optim.AdamW([
        {"params": model.get_backbone_params(), "lr": args.lr_dino,   "weight_decay": 1e-4},
        {"params": model.get_head_params(),     "lr": args.lr_head,   "weight_decay": 1e-4},
    ])

    # Scheduler: cosine annealing sobre el lr de la cabeza
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_head * 0.01
    )

    # ── Entrenamiento ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Fine-tuning DINOv2 — {args.epochs} epocas")
    print(f"  LR backbone: {args.lr_dino} | LR cabeza: {args.lr_head}")
    print(f"{'='*60}")

    args.output.mkdir(parents=True, exist_ok=True)

    history      = []
    best_val_f1  = 0.0
    best_epoch   = 0

    for epoch in range(1, args.epochs + 1):
        tr_metrics  = train_epoch(model, tr_loader, optimizer, criterion, device)
        val_metrics = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        lr_dino = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]

        print(
            f"  Ep {epoch:3d}/{args.epochs} | "
            f"loss train={tr_metrics['loss']:.4f} val={val_metrics['loss']:.4f} | "
            f"F1 train={tr_metrics['macro_f1']:.4f} val={val_metrics['macro_f1']:.4f} | "
            f"lr_dino={lr_dino:.1e} lr_head={lr_head:.1e}"
        )

        epoch_log = {
            "epoch":       epoch,
            "train_loss":  tr_metrics["loss"],
            "val_loss":    val_metrics["loss"],
            "train_f1":    tr_metrics["macro_f1"],
            "val_f1":      val_metrics["macro_f1"],
            "val_metrics": val_metrics,
            "lr_dino":     lr_dino,
            "lr_head":     lr_head,
        }
        history.append(epoch_log)

        # Guardar mejor modelo según val F1
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch  = epoch
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_f1":           best_val_f1,
                "args":             vars(args),
            }, args.output / "best_model.pt")
            print(f"  ✓ Nuevo mejor modelo guardado (val F1={best_val_f1:.4f})")

    # Guardar modelo final
    torch.save({
        "epoch":            args.epochs,
        "model_state_dict": model.state_dict(),
        "val_f1":           val_metrics["macro_f1"],
        "args":             vars(args),
    }, args.output / "final_model.pt")

    # ── Evaluación final en test ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Cargando mejor modelo (epoca {best_epoch}, val F1={best_val_f1:.4f})")
    checkpoint = torch.load(args.output / "best_model.pt", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("  Evaluacion en TEST...")
    test_metrics = eval_epoch(model, te_loader, criterion, device)

    print(f"\n  Test Macro F1:    {test_metrics['macro_f1']:.4f}")
    print(f"  Exact accuracy:   {test_metrics['exact_accuracy']:.4f}")
    print("\n  Por clase:")
    for cls in CLASSES:
        m = test_metrics[cls]
        print(f"    {cls} → P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")

    # ── Guardar reporte ───────────────────────────────────────────────
    report = {
        "args":         {**vars(args), "dataset": str(args.dataset), "output": str(args.output)},
        "best_epoch":   best_epoch,
        "best_val_f1":  best_val_f1,
        "history":      history,
        "test_metrics": test_metrics,
    }
    report_path = args.output / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Reporte guardado: {report_path}")

    # ── Info para el siguiente paso ───────────────────────────────────
    print(f"\n{'='*60}")
    print("  SIGUIENTE PASO: extraer features con el modelo fine-tuneado")
    print(f"  El modelo esta en: {args.output / 'best_model.pt'}")
    print(f"  Usa extract_features_finetuned.py para extraer las features")
    print(f"{'='*60}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tuning nivel 2 de DINOv2 para clasificacion de grietas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Configuracion recomendada para RTX 4060 (8GB)
  python finetune_dinov2.py --dataset C:/RDD2022_clean --output C:/models/dinov2-finetuned

  # Si hay OOM, reducir batch
  python finetune_dinov2.py --dataset C:/RDD2022_clean --output C:/models/dinov2-finetuned --batch-size 8

  # Mas epocas para convergencia mas solida
  python finetune_dinov2.py --dataset C:/RDD2022_clean --output C:/models/dinov2-finetuned --epochs 40
        """
    )
    parser.add_argument("--dataset",          type=Path,  required=True,
                        help="Ruta al dataset limpio RDD2022_clean")
    parser.add_argument("--output",           type=Path,  required=True,
                        help="Carpeta donde guardar el modelo y el reporte")
    parser.add_argument("--epochs",           type=int,   default=30,
                        help="Numero de epocas (default: 30)")
    parser.add_argument("--batch-size",       type=int,   default=16,
                        help="Batch size (default: 16, reducir a 8 si hay OOM)")
    parser.add_argument("--lr-dino",          type=float, default=1e-5,
                        help="LR para capas descongeladas de DINOv2 (default: 1e-5)")
    parser.add_argument("--lr-head",          type=float, default=1e-3,
                        help="LR para la cabeza clasificadora (default: 1e-3)")
    parser.add_argument("--n-unfrozen-blocks",type=int,   default=4,
                        help="Numero de bloques a descongelar desde el final (default: 4)")
    parser.add_argument("--dropout",          type=float, default=0.3,
                        help="Dropout en la cabeza clasificadora (default: 0.3)")
    parser.add_argument("--num-workers",      type=int,   default=4,
                        help="Workers DataLoader (default: 4, usar 0 en Windows si hay errores)")
    parser.add_argument("--cpu",              action="store_true",
                        help="Forzar CPU aunque haya GPU")
    return parser.parse_args()


if __name__ == "__main__":
    main()

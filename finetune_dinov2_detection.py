"""
RDD2022 — Fine-tuning de detección sobre DINOv2 (estilo DETR)
==============================================================
Entrena una detection head tipo DETR sobre DINOv2 con los bloques
10-11 descongelados. Predice bounding boxes y clases directamente
desde los patch tokens del transformer.

Arquitectura:
  DINOv2 backbone (bloques 0-9 congelados, 10-11 descongelados)
    → patch tokens (196, 768)
    → Transformer decoder con N_QUERIES queries aprendibles
    → Para cada query: clase (D00/D10/D20/fondo) + bbox (cx, cy, w, h)
    → Hungarian matching loss durante entrenamiento

Métricas:
  mAP (mean Average Precision) con IoU threshold 0.5
  AP por clase (D00, D10, D20)
  Precision y Recall por clase

Tiempo estimado RTX 4060:
  ~3-6 horas para 50 épocas con batch size 8

Uso:
  python finetune_dinov2_detection.py
      --dataset C:/RDD2022_clean
      --output  C:/models/dinov2-detection

  # Batch más pequeño si hay OOM
  python finetune_dinov2_detection.py
      --dataset C:/RDD2022_clean
      --output  C:/models/dinov2-detection
      --batch-size 4

Dependencias:
  pip install torch torchvision pillow tqdm numpy scipy

Salida:
  models/dinov2-detection/
    best_model.pt          ← mejor checkpoint según val mAP
    final_model.pt         ← modelo última época
    training_report.json   ← métricas por época
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────

CLASSES         = ["D00", "D10", "D20"]
NUM_CLASSES     = len(CLASSES)          # sin contar fondo
BG_CLASS        = NUM_CLASSES           # índice de clase fondo = 3
N_QUERIES       = 20                    # nº de detecciones posibles por imagen
IMG_SIZE        = 224
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]
SEED            = 42
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15

VALID_COUNTRIES = [
    "China_MotorBike", "Czech", "India",
    "Japan", "Norway", "United_States",
]

# Pesos de la loss
LAMBDA_CLASS = 1.0   # peso de la loss de clasificación
LAMBDA_BBOX  = 5.0   # peso de la loss L1 de bounding box
LAMBDA_GIOU  = 2.0   # peso de la loss GIoU


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class RDDDetectionDataset(Dataset):
    """
    Dataset que carga imágenes y sus bounding boxes para detección.
    Devuelve (imagen, targets) donde targets es un dict con:
      - boxes:  tensor (N, 4) en formato [cx, cy, w, h] normalizado [0,1]
      - labels: tensor (N,) con índices de clase 0=D00, 1=D10, 2=D20
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
            w0, h0 = img.size
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            w0, h0 = IMG_SIZE, IMG_SIZE

        # Normalizar bounding boxes a [0,1] y convertir a cx,cy,w,h
        boxes  = []
        labels = []
        for box in sample["boxes"]:
            cls = box["class"]
            if cls not in CLASSES:
                continue
            xmin = box["xmin"] / w0
            ymin = box["ymin"] / h0
            xmax = box["xmax"] / w0
            ymax = box["ymax"] / h0
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            bw = xmax - xmin
            bh = ymax - ymin
            # Clamp para evitar valores fuera de [0,1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            bw = max(0.01, min(1, bw))
            bh = max(0.01, min(1, bh))
            boxes.append([cx, cy, bw, bh])
            labels.append(CLASSES.index(cls))

        if boxes:
            boxes  = torch.tensor(boxes,  dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.long)

        return img, {"boxes": boxes, "labels": labels}


def collate_fn(batch):
    """Collate que maneja targets de distinto número de boxes por imagen."""
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs)
    return imgs, list(targets)


def scan_dataset_detection(dataset_root: Path) -> list:
    """Escanea el dataset y devuelve muestras con bounding boxes."""
    samples = []

    for country in VALID_COUNTRIES:
        images_dir = dataset_root / country / "train" / "images"
        xml_dir    = dataset_root / country / "train" / "annotations" / "xmls"

        if not images_dir.exists():
            continue

        for img_path in sorted(images_dir.glob("*.[jJpP][pPnN][gG]*")):
            xml_path = xml_dir / (img_path.stem + ".xml")
            boxes    = []

            if xml_path.exists():
                try:
                    root = ET.parse(xml_path).getroot()
                    for obj in root.findall("object"):
                        name_el = obj.find("name")
                        bnd     = obj.find("bndbox")
                        if name_el is None or bnd is None:
                            continue
                        cls = name_el.text.strip()
                        if cls not in CLASSES:
                            continue
                        boxes.append({
                            "class": cls,
                            "xmin":  int(float(bnd.find("xmin").text)),
                            "ymin":  int(float(bnd.find("ymin").text)),
                            "xmax":  int(float(bnd.find("xmax").text)),
                            "ymax":  int(float(bnd.find("ymax").text)),
                        })
                except Exception:
                    pass

            if boxes:  # solo imágenes con al menos un bbox válido
                samples.append({
                    "image_path": str(img_path),
                    "boxes":      boxes,
                    "country":    country,
                })

    return samples


def make_splits(samples: list) -> tuple:
    torch.manual_seed(SEED)
    n       = len(samples)
    n_test  = int(n * TEST_SPLIT)
    n_val   = int(n * VAL_SPLIT)
    n_train = n - n_val - n_test
    perm    = torch.randperm(n).tolist()
    return (
        [samples[i] for i in perm[:n_train]],
        [samples[i] for i in perm[n_train:n_train + n_val]],
        [samples[i] for i in perm[n_train + n_val:]],
    )


def get_transforms(augment: bool = True):
    if augment:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─────────────────────────────────────────────
# Modelo — DINOv2 + Detection Head (DETR-style)
# ─────────────────────────────────────────────

class DINOv2DetectionHead(nn.Module):
    """
    Detection head tipo DETR sobre patch tokens de DINOv2.

    Transformer decoder con N_QUERIES queries aprendibles que
    atienden a los patch tokens del backbone. Cada query predice:
      - Una clase: D00 / D10 / D20 / fondo (4 clases)
      - Un bounding box: (cx, cy, w, h) normalizado en [0,1]
    """

    def __init__(self, d_model: int = 768, n_queries: int = N_QUERIES,
                 n_heads: int = 8, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()

        self.n_queries = n_queries

        # Queries aprendibles — se inicializan aleatoriamente y aprenden
        # a especializarse en detectar distintos tipos de grietas
        self.queries = nn.Embedding(n_queries, d_model)

        # Transformer decoder: las queries atienden a los patch tokens
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Cabeza de clasificación: query → clase
        self.class_head = nn.Linear(d_model, NUM_CLASSES + 1)  # +1 para fondo

        # Cabeza de bounding box: query → (cx, cy, w, h)
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 4),
            nn.Sigmoid(),  # normalizar a [0,1]
        )

    def forward(self, patch_tokens: torch.Tensor):
        """
        patch_tokens: (B, N_patches, d_model) — patch tokens de DINOv2
        Devuelve:
          logits: (B, N_queries, NUM_CLASSES+1)
          boxes:  (B, N_queries, 4) en [0,1]
        """
        B = patch_tokens.size(0)

        # Expandir queries para el batch
        queries = self.queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)

        # Decoder: queries atienden a patch tokens
        decoded = self.decoder(queries, patch_tokens)  # (B, Q, D)

        logits = self.class_head(decoded)   # (B, Q, NUM_CLASSES+1)
        boxes  = self.bbox_head(decoded)    # (B, Q, 4)

        return logits, boxes


class DINOv2Detector(nn.Module):
    """
    Modelo completo: DINOv2 backbone + Detection Head.
    Bloques 0-9 congelados, bloques 10-11 descongelados.
    """

    def __init__(self, n_unfrozen_blocks: int = 2, dropout: float = 0.1):
        super().__init__()

        print("  Cargando DINOv2 (ViT-B/14)...")
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14", pretrained=True, verbose=False,
        )

        # Congelar todo el backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Descongelar últimas n capas
        n_blocks = len(self.backbone.blocks)
        for i in range(n_blocks - n_unfrozen_blocks, n_blocks):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

        d_model = self.backbone.embed_dim  # 768

        self.detection_head = DINOv2DetectionHead(
            d_model=d_model, n_queries=N_QUERIES, dropout=dropout
        )

        frozen   = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        unfrozen = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        head_p   = sum(p.numel() for p in self.detection_head.parameters())

        print(f"  Backbone congelado:    bloques 0-{n_blocks-n_unfrozen_blocks-1} ({frozen:,} params)")
        print(f"  Backbone descongelado: bloques {n_blocks-n_unfrozen_blocks}-{n_blocks-1} ({unfrozen:,} params)")
        print(f"  Detection head:        {head_p:,} params")
        print(f"  Total entrenable:      {unfrozen + head_p:,} params")

    def forward(self, x: torch.Tensor):
        # Extraer patch tokens (sin CLS)
        out         = self.backbone.forward_features(x)
        patch_tokens = out["x_norm_patchtokens"]  # (B, 196, 768)

        logits, boxes = self.detection_head(patch_tokens)
        return logits, boxes

    def get_backbone_params(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def get_head_params(self):
        return list(self.detection_head.parameters())


# ─────────────────────────────────────────────
# Hungarian Matching + Loss
# ─────────────────────────────────────────────

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convierte (cx, cy, w, h) a (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def generalized_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calcula GIoU entre dos conjuntos de boxes en formato xyxy.
    GIoU penaliza más cuando los boxes no se solapan en absoluto.
    """
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter + 1e-6

    iou = inter / union

    # Caja envolvente más grande (para GIoU)
    enc_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enc_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enc_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enc_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1) + 1e-6

    giou = iou - (enc_area - union) / enc_area
    return giou


def hungarian_matching(pred_logits: torch.Tensor, pred_boxes: torch.Tensor,
                        gt_labels: torch.Tensor, gt_boxes: torch.Tensor):
    """
    Asignación óptima Hungarian entre predicciones y ground truth.
    Minimiza: λ_class * (-prob_clase) + λ_bbox * L1 + λ_giou * (1 - GIoU)

    Devuelve (pred_indices, gt_indices) para alinear predicciones con GT.
    """
    n_pred = pred_logits.size(0)
    n_gt   = gt_labels.size(0)

    if n_gt == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    # Probabilidades de clase para cada GT
    probs     = pred_logits.softmax(-1)  # (n_pred, NUM_CLASSES+1)
    class_cost = -probs[:, gt_labels]    # (n_pred, n_gt)

    # L1 entre boxes predichos y GT
    bbox_cost = torch.cdist(pred_boxes, gt_boxes, p=1)  # (n_pred, n_gt)

    # GIoU
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    gt_xyxy   = box_cxcywh_to_xyxy(gt_boxes)
    giou_cost = torch.zeros(n_pred, n_gt, device=pred_logits.device)
    for j in range(n_gt):
        giou_cost[:, j] = -generalized_iou(
            pred_xyxy,
            gt_xyxy[j:j+1].expand(n_pred, -1)
        )

    # Coste total
    cost = (LAMBDA_CLASS * class_cost +
            LAMBDA_BBOX  * bbox_cost  +
            LAMBDA_GIOU  * giou_cost)

    # Resolver asignación óptima
    cost_np  = cost.detach().cpu().numpy()
    pred_idx, gt_idx = linear_sum_assignment(cost_np)

    return (torch.tensor(pred_idx, dtype=torch.long),
            torch.tensor(gt_idx,   dtype=torch.long))


def compute_detection_loss(pred_logits: torch.Tensor, pred_boxes: torch.Tensor,
                            targets: list) -> dict:
    """
    Calcula la loss de detección usando Hungarian matching.

    pred_logits: (B, N_queries, NUM_CLASSES+1)
    pred_boxes:  (B, N_queries, 4)
    targets: lista de dicts con 'boxes' y 'labels'
    """
    B = pred_logits.size(0)
    device = pred_logits.device

    total_class_loss = torch.tensor(0.0, device=device)
    total_bbox_loss  = torch.tensor(0.0, device=device)
    total_giou_loss  = torch.tensor(0.0, device=device)
    n_matched        = 0

    # Labels objetivo para clasificación (por defecto todo fondo)
    target_classes = torch.full(
        (B, N_QUERIES), BG_CLASS, dtype=torch.long, device=device
    )

    for b in range(B):
        gt_boxes  = targets[b]["boxes"].to(device)
        gt_labels = targets[b]["labels"].to(device)

        if len(gt_boxes) == 0:
            continue

        # Hungarian matching para esta imagen
        pred_idx, gt_idx = hungarian_matching(
            pred_logits[b].detach(),
            pred_boxes[b].detach(),
            gt_labels, gt_boxes,
        )

        if len(pred_idx) == 0:
            continue

        pred_idx = pred_idx.to(device)
        gt_idx   = gt_idx.to(device)

        # Asignar clases GT a las queries matched
        target_classes[b, pred_idx] = gt_labels[gt_idx]

        # Loss de bounding box (solo queries matched)
        matched_pred_boxes = pred_boxes[b, pred_idx]
        matched_gt_boxes   = gt_boxes[gt_idx]

        bbox_loss = F.l1_loss(matched_pred_boxes, matched_gt_boxes, reduction="sum")
        total_bbox_loss += bbox_loss

        # Loss GIoU
        pred_xyxy = box_cxcywh_to_xyxy(matched_pred_boxes)
        gt_xyxy   = box_cxcywh_to_xyxy(matched_gt_boxes)
        giou = generalized_iou(pred_xyxy, gt_xyxy)
        giou_loss = (1 - giou).sum()
        total_giou_loss += giou_loss

        n_matched += len(pred_idx)

    # Loss de clasificación (todas las queries, matched y no matched)
    class_loss = F.cross_entropy(
        pred_logits.view(-1, NUM_CLASSES + 1),
        target_classes.view(-1),
        weight=torch.tensor(
            [1.0] * NUM_CLASSES + [0.1],  # peso menor para clase fondo
            device=device
        ),
    )
    total_class_loss = class_loss * B

    # Normalizar por número de matches
    norm = max(n_matched, 1)
    total_loss = (LAMBDA_CLASS * total_class_loss +
                  LAMBDA_BBOX  * total_bbox_loss  / norm +
                  LAMBDA_GIOU  * total_giou_loss  / norm)

    return {
        "loss":       total_loss,
        "class_loss": total_class_loss.item(),
        "bbox_loss":  (total_bbox_loss / norm).item(),
        "giou_loss":  (total_giou_loss / norm).item(),
        "n_matched":  n_matched,
    }


# ─────────────────────────────────────────────
# Métricas — mAP
# ─────────────────────────────────────────────

def compute_map(pred_logits: torch.Tensor, pred_boxes: torch.Tensor,
                targets: list, iou_threshold: float = 0.5) -> dict:
    """
    Calcula mAP@0.5 por clase.
    Solo cuenta predicciones con confianza > 0.3 y clase != fondo.
    """
    all_preds  = {cls: [] for cls in CLASSES}  # (score, tp/fp)
    all_n_gt   = {cls: 0  for cls in CLASSES}

    probs = pred_logits.softmax(-1)  # (B, Q, NUM_CLASSES+1)
    B     = pred_logits.size(0)

    for b in range(B):
        gt_boxes    = targets[b]["boxes"]
        gt_labels   = targets[b]["labels"]
        gt_matched  = [False] * len(gt_boxes)

        for q in range(N_QUERIES):
            cls_probs  = probs[b, q, :NUM_CLASSES]
            pred_class = cls_probs.argmax().item()
            score      = cls_probs[pred_class].item()

            if score < 0.3:  # umbral de confianza
                continue

            cls_name = CLASSES[pred_class]
            pred_box = pred_boxes[b, q].unsqueeze(0)

            # Buscar GT del mismo clase y mayor IoU
            best_iou = 0.0
            best_j   = -1
            for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_label.item() != pred_class:
                    continue
                if gt_matched[j]:
                    continue
                pred_xyxy = box_cxcywh_to_xyxy(pred_box)
                gt_xyxy   = box_cxcywh_to_xyxy(gt_box.unsqueeze(0))
                iou = generalized_iou(pred_xyxy, gt_xyxy)[0].item()
                if iou > best_iou:
                    best_iou = iou
                    best_j   = j

            tp = 1 if best_iou >= iou_threshold and best_j >= 0 else 0
            if tp and best_j >= 0:
                gt_matched[best_j] = True

            all_preds[cls_name].append((score, tp))

        # Contar GT por clase
        for gt_label in gt_labels:
            all_n_gt[CLASSES[gt_label.item()]] += 1

    # Calcular AP por clase
    ap_per_class = {}
    for cls in CLASSES:
        preds  = sorted(all_preds[cls], key=lambda x: -x[0])
        n_gt   = all_n_gt[cls]
        if n_gt == 0 or not preds:
            ap_per_class[cls] = 0.0
            continue

        tp_cumsum = np.cumsum([p[1] for p in preds])
        precision = tp_cumsum / (np.arange(len(preds)) + 1)
        recall    = tp_cumsum / n_gt

        # AP como área bajo la curva precision-recall
        ap = 0.0
        for i in range(len(precision) - 1):
            ap += (recall[i+1] - recall[i]) * precision[i+1]
        ap_per_class[cls] = float(ap)

    map_score = float(np.mean(list(ap_per_class.values())))
    return {"mAP": round(map_score, 4), **{f"AP_{cls}": round(v, 4) for cls, v in ap_per_class.items()}}


# ─────────────────────────────────────────────
# Entrenamiento
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device) -> dict:
    model.train()
    total_loss = total_class = total_bbox = total_giou = 0.0
    n_batches  = 0

    for imgs, targets in tqdm(loader, desc="  Train", leave=False):
        imgs = imgs.to(device)
        optimizer.zero_grad()

        pred_logits, pred_boxes = model(imgs)
        loss_dict = compute_detection_loss(pred_logits, pred_boxes, targets)

        loss_dict["loss"].backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=0.1
        )
        optimizer.step()

        total_loss  += loss_dict["loss"].item()
        total_class += loss_dict["class_loss"]
        total_bbox  += loss_dict["bbox_loss"]
        total_giou  += loss_dict["giou_loss"]
        n_batches   += 1

    n = max(n_batches, 1)
    return {
        "loss":       round(total_loss  / n, 4),
        "class_loss": round(total_class / n, 4),
        "bbox_loss":  round(total_bbox  / n, 4),
        "giou_loss":  round(total_giou  / n, 4),
    }


@torch.no_grad()
def eval_epoch(model, loader, device) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    all_logits, all_boxes, all_targets = [], [], []

    for imgs, targets in tqdm(loader, desc="  Eval ", leave=False):
        imgs = imgs.to(device)
        pred_logits, pred_boxes = model(imgs)

        loss_dict   = compute_detection_loss(pred_logits, pred_boxes, targets)
        total_loss += loss_dict["loss"].item()
        n_batches  += 1

        all_logits.append(pred_logits.cpu())
        all_boxes.append(pred_boxes.cpu())
        all_targets.extend(targets)

    avg_loss = round(total_loss / max(n_batches, 1), 4)

    # Calcular mAP sobre todo el split
    all_logits_cat = torch.cat(all_logits, dim=0)
    all_boxes_cat  = torch.cat(all_boxes,  dim=0)
    map_metrics    = compute_map(all_logits_cat, all_boxes_cat, all_targets)

    return {"loss": avg_loss, **map_metrics}


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
        print("\nSin GPU, usando CPU (muy lento para deteccion)")

    print(f"Dataset: {args.dataset}")
    print(f"Output:  {args.output}")
    print(f"Epocas:  {args.epochs} | Batch: {args.batch_size}")

    # ── Dataset ───────────────────────────────────────────────────────
    print("\nEscaneando dataset...")
    all_samples = scan_dataset_detection(args.dataset)
    print(f"  {len(all_samples)} imagenes con bounding boxes validos")

    tr_samples, val_samples, te_samples = make_splits(all_samples)
    print(f"  Split → train: {len(tr_samples)} | val: {len(val_samples)} | test: {len(te_samples)}")

    # Estadísticas de boxes por clase
    class_counts = {cls: 0 for cls in CLASSES}
    for s in tr_samples:
        for b in s["boxes"]:
            if b["class"] in CLASSES:
                class_counts[b["class"]] += 1
    print(f"  Boxes en train: " + " | ".join(f"{cls}={n}" for cls, n in class_counts.items()))

    tr_dataset  = RDDDetectionDataset(tr_samples,  get_transforms(augment=True))
    val_dataset = RDDDetectionDataset(val_samples, get_transforms(augment=False))
    te_dataset  = RDDDetectionDataset(te_samples,  get_transforms(augment=False))

    tr_loader  = DataLoader(tr_dataset,  batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate_fn,
                            pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn,
                            pin_memory=(device.type == "cuda"))
    te_loader  = DataLoader(te_dataset,  batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn,
                            pin_memory=(device.type == "cuda"))

    # ── Modelo ────────────────────────────────────────────────────────
    print("\nCreando modelo...")
    model = DINOv2Detector(
        n_unfrozen_blocks=args.n_unfrozen_blocks,
        dropout=args.dropout,
    ).to(device)

    # ── Optimizador ───────────────────────────────────────────────────
    optimizer = optim.AdamW([
        {"params": model.get_backbone_params(), "lr": args.lr_dino,  "weight_decay": 1e-4},
        {"params": model.get_head_params(),     "lr": args.lr_head,  "weight_decay": 1e-4},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_head * 0.01
    )

    # ── Entrenamiento ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Fine-tuning deteccion DINOv2 — {args.epochs} epocas")
    print(f"  LR backbone: {args.lr_dino} | LR head: {args.lr_head}")
    print(f"  N queries: {N_QUERIES} | Lambda bbox: {LAMBDA_BBOX} | Lambda giou: {LAMBDA_GIOU}")
    print(f"{'='*60}")

    args.output.mkdir(parents=True, exist_ok=True)

    history      = []
    best_map     = 0.0
    best_epoch   = 0

    for epoch in range(1, args.epochs + 1):
        tr_metrics  = train_epoch(model, tr_loader, optimizer, device)
        val_metrics = eval_epoch(model, val_loader, device)
        scheduler.step()

        lr_dino = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]

        print(
            f"  Ep {epoch:3d}/{args.epochs} | "
            f"loss train={tr_metrics['loss']:.4f} val={val_metrics['loss']:.4f} | "
            f"mAP={val_metrics['mAP']:.4f} "
            f"(D00={val_metrics['AP_D00']:.3f} D10={val_metrics['AP_D10']:.3f} D20={val_metrics['AP_D20']:.3f})"
        )

        epoch_log = {
            "epoch":      epoch,
            "train_loss": tr_metrics["loss"],
            "val_loss":   val_metrics["loss"],
            "val_mAP":    val_metrics["mAP"],
            "val_AP_D00": val_metrics["AP_D00"],
            "val_AP_D10": val_metrics["AP_D10"],
            "val_AP_D20": val_metrics["AP_D20"],
            "lr_dino":    lr_dino,
            "lr_head":    lr_head,
        }
        history.append(epoch_log)

        # Guardar mejor modelo según val mAP
        if val_metrics["mAP"] > best_map:
            best_map   = val_metrics["mAP"]
            best_epoch = epoch
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_map":          best_map,
                "args":             {k: str(v) if isinstance(v, Path) else v
                                     for k, v in vars(args).items()},
            }, args.output / "best_model.pt")
            print(f"  ✓ Mejor modelo guardado (mAP={best_map:.4f})")

    # Guardar modelo final
    torch.save({
        "epoch":            args.epochs,
        "model_state_dict": model.state_dict(),
        "val_map":          val_metrics["mAP"],
        "args":             {k: str(v) if isinstance(v, Path) else v
                             for k, v in vars(args).items()},
    }, args.output / "final_model.pt")

    # ── Evaluación en test ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Cargando mejor modelo (epoca {best_epoch}, mAP={best_map:.4f})")
    checkpoint = torch.load(args.output / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("  Evaluacion en TEST...")
    test_metrics = eval_epoch(model, te_loader, device)

    print(f"\n  Test mAP@0.5:  {test_metrics['mAP']:.4f}")
    print(f"  AP D00:        {test_metrics['AP_D00']:.4f}")
    print(f"  AP D10:        {test_metrics['AP_D10']:.4f}")
    print(f"  AP D20:        {test_metrics['AP_D20']:.4f}")

    # ── Guardar reporte ───────────────────────────────────────────────
    report = {
        "task":         "detection",
        "best_epoch":   best_epoch,
        "best_val_mAP": best_map,
        "history":      history,
        "test_metrics": test_metrics,
        "args":         {k: str(v) if isinstance(v, Path) else v
                         for k, v in vars(args).items()},
    }
    report_path = args.output / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Reporte guardado: {report_path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tuning de deteccion sobre DINOv2 para RDD2022",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Configuracion recomendada RTX 4060
  python finetune_dinov2_detection.py --dataset C:/RDD2022_clean --output C:/models/dinov2-detection

  # Si hay OOM reducir batch
  python finetune_dinov2_detection.py --dataset C:/RDD2022_clean --output C:/models/dinov2-detection --batch-size 4

Tiempo estimado RTX 4060:
  batch=8,  50 epocas → ~3-4 horas
  batch=4,  50 epocas → ~5-6 horas
        """
    )
    parser.add_argument("--dataset",           type=Path,  required=True)
    parser.add_argument("--output",            type=Path,  required=True)
    parser.add_argument("--epochs",            type=int,   default=50)
    parser.add_argument("--batch-size",        type=int,   default=8,
                        help="Batch size (default: 8, reducir a 4 si hay OOM)")
    parser.add_argument("--lr-dino",           type=float, default=1e-5)
    parser.add_argument("--lr-head",           type=float, default=1e-4,
                        help="LR detection head (mas bajo que clasificacion, default: 1e-4)")
    parser.add_argument("--n-unfrozen-blocks", type=int,   default=2)
    parser.add_argument("--dropout",           type=float, default=0.1)
    parser.add_argument("--num-workers",       type=int,   default=4)
    parser.add_argument("--cpu",               action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

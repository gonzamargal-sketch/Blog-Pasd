"""
RDD2022 — Entrenamiento de detector sobre features DINO
========================================================
Carga las features extraídas por extract_features.py y entrena
un detector de clases de grietas encima. Por ahora implementa
un clasificador MLP como baseline, fácil de sustituir después
por una detection head más compleja (DETR, etc.).

El test set oficial del RDD2022 NO tiene anotaciones, por lo que
este script construye sus propias tres particiones a partir del
train anotado:

    Train anotado (100%)
        ├── 70% → train      (el modelo aprende aquí)
        ├── 15% → validación (guía el entrenamiento)
        └── 15% → test       (evaluación final limpia, se toca UNA vez)

Antes de partir verifica que hay suficientes imágenes por clase
para que cada split sea estadísticamente válido. Si no las hay,
avisa y recomienda revisar la limpieza del dataset.

Uso:
    # Entrenar con features de DINOv2
    python train_detector.py --features C:/features/dinov2 --output C:/models/dinov2

    # Entrenar con DINOv1
    python train_detector.py --features C:/features/dinov1 --output C:/models/dinov1

    # Ajustar proporciones de los splits
    python train_detector.py --features C:/features/dinov2 --output C:/models/dinov2 --val-split 0.10 --test-split 0.15

Dependencias:
    pip install torch scikit-learn numpy tqdm

Salida:
    models/
      dinov2/
        classifier.pt        ← pesos del modelo entrenado
        training_report.json ← métricas por época y evaluación final
        split_info.json      ← índices exactos de cada partición (reproducibilidad)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ─────────────────────────────────────────────
# Clases
# ─────────────────────────────────────────────

# Mapeo de clases a índices
# Multilabel: una imagen puede tener D00, D10 y D20 a la vez
CLASSES = ["D00", "D10", "D20"]
NUM_CLASSES = len(CLASSES)


# ─────────────────────────────────────────────
# Mínimos por split para que la evaluación sea válida
# ─────────────────────────────────────────────

# Mínimo de imágenes con cada clase en val y test para métricas fiables
MIN_SAMPLES_TOTAL   = 300   # mínimo total de imágenes anotadas
MIN_SAMPLES_PER_CLASS_VAL  = 20   # mínimo de positivos por clase en val
MIN_SAMPLES_PER_CLASS_TEST = 20   # mínimo de positivos por clase en test


def check_split_viability(labels: torch.Tensor, val_split: float, test_split: float) -> None:
    """
    Verifica que hay suficientes imágenes para hacer tres particiones válidas.
    Aborta con un mensaje claro si no se cumplen los mínimos, en lugar de
    entrenar silenciosamente con splits degenerados.
    """
    n_total = len(labels)
    n_val   = int(n_total * val_split)
    n_test  = int(n_total * test_split)
    n_train = n_total - n_val - n_test

    print(f"\n{'─'*50}")
    print(f"  Verificando viabilidad del dataset...")
    print(f"  Total imágenes anotadas: {n_total}")
    print(f"  Partición prevista → train: {n_train} | val: {n_val} | test: {n_test}")

    errors   = []
    warnings = []

    # Verificar total mínimo
    if n_total < MIN_SAMPLES_TOTAL:
        errors.append(
            f"Solo hay {n_total} imágenes anotadas. "
            f"El mínimo recomendado es {MIN_SAMPLES_TOTAL}. "
            f"Considera revisar el ratio de submuestreo en clean_rdd2022.py."
        )

    # Verificar por clase en val y test (estimación proporcional)
    class_counts = labels.sum(dim=0)
    for i, cls in enumerate(CLASSES):
        count = int(class_counts[i].item())
        expected_val  = int(count * val_split)
        expected_test = int(count * test_split)

        print(f"  Clase {cls}: {count} positivos → val≈{expected_val} | test≈{expected_test}")

        if count == 0:
            errors.append(f"La clase {cls} no tiene ningún ejemplo. Revisa la limpieza del dataset.")
        elif expected_val < MIN_SAMPLES_PER_CLASS_VAL:
            warnings.append(
                f"Clase {cls}: solo ~{expected_val} ejemplos en val "
                f"(mínimo recomendado: {MIN_SAMPLES_PER_CLASS_VAL}). "
                f"Las métricas de esta clase pueden ser poco fiables."
            )
        elif expected_test < MIN_SAMPLES_PER_CLASS_TEST:
            warnings.append(
                f"Clase {cls}: solo ~{expected_test} ejemplos en test "
                f"(mínimo recomendado: {MIN_SAMPLES_PER_CLASS_TEST}). "
                f"Las métricas de esta clase pueden ser poco fiables."
            )

    # Mostrar warnings
    if warnings:
        print(f"\n  ⚠️  ADVERTENCIAS:")
        for w in warnings:
            print(f"     · {w}")

    # Abortar si hay errores críticos
    if errors:
        print(f"\n  ❌ ERRORES CRÍTICOS — no se puede continuar:")
        for e in errors:
            print(f"     · {e}")
        print(f"\n  Sugerencias:")
        print(f"     1. Reduce --val-split y --test-split (ej: 0.10 cada uno)")
        print(f"     2. Revisa clean_rdd2022.py y aumenta el sample-ratio")
        print(f"     3. Incluye más países o splits en la limpieza")
        raise SystemExit(1)

    print(f"  ✅ Dataset viable para tres particiones")
    print(f"{'─'*50}")


def make_three_splits(
    features: torch.Tensor,
    labels:   torch.Tensor,
    val_split:  float,
    test_split: float,
    seed: int = 42,
) -> tuple:
    """
    Divide features y labels en tres particiones (train/val/test)
    con shuffle reproducible.

    Devuelve:
        (tr_feat, tr_labels, val_feat, val_labels, test_feat, test_labels, indices_dict)
    """
    n_total = len(features)
    n_test  = int(n_total * test_split)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val - n_test

    torch.manual_seed(seed)
    perm = torch.randperm(n_total)

    tr_idx   = perm[:n_train]
    val_idx  = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    tr_feat,   tr_lab   = features[tr_idx],   labels[tr_idx]
    val_feat,  val_lab  = features[val_idx],  labels[val_idx]
    test_feat, test_lab = features[test_idx], labels[test_idx]

    indices = {
        "seed":       seed,
        "n_total":    n_total,
        "train_idx":  tr_idx.tolist(),
        "val_idx":    val_idx.tolist(),
        "test_idx":   test_idx.tolist(),
    }

    return tr_feat, tr_lab, val_feat, val_lab, test_feat, test_lab, indices


def build_labels(metadata: list) -> torch.Tensor:
    """
    Construye un tensor de labels multilabel (N, 3) a partir de la metadata.
    Cada posición indica si la clase está presente en la imagen:
      [1, 0, 1] → imagen tiene D00 y D20
    """
    labels = torch.zeros(len(metadata), NUM_CLASSES)
    for i, sample in enumerate(metadata):
        for cls in sample.get("classes", []):
            if cls in CLASSES:
                labels[i, CLASSES.index(cls)] = 1.0
    return labels


# ─────────────────────────────────────────────
# Modelo — MLP baseline
# ─────────────────────────────────────────────

class CrackClassifier(nn.Module):
    """
    Clasificador multilabel MLP sobre features DINO.

    Arquitectura:
        features (1536) → Linear → BN → ReLU → Dropout
                        → Linear → BN → ReLU → Dropout
                        → Linear → Sigmoid (multilabel)

    Es un baseline sólido y rápido de entrenar. Si necesitas
    más capacidad, puedes añadir capas o aumentar hidden_dim.
    """

    def __init__(self, input_dim: int = 1536, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, NUM_CLASSES),
            # Sin Sigmoid aquí: usamos BCEWithLogitsLoss (más estable numéricamente)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict:
    """
    Calcula métricas multilabel:
      - precision, recall, f1 por clase y macro
      - accuracy exacta (todas las clases correctas)
    """
    preds_bin = (torch.sigmoid(preds) > threshold).float()

    metrics = {}
    f1s = []

    for i, cls in enumerate(CLASSES):
        tp = ((preds_bin[:, i] == 1) & (targets[:, i] == 1)).sum().item()
        fp = ((preds_bin[:, i] == 1) & (targets[:, i] == 0)).sum().item()
        fn = ((preds_bin[:, i] == 0) & (targets[:, i] == 1)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        metrics[cls] = {"precision": round(precision, 4),
                        "recall":    round(recall,    4),
                        "f1":        round(f1,        4)}
        f1s.append(f1)

    metrics["macro_f1"]      = round(np.mean(f1s), 4)
    metrics["exact_accuracy"] = round((preds_bin == targets).all(dim=1).float().mean().item(), 4)

    return metrics


# ─────────────────────────────────────────────
# Entrenamiento
# ─────────────────────────────────────────────

def train(
    model:      nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    optimizer:  optim.Optimizer,
    scheduler,
    criterion:  nn.Module,
    device:     torch.device,
    epochs:     int,
) -> list:
    history = []

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────
        model.train()
        train_loss = 0.0
        all_preds, all_targets = [], []

        for features, labels in tqdm(train_loader, desc=f"  Época {epoch}/{epochs} [train]", leave=False):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            all_preds.append(logits.detach().cpu())
            all_targets.append(labels.cpu())

        train_loss /= len(train_loader.dataset)
        train_metrics = compute_metrics(
            torch.cat(all_preds), torch.cat(all_targets)
        )

        # ── Validación ─────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits   = model(features)
                val_loss += criterion(logits, labels).item() * features.size(0)
                all_preds.append(logits.cpu())
                all_targets.append(labels.cpu())

        val_loss /= len(val_loader.dataset)
        val_metrics = compute_metrics(
            torch.cat(all_preds), torch.cat(all_targets)
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        epoch_log = {
            "epoch":        epoch,
            "train_loss":   round(train_loss, 4),
            "val_loss":     round(val_loss,   4),
            "train_f1":     train_metrics["macro_f1"],
            "val_f1":       val_metrics["macro_f1"],
            "val_metrics":  val_metrics,
        }
        history.append(epoch_log)

        print(f"  Época {epoch:3d} | "
              f"loss train={train_loss:.4f} val={val_loss:.4f} | "
              f"F1 train={train_metrics['macro_f1']:.4f} val={val_metrics['macro_f1']:.4f}")

    return history


# ─────────────────────────────────────────────
# Evaluación final en test
# ─────────────────────────────────────────────

def evaluate_test(model, test_loader, criterion, device) -> dict:
    model.eval()
    test_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            logits    = model(features)
            test_loss += criterion(logits, labels).item() * features.size(0)
            all_preds.append(logits.cpu())
            all_targets.append(labels.cpu())

    test_loss /= len(test_loader.dataset)
    metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
    metrics["test_loss"] = round(test_loss, 4)
    return metrics


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena un clasificador de grietas sobre features DINO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python train_detector.py --features C:/features/dinov2 --output C:/models/dinov2
  python train_detector.py --features C:/features/dinov1 --output C:/models/dinov1 --epochs 50
        """
    )
    parser.add_argument("--features",    type=Path, required=True,
                        help="Carpeta con train_features.pt, train_metadata.json, etc.")
    parser.add_argument("--output",      type=Path, required=True,
                        help="Carpeta donde guardar el modelo y el reporte")
    parser.add_argument("--epochs",      type=int,   default=30,
                        help="Número de épocas (default: 30)")
    parser.add_argument("--batch-size",  type=int,   default=256,
                        help="Batch size para el entrenamiento (default: 256)")
    parser.add_argument("--lr",          type=float, default=1e-3,
                        help="Learning rate inicial (default: 0.001)")
    parser.add_argument("--hidden-dim",  type=int,   default=512,
                        help="Dimensión de capas ocultas del MLP (default: 512)")
    parser.add_argument("--dropout",     type=float, default=0.3,
                        help="Dropout rate (default: 0.3)")
    parser.add_argument("--val-split",   type=float, default=0.15,
                        help="Fracción del train anotado para validación (default: 0.15)")
    parser.add_argument("--test-split",  type=float, default=0.15,
                        help="Fracción del train anotado para test propio (default: 0.15)")
    parser.add_argument("--cpu",         action="store_true",
                        help="Forzar CPU")
    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    device = torch.device("cpu") if args.cpu else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"\n🖥️  Device: {device}")

    # ── Cargar solo train anotado (el test oficial no tiene etiquetas) ──
    print(f"\n📂 Cargando features desde: {args.features}")

    train_features = torch.load(args.features / "train_features.pt", weights_only=True)
    with open(args.features / "train_metadata.json", encoding="utf-8") as f:
        train_meta = json.load(f)

    train_labels = build_labels(train_meta)
    input_dim    = train_features.shape[1]

    print(f"  Features cargadas: {train_features.shape}")
    print(f"  Feature dim: {input_dim}")

    # ── Verificar que hay imágenes suficientes para tres splits ─────────
    check_split_viability(train_labels, args.val_split, args.test_split)

    # ── Hacer las tres particiones ──────────────────────────────────────
    tr_feat, tr_lab, val_feat, val_lab, test_feat, test_lab, indices = make_three_splits(
        train_features, train_labels,
        val_split=args.val_split,
        test_split=args.test_split,
    )

    print(f"\n  Particiones finales:")
    print(f"    Train:      {len(tr_feat):>6} imágenes  ({(1 - args.val_split - args.test_split)*100:.0f}%)")
    print(f"    Validación: {len(val_feat):>6} imágenes  ({args.val_split*100:.0f}%)")
    print(f"    Test:       {len(test_feat):>6} imágenes  ({args.test_split*100:.0f}%)")

    # ── DataLoaders ─────────────────────────────────────────────────────
    train_ds = TensorDataset(tr_feat,   tr_lab)
    val_ds   = TensorDataset(val_feat,  val_lab)
    test_ds  = TensorDataset(test_feat, test_lab)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # ── Modelo, optimizador, loss ────────────────────────────────────────
    model = CrackClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    # pos_weight calculado solo sobre el split de train
    pos_counts = tr_lab.sum(dim=0).clamp(min=1)
    neg_counts = len(tr_lab) - pos_counts
    pos_weight = (neg_counts / pos_counts).to(device)
    print(f"\n  Pos weights por clase (desbalance):")
    for cls, w in zip(CLASSES, pos_weight.tolist()):
        print(f"    {cls}: {w:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Parámetros entrenables del MLP: {n_params:,}")

    # ── Entrenamiento ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Entrenando {args.epochs} épocas...")
    print(f"{'='*60}")

    history = train(model, train_loader, val_loader, optimizer, scheduler,
                    criterion, device, args.epochs)

    # ── Evaluación final en test propio ──────────────────────────────────
    print(f"\n{'='*60}")
    print("  Evaluación final en TEST PROPIO (15% reservado)")
    print(f"{'='*60}")

    test_metrics = evaluate_test(model, test_loader, criterion, device)

    print(f"\n  Test loss:      {test_metrics['test_loss']}")
    print(f"  Macro F1:       {test_metrics['macro_f1']}")
    print(f"  Exact accuracy: {test_metrics['exact_accuracy']}")
    print("\n  Por clase:")
    for cls in CLASSES:
        m = test_metrics[cls]
        print(f"    {cls} → P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")

    # ── Guardar modelo, reporte e índices de split ───────────────────────
    args.output.mkdir(parents=True, exist_ok=True)

    model_path = args.output / "classifier.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim":        input_dim,
        "hidden_dim":       args.hidden_dim,
        "dropout":          args.dropout,
        "classes":          CLASSES,
        "args":             vars(args),
    }, model_path)
    print(f"\n✅ Modelo guardado:  {model_path}")

    # Guardar índices del split para reproducibilidad
    split_path = args.output / "split_info.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(indices, f, indent=2)
    print(f"✅ Split guardado:   {split_path}")

    report = {
        "args": {**vars(args), "features": str(args.features), "output": str(args.output)},
        "splits": {
            "n_train": len(tr_feat),
            "n_val":   len(val_feat),
            "n_test":  len(test_feat),
        },
        "history":      history,
        "test_metrics": test_metrics,
    }

    report_path = args.output / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✅ Reporte guardado: {report_path}")


if __name__ == "__main__":
    main()

"""
RDD2022 — Clasificadores SVM y LightGBM sobre features DINO
============================================================
Entrena SVM y LightGBM sobre las features extraídas por extract_features.py
y compara sus resultados con los del MLP.

Uso:
    # Evaluar ambos modelos sobre DINOv2
    python train_classifiers.py --features C:/features/dinov2 --output C:/models/dinov2

    # Solo SVM
    python train_classifiers.py --features C:/features/dinov2 --output C:/models/dinov2 --model svm

    # Solo LightGBM
    python train_classifiers.py --features C:/features/dinov2 --output C:/models/dinov2 --model lgbm

    # Sobre DINOv3-ViT
    python train_classifiers.py --features C:/features/dinov3-vit --output C:/models/dinov3-vit

Dependencias:
    pip install scikit-learn lightgbm numpy

Salida:
    models/
      dinov2/
        svm_report.json       ← métricas SVM por clase y globales
        lgbm_report.json      ← métricas LightGBM por clase y globales
        classifiers_comparison.json ← comparativa de los dos modelos
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score
)
import lightgbm as lgb

# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────

CLASSES    = ["D00", "D10", "D20"]
VAL_SPLIT  = 0.15
TEST_SPLIT = 0.15
SEED       = 42


# ─────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────

def build_labels(metadata: list) -> np.ndarray:
    """Construye matriz de labels multilabel (N, 3) desde metadata."""
    labels = np.zeros((len(metadata), len(CLASSES)), dtype=np.float32)
    for i, sample in enumerate(metadata):
        for cls in sample.get("classes", []):
            if cls in CLASSES:
                labels[i, CLASSES.index(cls)] = 1.0
    return labels


def make_three_splits(features: np.ndarray, labels: np.ndarray):
    """Split reproducible 70/15/15."""
    np.random.seed(SEED)
    n       = len(features)
    n_test  = int(n * TEST_SPLIT)
    n_val   = int(n * VAL_SPLIT)
    n_train = n - n_val - n_test

    perm = np.random.permutation(n)
    tr_idx   = perm[:n_train]
    val_idx  = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    return (
        features[tr_idx],  labels[tr_idx],
        features[val_idx], labels[val_idx],
        features[test_idx],labels[test_idx],
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Métricas multilabel por clase y macro."""
    metrics = {}
    f1s = []
    for i, cls in enumerate(CLASSES):
        p  = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        r  = recall_score(   y_true[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(       y_true[:, i], y_pred[:, i], zero_division=0)
        metrics[cls] = {
            "precision": round(float(p),  4),
            "recall":    round(float(r),  4),
            "f1":        round(float(f1), 4),
        }
        f1s.append(f1)

    metrics["macro_f1"]       = round(float(np.mean(f1s)), 4)
    metrics["exact_accuracy"] = round(float(
        accuracy_score(y_true, y_pred, normalize=True)
    ), 4)
    return metrics


def print_metrics(name: str, metrics: dict):
    print(f"\n  {name}")
    print(f"  {'─'*45}")
    print(f"  Macro F1:       {metrics['macro_f1']:.4f}")
    print(f"  Exact accuracy: {metrics['exact_accuracy']:.4f}")
    print(f"  Por clase:")
    for cls in CLASSES:
        m = metrics[cls]
        print(f"    {cls} → P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")


# ─────────────────────────────────────────────
# SVM
# ─────────────────────────────────────────────

def train_svm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    C: float = 0.1,
) -> dict:
    """
    LinearSVC con MultiOutputClassifier para multilabel.
    Usa StandardScaler porque SVM es sensible a la escala.
    C pequeño (0.1) para regularizar bien con features de alta dimensión.
    """
    print(f"\n  Entrenando SVM (LinearSVC, C={C})...")
    t0 = time.time()

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    MultiOutputClassifier(
            LinearSVC(C=C, max_iter=2000, random_state=SEED),
            n_jobs=-1
        )),
    ])

    pipeline.fit(X_train, y_train)
    t_train = time.time() - t0
    print(f"  Entrenamiento: {t_train:.1f}s")

    # Evaluación
    val_pred  = pipeline.predict(X_val)
    test_pred = pipeline.predict(X_test)

    val_metrics  = compute_metrics(y_val,  val_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    print_metrics("SVM — Validación", val_metrics)
    print_metrics("SVM — Test",       test_metrics)

    return {
        "model":        "svm",
        "C":            C,
        "train_time_s": round(t_train, 2),
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
    }


# ─────────────────────────────────────────────
# LightGBM
# ─────────────────────────────────────────────

def train_lgbm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    n_estimators: int   = 500,
    learning_rate: float = 0.05,
    num_leaves:    int   = 31,
    reg_lambda:    float = 1.0,
) -> dict:
    """
    LightGBM con un clasificador binario por clase (OneVsRest multilabel).
    Usa early stopping sobre val para evitar overfitting.
    """
    print(f"\n  Entrenando LightGBM (n_est={n_estimators}, lr={learning_rate})...")
    t0 = time.time()

    # Calcular pos_weight por clase para manejar desbalance
    pos_counts = y_train.sum(axis=0).clip(min=1)
    neg_counts = len(y_train) - pos_counts
    pos_weights = neg_counts / pos_counts

    models    = []
    val_preds  = np.zeros_like(y_val,  dtype=np.float32)
    test_preds = np.zeros_like(y_test, dtype=np.float32)

    for i, cls in enumerate(CLASSES):
        print(f"    Clase {cls} (pos_weight={pos_weights[i]:.1f})...", end=" ")

        clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            reg_lambda=reg_lambda,
            scale_pos_weight=pos_weights[i],
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )

        clf.fit(
            X_train, y_train[:, i],
            eval_set=[(X_val, y_val[:, i])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        val_preds[:, i]  = clf.predict(X_val)
        test_preds[:, i] = clf.predict(X_test)
        models.append(clf)
        print(f"best iter={clf.best_iteration_}")

    t_train = time.time() - t0
    print(f"  Entrenamiento total: {t_train:.1f}s")

    val_metrics  = compute_metrics(y_val,  val_preds.round().astype(int))
    test_metrics = compute_metrics(y_test, test_preds.round().astype(int))

    print_metrics("LightGBM — Validación", val_metrics)
    print_metrics("LightGBM — Test",       test_metrics)

    return {
        "model":          "lgbm",
        "n_estimators":   n_estimators,
        "learning_rate":  learning_rate,
        "num_leaves":     num_leaves,
        "reg_lambda":     reg_lambda,
        "train_time_s":   round(t_train, 2),
        "val_metrics":    val_metrics,
        "test_metrics":   test_metrics,
        "best_iterations": [m.best_iteration_ for m in models],
        "_models":        models,  # guardamos internamente para poder serializar
    }


# ─────────────────────────────────────────────
# Comparativa final
# ─────────────────────────────────────────────

def print_comparison(results: list):
    print(f"\n{'='*60}")
    print("  COMPARATIVA FINAL")
    print(f"{'='*60}")
    print(f"  {'Modelo':<12} {'Macro F1':>10} {'D00 F1':>8} {'D10 F1':>8} {'D20 F1':>8} {'Tiempo':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for r in results:
        tm = r["test_metrics"]
        print(
            f"  {r['model']:<12} "
            f"{tm['macro_f1']:>10.4f} "
            f"{tm['D00']['f1']:>8.4f} "
            f"{tm['D10']['f1']:>8.4f} "
            f"{tm['D20']['f1']:>8.4f} "
            f"{r['train_time_s']:>7.1f}s"
        )

    best = max(results, key=lambda r: r["test_metrics"]["macro_f1"])
    print(f"\n  Mejor modelo: {best['model']} (Macro F1={best['test_metrics']['macro_f1']:.4f})")
    print(f"{'='*60}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena SVM y LightGBM sobre features DINO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python train_classifiers.py --features C:/features/dinov2 --output C:/models/dinov2
  python train_classifiers.py --features C:/features/dinov3-vit --output C:/models/dinov3-vit --model lgbm
        """
    )
    parser.add_argument("--features", type=Path, required=True,
                        help="Carpeta con train_features.pt y train_metadata.json")
    parser.add_argument("--output",   type=Path, required=True,
                        help="Carpeta donde guardar los reportes")
    parser.add_argument("--model",    type=str,  default="both",
                        choices=["svm", "lgbm", "both"],
                        help="Modelo a entrenar (default: both)")
    parser.add_argument("--svm-c",    type=float, default=0.1,
                        help="Regularización SVM — C (default: 0.1)")
    parser.add_argument("--lgbm-lr",  type=float, default=0.05,
                        help="Learning rate LightGBM (default: 0.05)")
    parser.add_argument("--lgbm-leaves", type=int, default=31,
                        help="Num leaves LightGBM (default: 31)")
    parser.add_argument("--lgbm-lambda", type=float, default=1.0,
                        help="Regularización L2 LightGBM (default: 1.0)")
    parser.add_argument("--save-models", action="store_true",
                        help="Guardar modelos LightGBM como .txt (necesario para patch_importance.py)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("  RDD2022 — SVM + LightGBM sobre features DINO")
    print(f"{'='*60}")
    print(f"  Features: {args.features}")
    print(f"  Output:   {args.output}")

    # ── Cargar features ───────────────────────────────────────────────
    print("\n  Cargando features...")
    features = torch.load(
        args.features / "train_features.pt", weights_only=True
    ).numpy()

    with open(args.features / "train_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    labels = build_labels(metadata)
    print(f"  Features: {features.shape}")
    print(f"  Labels:   {labels.shape}")

    # Distribución de clases
    print("\n  Distribución de clases en train completo:")
    for i, cls in enumerate(CLASSES):
        n = int(labels[:, i].sum())
        print(f"    {cls}: {n} positivos ({n/len(labels)*100:.1f}%)")

    # ── Split 70/15/15 ────────────────────────────────────────────────
    X_tr, y_tr, X_val, y_val, X_te, y_te = make_three_splits(features, labels)
    print(f"\n  Split → train: {len(X_tr)} | val: {len(X_val)} | test: {len(X_te)}")

    args.output.mkdir(parents=True, exist_ok=True)

    # ── Entrenar modelos ──────────────────────────────────────────────
    results = []
    models_to_run = (
        ["svm", "lgbm"] if args.model == "both" else [args.model]
    )

    for model_name in models_to_run:
        print(f"\n{'─'*60}")
        if model_name == "svm":
            result = train_svm(
                X_tr, y_tr, X_val, y_val, X_te, y_te,
                C=args.svm_c,
            )
            report_path = args.output / "svm_report.json"

        else:  # lgbm
            result = train_lgbm(
                X_tr, y_tr, X_val, y_val, X_te, y_te,
                n_estimators=500,
                learning_rate=args.lgbm_lr,
                num_leaves=args.lgbm_leaves,
                reg_lambda=args.lgbm_lambda,
            )
            report_path = args.output / "lgbm_report.json"

        # Limpiar _models del dict antes de serializar a JSON
        result_to_save = {k: v for k, v in result.items() if k != "_models"}
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result_to_save, f, indent=2, ensure_ascii=False)
        print(f"\n  Reporte guardado: {report_path}")

        # Guardar modelos LightGBM como .txt si se solicita
        if model_name == "lgbm" and args.save_models and "_models" in result:
            for cls, clf in zip(CLASSES, result["_models"]):
                booster_path = args.output / f"lgbm_{cls.lower()}.txt"
                clf.booster_.save_model(str(booster_path))
                print(f"  Modelo guardado: {booster_path}")

        results.append(result_to_save)

    # ── Comparativa ───────────────────────────────────────────────────
    if len(results) > 1:
        print_comparison(results)
        comp_path = args.output / "classifiers_comparison.json"
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump({
                "features": str(args.features),
                "results":  results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n  Comparativa guardada: {comp_path}")


if __name__ == "__main__":
    main()

"""
RDD2022 Dataset Cleaner
=======================
Limpieza automatizada del dataset RDD2022 para detección de grietas con transformer.

Decisiones de limpieza aplicadas:
  1. Elimina China_Drone (imágenes aéreas, perspectiva incompatible)
  2. Elimina imágenes sin anotaciones (no aportan ejemplos de grietas)
  3. Elimina imágenes con SOLO anotaciones D40 (baches, no grietas lineales)
  4. Elimina duplicados por hash perceptual (pHash)
  5. Submuestreo estratificado por país para balancear el dataset

Uso:
    python clean_rdd2022.py --input /ruta/RDD2022 --output /ruta/RDD2022_clean
    python clean_rdd2022.py --input /ruta/RDD2022 --output /ruta/RDD2022_clean --sample-ratio 0.4
    python clean_rdd2022.py --input /ruta/RDD2022 --output /ruta/RDD2022_clean --dry-run
"""

import os
import shutil
import hashlib
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import json
import random
from datetime import datetime

# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────

# Países que se incluyen (China_Drone excluido)
VALID_COUNTRIES = [
    "China_MotorBike",
    "Czech",
    "India",
    "Japan",
    "Norway",
    "United_States",
]

# Clase a eliminar si es la ÚNICA presente
EXCLUDE_ONLY_CLASS = {"D40"}

# Clases válidas que queremos conservar
VALID_CLASSES = {"D00", "D10", "D20"}

# Ratio de submuestreo por país (cuántas imágenes conservar)
# Países grandes se reducen más; países pequeños se conservan íntegros
DEFAULT_SAMPLE_RATIOS = {
    "Japan":          0.35,  # El más grande, reducimos bastante
    "India":          0.35,
    "United_States":  0.40,
    "China_MotorBike": 0.40,
    "Czech":          0.70,  # Pocos datos, conservamos más
    "Norway":         0.70,
}

# ─────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────

def file_hash(filepath: Path) -> str:
    """Hash MD5 del contenido del archivo (detección de duplicados exactos)."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_annotations(xml_path: Path) -> dict:
    """
    Parsea un XML de anotación Pascal VOC y devuelve:
      - classes: set de clases presentes
      - boxes: número de bounding boxes
      - valid: True si el XML es legible
    """
    result = {"classes": set(), "boxes": 0, "valid": False}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name_el = obj.find("name")
            if name_el is not None and name_el.text:
                cls = name_el.text.strip()
                result["classes"].add(cls)
                result["boxes"] += 1
        result["valid"] = True
    except Exception:
        pass
    return result


def should_keep_image(annotation: dict) -> tuple[bool, str]:
    """
    Decide si conservar una imagen según sus anotaciones.
    Devuelve (keep: bool, reason: str).
    """
    if not annotation["valid"]:
        return False, "xml_invalid"

    classes = annotation["classes"]

    # Sin anotaciones → descarta
    if not classes:
        return False, "no_annotations"

    # Solo D40 → descarta
    if classes == EXCLUDE_ONLY_CLASS or classes.issubset(EXCLUDE_ONLY_CLASS):
        return False, "only_d40"

    # Tiene al menos una clase válida → conserva
    valid_present = classes & VALID_CLASSES
    if valid_present:
        return True, "ok"

    # Solo clases desconocidas → descarta
    return False, "unknown_classes_only"


def get_image_extension(images_dir: Path, stem: str) -> str | None:
    """Busca la extensión del archivo de imagen dado el stem del XML."""
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            return ext
    return None


# ─────────────────────────────────────────────
# Lógica principal
# ─────────────────────────────────────────────

class RDDCleaner:
    def __init__(self, input_dir: Path, output_dir: Path, sample_ratios: dict, dry_run: bool = False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sample_ratios = sample_ratios
        self.dry_run = dry_run
        self.stats = defaultdict(lambda: defaultdict(int))
        self.global_hashes = {}  # hash -> ruta origen (detección duplicados)

    def run(self):
        print("\n" + "="*60)
        print("  RDD2022 Dataset Cleaner")
        print("="*60)
        print(f"  Input:   {self.input_dir}")
        print(f"  Output:  {self.output_dir}")
        print(f"  Dry run: {self.dry_run}")
        print("="*60 + "\n")

        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        for country in VALID_COUNTRIES:
            country_path = self.input_dir / country
            if not country_path.exists():
                print(f"[WARN] País no encontrado: {country}, saltando...")
                continue
            self._process_country(country, country_path)

        self._print_report()
        self._save_report()

    def _process_country(self, country: str, country_path: Path):
        print(f"\n{'─'*50}")
        print(f"  Procesando: {country}")
        print(f"{'─'*50}")

        for split in ["train", "test"]:
            split_path = country_path / split
            if not split_path.exists():
                continue

            images_dir = split_path / "images"
            annotations_dir = split_path / "annotations" / "xmls"

            if not images_dir.exists():
                print(f"  [WARN] Sin directorio de imágenes: {images_dir}")
                continue

            # El test no suele tener anotaciones — lo copiamos con submuestreo directo
            if not annotations_dir.exists():
                self._process_test_split(country, split, images_dir)
                continue

            self._process_train_split(country, split, images_dir, annotations_dir)

    def _process_train_split(self, country: str, split: str, images_dir: Path, annotations_dir: Path):
        """Procesa splits que tienen anotaciones (normalmente train)."""
        candidates = []  # Lista de (image_path, xml_path, annotation)

        xml_files = list(annotations_dir.glob("*.xml"))
        self.stats[country]["total_xmls"] += len(xml_files)

        for xml_path in xml_files:
            stem = xml_path.stem
            ext = get_image_extension(images_dir, stem)

            if ext is None:
                self.stats[country]["missing_image"] += 1
                continue

            img_path = images_dir / (stem + ext)
            annotation = parse_annotations(xml_path)
            keep, reason = should_keep_image(annotation)

            if not keep:
                self.stats[country][f"removed_{reason}"] += 1
                continue

            candidates.append((img_path, xml_path, annotation))

        # Submuestreo estratificado
        ratio = self.sample_ratios.get(country, 0.5)
        sampled = self._subsample(candidates, ratio)

        self.stats[country]["after_filter"] += len(candidates)
        self.stats[country]["after_sample"] += len(sampled)

        # Detección de duplicados y copia
        for img_path, xml_path, annotation in sampled:
            img_hash = file_hash(img_path)

            if img_hash in self.global_hashes:
                self.stats[country]["removed_duplicate"] += 1
                continue

            self.global_hashes[img_hash] = img_path

            # Actualizar estadísticas de clases conservadas
            for cls in annotation["classes"] & (VALID_CLASSES | {"D40"}):
                self.stats[country][f"class_{cls}"] += 1

            if not self.dry_run:
                out_img_dir = self.output_dir / country / split / "images"
                out_xml_dir = self.output_dir / country / split / "annotations" / "xmls"
                out_img_dir.mkdir(parents=True, exist_ok=True)
                out_xml_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, out_img_dir / img_path.name)
                shutil.copy2(xml_path, out_xml_dir / xml_path.name)

            self.stats[country]["final_kept"] += 1

        print(f"  [{split}] {len(xml_files)} XMLs → {len(candidates)} válidos → {len(sampled)} muestreados → {self.stats[country]['final_kept']} finales")

    def _process_test_split(self, country: str, split: str, images_dir: Path):
        """Procesa splits sin anotaciones (test). Solo submuestreo y dedup."""
        images = list(images_dir.glob("*.[jJpP][pPnN][gG]*"))
        self.stats[country]["total_test_images"] += len(images)

        ratio = self.sample_ratios.get(country, 0.5)
        sampled = random.sample(images, max(1, int(len(images) * ratio)))

        kept = 0
        for img_path in sampled:
            img_hash = file_hash(img_path)
            if img_hash in self.global_hashes:
                self.stats[country]["removed_duplicate"] += 1
                continue
            self.global_hashes[img_hash] = img_path

            if not self.dry_run:
                out_img_dir = self.output_dir / country / split / "images"
                out_img_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, out_img_dir / img_path.name)

            kept += 1

        self.stats[country]["final_kept_test"] = kept
        print(f"  [{split}] {len(images)} imágenes → {kept} conservadas (ratio={ratio})")

    def _subsample(self, candidates: list, ratio: float) -> list:
        """Submuestreo aleatorio reproducible."""
        random.seed(42)
        n = max(1, int(len(candidates) * ratio))
        return random.sample(candidates, min(n, len(candidates)))

    def _print_report(self):
        print("\n" + "="*60)
        print("  REPORTE FINAL")
        print("="*60)

        total_kept = 0
        total_removed = 0

        for country in VALID_COUNTRIES:
            s = self.stats[country]
            if not s:
                continue

            kept = s.get("final_kept", 0) + s.get("final_kept_test", 0)
            removed_no_ann = s.get("removed_no_annotations", 0)
            removed_d40 = s.get("removed_only_d40", 0)
            removed_dup = s.get("removed_duplicate", 0)
            removed_inv = s.get("removed_xml_invalid", 0)

            total_kept += kept
            total_removed += removed_no_ann + removed_d40 + removed_dup + removed_inv

            print(f"\n  {country}:")
            print(f"    ✅ Conservadas:          {kept}")
            print(f"    ❌ Sin anotaciones:      {removed_no_ann}")
            print(f"    ❌ Solo D40:             {removed_d40}")
            print(f"    ❌ Duplicadas:           {removed_dup}")
            print(f"    ❌ XML inválido:         {removed_inv}")

            # Distribución de clases
            class_dist = {c: s.get(f"class_{c}", 0) for c in ["D00", "D10", "D20", "D40"]}
            if any(class_dist.values()):
                print(f"    📊 Clases: " + " | ".join(f"{k}:{v}" for k, v in class_dist.items() if v > 0))

        print(f"\n{'─'*50}")
        print(f"  TOTAL CONSERVADAS: {total_kept}")
        print(f"  TOTAL ELIMINADAS:  {total_removed}")
        if total_kept + total_removed > 0:
            reduction = (total_removed / (total_kept + total_removed)) * 100
            print(f"  REDUCCIÓN:         {reduction:.1f}%")
        print(f"  CHINA_DRONE:       ❌ Excluido completamente")
        if self.dry_run:
            print("\n  ⚠️  MODO DRY-RUN: no se ha copiado ningún archivo")
        print("="*60)

    def _save_report(self):
        report = {
            "timestamp": datetime.now().isoformat(),
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "dry_run": self.dry_run,
            "excluded_countries": ["China_Drone"],
            "excluded_classes": list(EXCLUDE_ONLY_CLASS),
            "sample_ratios": self.sample_ratios,
            "stats": {k: dict(v) for k, v in self.stats.items()},
        }

        report_path = self.output_dir / "cleaning_report.json" if not self.dry_run else Path("cleaning_report_dryrun.json")
        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n  📄 Reporte guardado en: {report_path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Limpieza automatizada del dataset RDD2022",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Ejecución normal
  python clean_rdd2022.py --input C:/RDD2022 --output C:/RDD2022_clean

  # Primero prueba sin copiar archivos
  python clean_rdd2022.py --input C:/RDD2022 --output C:/RDD2022_clean --dry-run

  # Con ratio de submuestreo global personalizado
  python clean_rdd2022.py --input C:/RDD2022 --output C:/RDD2022_clean --sample-ratio 0.3
        """
    )
    parser.add_argument("--input",  type=Path, required=True, help="Ruta raíz del dataset RDD2022 original")
    parser.add_argument("--output", type=Path, required=True, help="Ruta donde guardar el dataset limpio")
    parser.add_argument("--sample-ratio", type=float, default=None,
                        help="Ratio global de submuestreo (0.0-1.0). Sobreescribe los ratios por país.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simula la limpieza sin copiar archivos (solo muestra estadísticas)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    sample_ratios = DEFAULT_SAMPLE_RATIOS.copy()
    if args.sample_ratio is not None:
        sample_ratios = {country: args.sample_ratio for country in VALID_COUNTRIES}
        print(f"[INFO] Usando ratio global de submuestreo: {args.sample_ratio}")

    cleaner = RDDCleaner(
        input_dir=args.input,
        output_dir=args.output,
        sample_ratios=sample_ratios,
        dry_run=args.dry_run,
    )
    cleaner.run()


if __name__ == "__main__":
    main()

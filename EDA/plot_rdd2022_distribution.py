"""
RDD2022 Dataset Distribution Plotter
=====================================
Genera gráficas comparativas antes/después de la limpieza del dataset.

Uso:
    python plot_rdd2022_distribution.py --before C:/RDD2022 --after C:/RDD2022_clean
    python plot_rdd2022_distribution.py --before C:/RDD2022 --after C:/RDD2022_clean --output report.png

Dependencias:
    pip install matplotlib numpy
"""

import argparse
import hashlib
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# ─────────────────────────────────────────────
# Paleta y estilo
# ─────────────────────────────────────────────

BEFORE_COLOR = "#E8503A"
AFTER_COLOR  = "#3AB8E8"

CLASS_COLORS = {
    "D00":   "#F4A261",
    "D10":   "#2A9D8F",
    "D20":   "#E76F51",
    "D40":   "#9B5DE5",
    "other": "#8D99AE",
}

COUNTRIES_ORDER = [
    "China_Drone", "China_MotorBike", "Czech",
    "India", "Japan", "Norway", "United_States",
]

# ─────────────────────────────────────────────
# Análisis del dataset
# ─────────────────────────────────────────────

def scan_dataset(root: Path) -> dict:
    stats = {
        "total_images": 0,
        "total_boxes":  0,
        "by_country":   defaultdict(lambda: {
            "images": 0, "boxes": 0,
            "classes": defaultdict(int),
            "empty": 0,
        }),
        "by_class":     defaultdict(int),
        "empty_images": 0,
    }

    if not root.exists():
        print(f"  [WARN] Ruta no encontrada: {root}")
        return stats

    for country_dir in sorted(root.iterdir()):
        if not country_dir.is_dir():
            continue
        country = country_dir.name

        for split in ["train", "test"]:
            images_dir = country_dir / split / "images"
            xml_dir    = country_dir / split / "annotations" / "xmls"

            if not images_dir.exists():
                continue

            images = list(images_dir.glob("*.[jJpP][pPnN][gG]*"))
            stats["by_country"][country]["images"] += len(images)
            stats["total_images"] += len(images)

            if xml_dir.exists():
                for xml_path in xml_dir.glob("*.xml"):
                    try:
                        root_el = ET.parse(xml_path).getroot()
                        objects = root_el.findall("object")
                        if not objects:
                            stats["by_country"][country]["empty"] += 1
                            stats["empty_images"] += 1
                        for obj in objects:
                            name_el = obj.find("name")
                            if name_el is not None and name_el.text:
                                cls = name_el.text.strip()
                                stats["by_country"][country]["classes"][cls] += 1
                                stats["by_country"][country]["boxes"] += 1
                                stats["by_class"][cls] += 1
                                stats["total_boxes"] += 1
                    except Exception:
                        pass

    return stats


# ─────────────────────────────────────────────
# Figura principal
# ─────────────────────────────────────────────

def make_figure(before: dict, after: dict, before_path: Path, after_path: Path, output: Path):
    matplotlib.rcParams.update({
        "font.family":       "monospace",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.22,
        "grid.linestyle":    "--",
        "axes.labelpad":     10,
        "figure.facecolor":  "#0F1117",
        "axes.facecolor":    "#171B26",
        "text.color":        "#E8EAF0",
        "axes.labelcolor":   "#E8EAF0",
        "xtick.color":       "#9AA0B4",
        "ytick.color":       "#9AA0B4",
        "axes.edgecolor":    "#2D3348",
        "grid.color":        "#2D3348",
    })

    fig = plt.figure(figsize=(22, 18), facecolor="#0F1117")
    fig.subplots_adjust(hspace=0.55, wspace=0.38, top=0.91, bottom=0.06, left=0.06, right=0.97)

    # Título
    fig.text(0.5, 0.975, "RDD2022 — Distribución del dataset: Antes vs Después de la limpieza",
             ha="center", va="top", fontsize=17, fontweight="bold", color="#E8EAF0")
    fig.text(0.5, 0.955, f"ANTES: {before_path}    ·    DESPUÉS: {after_path}",
             ha="center", va="top", fontsize=9, color="#6B7394")

    gs = gridspec.GridSpec(3, 3, figure=fig)

    # KPI cards (fila superior)
    ax_dummy = fig.add_subplot(gs[0, :])
    ax_dummy.set_visible(False)
    _draw_kpi_cards(fig, before, after)

    # Fila media
    ax1 = fig.add_subplot(gs[1, :2])
    _plot_images_by_country(ax1, before, after)

    ax2 = fig.add_subplot(gs[1, 2])
    _plot_reduction_pie(ax2, before, after)

    # Fila inferior
    ax3 = fig.add_subplot(gs[2, 0])
    _plot_class_distribution(ax3, before, after)

    ax4 = fig.add_subplot(gs[2, 1])
    _plot_empty_images(ax4, before, after)

    ax5 = fig.add_subplot(gs[2, 2])
    _plot_class_country_heatmap(ax5, after)

    plt.savefig(output, dpi=160, bbox_inches="tight", facecolor="#0F1117")
    print(f"\n✅ Figura guardada en: {output}")
    plt.close()


# ─────────────────────────────────────────────
# Subplots
# ─────────────────────────────────────────────

def _draw_kpi_cards(fig, before, after):
    b_img = before["total_images"]
    a_img = after["total_images"]
    b_box = before["total_boxes"]
    a_box = after["total_boxes"]
    b_emp = before["empty_images"]
    a_emp = after["empty_images"]
    b_cnt = len(before["by_country"])
    a_cnt = len(after["by_country"])

    pct_img = (1 - a_img / max(b_img, 1)) * 100
    pct_box = (1 - a_box / max(b_box, 1)) * 100

    cards = [
        ("IMÁGENES TOTALES",
         f"{b_img:,}", f"{a_img:,}",
         f"−{b_img - a_img:,}", f"{pct_img:.0f}% reducción"),
        ("BOUNDING BOXES",
         f"{b_box:,}", f"{a_box:,}",
         f"−{b_box - a_box:,}", f"{pct_box:.0f}% reducción"),
        ("IMÁGENES VACÍAS",
         f"{b_emp:,}", f"{a_emp:,}",
         f"−{b_emp - a_emp:,}", "eliminadas"),
        ("PAÍSES ACTIVOS",
         f"{b_cnt}", f"{a_cnt}",
         f"−{b_cnt - a_cnt}", "China_Drone excluido"),
    ]

    for i, (title, val_b, val_a, delta, note) in enumerate(cards):
        x = 0.045 + i * 0.237
        y = 0.875

        rect = mpatches.FancyBboxPatch(
            (x, y - 0.058), 0.215, 0.068,
            boxstyle="round,pad=0.012",
            facecolor="#1E2336", edgecolor="#2D3348",
            linewidth=1.3, transform=fig.transFigure, clip_on=False
        )
        fig.add_artist(rect)

        fig.text(x + 0.008, y + 0.004, title,
                 fontsize=7.5, color="#6B7394", fontweight="bold", transform=fig.transFigure)
        fig.text(x + 0.008, y - 0.018, val_b,
                 fontsize=14, color=BEFORE_COLOR, fontweight="bold", transform=fig.transFigure)
        fig.text(x + 0.008, y - 0.040, f"→  {val_a}",
                 fontsize=14, color=AFTER_COLOR, fontweight="bold", transform=fig.transFigure)
        fig.text(x + 0.145, y - 0.018, delta,
                 fontsize=10, color="#F4E04D", transform=fig.transFigure)
        fig.text(x + 0.145, y - 0.036, note,
                 fontsize=7.5, color="#6B7394", transform=fig.transFigure)


def _plot_images_by_country(ax, before, after):
    countries = COUNTRIES_ORDER
    b_vals = [before["by_country"].get(c, {}).get("images", 0) for c in countries]
    a_vals = [after["by_country"].get(c, {}).get("images", 0) for c in countries]

    x = np.arange(len(countries))
    w = 0.38

    bars_b = ax.bar(x - w/2, b_vals, w, color=BEFORE_COLOR, alpha=0.82, label="Antes", zorder=3)
    bars_a = ax.bar(x + w/2, a_vals, w, color=AFTER_COLOR,  alpha=0.82, label="Después", zorder=3)

    max_val = max(b_vals + a_vals) if b_vals + a_vals else 1
    for bar, color in [(bars_b, BEFORE_COLOR), (bars_a, AFTER_COLOR)]:
        for b in bar:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width()/2, h + max_val * 0.01,
                        f"{int(h):,}", ha="center", va="bottom",
                        fontsize=7, color=color)

    # Resaltar China_Drone
    if "China_Drone" in countries:
        idx = countries.index("China_Drone")
        ax.axvspan(idx - 0.5, idx + 0.5, color="#E8503A", alpha=0.07, zorder=0)
        ax.text(idx, max_val * 0.45, "EXCLUIDO", ha="center", va="center",
                fontsize=8, color="#E8503A", alpha=0.75, rotation=90, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in countries], fontsize=8.5)
    ax.set_title("Imágenes por país", fontsize=12, color="#E8EAF0", pad=12)
    ax.set_ylabel("Nº imágenes", fontsize=9)
    ax.legend(fontsize=9, facecolor="#1E2336", edgecolor="#2D3348", labelcolor="#E8EAF0")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))


def _plot_reduction_pie(ax, before, after):
    kept    = after["total_images"]
    removed = before["total_images"] - kept

    if before["total_images"] == 0:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", color="#6B7394")
        return

    wedges, texts, autotexts = ax.pie(
        [kept, removed],
        explode=(0.04, 0.04),
        colors=[AFTER_COLOR, BEFORE_COLOR],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "#0F1117", "linewidth": 2},
        textprops={"color": "#E8EAF0", "fontsize": 9},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")

    ax.set_title("Reducción global\nde imágenes", fontsize=12, color="#E8EAF0", pad=12)
    ax.legend(
        wedges,
        [f"Conservadas ({kept:,})", f"Eliminadas ({removed:,})"],
        loc="lower center", bbox_to_anchor=(0.5, -0.18),
        fontsize=8, facecolor="#1E2336", edgecolor="#2D3348", labelcolor="#E8EAF0"
    )


def _plot_class_distribution(ax, before, after):
    classes = ["D00", "D10", "D20", "D40"]
    b_vals  = [before["by_class"].get(c, 0) for c in classes]
    a_vals  = [after["by_class"].get(c, 0) for c in classes]
    colors  = [CLASS_COLORS.get(c, "#8D99AE") for c in classes]

    x = np.arange(len(classes))
    w = 0.35

    bars_b = ax.bar(x - w/2, b_vals, w, color=colors, alpha=0.45, label="Antes", zorder=3)
    bars_a = ax.bar(x + w/2, a_vals, w, color=colors, alpha=0.95, label="Después", zorder=3,
                    edgecolor=colors, linewidth=1.5)

    max_val = max(b_vals + a_vals) if b_vals + a_vals else 1
    for bar in bars_b:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + max_val * 0.012,
                    f"{int(h):,}", ha="center", va="bottom", fontsize=6.5, color="#9AA0B4")
    for bar in bars_a:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + max_val * 0.012,
                    f"{int(h):,}", ha="center", va="bottom", fontsize=6.5, color=AFTER_COLOR)

    # Marcar D40
    ax.axvspan(2.5, 3.5, color="#9B5DE5", alpha=0.08, zorder=0)
    ax.text(3, max_val * 0.5, "EXCLUIDO\n(solo D40)", ha="center", va="center",
            fontsize=7.5, color="#9B5DE5", alpha=0.85, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_title("Bounding boxes por clase", fontsize=12, color="#E8EAF0", pad=12)
    ax.set_ylabel("Nº bounding boxes", fontsize=9)
    ax.legend(fontsize=9, facecolor="#1E2336", edgecolor="#2D3348", labelcolor="#E8EAF0")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))


def _plot_empty_images(ax, before, after):
    countries = [c for c in COUNTRIES_ORDER if c != "China_Drone"]
    b_vals = [before["by_country"].get(c, {}).get("empty", 0) for c in countries]
    a_vals = [after["by_country"].get(c, {}).get("empty", 0) for c in countries]

    x = np.arange(len(countries))
    w = 0.35

    ax.bar(x - w/2, b_vals, w, color=BEFORE_COLOR, alpha=0.75, label="Antes", zorder=3)
    ax.bar(x + w/2, a_vals, w, color=AFTER_COLOR,  alpha=0.75, label="Después", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in countries], fontsize=7.5)
    ax.set_title("Imágenes sin anotaciones\n(eliminadas en limpieza)", fontsize=12,
                 color="#E8EAF0", pad=12)
    ax.set_ylabel("Nº imágenes vacías", fontsize=9)
    ax.legend(fontsize=9, facecolor="#1E2336", edgecolor="#2D3348", labelcolor="#E8EAF0")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))


def _plot_class_country_heatmap(ax, after):
    countries = [c for c in COUNTRIES_ORDER if c in after["by_country"]]
    classes   = ["D00", "D10", "D20"]

    matrix = np.zeros((len(classes), len(countries)))
    for j, country in enumerate(countries):
        cls_data = after["by_country"][country].get("classes", {})
        for i, cls in enumerate(classes):
            matrix[i, j] = cls_data.get(cls, 0)

    col_sums = matrix.sum(axis=0)
    col_sums[col_sums == 0] = 1
    matrix_norm = matrix / col_sums

    im = ax.imshow(matrix_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(countries)))
    ax.set_xticklabels([c.replace("_", "\n") for c in countries], fontsize=7)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=10)

    for i in range(len(classes)):
        for j in range(len(countries)):
            val = matrix[i, j]
            pct = matrix_norm[i, j]
            txt_color = "white" if pct > 0.55 else "#1E2336"
            ax.text(j, i, f"{int(val):,}\n({pct*100:.0f}%)",
                    ha="center", va="center", fontsize=6.5, color=txt_color)

    ax.set_title("Clases × país (después limpieza)\n% por país",
                 fontsize=12, color="#E8EAF0", pad=12)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Proporción", color="#9AA0B4", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#9AA0B4")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#9AA0B4", fontsize=7)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera gráficas comparativas antes/después de la limpieza del RDD2022",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python plot_rdd2022_distribution.py --before C:/RDD2022 --after C:/RDD2022_clean
  python plot_rdd2022_distribution.py --before C:/RDD2022 --after C:/RDD2022_clean --output mi_reporte.png
        """
    )
    parser.add_argument("--before", type=Path, required=True,
                        help="Ruta al dataset original RDD2022")
    parser.add_argument("--after",  type=Path, required=True,
                        help="Ruta al dataset limpio RDD2022_clean")
    parser.add_argument("--output", type=Path, default=Path("rdd2022_distribution.png"),
                        help="Archivo de salida (default: rdd2022_distribution.png)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("📊 Analizando dataset ORIGINAL...")
    before_stats = scan_dataset(args.before)
    print(f"   → {before_stats['total_images']:,} imágenes | {before_stats['total_boxes']:,} bboxes")

    print("📊 Analizando dataset LIMPIO...")
    after_stats = scan_dataset(args.after)
    print(f"   → {after_stats['total_images']:,} imágenes | {after_stats['total_boxes']:,} bboxes")

    print("\n🎨 Generando figura...")
    make_figure(before_stats, after_stats, args.before, args.after, args.output)


if __name__ == "__main__":
    main()

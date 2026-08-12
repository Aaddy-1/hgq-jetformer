#!/usr/bin/env python3
"""
scripts/analyze_ebops.py
------------------------
Analyzes and visualizes Effective Bit Operations (EBOPs) per layer for HGQ-quantized Keras models (e.g. JetFormer).

Features:
- Accurate Top-Level Accounting: Respects top-level layer `.ebops` properties (matching HGQ library totals).
- Optional Sub-layer Drilldown: Optional `--sublayers` flag to inspect inner sub-layer operations without double-counting.
- Formatted ASCII summary tables (Top-level layer breakdown & Architectural Section summary).
- High-resolution Matplotlib visualizations saved to disk (Bar charts & Section distribution).

Usage:
    python scripts/analyze_ebops.py --model_path .agents/EXP-18-final/models/quantized/128_17f.keras
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Ensure Keras environment is configured
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import keras
import hgq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def extract_top_level_ebops(model):
    """
    Extracts top-level layer EBOP contributions matching `scripts/s.py` and HGQ standards.
    If a top-level layer exposes `.ebops`, that value is used as the layer's official count.
    If a top-level layer does not have `.ebops`, but contains child sub-layers with `.ebops`,
    its child sub-layer `.ebops` are aggregated.
    """
    entries = []

    for idx, layer in enumerate(model.layers):
        if hasattr(layer, "ebops") and layer.ebops is not None:
            try:
                val = float(layer.ebops)
                entries.append({
                    "name": layer.name,
                    "layer": layer,
                    "type": layer.__class__.__name__,
                    "ebops": val,
                    "group": categorize_layer(layer.name),
                    "is_composite": len(getattr(layer, "layers", []) or getattr(layer, "_flatten_layers", lambda: [])()) > 1
                })
                continue
            except Exception:
                pass

        # If top-level layer does not expose .ebops, check child layers
        sub_layers = getattr(layer, "layers", None) or getattr(layer, "_flatten_layers", lambda: [])()
        sub_layers = [sl for sl in sub_layers if sl is not layer]
        
        child_ebops = 0.0
        found_child = False
        for sl in sub_layers:
            if hasattr(sl, "ebops") and sl.ebops is not None:
                try:
                    child_ebops += float(sl.ebops)
                    found_child = True
                except Exception:
                    pass

        if found_child:
            entries.append({
                "name": layer.name,
                "layer": layer,
                "type": layer.__class__.__name__,
                "ebops": child_ebops,
                "group": categorize_layer(layer.name),
                "is_composite": True
            })

    return entries


def extract_sublayer_drilldown(composite_layer):
    """
    Extracts leaf sub-layer EBOP breakdown for a composite layer (e.g. QLinformerAttention).
    """
    sub_layers = getattr(composite_layer, "layers", None) or getattr(composite_layer, "_flatten_layers", lambda: [])()
    sub_layers = [sl for sl in sub_layers if sl is not composite_layer]

    sub_entries = []
    for sl in sub_layers:
        if hasattr(sl, "ebops") and sl.ebops is not None:
            try:
                sub_entries.append({
                    "name": sl.name,
                    "type": sl.__class__.__name__,
                    "ebops": float(sl.ebops)
                })
            except Exception:
                pass
    return sub_entries


def categorize_layer(layer_name):
    """
    Categorizes a layer into high-level architectural components based on name patterns.
    """
    name_lower = layer_name.lower()

    if "embedding" in name_lower or "cls_token" in name_lower or "prepend" in name_lower:
        return "Embedding & Prep"
    elif "classifier" in name_lower or "embed_dense" in name_lower or "extract_cls" in name_lower or "linformer_pool" in name_lower:
        return "Classifier Head"
    elif "transformer_block" in name_lower:
        import re
        match = re.search(r"transformer_block_(\d+)", name_lower)
        block_idx = match.group(1) if match else ""
        block_prefix = f"Block {block_idx}" if block_idx else "Transformer Block"

        if "attn" in name_lower or "linformer" in name_lower or "attention" in name_lower:
            return f"{block_prefix} (Attention)"
        elif "ffn" in name_lower:
            return f"{block_prefix} (FFN)"
        elif "residual" in name_lower or "add" in name_lower:
            return f"{block_prefix} (Residual)"
        else:
            return f"{block_prefix} (Other)"
    else:
        return "Other / Miscellaneous"


def print_table(title, headers, rows):
    """
    Prints a clean formatted text table with borders.
    """
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
    separator = "-+-".join("-" * col_widths[i] for i in range(len(headers)))

    print(f"\n=== {title} ===")
    print(separator)
    print(header_line)
    print(separator)
    for row in rows:
        print(" | ".join(f"{str(row[i]):<{col_widths[i]}}" for i in range(len(row))))
    print(separator)


def main():
    parser = argparse.ArgumentParser(description="Analyze EBOPs per layer for HGQ Keras models.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, ".agents/EXP-18-final/models/quantized/128_17f.keras"),
        help="Path to the saved quantized .keras model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plot PNG (defaults to same folder or output folder)",
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default="ebops_per_layer.png",
        help="Filename for saved plot image",
    )
    parser.add_argument(
        "--sublayers",
        action="store_true",
        help="Display detailed sub-layer drilldown for composite layers",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)

    print(f"[HGQ EBOP Analyzer] Loading model: {args.model_path}")
    model = keras.models.load_model(args.model_path, compile=False)

    # 1. Collect Top-Level EBOP info
    ebop_entries = extract_top_level_ebops(model)

    if not ebop_entries:
        print("Warning: No layers with 'ebops' attribute were found in the model.")
        sys.exit(0)

    total_ebops = sum(e["ebops"] for e in ebop_entries)
    print(f"\n>>> Total Model EBOPs: {total_ebops:,.2f} ({total_ebops/1e3:.2f} kEBOPs) <<<")

    for e in ebop_entries:
        e["pct"] = (e["ebops"] / total_ebops * 100.0) if total_ebops > 0 else 0.0

    # 2. Layer Table
    layer_rows = []
    cum_pct = 0.0
    for idx, e in enumerate(ebop_entries, 1):
        cum_pct += e["pct"]
        layer_rows.append([
            f"{idx:02d}. {e['name']}",
            e["group"],
            e["type"],
            f"{e['ebops']:,.1f}",
            f"{e['pct']:.2f}%",
            f"{cum_pct:.2f}%"
        ])

    print_table(
        "Top-Level Model Layer EBOP Breakdown",
        ["Layer Name", "Module Group", "Layer Type", "EBOPs", "% Total", "Cum %"],
        layer_rows
    )

    # 3. Grouped Module Summary Table
    group_totals = {}
    group_counts = {}
    for e in ebop_entries:
        grp = e["group"]
        group_totals[grp] = group_totals.get(grp, 0.0) + e["ebops"]
        group_counts[grp] = group_counts.get(grp, 0) + 1

    group_rows = []
    for grp, grp_ebops in sorted(group_totals.items(), key=lambda x: x[1], reverse=True):
        grp_pct = (grp_ebops / total_ebops * 100.0) if total_ebops > 0 else 0.0
        group_rows.append([
            grp,
            group_counts[grp],
            f"{grp_ebops:,.1f}",
            f"{grp_pct:.2f}%"
        ])

    print_table(
        "Architectural Component Summary",
        ["Module Section", "Layers", "Total EBOPs", "% Total"],
        group_rows
    )

    # 4. Optional Sub-layer Drilldown Output
    if args.sublayers:
        for e in ebop_entries:
            if e["is_composite"]:
                sub_entries = extract_sublayer_drilldown(e["layer"])
                if sub_entries:
                    sub_rows = [
                        [se["name"], se["type"], f"{se['ebops']:,.1f}"]
                        for se in sub_entries
                    ]
                    print_table(
                        f"Sub-Layer Drilldown for Composite Layer: {e['name']} (Official EBOPs: {e['ebops']:,.1f})",
                        ["Inner Layer Name", "Layer Type", "Inner Raw EBOPs"],
                        sub_rows
                    )

    # 5. Generate Visualizations
    output_dir = args.output_dir
    if output_dir is None:
        exp_output_dir = os.path.join(PROJECT_ROOT, ".agents/EXP-18-final/outputs/quantized")
        if os.path.exists(exp_output_dir):
            output_dir = exp_output_dir
        else:
            output_dir = os.path.dirname(args.model_path)

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, args.save_plot)

    # Plot setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.6, 1]})

    # Palette
    unique_groups = list(set(e["group"] for e in ebop_entries))
    cmap = plt.get_cmap("tab10")
    color_map = {grp: cmap(i % 10) for i, grp in enumerate(unique_groups)}

    # Subplot 1: Bar chart per layer
    layer_names = [f"{e['name']} ({e['type']})" for e in ebop_entries]
    ebop_vals = [e["ebops"] for e in ebop_entries]
    bar_colors = [color_map[e["group"]] for e in ebop_entries]

    y_pos = np.arange(len(layer_names))
    bars = ax1.barh(y_pos, ebop_vals, color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.85)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(layer_names, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Effective Bit Operations (EBOPs)", fontsize=11, fontweight="bold")
    ax1.set_title("EBOP Breakdown per Layer", fontsize=13, fontweight="bold", pad=12)
    ax1.grid(True, which="both", axis="x", linestyle=":", alpha=0.6)

    max_val = max(ebop_vals) if ebop_vals else 1.0
    for bar, val, pct in zip(bars, ebop_vals, [e["pct"] for e in ebop_entries]):
        if val > 0:
            ax1.text(
                bar.get_width() + max_val * 0.01,
                bar.get_y() + bar.get_height() / 2.0,
                f"{val:,.0f} ({pct:.1f}%)",
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="bold"
            )

    # Subplot 2: Donut / Pie chart by Architectural Component
    grp_labels = list(group_totals.keys())
    grp_vals = [group_totals[g] for g in grp_labels]
    grp_colors = [color_map[g] for g in grp_labels]

    wedges, texts, autotexts = ax2.pie(
        grp_vals,
        labels=grp_labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=grp_colors,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
        pctdistance=0.75
    )

    for text in texts:
        text.set_fontsize(9)
        text.set_fontweight="bold"
    for autotext in autotexts:
        autotext.set_fontsize(8.5)
        autotext.set_fontweight="bold"

    ax2.set_title(f"EBOP Distribution by Section\n(Total: {total_ebops:,.0f} EBOPs)", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[HGQ EBOP Analyzer] Plot successfully saved to: {plot_path}")


if __name__ == "__main__":
    main()

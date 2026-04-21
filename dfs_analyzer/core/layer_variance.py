"""
Layer variance analysis: how within-layer variance changes with BFS distance.

Usage:
    from dfs_analyzer.core.layer_variance import run_layer_variance_analysis
    result = run_layer_variance_analysis("data_output/my_exp")
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def analyze_layer_variance(layers, layer_stds, layer_means, n,
                           graph_name="", output_path=None):
    """
    Two-panel plot of within-layer spread:
      left  – std dev of vertex means per layer
      right – coefficient of variation (CV = std/mean × 100 %)

    Returns a dict with arrays and the figure.
    """
    L    = np.array(layers,      dtype=float)
    stds = np.array(layer_stds,  dtype=float)
    means = np.array(layer_means, dtype=float)
    cvs  = np.where(means > 0, stds / means * 100.0, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.scatter(L, stds, s=40, color='steelblue', zorder=10)
    ax1.plot(L, stds, color='steelblue', linewidth=1.5, alpha=0.5)
    ax1.set_xlabel('BFS Layer', fontsize=12)
    ax1.set_ylabel('Within-Layer Std Dev', fontsize=12)
    ax1.set_title(f'{graph_name} — Within-Layer Std Dev', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(L, cvs, s=40, color='darkorange', zorder=10)
    ax2.plot(L, cvs, color='darkorange', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('BFS Layer', fontsize=12)
    ax2.set_ylabel('Coefficient of Variation (%)', fontsize=12)
    ax2.set_title(f'{graph_name} — CV per Layer', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Layer Variance Analysis — {graph_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return {'layers': L, 'stds': stds, 'cvs': cvs, 'fig': fig}


def run_layer_variance_analysis(data_dir, graph_name="", output_dir=None):
    """
    Run layer variance analysis on an experiment output directory.

    Expects layer_statistics_bfs.csv with columns Layer, Mean, and optionally Std.
    """
    csv_path = os.path.join(data_dir, 'layer_statistics_bfs.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No layer_statistics_bfs.csv in {data_dir}")

    df = pd.read_csv(csv_path)
    df = df[df['Layer'] > 0].copy()

    layers = df['Layer'].values
    means  = df['Mean'].values
    stds   = df['Std'].values if 'Std' in df.columns else np.zeros(len(layers))

    from dfs_analyzer.core.deviation_analysis import _infer_n
    n = _infer_n(data_dir, df)

    out_path = os.path.join(output_dir, 'layer_variance.png') if output_dir else None
    return analyze_layer_variance(layers, stds, means, n, graph_name, out_path)

"""
Deviation analysis: fit cubic polynomial to layer deviations from (n-1)/2.

Usage:
    from dfs_analyzer.core.deviation_analysis import run_deviation_analysis
    fig, result, layers, means, n = run_deviation_analysis("data_output/my_exp")
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Core fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_deviation_polynomial(layers, layer_means, n):
    """
    Fit a cubic polynomial to deviation(L) = mean(L) - (n-1)/2.

    Returns a dict with:
        a, b, c, d       – cubic coefficients (highest degree first)
        r_squared        – goodness of fit
        deviations       – observed deviation per layer
        fitted           – polynomial values at each layer
        expected         – (n-1)/2
    """
    expected = (n - 1) / 2
    L = np.array(layers, dtype=float)
    deviations = np.array(layer_means, dtype=float) - expected

    coeffs = np.polyfit(L, deviations, 3)
    a, b, c, d = coeffs
    fitted = np.polyval(coeffs, L)

    ss_res = np.sum((deviations - fitted) ** 2)
    ss_tot = np.sum((deviations - np.mean(deviations)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        'a': a, 'b': b, 'c': c, 'd': d,
        'r_squared': r_squared,
        'deviations': deviations,
        'fitted': fitted,
        'expected': expected,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_deviation(layers, layer_means, n, graph_name="", output_path=None):
    """
    Two-panel plot:
      left  – layer means with (n-1)/2 baseline
      right – deviation from (n-1)/2 with cubic fit

    Returns (fig, result_dict).
    """
    result = fit_deviation_polynomial(layers, layer_means, n)
    L = np.array(layers, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left – layer means
    ax1 = axes[0]
    ax1.scatter(L, layer_means, s=40, color='steelblue', zorder=10, label='Layer means')
    ax1.axhline(result['expected'], color='red', linestyle='--', linewidth=2,
                label=f'(n-1)/2 = {result["expected"]:.0f}')
    ax1.set_xlabel('BFS Layer', fontsize=12)
    ax1.set_ylabel('Mean Discovery Number', fontsize=12)
    ax1.set_title(f'{graph_name} — Layer Means', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right – deviation + cubic fit
    ax2 = axes[1]
    ax2.scatter(L, result['deviations'], s=40, color='steelblue', zorder=10,
                label='Observed deviation')
    ax2.plot(L, result['fitted'], color='red', linewidth=2.5,
             label=f'Cubic fit  R²={result["r_squared"]:.4f}')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.4)
    ax2.set_xlabel('BFS Layer', fontsize=12)
    ax2.set_ylabel('Deviation from (n-1)/2', fontsize=12)
    ax2.set_title(f'{graph_name} — Deviation from (n-1)/2', fontsize=13, fontweight='bold')

    r = result
    coeff_text = (f'a = {r["a"]:.4e}\nb = {r["b"]:.4e}\n'
                  f'c = {r["c"]:.4e}\nd = {r["d"]:.4e}\nR² = {r["r_squared"]:.4f}')
    ax2.text(0.05, 0.95, coeff_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Deviation Analysis — {graph_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig, result


# ─────────────────────────────────────────────────────────────────────────────
#  High-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_deviation_analysis(data_dir, graph_name="", output_dir=None):
    """
    Run deviation analysis on an experiment output directory.

    Reads layer_statistics_bfs.csv (columns: Layer, Mean).
    Infers n from summary.txt if available.

    Returns (fig, result, layers, means, n).
    """
    csv_path = os.path.join(data_dir, 'layer_statistics_bfs.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No layer_statistics_bfs.csv in {data_dir}")

    df = pd.read_csv(csv_path)
    df = df[df['Layer'] > 0].copy()
    layers = df['Layer'].values
    means  = df['Mean'].values
    n      = _infer_n(data_dir, df)

    out_path = os.path.join(output_dir, 'deviation_analysis.png') if output_dir else None
    fig, result = plot_deviation(layers, means, n, graph_name, out_path)
    return fig, result, layers, means, n


def _infer_n(data_dir, df):
    """Read n from summary.txt, or fall back to counting the Count column."""
    summary = os.path.join(data_dir, 'summary.txt')
    if os.path.exists(summary):
        with open(summary) as f:
            for line in f:
                for kw in ('Vertices:', 'vertices:', 'n =', 'n=', 'Total vertices:'):
                    if kw in line:
                        try:
                            part = line.split(kw, 1)[1].strip().split()[0]
                            return int(part.replace(',', ''))
                        except Exception:
                            pass
    if 'Count' in df.columns:
        return int(df['Count'].sum())
    return int(2 * float(df['Mean'].values[-1]))

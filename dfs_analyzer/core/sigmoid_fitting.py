"""
Sigmoid model fitting with L, sqrt(L), and log(L) transformations.

Usage:
    from dfs_analyzer.core.sigmoid_fitting import run_sigmoid_fitting
    best, results, fig, layers, means, n = run_sigmoid_fitting("data_output/my_exp")
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


# ─────────────────────────────────────────────────────────────────────────────
#  Model functions
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid_L(L, A, k):
    return A / (1.0 + np.exp(-k * L))

def _sigmoid_sqrtL(L, A, k):
    return A / (1.0 + np.exp(-k * np.sqrt(np.maximum(L, 0.0))))

def _sigmoid_logL(L, A, k):
    return A / (1.0 + np.exp(-k * np.log(np.maximum(L, 1e-10))))

TRANSFORMS = {
    'L':       (_sigmoid_L,     'f(L) = L'),
    'sqrt(L)': (_sigmoid_sqrtL, 'f(L) = √L'),
    'log(L)':  (_sigmoid_logL,  'f(L) = log(L)'),
}


# ─────────────────────────────────────────────────────────────────────────────
#  Fitting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fit_one(func, L, y, n):
    expected = (n - 1) / 2
    A_init   = float(y[-1]) if len(y) > 0 else expected
    try:
        popt, _ = curve_fit(
            func, L, y,
            p0=[A_init, 0.1],
            bounds=([expected * 0.5, 1e-6], [expected * 2.0, 50.0]),
            maxfev=5000,
        )
        A, k   = popt
        y_pred = func(L, A, k)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {
            'A': A, 'k': k, 'r2': r2,
            'rmse': float(np.sqrt(np.mean((y - y_pred) ** 2))),
            'predictions': y_pred,
            'success': True,
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def fit_sigmoid_transforms(layers, layer_means, n, graph_name="", output_path=None):
    """
    Fit sigmoid A/(1+e^{-k·f(L)}) with f = L, sqrt(L), log(L).

    Returns (best_transform_name, results_dict, fig).
    """
    L = np.array(layers,      dtype=float)
    y = np.array(layer_means, dtype=float)
    expected = (n - 1) / 2

    results = {}
    for name, (func, label) in TRANSFORMS.items():
        res = _fit_one(func, L, y, n)
        res['label'] = label
        results[name] = res

    successful = [(name, r) for name, r in results.items() if r['success']]
    best_name  = max(successful, key=lambda x: x[1]['r2'])[0] if successful else None

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    L_smooth  = np.linspace(L.min(), L.max(), 300)

    for ax, (name, (func, label)) in zip(axes, TRANSFORMS.items()):
        r = results[name]
        ax.scatter(L, y, s=40, color='steelblue', zorder=10, label='Data')
        ax.axhline(expected, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'(n-1)/2 = {expected:.0f}')

        if r['success']:
            ax.plot(L_smooth, func(L_smooth, r['A'], r['k']),
                    color='red', linewidth=2.5, label='Sigmoid fit')
            star = '  ★ BEST' if name == best_name else ''
            info = f'A = {r["A"]:.0f}\nk = {r["k"]:.4f}\nR² = {r["r2"]:.4f}{star}'
            ax.text(0.05, 0.05, info, transform=ax.transAxes, fontsize=9,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
        else:
            ax.text(0.5, 0.5, 'Fit failed', ha='center', va='center',
                    transform=ax.transAxes, color='red', fontsize=11)

        ax.set_xlabel('BFS Layer (L)', fontsize=11)
        ax.set_ylabel('Mean Discovery Number', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Sigmoid Fitting — {graph_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return best_name, results, fig


def run_sigmoid_fitting(data_dir, graph_name="", output_dir=None):
    """
    Run sigmoid fitting on an experiment output directory.

    Returns (best_name, results, fig, layers, means, n).
    """
    csv_path = os.path.join(data_dir, 'layer_statistics_bfs.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No layer_statistics_bfs.csv in {data_dir}")

    df = pd.read_csv(csv_path)
    df = df[df['Layer'] > 0].copy()
    layers = df['Layer'].values
    means  = df['Mean'].values

    from dfs_analyzer.core.deviation_analysis import _infer_n
    n = _infer_n(data_dir, df)

    out_path = os.path.join(output_dir, 'sigmoid_fitting.png') if output_dir else None
    best_name, results, fig = fit_sigmoid_transforms(layers, means, n, graph_name, out_path)
    return best_name, results, fig, layers, means, n

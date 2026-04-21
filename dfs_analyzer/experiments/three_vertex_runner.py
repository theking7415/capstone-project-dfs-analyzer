"""
Three-Vertex Distribution Runner.

Runs RDFS repeatedly and collects full discovery-number distributions for
three user-specified vertices.  The resulting histograms reveal how the
shape of the distribution changes with BFS distance from the origin.

Usage:
    from dfs_analyzer.experiments.three_vertex_runner import ThreeVertexRunner
    runner = ThreeVertexRunner()
    result = runner.run(graph, vertices, labels, n_samples=10000, ...)
"""

import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class ThreeVertexRunner:
    """Collect full discovery-number distributions for three chosen vertices."""

    def run(self, graph, vertices, labels, n_samples=10_000, rng_seed=42,
            output_dir="data_output", graph_name="", progress_callback=None):
        """
        Parameters
        ----------
        graph            : Graph instance (any graph type).
        vertices         : list of three vertex identifiers.
        labels           : list of three display strings, e.g. ['L=1', 'L=60', 'L=80'].
        n_samples        : number of independent RDFS runs.
        rng_seed         : random seed for reproducibility.
        output_dir       : directory for saving the PNG.
        graph_name       : used in plot titles.
        progress_callback: optional callable(current, total).

        Returns
        -------
        dict with keys: collected, arrays, labels, vertices, n,
                        output_path, summary_stats.
        """
        from dfs_analyzer.core.rdfs import rdfs

        rng   = np.random.default_rng(rng_seed)
        start = graph.get_start_vertex()
        n     = graph.number_vertices()

        collected  = {v: [] for v in vertices}
        dist_stats = defaultdict(list)

        for i in range(n_samples):
            if progress_callback and i % 500 == 0:
                progress_callback(i, n_samples)
            dist_stats.clear()
            rdfs(graph, start, dist_stats=dist_stats, rng=rng)
            for v in vertices:
                if v in dist_stats and dist_stats[v]:
                    collected[v].append(dist_stats[v][-1])

        if progress_callback:
            progress_callback(n_samples, n_samples)

        arrays = [np.array(collected[v]) for v in vertices]

        # ── print summary ─────────────────────────────────────────────────────
        half = (n - 1) / 2
        print(f"\n{'':30} {labels[0]:>12} {labels[1]:>12} {labels[2]:>12}")
        print('-' * 70)
        rows = [
            ('mean',           [f'{a.mean():.1f}' for a in arrays]),
            ('std dev',        [f'{a.std():.1f}'  for a in arrays]),
            ('min',            [str(int(a.min()))  for a in arrays]),
            ('max',            [str(int(a.max()))  for a in arrays]),
            ('mean - (n-1)/2', [f'{a.mean()-half:+.1f}' for a in arrays]),
        ]
        for lbl_stat, vals in rows:
            print(f'{lbl_stat:30} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}')

        # ── build summary_stats dict ───────────────────────────────────────────
        summary_stats = {}
        for lbl, v, arr in zip(labels, vertices, arrays):
            summary_stats[lbl] = {
                'vertex': v,
                'mean': float(arr.mean()),
                'std':  float(arr.std()),
                'min':  int(arr.min()),
                'max':  int(arr.max()),
                'deviation_from_half': float(arr.mean() - half),
            }

        # ── figure ────────────────────────────────────────────────────────────
        os.makedirs(output_dir, exist_ok=True)
        colors = ['#2196F3', '#FF9800', '#E91E63']

        fig = plt.figure(figsize=(16, 14))
        gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)

        # top row – individual histograms
        for col, (lbl, v, arr, c) in enumerate(zip(labels, vertices, arrays, colors)):
            ax = fig.add_subplot(gs[0, col])
            n_bins  = min(80, max(10, len(set(arr.tolist()))))
            weights = np.ones(len(arr)) / len(arr) * 100
            ax.hist(arr, bins=n_bins, color=c, alpha=0.75,
                    edgecolor='white', linewidth=0.4, weights=weights)
            ax.axvline(arr.mean(),    color='black', linewidth=2, linestyle='--',
                       label=f'mean={arr.mean():.0f}')
            ax.axvline(half,          color='green', linewidth=2, linestyle=':',
                       label=f'(n-1)/2={int(half)}')
            ax.axvline(float(arr.min()), color='red', linewidth=1.5, linestyle='-',
                       alpha=0.7, label=f'min={int(arr.min())}')
            ax.set_title(f'{lbl}\nVertex {v}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Discovery number')
            ax.set_ylabel('Percentage (%)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # middle – overlaid comparison
        ax_main = fig.add_subplot(gs[1, :])
        for lbl, v, arr, c in zip(labels, vertices, arrays, colors):
            weights = np.ones(len(arr)) / len(arr) * 100
            ax_main.hist(arr, bins=100, color=c, alpha=0.45, weights=weights,
                         label=lbl.replace('\n', ' '))
            ax_main.axvline(arr.mean(), color=c, linewidth=2.5, linestyle='--')
        ax_main.axvline(half, color='green', linewidth=2.5, linestyle=':',
                        label=f'(n-1)/2 = {int(half)}', zorder=10)
        ax_main.set_xlabel('Discovery number', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('Percentage (%)',   fontsize=12, fontweight='bold')
        ax_main.set_title('Overlaid Distributions — All Three Vertices',
                          fontsize=12, fontweight='bold')
        ax_main.legend(fontsize=10)
        ax_main.grid(True, alpha=0.3)

        # bottom – zoomed early positions for the nearest vertex
        ax_zoom = fig.add_subplot(gs[2, :])
        arr0     = arrays[0]
        zoom_max = max(5, min(int(float(arr0.max()) * 0.05) + 1, 150))
        positions = np.arange(1, zoom_max + 1)
        pct       = np.array([np.sum(arr0 == p) for p in positions]) / len(arr0) * 100
        bar_cols  = ['#2196F3' if p % 2 == 1 else '#BDBDBD' for p in positions]
        ax_zoom.bar(positions, pct, color=bar_cols, edgecolor='white', linewidth=0.3)
        for p, pc in zip(positions, pct):
            if pc > 0.4:
                ax_zoom.text(p, pc + 0.05, f'{pc:.1f}%', ha='center',
                             va='bottom', fontsize=7, fontweight='bold')
        ax_zoom.set_xlabel(f'Discovery number (first {zoom_max} positions)',
                           fontsize=11, fontweight='bold')
        ax_zoom.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax_zoom.set_title(
            f'Zoomed: {labels[0]} vertex — early discovery positions',
            fontsize=11, fontweight='bold',
        )
        ax_zoom.set_xlim(0, zoom_max + 1)
        ax_zoom.grid(True, alpha=0.3, axis='y')

        fig.suptitle(
            f'Three-Vertex Discovery Distributions — {graph_name}\n'
            f'n = {n:,}  |  samples = {n_samples:,}',
            fontsize=13, fontweight='bold',
        )

        out_path = os.path.join(output_dir, 'three_vertex_distributions.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'\n✓ Saved to {out_path}')

        return {
            'collected':     collected,
            'arrays':        arrays,
            'labels':        labels,
            'vertices':      vertices,
            'n':             n,
            'output_path':   output_dir,
            'summary_stats': summary_stats,
            'fig':           fig,
        }

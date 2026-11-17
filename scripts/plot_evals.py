#!/usr/bin/env python3
"""Plot a saved eval_report.py JSON without running models.

Usage: python scripts/plot_evals.py evals/2026-11-14.json
Requires matplotlib. Writes PNG and SVG beside the input by default.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--out', type=Path, help='output filename stem')
    args = parser.parse_args()
    rows = json.loads(args.report.read_text())
    if not rows:
        parser.error('report contains no rows')
    stem = args.out or args.report.with_name(args.report.stem + '-chart')
    stem.parent.mkdir(parents=True, exist_ok=True)
    labels = [r['name'].removesuffix('.gguf').replace('SmolLM2-135M-Instruct-', 'Smol135M ').replace('Qwen3-0.6B-', 'Qwen0.6B ') for r in rows]
    y = np.arange(len(rows))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(17, max(6, len(rows) * .47)), sharey=True)
    colors = ['#167d9a', '#e49634']
    for ax, metric, title in zip(axes[:2], ['pp', 'tg'], ['Prefill pp128: relative speed', 'Decode tg64: relative speed']):
        for i, row in enumerate(rows):
            a, b = row.get(metric), row.get(metric + '_ref')
            if a is None or b is None or b <= 0:
                ax.text(.03, i, 'missing', va='center')
                continue
            value = a / b
            ax.barh(i, value, color=colors[value < 1], height=.65)
            ax.text(value + .025, i, f'{value:.2f}×', va='center', fontsize=9)
        ax.axvline(1, color='#555555', linestyle='--', linewidth=1)
        ax.set_xlim(0, max([r.get(metric, 0) / r[metric + '_ref'] for r in rows if r.get(metric) is not None and r.get(metric + '_ref', 0) > 0] or [1]) * 1.22)
        ax.set_title(title, loc='left', weight='bold')
        ax.set_xlabel('kLLM / llama.cpp · higher is faster')
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
    ax = axes[2]
    errors = []
    for i, row in enumerate(rows):
        a, b = row.get('ppl'), row.get('ppl_ref')
        if a is None or b is None or b <= 0:
            ax.text(0, i, 'missing', va='center')
            continue
        delta = 100 * (a / b - 1)
        errors.append(abs(delta))
        ax.barh(i, delta, color=colors[delta > 0], height=.65)
        ax.annotate(f'{delta:+.3f}%', (delta, i), xytext=(4 if delta >= 0 else -4, 0), textcoords='offset points', va='center', ha='left' if delta >= 0 else 'right', fontsize=9)
    limit = max(errors or [1]) * 1.5 + .1
    ax.set_xlim(-limit, limit)
    ax.axvline(0, color='#555555', linewidth=1)
    ax.set_title('Perplexity difference', loc='left', weight='bold')
    ax.set_xlabel('(kLLM / llama.cpp − 1) × 100 · closer to 0 is closer')
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle(f'kLLM vs llama.cpp — recorded evaluation {args.report.stem}', fontsize=17, weight='bold', x=.02, ha='left')
    fig.text(.02, .025, 'Source: ' + str(args.report) + '\nSpeed: pp128 / tg64, 4 threads, 3 repeats (see source report for aggregation). PPL: WikiText excerpt, 8 × 256.\nEngine fidelity on a short corpus; this does not measure instruction following, coding, or tool-use quality.', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .13, 1, .93), w_pad=2)
    for suffix in ['.png', '.svg']:
        target = stem.with_suffix(suffix)
        fig.savefig(target, dpi=170, facecolor='white')
        print(target)
    plt.close(fig)


if __name__ == '__main__':
    main()

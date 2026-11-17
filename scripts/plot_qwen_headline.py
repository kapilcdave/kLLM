#!/usr/bin/env python3
"""Render the documented Qwen3 MoE result; does not run models.

Data source: README.md and docs/MOE.md, 2026-11-16.
These notes are not a fresh, controlled benchmark dataset.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
BG, FG, MUTED, GREEN = '#0b1019', '#f4f7fb', '#9daabd', '#8df7b5'
fig = plt.figure(figsize=(16, 9), facecolor=BG)
fig.text(.065, .915, 'kLLM  /  LOCAL INFERENCE', color=GREEN, fontsize=15, weight='bold')
fig.text(.065, .805, '30B parameters. 16 GiB laptop.', color=FG, fontsize=37, weight='bold')
fig.text(.065, .748, 'Qwen3-30B-A3B  ·  Q4_K_M  ·  17.3 GiB model file', color=MUTED, fontsize=18)
fig.text(.79, .52, '~6×', color=GREEN, fontsize=77, weight='bold', ha='center')
fig.text(.79, .466, 'FASTER DECODE', color=FG, fontsize=16, weight='bold', ha='center')
fig.text(.79, .421, '5.5–6.1× reported range', color=MUTED, fontsize=12, ha='center')
ax = fig.add_axes([.065, .315, .54, .31], facecolor=BG)
ax.barh([1, 0], [10, 1.81], color=[GREEN, '#67758a'], height=.40)
ax.barh([1], [1], left=[10], height=.40, color=GREEN, alpha=.45)
ax.text(0, 1.35, 'kLLM', color=FG, weight='bold', fontsize=17)
ax.text(0, .35, 'llama.cpp', color=FG, fontsize=17)
ax.text(11.2, 1, '10–11', color=GREEN, fontsize=21, weight='bold', va='center')
ax.text(2.05, 0, '1.81', color=FG, fontsize=21, weight='bold', va='center')
ax.set_xlim(0, 13.3); ax.set_ylim(-.45, 1.7)
ax.set_xticks([0, 3, 6, 9, 12]); ax.tick_params(axis='x', colors=MUTED, length=0, labelsize=11)
ax.set_yticks([])
ax.set_xlabel('Decode tokens / second · higher is faster', color=MUTED, fontsize=12, labelpad=12)
for spine in ax.spines.values(): spine.set_visible(False)
ax.set_axisbelow(True); ax.grid(axis='x', color='#233041', linewidth=.7)
fig.text(.065, .205, 'Dell Precision 5520  ·  Xeon E3-1505M v6  ·  CPU inference', color=FG, fontsize=15)
fig.text(.065, .145, 'Warm-cache kLLM decode. Same model file and machine reported in project notes.', color=MUTED, fontsize=12)
fig.text(.065, .106, 'Source: README.md + docs/MOE.md, 16 Nov 2026. Documented measurements; no fresh matched rerun.', color=MUTED, fontsize=11)
fig.text(.065, .069, 'MoE: ~3.3B active parameters per token. Decode speed only; prefill and cold-start performance differ.', color=MUTED, fontsize=11)
for ext in ['png', 'svg']:
    out = ROOT / 'evals' / f'qwen30b-headline.{ext}'
    fig.savefig(out, dpi=180, facecolor=BG)
    print(out)

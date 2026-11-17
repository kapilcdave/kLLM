#!/usr/bin/env python3
"""Render three engine columns from Dell microbenchmarks and cited MoE notes."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
BG, PANEL, FG, MUTED, ACCENT = '#0b1019', '#141e2c', '#f3f7fc', '#a3b0c2', '#8df7b5'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=ROOT / 'evals/dell-4b/results.json')
    p.add_argument('--moe-results', type=Path, default=ROOT / 'evals/dell-30b/results.json')
    p.add_argument('--out', type=Path, default=ROOT / 'evals/dell-engine-comparison')
    args = p.parse_args()
    data = json.loads(args.results.read_text())
    summary = data['summary']
    moe = json.loads(args.moe_results.read_text())['summary'] if args.moe_results.exists() else None
    fig = plt.figure(figsize=(17, 11), facecolor=BG)
    def text(x, y, s, size=15, color=FG, **kw):
        fig.text(x, y, s, fontsize=size, color=color, **kw)
    text(.055, .94, 'kLLM  /  THE LAPTOP TEST', 16, ACCENT, weight='bold')
    text(.055, .877, 'Big models. Old laptop.', 40, weight='bold')
    text(.055, .832, 'Dell Precision 5520 · Xeon E3-1505M v6 · 16 GiB RAM · CPU inference', 16, MUTED)
    xs = [.46, .665, .865]
    for x, name in zip(xs, ['vLLM', 'llama.cpp', 'kLLM']):
        text(x, .755, name, 25, ACCENT if name == 'kLLM' else FG, ha='center', weight='bold')
    # Both model sections use fresh measurements and identical aggregation.
    rect = FancyBboxPatch((.045, .505), .91, .203, boxstyle='round,pad=0.012',
                          transform=fig.transFigure, facecolor=PANEL, edgecolor='#28394a')
    fig.add_artist(rect)
    text(.065, .674, 'Qwen3-30B-A3B', 21, weight='bold')
    text(.065, .644, 'Q4_K_M · 17.3 GiB file · fresh benchmark¹', 11, MUTED)
    if not moe or any(moe[eng]['successful_rounds'] != 3 for eng in ['llama.cpp', 'kllm']):
        raise SystemExit('30B results require three successful rounds per engine')
    for y, metric, label in [(.597, 'pp', 'Prefill pp128'), (.530, 'tg', 'Decode tg64')]:
        text(.065, y+.005, label, 15, weight='bold')
        text(.065, y-.017, 'tokens/sec · median of 3', 10, MUTED)
        text(xs[0], y, 'N/A²', 23, MUTED, ha='center')
        winner = max(['llama.cpp', 'kllm'], key=lambda eng: moe[eng][metric]['median'])
        for x, eng in zip(xs[1:], ['llama.cpp', 'kllm']):
            r = moe[eng][metric]
            text(x, y, f"{r['median']:.1f}", 27, ACCENT if eng == winner else FG, ha='center', weight='bold')
            text(x, y-.023, f"range {r['min']:.1f}–{r['max']:.1f}", 9, MUTED, ha='center')
    text(.065, .464, 'Qwen3.5-4B', 21, weight='bold')
    text(.065, .432, 'Q4_K_M · fresh CPU microbenchmark³', 12, MUTED)
    for y, metric, label in [(.365, 'pp', 'Prefill pp128'), (.245, 'tg', 'Decode tg64')]:
        text(.065, y+.012, label, 16, weight='bold')
        text(.065, y-.019, 'tokens/sec · median of 3 rounds', 11, MUTED)
        text(xs[0], y, 'N/A²', 24, MUTED, ha='center')
        winner = max(['llama.cpp', 'kllm'], key=lambda eng: summary[eng][metric]['median'])
        for x, engine in zip(xs[1:], ['llama.cpp', 'kllm']):
            r = summary[engine][metric]
            text(x, y, f"{r['median']:.1f}", 29, ACCENT if engine == winner else FG,
                 ha='center', weight='bold' if engine == winner else 'normal')
            text(x, y-.035, f"range {r['min']:.1f}–{r['max']:.1f}", 11, MUTED, ha='center')
    text(.055, .141, '¹ Fresh pp128 / tg64, 3 interleaved rounds. llama.cpp --repack 0 avoids a large heap copy.', 11, MUTED)
    text(.055, .112, '² vLLM: M1200 CUDA unsupported; GGUF unsupported on x86 CPU per upstream docs. Not a zero score.', 11, MUTED)
    text(.055, .083, '³ 4 threads, CPU only; kLLM bench vs llama-bench b12099. Equivalent sizes; prompts / decode contexts differ.', 11, MUTED)
    text(.055, .054, 'Background workloads active; no cache flushing. Green marks faster results. Commands, hashes, logs: evals/dell-{4b,30b}/.', 11, MUTED)
    for suffix in ['.png', '.svg']:
        path = args.out.with_suffix(suffix)
        fig.savefig(path, dpi=160, facecolor=BG)
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()

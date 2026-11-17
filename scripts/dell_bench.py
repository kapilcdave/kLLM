#!/usr/bin/env python3
"""Interleave CPU pp128/tg64 benchmarks; retain commands, raw logs and medians.

python scripts/dell_bench.py --model /path/model.gguf --out evals/dell-4b
Each engine uses its own synthetic prompt/warmup: equivalent workload sizes,
not identical token streams. kLLM tg includes its prefilled prompt; llama-bench
runs standalone tg. These are microbenchmarks, not matched chat latency.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--llama-repack', choices=['0', '1'], default='1')
    p.add_argument('--llama-bench', default='/home/kapil/ai/llama.cpp/cpu/llama-b12099/llama-bench')
    args = p.parse_args()
    if args.rounds < 1:
        p.error('rounds must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    binary = args.out.resolve() / 'kllm-snapshot'
    shutil.copy2(ROOT / 'target/release/kllm', binary)
    data = {'model': str(args.model.resolve()), 'model_sha256': digest(args.model), 'binary_sha256': digest(binary),
            'kllm_binary_sha256': digest(binary), 'date': time.strftime('%Y-%m-%d'),
            'hardware': 'Dell Precision 5520 / Xeon E3-1505M v6 / 16 GiB / CPU, 4 threads',
            'workload': 'pp128 / tg64, one repetition per process; alternating engine order',
            'notes': __doc__, 'samples': [], 'summary': {},
            'vllm': {'status': 'not_benchmarked', 'cuda': 'M1200 compute capability 5.0 below supported minimum',
                     'cpu': 'AVX2 available but GGUF unsupported on x86 CPU per upstream quantization matrix',
                     'source': 'https://docs.vllm.ai/en/latest/getting_started/installation/cpu/'}}
    commands = {
        'kllm': [str(binary), 'bench', str(args.model), '-p', '128', '-n', '64', '-t', '4', '-r', '1', '--gpu-layers', '0'],
        'llama.cpp': [args.llama_bench, '-m', str(args.model), '-p', '128', '-n', '64', '-t', '4', '-r', '1', '-ngl', '0', '-o', 'json', '--repack', args.llama_repack],
    }
    for round_id in range(args.rounds):
        for engine in (['kllm', 'llama.cpp'] if round_id % 2 == 0 else ['llama.cpp', 'kllm']):
            print(f'round {round_id + 1}: {engine}', flush=True)
            command = commands[engine]
            started = time.monotonic()
            sample = {'engine': engine, 'round': round_id + 1, 'command': command}
            raw = ''
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
                raw = result.stdout + '\n' + result.stderr
                sample['exit_code'] = result.returncode
                if result.returncode:
                    raise ValueError(f'exit {result.returncode}')
                if engine == 'kllm':
                    sample.update({key: float(re.search(pattern, raw)[1]) for key, pattern in
                                   [('pp', r'pp128:\s+([\d.]+)'), ('tg', r'tg64:\s+([\d.]+)')]})
                else:
                    for row in json.loads(result.stdout):
                        sample['pp' if row['n_prompt'] else 'tg'] = row['avg_ts']
                    if not all(key in sample for key in ['pp', 'tg']):
                        raise ValueError('missing benchmark metrics')
            except (subprocess.TimeoutExpired, ValueError, TypeError, AttributeError, KeyError) as e:
                raw = raw if not isinstance(e, subprocess.TimeoutExpired) else str(e)
                sample['error'] = str(e)
            sample['wall_seconds'] = time.monotonic() - started
            name = f'{round_id + 1}-{engine}.log'
            (args.out / name).write_text(raw)
            sample['raw_log'] = name
            data['samples'].append(sample)
            for eng in commands:
                good = [s for s in data['samples'] if s['engine'] == eng and not s.get('error')]
                if good:
                    data['summary'][eng] = {metric: {'median': statistics.median(s[metric] for s in good),
                                                     'min': min(s[metric] for s in good),
                                                     'max': max(s[metric] for s in good)} for metric in ['pp', 'tg']}
                    data['summary'][eng]['successful_rounds'] = len(good)
            (args.out / 'results.json').write_text(json.dumps(data, indent=2) + '\n')
            print(sample, flush=True)
    if any(s.get('error') for s in data['samples']):
        raise SystemExit('one or more benchmark attempts failed; see raw logs')


if __name__ == '__main__':
    main()

# Interleaved CPU comparison, 2026-11-17

This run tests the claim that kLLM is faster than llama.cpp on the reference
Dell Precision 5520. The result does **not** support a blanket speed claim.
kLLM leads prompt processing on three SmolLM2 weight formats; llama.cpp leads
generation on every tested model and leads both metrics on Qwen3-0.6B.

## Method

- CPU only, four threads, pinned to physical cores 0-3 with `taskset`.
- kLLM commit `29e7ed6`; llama.cpp build b12099.
- SmolLM2 and Qwen3-0.6B: pp512/tg128, five alternating process rounds,
  three timed repetitions per process.
- Qwen3-30B-A3B: pp128/tg64, three alternating process rounds, two timed
  repetitions per process. llama.cpp uses `--repack 0`; its default repacked
  copy does not fit comfortably beside this 17.3 GiB file in 16 GiB RAM.
- The table reports the median of each process's mean throughput. Every
  `results.json` includes exact commands, binary and model SHA-256 hashes,
  host state before each sample, all samples, and raw-log filenames.
- The laptop was on AC power with the `powersave` governor. Other user
  workloads remained active. One Q4_0 kLLM process was heavily interrupted;
  process-level medians limit its effect, but this is not an exclusive-hardware
  laboratory run.

The engines use equivalent workload sizes, not identical synthetic token
streams or internal batches. Treat this as an engine microbenchmark, not a
matched end-to-end chat latency test.

## Results

| Model | Workload | kLLM | llama.cpp | kLLM / llama.cpp |
|---|---|---:|---:|---:|
| SmolLM2-135M Q4_0 | prompt processing | 677.6 | 621.7 | **1.090x** |
|  | generation | 124.0 | 173.2 | 0.716x |
| SmolLM2-135M Q6_K | prompt processing | 637.2 | 610.8 | **1.043x** |
|  | generation | 112.1 | 152.5 | 0.735x |
| SmolLM2-135M Q8_0 | prompt processing | 685.9 | 631.8 | **1.086x** |
|  | generation | 111.9 | 146.1 | 0.766x |
| Qwen3-0.6B Q4_K_M | prompt processing | 211.7 | 219.6 | 0.964x |
|  | generation | 35.5 | 52.7 | 0.674x |
| Qwen3-30B-A3B Q4_K_M | prompt processing | 11.6 | 11.8 | 0.985x |
|  | generation | 4.4 | 6.3 | 0.704x |

All values are tokens/second. “Prompt processing” is llama-bench `pp`; it
controls time to first output on a long prompt. “Generation” is `tg`; it is
the rate at which the answer appears after prompt processing.

## Conclusion

kLLM's demonstrated advantage is memory behavior, not universal throughput.
It runs the 17.3 GiB Qwen3-30B-A3B file with its default planner on a 16 GiB
machine. llama.cpp can also run the comparison when explicitly configured
with `--repack 0`, and it is faster at generation in this test. kLLM has
small, repeatable prompt-processing leads on the tested SmolLM2 formats.

Before publishing a stronger speed claim, repeat this run with exclusive use
of the laptop and a fixed performance governor. The checked-in script is
`scripts/credible_cpu_bench.py`.

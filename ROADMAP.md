# kLLM roadmap

## Current release: 0.1

The 0.1 line proves that kLLM can inspect a model, plan memory conservatively,
and execute a minimal Llama-family model without exceeding a configured budget.

| Milestone | Deliverable | Exit condition | Status |
|---|---|---|---|
| M1 | Deterministic memory planner | Bounded plans and passing tests | Done |
| M2 | GGUF inspector | Metadata and tensor inventory without loading weights | In progress |
| M3 | CPU reference backend | Greedy token generation matches a fixture | Planned |
| M4 | Q4_0 kernels | Quantized operations match float reference tolerances | Planned |
| M5 | Legacy CUDA backend | Q4_0 decode runs without tensor cores | Planned |
| M6 | Hybrid placement | Planned layers execute across CPU and GPU | Planned |
| M7 | Runtime CLI | Tokenization, sampling, streaming, and benchmarks | Planned |

## Immediate work queue

1. Complete GGUF type coverage and malformed-file fixtures.
2. Feed real tensor sizes into the memory planner instead of uniform estimates.
3. Define a small backend trait and a plain-f32 CPU reference implementation.
4. Add a reproducible correctness fixture for one tiny Llama-family model.
5. Profile only after end-to-end generation is correct.

## Release gates

- All file offsets and allocation arithmetic are checked.
- CPU-only builds require neither CUDA nor an NVIDIA driver.
- Model weights are never copied wholesale merely to inspect a GGUF file.
- Benchmark results include the exact command, model hash, and hardware details.


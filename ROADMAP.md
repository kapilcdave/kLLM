# kLLM roadmap

## Current release: 0.1

The 0.1 line proves that kLLM can inspect a model, plan memory conservatively,
and execute Llama-family models without exceeding a configured budget.

| Milestone | Deliverable | Exit condition | Status |
|---|---|---|---|
| M1 | Deterministic memory planner | Bounded plans and passing tests | Done |
| M2 | GGUF parser | Metadata and tensor inventory without loading weights; malformed files rejected | Done |
| M3 | CPU reference backend | Greedy generation matches llama.cpp on a fixture | Done (F16 token-exact) |
| M4 | Quantized kernels | Quantized operations match references; perplexity within llama.cpp's own spread | Done (Q4_0..Q8_0, Q4_K, Q5_K, Q6_K) |
| M5 | Legacy CUDA backend | Q4_0 decode runs without tensor cores | Planned |
| M6 | Hybrid placement | Planned layers execute across CPU and GPU | Planned |
| M7 | Runtime CLI | Tokenization, sampling, streaming, and benchmarks | Done (CPU) |

## Immediate work queue

1. **Prefill GEMM.** Prefill runs at about half of llama.cpp. Replace the
   per-(row, column) reductions with row-interleaved tiles (e.g. 8 rows x 4
   columns) so block scales live in one vector and no horizontal sums are
   needed per output.
2. **CUDA backend (M5).** Device probe, allocation guard, Q4_0/Q8_0 matvec on
   Maxwell (sm_50, no tensor cores), tested against `CpuBackend` through the
   `Backend` trait.
3. **Hybrid execution (M6).** Run the planner's GPU suffix of layers plus the
   output head on the device; one activation transfer each way per token.
4. **Q3_K kernels**, the remaining common K-quant.
5. **More architectures**: Gemma, Phi-3, and the hybrid Qwen3.5, whose
   linear-attention layers need new kernels.
6. **Real-model checks** for the kernel-tested-only types (BF16, Q5_0, Q5_1,
   Q5_K, Q2_K).
7. `serve`: an OpenAI-compatible local HTTP endpoint (spec milestone 8).

## Release gates

- All file offsets and allocation arithmetic are checked.
- CPU-only builds require neither CUDA nor an NVIDIA driver.
- Model weights are never copied wholesale merely to inspect a GGUF file.
- Benchmark results include the exact command, model hash, and hardware details.
- Correctness is checked against llama.cpp, not only against kLLM itself.

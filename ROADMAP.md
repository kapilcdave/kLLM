# kLLM roadmap

## Current release: 0.1

The 0.1 line proves that kLLM can inspect a model, plan memory conservatively,
and execute Llama-family models without exceeding a configured budget.

| Milestone | Deliverable | Exit condition | Status |
|---|---|---|---|
| M1 | Deterministic memory planner | Bounded plans and passing tests | Done |
| M2 | GGUF parser | Metadata and tensor inventory without loading weights; malformed files rejected | Done |
| M3 | CPU reference backend | Greedy generation matches llama.cpp on a fixture | Done (F16 token-exact) |
| M4 | Quantized kernels | Quantized operations match references; perplexity within llama.cpp's own spread | Done; every type also checked on a real model by KL divergence |
| M5 | Legacy CUDA backend | Q4_0 decode runs without tensor cores | Kernels done and tested; not yet in the forward pass |
| M6 | Hybrid placement | Planned layers execute across CPU and GPU | Planned |
| M7 | Runtime CLI | Tokenization, sampling, streaming, and benchmarks | Done (CPU) |

## Immediate work queue

1. **Decode bandwidth.** Decode is the remaining gap: 81-90% of llama.cpp,
   where prefill is now 1.2-1.7x. It is bandwidth-bound and reaches about
   17.8 GB/s against llama.cpp's 20.8 on the same machine, scaling 1.48x on
   four threads where pure compute scales 3.88x. Software prefetching the next
   weight group made it slightly worse, so the stream itself is fine. Two
   things left to try: a single contiguous arena for the x8-repacked weights
   instead of one allocation per tensor (210 of them per token, so the TLB
   takes ~21k misses that 2 MiB pages would remove — worth about 3% by
   estimate), and shrinking the Q4_K x8 layout, which is 1248 bytes per eight
   rows against 1152 row-major: in a bandwidth-bound loop that 8.3% is paid
   directly, and storing the unpacked mins as `i8` rather than `i16` would
   recover most of it.
2. **Hybrid execution (M6).** Run the planner's GPU suffix of layers plus the
   output head on the device; one activation transfer each way per token.
   Before building it, note what `examples/gpu_bench` measured on the
   reference machine: the quantized CUDA matvecs manage 0.56-0.93 GB/s against
   F16's 19 GB/s, because every lane walks a whole block byte by byte and the
   loads do not coalesce. Repacking the weights into a struct-of-arrays layout
   on upload (scales in one array, quants in another, as the CPU side already
   does for x8) is the fix, and it should come first. Also worth knowing: this
   laptop's M1200 is clamped to 287 MHz on a 65 W charger — a quarter of its
   1150 MHz — while its memory stays at full 2505 MHz, so GPU work here is
   instruction-starved rather than bandwidth-starved, and kernels should be
   judged on instructions per byte.
3. **Q3_K kernels**, the remaining common K-quant.
4. **More architectures**: Gemma, Phi-3, and the hybrid Qwen3.5, whose
   linear-attention layers need new kernels.
5. `serve`: an OpenAI-compatible local HTTP endpoint (spec milestone 8).

## Closed

- **Prefill GEMM.** Prefill was at 65% of llama.cpp. The kernels were never
  the problem -- at one thread kLLM already beat llama.cpp, and the Q4_0
  packed kernel runs at ~22 MAC/cycle, near the `vpmaddubsw` port limit. The
  bottleneck was that activation quantization ran on the calling thread before
  any worker woke: 48% of a 1536x576 matmul and 71% of a 576x1536 one. Making
  it parallel and then vectorizing it took prefill to 1.2-1.7x llama.cpp.
- **Real-model checks** for every weight type, by perplexity and KL
  divergence (`tests/divergence.rs`).

## Release gates

- All file offsets and allocation arithmetic are checked.
- CPU-only builds require neither CUDA nor an NVIDIA driver.
- Model weights are never copied wholesale merely to inspect a GGUF file.
- Benchmark results include the exact command, model hash, and hardware details.
- Correctness is checked against llama.cpp, not only against kLLM itself.

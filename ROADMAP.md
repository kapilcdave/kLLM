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
| M5 | Legacy CUDA backend | Q4_0 decode runs without tensor cores | Done |
| M6 | Hybrid placement | Planned layers execute across CPU and GPU | Done (`--gpu-layers N|auto`); slower than the CPU at the 65 W charger clock |
| M7 | Runtime CLI | Tokenization, sampling, streaming, and benchmarks | Done (CPU) |

## Immediate work queue

0. **Disk tier: done.** Models larger than the RAM budget keep the head and a
   block prefix resident and stream the rest, with read-ahead and immediate
   release (README, "Models bigger than RAM"). It is 1.45x llama.cpp on
   Qwen3-14B Q4_K_M in a 6 GiB cgroup, running at 95% of disk bandwidth. An
   open question: near the edge (a model barely over MemAvailable), streaming
   loses to plain mmap because the kernel can reclaim other caches. Deciding
   that at run time, from observed major faults, would beat a static budget.

1. **Routed decode compute.** Mixture-of-experts runs (`qwen3moe`, designed
   in [docs/MOE.md](docs/MOE.md)): Qwen3-30B-A3B Q4_K_M, a 17.3 GiB model on
   a 16 GiB laptop, at **11.3 tok/s against llama.cpp's 1.81** -- 6x, up from
   1.4 -- with perplexity 13.0682 against llama.cpp's 13.0579 (+0.08%, and
   its own flash-attention spread on this model is 1.23%).

   Decode is now compute-bound, not traffic-bound, so the next lever is the
   routed path itself: one parallel region per layer instead of `k` separate
   matmuls, which also shares the activation quantization across experts
   rather than repeating it 16 times per layer. This model wants ~2.1 G MAC
   per token, which is where the ~10-14 tok/s ceiling comes from.

   An explicit expert cache is deliberately *not* next. The locality curve
   (an LRU over (layer, expert) reaching 61% at 1.9 GiB and 92% at 7.2 GiB,
   with the hottest 10% of a layer's experts taking 38% of its reads) said
   caching pays -- and the kernel's own page-cache LRU already delivers it,
   once the planner stopped reserving budget for expert pages it never
   prefetched. Building a second cache on top would be replacing a free one
   that already reaches the compute ceiling.

2. **GPU speed.** M6 works (`tests/gpu.rs`: perplexity matches llama.cpp
   on the GPU as on the CPU), and the quantized matvecs were rewritten to be
   coalesced over a struct-of-arrays layout (Q4_0 0.80 -> 5.9 GB/s, Q8_0
   0.92 -> 10.2 GB/s, batch-64 matmul 1.8 -> 22 GOP/s). But this laptop's
   M1200 is clamped to 287 MHz on a 65 W charger -- a quarter of its 1150 MHz
   -- while its memory stays at full 2505 MHz. At that clock the GPU is the
   slower device: SmolLM2-135M Q4_0 decodes at 37 tok/s on it against 215 on
   the CPU, and Qwen3-0.6B Q4_K_M at 10.5 against 46. Next steps, in order:
   re-measure with the 130 W charger; then cut instructions per byte
   further (the Q4_0 kernel costs about what F16's does per element, so it
   is ALU-bound here); then fuse the small per-layer kernels (about 15
   launches per block at ~5 µs each).
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
  it parallel and then vectorizing it took prefill to 1.22-1.65x llama.cpp on
  every quantized type. See `evals/` for the full table.
- **Decode bandwidth.** Decode is at parity or better on every weight type
  and there is no headroom left to chase. Qwen3-0.6B Q4_K_M, the type that
  trailed longest, streams 22.2 GB/s of weights against a measured ceiling of
  23.4-23.8 GB/s on this machine (`examples/kernel_bench`, 4096x4096, four
  threads) -- 94% of what the memory system delivers -- and best-of-four lands
  at 56.8 tok/s against llama.cpp's 57.4, inside llama.cpp's own 4% spread.
  Q5_K_M is 1.08-1.18x and Q6_K 1.11-1.13x. What closed it was shrinking the
  Q4_K x8 layout (u8 mins, 1248 -> 1184 bytes per eight rows), not kernel
  instruction count. Tried and closed on this line: software prefetching the
  next weight group (slightly worse), and one contiguous huge-page arena for
  the repacked weights (noise, because transparent huge pages are already
  `always` here).
- **Prefill for the float types.** F16 and BF16 prefill were the last place
  kLLM lost to llama.cpp (0.58-0.61x). A float row has no x8-packed layout,
  so it was widened to f32 and reduced against one column at a time: one FMA
  per two loads. A 3x3 register tile (nine accumulators, 13 of 16 YMM
  registers; 4x3 spills) made it 1.11-1.15x. The tile only pays once there
  are columns to fill it -- at batch one, widening a row block to f32 is 3x
  the bytes of the f16 it came from, and it cost bandwidth-bound decode
  1.04x -> 0.82x before being gated on batch size.
- **Real-model checks** for every weight type, by perplexity and KL
  divergence (`tests/divergence.rs`). This is what found the two weight types
  that were quietly running on scalar code -- BF16 (no entry in the AVX2
  dispatch) and Q2_K (no kernel against quantized activations at all, which
  also explained a 2.3% perplexity gap previously written off as noise).

## Release gates

- All file offsets and allocation arithmetic are checked.
- CPU-only builds require neither CUDA nor an NVIDIA driver.
- Model weights are never copied wholesale merely to inspect a GGUF file.
- Benchmark results include the exact command, model hash, and hardware details.
- Correctness is checked against llama.cpp, not only against kLLM itself.

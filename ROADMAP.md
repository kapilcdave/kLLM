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
   Qwen3-14B Q4_K_M in a 6 GiB cgroup, running at 95% of disk bandwidth. Next
   on this line: Mixture-of-Experts, where a token reads only its active
   experts, so streaming cold experts from disk while caching hot ones costs
   a fraction of a dense model's traffic. kLLM does not run any MoE
   architecture yet (Qwen3-MoE would be the first). An open question: near
   the edge (a model barely over MemAvailable), streaming loses to plain
   mmap because the kernel can reclaim other caches. Deciding that at run
   time, from observed major faults, would beat a static budget.

1. **Decode bandwidth.** Decode is now parity or better on most types
   (Q8_0 1.18x, Q6_K 1.14x, Q4_0 1.05x) but trails on the K-quants that
   matter most: Qwen3-0.6B Q4_K_M at 0.88x and Q5_K_M at 0.94x. It is
   bandwidth-bound and scales 1.48x on four threads where pure compute
   scales 3.88x. Software prefetching the next weight group made it slightly
   worse, so the stream itself is fine. Tried and
   closed: one contiguous huge-page arena for the x8-repacked weights
   instead of a heap block per tensor measured as noise (-2% to +5% across
   interleaved runs) because transparent huge pages are already `always` on
   the reference machine; shrinking the Q4_K x8 layout from 1248 to 1184
   bytes per eight rows (u8 mins) gave +3-4% on Qwen3-0.6B Q4_K_M. The
   remaining K-quant gap is in the kernels themselves.
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
3. **Prefill for the float types.** F16 and BF16 prefill run at 0.58-0.61x
   of llama.cpp while every quantized type is now ahead of it (1.22-1.65x). The x8 packing
   covers only Q4_0, Q8_0, Q4_K and Q6_K, so a float row is dequantized to f32
   and then reduced once per activation column; llama.cpp runs an f16 GEMM at
   half the bytes with register tiling. Decode for these types is fine
   (1.0x), so this is specifically the batched path.
4. **Q3_K kernels**, the remaining common K-quant.
5. **More architectures**: Gemma, Phi-3, and the hybrid Qwen3.5, whose
   linear-attention layers need new kernels.
6. `serve`: an OpenAI-compatible local HTTP endpoint (spec milestone 8).

## Closed

- **Prefill GEMM.** Prefill was at 65% of llama.cpp. The kernels were never
  the problem -- at one thread kLLM already beat llama.cpp, and the Q4_0
  packed kernel runs at ~22 MAC/cycle, near the `vpmaddubsw` port limit. The
  bottleneck was that activation quantization ran on the calling thread before
  any worker woke: 48% of a 1536x576 matmul and 71% of a 576x1536 one. Making
  it parallel and then vectorizing it took prefill to 1.22-1.65x llama.cpp on
  every quantized type. See `evals/` for the full table.
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

# Mixture-of-Experts in kLLM

A design note, not a plan of record. It exists because MoE is the strongest
case kLLM has for its own thesis, and because most of the work is not the
part that looks hard.

## Why this matters more to kLLM than to a server engine

A dense decoder reads every weight once per token. An MoE reads the router,
the shared expert if there is one, and `k` of `N` experts. The ratio is the
whole argument:

| Model | Total | Active/token | Q4_K total | Q4_K active |
|---|---|---|---|---|
| Qwen3-30B-A3B | 30B | 3.3B | ~18 GiB | ~1.9 GiB |
| Qwen3.5-35B-A3B | 35B | ~3B | (IQ2_XXS: 9.9 GiB) | ~0.9 GiB |

On a 16 GiB machine a dense 30B is simply out of reach: every token needs all
18 GiB, so a disk tier streams 18 GiB per token and the thing runs at
one token every several seconds. An MoE needs ~1.9 GiB per token, and *which*
1.9 GiB is data-dependent but repeats. That is the difference between
"impossible" and "a caching problem", and caching problems are what kLLM's
planner and disk tier already exist for.

So the interesting claim is not "kLLM runs MoE" — it is "kLLM runs a 30B MoE
on a 16 GiB laptop at a useful rate, because it only resides the experts that
get used."

## What the format actually requires

Metadata, under the `{arch}.` prefix (verified against llama.cpp b12099):

| Key | Meaning |
|---|---|
| `expert_count` | `N`, experts per layer |
| `expert_used_count` | `k`, experts chosen per token |
| `expert_feed_forward_length` | each expert's FFN width, *not* `feed_forward_length` |
| `expert_shared_count`, `expert_shared_feed_forward_length` | an always-on expert alongside the routed ones |
| `expert_weights_norm` | renormalize the top-`k` router weights to sum 1 |
| `expert_weights_scale` | scale applied to the router weights |
| `expert_gating_func` | softmax or sigmoid (DeepSeek-V3 uses sigmoid) |

Tensors, per block:

| Tensor | Shape (GGML order) |
|---|---|
| `ffn_gate_inp.weight` | `[n_embd, n_expert]` — the router |
| `ffn_gate_exps.weight`, `ffn_up_exps.weight` | `[n_embd, n_ff_exp, n_expert]` |
| `ffn_down_exps.weight` | `[n_ff_exp, n_embd, n_expert]` |
| `ffn_{gate,up,down}_shexp.weight`, `ffn_gate_inp_shexp.weight` | the shared expert |

**The 3-D tensors are the good news.** `[n_embd, n_ff_exp, n_expert]` is
`n_expert` contiguous 2-D matrices of `[n_embd, n_ff_exp]`, so expert `e` is a
byte offset: `e * n_ff_exp * row_bytes(n_embd)`. `src/gguf.rs` already parses
3-D tensors and `TensorInfo::byte_size` already computes their size
correctly. Nothing in the parser needs touching, and the existing `Matrix`
type describes an expert slice unchanged.

What does need changing is small and well-bounded:

- `Loader::matrix` requires `dimensions == [cols, rows]` exactly. It needs a
  sibling that accepts a third dimension and returns the per-expert stride.
- `Layer` grows a variant: routed FFN (router + three expert blocks +
  optional shared expert) instead of the three dense matrices.
- `Model::weight_bytes` and `layer_bytes` currently assume every weight is
  read every token, which is exactly what stops being true. They feed the
  planner and the bench bandwidth figure, so both need an "active bytes"
  notion distinct from "resident bytes".

## The forward pass

Per token, per layer: `logits = ffn_gate_inp · x`; softmax or sigmoid; top-`k`;
optionally renormalize and scale; then `y = Σ_e w_e · SwiGLU_e(x)`, plus the
shared expert's unweighted contribution.

Two things here are easy to get quietly wrong, and both change output without
failing anything: the order of (top-k, normalize, scale) differs between
architectures, and ties in top-k selection have to break the same way
llama.cpp breaks them. This is precisely what `kllm compare` is for — a
router that picks a different 8th expert on 2% of tokens will show up as a KL
divergence against llama.cpp on the same file, and as nothing at all in a
perplexity check.

## The two real engineering problems

**Decode is a latency problem, not a bandwidth one.** Qwen3-30B-A3B has
`n_ff_exp = 768`, so one expert's `gate` is 768x2048 — about 0.8 MiB at Q4_K.
Per layer the engine does one router matvec plus 8 experts x 3 matvecs = 25
matmul calls where a dense model does 3. Over 48 layers that is ~1200 parallel
regions per token against a dense model's ~150. kLLM's pool dispatch is
0.47 µs, so that alone is sub-millisecond, but `MIN_TASK_BYTES` (48 KiB) would
carve each 0.8 MiB expert into ~16 tasks and the scheduler would spend its
time on bookkeeping. The right shape is one parallel region per layer over all
`k x rows` expert rows, not `k` separate `matmul_many` calls — i.e. the
`Job` list already supported by `matmul_many`, extended to let each job carry
its own input scaling.

**Prefill needs tokens sorted by expert.** Each of `B` tokens picks its own
`k` experts, so the naive loop does `B x k` narrow matvecs and throws away
every bit of the register tiling that just doubled float prefill. The standard
fix is a gather: for each expert, collect the tokens that chose it, run one
dense GEMM over that subset, scatter the results back weighted. That also
turns expert weight traffic from `B x k` reads into `N` reads per batch, which
matters enormously when the weights come from disk.

But note the arithmetic: the average tokens-per-expert is `B x k / N`. For
Qwen3-30B-A3B at the current default batch of 64, that is `64 x 8 / 128 = 4`
— still too narrow to tile well. MoE prefill wants a much larger batch (512
gives 32 per expert), which interacts with the planner's scratch budget. The
batch default should become architecture-aware.

And one correction to the obvious reading of "token sorting saves traffic".
It does not change *which* experts a batch reads — across 256 tokens picking
8 of 128, essentially all 128 get touched — only *how many times* each is
read. Naive, a 256-token chunk makes `256 x 8 = 2048` expert visits per layer
against ~128 distinct ones: 16x redundant. Sorted, it makes 128.

Whether that redundancy costs anything depends on a number worth having to
hand for Qwen3-30B-A3B at Q4_K: an expert is `3 x 768 x 2048` parameters,
about 2.6 MiB, so one layer's 128 experts are ~337 MiB and the whole model is
~16 GiB of experts. A single layer's expert set therefore fits in RAM *even
when the model does not*, which means the 16x redundancy lands on the page
cache rather than the disk, and sorting mainly buys back dispatch and compute
rather than bandwidth. That is a much weaker reason to do it than it first
appears, and it should be measured before being built — the same discipline
this note asks for everywhere else.

## The decisive unknown: expert locality

Everything above is ordinary work. The thing that decides whether the
headline claim is true is a question about data, not code: **across
single-stream decode, how concentrated is expert selection?**

If the hot working set is small and stable, an LRU over experts serves most
reads from RAM and a 30B MoE runs at a useful rate on this laptop. If
selection is close to uniform and shifts every token, then every token needs
~1.9 GiB from disk, NVMe gives ~2-3 GiB/s, and the answer is ~1 token/s no
matter how good the cache is.

Published routing analyses report real skew, but skew measured over a corpus
with large batches is not the same as temporal locality in one stream, and
the second is what a cache sees. So this should be measured, and it is cheap
to measure the moment the router works:

1. Implement the router and the MoE FFN with no caching at all, on a model
   that fits in RAM.
2. Log `(layer, token, expert[k])` for a few thousand tokens of real text.
3. Offline, replay the trace against LRU caches of various sizes and plot hit
   rate against cache bytes.

That is a day's work and it decides whether to build the expert cache at all.
Doing it before the cache, rather than after, is the whole point — the cache
is the expensive part and it is worthless if the hit rate is flat.

A secondary measurement from the same trace: whether the top-`k` set for
layer `L+1` is predictable from layer `L`'s hidden state early enough to
prefetch. If it is, the disk tier can issue reads ahead of need, which is
worth more than cache size.

## What to target first

**Not the file already on disk.** `Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf` looks
tempting but needs four new things at once: MoE, `IQ2_XXS` (a codebook quant,
unlike every type kLLM has), M-RoPE (`rope.dimension_sections = [11,11,10,0]`),
and whatever Qwen3.5's hybrid attention layers need. Its 120 expert tensors
are all IQ2_XXS, so it cannot run until that kernel exists.

**Qwen3-30B-A3B Q4_K_M is the smallest code delta.** The arch is `qwen3moe`,
whose attention is identical to the `qwen3` kLLM already runs — per-head q/k
RMSNorm, NeoX rotary — so the only new code is the FFN. 128 experts, 8 used,
and standard K-quants throughout. It is ~18 GiB, which means it exercises the
disk tier from day one; that is the demo, but it makes the correctness loop
slow.

So, pick one of two openings:

- **Correctness-first:** add `qwen3moe` and validate on Qwen3-30B-A3B via
  perplexity and greedy fixtures against llama.cpp (one model resident at a
  time, so `kllm compare`'s two-model requirement does not apply). ~18 GiB
  download, slow iteration, but it is the model the thesis is about.
- **Loop-speed-first:** add `olmoe` (OLMoE-1B-7B, 64 experts, 8 used, ~4 GiB
  at Q4_K_M) which fits in RAM, get the router and the locality trace quickly,
  then add `qwen3moe`. Costs one extra architecture, buys a fast iteration
  loop and an early answer on locality.

Either way the staging is: router and routed FFN correct on a RAM-resident
model → locality trace → expert-granularity residency in the planner and
disk tier → prefill token sorting.

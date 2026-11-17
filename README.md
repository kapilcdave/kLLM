<p align="center">
  <img src="docs/assets/banner.svg" alt="kLLM" width="100%">
</p>

<h3 align="center">Memory-first LLM inference in Rust, for the machine you already own</h3>

<p align="center">
  <a href="Cargo.toml"><img src="https://img.shields.io/badge/license-Apache--2.0%20%2F%20MIT-blue" alt="license"></a>
  <a href="Cargo.toml"><img src="https://img.shields.io/badge/version-v0.1.0-green" alt="version"></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/rust-2024%20edition-f74c00?logo=rust&logoColor=white" alt="rust"></a>
  <img src="https://img.shields.io/badge/CPU-AVX2%20%7C%20scalar-lightgrey" alt="cpu">
  <img src="https://img.shields.io/badge/CUDA-optional,%20no%20toolkit-76b900?logo=nvidia&logoColor=white" alt="cuda">
  <img src="https://img.shields.io/badge/format-GGUF-8a2be2" alt="gguf">
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> |
  <a href="#supported-models">Models</a> |
  <a href="#a-30b-model-on-a-16-gib-laptop">30B on 16 GiB</a> |
  <a href="#correctness">Correctness</a> |
  <a href="#performance">Performance</a> |
  <a href="docs/MOE.md">MoE design</a> |
  <a href="SPEC.md">Spec</a> |
  <a href="ROADMAP.md">Roadmap</a>
</p>

---

## About

kLLM (Kapil LLM) is an inference engine for quantized models on older,
memory-constrained machines. It reads GGUF files in place (memory-mapped,
never copied), plans every large allocation before making it, and runs
Llama- and Qwen-family decoders with AVX2 kernels whose output is checked
against llama.cpp.

Its reference machine is a Dell Precision 5520: a Xeon E3-1505M v6, 16 GiB of
RAM and a 4 GiB Quadro M1200. On it, kLLM runs **Qwen3-30B-A3B at 10-11 tok/s,
6x llama.cpp** on the same file.

<table>
  <tr>
    <td width="33%" valign="top">
      <h4>🧠 Memory-first</h4>
      Every large allocation is planned from real tensor sizes and KV costs
      before it happens. <code>run</code> prints its plan and refuses a
      context that will not fit, instead of swapping the machine to death.
    </td>
    <td width="33%" valign="top">
      <h4>💽 Three weight tiers</h4>
      VRAM, RAM, and the file itself. Models bigger than RAM stream blocks
      from disk with read-ahead and immediate release, so the stream never
      evicts the resident set.
    </td>
    <td width="33%" valign="top">
      <h4>🧩 Mixture of experts</h4>
      Experts are demand-paged, and the kernel page cache becomes the expert
      cache for free. A 17.3 GiB 30B MoE runs on a 16 GiB laptop.
    </td>
  </tr>
  <tr>
    <td width="33%" valign="top">
      <h4>⚡ Faster prefill than llama.cpp</h4>
      1.15-1.61x on every weight type measured, float types included. Decode is
      at parity or better on most types.
    </td>
    <td width="33%" valign="top">
      <h4>✅ Checked against llama.cpp</h4>
      Perplexity <i>and</i> KL divergence on real models, for 11 weight types.
      Within 2% of llama.cpp on every one, and lower on eight of ten.
    </td>
    <td width="33%" valign="top">
      <h4>🎮 CUDA without a toolkit</h4>
      Whole blocks and the output head offload with
      <code>--gpu-layers N|auto</code>. Kernels ship as PTX and load through the
      driver, so building needs no CUDA install.
    </td>
  </tr>
</table>

## Quick start

```sh
git clone https://github.com/kapilcdave/kLLM.git && cd kLLM
cargo build --release

# one-shot
./target/release/kllm run model.gguf --chat --prompt "Explain RoPE in one paragraph." -n 200

# a conversation (ChatML models): keeps the session, so each turn feeds only what is new
./target/release/kllm chat model.gguf

# what fits where, for any GGUF
./target/release/kllm plan-model model.gguf --vram-gib 3.7 --context 8192
```

Building without the GPU backend: `cargo build --release --no-default-features`.

## Supported models

| | |
|---|---|
| **Architectures** | `llama` (Llama 1–3, Mistral, SmolLM, TinyLlama), `qwen2`, `qwen3`, `qwen3moe`. A 30B mixture of experts runs on a 16 GiB machine at 6x llama.cpp: see [docs/MOE.md](docs/MOE.md) |
| **Weight types** | Validated end to end against llama.cpp on real models, by perplexity *and* KL divergence: F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K. F32 is kernel-tested only |
| **Tokenizers** | GPT-2 byte-level BPE (gpt2, smollm, llama3, qwen2 pre-tokenizers) and SentencePiece; special tokens; byte-exact with llama.cpp on the fixtures |
| **GGUF parsing** | Full metadata and tensor directory; every length is bounds-checked against the file before allocation; payloads are memory-mapped |
| **Execution** | CPU: batched prefill, f16 KV cache, AVX2/FMA/F16C kernels with a portable scalar fallback |
| **Sampling** | Greedy, temperature, top-k, top-p, min-p, repetition penalty, seeded |
| **CUDA** | Kernels for all weight types, coalesced struct-of-arrays matvecs for Q4_0/Q8_0/Q4_K/Q6_K, KV cache in VRAM. Perplexity matches llama.cpp on the GPU as closely as on the CPU (`tests/gpu.rs`) |
| **Qwen3.5** | `qwen35` dense 4B/9B Q4_K_M runs on CPU with gated attention and DeltaNet; 4B perplexity is within 0.27% of llama.cpp. The 35B-A3B IQ2_XXS file still needs that weight type. See [docs/QWEN35.md](docs/QWEN35.md) |

## Commands

```text
kllm inspect MODEL.gguf [--tensors]
kllm plan-model MODEL.gguf --vram-gib N [--context N]
kllm run MODEL.gguf --prompt TEXT [-n N] [--chat] [--system TEXT] [--temp T] [--no-think] ...
kllm chat MODEL.gguf
kllm serve MODEL.gguf [--host 127.0.0.1] [--port 8787] [--context N] [--api-key KEY]
kllm perplexity MODEL.gguf --file TEXT [--context 512] [--chunks N]
kllm compare REFERENCE.gguf TEST.gguf --file TEXT [--context 512] [--chunks N]
kllm bench MODEL.gguf [-p 128] [-n 64] [--profile]
kllm devices
```

`run`, `perplexity` and `bench` take `--gpu-layers N` (or `-ngl N`): the
last N blocks run on the GPU, and N at or above the block count also moves
the output head, as with llama.cpp. `--gpu-layers auto` lets the planner
place as many blocks as fit in the GPU's free memory. It is off by default.
On the reference laptop the M1200 is clock-limited to 287 MHz of its 1150,
and the CPU is faster by a wide margin: SmolLM2-135M Q4_0 decodes at 215
tok/s on the CPU against 37 on the GPU, and for Qwen3-0.6B Q4_K_M with all
28 blocks and the head offloaded, prefill is 308 tok/s on the CPU against
24.8 and decode 50.6 against 10.9. The clamp is not a battery-saving
decision that going on mains undoes -- `nvidia-smi` reports `HW Slowdown:
Active` with AC online and at idle, and the memory clock still runs at its
full 2505 MHz, so the card is instruction-starved rather than
bandwidth-starved. At the GPU's full clock that balance should change.
Re-measure if a charger the machine accepts at more than 65 W ever turns up.

`run` prints its memory plan before allocating anything and refuses a context
that does not fit in available RAM. `plan-model` works for any GGUF, including
architectures kLLM cannot run yet:

```text
$ kllm plan-model Qwen3.5-9B-Q4_K_M.gguf --vram-gib 3.7 --context 8192
  GPU layers:             20 (blocks 12..31)
  CPU layers:             12
  GPU weights:          2.48 GiB
  GPU KV cache:         0.62 GiB
  ...
```

## Models bigger than RAM

kLLM has a third weight tier below VRAM and RAM: the file itself. When a
model does not fit the RAM budget, it keeps the output head and a prefix of
blocks resident and streams the rest from disk every token. Each streamed
block is read ahead two blocks early (`MADV_WILLNEED`) and dropped from the
page cache right after use, so the stream never evicts the resident set,
which is what plain least-recently-used caching does on a pass larger
than RAM. Streamed blocks stay in their original row-major form, so they
take no heap memory. The budget defaults to what is available now, including
any cgroup limit, minus the session and a 1 GiB margin. Set it with
`--ram-gib N`, or turn streaming off with `--ram-gib all`.

Qwen3-14B Q4_K_M (9.0 GB) in a 6 GiB memory cgroup, cold page cache, 4
threads, on a 1.4 GB/s NVMe:

| Engine | Decode |
|---|---|
| llama.cpp b12099, default | killed: its CPU repack needs the whole model in heap |
| llama.cpp `--repack 0` (mmap) | 4.35 s/token |
| kLLM, mmap only (`KLLM_NO_REPACK=1 --ram-gib all`) | 4.58 s/token |
| kLLM, streaming (`--ram-gib 4.5`) | **3.00 s/token** |

Streaming is bounded by the disk: 3.98 GB streamed per token at 1.4 GB/s is
2.84 s. It only helps when the model really does not fit. Near the edge, the
kernel can often free enough other cache to hold the whole model, and
`--ram-gib all` is then faster.

## A 30B model on a 16 GiB laptop

A dense 30B needs all of itself for every token, so on a machine this size it
is hopeless. A mixture of experts needs a routed fraction -- 8 of 128 experts
for Qwen3-30B-A3B, about 6% of the expert weights -- and *which* fraction
repeats enough for the page cache to absorb it. That is the whole reason this
works:

```sh
kllm run Qwen3-30B-A3B-Q4_K_M.gguf --chat --prompt "..." -n 400 --context 4096
```

17.3 GiB of weights, 16 GiB of RAM, **10-11 tok/s** against llama.cpp's 1.81
on the same file and machine. Three things matter in practice:

- **Let the cache warm.** The first run of a model reads experts from disk and
  lands at 4-7 tok/s; by the second or third it is at 10-11. Switching
  between two 18 GiB models resets that, so expect a slow first answer.
- **Leave the budget alone.** `--ram-gib` defaults to something sensible, and
  for a routed model it only governs the non-expert weights (about 1 GiB);
  everything else is better spent on page cache for the experts. Forcing it
  smaller makes kLLM stream blocks it had no need to and costs 6x.
- **Prompt length picks the batch.** A long prompt gets a wide batch, which
  is what makes routed prefill fast; a short one gets a narrow batch so the
  session does not spend 155 MiB of logits scratch it will not use.

For a conversation, use `kllm chat` rather than repeated `run` calls. It keeps
the session alive, so each turn feeds only what is new: in a three-turn
exchange the third turn fed 18 tokens instead of re-processing all 174.
Re-sending the history every turn is what makes a one-shot CLI unusable on a
slow machine -- a thousand tokens of context costs minutes to re-read at
prefill speed, every single turn.

For a reasoning model such as the base Qwen3-30B-A3B, pass `--no-think`.
Otherwise it answers by thinking first, and at ten tokens a second a 100-token
budget is spent before it has said anything. Its own chat template suppresses
reasoning by pre-filling an empty `<think></think>`, and that is what the flag
does. Instruct builds that do not reason (Qwen3's `-2507-Instruct`, for one)
need nothing.

## Serving the Anthropic Messages API

`kllm serve` exposes the local model over the API Claude Code speaks, so a
coding client can drive it with no proxy in between:

```sh
kllm serve Qwen3-30B-A3B-Q4_K_M.gguf --context 16384
ANTHROPIC_BASE_URL=http://127.0.0.1:8787 ANTHROPIC_AUTH_TOKEN=local claude
```

For the version you keep -- a systemd unit, a memory ceiling, and a
`claude-kllm` wrapper that starts the server and sets the dozen environment
variables Claude Code needs (including the ones that must be *un*set, or it
quietly talks to a cloud provider instead) -- see [contrib/](contrib/).

`POST /v1/messages` streams over SSE or returns a single JSON message, and
`/v1/messages/count_tokens`, `/v1/models` and `/health` are there too. The
wire format is Anthropic's; underneath it is the same ChatML the model was
trained on. `--api-key` requires an `x-api-key` (or bearer) match; without it
the server trusts whoever can reach the port, so keep the default host.

**Tool calls work through the model's own format.** Qwen3 was trained to emit
`<tool_call>{"name": ..., "arguments": ...}</tool_call>`, so Anthropic `tools`
are rendered into the system prompt exactly as its chat template does, and
`<tool_call>` output is translated back into `tool_use` content blocks --
streaming included. `tool_result` blocks come back the other way as Qwen's
`<tool_response>` turns.

Reasoning is **off by default here**, the opposite of `run` and `chat`: a
client sends a `max_tokens` budget and expects an answer within it, and a
reasoning model will spend the entire budget deliberating instead. Asked to
call a tool with 80 tokens to spend, Qwen3-0.6B used all 80 inside `<think>`
and never emitted the call; suppressed, it answered with the `tool_use` block
in 21. `--think` puts reasoning back, as visible text rather than as
Anthropic `thinking` blocks.

**Prefix reuse is the only reason this is usable.** A client resends the whole
conversation on every turn, so the server keeps one session alive and rewinds
it to the longest prefix the new request still shares. Measured on the 30B
with a 3.5k-token system prompt:

| | prompt tokens | reused | time |
|---|---|---|---|
| first request | 3576 | 1 | 291 s |
| second turn | 3594 | 3577 | 4.7 s |

Every response logs `prompt N reused + M read`. Watch it: a client that
perturbs its own prefix between turns -- a changing system prompt, or two
conversations interleaved on one server -- turns every turn back into a full
prefill, and the only symptom is that things got slow.

On a **recurrent** model, `N reused` is not by itself proof that the reuse was
cheap. A KV cache rewinds for free, but a recurrence keeps no per-position
record of its state, so rewinding to a divergence point means replaying
everything below it -- a fresh session reported `8496 reused + 2386 read` and
still took 739 s. The server therefore saves the recurrent state at the
divergence point (`Session::checkpoint`, about 66 MiB on a 40-block hybrid) and
resumes from it next time, replaying only what sits between. In a controlled
run the same `682 reused + 10 read` went from 44.4 s to **1.1 s**. The second
request after a restart still pays one replay; that is when the snapshot is
taken.

Which is the honest limit here. **Decode is comfortable at 10.2 tok/s warm but
prefill on a demand-paged MoE runs at 12-18 tok/s**, because a prefill batch
routes to all 128 experts in every layer and so reads the whole 16.7 GiB
expert set, while a single decoded token reads only the 8 it routed to. The
first request of a session therefore costs minutes and every one after it is
seconds. Widening the batch does not fix it (16 / 64 / 256 measured at 10-14 /
13-15 / 17-18 tok/s: wider is better, but it saturates), and narrowing it does
not either -- 16 tokens still touch most of a layer's experts. A model whose
experts fit in RAM would not have the problem.

## Correctness

The test-suite compares kLLM with llama.cpp (build 12099) on recorded
fixtures; see [tests/fixtures/README.md](tests/fixtures/README.md). It checks
two different things, because one of them is not enough:

- **Perplexity** says whether a model predicts text well. It is an average
  over a corpus, which is what makes it robust to the summation-order noise
  that 8-bit activations introduce.
- **KL divergence** (`kllm compare`) says how far a quantization moved the
  distribution away from the full-precision model. An average over a corpus
  can hide that: a kernel can hold perplexity and still reorder the top of the
  distribution.

Mean KL divergence against each family's reference model, kLLM against
`llama-perplexity --kl-divergence` on the same text (4 chunks x 512 tokens):

| Model | reference | llama.cpp | kLLM | top-1 agreement |
|---|---|---|---|---|
| SmolLM2-135M BF16 | f16 | 0.000081 | 0.000000 | 100.00% / 99.61% |
| SmolLM2-135M Q8_0 | f16 | 0.003777 | 0.003630 | 97.16% / 96.57% |
| SmolLM2-135M Q6_K | f16 | 0.008890 | 0.008736 | 93.73% / 94.12% |
| SmolLM2-135M Q5_K_M | f16 | 0.063889 | 0.063725 | 85.10% / 85.39% |
| SmolLM2-135M Q5_1 | f16 | 0.069482 | 0.070725 | 85.49% / 85.69% |
| SmolLM2-135M Q5_0 | f16 | 0.088250 | 0.087968 | 83.14% / 82.45% |
| SmolLM2-135M Q4_0 | f16 | 0.148069 | 0.146500 | 81.08% / 81.08% |
| SmolLM2-135M Q4_1 | f16 | 0.271327 | 0.270642 | 72.65% / 74.12% |
| SmolLM2-135M Q2_K | f16 | 2.130084 | 2.124023 | 30.10% / 30.69% |
| Qwen3-0.6B Q4_K_M | Q8_0 | 0.076557 | 0.076500 | 86.76% / 86.77% |

kLLM is within 2% of llama.cpp on every one, and lower on eight of the ten.
BF16 is exactly zero because it should be: SmolLM2 was trained in bf16, so its
f16 GGUF holds exactly representable bf16 values (f16 keeps 10 mantissa bits
to bf16's 7) and converting back is lossless — 100k weights checked, none
differ. llama.cpp's 8e-5 there is its own F16-vs-BF16 kernel noise.

On F16, greedy decoding reproduces llama.cpp's tokens exactly, with a mean
top-10 log-probability difference of 0.003. Quantized paths quantize
activations to 8 bits (as llama.cpp does), which amplifies summation-order
differences: llama.cpp disagrees with *itself* by up to 0.4% perplexity when
only its flash-attention switch is flipped. kLLM's differences are within that
spread.

## Performance

Same machine, 4 threads, `kllm bench MODEL -p 128 -n 64 -t 4 -r 3` against
`llama-bench -t 4 -p 128 -n 64 -r 3` (llama.cpp b12099 CPU build), tokens per
second, best of three. The full table with model hashes and the exact commands
is in [evals/](evals/); regenerate it with `scripts/eval_report.py`.

| Model | pp128 kLLM / llama.cpp | ratio | tg64 kLLM / llama.cpp | ratio |
|---|---|---|---|---|
| SmolLM2-135M Q4_0 | 1145 / 750 | 1.53x | 229 / 219 | 1.05x |
| SmolLM2-135M Q8_0 | 1060 / 658 | 1.61x | 165 / 140 | 1.18x |
| SmolLM2-135M Q6_K | 988 / 667 | 1.48x | 165 / 145 | 1.14x |
| SmolLM2-135M Q2_K | 1089 / 814 | 1.34x | 247 / 244 | 1.01x |
| SmolLM2-135M Q5_K_M | 462 / 352 | 1.31x | 157 / 167 | 0.94x |
| SmolLM2-135M f16 | 629 / 547 | 1.15x | 85.5 / 82.3 | 1.04x |
| Qwen3-0.6B Q4_K_M | 324 / 266 | 1.22x | 48.3 / 55.2 | 0.88x |
| Qwen3-0.6B Q8_0 | 284 / 184 | 1.54x | 36.4 / 37.0 | 0.98x |

Prefill is ahead of llama.cpp on every weight type, float ones included
(1.11-1.65x). Decode is parity or better on most types and trails on the
K-quants (Q4_K 0.88x); it is memory-bandwidth-bound, where prefill is
compute-bound, which is why the two respond to different things -- a register
tile doubled float prefill and *cost* float decode until it was gated on
batch size.

Two notes on reading these numbers. Prefill was at 65% of llama.cpp until the
real bottleneck turned up, and it was not the kernels: at *one* thread kLLM
already beat llama.cpp, and the packed Q4_0 kernel runs near the
`vpmaddubsw` port limit. Activation quantization was running on the calling
thread while the workers idled — 71% of a 576x1536 matmul. And this machine
usually has other work on it, so `kllm bench` reports the best run rather than
the mean (interference can only ever slow a run down) and comparisons against
llama.cpp are interleaved in the same window.

`kllm bench --profile` prints a per-phase breakdown.

## Development

```sh
cargo test            # unit tests; model tests skip if models are absent
KLLM_REQUIRE_MODELS=1 cargo test   # fail instead of skipping
KLLM_NO_SIMD=1 cargo test          # exercise the portable scalar kernels
```

Model-backed tests look in `models/` (git-ignored; symlinks are fine) or in
`$KLLM_TEST_MODELS`. `scripts/make_fixtures.py` and
`scripts/make_eval_fixtures.py` re-record the llama.cpp references;
`scripts/eval_report.py` writes one report with perplexity, divergence, speed,
model hashes and hardware beside llama.cpp's numbers (see `evals/`).

`examples/` holds the micro-benchmarks that located the bottlenecks above, and
they are the right first stop before optimizing anything:

| Example | Answers |
|---|---|
| `kernel_bench` | matvec GB/s per weight type — is a shape bandwidth-bound? |
| `gemm_bench` | batched matmul GOP/s |
| `tile_bench` | splits a matmul into kernel / row-streaming / full path, and times `prepare` separately |
| `attn_bench` | attention primitives in GMAC/s, per head width |
| `scale_bench` | parallel speedup available on this machine, and pool dispatch latency |
| `gpu_bench` | CUDA matvec GB/s and kernel launch overhead |

See [SPEC.md](SPEC.md) for the specification and [ROADMAP.md](ROADMAP.md) for
the plan.

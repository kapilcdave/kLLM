# Running Qwen3.5 (`qwen35`, `qwen35moe`)

Qwen3.5 dense text inference now runs on the CPU. The 4B Q4_K_M model is
checked against llama.cpp; the gated DeltaNet recurrence, causal convolution,
gated softmax attention and text RoPE are implemented. This work follows
steps 1–3 of the original staged plan. IQ2_XXS for the local 35B-A3B file
remains separate work.

## Run it

```sh
cargo build --release
./target/release/kllm run ~/ai/models/Qwen3.5-4B-Q4_K_M.gguf \
  --chat --no-think --prompt "What is the capital of France?" \
  --temp 0 -n 64 --context 512 --threads 4
./target/release/kllm chat ~/ai/models/Qwen3.5-9B-Q4_K_M.gguf \
  --no-think --context 2048 --threads 4
```

Use CPU execution for these hybrid blocks. GPU layer offload rejects gated
attention and DeltaNet explicitly; output-head-only offload remains possible.
Image inputs and multimodal positions are outside this text-only path.

## Tensor layout

Layers are selected by their tensors, rather than by assuming a particular
attention interval. On the 4B, three SSM blocks alternate with one attention
block across 32 layers. All blocks use `attn_norm`, `post_attention_norm`
(the pre-FFN norm), and the ordinary dense feed-forward weights.

The 4B linear-attention hyperparameters are:

| Parameter | Value | Meaning |
|---|---:|---|
| `ssm.state_size` | 128 | Key and value head dimension |
| `ssm.group_count` | 16 | Number of Q/K heads |
| `ssm.time_step_rank` | 32 | Number of value heads |
| `ssm.inner_size` | 4096 | Value width, 32 × 128 |
| `ssm.conv_kernel` | 4 | Causal depthwise convolution taps |

The fused `attn_qkv` output is Q (2048), K (2048), V (4096): 8192 channels.
The loader calculates this as `2 * group_count * state_size + inner_size`;
it does not assume the value width is always equal to the combined Q/K width.

| SSM tensor | 4B shape (GGUF order) |
|---|---|
| `attn_qkv.weight` | `[2560, 8192]` |
| `attn_gate.weight` | `[2560, 4096]` |
| `ssm_conv1d.weight` | `[4, 8192]` |
| `ssm_a`, `ssm_dt.bias` | `[32]` |
| `ssm_alpha.weight`, `ssm_beta.weight` | `[2560, 32]` |
| `ssm_norm.weight` | `[128]` |
| `ssm_out.weight` | `[4096, 2560]` |

## Recurrence and conventions

The reference is llama.cpp's
[`qwen35.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp),
[`build_gdn_l2_norm`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/models.h),
and CPU gated DeltaNet implementation in
[`ops.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ops.cpp).
Its [converter](https://github.com/ggml-org/llama.cpp/blob/master/conversion/qwen.py)
explains the GGUF-specific weight transformations.

For each token:

1. Project the normalized hidden state into fused QKV, output gate Z, alpha
   and beta. Convolve each QKV channel over the previous three inputs and
   the current input, using the stored taps oldest to newest, then apply SiLU.
2. Normalize Q and K by `sqrt(sum(x²) + rms_eps)` per head.
3. Set `decay = exp(ssm_a * softplus(alpha + dt_bias))` and
   `beta = sigmoid(beta_projection)`. **GGUF already stores `ssm_a` as
   `-exp(A_log)`**, so do not exponentiate that weight again.
4. Decay the matrix state, predict the value with the decayed state, then
   apply the delta correction. In key-by-value notation:

   ```text
   S_decay = decay * S
   delta   = beta * (v - kᵀ S_decay)
   S       = S_decay + k deltaᵀ
   output  = qᵀ S / sqrt(head_dim)
   ```

5. Apply weighted RMSNorm over each output head, multiply by `SiLU(Z)`,
   and project back to the hidden width. The SSM output gate is SiLU.

GGUF reorders V heads, their gates and the output projection for tiled
broadcasting: value head `h` uses Q/K head `h % n_key_heads`. Applying
Transformers' original `repeat_interleave` convention directly to these
converted tensors produces badly wrong output.

The attention blocks project **interleaved Q and gate per head**:
`[q_head_0, gate_head_0, q_head_1, gate_head_1, ...]`. Split these before
Q normalization and RoPE; sigmoid-gate the attention result before the output
projection. For text, all M-RoPE position axes agree, so ordinary partial
NeoX RoPE over the first 64 dimensions is sufficient.

## Session state and memory

Each SSM layer owns a matrix state of `32 × 128 × 128` f32s (2 MiB), plus
8192 channels of three-position convolution history. Across the 24 SSM
layers this is 51 MiB including convolution scratch. Projection
scratch is reused across layers and forward calls. Session allocation
reporting and the preflight estimate include these buffers and the gated
attention buffers.

`truncate(0)` clears convolution and matrix state. Truncating to a nonzero
prefix rebuilds that prefix on the next forward call, using the retained
token IDs. This keeps rewind correct without storing a matrix snapshot per
token; rewinding consequently costs a prefix prefill.

## Validation

On `tests/fixtures/wiki.test.head.txt`, CPU, 4 chunks of context 256 (508
scored tokens), same `Qwen3.5-4B-Q4_K_M.gguf`:

| Engine | Perplexity |
|---|---:|
| llama.cpp b12099, `-t 8 -ngl 0` | 12.8886 |
| kLLM, `--threads 8` | 12.9229 |

Both local dense models also pass a greedy chat smoke test: 4B and 9B
answer “The capital of France is Paris.” with `--chat --no-think --temp 0`.

Difference: **0.27%**. The recorded baseline is
`tests/fixtures/qwen35-perplexity.json`; the regression tolerance is 1%.
This is a short correctness sample, not a broad model-quality evaluation.

```sh
KLLM_QWEN35_MODEL=~/ai/models/Qwen3.5-4B-Q4_K_M.gguf \
  cargo test --test qwen35 -- --test-threads=1 --nocapture
```

The real-model tests also check batched prefill against single-token decode,
exact reset/rewind results with the same batching, finite logits and memory
accounting. Without a local model, the small two-step numerical recurrence
fixture still runs in `cargo test --lib`.

## Remaining work

The existing routed FFN loader recognizes `qwen35moe`, but the local
`Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf` still requires IQ2_XXS weight support.
Its expert tensors cannot currently load. Dense-model validation does not
establish correctness for that MoE file.

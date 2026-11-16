# Running Qwen3.5 (`qwen35`, `qwen35moe`)

Scoping notes, so this can be built without rediscovering the shape of it.
Nothing here is implemented yet.

## Why it is wanted

Two reasons, and the second is the interesting one.

`Qwen3.5-4B` and `Qwen3.5-9B` are already on this machine as Q4_K_M — types
kLLM reads. At 2.5 and 5.5 GiB they fit in RAM outright, so they would run
*fast*, not merely run. That is the cheap win.

`Qwen3.5-35B-A3B` is the one actually asked for. It needs this architecture
**and** IQ2_XXS, because its 120 expert tensors are all IQ2_XXS and nothing
else in the file is the bulk. Two separate pieces of work; this note covers
only the first.

## The shape

`qwen35` is a layer-level hybrid, and `{arch}.full_attention_interval = 4`
says how: three of every four blocks are a linear-attention (SSM) layer, and
every fourth is ordinary softmax attention. Measured on the 4B, 32 blocks:

```
0:SSM 1:SSM 2:SSM 3:ATT 4:SSM 5:SSM 6:SSM 7:ATT ... 28:SSM 29:SSM 30:SSM 31:ATT
```

So a block is identified by its tensors, which is the rule to follow rather
than arithmetic on the interval:

| Tensor | Which blocks | Shape (4B) |
|---|---|---|
| `attn_norm`, `post_attention_norm`, `ffn_{gate,up,down}` | all | — |
| `attn_q`, `attn_k`, `attn_v`, `attn_output`, `attn_q_norm`, `attn_k_norm` | attention only | as `qwen3` |
| `attn_qkv` | SSM only | `[2560, 8192]` |
| `attn_gate` | SSM only | `[2560, 4096]` |
| `ssm_conv1d` | SSM only | `[4, 8192]` |
| `ssm_a`, `ssm_dt.bias` | SSM only | `[32]` |
| `ssm_alpha`, `ssm_beta` | SSM only | `[2560, 32]` |
| `ssm_norm` | SSM only | `[128]` |
| `ssm_out` | SSM only | `[4096, 2560]` |

Hyper-parameters (4B):

```
embedding_length 2560      feed_forward_length 9216
attention.head_count 16    head_count_kv 4
attention.key_length 256   value_length 256
ssm.conv_kernel 4          ssm.state_size 128
ssm.group_count 16         ssm.time_step_rank 32
ssm.inner_size 4096        full_attention_interval 4
rope.dimension_count 64    rope.dimension_sections [11, 11, 10, 0]
rope.freq_base 10000000
```

Note `key_length = 256` with `head_count = 16`, so the attention blocks have
`q_dim = 4096` and `kv_dim = 1024` -- a head dimension kLLM has not run
before, but nothing structural.

## The three pieces of work, easiest first

**1. The attention blocks are nearly free.** Separate q/k/v, an output
projection, per-head q/k RMSNorm: that is `qwen3`, which already runs. What
is new is only that they sit in a layer loop alongside something else.

**2. M-RoPE reduces to RoPE for text.** `rope.dimension_sections`
`[11, 11, 10, 0]` splits the 32 rotary pairs across the time, height and
width position axes of a multimodal model. For text-only input all three
axes carry the same position, so every section rotates by the same angle and
the whole thing collapses to ordinary RoPE over `rope_dimension_count = 64`.
Implement it that way and it is exact for text; the sections only start to
matter when images are in the prompt, which kLLM does not take.

**3. The SSM block is the real work.** From the tensor inventory it is a
gated DeltaNet, the Qwen3-Next shape. The roles and widths are now pinned
down (4B numbers):

- `attn_qkv` projects to 8192 = q, k and v for the linear-attention path.
- `ssm_conv1d` `[4, 8192]` is a causal depthwise convolution, four taps, over
  that 8192-wide stream -- so the session needs a per-layer ring buffer of
  the last three positions, not just a recurrent state.
- `ssm_a` `[32]` and `ssm_dt.bias` `[32]` with `time_step_rank = 32` give the
  per-step decay; `ssm_alpha` and `ssm_beta` `[2560, 32]` project the hidden
  state to the gates that modulate it.
- The recurrent state is `group_count = 16` by `state_size = 128`, so
  2048 floats per layer per sequence, times 24 SSM layers.
- `ssm_norm` `[128]` is an RMSNorm over the state dimension before
  `ssm_out` `[4096, 2560]` projects back, and `attn_gate` `[2560, 4096]`
  supplies a sigmoid gate on the way out.

### What the widths say, exactly

The hyper-parameters name the same three numbers three different ways, and
putting them together fixes the whole layout:

- `ssm.state_size = 128` is the head dimension.
- `ssm.time_step_rank = 32` is the number of **v** heads, and
  `32 x 128 = 4096 = ssm.inner_size`.
- `ssm.group_count = 16` is the number of **q** and **k** heads, so two v
  heads share each k head.

Which accounts for `attn_qkv`'s 8192 outputs exactly:

| part | heads x dim | width |
|---|---|---|
| q | 16 x 128 | 2048 |
| k | 16 x 128 | 2048 |
| v | 32 x 128 | 4096 |

and for everything else: `ssm_conv1d [4, 8192]` convolves all three
depthwise, `ssm_a` and `ssm_dt.bias` are `[32]` because they are per v head,
`ssm_alpha` and `ssm_beta` are `[n_embd, 32]` because they project the hidden
state to a scalar per v head, `ssm_norm [128]` normalizes over the head
dimension, and `ssm_out [4096, n_embd]` projects the 32 heads back.

### What is left is one formula, and it should not be guessed

Everything above is read off shapes and is therefore safe. The recurrence is
not: it is about five coupled choices, and the plausible-looking options are
indistinguishable from the file.

- whether q and k are L2-normalized per head before use (gated DeltaNet
  usually does);
- whether the convolution is followed by an activation, and which;
- how the decay is parameterized -- something of the shape
  `g = exp(-exp(ssm_a) * softplus(alpha . x + dt_bias))`, but the exact
  nesting matters;
- the delta rule itself, `S <- S g - beta (S k) k^T + beta v k^T`, and
  whether beta comes from a sigmoid;
- where the output gate applies, before or after `ssm_norm`.

Get any one wrong and perplexity explodes rather than drifting, so the
failure tells you nothing about *which* one. That is the argument for not
shipping a guess: five binary-ish choices is up to thirty-odd combinations,
each needing a multi-minute perplexity run to reject, and no signal to
order the search. It wants the reference implementation open alongside,
not an evening of bisection.

It also needs a new `Session` concern either way -- a recurrent state of
`32 x 128 x 128` floats per layer (about 2 MiB, so ~50 MiB over 24 SSM
layers) and a ring buffer of the last three positions for the convolution,
neither of which the KV cache covers.

## Status

Steps 1 and 2 below are done. Qwen3.5-4B Q4_K_M loads every tensor at the
right shape in 3.7 s and stops with an explicit "the recurrence is not
implemented yet"; the config, the per-block attention/SSM split, the gated
attention and the norm naming are all in. Two things turned up that the
first pass of this note had wrong:

- **The attention blocks are gated too.** `attn_q` is *twice* `q_dim` wide
  (`[2560, 8192]` on the 4B against a `q_dim` of 4096), and the second half
  is a sigmoid gate on the attention output -- the same gate the SSM blocks
  keep in a separate `attn_gate` tensor. Detected by shape, not by
  architecture.
- **There is no `ffn_norm`.** `post_attention_norm` *is* the norm before the
  feed-forward block; it is simply named for where it sits relative to the
  mixer. Nothing else in the file could play that role.

And the 35B's other blocker is confirmed rather than assumed: it stops at
`blk.0.ffn_gate_exps.weight has type UNKNOWN`, which is IQ2_XXS.

## Order to build it

1. `Arch::Qwen35`, the config, and the per-block attention/SSM split, with
   the SSM path returning "unsupported" -- so the file loads and the shape
   can be checked against `kllm inspect` before any arithmetic.
2. The attention blocks and M-RoPE-as-RoPE. Still unrunnable, but every
   non-SSM tensor is then proven to load at the right shape.
3. The conv ring buffer and the recurrence, validated by perplexity against
   llama.cpp on Qwen3.5-4B Q4_K_M at every step.
4. Only then `qwen35moe` (the 35B adds routed experts on top, which kLLM
   already does) and IQ2_XXS.

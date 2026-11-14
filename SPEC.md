# kLLM v0.1 specification

## 1. Product

kLLM is a local LLM inference engine written in Rust. It targets machines with
older NVIDIA GPUs, no tensor cores, small VRAM pools, and substantially more
system RAM than VRAM. Its initial reference machine is a Dell Precision 5520
with a 4 GiB NVIDIA Quadro M1200.

kLLM does not promise that an arbitrarily large dense model will run quickly in
4 GiB. It makes the capacity/speed trade-off explicit and predictable by
quantizing weights, placing only selected layers on the GPU, and budgeting every
large allocation before inference starts.

## 2. User experience

The primary interface is a single CLI:

```text
kllm inspect model.gguf
kllm plan model.gguf --vram-limit 3.7GiB --context 4096
kllm run model.gguf --prompt "Hello" --vram-limit 3.7GiB
kllm serve model.gguf --host 127.0.0.1 --port 8080
```

`plan` must work without CUDA. `run` must print its chosen memory plan before
allocating model buffers. If the requested configuration cannot fit, kLLM must
return a useful error containing the estimated shortfall and safe settings.

## 3. v0.1 scope

### Included

- Linux x86-64 as the first runtime target.
- GGUF parsing and memory-mapped weight access.
- Decoder-only Llama-family models.
- Common GGUF integer quantizations, beginning with Q4_0.
- CPU execution plus optional CUDA execution on legacy NVIDIA hardware.
- Whole-layer CPU/GPU placement.
- Single-user, batch-size-one token generation.
- Bounded KV cache and streaming text output.
- Reproducible benchmarks reporting prompt and generation tokens/second.

### Excluded

- Training and fine-tuning.
- Multi-GPU or distributed inference.
- Tensor-parallel execution.
- Continuous batching in v0.1.
- Compatibility with every GGUF architecture and quantization.
- Windows support in the first milestone.

## 4. Functional requirements

### Model loader

- Validate GGUF magic, version, tensor offsets, dimensions, and file bounds.
- Read metadata without copying tensor data.
- Memory-map tensor payloads and expose typed tensor descriptors.
- Reject unsupported architectures and quantizations before allocation.

### Memory planner

- Inputs: per-layer weight bytes, head and embedding bytes, VRAM limit,
  context length, KV bytes per token per layer (derived from GGUF metadata),
  fixed GPU workspace, and safety margin.
- Reserve KV cache, workspace, and safety margin before placing weights.
- Place the largest possible whole *suffix* of transformer layers in VRAM,
  using each layer's real tensor sizes and its own KV-cache share. A suffix
  keeps the GPU layers adjacent to the output head, so a token crosses the bus
  once in each direction.
- Account for embeddings and output tensors separately: the embedding table is
  a row lookup and stays in RAM; the output head goes to VRAM only once every
  layer fits.
- Never intentionally exceed the configured VRAM limit.
- Produce a stable, human-readable plan and machine-readable plan later.

### Execution

- Implement tokenizer, embedding lookup, RMSNorm, RoPE, causal attention,
  SwiGLU feed-forward blocks, and output projection.
- Use preallocated scratch buffers in the decode loop.
- Permit CPU-only execution.
- Stream generated tokens without retaining duplicate output buffers.

### CUDA backend

- Compile as an optional feature; the CPU build must remain usable without a
  CUDA toolkit.
- Target scalar CUDA cores and avoid requiring tensor-core instructions.
- Start with quantized matrix-vector kernels because batch-one decoding is the
  principal workload.
- Keep transfers explicit and measurable.

## 5. Architecture

```text
CLI
 +-- GGUF metadata/parser
 +-- hardware probe
 +-- memory planner
 +-- model graph
      +-- CPU backend
      +-- CUDA backend
 +-- tokenizer/sampler
 +-- benchmark and metrics
```

Core crates/modules should not depend on the CLI. Backends implement a narrow
tensor-operation trait so CPU correctness can serve as the reference for CUDA.

## 6. Performance policy

Correctness and bounded memory come before throughput. Every benchmark records:

- model and quantization;
- context and generated-token counts;
- CPU, RAM, GPU, driver, and VRAM limit;
- time to first token;
- prompt-processing tokens/second;
- generation tokens/second; and
- peak host RAM and VRAM.

No performance claim is accepted without a reproducible command and comparison
against a current `llama.cpp` build on the same machine.

## 7. Milestones

1. **Planner:** standalone Rust CLI, deterministic memory plans, unit tests.
2. **Inspect:** safe GGUF metadata parser and tensor inventory.
3. **CPU reference:** minimal Llama forward pass and greedy generation.
4. **Quantization:** Q4_0 CPU kernels with correctness fixtures.
5. **Legacy CUDA:** device probe, allocation guard, Q4_0 matrix-vector kernel.
6. **Hybrid execution:** planned layer offload and overlapped transfers.
7. **Usable runtime:** tokenizer, sampling, streaming CLI, benchmark harness.
8. **Server:** optional OpenAI-compatible local HTTP endpoint.

## 8. Acceptance criteria for milestone 1

- `cargo test` passes without network access or CUDA.
- The planner never reports allocations above the supplied VRAM limit.
- Increasing the context cannot increase GPU-resident layer count.
- Impossible configurations produce an error instead of underflow or panic.
- The CLI emits model, KV-cache, workspace, safety, GPU-weight, and CPU-weight
  estimates.


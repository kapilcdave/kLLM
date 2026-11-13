# KLLM

KLLM (Kapil LLM) is a memory-first inference engine in Rust for quantized
language models on older, memory-constrained GPUs.

The first executable milestone is a deterministic memory planner. It divides a
model between VRAM and system RAM while reserving room for the KV cache and
runtime workspace.

```sh
cargo run -- plan \
  --model-gib 4.1 \
  --layers 32 \
  --vram-gib 4 \
  --context 4096 \
  --kv-bytes-per-token 131072
```

Run the tests with `cargo test`.

Inspect a GGUF file without loading its tensor data:

```sh
cargo run -- inspect model.gguf
```

See [SPEC.md](SPEC.md) for the product and implementation specification.


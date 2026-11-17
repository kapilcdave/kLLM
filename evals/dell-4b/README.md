# Dell Precision 5520 engine comparison

The three-column chart uses fresh Qwen3-30B-A3B and Qwen3.5-4B CPU microbenchmarks once both runs complete. An earlier chart draft used recorded Qwen3-30B-A3B results. It is not an official cross-engine llama-bench score: llama-bench is llama.cpp's benchmark executable, and kLLM is measured with its own bench command.

## Fresh 4B protocol

- Identical local Q4_K_M GGUF file, SHA-256 in results.json.
- Four CPU threads, zero GPU layers, prompt length 128, generation length 64.
- Three process launches per engine, interleaved AB / BA / AB; each launch uses one measured repetition after its engine's warmup. Chart shows median and full observed range, using the same aggregation for both engines.
- kLLM binary copied before the run; binary SHA-256 in results.json. No rebuilds or source edits to the other session's engine work.
- Engine-specific synthetic inputs: kLLM uses deterministic IDs and greedy decode following pp128; llama-bench uses its own inputs and standalone tg64. These results approximate equivalent-size microbenchmarks, not identical-token end-to-end chat trials.
- Default engine execution strategies remain enabled. Batch/ubatch sizes are engine defaults, not a matched batching experiment. Read raw logs for effective configurations.
- Normal background workloads were active. No cache flushing, swap changes, process termination, or exclusive machine allocation. Results are preliminary, not an idle-machine publication run.
- Memory utilization is not measured by this suite yet. The headline 16 GiB refers to machine capacity, not engine peak RAM.

Run:

```sh
python scripts/dell_bench.py --model /home/kapil/ai/models/Qwen3.5-4B-Q4_K_M.gguf --out evals/dell-4b
python scripts/plot_dell_comparison.py
```

## Recorded 30B result

README.md and docs/MOE.md report Qwen3-30B-A3B Q4_K_M, 17.3 GiB file, 10–11 tok/s warm kLLM decode versus 1.81 tok/s llama.cpp on the same machine/file. The chart cites this historical result separately. Exact baseline flags and matched cache states are not available for that historical number. The fresh no-repack comparison in ../dell-30b/ is the relevant current baseline; do not present the old ~6× claim as the current controlled result.

## vLLM compatibility

No vLLM runtime score was measured. The label N/A is a compatibility statement about this hardware plus these GGUF files, not proof that vLLM cannot run any model on this CPU.

- Local GPU: NVIDIA Quadro M1200, compute capability 5.0, 4096 MiB VRAM. Current vLLM CUDA requirements list compute capability 7.5 or higher: https://docs.vllm.ai/en/stable/getting_started/installation/gpu/
- Local CPU supports AVX2. vLLM documents limited AVX2 CPU support, so a blanket CPU-incompatible claim would be wrong: https://docs.vllm.ai/en/latest/getting_started/installation/cpu/
- vLLM's quantization compatibility matrix lists GGUF unsupported on x86 CPU: https://docs.vllm.ai/en/latest/features/quantization/

A vLLM CPU comparison with a supported weight format would be a separate experiment; different quantization would also require quality/memory disclosure.

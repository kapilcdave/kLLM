# Dell Precision 5520: fresh engine comparison

Four CPU threads, Q4_K_M files, pp128 and tg64, three interleaved rounds per engine. Values are medians in tokens/sec. Background workloads active; engine-specific prompts and decode contexts differ.

| Model / metric | vLLM | llama.cpp | kLLM |
|---|---:|---:|---:|
| Qwen3-30B-A3B / pp | N/A¹ | 10.87 | 12.40 |
| Qwen3-30B-A3B / tg | N/A¹ | 5.98 | 4.80 |
| Qwen3.5-4B / pp | N/A¹ | 33.93 | 19.40 |
| Qwen3.5-4B / tg | N/A¹ | 7.64 | 7.30 |

¹ vLLM was not executed: the M1200 is below its CUDA requirement, and its upstream quantization matrix lists GGUF unsupported on x86 CPU. This does not imply that vLLM cannot run supported CPU weight formats.

llama.cpp uses --repack 0 for 30B, allowing a low-memory mmap configuration. Its defaults are used for 4B. kLLM uses automatic memory planning. See per-run logs and README files for limitations.

The fresh no-repack comparison does not reproduce the historical ~6× decode claim used in qwen30b-headline.png. Treat that older graphic as superseded for current performance claims. kLLM leads median 30B prefill here, while llama.cpp leads median decode on both models and 4B prefill. Wide 30B ranges make an exclusive-hardware, cache-controlled follow-up useful before a broad performance claim.

Reproduce:

```sh
python scripts/dell_bench.py --model /path/Qwen3.5-4B-Q4_K_M.gguf --out evals/dell-4b
python scripts/dell_bench.py --model /path/Qwen3-30B-A3B-Q4_K_M.gguf --out evals/dell-30b --llama-repack 0
python scripts/plot_dell_comparison.py
```

Compatibility references: https://docs.vllm.ai/en/stable/getting_started/installation/gpu/ and https://docs.vllm.ai/en/latest/features/quantization/

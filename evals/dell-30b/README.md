# Fresh Qwen3-30B-A3B CPU microbenchmark

Same 17.3 GiB Q4_K_M file for both engines on the Dell Precision 5520, four CPU threads, pp128/tg64, three interleaved process launches each. Each launch measures one repetition after engine-specific warmup; report median and observed min/max across launches.

llama.cpp uses `--repack 0` to avoid a large heap copy on this 16 GiB machine. kLLM uses its default memory planner. No page-cache flushing occurs between launches: each engine starts after the preceding workload. File hashing reads the entire model before the first launch; do not label these as controlled cold-cache results. All normal background workloads remain active. This is an indicative laptop test, not exclusive use of hardware.

These are equivalent-size engine microbenchmarks with different synthetic tokens, warmup routines and decode contexts. They do not establish identical end-to-end chat latency. The lower-RAM non-repacked llama.cpp configuration is explicit; results must not be described as every possible llama.cpp configuration.

`results.json` retains full SHA-256 hashes of the model and kLLM binary snapshot, exact commands, exit status, elapsed wall time and parsed metrics. Per-attempt raw logs retain llama.cpp build metadata and engine configuration. The executable snapshot stays local and is ignored by git. See ../dell-4b/README.md for vLLM compatibility sources.

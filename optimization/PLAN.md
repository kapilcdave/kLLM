# kLLM optimisation log

## 2026-11-17 — AArch64 Q4_0 × Q8_0 matvec

Target: accelerate the existing end-to-end Q4_0 CPU path without changing its
quantized activation format, matrix layout, or results.

The accepted comparison used control/candidate/control processes, nine samples
per process, a 4096 x 4096 matrix, and five worker threads on
`aarch64-apple-darwin`:

| Run | Kernel | Median latency | Rows/s |
|---|---|---:|---:|
| Control 1 | scalar | 2.262 ms | 1,810,620 |
| Candidate | NEON | 1.422 ms | 2,879,521 |
| Control 2 | scalar | 2.518 ms | 1,626,769 |

The candidate beat both neighbouring controls: 1.59x versus control 1 and
1.77x versus control 2. Earlier warm runs reached 0.361 ms, but that number is
not used as the accepted result because the adjacent controls drifted.

Correctness: `kernels::tests::kernels_agree_with_reference` and
`kernels::tests::multi_column_kernels_match_single_column` pass. The entire
library suite has one pre-existing AArch64 failure in
`vector_exp_and_swiglu_are_accurate` (absolute difference 0.000057 in an
unrelated exponential approximation).

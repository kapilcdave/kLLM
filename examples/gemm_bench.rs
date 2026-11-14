// Batched matmul (prefill) throughput in effective GOP/s (2 ops per MAC).
use kllm::backend::{Backend, Matrix};
use kllm::cpu::CpuBackend;
use kllm::quant::{GgmlType, encode_f16, quantize_q4_0, quantize_q8_0};
use std::time::Instant;
fn main() {
    let threads: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(4);
    let batch: usize = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(64);
    kllm::pool::init(threads);
    let (rows, cols) = (1536, 576);
    let values: Vec<f32> = (0..rows * cols)
        .map(|i| ((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0)
        .collect();
    let x: Vec<f32> = (0..cols * batch).map(|i| (i as f32 * 0.01).sin()).collect();
    for (kind, data) in [
        (GgmlType::Q4_0, quantize_q4_0(&values)),
        (GgmlType::Q8_0, quantize_q8_0(&values)),
        (GgmlType::F16, encode_f16(&values)),
    ] {
        let m = Matrix::new(kind, rows, cols, &data).unwrap();
        let mut y = vec![0.0; rows * batch];
        CpuBackend.matmul(m, &x, &mut y, batch).unwrap();
        let reps = 50;
        let t = Instant::now();
        for _ in 0..reps {
            CpuBackend.matmul(m, &x, &mut y, batch).unwrap();
        }
        let s = t.elapsed().as_secs_f64() / reps as f64;
        println!(
            "{rows}x{cols} batch {batch} {:5} {:8.1} us  {:6.1} GOP/s",
            kind.name(),
            s * 1e6,
            2.0 * (rows * cols * batch) as f64 / s / 1e9
        );
    }
}

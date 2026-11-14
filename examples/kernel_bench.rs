// Matvec throughput per weight type: GB/s of weights streamed.
use kllm::backend::{Backend, Matrix};
use kllm::cpu::CpuBackend;
use kllm::quant::{GgmlType, encode_f16, quantize_q4_0, quantize_q8_0};
use std::time::Instant;

fn main() {
    let threads: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(4);
    kllm::pool::init(threads);
    for (rows, cols) in [(1536, 576), (4096, 4096)] {
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01).sin()).collect();
        for (kind, data) in [
            (GgmlType::Q4_0, quantize_q4_0(&values)),
            (GgmlType::Q8_0, quantize_q8_0(&values)),
            (GgmlType::F16, encode_f16(&values)),
        ] {
            let m = Matrix::new(kind, rows, cols, &data).unwrap();
            let mut y = vec![0.0; rows];
            let reps = (200_000_000 / data.len()).max(3);
            CpuBackend.matvec(m, &x, &mut y).unwrap();
            let t = Instant::now();
            for _ in 0..reps {
                CpuBackend.matvec(m, &x, &mut y).unwrap();
            }
            let s = t.elapsed().as_secs_f64() / reps as f64;
            println!(
                "{rows:>5}x{cols:<5} {:5} {:8.1} us  {:6.2} GB/s",
                kind.name(),
                s * 1e6,
                data.len() as f64 / s / 1e9
            );
        }
    }
}

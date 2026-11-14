// Where does prefill time go: the packed kernel, or the loop around it?
//
// Three measurements of the same arithmetic, all with the weights and
// activations hot in cache:
//   kernel  - dot_x8 called in a tight loop on one group of eight rows.
//             This is the kernel's ceiling: no scheduling, no output scatter.
//   rows    - the same, walking every row group of a real-sized matrix, so
//             the weights stream from L2/L3 but nothing is written out.
//   matmul  - the backend's full path including the output scatter and pool.
//
// A gap between `kernel` and `rows` is a memory problem; a gap between
// `rows` and `matmul` is a bookkeeping problem.
use kllm::backend::{Backend, Matrix};
use kllm::cpu::CpuBackend;
use kllm::kernels::{self, Prepared};
use kllm::quant::{GgmlType, quantize_q4_0, quantize_q8_0};
use std::time::Instant;

fn main() {
    let threads: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(1);
    let batch: usize = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(8);
    kllm::pool::init(threads);
    let rows: usize = std::env::args()
        .nth(3)
        .and_then(|a| a.parse().ok())
        .unwrap_or(1536);
    let cols: usize = std::env::args()
        .nth(4)
        .and_then(|a| a.parse().ok())
        .unwrap_or(576);
    println!("{rows}x{cols}, batch {batch}, {threads} thread(s)");
    let values: Vec<f32> = (0..rows * cols)
        .map(|i| ((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0)
        .collect();
    let x: Vec<f32> = (0..cols * batch).map(|i| (i as f32 * 0.01).sin()).collect();

    for (kind, data) in [
        (GgmlType::Q4_0, quantize_q4_0(&values)),
        (GgmlType::Q8_0, quantize_q8_0(&values)),
    ] {
        let packed = kernels::repack_x8(kind, &data, rows, cols).expect("packable");
        let group_bytes = kernels::packed_len(kind, 8 * kind.row_bytes(cols).unwrap());
        let columns: Vec<Prepared<'_>> = x
            .chunks_exact(cols)
            .map(|c| kernels::prepare(kind, c))
            .collect();
        let macs = (rows * cols * batch) as f64;
        let mut out = vec![0.0_f32; 8 * batch];

        // kernel: one group, repeatedly. Everything stays in L1.
        let groups = rows / 8;
        let reps = 2000;
        let t = Instant::now();
        for _ in 0..reps {
            kernels::dot_x8(kind, &packed[..group_bytes], &columns, &mut out);
        }
        let per_group = t.elapsed().as_secs_f64() / reps as f64;
        let kernel_gops = 2.0 * (8 * cols * batch) as f64 / per_group / 1e9;

        // rows: every group once, no output scatter.
        let reps = 200;
        let t = Instant::now();
        for _ in 0..reps {
            for g in 0..groups {
                kernels::dot_x8(
                    kind,
                    &packed[g * group_bytes..(g + 1) * group_bytes],
                    &columns,
                    &mut out,
                );
            }
            std::hint::black_box(&out);
        }
        let rows_time = t.elapsed().as_secs_f64() / reps as f64;

        // matmul: the real path.
        let m = Matrix::new(kind, rows, cols, &data)
            .unwrap()
            .with_packed(&packed)
            .unwrap();
        let mut y = vec![0.0; rows * batch];
        CpuBackend.matmul(m, &x, &mut y, batch).unwrap();
        let reps = 200;
        let t = Instant::now();
        for _ in 0..reps {
            CpuBackend.matmul(m, &x, &mut y, batch).unwrap();
        }
        let matmul_time = t.elapsed().as_secs_f64() / reps as f64;

        // prepare: quantizing the activations, which run_jobs does serially
        // on the calling thread before any parallel work starts.
        let reps = 200;
        let t = Instant::now();
        for _ in 0..reps {
            let prepared: Vec<Prepared<'_>> = x
                .chunks_exact(cols)
                .map(|c| kernels::prepare(kind, c))
                .collect();
            std::hint::black_box(&prepared);
        }
        let prepare_time = t.elapsed().as_secs_f64() / reps as f64;

        println!(
            "{:5}  kernel {kernel_gops:6.1} | rows {:6.1} ({:5.1}% of kernel) | \
             matmul {:6.1} GOP/s | prepare {:7.1} us = {:4.1}% of matmul",
            kind.name(),
            2.0 * macs / rows_time / 1e9,
            100.0 * (2.0 * macs / rows_time / 1e9) / kernel_gops,
            2.0 * macs / matmul_time / 1e9,
            prepare_time * 1e6,
            100.0 * prepare_time / matmul_time,
        );
        let _ = groups;
    }
}

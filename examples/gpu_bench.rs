// GPU matvec/matmul throughput: GB/s of weights streamed from device memory.
//
// Weights stay resident; only the kernel is timed, so this is the ceiling a
// GPU-resident layer stack could reach. Compare against examples/kernel_bench
// (the same shapes on the CPU).
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use kllm::cuda::default_gpu;
    use kllm::quant::{GgmlType, encode_f16, quantize_q4_0, quantize_q8_0};
    use std::time::Instant;

    let gpu = match default_gpu() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };
    println!(
        "{} | compute {}.{} | {} SMs @ {} MHz",
        gpu.name,
        gpu.compute_capability.0,
        gpu.compute_capability.1,
        gpu.multiprocessors,
        gpu.clock_mhz
    );

    // Pure read bandwidth: one f32 matvec over a large matrix.
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
            let w = gpu.upload_matrix(kind, &data, rows, cols).unwrap();
            let xd = gpu.upload(&x).unwrap();
            let y = gpu.alloc(rows * 4).unwrap();
            let reps = (400_000_000 / data.len()).clamp(5, 2000);
            gpu.matmul_matrix(&w, xd.ptr, y.ptr, 1).unwrap();
            gpu.synchronize().unwrap();
            let t = Instant::now();
            for _ in 0..reps {
                gpu.matmul_matrix(&w, xd.ptr, y.ptr, 1).unwrap();
            }
            gpu.synchronize().unwrap();
            let s = t.elapsed().as_secs_f64() / reps as f64;
            println!(
                "{rows:>5}x{cols:<5} {:5} matvec {:8.1} us  {:6.2} GB/s",
                kind.name(),
                s * 1e6,
                data.len() as f64 / s / 1e9
            );
        }
    }

    // Prefill-shaped: batched matmul, effective GOP/s.
    let (rows, cols) = (1536, 576);
    let values: Vec<f32> = (0..rows * cols)
        .map(|i| ((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0)
        .collect();
    for batch in [16, 64] {
        let x: Vec<f32> = (0..cols * batch).map(|i| (i as f32 * 0.01).sin()).collect();
        for (kind, data) in [
            (GgmlType::Q4_0, quantize_q4_0(&values)),
            (GgmlType::Q8_0, quantize_q8_0(&values)),
        ] {
            let w = gpu.upload_matrix(kind, &data, rows, cols).unwrap();
            let xd = gpu.upload(&x).unwrap();
            let y = gpu.alloc(rows * batch * 4).unwrap();
            let reps = 200;
            gpu.matmul_matrix(&w, xd.ptr, y.ptr, batch).unwrap();
            gpu.synchronize().unwrap();
            let t = Instant::now();
            for _ in 0..reps {
                gpu.matmul_matrix(&w, xd.ptr, y.ptr, batch).unwrap();
            }
            gpu.synchronize().unwrap();
            let s = t.elapsed().as_secs_f64() / reps as f64;
            println!(
                "{rows}x{cols} batch {batch:<3} {:5} {:8.1} us  {:6.1} GOP/s",
                kind.name(),
                s * 1e6,
                2.0 * (rows * cols * batch) as f64 / s / 1e9
            );
        }
    }

    // Launch overhead: the floor on per-token time for a GPU-resident stack.
    let w = gpu.upload(&[0.0f32; 64]).unwrap();
    let y = gpu.alloc(64).unwrap();
    let reps = 2000;
    let t = Instant::now();
    for _ in 0..reps {
        gpu.matmul(GgmlType::F32, w.ptr, 4, 16, w.ptr, y.ptr, 1)
            .unwrap();
    }
    gpu.synchronize().unwrap();
    println!(
        "launch+sync overhead: {:.1} us/kernel",
        t.elapsed().as_secs_f64() / reps as f64 * 1e6
    );
}

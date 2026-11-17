use std::{hint::black_box, time::Instant};

use kllm::{
    backend::{Backend, Matrix},
    cpu::CpuBackend,
    kernels,
    quant::{GgmlType, quantize_q4_0},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rows = argument(1, 4096)?;
    let columns = argument(2, 4096)?;
    let samples = argument(3, 9)?;
    if !columns.is_multiple_of(32) || samples < 6 {
        return Err("columns must be a multiple of 32 and samples must be at least 6".into());
    }

    let weights: Vec<f32> = (0..rows * columns)
        .map(|index| ((index * 13 % 257) as f32 - 128.0) / 128.0)
        .collect();
    let data = quantize_q4_0(&weights);
    drop(weights);
    let matrix = Matrix::new(GgmlType::Q4_0, rows, columns, &data)?;
    let input: Vec<f32> = (0..columns)
        .map(|index| ((index * 17 % 257) as f32 - 128.0) / 128.0)
        .collect();
    let mut output = vec![0.0; rows];
    let threads = kllm::pool::init(kllm::pool::default_threads());

    CpuBackend.matvec(matrix, &input, &mut output)?;
    let mut times = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        CpuBackend.matvec(matrix, &input, &mut output)?;
        black_box(&output);
        times.push(start.elapsed().as_secs_f64());
    }
    times.sort_unstable_by(f64::total_cmp);
    let seconds = times[times.len() / 2];

    println!("kLLM Q4_0 matvec benchmark");
    println!("  kernel:     {}", kernels::kernel_name());
    println!("  shape:      {rows} x {columns}");
    println!("  threads:    {threads}");
    println!("  samples:    {samples} (median)");
    println!("  latency:    {:.3} ms", seconds * 1_000.0);
    println!("  rows/s:     {:.0}", rows as f64 / seconds);
    Ok(())
}

fn argument(index: usize, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    match std::env::args().nth(index) {
        Some(value) => Ok(value.parse()?),
        None => Ok(default),
    }
}

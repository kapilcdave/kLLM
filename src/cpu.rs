//! CPU backend: parallel rows over a rayon pool, SIMD dot kernels.

use std::cell::RefCell;

use crate::backend::{Backend, BackendError, Job, Matrix, require_len};
use crate::kernels::{self, Prepared};

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

/// Weight bytes a task should cover at minimum: below this, scheduling
/// overhead outweighs the work (a 576-column Q4_0 row is only ~324 bytes).
const MIN_TASK_BYTES: usize = 48 * 1024;

/// Activation columns processed together by one pass over a task's rows.
const COLUMN_TILE: usize = 8;

/// Raw output pointer shared across tasks that write disjoint rows.
#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: every task writes only the rows it owns (see `run_jobs`).
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

thread_local! {
    static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

struct Task {
    job: usize,
    first_row: usize,
    rows: usize,
}

impl CpuBackend {
    /// Compute every job: `outputs[b * rows + r] = matrix.row(r) · inputs[b]`.
    fn run_jobs(
        &self,
        jobs: &mut [Job<'_, '_>],
        inputs: &[f32],
        batch: usize,
    ) -> Result<(), BackendError> {
        for job in jobs.iter() {
            require_len("matmul inputs", job.matrix.cols * batch, inputs.len())?;
            require_len("matmul outputs", job.matrix.rows * batch, job.output.len())?;
        }
        // Prepare the inputs once per distinct activation format.
        let mut prepared: Vec<(kernels::DotInput, Vec<Prepared<'_>>)> = Vec::new();
        for job in jobs.iter() {
            let format = kernels::dot_input(job.matrix.kind);
            if !prepared.iter().any(|(f, _)| *f == format) {
                let cols = job.matrix.cols;
                let columns = inputs
                    .chunks_exact(cols)
                    .map(|x| kernels::prepare(job.matrix.kind, x))
                    .collect();
                prepared.push((format, columns));
            }
        }
        let threads = crate::pool::global().threads();
        let total_bytes: usize = jobs.iter().map(|j| j.matrix.data.len()).sum::<usize>() * batch;
        let target = (total_bytes / (threads * 3)).max(MIN_TASK_BYTES);
        let mut tasks = Vec::new();
        for (index, job) in jobs.iter().enumerate() {
            let row_cost = (job.matrix.row_bytes() * batch).max(1);
            let mut rows_per_task = target.div_ceil(row_cost).clamp(1, job.matrix.rows);
            if job.matrix.packed.is_some() {
                // Packed matrices are processed in whole groups of eight rows.
                rows_per_task = rows_per_task.next_multiple_of(8);
            }
            let mut first_row = 0;
            while first_row < job.matrix.rows {
                let rows = rows_per_task.min(job.matrix.rows - first_row);
                tasks.push(Task {
                    job: index,
                    first_row,
                    rows,
                });
                first_row += rows;
            }
        }
        let outputs: Vec<OutPtr> = jobs
            .iter_mut()
            .map(|j| OutPtr(j.output.as_mut_ptr()))
            .collect();
        let matrices: Vec<Matrix<'_>> = jobs.iter().map(|j| j.matrix).collect();
        let work = |task: &Task| {
            let matrix = matrices[task.job];
            let out = outputs[task.job];
            let format = kernels::dot_input(matrix.kind);
            let columns = &prepared.iter().find(|(f, _)| *f == format).unwrap().1;
            // Quantized types: walk the rows once per group of COLUMN_TILE
            // columns, so the group's activations stay in L1 while the task's
            // weight rows stream from L2 (wide inputs, e.g. ffn_down with
            // 128 columns, otherwise overflow L2). Float types dequantize the
            // row first, so they take every column per row instead.
            if let Some(packed) = matrix.packed {
                let group_bytes = 8 * matrix.row_bytes();
                let mut values = [0.0_f32; 8 * COLUMN_TILE];
                for (tile, chunk) in columns.chunks(COLUMN_TILE).enumerate() {
                    let start = tile * COLUMN_TILE;
                    let values = &mut values[..8 * chunk.len()];
                    for g in (task.first_row..task.first_row + task.rows).step_by(8) {
                        let group = &packed[(g / 8) * group_bytes..][..group_bytes];
                        kernels::dot_x8(matrix.kind, group, chunk, values);
                        for c in 0..chunk.len() {
                            for r in 0..8 {
                                // SAFETY: rows g..g+8 belong to this task alone.
                                unsafe {
                                    *out.0.add((start + c) * matrix.rows + g + r) =
                                        values[c * 8 + r]
                                };
                            }
                        }
                    }
                }
                return;
            }
            let tile = if format == kernels::DotInput::F32 {
                batch
            } else {
                COLUMN_TILE
            };
            let mut values = vec![0.0_f32; tile.min(batch)];
            SCRATCH.with_borrow_mut(|scratch| {
                for (group, chunk) in columns.chunks(tile).enumerate() {
                    let start = group * tile;
                    let values = &mut values[..chunk.len()];
                    for r in task.first_row..task.first_row + task.rows {
                        kernels::dot_multi(matrix.kind, matrix.row(r), chunk, values, scratch);
                        for (b, v) in values.iter().enumerate() {
                            // SAFETY: row `r` belongs to this task alone, and
                            // `(start + b) * rows + r` is in bounds (checked above).
                            unsafe { *out.0.add((start + b) * matrix.rows + r) = *v };
                        }
                    }
                }
            });
        };
        crate::pool::global().for_each(tasks.len(), |i| work(&tasks[i]));
        Ok(())
    }
}

impl Backend for CpuBackend {
    fn matvec(
        &self,
        matrix: Matrix<'_>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), BackendError> {
        self.run_jobs(&mut [Job { matrix, output }], input, 1)
    }

    fn matmul(
        &self,
        matrix: Matrix<'_>,
        inputs: &[f32],
        outputs: &mut [f32],
        batch: usize,
    ) -> Result<(), BackendError> {
        self.run_jobs(
            &mut [Job {
                matrix,
                output: outputs,
            }],
            inputs,
            batch,
        )
    }

    fn matmul_many(
        &self,
        jobs: &mut [Job<'_, '_>],
        inputs: &[f32],
        batch: usize,
    ) -> Result<(), BackendError> {
        self.run_jobs(jobs, inputs, batch)
    }

    fn rms_norm(
        &self,
        input: &[f32],
        weights: &[f32],
        epsilon: f32,
        output: &mut [f32],
    ) -> Result<(), BackendError> {
        if input.is_empty() {
            return Err(BackendError::InvalidParameter(
                "RMSNorm input cannot be empty",
            ));
        }
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(BackendError::InvalidParameter(
                "RMSNorm epsilon must be finite and non-negative",
            ));
        }
        require_len("RMSNorm weights", input.len(), weights.len())?;
        require_len("RMSNorm output", input.len(), output.len())?;
        rms_norm(input, weights, epsilon, output);
        Ok(())
    }

    fn swiglu(&self, gate: &[f32], up: &[f32], output: &mut [f32]) -> Result<(), BackendError> {
        require_len("SwiGLU up", gate.len(), up.len())?;
        require_len("SwiGLU output", gate.len(), output.len())?;
        for ((destination, gate_value), up_value) in output.iter_mut().zip(gate).zip(up) {
            *destination = silu(*gate_value) * up_value;
        }
        Ok(())
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `output = input / rms(input) * weights` (shapes already validated).
pub fn rms_norm(input: &[f32], weights: &[f32], epsilon: f32, output: &mut [f32]) {
    let mean_square = input
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        / input.len() as f64;
    let inverse_rms = (mean_square + f64::from(epsilon)).sqrt().recip() as f32;
    for ((destination, value), weight) in output.iter_mut().zip(input).zip(weights) {
        *destination = value * inverse_rms * weight;
    }
}

/// In-place RMSNorm.
pub fn rms_norm_in_place(values: &mut [f32], weights: &[f32], epsilon: f32) {
    let mean_square = values
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        / values.len() as f64;
    let inverse_rms = (mean_square + f64::from(epsilon)).sqrt().recip() as f32;
    for (value, weight) in values.iter_mut().zip(weights) {
        *value *= inverse_rms * weight;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{GgmlType, encode_f32, quantize_q8_0};

    fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn computes_matrix_vector_product() {
        let data = encode_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let m = Matrix::new(GgmlType::F32, 2, 3, &data).unwrap();
        let mut output = [0.0; 2];
        CpuBackend
            .matvec(m, &[2.0, 1.0, -1.0], &mut output)
            .unwrap();
        close(&output, &[1.0, 7.0], 1e-6);
    }

    #[test]
    fn matmul_equals_repeated_matvec() {
        let values: Vec<f32> = (0..64 * 96)
            .map(|i| ((i * 31 % 97) as f32 - 48.0) / 50.0)
            .collect();
        let data = quantize_q8_0(&values);
        let m = Matrix::new(GgmlType::Q8_0, 96, 64, &data).unwrap();
        let inputs: Vec<f32> = (0..64 * 3).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut batched = vec![0.0; 96 * 3];
        CpuBackend.matmul(m, &inputs, &mut batched, 3).unwrap();
        for b in 0..3 {
            let mut single = vec![0.0; 96];
            CpuBackend
                .matvec(m, &inputs[b * 64..(b + 1) * 64], &mut single)
                .unwrap();
            close(&batched[b * 96..(b + 1) * 96], &single, 0.0);
        }
    }

    #[test]
    fn computes_rms_norm() {
        let mut output = [0.0; 2];
        CpuBackend
            .rms_norm(&[3.0, 4.0], &[1.0, 2.0], 0.0, &mut output)
            .unwrap();
        let inverse_rms = (12.5_f32).sqrt().recip();
        close(&output, &[3.0 * inverse_rms, 8.0 * inverse_rms], 1e-6);
    }

    #[test]
    fn computes_swiglu() {
        let mut output = [0.0; 2];
        CpuBackend
            .swiglu(&[0.0, 1.0], &[4.0, 2.0], &mut output)
            .unwrap();
        close(&output, &[0.0, 1.462_117_2], 1e-6);
    }

    #[test]
    fn rejects_shape_mismatch() {
        let data = encode_f32(&[1.0, 2.0, 3.0, 4.0]);
        let m = Matrix::new(GgmlType::F32, 2, 2, &data).unwrap();
        let mut output = [0.0; 2];
        assert!(matches!(
            CpuBackend.matvec(m, &[1.0, 1.0, 1.0], &mut output),
            Err(BackendError::ShapeMismatch { .. })
        ));
        assert!(Matrix::new(GgmlType::F32, 3, 2, &data).is_err());
    }
}

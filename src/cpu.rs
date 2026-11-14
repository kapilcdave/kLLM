//! Plain f32 CPU backend used as the correctness reference.

use crate::backend::{Backend, BackendError};

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn matvec_f32(
        &self,
        matrix: &[f32],
        rows: usize,
        columns: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), BackendError> {
        let matrix_values = rows
            .checked_mul(columns)
            .ok_or(BackendError::ArithmeticOverflow)?;
        require_len("matvec matrix", matrix_values, matrix.len())?;
        require_len("matvec input", columns, input.len())?;
        require_len("matvec output", rows, output.len())?;

        for (row, destination) in matrix.chunks_exact(columns).zip(output.iter_mut()) {
            *destination = row
                .iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum();
        }
        Ok(())
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

        let mean_square = input
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            / input.len() as f64;
        let inverse_rms = (mean_square + f64::from(epsilon)).sqrt().recip() as f32;
        for ((destination, value), weight) in output.iter_mut().zip(input).zip(weights) {
            *destination = value * inverse_rms * weight;
        }
        Ok(())
    }

    fn swiglu(&self, gate: &[f32], up: &[f32], output: &mut [f32]) -> Result<(), BackendError> {
        require_len("SwiGLU up", gate.len(), up.len())?;
        require_len("SwiGLU output", gate.len(), output.len())?;
        for ((destination, gate_value), up_value) in output.iter_mut().zip(gate).zip(up) {
            let silu = gate_value / (1.0 + (-gate_value).exp());
            *destination = silu * up_value;
        }
        Ok(())
    }
}

fn require_len(
    operation: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), BackendError> {
    if expected == actual {
        Ok(())
    } else {
        Err(BackendError::ShapeMismatch {
            operation,
            expected,
            actual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut output = [0.0; 2];
        CpuBackend
            .matvec_f32(
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                2,
                3,
                &[2.0, 1.0, -1.0],
                &mut output,
            )
            .unwrap();
        close(&output, &[1.0, 7.0], 1e-6);
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
        let mut output = [0.0; 2];
        assert!(matches!(
            CpuBackend.matvec_f32(&[1.0, 2.0], 2, 2, &[1.0, 1.0], &mut output),
            Err(BackendError::ShapeMismatch { .. })
        ));
    }
}

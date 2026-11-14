//! Backend contract shared by reference CPU and accelerated implementations.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    ShapeMismatch {
        operation: &'static str,
        expected: usize,
        actual: usize,
    },
    ArithmeticOverflow,
    InvalidParameter(&'static str),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch {
                operation,
                expected,
                actual,
            } => write!(
                f,
                "{operation} expected {expected} values but received {actual}"
            ),
            Self::ArithmeticOverflow => write!(f, "tensor shape arithmetic overflow"),
            Self::InvalidParameter(message) => write!(f, "invalid backend parameter: {message}"),
        }
    }
}

impl std::error::Error for BackendError {}

pub trait Backend {
    fn matvec_f32(
        &self,
        matrix: &[f32],
        rows: usize,
        columns: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), BackendError>;

    fn rms_norm(
        &self,
        input: &[f32],
        weights: &[f32],
        epsilon: f32,
        output: &mut [f32],
    ) -> Result<(), BackendError>;

    fn swiglu(&self, gate: &[f32], up: &[f32], output: &mut [f32]) -> Result<(), BackendError>;
}

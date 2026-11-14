//! Backend contract shared by the reference CPU and accelerated
//! implementations.
//!
//! The contract is deliberately narrow: the operations a batch-one decoder
//! spends its time in. A CUDA backend implements the same trait and is tested
//! against [`crate::cpu::CpuBackend`].

use std::fmt;

use crate::quant::GgmlType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    ShapeMismatch {
        operation: &'static str,
        expected: usize,
        actual: usize,
    },
    ArithmeticOverflow,
    InvalidParameter(&'static str),
    UnsupportedType(GgmlType),
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
            Self::UnsupportedType(kind) => write!(f, "no kernel for {}", kind.name()),
        }
    }
}

impl std::error::Error for BackendError {}

/// A row-major weight matrix stored in a GGML type: `rows` rows of `cols`
/// elements each (GGUF `dimensions = [cols, rows]`).
#[derive(Debug, Clone, Copy)]
pub struct Matrix<'a> {
    pub kind: GgmlType,
    pub rows: usize,
    pub cols: usize,
    pub data: &'a [u8],
}

impl<'a> Matrix<'a> {
    pub fn new(
        kind: GgmlType,
        rows: usize,
        cols: usize,
        data: &'a [u8],
    ) -> Result<Self, BackendError> {
        if !kind.is_supported() {
            return Err(BackendError::UnsupportedType(kind));
        }
        let row_bytes = kind.row_bytes(cols).ok_or(BackendError::InvalidParameter(
            "columns are not a whole number of blocks",
        ))?;
        let total = row_bytes
            .checked_mul(rows)
            .ok_or(BackendError::ArithmeticOverflow)?;
        require_len("matrix bytes", total, data.len())?;
        Ok(Self {
            kind,
            rows,
            cols,
            data,
        })
    }

    pub fn row_bytes(&self) -> usize {
        self.data.len() / self.rows.max(1)
    }

    pub fn row(&self, index: usize) -> &'a [u8] {
        let n = self.row_bytes();
        &self.data[index * n..(index + 1) * n]
    }
}

/// One matrix of a [`Backend::matmul_many`] call and where its result goes.
pub struct Job<'m, 'o> {
    pub matrix: Matrix<'m>,
    pub output: &'o mut [f32],
}

pub trait Backend {
    /// `output = matrix · input`.
    fn matvec(
        &self,
        matrix: Matrix<'_>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), BackendError>;

    /// `outputs[b] = matrix · inputs[b]` for a batch of vectors, laid out
    /// contiguously. Backends override this to read the weights once.
    fn matmul(
        &self,
        matrix: Matrix<'_>,
        inputs: &[f32],
        outputs: &mut [f32],
        batch: usize,
    ) -> Result<(), BackendError> {
        require_len("matmul inputs", matrix.cols * batch, inputs.len())?;
        require_len("matmul outputs", matrix.rows * batch, outputs.len())?;
        for (x, y) in inputs
            .chunks_exact(matrix.cols)
            .zip(outputs.chunks_exact_mut(matrix.rows))
        {
            self.matvec(matrix, x, y)?;
        }
        Ok(())
    }

    /// Several matrices applied to the same inputs (e.g. Q, K and V), so a
    /// backend can prepare the inputs once and schedule all rows together.
    fn matmul_many(
        &self,
        jobs: &mut [Job<'_, '_>],
        inputs: &[f32],
        batch: usize,
    ) -> Result<(), BackendError> {
        for job in jobs {
            self.matmul(job.matrix, inputs, job.output, batch)?;
        }
        Ok(())
    }

    fn rms_norm(
        &self,
        input: &[f32],
        weights: &[f32],
        epsilon: f32,
        output: &mut [f32],
    ) -> Result<(), BackendError>;

    fn swiglu(&self, gate: &[f32], up: &[f32], output: &mut [f32]) -> Result<(), BackendError>;
}

pub(crate) fn require_len(
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

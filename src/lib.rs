//! Core library for kLLM.

// `chunks_exact(N)` reads better than `as_chunks::<N>()` here and compiles the same.
#![allow(clippy::chunks_exact_to_as_chunks)]

#[cfg(not(target_endian = "little"))]
compile_error!("kLLM reads GGUF tensors in place and requires a little-endian target");

pub mod backend;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod eval;
pub mod gguf;
pub mod json;
pub mod kernels;
pub mod model;
pub mod planner;
pub mod pool;
pub mod quant;
pub mod sampler;
pub mod serve;
pub mod tokenizer;

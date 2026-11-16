//! Perplexity against llama.cpp on a WikiText-2 excerpt.
//!
//! Quantized decoders quantize activations to 8 bits, which makes individual
//! log-probabilities sensitive to summation order; perplexity over a corpus
//! averages that noise away and is the meaningful agreement check.
//!
//! The fixture holds llama.cpp's result with flash attention on (its default)
//! and off. On quantized models those two differ by up to 0.4%: that spread is
//! llama.cpp's own implementation noise, so kLLM is held to 1% of both. F16
//! has no activation quantization and is held to 0.1%.
//!
//! Resist widening these bounds. Q2_K once sat 2.3% out and looked like the
//! hypersensitivity of a wrecked model (pure Q2_K of a 135M model perplexes
//! at 228 against f16's 24.5, and llama.cpp disagreed with itself by 0.56%
//! there against 0.1-0.4% elsewhere). It was a real defect: Q2_K had no
//! kernel against quantized activations, so it ran on exact f32 activations
//! where llama.cpp quantizes them to Q8_K. Giving it one moved perplexity
//! from 222.65 to 226.24 and the gap from 2.31% to 0.74%.

mod common;

use kllm::{eval::perplexity, model::Model, tokenizer::Tokenizer};

fn check(model_file: &str, tolerance: f64) {
    let Some(path) = common::model(model_file) else {
        return;
    };
    let reference = common::fixture("perplexity.json");
    let entry = &reference["models"][model_file];
    let expected = [
        entry["ppl_fa_on"].as_f64().unwrap(),
        entry["ppl_fa_off"].as_f64().unwrap(),
    ];
    let n_ctx = reference["context"].as_u64().unwrap() as usize;
    let chunks = reference["chunks"].as_u64().unwrap() as usize;
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wiki.test.head.txt"),
    )
    .unwrap();
    let model = Model::load(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(model.info()).unwrap();
    let tokens = tokenizer.encode(&text, true, false);
    let result = perplexity(&model, &tokenizer, &tokens, n_ctx, Some(chunks), |_, _| {}).unwrap();
    assert_eq!(result.chunks, chunks);
    for (label, reference) in ["fa on", "fa off"].iter().zip(expected) {
        let relative = (result.ppl - reference).abs() / reference;
        eprintln!(
            "{model_file}: PPL {:.4} vs llama.cpp ({label}) {reference:.4}: {:+.3}%",
            result.ppl,
            (result.ppl / reference - 1.0) * 100.0
        );
        assert!(
            relative < tolerance,
            "{model_file}: PPL {} is {:.3}% from llama.cpp ({label})",
            result.ppl,
            relative * 100.0
        );
    }
}

/// The float types carry no activation quantization, so they are held an
/// order of magnitude tighter than the quantized ones.
#[test]
fn smollm2_f16() {
    check("SmolLM2-135M-Instruct-f16.gguf", 0.001);
}

#[test]
fn smollm2_bf16() {
    check("SmolLM2-135M-Instruct-BF16.gguf", 0.001);
}

#[test]
fn smollm2_q8_0() {
    check("SmolLM2-135M-Instruct-Q8_0.gguf", 0.01);
}

#[test]
fn smollm2_q4_0() {
    check("SmolLM2-135M-Instruct-Q4_0.gguf", 0.01);
}

#[test]
fn smollm2_q4_1() {
    check("SmolLM2-135M-Instruct-Q4_1.gguf", 0.01);
}

#[test]
fn smollm2_q5_0() {
    check("SmolLM2-135M-Instruct-Q5_0.gguf", 0.01);
}

#[test]
fn smollm2_q5_1() {
    check("SmolLM2-135M-Instruct-Q5_1.gguf", 0.01);
}

/// SmolLM2's 576-wide rows are not a multiple of 256, so llama.cpp can only
/// put a K-quant in the 1536-wide `ffn_down`; the rest of the model falls
/// back to a legacy type. These three are still the only coverage the Q2_K,
/// Q5_K and Q6_K kernels get on a real model's weights.
#[test]
fn smollm2_q2_k() {
    check("SmolLM2-135M-Instruct-Q2_K.gguf", 0.01);
}

#[test]
fn smollm2_q5_k_m() {
    check("SmolLM2-135M-Instruct-Q5_K_M.gguf", 0.01);
}

#[test]
fn smollm2_q6_k() {
    check("SmolLM2-135M-Instruct-Q6_K.gguf", 0.01);
}

#[test]
fn qwen3_q4_k_m() {
    check("Qwen3-0.6B-Q4_K_M.gguf", 0.01);
}

/// Q3_K, the last common K-quant. SmolLM2 cannot carry it (576-wide rows
/// are not a multiple of 256, so llama.cpp falls back to legacy types), so
/// this one is Qwen3-0.6B requantized from Q8_0 with `--allow-requantize`.
/// The doubly-quantized weights make the absolute perplexity meaningless and
/// the agreement with llama.cpp exactly as meaningful as ever.
#[test]
fn qwen3_q3_k_m() {
    check("Qwen3-0.6B-Q3_K_M.gguf", 0.01);
}

#[test]
fn qwen3_q8_0() {
    check("Qwen3-0.6B-Q8_0.gguf", 0.01);
}

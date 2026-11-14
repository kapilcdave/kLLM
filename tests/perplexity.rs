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

#[test]
fn smollm2_f16() {
    check("SmolLM2-135M-Instruct-f16.gguf", 0.001);
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
fn qwen3_q4_k_m() {
    check("Qwen3-0.6B-Q4_K_M.gguf", 0.01);
}

#[test]
fn qwen3_q8_0() {
    check("Qwen3-0.6B-Q8_0.gguf", 0.01);
}

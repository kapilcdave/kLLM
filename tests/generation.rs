//! End-to-end correctness against llama.cpp.
//!
//! For each recorded prompt the test feeds the same prompt tokens, decodes
//! greedily, and compares every step with llama.cpp's token and top-10
//! log-probabilities. Two engines with different (but both valid) float
//! summation orders may legitimately pick different tokens where the top two
//! candidates are nearly tied, so a divergence is accepted only at a step
//! whose reference margin is below the model's tie margin; comparison stops
//! there.
//!
//! How tight can this be? llama.cpp disagrees with *itself* when only its
//! flash-attention switch is flipped (measured on these fixtures, b12099):
//!
//! | model          | mean top-10 |Δ log p| | max   | greedy run before divergence |
//! |----------------|-----------------------|-------|------------------------------|
//! | SmolLM2 F16    | 0.003 - 0.005         | 0.03  | 32/32 on every prompt        |
//! | SmolLM2 Q8_0   | 0.09 - 0.14           | 0.78  | as few as 11/32              |
//! | Qwen3 Q4_K_M   | 0.14 - 0.18           | 0.90  | as few as 8/32               |
//!
//! Quantized paths quantize activations to 8 bits, which amplifies tiny
//! summation-order differences, so their bounds sit at about 1.5x llama.cpp's
//! own spread and this test only catches gross breakage there; the precise
//! check for quantized models is `tests/perplexity.rs`. F16 is held tight.

mod common;

use kllm::model::{Logits, Model};
use kllm::sampler::log_prob;
use kllm::tokenizer::Tokenizer;

// About 1.5x llama.cpp's own flash-attention on/off spread (see above).
const QUANTIZED_TIE_MARGIN: f64 = 0.5;
const QUANTIZED_TOP1_ERROR: f64 = 0.6;
const QUANTIZED_MEAN_ERROR: f64 = 0.3;

struct Report {
    matched: usize,
    total: usize,
    max_logprob_error: f64,
    mean_logprob_error: f64,
}

/// `top1_error`: bound on |Δ log p| of the reference's chosen token at every
/// matched step. `mean_error`: bound on the mean |Δ log p| over the top 10.
fn check(
    model_file: &str,
    fixture: &str,
    tie_margin: f64,
    top1_error: f64,
    mean_error: f64,
) -> Option<Vec<Report>> {
    let path = common::model(model_file)?;
    let model = Model::load(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(model.info()).unwrap();
    let reference = common::fixture(fixture);
    let mut reports = Vec::new();
    for case in reference["cases"].as_array().unwrap() {
        let prompt = case["prompt"].as_str().unwrap();
        let ids = |v: &serde_json::Value| -> Vec<u32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect()
        };
        let prompt_ids = ids(&case["prompt_ids"]);
        assert_eq!(
            tokenizer.encode(prompt, true, true),
            prompt_ids,
            "prompt tokens for {prompt:?}"
        );

        let mut session = model.new_session(512, 64).unwrap();
        let mut logits = session.forward(&prompt_ids, Logits::Last).unwrap().to_vec();
        let steps = case["steps"].as_array().unwrap();
        let (mut matched, mut errors, mut top1) = (0, Vec::new(), 0.0_f64);
        for step in steps {
            let expected = step["id"].as_u64().unwrap() as u32;
            top1 =
                top1.max((log_prob(&logits, expected) - step["logprob"].as_f64().unwrap()).abs());
            for top in step["top"].as_array().unwrap() {
                let id = top[0].as_u64().unwrap() as u32;
                let lp = top[1].as_f64().unwrap();
                errors.push((log_prob(&logits, id) - lp).abs());
            }
            let ours = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            if ours != expected {
                let top = step["top"].as_array().unwrap();
                let ref_lp = top[0][1].as_f64().unwrap();
                let our_lp = top
                    .iter()
                    .find(|t| t[0].as_u64() == Some(ours as u64))
                    .map(|t| t[1].as_f64().unwrap())
                    .unwrap_or(f64::NEG_INFINITY);
                assert!(
                    ref_lp - our_lp < tie_margin,
                    "{prompt:?} step {matched}: chose {ours}, llama.cpp chose {expected} with margin {}",
                    ref_lp - our_lp
                );
                break;
            }
            matched += 1;
            logits = session.forward(&[expected], Logits::Last).unwrap().to_vec();
        }
        let max = errors.iter().cloned().fold(0.0, f64::max);
        let mean = errors.iter().sum::<f64>() / errors.len() as f64;
        assert!(
            top1 < top1_error,
            "{prompt:?}: chosen-token log-prob error {top1} exceeds {top1_error}"
        );
        assert!(
            mean < mean_error,
            "{prompt:?}: mean top-10 log-prob error {mean} exceeds {mean_error}"
        );
        reports.push(Report {
            matched,
            total: steps.len(),
            max_logprob_error: max,
            mean_logprob_error: mean,
        });
    }
    for r in &reports {
        eprintln!(
            "{model_file}: {}/{} greedy tokens identical, top-10 log-prob error mean {:.4} max {:.4}",
            r.matched, r.total, r.mean_logprob_error, r.max_logprob_error
        );
    }
    Some(reports)
}

#[test]
fn smollm2_f16_matches_llama_cpp() {
    // F16 has no activation quantization: agreement is tight.
    check(
        "SmolLM2-135M-Instruct-f16.gguf",
        "smollm2-135m-f16.greedy.json",
        0.05,
        0.03,
        0.01,
    );
}

#[test]
fn smollm2_q8_0_matches_llama_cpp() {
    check(
        "SmolLM2-135M-Instruct-Q8_0.gguf",
        "smollm2-135m-Q8_0.greedy.json",
        QUANTIZED_TIE_MARGIN,
        QUANTIZED_TOP1_ERROR,
        QUANTIZED_MEAN_ERROR,
    );
}

#[test]
fn smollm2_q4_0_matches_llama_cpp() {
    check(
        "SmolLM2-135M-Instruct-Q4_0.gguf",
        "smollm2-135m-Q4_0.greedy.json",
        QUANTIZED_TIE_MARGIN,
        QUANTIZED_TOP1_ERROR,
        QUANTIZED_MEAN_ERROR,
    );
}

#[test]
fn qwen3_q4_k_m_matches_llama_cpp() {
    check(
        "Qwen3-0.6B-Q4_K_M.gguf",
        "qwen3-0.6b-q4_k_m.greedy.json",
        QUANTIZED_TIE_MARGIN,
        QUANTIZED_TOP1_ERROR,
        QUANTIZED_MEAN_ERROR,
    );
}

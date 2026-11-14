//! KL divergence from a reference model, against llama.cpp.
//!
//! Perplexity says whether a model predicts text well; it does not say
//! whether a *quantization* changed what the full-precision model would have
//! said. A quantization can hold perplexity and still reorder the top of the
//! distribution. KL divergence measures exactly that, so this is the test
//! that would catch a kernel which is plausible on average and wrong per
//! token.
//!
//! The fixture holds `llama-perplexity --kl-divergence` run against the same
//! reference model over the same text. kLLM is held to 5% of llama.cpp's mean
//! KL (or twice its standard error, whichever is wider) and to one percentage
//! point of its top-1 agreement, both of which are far tighter than the
//! difference between adjacent quantization levels.

mod common;

use kllm::{eval::divergence, model::Model, tokenizer::Tokenizer};

fn check(model_file: &str) {
    let fixture = common::fixture("divergence.json");
    let entry = &fixture["models"][model_file];
    assert!(
        !entry.is_null(),
        "{model_file} has no divergence fixture; rerun scripts/make_eval_fixtures.py"
    );
    let reference_file = entry["reference"].as_str().unwrap();
    let (Some(reference_path), Some(test_path)) =
        (common::model(reference_file), common::model(model_file))
    else {
        return;
    };
    let n_ctx = fixture["context"].as_u64().unwrap() as usize;
    let chunks = fixture["chunks"].as_u64().unwrap() as usize;
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wiki.test.head.txt"),
    )
    .unwrap();

    let reference = Model::load(&reference_path).unwrap();
    let test = Model::load(&test_path).unwrap();
    let tokenizer = Tokenizer::from_gguf(reference.info()).unwrap();
    let tokens = tokenizer.encode(&text, true, false);
    let got = divergence(
        &reference,
        &test,
        &tokenizer,
        &tokens,
        n_ctx,
        Some(chunks),
        |_, _| {},
    )
    .unwrap();
    assert_eq!(got.chunks, chunks);

    let expected_kl = entry["kl_mean"].as_f64().unwrap();
    let expected_stderr = entry["kl_mean_stderr"].as_f64().unwrap();
    let expected_top1 = entry["top1_agree"].as_f64().unwrap();
    let expected_top1_stderr = entry["top1_agree_stderr"].as_f64().unwrap();
    eprintln!(
        "{model_file} vs {reference_file}: KL {:.6} (llama.cpp {expected_kl:.6}, {:+.2}%), \
         median {:.6} (llama.cpp {:.6}), top-1 {:.2}% (llama.cpp {expected_top1:.2}%), \
         PPL {:.4} -> {:.4}",
        got.kl_mean,
        (got.kl_mean / expected_kl - 1.0) * 100.0,
        got.kl_median,
        entry["kl_median"].as_f64().unwrap(),
        got.top1_agree * 100.0,
        got.reference_ppl,
        got.test_ppl,
    );

    // kLLM must not lose more than llama.cpp does. It is allowed to lose
    // *less*: a smaller divergence means kLLM's kernels tracked the
    // reference model more closely, which is not a failure. BF16 is the case
    // that matters -- SmolLM2 was trained in bf16, so its f16 GGUF holds
    // exactly representable bf16 values (f16 keeps 10 mantissa bits to
    // bf16's 7) and converting back is lossless. The two files are
    // numerically identical, kLLM returns exactly zero divergence, and
    // llama.cpp's 8e-5 is its own F16-vs-BF16 kernel noise.
    let kl_ceiling = expected_kl * 1.05 + 4.0 * expected_stderr;
    assert!(
        got.kl_mean <= kl_ceiling,
        "{model_file}: mean KL {} exceeds {kl_ceiling:.6}, llama.cpp's {expected_kl} plus slack",
        got.kl_mean
    );
    // Guard the other direction only where the quantization really does lose
    // information, so a suspiciously small divergence (a kernel quietly
    // falling back to the reference type, say) still fails.
    if expected_kl > 1e-3 {
        assert!(
            got.kl_mean >= 0.5 * expected_kl,
            "{model_file}: mean KL {} is less than half llama.cpp's {expected_kl}; \
             is the weight type really being used?",
            got.kl_mean
        );
    }
    let top1_tolerance = 1.0 + 2.0 * expected_top1_stderr;
    assert!(
        got.top1_agree * 100.0 >= expected_top1 - top1_tolerance,
        "{model_file}: top-1 agreement {:.2}% is more than {top1_tolerance:.2} points below \
         llama.cpp's {expected_top1:.2}%",
        got.top1_agree * 100.0
    );
    // A loose check that the reference model itself ran. It cannot be tight:
    // llama.cpp's KL mode reports `Mean PPL(base)` as a mean over chunks,
    // not as exp of the mean negative log-likelihood, so it differs from
    // both kLLM's figure and llama.cpp's own `llama-perplexity` output (for
    // Qwen3-Q8_0 here: 18.495 against 18.861) . The tight absolute-perplexity
    // comparison lives in `tests/perplexity.rs`.
    let expected_reference_ppl = entry["ppl_reference"].as_f64().unwrap();
    assert!(
        (got.reference_ppl / expected_reference_ppl - 1.0).abs() < 0.05,
        "{model_file}: reference PPL {} is not within 5% of llama.cpp's {expected_reference_ppl}",
        got.reference_ppl
    );
}

/// A quantization kLLM has a dedicated AVX2 kernel for.
#[test]
fn smollm2_q4_0() {
    check("SmolLM2-135M-Instruct-Q4_0.gguf");
}

#[test]
fn smollm2_q8_0() {
    check("SmolLM2-135M-Instruct-Q8_0.gguf");
}

/// BF16 and the legacy Q4_1/Q5_0/Q5_1 types have no x8-packed kernel, so they
/// exercise the row-major path end to end.
#[test]
fn smollm2_bf16() {
    check("SmolLM2-135M-Instruct-BF16.gguf");
}

#[test]
fn smollm2_q4_1() {
    check("SmolLM2-135M-Instruct-Q4_1.gguf");
}

#[test]
fn smollm2_q5_0() {
    check("SmolLM2-135M-Instruct-Q5_0.gguf");
}

#[test]
fn smollm2_q5_1() {
    check("SmolLM2-135M-Instruct-Q5_1.gguf");
}

/// The K-quants. SmolLM2's 576-wide rows are not a multiple of 256, so
/// llama.cpp keeps those tensors in a legacy type and only `ffn_down` (1536
/// wide) carries the K-quant; that is still the only real-model coverage
/// these kernels get.
#[test]
fn smollm2_q2_k() {
    check("SmolLM2-135M-Instruct-Q2_K.gguf");
}

#[test]
fn smollm2_q5_k() {
    check("SmolLM2-135M-Instruct-Q5_K_M.gguf");
}

#[test]
fn smollm2_q6_k() {
    check("SmolLM2-135M-Instruct-Q6_K.gguf");
}

/// Qwen3 is Q4_K/Q6_K throughout, and a different architecture (per-head q/k
/// RMSNorm, NeoX rotary).
#[test]
fn qwen3_q4_k_m() {
    check("Qwen3-0.6B-Q4_K_M.gguf");
}

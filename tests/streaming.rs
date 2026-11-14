//! Streaming weights from disk (the RAM tier below VRAM and RAM).
//!
//! A model larger than its RAM budget keeps the head and a prefix of blocks
//! resident and re-reads the rest from the file every token, releasing each
//! streamed block after use. Streaming changes where bytes come from, never
//! what is computed.

mod common;

use kllm::model::{LoadOptions, Logits, Model, Residency};

#[test]
fn residency_plan_respects_the_budget() {
    let Some(path) = common::model("SmolLM2-135M-Instruct-Q4_0.gguf") else {
        return;
    };
    let info = kllm::gguf::inspect_file(&path).unwrap();
    let total = info.known_weight_bytes().unwrap();
    let all = Residency::plan(&info, 30, None);
    assert_eq!(
        (all.first_streamed, all.head_streamed, all.streamed_bytes),
        (30, false, 0)
    );
    assert_eq!(Residency::plan(&info, 30, Some(total)).first_streamed, 30);
    // Norms and the embedding lookup always stay resident; they are tiny.
    let fixed = Residency::plan(&info, 30, Some(0)).resident_bytes;
    let mut previous = u64::MAX;
    for fraction in [0.9, 0.6, 0.4, 0.1, 0.0] {
        let budget = (total as f64 * fraction) as u64;
        let r = Residency::plan(&info, 30, Some(budget));
        assert!(
            r.resident_bytes <= budget.max(fixed),
            "{} resident over a {budget} budget",
            r.resident_bytes
        );
        assert_eq!(r.resident_bytes + r.streamed_bytes, total);
        // Block counts need not fall monotonically (once the head no longer
        // fits, its share goes to blocks), but resident bytes must.
        assert!(
            r.resident_bytes <= previous,
            "less RAM must not keep more bytes"
        );
        previous = r.resident_bytes;
    }
    let none = Residency::plan(&info, 30, Some(0));
    assert_eq!(none.first_streamed, 0);
    assert!(none.head_streamed);
}

/// F16 has no repacked layout, so a fully streamed model runs the very same
/// kernels on the very same bytes: logits must match bit for bit, through
/// prefill and decode.
#[test]
fn streamed_f16_logits_are_identical() {
    let Some(path) = common::model("SmolLM2-135M-Instruct-f16.gguf") else {
        return;
    };
    let resident = Model::load(&path).unwrap();
    let streamed = Model::load_with(
        &path,
        LoadOptions {
            ram_budget: Some(0),
        },
    )
    .unwrap();
    assert_eq!(streamed.residency.first_streamed, 0);
    let tokens: Vec<u32> = (0..24).map(|i| (i * 1931 + 7) % 49152).collect();
    let mut a = resident.new_session(64, 16).unwrap();
    let mut b = streamed.new_session(64, 16).unwrap();
    let mut steps: Vec<&[u32]> = vec![&tokens[..16]];
    steps.extend(tokens[16..].chunks(1));
    for step in steps {
        let la = a.forward(step, Logits::Last).unwrap().to_vec();
        let lb = b.forward(step, Logits::Last).unwrap();
        assert!(
            la.iter().zip(lb).all(|(x, y)| x.to_bits() == y.to_bits()),
            "streamed logits differ"
        );
    }
}

/// Partially streamed Q4_0: resident blocks run the packed kernels, streamed
/// ones the row-major kernels. Perplexity must stay within the usual bounds.
#[test]
fn half_streamed_q4_0_matches_llama_cpp() {
    let Some(path) = common::model("SmolLM2-135M-Instruct-Q4_0.gguf") else {
        return;
    };
    let info = kllm::gguf::inspect_file(&path).unwrap();
    let budget = info.known_weight_bytes().unwrap() / 2;
    let model = Model::load_with(
        &path,
        LoadOptions {
            ram_budget: Some(budget),
        },
    )
    .unwrap();
    assert!(model.residency.first_streamed > 0 && model.residency.first_streamed < 30);
    let reference = common::fixture("perplexity.json");
    let entry = &reference["models"]["SmolLM2-135M-Instruct-Q4_0.gguf"];
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wiki.test.head.txt"),
    )
    .unwrap();
    let tokenizer = kllm::tokenizer::Tokenizer::from_gguf(model.info()).unwrap();
    let tokens = tokenizer.encode(&text, true, false);
    let n_ctx = reference["context"].as_u64().unwrap() as usize;
    let chunks = reference["chunks"].as_u64().unwrap() as usize;
    let got = kllm::eval::perplexity(&model, &tokenizer, &tokens, n_ctx, Some(chunks), |_, _| {})
        .unwrap()
        .ppl;
    for label in ["ppl_fa_on", "ppl_fa_off"] {
        let expected = entry[label].as_f64().unwrap();
        let relative = (got - expected).abs() / expected;
        eprintln!("streamed Q4_0: PPL {got:.4} vs llama.cpp {label} {expected:.4}");
        assert!(
            relative < 0.01,
            "streamed PPL {got} is {:.3}% from {expected}",
            relative * 100.0
        );
    }
}

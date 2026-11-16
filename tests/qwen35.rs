//! Real-model regression for GGUF's tiled DeltaNet head layout, gated
//! attention, convolution history, and recurrent session reset/rewind.
mod common;
use kllm::{
    model::{Logits, Model},
    tokenizer::Tokenizer,
};

fn model() -> Option<Model> {
    let path = std::env::var_os("KLLM_QWEN35_MODEL")
        .map(std::path::PathBuf::from)
        .or_else(|| common::model("Qwen3.5-4B-Q4_K_M.gguf"))?;
    kllm::pool::init(8);
    Some(Model::load(path).unwrap())
}

#[test]
fn recurrent_sessions_reset_rewind_and_decode() {
    let Some(model) = model() else {
        return;
    };
    let tokenizer = Tokenizer::from_gguf(model.info()).unwrap();
    let tokens = tokenizer.encode(
        "The capital of France is Paris. The capital of Italy is",
        true,
        false,
    );
    assert!(tokens.len() > 8);
    let mut session = model.new_session(64, 4).unwrap();
    let (kv, scratch) = kllm::model::Session::estimate(&model.config, 64, 4);
    assert_eq!(session.allocated_bytes(), kv + scratch);
    let expected = session.forward(&tokens, Logits::All).unwrap().to_vec();
    assert!(expected.iter().all(|v| v.is_finite()));
    let vocab = model.config.n_vocab;
    session.truncate(0);
    let reset = session.forward(&tokens, Logits::All).unwrap();
    assert_eq!(
        expected, reset,
        "reset must clear both convolution and recurrent state"
    );
    session.truncate(4);
    let rewound = session.forward(&tokens[4..], Logits::All).unwrap();
    assert_eq!(
        &expected[4 * vocab..],
        rewound,
        "rewind must reconstruct the retained prefix"
    );
    // Single-token decoding and prefill can differ slightly because the
    // quantized matrix kernels accumulate in different orders.
    let mut decode = model.new_session(64, 1).unwrap();
    let mut agreed = 0;
    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    for (i, token) in tokens.iter().enumerate() {
        let got = decode.forward(&[*token], Logits::Last).unwrap();
        let reference = &expected[i * vocab..(i + 1) * vocab];
        if argmax(got) == argmax(reference) {
            agreed += 1;
        }
        let rms = (got
            .iter()
            .zip(reference)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / vocab as f32)
            .sqrt();
        assert!(rms < 0.3, "token {i}: batch/decode logits RMS {rms}");
    }
    assert!(
        agreed * 5 >= tokens.len() * 4,
        "only {agreed}/{} argmax predictions agree",
        tokens.len()
    );
    assert!(
        session.allocated_bytes() >= 48 * 1024 * 1024,
        "account for recurrent state"
    );
}

#[test]
fn perplexity_matches_llama_cpp() {
    let Some(model) = model() else {
        return;
    };
    let tokenizer = Tokenizer::from_gguf(model.info()).unwrap();
    let text = include_str!("fixtures/wiki.test.head.txt");
    let tokens = tokenizer.encode(text, true, false);
    let reference = common::fixture("qwen35-perplexity.json");
    let context = reference["context"].as_u64().unwrap() as usize;
    let chunks = reference["chunks"].as_u64().unwrap() as usize;
    let result = kllm::eval::perplexity(
        &model,
        &tokenizer,
        &tokens,
        context,
        Some(chunks),
        |_, _| {},
    )
    .unwrap();
    let expected = reference["ppl"].as_f64().unwrap();
    eprintln!("Qwen3.5-4B: {} vs llama.cpp {expected}", result.ppl);
    assert_eq!(result.chunks, chunks);
    assert!((result.ppl / expected - 1.0).abs() < reference["tolerance"].as_f64().unwrap());
}

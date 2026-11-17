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

/// Resuming a rewind from a checkpoint has to land on exactly the state a
/// replay would have rebuilt, or the saving is a wrong answer.
///
/// This is the property the server leans on: a chat client sends a prompt that
/// shares a long prefix with the last one and then diverges, and on a
/// recurrent model that rewind otherwise replays the whole prefix. One
/// measured case on an 11k-token prompt reused 8496 tokens and still took
/// 739 s because of it.
#[test]
fn a_checkpoint_resumes_a_rewind_exactly() {
    let Some(model) = model() else {
        return;
    };
    let tokenizer = Tokenizer::from_gguf(model.info()).unwrap();
    let prefix = tokenizer.encode("The capital of France is Paris, and the", true, false);
    let first = tokenizer.encode(" capital of Italy is Rome too", false, false);
    let second = tokenizer.encode(" largest city in Germany is", false, false);
    assert!(prefix.len() > 4 && !first.is_empty() && !second.is_empty());

    // What a session that only ever saw `prefix + second` computes.
    let mut whole = prefix.clone();
    whole.extend_from_slice(&second);
    let mut fresh = model.new_session(128, 8).unwrap();
    let expected = fresh.forward(&whole, Logits::Last).unwrap().to_vec();
    assert!(expected.iter().all(|v| v.is_finite()));

    // The replay path: rewind with no checkpoint, so the recurrence is
    // rebuilt from the beginning of the stream.
    let mut replayed = model.new_session(128, 8).unwrap();
    replayed.forward(&prefix, Logits::Last).unwrap();
    replayed.forward(&first, Logits::Last).unwrap();
    assert_eq!(replayed.checkpoint_pos(), None);
    replayed.truncate(prefix.len());
    let by_replay = replayed.forward(&second, Logits::Last).unwrap().to_vec();
    assert_eq!(
        expected, by_replay,
        "a rewind without a checkpoint must rebuild the recurrence"
    );

    // The checkpoint path: same rewind, resumed from a saved state.
    let mut resumed = model.new_session(128, 8).unwrap();
    resumed.forward(&prefix, Logits::Last).unwrap();
    resumed.checkpoint();
    assert_eq!(resumed.checkpoint_pos(), Some(prefix.len()));
    resumed.forward(&first, Logits::Last).unwrap();
    resumed.truncate(prefix.len());
    let by_checkpoint = resumed.forward(&second, Logits::Last).unwrap().to_vec();
    assert_eq!(
        expected, by_checkpoint,
        "resuming from a checkpoint must equal replaying"
    );
}

/// A checkpoint describes the state after a particular run of tokens. Once
/// those positions are discarded they may be replaced by different tokens, so
/// keeping it would be describing a stream this session never saw.
#[test]
fn a_discarded_checkpoint_is_not_reused() {
    let Some(model) = model() else {
        return;
    };
    let tokenizer = Tokenizer::from_gguf(model.info()).unwrap();
    let a = tokenizer.encode("The capital of France is", true, false);
    let b = tokenizer.encode(" Paris and also", false, false);
    let other = tokenizer.encode("Mountains in Nepal are", true, false);

    let mut session = model.new_session(128, 8).unwrap();
    session.forward(&a, Logits::Last).unwrap();
    session.forward(&b, Logits::Last).unwrap();
    session.checkpoint();
    assert_eq!(session.checkpoint_pos(), Some(a.len() + b.len()));

    // Rewinding below the checkpoint must drop it.
    session.truncate(a.len());
    assert_eq!(session.checkpoint_pos(), None);

    // And starting over must still agree with a session that never saw `a`.
    session.truncate(0);
    let got = session.forward(&other, Logits::Last).unwrap().to_vec();
    let mut clean = model.new_session(128, 8).unwrap();
    let expected = clean.forward(&other, Logits::Last).unwrap();
    assert_eq!(expected, got, "a stale checkpoint must not survive a reset");
}

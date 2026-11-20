//! GPU-resident layers (M6) against llama.cpp and against kLLM's own CPU path.
//!
//! Skipped (passes with a notice) when no GPU or model is available. The GPU
//! keeps activations in f32 where the CPU quantizes them to 8 bits for the
//! quantized types, so the two paths are compared by perplexity, which
//! averages that difference out, with the same bounds as tests/perplexity.rs.

#![cfg(feature = "cuda")]

mod common;

use kllm::{
    cuda::default_gpu,
    eval::perplexity,
    model::{Logits, Model},
    tokenizer::Tokenizer,
};

fn ppl(model: &Model, n_ctx: usize, chunks: usize) -> f64 {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wiki.test.head.txt"),
    )
    .unwrap();
    let tokenizer = Tokenizer::from_gguf(model.info()).unwrap();
    let tokens = tokenizer.encode(&text, true, false);
    perplexity(model, &tokenizer, &tokens, n_ctx, Some(chunks), |_, _| {})
        .unwrap()
        .ppl
}

/// Offload `fraction` of the blocks (and the head when all of them go) and
/// check perplexity against llama.cpp.
fn check(model_file: &str, fraction: f64, tolerance: f64) {
    let Some(path) = common::model(model_file) else {
        return;
    };
    let gpu = match default_gpu() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let reference = common::fixture("perplexity.json");
    let entry = &reference["models"][model_file];
    let n_ctx = reference["context"].as_u64().unwrap() as usize;
    let chunks = reference["chunks"].as_u64().unwrap() as usize;
    let mut model = Model::load(&path).unwrap();
    let layers = (model.config.n_layer as f64 * fraction).round() as usize;
    let head = layers == model.config.n_layer;
    let placed = model.offload(gpu, layers, head).unwrap();
    assert_eq!(model.cpu_layers(), model.config.n_layer - layers);
    let got = ppl(&model, n_ctx, chunks);
    for label in ["ppl_fa_on", "ppl_fa_off"] {
        let expected = entry[label].as_f64().unwrap();
        let relative = (got - expected).abs() / expected;
        eprintln!(
            "{model_file} ({} GPU layers, head {}): PPL {got:.4} vs llama.cpp {label} {expected:.4}: {:+.3}%",
            placed.gpu_layers,
            placed.head,
            (got / expected - 1.0) * 100.0
        );
        assert!(
            relative < tolerance,
            "{model_file}: GPU PPL {got} is {:.3}% from {expected}",
            relative * 100.0
        );
    }
}

#[test]
fn smollm2_f16_all_on_gpu() {
    check("SmolLM2-135M-Instruct-f16.gguf", 1.0, 0.001);
}

#[test]
fn smollm2_q4_0_all_on_gpu() {
    check("SmolLM2-135M-Instruct-Q4_0.gguf", 1.0, 0.01);
}

#[test]
fn smollm2_q8_0_half_on_gpu() {
    check("SmolLM2-135M-Instruct-Q8_0.gguf", 0.5, 0.01);
}

#[test]
fn qwen3_q4_k_m_all_on_gpu() {
    check("Qwen3-0.6B-Q4_K_M.gguf", 1.0, 0.01);
}

#[test]
fn qwen3_q8_0_half_on_gpu() {
    check("Qwen3-0.6B-Q8_0.gguf", 0.5, 0.01);
}

/// F16 has no activation quantization on either side, so the GPU and CPU
/// paths must agree closely token by token, including through decode steps
/// that read a KV cache split between host and device.
#[test]
fn f16_gpu_and_cpu_logits_agree() {
    let Some(path) = common::model("SmolLM2-135M-Instruct-f16.gguf") else {
        return;
    };
    let Ok(gpu) = default_gpu() else { return };
    let cpu = Model::load(&path).unwrap();
    let mut split = Model::load(&path).unwrap();
    split.offload(gpu, cpu.config.n_layer / 2, false).unwrap();
    let tokens: Vec<u32> = (0..40).map(|i| (i * 977 + 31) % 49152).collect();
    let mut a = cpu.new_session(128, 16).unwrap();
    let mut b = split.new_session(128, 16).unwrap();
    // Batched prefill, then single-token decode steps.
    let mut steps: Vec<&[u32]> = vec![&tokens[..32]];
    steps.extend(tokens[32..].chunks(1));
    for step in steps {
        let la = a.forward(step, kllm::model::Logits::Last).unwrap().to_vec();
        let lb = b.forward(step, kllm::model::Logits::Last).unwrap();
        let max = la
            .iter()
            .zip(lb)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        assert!(max < 2e-2, "CPU and GPU logits differ by {max}");
    }
}

/// Qwen3.5 exercises both GPU mixer paths: gated softmax attention and the
/// recurrent gated DeltaNet blocks. It also checks that reset clears the
/// device-resident convolution rings and recurrent matrices.
#[test]
fn qwen35_hybrid_all_on_gpu() {
    let Some(path) = std::env::var_os("KLLM_QWEN35_MODEL")
        .map(std::path::PathBuf::from)
        .or_else(|| common::model("Qwen3.5-4B-Q4_K_M.gguf"))
    else {
        return;
    };
    let Ok(gpu) = default_gpu() else { return };
    let cpu = Model::load(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(cpu.info()).unwrap();
    let tokens = tokenizer.encode("The capital of France is", true, false);
    let mut device = Model::load(&path).unwrap();
    device.offload(gpu, device.config.n_layer, true).unwrap();

    let mut cpu_session = cpu.new_session(64, 4).unwrap();
    let expected = cpu_session.forward(&tokens, Logits::All).unwrap().to_vec();
    let mut gpu_session = device.new_session(64, 4).unwrap();
    let got = gpu_session.forward(&tokens, Logits::All).unwrap().to_vec();
    assert!(got.iter().all(|x| x.is_finite()));

    let vocab = device.config.n_vocab;
    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    let agreed = expected
        .chunks_exact(vocab)
        .zip(got.chunks_exact(vocab))
        .filter(|(a, b)| argmax(a) == argmax(b))
        .count();
    assert!(
        agreed * 2 >= tokens.len(),
        "CPU/GPU argmax agreed on only {agreed}/{} positions",
        tokens.len()
    );

    gpu_session.truncate(0);
    let reset = gpu_session.forward(&tokens, Logits::All).unwrap();
    assert_eq!(got, reset, "GPU reset must clear every recurrent state");
}

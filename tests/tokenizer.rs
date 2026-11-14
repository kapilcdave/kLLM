mod common;

use kllm::{gguf::inspect_file, tokenizer::Tokenizer};

fn check(model: &str, fixture: &str) {
    let Some(path) = common::model(model) else {
        return;
    };
    let info = inspect_file(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&info).unwrap();
    let cases = common::fixture(fixture);
    let mut failures = Vec::new();
    for case in cases["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let expected: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let got = tokenizer.encode(text, true, true);
        if got != expected {
            failures.push(format!(
                "{text:?}\n   expected {expected:?}\n   got      {got:?}"
            ));
        }
        // Decoding the reference IDs must reproduce the input exactly.
        let decoded = tokenizer.decode(&expected, true);
        let bos = tokenizer
            .bos
            .filter(|_| tokenizer.add_bos)
            .map(|b| tokenizer.token_text(b).unwrap().to_string());
        let decoded = bos
            .map(|b| decoded.strip_prefix(&b).unwrap_or(&decoded).to_string())
            .unwrap_or(decoded);
        if decoded != text {
            failures.push(format!("decode {text:?} gave {decoded:?}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn smollm2_matches_llama_cpp() {
    check("SmolLM2-135M-Instruct-f16.gguf", "smollm2-135m.tokens.json");
}

#[test]
fn qwen3_matches_llama_cpp() {
    check("Qwen3-0.6B-Q4_K_M.gguf", "qwen3-0.6b-q4_k_m.tokens.json");
}

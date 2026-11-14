// Print token IDs for a file, one per line (for diffing against llama-tokenize).
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let info = kllm::gguf::inspect_file(&a[1]).unwrap();
    let tok = kllm::tokenizer::Tokenizer::from_gguf(&info).unwrap();
    let text = std::fs::read_to_string(&a[2]).unwrap();
    for id in tok.encode(&text, true, false) {
        println!("{id}");
    }
}

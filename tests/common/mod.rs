//! Helpers for tests that need real model files.
//!
//! Models are looked up in `$KLLM_TEST_MODELS` or `<repo>/models`. A test
//! whose model is absent prints a notice and passes, so `cargo test` works
//! offline; set `KLLM_REQUIRE_MODELS=1` to turn absence into failure.

#![allow(dead_code)]

use std::path::PathBuf;

pub fn model(file: &str) -> Option<PathBuf> {
    let dir = std::env::var_os("KLLM_TEST_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models"));
    let path = dir.join(file);
    if path.exists() {
        return Some(path);
    }
    if std::env::var_os("KLLM_REQUIRE_MODELS").is_some() {
        panic!("required test model {} is missing", path.display());
    }
    eprintln!("skipping: {} not found", path.display());
    None
}

pub fn fixture(name: &str) -> serde_json::Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

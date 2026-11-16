//! Mixture-of-experts correctness, on GGUF files built by the test.
//!
//! The real target (Qwen3-30B-A3B) is 18 GiB, which is a poor place to find
//! out that an expert offset is wrong. These tests pin the routed path
//! against the dense one using two invariants that hold whatever the router
//! does, so they need no reference implementation and no model download:
//!
//! 1. If every expert holds the same weights and all of them are used with
//!    normalized weights, the routed output equals the dense output exactly
//!    -- the expert weights sum to one. This checks the router, the
//!    normalization and the accumulation.
//! 2. If the router is rigged so only expert `e` can win and one expert is
//!    used, the routed output equals a dense model holding expert `e`'s
//!    weights. This checks the per-expert byte offsets, which invariant 1
//!    cannot see because all its experts are identical.
//!
//! Both models in a pair share an architecture and every non-FFN weight, so
//! any difference is the feed-forward block.

use kllm::model::{Logits, Model};

const DIM: usize = 32;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = DIM / HEADS;
const FF: usize = 48;
const VOCAB: usize = 24;
const LAYERS: usize = 2;

/// A GGUF writer just complete enough for these fixtures.
struct Gguf {
    metadata: Vec<(String, Vec<u8>)>,
    tensors: Vec<(String, Vec<u64>, Vec<u8>)>,
}

fn string_bytes(s: &str) -> Vec<u8> {
    let mut out = (s.len() as u64).to_le_bytes().to_vec();
    out.extend(s.as_bytes());
    out
}

impl Gguf {
    fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
        }
    }

    fn string(&mut self, key: &str, value: &str) -> &mut Self {
        let mut v = 8_u32.to_le_bytes().to_vec();
        v.extend(string_bytes(value));
        self.metadata.push((key.into(), v));
        self
    }

    fn u32(&mut self, key: &str, value: u32) -> &mut Self {
        let mut v = 4_u32.to_le_bytes().to_vec();
        v.extend(value.to_le_bytes());
        self.metadata.push((key.into(), v));
        self
    }

    fn f32(&mut self, key: &str, value: f32) -> &mut Self {
        let mut v = 6_u32.to_le_bytes().to_vec();
        v.extend(value.to_le_bytes());
        self.metadata.push((key.into(), v));
        self
    }

    /// An F32 tensor. `dimensions` is in GGML order (innermost first).
    fn tensor(&mut self, name: &str, dimensions: &[usize], values: &[f32]) -> &mut Self {
        assert_eq!(
            dimensions.iter().product::<usize>(),
            values.len(),
            "{name}: {dimensions:?} does not match {} values",
            values.len()
        );
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.tensors.push((
            name.into(),
            dimensions.iter().map(|d| *d as u64).collect(),
            bytes,
        ));
        self
    }

    fn write(&self, path: &std::path::Path) {
        const ALIGN: usize = 32;
        let mut head = b"GGUF".to_vec();
        head.extend(3_u32.to_le_bytes());
        head.extend((self.tensors.len() as u64).to_le_bytes());
        head.extend((self.metadata.len() as u64).to_le_bytes());
        for (key, value) in &self.metadata {
            head.extend(string_bytes(key));
            head.extend(value);
        }
        // Tensor offsets are relative to the start of the data section, which
        // begins at the first alignment boundary after the directory. The
        // directory's own size depends on the offsets only through their
        // width, which is fixed, so one pass suffices.
        let mut directory = Vec::new();
        let mut offset = 0_u64;
        for (name, dimensions, bytes) in &self.tensors {
            directory.extend(string_bytes(name));
            directory.extend((dimensions.len() as u32).to_le_bytes());
            for d in dimensions {
                directory.extend(d.to_le_bytes());
            }
            directory.extend(0_u32.to_le_bytes()); // F32
            directory.extend(offset.to_le_bytes());
            let size = bytes.len() as u64;
            offset += size.next_multiple_of(ALIGN as u64);
        }
        let mut out = head;
        out.extend(directory);
        while !out.len().is_multiple_of(ALIGN) {
            out.push(0);
        }
        for (_, _, bytes) in &self.tensors {
            out.extend(bytes);
            while !out.len().is_multiple_of(ALIGN) {
                out.push(0);
            }
        }
        std::fs::write(path, out).unwrap();
    }
}

/// Deterministic pseudo-random weights, stable across runs.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as u32;
            (bits as f32 / 8388608.0) - 1.0
        })
        .collect()
}

/// Everything except the feed-forward block. `arch` is shared by both models
/// of a pair so attention is identical.
fn common(g: &mut Gguf, arch: &str, layers: usize) {
    g.string("general.architecture", arch)
        .u32(&format!("{arch}.block_count"), layers as u32)
        .u32(&format!("{arch}.embedding_length"), DIM as u32)
        .u32(&format!("{arch}.attention.head_count"), HEADS as u32)
        .u32(&format!("{arch}.attention.head_count_kv"), KV_HEADS as u32)
        .u32(&format!("{arch}.context_length"), 64)
        .u32(&format!("{arch}.vocab_size"), VOCAB as u32)
        .f32(&format!("{arch}.attention.layer_norm_rms_epsilon"), 1e-5)
        .f32(&format!("{arch}.rope.freq_base"), 10000.0);
    g.tensor("token_embd.weight", &[DIM, VOCAB], &values(DIM * VOCAB, 1));
    g.tensor("output_norm.weight", &[DIM], &values(DIM, 2));
    g.tensor("output.weight", &[DIM, VOCAB], &values(DIM * VOCAB, 3));
    for l in 0..layers {
        let kv = KV_HEADS * HEAD_DIM;
        g.tensor(
            &format!("blk.{l}.attn_norm.weight"),
            &[DIM],
            &values(DIM, 10 + l as u64),
        );
        g.tensor(
            &format!("blk.{l}.attn_q.weight"),
            &[DIM, DIM],
            &values(DIM * DIM, 20 + l as u64),
        );
        g.tensor(
            &format!("blk.{l}.attn_k.weight"),
            &[DIM, kv],
            &values(DIM * kv, 30 + l as u64),
        );
        g.tensor(
            &format!("blk.{l}.attn_v.weight"),
            &[DIM, kv],
            &values(DIM * kv, 40 + l as u64),
        );
        g.tensor(
            &format!("blk.{l}.attn_output.weight"),
            &[DIM, DIM],
            &values(DIM * DIM, 50 + l as u64),
        );
        g.tensor(
            &format!("blk.{l}.ffn_norm.weight"),
            &[DIM],
            &values(DIM, 60 + l as u64),
        );
    }
}

/// Dense feed-forward weights for layer `l`, seeded by `expert`.
fn dense_ffn(g: &mut Gguf, l: usize, expert: u64) {
    let seed = 100 + expert * 7 + l as u64;
    g.tensor(
        &format!("blk.{l}.ffn_gate.weight"),
        &[DIM, FF],
        &values(DIM * FF, seed),
    );
    g.tensor(
        &format!("blk.{l}.ffn_up.weight"),
        &[DIM, FF],
        &values(DIM * FF, seed + 1000),
    );
    g.tensor(
        &format!("blk.{l}.ffn_down.weight"),
        &[FF, DIM],
        &values(FF * DIM, seed + 2000),
    );
}

const ARCH: &str = "qwen3moe";

fn dense_model(path: &std::path::Path, expert: u64, layers: usize) {
    let mut g = Gguf::new();
    common(&mut g, ARCH, layers);
    g.u32(&format!("{ARCH}.feed_forward_length"), FF as u32);
    for l in 0..layers {
        dense_ffn(&mut g, l, expert);
    }
    g.write(path);
}

/// A routed model. `experts` seeds each expert's weights; `router` gives the
/// router rows, so a test can decide which expert wins.
fn routed_model(
    path: &std::path::Path,
    experts: &[u64],
    used: usize,
    layers: usize,
    router: &dyn Fn(usize) -> f32,
) {
    let n = experts.len();
    let mut g = Gguf::new();
    common(&mut g, ARCH, layers);
    g.u32(&format!("{ARCH}.feed_forward_length"), FF as u32)
        .u32(&format!("{ARCH}.expert_count"), n as u32)
        .u32(&format!("{ARCH}.expert_used_count"), used as u32)
        .u32(&format!("{ARCH}.expert_feed_forward_length"), FF as u32);
    for l in 0..layers {
        // Router: [n_embd, n_expert], so expert e's row is contiguous.
        let mut rows = Vec::with_capacity(DIM * n);
        for e in 0..n {
            rows.extend(std::iter::repeat_n(router(e), DIM));
        }
        g.tensor(&format!("blk.{l}.ffn_gate_inp.weight"), &[DIM, n], &rows);
        // Expert tensors are the dense ones stacked along the last axis.
        for (name, dims, pick) in [
            ("ffn_gate_exps", [DIM, FF, n], 0_u64),
            ("ffn_up_exps", [DIM, FF, n], 1000),
            ("ffn_down_exps", [FF, DIM, n], 2000),
        ] {
            let mut stacked = Vec::with_capacity(DIM * FF * n);
            for e in experts {
                let seed = 100 + e * 7 + l as u64 + pick;
                stacked.extend(values(DIM * FF, seed));
            }
            g.tensor(&format!("blk.{l}.{name}.weight"), &dims, &stacked);
        }
    }
    g.write(path);
}

fn logits_for(path: &std::path::Path, tokens: &[u32]) -> Vec<f32> {
    let model = Model::load(path).unwrap();
    let mut session = model.new_session(32, 8).unwrap();
    session.forward(tokens, Logits::Last).unwrap().to_vec()
}

fn close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tolerance * x.abs().max(1.0),
            "logit {i}: {x} vs {y}"
        );
    }
}

/// Invariant 1: identical experts, all of them used, weights normalized ->
/// the same answer as the dense model they were copied from.
#[test]
fn all_experts_identical_matches_dense() {
    let dir = std::env::temp_dir().join("kllm-moe-identical");
    std::fs::create_dir_all(&dir).unwrap();
    let (dense, routed) = (dir.join("dense.gguf"), dir.join("routed.gguf"));
    dense_model(&dense, 0, LAYERS);
    // Four copies of expert 0, all used. Router values are deliberately
    // equal, so selection order cannot matter either.
    routed_model(&routed, &[0, 0, 0, 0], 4, LAYERS, &|_| 0.25);
    let tokens = [1_u32, 5, 9, 3];
    close(
        &logits_for(&dense, &tokens),
        &logits_for(&routed, &tokens),
        1e-5,
    );
}

/// Invariant 2: with one expert used and four *distinct* experts, whatever
/// the router picks the output must equal that expert's dense model -- and
/// no other's. Reading the wrong slice matches nobody, which is what pins
/// the per-expert byte offsets.
///
/// The router is not rigged to choose a particular expert: its rows are
/// constant across `n_embd`, so a logit is the row value times the sum of
/// the normalized input and flips sign with it. Asserting a specific winner
/// would be asserting the test's own premise. Instead several router
/// configurations are tried and the winners are collected, so more than one
/// offset gets exercised.
///
/// One layer, deliberately: routing is per layer, so a two-layer model picks
/// its experts independently per block and matches no single dense model.
#[test]
fn single_routed_expert_matches_its_dense_equivalent() {
    let dir = std::env::temp_dir().join("kllm-moe-offsets");
    std::fs::create_dir_all(&dir).unwrap();
    let tokens = [2_u32, 7, 4];

    // One dense model per expert, to compare against.
    let dense: Vec<Vec<f32>> = (0..4_u64)
        .map(|e| {
            let path = dir.join(format!("dense{e}.gguf"));
            dense_model(&path, e, 1);
            logits_for(&path, &tokens)
        })
        .collect();
    // The experts must actually differ, or this proves nothing.
    for e in 1..4 {
        assert!(
            dense[e]
                .iter()
                .zip(&dense[0])
                .any(|(a, b)| (a - b).abs() > 1e-3),
            "expert {e}'s dense model is indistinguishable from expert 0's"
        );
    }

    let matches_one = |got: &[f32]| -> Option<usize> {
        let hits: Vec<usize> = (0..4)
            .filter(|p| {
                got.iter()
                    .zip(&dense[*p])
                    .all(|(x, y)| (x - y).abs() <= 1e-5 * x.abs().max(1.0))
            })
            .collect();
        match hits.len() {
            1 => Some(hits[0]),
            _ => None,
        }
    };

    let routed = dir.join("routed.gguf");
    let mut seen = std::collections::BTreeSet::new();
    for trial in 0..6_usize {
        // Distinct, non-degenerate router rows, varied per trial.
        routed_model(&routed, &[0, 1, 2, 3], 1, 1, &|e| {
            ((e + 1) as f32) * if (trial + e) % 2 == 0 { 1.0 } else { -1.0 } + trial as f32 * 0.5
        });
        let got = logits_for(&routed, &tokens);
        let which = matches_one(&got).unwrap_or_else(|| {
            panic!(
                "trial {trial}: routed logits match no single expert's dense model, \
                 so the expert slice is wrong: got {:?}, expert 0 {:?}",
                &got[..4.min(got.len())],
                &dense[0][..4.min(dense[0].len())]
            )
        });
        seen.insert(which);
    }
    assert!(
        seen.len() >= 2,
        "only expert(s) {seen:?} were ever routed to, so the other offsets are untested"
    );
}

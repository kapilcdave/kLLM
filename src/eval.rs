//! Model evaluation shared by the CLI and the test-suite.
//!
//! Two measures, because they answer different questions:
//!
//! * [`perplexity`] scores a model against a corpus. It says whether a model
//!   predicts text well, and it is the right way to compare a quantized build
//!   against llama.cpp, because averaging over a corpus washes out the
//!   summation-order noise that 8-bit activations introduce.
//! * [`divergence`] scores a model against *another model* token by token. It
//!   says how much a quantization changed the distribution the full-precision
//!   model would have produced, which perplexity can hide: a quantization can
//!   keep perplexity and still reorder the top of the distribution.

use crate::model::{Logits, Model, ModelError};
use crate::sampler::log_prob;
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, Copy)]
pub struct Perplexity {
    pub ppl: f64,
    /// Standard error of the PPL estimate (delta method, as llama.cpp).
    pub stderr: f64,
    pub tokens_scored: usize,
    pub chunks: usize,
}

/// Perplexity with llama.cpp's `llama-perplexity` protocol: split `tokens`
/// into `n_ctx`-token chunks, evaluate each from an empty cache with its first
/// token replaced by BOS (if the model uses one), and score only the second
/// half of every chunk. `progress` receives `(chunk, running_ppl)`.
pub fn perplexity(
    model: &Model,
    tokenizer: &Tokenizer,
    tokens: &[u32],
    n_ctx: usize,
    max_chunks: Option<usize>,
    mut progress: impl FnMut(usize, f64),
) -> Result<Perplexity, ModelError> {
    let available = tokens.len() / n_ctx;
    let chunks = max_chunks.map_or(available, |m| m.min(available));
    if chunks == 0 || n_ctx < 4 {
        return Err(ModelError::Shape(format!(
            "need at least one chunk of {n_ctx} tokens, have {}",
            tokens.len()
        )));
    }
    let first = n_ctx / 2;
    let vocab = model.config.n_vocab;
    let mut session = model.new_session(n_ctx, n_ctx)?;
    let mut nlls = Vec::with_capacity(chunks * (n_ctx - 1 - first));
    for chunk in 0..chunks {
        let mut batch = tokens[chunk * n_ctx..(chunk + 1) * n_ctx].to_vec();
        if tokenizer.add_bos
            && let Some(bos) = tokenizer.bos
        {
            batch[0] = bos;
        }
        session.truncate(0);
        let logits = session.forward(&batch, Logits::All)?;
        for j in first..n_ctx - 1 {
            nlls.push(-log_prob(&logits[j * vocab..(j + 1) * vocab], batch[j + 1]));
        }
        let mean = nlls.iter().sum::<f64>() / nlls.len() as f64;
        progress(chunk + 1, mean.exp());
    }
    let n = nlls.len() as f64;
    let mean = nlls.iter().sum::<f64>() / n;
    let mean_sq = nlls.iter().map(|x| x * x).sum::<f64>() / n;
    let variance = ((mean_sq - mean * mean) / (n - 1.0)).max(0.0);
    let ppl = mean.exp();
    Ok(Perplexity {
        ppl,
        stderr: variance.sqrt() * ppl,
        tokens_scored: nlls.len(),
        chunks,
    })
}

/// How far a model's predictions moved from a reference model's, measured
/// over the same tokens. All divergences are in nats.
#[derive(Debug, Clone)]
pub struct Divergence {
    pub tokens_scored: usize,
    pub chunks: usize,
    /// Mean KL(reference ‖ test): the information lost by using `test` where
    /// the reference was right. This is the headline quantization-damage
    /// number and matches llama.cpp's `--kl-divergence`.
    pub kl_mean: f64,
    /// Standard error of `kl_mean`.
    pub kl_stderr: f64,
    pub kl_median: f64,
    pub kl_p99: f64,
    pub kl_max: f64,
    /// Fraction of positions where both models' most likely token agrees.
    pub top1_agree: f64,
    /// Fraction where the reference's most likely token is in the test
    /// model's top five.
    pub top5_agree: f64,
    /// Perplexity of each model over the scored positions.
    pub reference_ppl: f64,
    pub test_ppl: f64,
}

/// Softmax in f64 into `out`, returning the index of the largest logit.
fn softmax(logits: &[f32], out: &mut [f64]) -> usize {
    let mut best = 0;
    let mut max = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > max {
            max = v;
            best = i;
        }
    }
    let mut sum = 0.0;
    for (o, &v) in out.iter_mut().zip(logits) {
        *o = (v as f64 - max as f64).exp();
        sum += *o;
    }
    let inv = 1.0 / sum;
    for o in out.iter_mut() {
        *o *= inv;
    }
    best
}

/// Rank of `target` in `probabilities`, counting from zero, capped at `cap`
/// (so the whole vocabulary is never sorted).
fn rank_within(probabilities: &[f64], target: usize, cap: usize) -> usize {
    let p = probabilities[target];
    let mut above = 0;
    for (i, &q) in probabilities.iter().enumerate() {
        if q > p || (q == p && i < target) {
            above += 1;
            if above >= cap {
                return cap;
            }
        }
    }
    above
}

/// Compare `test` against `reference` over `tokens`, using the same chunking
/// protocol as [`perplexity`] so the two numbers are directly comparable.
///
/// Both models must share a vocabulary; the caller's `tokenizer` is the
/// reference model's. `progress` receives `(chunk, running_mean_kl)`.
pub fn divergence(
    reference: &Model,
    test: &Model,
    tokenizer: &Tokenizer,
    tokens: &[u32],
    n_ctx: usize,
    max_chunks: Option<usize>,
    mut progress: impl FnMut(usize, f64),
) -> Result<Divergence, ModelError> {
    let vocab = reference.config.n_vocab;
    if test.config.n_vocab != vocab {
        return Err(ModelError::Shape(format!(
            "vocabularies differ: reference has {vocab}, test has {}",
            test.config.n_vocab
        )));
    }
    let available = tokens.len() / n_ctx;
    let chunks = max_chunks.map_or(available, |m| m.min(available));
    if chunks == 0 || n_ctx < 4 {
        return Err(ModelError::Shape(format!(
            "need at least one chunk of {n_ctx} tokens, have {}",
            tokens.len()
        )));
    }
    let first = n_ctx / 2;
    let mut reference_session = reference.new_session(n_ctx, n_ctx)?;
    let mut test_session = test.new_session(n_ctx, n_ctx)?;
    let mut kls: Vec<f64> = Vec::with_capacity(chunks * (n_ctx - 1 - first));
    let (mut top1, mut top5) = (0_usize, 0_usize);
    let (mut reference_nll, mut test_nll) = (0.0_f64, 0.0_f64);
    let (mut p, mut q) = (vec![0.0_f64; vocab], vec![0.0_f64; vocab]);

    for chunk in 0..chunks {
        let mut batch = tokens[chunk * n_ctx..(chunk + 1) * n_ctx].to_vec();
        if tokenizer.add_bos
            && let Some(bos) = tokenizer.bos
        {
            batch[0] = bos;
        }
        reference_session.truncate(0);
        test_session.truncate(0);
        // Score one chunk at a time rather than holding two sets of
        // all-position logits for the whole corpus.
        let reference_logits = reference_session.forward(&batch, Logits::All)?.to_vec();
        let test_logits = test_session.forward(&batch, Logits::All)?;
        for j in first..n_ctx - 1 {
            let a = &reference_logits[j * vocab..(j + 1) * vocab];
            let b = &test_logits[j * vocab..(j + 1) * vocab];
            let best_a = softmax(a, &mut p);
            let best_b = softmax(b, &mut q);
            // KL(p ‖ q) = sum p * (ln p - ln q), skipping p == 0 terms.
            let mut kl = 0.0;
            for (&pi, &qi) in p.iter().zip(&q) {
                if pi > 0.0 {
                    kl += pi * (pi.ln() - qi.max(f64::MIN_POSITIVE).ln());
                }
            }
            kls.push(kl.max(0.0));
            top1 += usize::from(best_a == best_b);
            top5 += usize::from(rank_within(&q, best_a, 5) < 5);
            let target = batch[j + 1];
            reference_nll -= log_prob(a, target);
            test_nll -= log_prob(b, target);
        }
        let mean = kls.iter().sum::<f64>() / kls.len() as f64;
        progress(chunk + 1, mean);
    }

    let n = kls.len() as f64;
    let mean = kls.iter().sum::<f64>() / n;
    let mean_sq = kls.iter().map(|x| x * x).sum::<f64>() / n;
    let variance = ((mean_sq - mean * mean) / (n - 1.0).max(1.0)).max(0.0);
    let mut sorted = kls.clone();
    sorted.sort_unstable_by(f64::total_cmp);
    let at = |f: f64| sorted[((sorted.len() - 1) as f64 * f).round() as usize];
    Ok(Divergence {
        tokens_scored: kls.len(),
        chunks,
        kl_mean: mean,
        kl_stderr: variance.sqrt(),
        kl_median: at(0.5),
        kl_p99: at(0.99),
        kl_max: *sorted.last().unwrap(),
        top1_agree: top1 as f64 / n,
        top5_agree: top5 as f64 / n,
        reference_ppl: (reference_nll / n).exp(),
        test_ppl: (test_nll / n).exp(),
    })
}

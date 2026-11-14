//! Model evaluation shared by the CLI and the test-suite.

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
    let mut session = model.new_session(n_ctx, n_ctx);
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

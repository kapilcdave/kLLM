//! Token sampling: greedy or temperature with top-k / top-p / min-p and a
//! repetition penalty. Deterministic for a given seed.

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// 0 selects greedy decoding.
    pub temperature: f32,
    /// 0 disables top-k.
    pub top_k: usize,
    /// 1.0 disables nucleus sampling.
    pub top_p: f32,
    /// 0.0 disables min-p.
    pub min_p: f32,
    /// 1.0 disables the repetition penalty.
    pub repeat_penalty: f32,
    /// How many recent tokens the penalty considers.
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: 0x6b4c_4c4d,
        }
    }
}

pub struct Sampler {
    config: SamplerConfig,
    state: u64,
    candidates: Vec<(u32, f32)>,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        let state = splitmix(config.seed);
        Self {
            config,
            state,
            candidates: Vec::new(),
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.config.temperature <= 0.0
    }

    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let bits = self.state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40;
        bits as f32 / (1u64 << 24) as f32
    }

    /// Choose the next token. `history` is the token sequence so far.
    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> u32 {
        let c = &self.config;
        let penalised = c.repeat_penalty != 1.0 && !history.is_empty();
        let recent = &history[history.len().saturating_sub(c.repeat_last_n)..];
        let logit = |id: usize, value: f32| -> f32 {
            if penalised && recent.contains(&(id as u32)) {
                if value > 0.0 {
                    value / c.repeat_penalty
                } else {
                    value * c.repeat_penalty
                }
            } else {
                value
            }
        };

        if self.is_greedy() {
            let mut best = (0, f32::NEG_INFINITY);
            for (id, &value) in logits.iter().enumerate() {
                let v = logit(id, value);
                if v > best.1 {
                    best = (id, v);
                }
            }
            return best.0 as u32;
        }

        self.candidates.clear();
        self.candidates.extend(
            logits
                .iter()
                .enumerate()
                .map(|(id, &v)| (id as u32, logit(id, v) / c.temperature)),
        );
        let candidates = &mut self.candidates;
        let k = if c.top_k == 0 {
            candidates.len()
        } else {
            c.top_k.min(candidates.len())
        };
        if k < candidates.len() {
            candidates.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
            candidates.truncate(k);
        }
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // Softmax over the survivors.
        let max = candidates[0].1;
        let mut sum = 0.0_f32;
        for cand in candidates.iter_mut() {
            cand.1 = (cand.1 - max).exp();
            sum += cand.1;
        }
        for cand in candidates.iter_mut() {
            cand.1 /= sum;
        }
        if c.min_p > 0.0 {
            let floor = candidates[0].1 * c.min_p;
            let keep = candidates
                .iter()
                .take_while(|cand| cand.1 >= floor)
                .count()
                .max(1);
            candidates.truncate(keep);
        }
        if c.top_p < 1.0 {
            let mut cumulative = 0.0;
            let mut keep = candidates.len();
            for (i, cand) in candidates.iter().enumerate() {
                cumulative += cand.1;
                if cumulative >= c.top_p {
                    keep = i + 1;
                    break;
                }
            }
            candidates.truncate(keep);
        }
        let total: f32 = candidates.iter().map(|cand| cand.1).sum();
        let mut target = self.next_f32() * total;
        let candidates = &self.candidates;
        for cand in candidates {
            target -= cand.1;
            if target <= 0.0 {
                return cand.0;
            }
        }
        candidates.last().map(|cand| cand.0).unwrap_or(0)
    }
}

fn splitmix(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (z ^ (z >> 31)) | 1
}

/// Log-softmax of one logit row at `target`, computed stably in f64.
pub fn log_prob(logits: &[f32], target: u32) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&v| (v as f64 - max).exp()).sum();
    logits[target as usize] as f64 - max - sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let mut s = Sampler::new(SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        });
        assert_eq!(s.sample(&[0.1, 2.0, -1.0, 1.9], &[]), 1);
    }

    #[test]
    fn repetition_penalty_changes_greedy_choice() {
        let mut s = Sampler::new(SamplerConfig {
            temperature: 0.0,
            repeat_penalty: 2.0,
            ..Default::default()
        });
        assert_eq!(s.sample(&[0.1, 2.0, -1.0, 1.9], &[1]), 3);
    }

    #[test]
    fn top_k_one_is_greedy_and_seed_is_deterministic() {
        let logits: Vec<f32> = (0..100).map(|i| ((i * 7919) % 101) as f32 / 10.0).collect();
        let mut s = Sampler::new(SamplerConfig {
            top_k: 1,
            ..Default::default()
        });
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        assert_eq!(s.sample(&logits, &[]), argmax);

        let run = |seed| {
            let mut s = Sampler::new(SamplerConfig {
                seed,
                temperature: 1.5,
                min_p: 0.0,
                ..Default::default()
            });
            (0..20).map(|_| s.sample(&logits, &[])).collect::<Vec<_>>()
        };
        assert_eq!(run(1), run(1));
        assert_ne!(run(1), run(2));
    }

    #[test]
    fn log_prob_is_normalised() {
        let logits = [1.0, 2.0, 3.0];
        let total: f64 = (0..3).map(|t| log_prob(&logits, t).exp()).sum();
        assert!((total - 1.0).abs() < 1e-12);
    }
}

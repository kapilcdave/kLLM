// Attention primitives in isolation, in GMAC/s (one multiply-add each).
//
// The point is to separate the kernels from the loop around them: if the
// kernels run fast here but attention is slow in a real forward pass, the
// cost is in the traffic or the bookkeeping, not the arithmetic.
use kllm::kernels;
use std::time::Instant;

fn main() {
    let head_dim: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(64);
    // A KV cache slab big enough to not sit entirely in L1, as in a real run.
    let positions = 512;
    let kv: Vec<u16> = (0..positions * head_dim)
        .map(|i| kllm::quant::f32_to_f16(((i % 97) as f32 - 48.0) / 64.0))
        .collect();
    let queries: Vec<f32> = (0..16 * head_dim)
        .map(|i| ((i % 89) as f32 - 44.0) / 64.0)
        .collect();
    let q = |j: usize| &queries[j * head_dim..(j + 1) * head_dim];
    let key = |t: usize| &kv[t * head_dim..(t + 1) * head_dim];
    println!("head_dim {head_dim}, {positions} positions");

    let reps = 4000;
    // One query per key, the old shape.
    let mut sink = 0.0_f32;
    let t0 = Instant::now();
    for _ in 0..reps {
        for t in 0..positions {
            sink += kernels::dot_f16_f32(key(t), q(0));
        }
    }
    let one = t0.elapsed().as_secs_f64() / reps as f64;
    println!(
        "dot_f16_f32        {:7.2} GMAC/s",
        (positions * head_dim) as f64 / one / 1e9
    );

    // N queries per key.
    macro_rules! many {
        ($n:literal) => {{
            let t0 = Instant::now();
            for _ in 0..reps {
                for t in 0..positions {
                    let got = kernels::dot_f16_f32_many(
                        key(t),
                        std::array::from_fn::<_, $n, _>(|j| q(j)),
                    );
                    sink += got[0];
                }
            }
            let s = t0.elapsed().as_secs_f64() / reps as f64;
            println!(
                "dot_f16_f32_many<{}>  {:7.2} GMAC/s",
                $n,
                (positions * head_dim * $n) as f64 / s / 1e9
            );
        }};
    }
    many!(1);
    many!(2);
    many!(4);
    many!(8);

    // The value pass.
    let mut out = vec![0.0_f32; head_dim];
    let t0 = Instant::now();
    for _ in 0..reps {
        for t in 0..positions {
            kernels::axpy_f16(&mut out, 0.5, key(t));
        }
    }
    let s = t0.elapsed().as_secs_f64() / reps as f64;
    println!(
        "axpy_f16           {:7.2} GMAC/s",
        (positions * head_dim) as f64 / s / 1e9
    );

    // Softmax-shaped work over the same number of scores.
    let mut scores = vec![0.5_f32; positions];
    let t0 = Instant::now();
    for _ in 0..reps {
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum = kernels::exp_shifted_sum(&mut scores, max);
        let inv = 1.0 / sum;
        for s in scores.iter_mut() {
            *s *= inv;
        }
        sink += scores[0];
    }
    let s = t0.elapsed().as_secs_f64() / reps as f64;
    println!(
        "softmax row        {:7.2} ns for {positions} scores ({:.2} ns/score)",
        s * 1e9,
        s * 1e9 / positions as f64
    );
    std::hint::black_box(sink);
    std::hint::black_box(out);
}

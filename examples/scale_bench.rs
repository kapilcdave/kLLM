// How much parallel speedup is actually available, measured through kLLM's
// own pool on work with no memory traffic at all.
//
// This is the ceiling any parallel phase can reach on this machine right now.
// Comparing a phase's speedup against this number separates "the pool or the
// phase is at fault" from "the machine has no idle cores to give".
use std::time::Instant;

fn main() {
    let max: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(4);
    // Per-task work: a dependent FMA chain, so it cannot be vectorised away
    // and touches no memory.
    let spin = |seed: f64| {
        let mut x = seed;
        for _ in 0..200_000 {
            x = x * 1.0000001 + 1e-9;
        }
        x
    };
    let tasks = 64;
    let mut baseline = 0.0;
    for threads in 1..=max {
        // A fresh process per thread count would be cleaner, but the pool is
        // a process-wide singleton; run the largest count last so spin-up is
        // not charged to it.
        let pool = kllm::pool::Pool::new(threads);
        let out: Vec<std::sync::atomic::AtomicU64> = (0..tasks)
            .map(|_| std::sync::atomic::AtomicU64::new(0))
            .collect();
        pool.for_each(tasks, |i| {
            out[i].store(
                spin(i as f64).to_bits(),
                std::sync::atomic::Ordering::Relaxed,
            );
        });
        let t = Instant::now();
        let reps = 5;
        for _ in 0..reps {
            pool.for_each(tasks, |i| {
                out[i].store(
                    spin(i as f64).to_bits(),
                    std::sync::atomic::Ordering::Relaxed,
                );
            });
        }
        let s = t.elapsed().as_secs_f64() / reps as f64;
        if threads == 1 {
            baseline = s;
        }
        println!(
            "{threads} thread(s): {:8.2} ms  speedup {:5.2}x",
            s * 1e3,
            baseline / s
        );
    }

    // Dispatch latency: what a parallel region costs before any work is
    // done. Decode issues a few hundred of these per token, so this is a
    // floor on per-token time.
    println!();
    for threads in 1..=max {
        let pool = kllm::pool::Pool::new(threads);
        let counter = std::sync::atomic::AtomicUsize::new(0);
        let reps = 20_000;
        // Warm the workers so they are spinning, not parked.
        for _ in 0..1000 {
            pool.for_each(threads, |_| {
                counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            });
        }
        let t = Instant::now();
        for _ in 0..reps {
            pool.for_each(threads, |_| {
                counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            });
        }
        println!(
            "{threads} thread(s): {:6.2} us per empty for_each ({} per token would cost {:.2} ms)",
            t.elapsed().as_secs_f64() / reps as f64 * 1e6,
            150,
            t.elapsed().as_secs_f64() / reps as f64 * 150.0 * 1e3,
        );
        std::hint::black_box(counter.load(std::sync::atomic::Ordering::Relaxed));
    }
}

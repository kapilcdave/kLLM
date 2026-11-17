//! A low-latency fork-join pool for the decode loop.
//!
//! Batch-one decoding issues a few hundred small parallel operations per
//! token. A general-purpose pool parks idle workers, and waking them costs
//! tens of microseconds per operation, which for small models exceeds the
//! work itself. Here workers spin (with `spin_loop` hints) for a short budget
//! after each operation, so back-to-back operations start within a
//! microsecond, and only park once the engine has gone quiet.
//!
//! The calling thread always takes part, so `threads = 1` runs inline with no
//! synchronisation at all.

use std::{
    cell::UnsafeCell,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle, Thread},
    time::{Duration, Instant},
};

/// How long an idle worker spins before parking.
const SPIN: Duration = Duration::from_micros(500);

type Job = dyn Fn(usize) + Sync;

struct Shared {
    /// Incremented once per `run`; workers wait for it to change.
    generation: AtomicUsize,
    /// Workers still running the current job.
    pending: AtomicUsize,
    /// The current job, valid while `pending > 0`.
    job: UnsafeCell<Option<*const Job>>,
    stop: AtomicBool,
}

// SAFETY: `job` is written only by the thread inside `run` while no worker
// is reading it (pending == 0), and read by workers only between the
// generation bump and their own decrement of `pending`.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<(JoinHandle<()>, Thread)>,
    /// Serialises concurrent `run` calls from different threads.
    lock: std::sync::Mutex<()>,
}

impl Pool {
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            generation: AtomicUsize::new(0),
            pending: AtomicUsize::new(0),
            job: UnsafeCell::new(None),
            stop: AtomicBool::new(false),
        });
        let workers = (1..threads)
            .map(|index| {
                let shared = Arc::clone(&shared);
                let handle = thread::Builder::new()
                    .name(format!("kllm-worker-{index}"))
                    .spawn(move || worker(&shared, index))
                    .expect("failed to spawn worker thread");
                let thread = handle.thread().clone();
                (handle, thread)
            })
            .collect();
        Self {
            shared,
            workers,
            lock: std::sync::Mutex::new(()),
        }
    }

    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Call `f(i)` once for every `i` in `0..threads()`, in parallel, and
    /// return when all calls have finished. The caller runs `f(0)`.
    pub fn run(&self, f: &(dyn Fn(usize) + Sync)) {
        if self.workers.is_empty() {
            f(0);
            return;
        }
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let shared = &*self.shared;
        // SAFETY: no worker reads `job` while pending == 0 (enforced by the
        // wait at the end of the previous run). The lifetime is erased; it is
        // sound because this function does not return until every worker
        // has finished calling it.
        unsafe {
            let erased = std::mem::transmute::<*const (dyn Fn(usize) + Sync + '_), *const Job>(f);
            *shared.job.get() = Some(erased);
        }
        shared.pending.store(self.workers.len(), Ordering::Release);
        shared.generation.fetch_add(1, Ordering::AcqRel);
        for (_, thread) in &self.workers {
            thread.unpark(); // cheap when the worker is spinning
        }
        f(0);
        while shared.pending.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
    }

    /// Run `count` tasks, claimed dynamically by every thread.
    pub fn for_each(&self, count: usize, task: impl Fn(usize) + Sync) {
        self.for_each_on(count, |_, i| task(i));
    }

    /// As [`Self::for_each`], but the task also learns which worker is
    /// running it, in `0..threads()`.
    ///
    /// That is what lets a task take a slice of scratch without the
    /// scratch being sized for every task: only `threads()` of them run at
    /// once, so a per-worker slot is enough. Attention's score rows are
    /// sized that way, which is the difference between a few hundred
    /// megabytes and a few gigabytes at a long context.
    pub fn for_each_on(&self, count: usize, task: impl Fn(usize, usize) + Sync) {
        match count {
            0 => {}
            1 => task(0, 0),
            _ if self.workers.is_empty() => (0..count).for_each(|i| task(0, i)),
            _ => {
                let next = AtomicUsize::new(0);
                self.run(&|worker| {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= count {
                            break;
                        }
                        task(worker, i);
                    }
                });
            }
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.generation.fetch_add(1, Ordering::AcqRel);
        for (handle, thread) in self.workers.drain(..) {
            thread.unpark();
            let _ = handle.join();
        }
    }
}

fn worker(shared: &Shared, index: usize) {
    // Start from 0, not the current value: a `run` issued before this thread
    // got here has already bumped the generation and must not be missed.
    let mut seen = 0;
    loop {
        // Wait for the next job: spin first, then park.
        let spin_start = Instant::now();
        let mut spins = 0_u32;
        loop {
            let current = shared.generation.load(Ordering::Acquire);
            if current != seen {
                seen = current;
                break;
            }
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(1024) && spin_start.elapsed() > SPIN {
                thread::park_timeout(Duration::from_millis(10));
            } else {
                std::hint::spin_loop();
            }
        }
        if shared.stop.load(Ordering::Acquire) {
            return;
        }
        // SAFETY: `run` stored a valid job before bumping the generation and
        // keeps it alive until `pending` reaches zero.
        let job = unsafe { (*shared.job.get()).expect("job published before generation bump") };
        unsafe { (*job)(index) };
        shared.pending.fetch_sub(1, Ordering::AcqRel);
    }
}

static GLOBAL: OnceLock<Pool> = OnceLock::new();

/// Configure the global pool. Only the first call has an effect; returns the
/// thread count actually in use.
pub fn init(threads: usize) -> usize {
    GLOBAL.get_or_init(|| Pool::new(threads)).threads()
}

/// The global pool (default size: physical cores, estimated as half the
/// logical CPUs).
pub fn global() -> &'static Pool {
    GLOBAL.get_or_init(|| Pool::new(default_threads()))
}

pub fn default_threads() -> usize {
    let logical = thread::available_parallelism().map_or(4, |n| n.get());
    (logical / 2).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_index_runs_once_per_call() {
        let pool = Pool::new(4);
        for _ in 0..1000 {
            let hits: Vec<AtomicUsize> = (0..4).map(|_| AtomicUsize::new(0)).collect();
            pool.run(&|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
        }
    }

    #[test]
    fn for_each_covers_all_tasks() {
        let pool = Pool::new(3);
        let sum = AtomicUsize::new(0);
        pool.for_each(10_000, |i| {
            sum.fetch_add(i, Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), 10_000 * 9_999 / 2);
    }

    #[test]
    fn single_thread_runs_inline() {
        let pool = Pool::new(1);
        let id = thread::current().id();
        pool.run(&|_| assert_eq!(thread::current().id(), id));
    }

    #[test]
    fn workers_wake_after_parking() {
        let pool = Pool::new(2);
        thread::sleep(SPIN * 4);
        let count = AtomicUsize::new(0);
        pool.run(&|_| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::Relaxed), 2);
    }
}

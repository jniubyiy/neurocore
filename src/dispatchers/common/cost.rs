use std::time::Instant;
use std::hint::black_box;

#[derive(Debug, Clone)]
pub struct CostModel {
    pub fmadd_ns: f64,
    pub task_overhead_ns: f64,
    pub barrier_overhead_ns: f64,
    pub num_cores: usize,
}

impl CostModel {
    pub fn calibrate() -> Self {
        let num_cores = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        let fmadd_ns = {
            let iterations = 10_000_000u64;
            let start = Instant::now();
            let mut sum = 0.0f64;
            for _ in 0..iterations {
                sum += black_box(1.0f64) * black_box(2.0f64);
            }
            black_box(sum);
            (start.elapsed().as_secs_f64() * 1e9) / (iterations as f64)
        };
        let task_overhead_ns = {
            let iterations = 100_000;
            let start = Instant::now();
            for _ in 0..iterations {
                black_box((|| {})());
            }
            (start.elapsed().as_secs_f64() * 1e9) / (iterations as f64)
        };
        let barrier_overhead_ns = 1000.0;
        CostModel { fmadd_ns, task_overhead_ns, barrier_overhead_ns, num_cores }
    }

    pub fn min_mac_per_task(&self) -> f64 {
        self.task_overhead_ns / self.fmadd_ns
    }
}
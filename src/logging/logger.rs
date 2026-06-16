use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

static LOG_LEVEL: AtomicUsize = AtomicUsize::new(1); // 0=off, 1=info, 2=debug, 3=trace

pub struct Logger {
    start: Instant,
}

impl Logger {
    pub fn new() -> Self {
        Logger { start: Instant::now() }
    }

    pub fn set_level(level: usize) {
        LOG_LEVEL.store(level.min(3), Ordering::Relaxed);
    }

    pub fn info(&self, msg: &str) {
        if LOG_LEVEL.load(Ordering::Relaxed) >= 1 {
            self.log("INFO", msg);
        }
    }

    pub fn debug(&self, msg: &str) {
        if LOG_LEVEL.load(Ordering::Relaxed) >= 2 {
            self.log("DEBUG", msg);
        }
    }

    pub fn trace(&self, msg: &str) {
        if LOG_LEVEL.load(Ordering::Relaxed) >= 3 {
            self.log("TRACE", msg);
        }
    }

    fn log(&self, level: &str, msg: &str) {
        let elapsed = self.start.elapsed().as_secs_f64();
        println!("[{:>8.4}s] [{}] {}", elapsed, level, msg);
    }
}

impl Default for Logger {
    fn default() -> Self {
        Self::new()
    }
}
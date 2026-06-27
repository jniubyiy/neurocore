// src/compute_manager/gpu/executor.rs

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use crate::compute_manager::executor::Executor;
use super::init::GpuContext;

#[derive(Clone)]
pub struct GpuExecutor {
    context: Arc<GpuContext>,
    active_tasks: Arc<AtomicUsize>,
}

impl GpuExecutor {
    pub fn new(context: GpuContext) -> Self {
        Self {
            context: Arc::new(context),
            active_tasks: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl Executor for GpuExecutor {
    fn execute_dyn(&self, f: Box<dyn FnOnce() + Send>) {
        self.active_tasks.fetch_add(1, Ordering::Release);
        let active = self.active_tasks.clone();
        std::thread::spawn(move || {
            f();
            active.fetch_sub(1, Ordering::Release);
        });
    }

    fn wait_all(&self) {
        while self.active_tasks.load(Ordering::Acquire) > 0 {
            std::hint::spin_loop();
        }
    }

    fn num_workers(&self) -> usize { 1 }

    fn plan_chunks_assignment(&self, total_tasks: usize) -> Vec<Vec<(usize, usize)>> {
        if total_tasks == 0 { return vec![Vec::new(); 1]; }
        vec![vec![(0, total_tasks)]]
    }

    fn clone_executor(&self) -> Box<dyn Executor> {
        Box::new(self.clone())
    }
}
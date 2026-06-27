// src/compute_manager/executor.rs

pub type ChunkAssignment = Vec<Vec<(usize, usize)>>;

pub trait Executor: Send + Sync + 'static {
    fn execute_dyn(&self, f: Box<dyn FnOnce() + Send>);
    fn wait_all(&self);
    fn num_workers(&self) -> usize;
    fn plan_chunks_assignment(&self, total_tasks: usize) -> ChunkAssignment;
    fn clone_executor(&self) -> Box<dyn Executor>;
}
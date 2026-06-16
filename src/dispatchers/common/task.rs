use std::sync::{Arc, Barrier};

pub struct LayerPlan {
    pub tasks: Vec<RangeTask>,
    pub barrier: Arc<Barrier>,
}

#[derive(Clone)]
pub struct RangeTask {
    pub row_start: usize,
    pub row_end: usize,
    pub col_start: usize,
    pub col_end: usize,
}
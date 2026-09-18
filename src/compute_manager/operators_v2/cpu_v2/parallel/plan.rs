// src/compute_manager/cpu/parallel/plan.rs
//
// Типы описания плана чанкования батча.

/// Описание одного чанка в плане forward.
#[derive(Clone, Copy, Debug)]
pub(super) struct ChunkSlice {
    pub chunk_id: usize,
    pub worker_id: usize,
    pub in_start: usize,
    pub in_end: usize,
    pub out_start: usize,
    pub out_end: usize,
}

impl ChunkSlice {
    #[inline]
    pub fn in_size(&self) -> usize {
        self.in_end - self.in_start
    }

    #[inline]
    pub fn as_layout_tuple(&self) -> (usize, usize, usize) {
        (self.in_start, self.in_size(), self.in_end)
    }
}

/// План распределения forward по чанкам.
#[derive(Clone, Debug)]
pub(super) struct LayerChunkPlan {
    pub chunks: Vec<ChunkSlice>,
}

impl LayerChunkPlan {
    pub fn to_layout(&self) -> Vec<(usize, usize, usize)> {
        self.chunks.iter().map(|c| c.as_layout_tuple()).collect()
    }
}

/// Строит стандартный план чанкования из раскладки планировщика.
pub(super) fn default_layer_chunk_plan(
    assignments: &[Vec<(usize, usize, usize)>],
) -> LayerChunkPlan {
    let mut chunks = Vec::new();
    let mut chunk_id = 0usize;
    for (worker_id, worker_chunks) in assignments.iter().enumerate() {
        for &(start, _size, end) in worker_chunks {
            chunks.push(ChunkSlice {
                chunk_id,
                worker_id,
                in_start: start,
                in_end: end,
                out_start: start,
                out_end: end,
            });
            chunk_id += 1;
        }
    }
    LayerChunkPlan { chunks }
}
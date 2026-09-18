// src/compute_manager/graph/types.rs

use crate::layers::buffered_context::BufferedContext;

/// Контекст, сохраняемый слоями для обратного прохода.
#[derive(Clone)]
pub enum DynamicContext {
    /// Буферизованный контекст на основе `MatrixBufferHandle`.
    Buffered(BufferedContext),
}

/// Контексты обратного прохода, сгруппированные по чанкам.
/// Внешний вектор содержит по одному элементу на каждый чанк батча.
/// Каждый элемент — вектор контекстов слоёв для соответствующего чанка.
pub type ChunkedContexts = Vec<Vec<DynamicContext>>;


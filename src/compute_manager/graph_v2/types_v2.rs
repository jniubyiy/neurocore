// src/compute_manager/graph_v2/types_v2.rs
//
// Типы графа v2.
//
// Граф — коллекция сегментов. Каждый сегмент — один шаг forward-цепочки.
// Все типы здесь иммутабельны после построения (кроме ForwardCacheV2,
// который перезаписывается между forward и backward).

use std::sync::Arc;

use crate::compute_manager::jobs_v2::{
    ConnectorOpKind, DimOpKind, ForwardContextsV2,
};
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;

// ============================================================================
// Сегмент
// ============================================================================

/// Вид сегмента.
pub enum SegmentKindV2 {
    /// UniversalProcessor: цепочка обычных слоёв (Linear, ReLU, ...).
    /// Слои идут подряд, между ними нет ветвлений и смены формы.
    Universal {
        layers: Arc<Vec<Box<dyn UniversalLayer>>>,
        slices: Vec<ParamSlice>,
    },

    /// Изменение размерности (Unsqueeze / ReduceMean).
    /// Не имеет параметров.
    DimOp { kind: DimOpKind },

    /// Коннектор (Splitter / Combiner / SplitterConnector / CombinerConnector).
    Connector { kind: ConnectorOpKind },
}

/// Один сегмент графа.
pub struct SegmentV2 {
    /// Порядковый номер в графе.
    pub index: usize,

    /// Вид сегмента.
    pub kind: SegmentKindV2,

    /// Форма входа без batch (например, `[4]` для Linear(4→2)).
    pub input_shape: Vec<usize>,

    /// Форма выхода без batch.
    pub output_shape: Vec<usize>,

    /// Сколько выходных потоков даёт сегмент.
    /// 1 для Universal/DimOp/Combiner/CombinerConnector.
    /// 2 для Splitter/SplitterConnector.
    pub stream_count: usize,

    /// Какие именно потоки обновляет сегмент.
    /// `None` — все потоки (обычно — единственный).
    pub stream_indices: Option<Vec<usize>>,
}

// ============================================================================
// Кэш forward
// ============================================================================

/// Состояние сегмента, сохранённое между forward и backward.
#[derive(Clone)]
pub enum SegmentForwardStateV2 {
    /// Для DimOp / Connector-no-op — ничего не нужно.
    None,

    /// Для Universal — контексты слоёв, возвращённые forward.
    Universal { contexts: ForwardContextsV2 },

    /// Для Splitter / Combiner — входы forward и внутренние
    /// пред-активации, необходимые backward'у.
    ///
    /// * Splitter: `inputs = [x]`, `pre = [pre_a, pre_b]`.
    /// * Combiner: `inputs = [a, b]`, `pre = [pre]`.
    /// * SplitterConnector / CombinerConnector: используется `None` (no-op).
    Connector {
        inputs: Vec<MatrixBufferHandle>,
        pre: Vec<MatrixBufferHandle>,
    },
}

/// Кэш forward-прохода.
///
/// Хранит:
///   * для каждого сегмента — его собственные state'ы (для backward);
///   * финальный выходной handle (для сверки / отладки).
#[derive(Clone)]
pub struct ForwardCacheV2 {
    /// Состояния сегментов (индекс = index сегмента в графе).
    pub segment_states: Vec<SegmentForwardStateV2>,

    /// Буфер выхода (последний результат forward).
    pub output: MatrixBufferHandle,
}

impl ForwardCacheV2 {
    /// Количество сегментов в кэше.
    #[inline]
    pub fn len(&self) -> usize {
        self.segment_states.len()
    }

    /// `true`, если кэш пуст.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.segment_states.is_empty()
    }
}
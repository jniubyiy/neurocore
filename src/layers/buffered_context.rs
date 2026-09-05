// src/layers/buffered_context.rs

use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

/// Контекст, сохраняемый слоями при буферизованном прямом проходе.
///
/// В отличие от `MatContext`, здесь хранятся не `faer::Mat`, а лёгкие
/// дескрипторы `MatrixBufferHandle`, которые ссылаются на данные в
/// `MemoryExecutor`. Дескрипторы можно свободно клонировать, что позволяет
/// разделять один буфер между контекстом и следующим слоем без копирования.
///
/// Используется только в варианте `DynamicContext::Buffered`.
#[derive(Clone)]
pub enum BufferedContext {
    /// Вход линейного слоя.
    Linear {
        input: MatrixBufferHandle,
    },

    /// Вход ReLU.
    ReLU {
        input: MatrixBufferHandle,
    },

    /// Выход Sigmoid (для обратного прохода нужен выход).
    Sigmoid {
        output: MatrixBufferHandle,
    },

    /// Выход Tanh (для обратного прохода нужен выход).
    Tanh {
        output: MatrixBufferHandle,
    },

    /// Выход Softmax (для обратного прохода нужен выход).
    Softmax {
        output: MatrixBufferHandle,
    },

    /// Вход Memory (текущий обратный проход может его не использовать, но сохранён для полноты).
    Memory {
        input: MatrixBufferHandle,
    },

    /// Вход LeakyReLU.
    LeakyReLU {
        input: MatrixBufferHandle,
    },

    /// Вход SoftSparseGate.
    SoftSparseGate {
        input: MatrixBufferHandle,
    },

    /// Вход SoftKeepGate.
    SoftKeepGate {
        input: MatrixBufferHandle,
    },

    /// Вход DualAnchor.
    DualAnchor1D {
        input: MatrixBufferHandle,
    },

    /// Вход AdaptivePerFeatureActivation.
    AdaptiveActivation {
        input: MatrixBufferHandle,
    },

    /// Вход AdaptiveNormalization.
    AdaptiveNormalization {
        input: MatrixBufferHandle,
    },

    /// Вход BatchRenorm1d, включая статистики, использованные при прямом проходе.
    BatchRenorm {
        input: MatrixBufferHandle,
        mean: Vec<f32>,
        var: Vec<f32>,
        use_batch_stats: bool,
    },

    /// Вход ConcreteDropout.
    ConcreteDropout {
        input: MatrixBufferHandle,
    },

    /// Вход Mamba (состояния и промежуточные данные хранятся в самом слое).
    Mamba {
        input: MatrixBufferHandle,
    },

    /// Вход LinearAttention (промежуточные данные хранятся в самом слое).
    LinearAttention {
        input: MatrixBufferHandle,
    },

    /// Вход RelativePositionAttention (промежуточные данные хранятся в самом слое).
    RelativePositionAttention {
        input: MatrixBufferHandle,
    },

    /// Вход IndRNN (промежуточные данные хранятся в самом слое).
    IndRNN {
        input: MatrixBufferHandle,
    },

    /// Вход SpectrallyNormalizedLinear (сохранение sigma производится в самом слое).
    SpectralNormLinear {
        input: MatrixBufferHandle,
    },

    /// Вход Identity.
    Identity {
        input: MatrixBufferHandle,
    },

    /// Вход SplitterConnector (первый входной поток).
    SplitterConnector {
        input: MatrixBufferHandle,
    },

    /// Входы CombinerConnector (все входные потоки).
    CombinerConnector {
        inputs: Vec<MatrixBufferHandle>,
    },

    /// Контекст обучаемого Splitter:
    /// вход + pre-activation для обеих веток.
    Splitter {
        input: MatrixBufferHandle,
        pre_a: MatrixBufferHandle,
        pre_b: MatrixBufferHandle,
    },

    /// Контекст обучаемого Combiner:
    /// оба входа + pre-activation перед ReLU.
    Combiner {
        input_a: MatrixBufferHandle,
        input_b: MatrixBufferHandle,
        pre_act: MatrixBufferHandle,
    },
}
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
        arg: MatrixBufferHandle,
    },

    /// Вход Mamba (состояния и промежуточные данные хранятся в самом слое).
    Mamba {
        input: MatrixBufferHandle,
        h_all: MatrixBufferHandle,
    },

    /// Вход LinearAttention.
    ///
    /// Для CPU-реализации промежуточные результаты хранятся внутри слоя,
    /// поэтому здесь достаточно только `input`.
    /// Для GPU-реализации все промежуточные буферы должны быть сохранены,
    /// так как GPU-обратный проход не имеет доступа к внутреннему состоянию слоя.
    /// Поэтому поля, начиная с `q_raw`, являются `Option<MatrixBufferHandle>`:
    /// - `None` для CPU
    /// - `Some(handle)` для GPU
    LinearAttention {
        input: MatrixBufferHandle,
        q_raw: Option<MatrixBufferHandle>,
        k_raw: Option<MatrixBufferHandle>,
        v_raw: Option<MatrixBufferHandle>,
        q_phi: Option<MatrixBufferHandle>,
        k_phi: Option<MatrixBufferHandle>,
        kv: Option<MatrixBufferHandle>,
        z: Option<MatrixBufferHandle>,
    },

    /// Вход RelativePositionAttention.
    ///
    /// Аналогично LinearAttention, для CPU достаточно только `input`,
    /// для GPU необходимо сохранять промежуточные буферы.
    /// Поля `q`, `k`, `v`, `scores`, `weights` являются `Option<MatrixBufferHandle>`.
    RelativePositionAttention {
        input: MatrixBufferHandle,
        q: Option<MatrixBufferHandle>,
        k: Option<MatrixBufferHandle>,
        v: Option<MatrixBufferHandle>,
        scores: Option<MatrixBufferHandle>,
        weights: Option<MatrixBufferHandle>,
    },

    /// Вход IndRNN (промежуточные данные хранятся в самом слое).
    IndRNN {
        input: MatrixBufferHandle,
        h_all: MatrixBufferHandle,
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

    // ================= Новые слои =================

    /// Вход DualSlopeReLU.
    DualSlopeReLU {
        input: MatrixBufferHandle,
    },

    /// Вход LearnableMish.
    LearnableMish {
        input: MatrixBufferHandle,
    },

    /// Вход LearnableSoftplus.
    LearnableSoftplus {
        input: MatrixBufferHandle,
    },

    /// Вход RMSNormWithLearnableEpsilon.
    RMSNormWithLearnableEpsilon {
        input: MatrixBufferHandle,
    },

    /// Вход AdaptiveDropout.
    AdaptiveDropout {
        input: MatrixBufferHandle,
        mask: MatrixBufferHandle,
        arg: MatrixBufferHandle,
    },

    /// Вход FeatureFusion.
    FeatureFusion {
        input: MatrixBufferHandle,
    },

    /// Вход SparseFeatureSelectionGate.
    SparseFeatureSelectionGate {
        input: MatrixBufferHandle,
    },

    /// Вход MultiResolutionKANLinear.
    MultiResolutionKANLinear {
        input: MatrixBufferHandle,
    },
}
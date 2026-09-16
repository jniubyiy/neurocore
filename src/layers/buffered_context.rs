// src/layers/buffered_context.rs

use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

/// Буферы промежуточных данных одной головы LinearAttention на CPU.
///
/// Все буферы — векторы-столбцы (`n × 1`), потому что в CPU-реализации
/// они читаются и пишутся целиком через `read_range`/`write_range`.
///
/// Используется только в варианте `BufferedContext::LinearAttention::cpu_heads`.
/// Для GPU-пути этот тип не задействован — там состояние хранится в
/// `Option<MatrixBufferHandle>`-полях того же варианта контекста.
#[derive(Clone)]
pub struct CpuLinearAttentionHead {
    /// Преобразованные запросы Q: (batch · seq · d_head), column-major.
    pub q: MatrixBufferHandle,
    /// Преобразованные ключи K: (batch · seq · d_head), column-major.
    pub k: MatrixBufferHandle,
    /// Значения V до φ: (batch · seq · d_head), column-major.
    pub v: MatrixBufferHandle,
    /// Внешнее произведение K_phi^T · V: (d_head · d_head), row-major.
    pub kv: MatrixBufferHandle,
    /// Сумма K_phi по токенам: (d_head · 1).
    pub z: MatrixBufferHandle,
    /// Результат attention до выходной проекции:
    /// (batch · seq · d_head), column-major.
    pub attn_out: MatrixBufferHandle,
    /// Выход головы после выходной проекции:
    /// (batch · seq · d_model), column-major.
    pub y: MatrixBufferHandle,
    /// Вес головы на момент forward.
    pub weight: f32,
    /// Индекс головы в общем списке.
    pub head_index: usize,
}

/// Контекст, сохраняемый слоями при буферизованном прямом проходе.
///
/// В отличие от `MatContext`, здесь хранятся не `faer::Mat`, а лёгкие
/// дескрипторы `MatrixBufferHandle`, которые ссылаются на данные в
/// `MemoryExecutor`. Дескрипторы можно свободно клонировать, что позволяет
/// разделять один буфер между контекстом и следующим слоем без копирования.
///
/// Используется только в варианте `DynamicContext::Buffered`.
///
/// # Per-chunk state
///
/// Слои с внутренним состоянием (`Memory`, `ConcreteDropout`,
/// `AdaptiveDropout`, `IndRNN`, `Mamba`, `LinearAttention`,
/// `RelativePositionAttention`, `SpectrallyNormalizedLinear`) создают
/// state-буферы per-chunk в `forward_buffered` из `TempMatrixPool` и кладут
/// их в соответствующий вариант этого enum. Backward читает состояние из
/// контекста, а не из полей слоя — это делает чанковое распараллеливание
/// безопасным (у каждого чанка свой изолированный state).
///
/// Исключение: `BatchRenorm1d` — его персистентные running-статистики
/// хранятся в самом слое (см. `BatchRenorm1d::state`), а в контексте
/// передаются только per-batch статистики `mean`/`var`, использованные
/// в данном forward.
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

    /// Вход Memory и per-chunk якоря.
    ///
    /// `min_cells`, `max_cells` — векторы длины `features`, созданные в
    /// `forward_buffered` и заполненные во время forward. В backward
    /// они не используются (gradient у Memory линейный), но контекст
    /// обязан держать дескрипторы, пока жив, — иначе буферы вернутся в
    /// пул раньше времени.
    Memory {
        input: MatrixBufferHandle,
        min_cells: MatrixBufferHandle,
        max_cells: MatrixBufferHandle,
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
    ///
    /// Персистентные running-статистики живут в `BatchRenorm1d::state`
    /// (аналогично весам). Здесь передаются только per-batch статистики
    /// `mean`/`var` текущего forward, нужные для backward-формулы.
    BatchRenorm {
        input: MatrixBufferHandle,
        mean: Vec<f32>,
        var: Vec<f32>,
        use_batch_stats: bool,
    },

    /// Вход ConcreteDropout и per-chunk аргументы сигмоиды.
    ///
    /// `arg` — буфер длины `batch · features`, созданный в forward.
    /// Backward читает его для восстановления `z = sigmoid(arg)`.
    ConcreteDropout {
        input: MatrixBufferHandle,
        arg: MatrixBufferHandle,
    },

    /// Вход Mamba и per-chunk state.
    ///
    /// - `h_all`: все скрытые состояния, column-major (batch·seq × state_dim).
    /// - `a_bar`: дискретизированная A, row-major (state_dim × state_dim).
    /// - `b_bar`: дискретизированная B, row-major (state_dim × input_dim).
    ///
    /// Все три буфера создаются в `forward_buffered` из пула и живут до
    /// конца backward чанка.
    Mamba {
        input: MatrixBufferHandle,
        h_all: MatrixBufferHandle,
        a_bar: MatrixBufferHandle,
        b_bar: MatrixBufferHandle,
    },

    /// Вход LinearAttention.
    ///
    /// Для CPU-пути: `cpu_heads` содержит по одной записи на активную
    /// голову; каждый per-head буфер хранится в пуле. `h_raw` и `h_soft` —
    /// числа голов, зафиксированные на момент forward; `batch`, `seq`,
    /// `d_model`, `d_head` — размерности того же forward.
    ///
    /// Для GPU-пути: `cpu_heads` пуст, состояние хранится в `Option`-полях
    /// `q_raw`, `k_raw`, `v_raw`, `q_phi`, `k_phi`, `kv`, `z` (глобальный
    /// FORWARD_CACHE в gpu/mod.rs).
    LinearAttention {
        input: MatrixBufferHandle,
        cpu_heads: Vec<CpuLinearAttentionHead>,
        h_raw: f32,
        h_soft: f32,
        batch: usize,
        seq: usize,
        d_model: usize,
        d_head: usize,
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
    /// Для CPU: `q`, `k`, `v`, `weights`, `attn_out` содержат дескрипторы
    /// per-chunk state-буферов (выделены из пула), `scores = None`
    /// (использован только внутри forward).
    ///
    /// Для GPU: используются те же поля — дескрипторы на GPU-буферы,
    /// `scores` также может быть `Some`.
    RelativePositionAttention {
        input: MatrixBufferHandle,
        q: Option<MatrixBufferHandle>,
        k: Option<MatrixBufferHandle>,
        v: Option<MatrixBufferHandle>,
        scores: Option<MatrixBufferHandle>,
        weights: Option<MatrixBufferHandle>,
        attn_out: Option<MatrixBufferHandle>,
    },

    /// Вход IndRNN и per-chunk скрытые состояния.
    ///
    /// `h_all` — буфер формы (batch · seq_len × input_dim), column-major:
    /// элемент `(r, t, j)` лежит по адресу
    /// `j * (batch * seq) + (r * seq + t)`.
    IndRNN {
        input: MatrixBufferHandle,
        h_all: MatrixBufferHandle,
    },

    /// Вход SpectrallyNormalizedLinear и per-chunk состояние степенного метода.
    ///
    /// - `u_state`: вектор u (in_features × 1).
    /// - `v_state`: вектор v (out_features × 1).
    /// - `sigma_state`: скаляр sigma (1 × 1).
    ///
    /// Все три буфера создаются в `forward_buffered` из пула и живут до
    /// конца backward чанка. В прежней версии эти значения хранились в
    /// `RwLock<SpectralNormState>` внутри слоя; теперь они изолированы
    /// по чанкам.
    SpectralNormLinear {
        input: MatrixBufferHandle,
        u_state: MatrixBufferHandle,
        v_state: MatrixBufferHandle,
        sigma_state: MatrixBufferHandle,
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

    /// Вход AdaptiveDropout и per-chunk state.
    ///
    /// - `mask`: бинарная маска z ∈ {0, 1} (batch · features).
    /// - `arg`:  аргумент сигмоиды a = (|x| − θ) / T (batch · features).
    ///
    /// Backward использует `mask` для восстановления z и `arg` для
    /// восстановления `p_keep = sigmoid(arg)`. Создаются в
    /// `forward_buffered` из пула.
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
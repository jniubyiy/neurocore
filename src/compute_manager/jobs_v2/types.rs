// src/compute_manager/jobs_v2/types.rs
//
// Базовые типы заданий (jobs) для архитектуры v2.
//
// Схема потока:
//
//     GraphV2 ──► SmartDistributor ──► [MemoryOperatorV2 | CpuOperatorV2 | GpuOperatorV2]
//                       ▲
//                       └── JobResult ◄──── JobHandle
//
// `GraphV2` формирует `Job`, передаёт его в `SmartDistributor::dispatch`.
// Распределитель по `Job::kind()` и `TopologySnapshot` выбирает оператора,
// вызывает `operator.submit(job) -> JobHandle`, затем `operator.drain(handle)
// -> JobResult`, и возвращает результат обратно в граф.
//
// Ни один тип из этого файла не знает про конкретного оператора — только
// про абстрактные `JobKind` и `OperatorKind`. Это обеспечивает принцип
// «через уровень не видно»: граф не различает CPU и GPU, а оператор не
// различает, от какого сегмента пришло задание.
//
// Все типы `Send + Sync`: `MatrixBufferHandle` — `Arc<RwLock<...>>` внутри,
// `Arc<Vec<Box<dyn UniversalLayer>>>` — `Send + Sync` (требование трейта),
// `DynamicContext` — содержит только дескрипторы.
//
// Ни один тип здесь не реализует `Debug`, потому что `MatrixBufferHandle`
// не реализует `Debug`. Для отладочного вывода используйте `Job::kind()`
// или `JobHandle` (он `Debug`).

use std::sync::Arc;

use crate::compute_manager::core::dynamic_context::ChunkedContexts;
use crate::compute_manager::operators_v2::gpu_v2::GpuCompute;
use crate::compute_manager::operators_v2::memory_v2::buffer::MatrixBufferHandle;
use crate::compute_manager::operators_v2::memory_v2::types::MemoryDeviceKind;
use crate::compute_manager::DynamicContext;

use crate::layers::UniversalLayer;
use crate::loss_plan::LossExpr;
use crate::model_plan::param_store::ParamSlice;
use crate::optimizer_plan::OptimizerDesc;
use crate::training_plan::Initializer;

// ============================================================================
// Идентификация
// ============================================================================

/// Тип задания. Плоский enum, по которому `SmartDistributor::strategy`
/// выбирает оператора без раскрытия payload'а.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JobKind {
    /// Миграция одного буфера на целевое устройство.
    Migrate,
    /// Прямой проход UniversalProcessor-сегмента.
    ForwardSegment,
    /// Обратный проход UniversalProcessor-сегмента.
    BackwardSegment,
    /// Вычисление функции потерь и градиента по pred.
    Loss,
    /// Фаза 1 оптимизации: модификация градиента (lr, momentum, adam, clip,
    /// weight_decay). Не обновляет параметры.
    ///
    /// Инвариант I-1 (MIGRATION_PLAN.md §2): обязательно предшествует
    /// `OptimizerApplyUpdate` и `AdapterPass`.
    OptimizerModifyGrads,
    /// Фаза 2 оптимизации: обновление параметров (`params -= grads`).
    /// Не модифицирует градиент.
    OptimizerApplyUpdate,
    /// Изменение размерности (Unsqueeze / ReduceMean).
    DimOp,
    /// Операция коннектора (Splitter / Combiner).
    ConnectorOp,
    /// Инициализация параметров сегмента.
    ParamInit,
}

/// Идентификатор оператора, принявшего job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperatorKind {
    Memory,
    Cpu,
    Gpu,
}

/// Хэндл, который оператор возвращает вызывающему.
///
/// Служит для последующего `drain(handle) -> JobResult`. Внутри оператора
/// слоты результатов индексируются по `id`; `kind` и `operator` нужны для
/// отладки и для того, чтобы вызывающий не перепутал результаты разных
/// операторов.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JobHandle {
    /// Локально-уникальный (в пределах оператора) идентификатор задания.
    pub id: u64,
    /// Тип задания (совпадает с `Job::kind()` исходного job'а).
    pub kind: JobKind,
    /// Оператор, принявший задание.
    pub operator: OperatorKind,
}

// ============================================================================
// Контейнер контекстов forward
// ============================================================================

/// Контексты, возвращённые forward-проходом сегмента.
///
/// Различает два режима исполнения:
///
/// * `Sequential` — весь батч обработан одним чанком. Используется, когда
///   чанковое распараллеливание неприменимо (один воркер, батч размером 1,
///   слой в чёрном списке `can_parallelize`). `Vec<DynamicContext>` — по
///   одному контексту на слой.
///
/// * `Chunked` — батч разбит на чанки. `contexts[c][i]` — контекст i-го слоя
///   для c-го чанка. `layout[c] = (in_start, size, in_end)` — диапазон
///   строк исходного входа, которому соответствует чанк `c`. Layout нужен
///   backward-проходу, чтобы правильно сшивать градиенты.
#[derive(Clone)]
pub enum ForwardContextsV2 {
    Sequential(Vec<DynamicContext>),
    Chunked {
        contexts: ChunkedContexts,
        layout: Vec<(usize, usize, usize)>,
    },
}

impl ForwardContextsV2 {
    /// Количество чанков (1 для `Sequential`).
    #[inline]
    pub fn num_chunks(&self) -> usize {
        match self {
            ForwardContextsV2::Sequential(_) => 1,
            ForwardContextsV2::Chunked { contexts, .. } => contexts.len(),
        }
    }

    /// Возвращает контексты первого чанка.
    ///
    /// Нужно GPU-оператору и другим потребителям, работающим без чанкования.
    #[inline]
    pub fn first_chunk(&self) -> Vec<DynamicContext> {
        match self {
            ForwardContextsV2::Sequential(v) => v.clone(),
            ForwardContextsV2::Chunked { contexts, .. } => {
                contexts.first().cloned().unwrap_or_default()
            }
        }
    }
}

// ============================================================================
// Payload'ы заданий
// ============================================================================

/// Прямой проход одного UniversalProcessor-сегмента.
pub struct ForwardSegmentJob {
    /// Индекс сегмента в графе (для tracing / observer).
    pub segment_index: usize,
    /// Слои сегмента.
    pub layers: Arc<Vec<Box<dyn UniversalLayer>>>,
    /// Срезы параметров, по одному на слой.
    pub slices: Vec<ParamSlice>,
    /// Общий буфер параметров сегмента.
    pub params: MatrixBufferHandle,
    /// Входной буфер (batch × in_features).
    pub input: MatrixBufferHandle,
}

/// Обратный проход одного UniversalProcessor-сегмента.
pub struct BackwardSegmentJob {
    /// Индекс сегмента в графе.
    pub segment_index: usize,
    /// Слои сегмента.
    pub layers: Arc<Vec<Box<dyn UniversalLayer>>>,
    /// Срезы параметров, по одному на слой.
    pub slices: Vec<ParamSlice>,
    /// Буфер параметров (для backward-функций слоёв).
    pub params: MatrixBufferHandle,
    /// Буфер градиентов параметров (аккумулируется слоями).
    pub grad_params: MatrixBufferHandle,
    /// Градиент по выходу сегмента.
    pub grad_output: MatrixBufferHandle,
    /// Контексты слоёв, сохранённые при forward.
    pub contexts: ForwardContextsV2,
}

/// Вычисление функции потерь.
pub struct LossJob {
    /// Выражение потерь.
    pub expr: Arc<LossExpr>,
    /// Предсказание модели (batch × pred_features).
    pub pred: MatrixBufferHandle,
    /// Целевые значения (batch × target_features).
    pub target: MatrixBufferHandle,
}

/// Фаза 1 оптимизации: модификация градиента.
///
/// Применяет все кубики цепочки, кроме `ApplyUpdate`:
/// `ScaleGradient`, `AddWeightDecay`, `GradientClip`, `Momentum`,
/// `NesterovMomentum`, `Adam`.
///
/// # Инвариант I-1 (MIGRATION_PLAN.md §2)
///
/// Эта фаза НЕ обновляет параметры. `params` здесь — только для чтения
/// (например, `AddWeightDecay` читает их для расчёта вклада в градиент).
/// Обновление параметров — исключительная ответственность
/// `OptimizerApplyUpdate`.
pub struct OptimizerModifyGradsJob {
    /// Индекс буфера в `ParamStore`.
    pub buffer_idx: usize,
    /// Буфер параметров.
    pub params: MatrixBufferHandle,
    /// Буфер градиентов (in-place модифицируется этой фазой).
    pub grads: MatrixBufferHandle,
    /// Описание оптимизатора.
    pub optimizer: OptimizerDesc,
    /// Ссылка на `GpuCompute` — заполняется распределителем только если
    /// `params.is_gpu()` или `grads.is_gpu()`. Иначе `None`.
    pub gpu_compute: Option<Arc<GpuCompute>>,
}

/// Фаза 2 оптимизации: обновление параметров.
///
/// Применяет только `ApplyUpdate` (`params -= grads`). Все модификации
/// градиента должны быть выполнены фазой `OptimizerModifyGrads` до этого.
///
/// # Инвариант I-1 (MIGRATION_PLAN.md §2)
///
/// Эта фаза НЕ модифицирует градиент. `grads` здесь — только для чтения.
/// После этой фазы `optimizer_step` завершён, следующий шаг начинается
/// снова с `forward`.
pub struct OptimizerApplyUpdateJob {
    /// Индекс буфера в `ParamStore`.
    pub buffer_idx: usize,
    /// Буфер параметров (in-place обновляется этой фазой).
    pub params: MatrixBufferHandle,
    /// Буфер градиентов.
    pub grads: MatrixBufferHandle,
    /// Описание оптимизатора.
    pub optimizer: OptimizerDesc,
    /// Ссылка на `GpuCompute` — заполняется распределителем только если
    /// `params.is_gpu()` или `grads.is_gpu()`. Иначе `None`.
    pub gpu_compute: Option<Arc<GpuCompute>>,
}

/// Миграция одного буфера.
pub struct MigrateJob {
    pub handle: MatrixBufferHandle,
    pub target: MemoryDeviceKind,
}

/// Вид операции изменения размерности.
#[derive(Debug, Clone)]
pub enum DimOpKind {
    /// Разворачивание размерности: reshape без потери элементов.
    Unsqueeze(Vec<usize>),
    /// Сжатие размерности: reshape без потери элементов.
    ReduceMean(Vec<usize>),
}

/// Операция изменения размерности одного буфера.
pub struct DimOpJob {
    pub kind: DimOpKind,
    pub input: MatrixBufferHandle,
}

// ============================================================================
// Коннекторы
// ============================================================================

/// Направление операции коннектора.
///
/// Forward и backward коннекторов имеют разные входы, разные выходы и
/// разный набор сохранённых буферов. Поле `direction` в `ConnectorOpJob`
/// различает эти два случая.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectorDirection {
    Forward,
    Backward,
}

/// Вид операции коннектора.
///
/// В v2 коннекторы с обучаемыми параметрами — только `Splitter` и
/// `Combiner`. `SplitterConnector` и `CombinerConnector` из старого пути
/// в v2 не сериализуются: они работают как чистые маркеры веток
/// внутри `builder_v2.rs` и не порождают `SegmentV2`.
#[derive(Debug, Clone)]
pub enum ConnectorOpKind {
    /// Splitter: 1 вход (batch, n) → 2 выхода (batch, p) и (batch, q).
    Splitter {
        input_dim: usize,
        output_dims: Vec<usize>,
        slice: ParamSlice,
    },
    /// Combiner: 2 входа (batch, n) → 1 выход (batch, m).
    Combiner {
        input_dim: usize,
        output_dim: usize,
        slice: ParamSlice,
    },
}

/// Операция коннектора.
///
/// # Forward
///
/// * `inputs`:
///   * `Splitter` — `[x]` (batch, n);
///   * `Combiner` — `[a, b]` (batch, n) каждый.
///
/// * `saved` — пусто (forward не нуждается в сохранённых буферах).
///
/// * Результат — `JobResult::Buffers(…)`:
///   * `Splitter` — `[out_a, out_b, pre_a, pre_b]`;
///   * `Combiner` — `[out, pre]`.
///
/// # Backward
///
/// * `inputs` — градиенты по выходам соответствующего forward:
///   * `Splitter` — `[delta_a, delta_b]`;
///   * `Combiner` — `[delta]`.
///
/// * `saved` — forward-состояние, необходимое для backward:
///   * `Splitter` — `[x, pre_a, pre_b]`;
///   * `Combiner` — `[a, b, pre]`.
///
/// * `grad_params` — обязательно для обоих видов.
///
/// * Результат — `JobResult::Buffers(…)`:
///   * `Splitter` — `[dx]`;
///   * `Combiner` — `[da, db]`.
pub struct ConnectorOpJob {
    pub direction: ConnectorDirection,
    pub kind: ConnectorOpKind,
    pub inputs: Vec<MatrixBufferHandle>,
    pub params: Option<MatrixBufferHandle>,
    pub grad_params: Option<MatrixBufferHandle>,
    pub saved: Vec<MatrixBufferHandle>,
}

/// Инициализация параметров одного буфера.
///
/// # Порядок применения
///
/// 1. Если `precomputed = Some(data)` — эти значения записываются напрямую,
///    а `initializer` и `seed` игнорируются. Это позволяет `GraphV2`
///    сгенерировать всю последовательность **одним RNG** (как v1 `execute.rs`)
///    и раздать буферам её куски. Критично для критерия ТЗ «loss-кривая v2
///    совпадает со старым execute при одинаковом `plan.seed`».
///
/// 2. Если `precomputed = None` — значения генерируются локально по
///    `initializer`. Для `RandomUniform` используется
///    `StdRng::seed_from_u64(seed.wrapping_add(buffer_idx))`: каждый буфер
///    получает **свою** последовательность (устраняет коррелированную
///    инициализацию), но детерминированно.
pub struct ParamInitJob {
    pub buffer_idx: usize,
    pub params: MatrixBufferHandle,
    pub initializer: Initializer,
    pub seed: Option<u64>,
    /// Предварительно сгенерированные значения (см. описание выше).
    pub precomputed: Option<Vec<f32>>,
}

// ============================================================================
// Задание
// ============================================================================

/// Полное задание: тип + payload. Создаётся графом, исполняется оператором.
pub enum Job {
    Migrate(MigrateJob),
    ForwardSegment(ForwardSegmentJob),
    BackwardSegment(BackwardSegmentJob),
    Loss(LossJob),
    OptimizerModifyGrads(OptimizerModifyGradsJob),
    OptimizerApplyUpdate(OptimizerApplyUpdateJob),
    DimOp(DimOpJob),
    ConnectorOp(ConnectorOpJob),
    ParamInit(ParamInitJob),
}

impl Job {
    /// Тип задания. Используется `SmartDistributor` для выбора оператора.
    #[inline]
    pub fn kind(&self) -> JobKind {
        match self {
            Job::Migrate(_) => JobKind::Migrate,
            Job::ForwardSegment(_) => JobKind::ForwardSegment,
            Job::BackwardSegment(_) => JobKind::BackwardSegment,
            Job::Loss(_) => JobKind::Loss,
            Job::OptimizerModifyGrads(_) => JobKind::OptimizerModifyGrads,
            Job::OptimizerApplyUpdate(_) => JobKind::OptimizerApplyUpdate,
            Job::DimOp(_) => JobKind::DimOp,
            Job::ConnectorOp(_) => JobKind::ConnectorOp,
            Job::ParamInit(_) => JobKind::ParamInit,
        }
    }
}

// ============================================================================
// Результат
// ============================================================================

/// Результат исполнения задания. Выбирается по `Job::kind()`.
pub enum JobResult {
    /// Задание без возвращаемого значения
    /// (`Migrate`, `OptimizerModifyGrads`, `OptimizerApplyUpdate`, `ParamInit`).
    Unit,

    /// Прямой проход сегмента: выходной буфер + контексты слоёв.
    Forward {
        output: MatrixBufferHandle,
        contexts: ForwardContextsV2,
    },

    /// Обратный проход сегмента: градиент по входу.
    Backward {
        grad_input: MatrixBufferHandle,
    },

    /// Loss: значение и градиент по предсказанию.
    Loss {
        value: f32,
        grad_pred: MatrixBufferHandle,
    },

    /// Один буфер (DimOp).
    Buffer(MatrixBufferHandle),

    /// Несколько буферов (ConnectorOp).
    Buffers(Vec<MatrixBufferHandle>),

    /// Ошибка исполнения.
    Failed(String),
}

impl JobResult {
    /// Возвращает `true`, если результат — ошибка.
    #[inline]
    pub fn is_failed(&self) -> bool {
        matches!(self, JobResult::Failed(_))
    }

    /// Если результат — ошибка, возвращает текст сообщения.
    #[inline]
    pub fn into_failed_message(self) -> Option<String> {
        match self {
            JobResult::Failed(msg) => Some(msg),
            _ => None,
        }
    }
}
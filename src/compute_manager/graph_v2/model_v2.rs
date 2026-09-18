// src/compute_manager/graph_v2/model_v2.rs
//
// GraphV2 — публичный фасад графа.
//
// Знает:
//   * Vec<SegmentV2>;
//   * Arc<SmartDistributor>;
//   * GraphObserverV2;
//   * Arc<Mutex<ParamStore>>;
//   * Arc<Mutex<TempMatrixPool>> (тот же, что у распределителя);
//   * input_shapes / output_shapes;
//   * forward_cache (перезаписывается между forward и backward).
//
// Делает:
//   * forward(DynamicTensor) -> DynamicTensor;
//   * loss(LossDesc, &pred, &target) -> (f32, DynamicTensor);
//   * backward(delta: DynamicTensor) -> DynamicTensor;
//   * optimizer_step(OptimizerDesc);
//   * observe_step(loss), observe_epoch(epoch), end_epoch() -> EpochReportV2.
//
// Не делает:
//   * не выбирает CPU/GPU (это делает distributor);
//   * не мигрирует буферы вручную (это делает distributor::ensure_inputs_local);
//   * не знает про потоки, scheduler, mini-model.
//
// ВАЖНО (правка): контракт «наружу из GraphV2 хендл уходит в HostRam»
// реализуется через КОПИРОВАНИЕ в отдельный CPU-буфер, а не через
// in-place миграцию оригинального хендла.
//
// Причина: `BufferedContext` ряда слоёв (Sigmoid, Tanh, Softmax) хранит
// клон `output`-хендла сегмента. `MemoryExecutor::move_matrix_handle`
// меняет storage в `MatrixEntry` in-place, поэтому все клоны одного и
// того же id видят новое устройство. Если мигрировать `cache.output`
// в HostRam, контекст последнего слоя тоже «переедет» на CPU, и
// `assert!(output.is_gpu())` в GPU-backward упадёт.
//
// Копирование в новый буфер оставляет оригинал (и все его клоны в
// контекстах) на том устройстве, где он был создан.

use std::sync::{Arc, Mutex};

use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::distributor_v2::{
    SegmentTopologyInfo, SmartDistributor,
};
use crate::compute_manager::jobs_v2::{
    Job, JobResult, LossJob, OptimizerStepJob, ParamInitJob,
};
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::compute_manager::memory_executor::types::MemoryDeviceKind;
use crate::device_plan::DevicePlan;
use crate::loss_plan::desc::LossDesc;
use crate::model_plan::layer_desc::LayerDesc;
use crate::model_plan::param_store::ParamStore;
use crate::optimizer_plan::OptimizerDesc;
use crate::training_plan::Initializer;

use super::builder_v2::build_segments;
use super::forward_v2::run_forward;
use super::backward_v2::run_backward;
use super::observer_v2::{
    EpochReportV2, GraphObserverV2, MonitorConfigV2, WarningV2,
};
use super::types_v2::{ForwardCacheV2, SegmentV2};

// ============================================================================
// GraphV2
// ============================================================================

/// Граф v2.
pub struct GraphV2 {
    /// Сегменты.
    pub(crate) segments: Vec<SegmentV2>,

    /// Распределитель заданий.
    pub(crate) distributor: Arc<SmartDistributor>,

    /// Наблюдатель.
    pub(crate) observer: GraphObserverV2,

    /// Хранилище параметров.
    pub(crate) param_store: Arc<Mutex<ParamStore>>,

    /// Temp-pool (тот же, что у распределителя).
    pub(crate) temp_pool: Arc<Mutex<TempMatrixPool>>,

    /// Форма входа (без batch).
    pub(crate) input_shape: Vec<usize>,

    /// Форма выхода (без batch).
    pub(crate) output_shape: Vec<usize>,

    /// Кэш forward-прохода. Заполняется в `forward`, потребляется в `backward`.
    pub(crate) forward_cache: Option<ForwardCacheV2>,

    /// Seed для детерминированной инициализации.
    pub(crate) seed: Option<u64>,
}

impl GraphV2 {
    // -----------------------------------------------------------------------
    // Построение
    // -----------------------------------------------------------------------

    /// Создаёт граф из описания слоёв.
    ///
    /// За кулисами:
    ///   1. Создаётся `SmartDistributor` на основе `DevicePlan`.
    ///   2. Строятся сегменты через `builder_v2::build_segments`.
    ///   3. Создаётся наблюдатель с конфигурацией по умолчанию.
    pub fn build(
        layers_desc: Vec<LayerDesc>,
        device_plan: &DevicePlan,
    ) -> Result<Self, String> {
        if layers_desc.is_empty() {
            return Err("GraphV2::build: empty layer list".into());
        }

        let distributor = Arc::new(SmartDistributor::new(device_plan)?);

        // Пустой ParamStore (аллокация буферов будет в build_segments).
        let param_store = Arc::new(Mutex::new(ParamStore::new(
            distributor.snapshot().memory_executor.clone(),
        )));

        // Пока все параметры в HostRam; миграция — на этапе on_epoch_boundary.
        let segments = build_segments(
            &layers_desc,
            &param_store,
            MemoryDeviceKind::HostRam,
        )?;

        let input_shape = layers_desc
            .first()
            .map(|l| l.input_shape.streams.clone())
            .unwrap_or_default();
        let output_shape = layers_desc
            .last()
            .map(|l| l.output_shape.streams.clone())
            .unwrap_or_default();

        let temp_pool = distributor.temp_pool();

        Ok(Self {
            segments,
            distributor,
            observer: GraphObserverV2::new(),
            param_store,
            temp_pool,
            input_shape,
            output_shape,
            forward_cache: None,
            seed: None,
        })
    }

    /// Настраивает наблюдателя.
    pub fn with_monitor_config(mut self, cfg: MonitorConfigV2) -> Self {
        self.observer = GraphObserverV2::with_config(cfg);
        self
    }

    /// Устанавливает seed для детерминированной инициализации.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    // -----------------------------------------------------------------------
    // Forward
    // -----------------------------------------------------------------------

    /// Прямой проход.
    ///
    /// Вход и выход — `DynamicTensor` с формой
    /// `(batch, product(input_shape))`.
    pub fn forward(&mut self, input: DynamicTensor) -> Result<DynamicTensor, String> {
        // Кэш предыдущего forward больше не нужен — перезаписываем.
        self.forward_cache = None;

        let input_handle = {
            let mut pool = self.temp_pool.lock().unwrap();
            dynamic_tensor_to_handle(&mut *pool, input)?
        };

        let (_out_handle, cache) = run_forward(
            &self.distributor,
            &self.param_store,
            &self.segments,
            input_handle,
        )?;

        // Контракт: наружу из GraphV2 хендл уходит в HostRam, потому что
        // пользователь получает DynamicTensor (то есть CPU-вектор).
        //
        // ВАЖНО: НЕ мигрируем `cache.output` in-place. Оригинальный
        // хендл может быть процитирован в `BufferedContext` последнего
        // слоя (Sigmoid/Tanh/Softmax хранят в контексте именно output).
        // In-place миграция изменила бы storage у всех клонов, включая
        // контекстный, и GPU-backward этого слоя упал бы на
        // assert!(output.is_gpu()).
        //
        // Вместо миграции — копируем данные в отдельный CPU-буфер.
        let out_handle_cpu = self.copy_handle_to_host_ram(&cache.output)?;

        self.forward_cache = Some(cache);

        let out_tensor =
            handle_to_dynamic_tensor(&out_handle_cpu, &self.output_shape)?;
        Ok(out_tensor)
    }

    // -----------------------------------------------------------------------
    // Loss
    // -----------------------------------------------------------------------

    /// Вычисляет loss и градиент по предсказанию.
    pub fn loss(
        &self,
        loss_desc: LossDesc,
        pred: &DynamicTensor,
        target: &DynamicTensor,
    ) -> Result<(f32, DynamicTensor), String> {
        // Конвертируем pred и target в handle'ы (в HostRam).
        // Миграцию в VRAM (если strategy выберет GPU-loss) сделает
        // distributor::ensure_inputs_local внутри dispatch.
        let (pred_handle, target_handle) = {
            let mut pool = self.temp_pool.lock().unwrap();
            let p = dynamic_tensor_to_handle(&mut *pool, pred.clone())?;
            let t = dynamic_tensor_to_handle(&mut *pool, target.clone())?;
            (p, t)
        };

        let expr = loss_desc.build();
        let job = Job::Loss(LossJob {
            expr,
            pred: pred_handle,
            target: target_handle,
        });

        let result = self.distributor.dispatch(job);
        match result {
            JobResult::Loss { value, grad_pred } => {
                // Контракт: grad_pred отдаётся наружу в HostRam.
                //
                // Здесь grad_pred — «свежий» буфер, созданный внутри loss
                // и нигде в контекстах не закэшированный. in-place миграция
                // безопасна, но для единообразия политики (см. forward)
                // копируем в отдельный CPU-буфер.
                let grad_pred_cpu = self.copy_handle_to_host_ram(&grad_pred)?;

                let grad_tensor =
                    handle_to_dynamic_tensor(&grad_pred_cpu, &self.output_shape)?;
                Ok((value, grad_tensor))
            }
            JobResult::Failed(msg) => Err(format!("GraphV2::loss: {}", msg)),
            _ => Err("GraphV2::loss: unexpected JobResult".into()),
        }
    }

    // -----------------------------------------------------------------------
    // Backward
    // -----------------------------------------------------------------------

    /// Обратный проход.
    ///
    /// `delta` — градиент по выходу (обычно из `loss`).
    /// Возвращает градиент по входу первого сегмента.
    pub fn backward(&mut self, delta: DynamicTensor) -> Result<DynamicTensor, String> {
        let cache = self
            .forward_cache
            .take()
            .ok_or_else(|| "GraphV2::backward: forward_cache is empty".to_string())?;

        let delta_handle = {
            let mut pool = self.temp_pool.lock().unwrap();
            dynamic_tensor_to_handle(&mut *pool, delta)?
        };

        let grad_input = run_backward(
            &self.distributor,
            &self.param_store,
            &self.segments,
            cache,
            delta_handle,
        )?;

        // Контракт: наружу grad_input отдаётся в HostRam.
        // Копирование, а не миграция — см. комментарий в `forward`.
        let grad_input_cpu = self.copy_handle_to_host_ram(&grad_input)?;

        let grad_tensor =
            handle_to_dynamic_tensor(&grad_input_cpu, &self.input_shape)?;
        Ok(grad_tensor)
    }

    // -----------------------------------------------------------------------
    // Optimizer step
    // -----------------------------------------------------------------------

    /// Один шаг оптимизатора по каждому буферу параметров.
    pub fn optimizer_step(&self, optimizer: OptimizerDesc) -> Result<(), String> {
        let num_buffers = {
            let ps = self.param_store.lock().unwrap();
            ps.num_buffers()
        };

        for buffer_idx in 0..num_buffers {
            let (params, grads) = {
                let ps = self.param_store.lock().unwrap();
                let buffer = ps.get_param_buffer_by_idx(buffer_idx);
                (buffer.params.clone(), buffer.grads.clone())
            };

            let job = Job::OptimizerStep(OptimizerStepJob {
                buffer_idx,
                params,
                grads,
                optimizer: optimizer.clone(),
                gpu_compute: None, // будет вложен в SmartDistributor::prepare_job
            });

            match self.distributor.dispatch(job) {
                JobResult::Unit => {}
                JobResult::Failed(msg) => {
                    return Err(format!(
                        "GraphV2::optimizer_step: buffer {}: {}",
                        buffer_idx, msg
                    ));
                }
                _ => {
                    return Err(format!(
                        "GraphV2::optimizer_step: buffer {} unexpected JobResult",
                        buffer_idx
                    ));
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Наблюдение
    // -----------------------------------------------------------------------

    /// Зафиксировать шаг наблюдения.
    ///
    /// `grad_norm` вычислять не обязательно — можно передать `None`.
    pub fn observe_step(&mut self, loss: f32, grad_norm: Option<f32>) {
        self.observer.record_step(loss, grad_norm);
    }

    /// Отметить начало эпохи.
    ///
    /// Проверяет `observer.should_reassign()` и, при необходимости,
    /// запрашивает перераспределение у распределителя.
    pub fn observe_epoch(&mut self, epoch: usize) -> Result<(), String> {
        // Регулярный reassign: первая эпоха всегда, плюс при срабатывании
        // эвристик наблюдателя.
        let should = epoch == 0 || self.observer.should_reassign();
        if should {
            let infos = self.collect_segment_infos();
            self.distributor.on_epoch_boundary(&infos)?;
            self.observer.mark_reassigned();
        }
        Ok(())
    }

    /// Завершить эпоху, получить отчёт.
    pub fn end_epoch(&mut self) -> EpochReportV2 {
        self.observer.end_epoch()
    }

    /// Вернуть сводку предупреждений за всё время.
    pub fn all_warnings(&self) -> &[WarningV2] {
        self.observer.all_warnings()
    }

    // -----------------------------------------------------------------------
    // Инициализация параметров
    // -----------------------------------------------------------------------

    /// Инициализирует параметры по правилу `initializer`.
    ///
    /// # Порядок применения (совпадает с v1 `execute.rs`)
    ///
    /// 1. **Одна последовательность RNG на всю модель.** `GraphV2` сам
    ///    создаёт `StdRng::seed_from_u64(self.seed)` и генерирует
    ///    `total_params()` значений подряд. Это даёт **точное совпадение**
    ///    начальных весов с v1 `execute.rs` при одинаковом `plan.seed` —
    ///    критично для критерия ТЗ «loss-кривая v2 совпадает со старым
    ///    execute на тех же данных».
    ///
    /// 2. **Раскладка по буферам.** Сгенерированный вектор нарезается по
    ///    `param_len` каждого буфера и передаётся в `ParamInitJob` как
    ///    `precomputed`. Диспетчер и `CpuOperatorV2` просто записывают
    ///    эти значения — никакой собственной генерации не происходит.
    ///
    /// 3. **Zero-size buffers.** Если у сегмента нет параметров, `ParamInitJob`
    ///    для него не создаётся — как в v1.
    pub fn init_params(&self, initializer: Initializer) -> Result<(), String> {
        // ---- Шаг 1: одна последовательность RNG на всю модель. ----
        let all_data: Vec<f32> = {
            let ps = self.param_store.lock().unwrap();
            let total = ps.total_params();

            if total == 0 {
                Vec::new()
            } else {
                match &initializer {
                    Initializer::Zeros => vec![0.0; total],
                    Initializer::Ones => vec![1.0; total],
                    Initializer::RandomUniform { min, max } => {
                        use rand::rngs::StdRng;
                        use rand::{Rng, SeedableRng};
                        let mut rng: Box<dyn rand::RngCore> = match self.seed {
                            Some(s) => Box::new(StdRng::seed_from_u64(s)),
                            None => Box::new(rand::thread_rng()),
                        };
                        (0..total).map(|_| rng.gen_range(*min..*max)).collect()
                    }
                }
            }
        };

        // ---- Шаг 2: раскладка по буферам через ParamInitJob. ----
        let num_buffers = {
            let ps = self.param_store.lock().unwrap();
            ps.num_buffers()
        };

        let mut offset = 0usize;
        for buffer_idx in 0..num_buffers {
            let (params, len) = {
                let ps = self.param_store.lock().unwrap();
                let b = ps.get_param_buffer_by_idx(buffer_idx);
                (b.params.clone(), b.params.rows() * b.params.cols())
            };

            // Zero-size buffer — ничего не инициализируем, как в v1.
            if len == 0 {
                continue;
            }

            let slice = all_data[offset..offset + len].to_vec();
            offset += len;

            let job = Job::ParamInit(ParamInitJob {
                buffer_idx,
                params,
                initializer: initializer.clone(),
                seed: self.seed,
                precomputed: Some(slice),
            });

            match self.distributor.dispatch(job) {
                JobResult::Unit => {}
                JobResult::Failed(msg) => {
                    return Err(format!(
                        "GraphV2::init_params: buffer {}: {}",
                        buffer_idx, msg
                    ));
                }
                _ => {
                    return Err(format!(
                        "GraphV2::init_params: buffer {} unexpected JobResult",
                        buffer_idx
                    ));
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Доступ
    // -----------------------------------------------------------------------

    /// Ссылка на распределитель.
    pub fn distributor(&self) -> &Arc<SmartDistributor> {
        &self.distributor
    }

    /// Ссылка на хранилище параметров.
    pub fn param_store(&self) -> &Arc<Mutex<ParamStore>> {
        &self.param_store
    }

    /// Количество сегментов.
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Форма входа (без batch).
    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    /// Форма выхода (без batch).
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    // -----------------------------------------------------------------------
    // Внутреннее
    // -----------------------------------------------------------------------

    /// Копирует данные из произвольного handle (GPU или CPU) в свежий
    /// HostRam-handle, НЕ трогая оригинал.
    ///
    /// Это ключевое отличие от `SmartDistributor::ensure_local`: тот
    /// мигрирует оригинал in-place (меняет storage в `MatrixEntry`),
    /// из-за чего все клоны хендла «переезжают» на новое устройство.
    /// Здесь же создаётся отдельный буфер в HostRam, в него копируются
    /// данные, а оригинал остаётся там, где был.
    fn copy_handle_to_host_ram(
        &self,
        handle: &MatrixBufferHandle,
    ) -> Result<MatrixBufferHandle, String> {
        // Создаём свежий CPU-буфер через общий temp-pool.
        let cpu_buf = {
            let mut pool = self.temp_pool.lock().unwrap();
            pool.acquire(handle.rows(), handle.cols())
        };

        if handle.is_gpu() {
            // GPU → CPU: используем GpuCompute.
            let gpu = self
                .distributor
                .gpu_compute()
                .ok_or_else(|| {
                    "GraphV2::copy_handle_to_host_ram: GPU handle but no GpuCompute"
                        .to_string()
                })?;
            gpu.copy_gpu_to_cpu_handle(handle, &cpu_buf);
        } else {
            // Уже CPU: просто копируем содержимое.
            let src = handle.read();
            let src_slice = src
                .as_slice()
                .expect("GraphV2::copy_handle_to_host_ram: expected CPU buffer");
            cpu_buf.write_range(0, src_slice);
        }

        Ok(cpu_buf)
    }

    /// Собирает `SegmentTopologyInfo` для передачи в `on_epoch_boundary`.
    ///
    /// Заполняет `param_handle`, `grad_handle` и `opt_state_handle`, чтобы
    /// распределитель мог мигрировать все три буфера одним проходом.
    fn collect_segment_infos(&self) -> Vec<SegmentTopologyInfo> {
        let ps = self.param_store.lock().unwrap();
        let mut infos = Vec::with_capacity(self.segments.len());
        for seg in &self.segments {
            match &seg.kind {
                super::types_v2::SegmentKindV2::Universal { layers, slices } => {
                    let param_count: usize = layers.iter().map(|l| l.param_len()).sum();
                    let (param_handle, grad_handle, opt_state_handle, current_loc) =
                        match slices.first() {
                            Some(s) => {
                                let buffer = ps.get_param_buffer(s);
                                (
                                    Some(buffer.params.clone()),
                                    Some(buffer.grads.clone()),
                                    buffer.opt_state.clone(),
                                    buffer.location,
                                )
                            }
                            None => (None, None, None, MemoryDeviceKind::HostRam),
                        };
                    infos.push(SegmentTopologyInfo {
                        index: seg.index,
                        param_count,
                        param_handle,
                        grad_handle,
                        opt_state_handle,
                        current_param_location: current_loc,
                    });
                }
                _ => {
                    infos.push(SegmentTopologyInfo {
                        index: seg.index,
                        param_count: 0,
                        param_handle: None,
                        grad_handle: None,
                        opt_state_handle: None,
                        current_param_location: MemoryDeviceKind::HostRam,
                    });
                }
            }
        }
        infos
    }
}

// ============================================================================
// Конвертации DynamicTensor ↔ MatrixBufferHandle
// ============================================================================

/// `DynamicTensor` → `MatrixBufferHandle` (col-major).
///
/// Всегда создаёт буфер в HostRam. Если стратегия решит, что job
/// должен идти на GPU, `SmartDistributor::ensure_inputs_local`
/// мигрирует буфер в VRAM перед отправкой оператору.
pub(crate) fn dynamic_tensor_to_handle(
    pool: &mut TempMatrixPool,
    tensor: DynamicTensor,
) -> Result<MatrixBufferHandle, String> {
    let batch = tensor.batch_size();
    let features = tensor.features();
    let flat = tensor.to_flat();
    if flat.len() != batch * features {
        return Err(format!(
            "dynamic_tensor_to_handle: flat len {} != batch*features {}",
            flat.len(),
            batch * features
        ));
    }

    let buf = pool.acquire(batch, features);
    {
        let mut guard = buf.write();
        let slice = guard.as_slice_mut().expect("CPU buffer");
        for r in 0..batch {
            for c in 0..features {
                slice[c * batch + r] = flat[r * features + c];
            }
        }
    }
    Ok(buf)
}

/// `MatrixBufferHandle` → `DynamicTensor` (row-major).
///
/// Требует, чтобы handle был в HostRam. Вызывающая сторона обязана
/// предварительно убедиться, что данные лежат в HostRam (например,
/// через `GraphV2::copy_handle_to_host_ram`).
pub(crate) fn handle_to_dynamic_tensor(
    handle: &MatrixBufferHandle,
    shape: &[usize],
) -> Result<DynamicTensor, String> {
    if handle.is_gpu() {
        return Err(
            "handle_to_dynamic_tensor: GPU handles not supported (copy to HostRam first)"
                .into(),
        );
    }

    let batch = handle.rows();
    let features = handle.cols();

    let feature_count: usize = shape.iter().product();
    if feature_count != features {
        return Err(format!(
            "handle_to_dynamic_tensor: shape {:?} product {} != cols {}",
            shape, feature_count, features
        ));
    }

    let guard = handle.read();
    let slice = guard.as_slice().expect("CPU buffer");
    let mut flat = vec![0.0f32; batch * features];
    for r in 0..batch {
        for c in 0..features {
            flat[r * features + c] = slice[c * batch + r];
        }
    }
    drop(guard);

    flat_to_dynamic_tensor(shape, flat)
}

fn flat_to_dynamic_tensor(shape: &[usize], flat: Vec<f32>) -> Result<DynamicTensor, String> {
    let feature_count: usize = shape.iter().product();
    if feature_count == 0 {
        return Err("flat_to_dynamic_tensor: zero feature_count".into());
    }
    if flat.len() % feature_count != 0 {
        return Err(format!(
            "flat_to_dynamic_tensor: flat len {} not divisible by feature_count {}",
            flat.len(),
            feature_count
        ));
    }
    let batch = flat.len() / feature_count;

    let dest = match shape.len() {
        1 => DynamicTensor::Dim1(crate::tensor::Tensor2D::zeros(batch, shape[0])),
        2 => DynamicTensor::Dim2(crate::tensor::Tensor3D::zeros(batch, shape[0], shape[1])),
        3 => DynamicTensor::Dim3(crate::tensor::Tensor4D::zeros(
            batch, shape[0], shape[1], shape[2],
        )),
        4 => DynamicTensor::Dim4(crate::tensor::Tensor5D::zeros(
            batch, shape[0], shape[1], shape[2], shape[3],
        )),
        _ => {
            return Err(format!(
                "flat_to_dynamic_tensor: unsupported spatial dims {}",
                shape.len()
            ));
        }
    };

    Ok(DynamicTensor::from_flat(&dest, flat))
}
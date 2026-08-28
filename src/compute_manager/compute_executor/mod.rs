// src/compute_manager/compute_executor/mod.rs

use std::sync::{Arc, Mutex, MutexGuard};

use crate::compute_manager::device_spec::DeviceId;
use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::gpu::pipeline::PipelineCache;
use crate::compute_manager::memory_executor::MemoryExecutor;
use crate::compute_manager::graph::types::Segment;
use crate::device_plan::plan::{ComputeDevice, DevicePlan};

pub mod placement;
pub mod profiling;
pub mod strategy;
pub mod migration;

pub use placement::SegmentPlacement;
pub use profiling::ProfilingState;

/// Центральный исполнитель для вычислительных устройств.
///
/// Управляет размещением сегментов модели по доступным вычислительным устройствам
/// (CPU, GPU) с целью максимизации общей производительности. Пересматривает
/// размещение перед каждой новой эпохой обучения, используя накопленную
/// профилировочную статистику.
pub struct ComputeExecutor {
    device_plan: DevicePlan,
    gpu_compute: Option<Mutex<GpuCompute>>,
    memory_executor: Arc<Mutex<MemoryExecutor>>,
    profiling: Mutex<ProfilingState>,
    current_placement: Mutex<Vec<SegmentPlacement>>,
    epoch_counter: Mutex<usize>,
}

impl ComputeExecutor {
    /// Создаёт новый исполнитель на основе плана устройств и менеджера памяти.
    ///
    /// Если в плане присутствует GPU, исполнитель попытается получить уже
    /// зарегистрированный в `MemoryExecutor` контекст GPU. Если контекст отсутствует,
    /// будет возвращена ошибка, так как GPU должен быть инициализирован заранее
    /// через `DevicePlan::build_memory_executor`.
    pub fn new(
        device_plan: DevicePlan,
        memory_executor: Arc<Mutex<MemoryExecutor>>,
    ) -> Result<Self, String> {
        // Инициализация GPU, если он указан в плане
        let gpu_compute = if let Some(gpu_device) = device_plan
            .compute_devices
            .iter()
            .find(|d| matches!(d, ComputeDevice::Gpu { .. }))
        {
            if let ComputeDevice::Gpu { id } = gpu_device {
                // Получаем уже существующий контекст GPU из MemoryExecutor.
                let ctx = {
                    let mem = memory_executor.lock().unwrap();
                    mem.gpu_context(DeviceId(*id))
                        .cloned()
                        .ok_or_else(|| {
                            format!(
                                "ComputeExecutor: GPU контекст для id {} не найден в MemoryExecutor",
                                id
                            )
                        })?
                };

                let pipeline_cache = Arc::new(PipelineCache::new(ctx.device.clone()));
                let compute = GpuCompute::new(
                    ctx,
                    pipeline_cache,
                    memory_executor.clone(),
                    DeviceId(*id),
                );
                Some(Mutex::new(compute))
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            device_plan,
            gpu_compute,
            memory_executor,
            profiling: Mutex::new(ProfilingState::new()),
            current_placement: Mutex::new(Vec::new()),
            epoch_counter: Mutex::new(0),
        })
    }

    /// Возвращает количество потоков CPU, суммарно по всем CPU в плане.
    pub fn cpu_threads(&self) -> usize {
        self.device_plan
            .compute_devices
            .iter()
            .filter_map(|d| match d {
                ComputeDevice::Cpu { threads, .. } => Some(*threads),
                _ => None,
            })
            .sum::<usize>()
            .max(1)
    }

    /// Проверяет, доступен ли GPU.
    pub fn has_gpu(&self) -> bool {
        self.gpu_compute.is_some()
    }

    /// Возвращает ссылку на `GpuCompute`, если он инициализирован.
    /// Блокирует внутренний мьютекс, поэтому вызывающий код должен быть осторожен,
    /// чтобы не удерживать блокировку длительное время.
    pub fn gpu_compute(&self) -> Option<MutexGuard<'_, GpuCompute>> {
        self.gpu_compute.as_ref().map(|m| m.lock().unwrap())
    }

    /// Вычисляет начальное размещение сегментов (статическая эвристика).
    pub fn initial_placement(
        &self,
        segments: &[Segment],
        batch_size: usize,
    ) -> Vec<SegmentPlacement> {
        placement::assign_initial(segments, &self.device_plan, batch_size)
    }

    /// Пересматривает размещение сегментов, если наступила новая эпоха или принудительно.
    ///
    /// # Аргументы
    /// * `segments` – список сегментов модели.
    /// * `batch_size` – текущий размер батча.
    /// * `force` – если `true`, перераспределение выполняется всегда,
    ///   независимо от номера эпохи.
    pub fn redistribute(
        &self,
        segments: &[Segment],
        batch_size: usize,
        force: bool,
    ) {
        let mut epoch_guard = self.epoch_counter.lock().unwrap();
        let current_epoch = *epoch_guard;

        // Всегда перераспределяем, если force или это первая эпоха,
        // а также если профилировщик считает, что пора.
        let should_reassign = force
            || current_epoch == 0
            || self.profiling.lock().unwrap().should_reassign(current_epoch);

        if should_reassign {
            let new_placement = self.compute_adaptive_placement(segments, batch_size);
            // Здесь можно выполнить миграцию данных между устройствами,
            // но поскольку параметры модели хранятся в одном общем CPU‑буфере,
            // фактическое перемещение не требуется для вычислительного процесса.
            // В будущем, если появятся сегментно-локальные буферы, здесь будет вызов
            // migration::migrate_segments(segments, &new_placement, &self.memory_executor);
            let mut placement_guard = self.current_placement.lock().unwrap();
            *placement_guard = new_placement.clone();
            self.profiling.lock().unwrap().mark_reassigned(current_epoch);
        }

        *epoch_guard = current_epoch + 1;
    }

    /// Внутренний метод для адаптивного выбора устройств.
    fn compute_adaptive_placement(
        &self,
        segments: &[Segment],
        batch_size: usize,
    ) -> Vec<SegmentPlacement> {
        let profiling = self.profiling.lock().unwrap();
        let current = self.current_placement.lock().unwrap();
        strategy::compute_adaptive_placement(
            segments,
            &self.device_plan,
            batch_size,
            &profiling,
            &current,
        )
    }

    /// Возвращает вычислительное устройство, назначенное сегменту с указанным индексом.
    pub fn device_for_segment(&self, seg_index: usize) -> ComputeDevice {
        let placement = self.current_placement.lock().unwrap();
        placement
            .get(seg_index)
            .map(|p| p.compute_device.clone())
            .unwrap_or_else(|| {
                // Если размещение ещё не задано, возвращаем первый CPU или GPU из плана.
                self.device_plan
                    .compute_devices
                    .first()
                    .cloned()
                    .unwrap_or(ComputeDevice::Cpu { id: 0, threads: 1 })
            })
    }

    /// Записывает время выполнения сегмента на заданном устройстве для профилирования.
    pub fn record_segment_time(
        &self,
        seg_index: usize,
        device: &ComputeDevice,
        duration_ns: f64,
    ) {
        self.profiling
            .lock()
            .unwrap()
            .add_timing(seg_index, device.clone(), duration_ns);
    }

    /// Возвращает копию текущего размещения.
    pub fn get_placement(&self) -> Vec<SegmentPlacement> {
        self.current_placement.lock().unwrap().clone()
    }

    /// Возвращает ссылку на менеджер памяти.
    pub fn memory_executor(&self) -> &Arc<Mutex<MemoryExecutor>> {
        &self.memory_executor
    }

    /// Возвращает ссылку на план устройств.
    pub fn device_plan(&self) -> &DevicePlan {
        &self.device_plan
    }
}

// Вспомогательный ключ для хеширования устройств в профилировщике.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct DeviceHashKey(ComputeDevice);

impl From<&ComputeDevice> for DeviceHashKey {
    fn from(dev: &ComputeDevice) -> Self {
        DeviceHashKey(dev.clone())
    }
}
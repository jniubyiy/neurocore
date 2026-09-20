// src/plans/model_plan/adapter_store/store.rs

use std::sync::{Arc, RwLock};

use crate::compute_manager::operators_v2::memory_v2::buffer::MatrixBufferHandle;
use crate::compute_manager::operators_v2::memory_v2::executor::MemoryExecutor;
use crate::compute_manager::operators_v2::memory_v2::policy::BufferPriority;
use crate::compute_manager::operators_v2::memory_v2::types::MemoryDeviceKind;

use super::buffer::AdapterStateBuffer;
use super::slice::AdapterSlice;

/// Менеджер персистентного состояния градиентных адаптеров.
///
/// # Назначение
///
/// Параллелен `ParamStore`, но хранит состояние адаптеров, а не
/// параметры. Один сегмент может содержать несколько адаптеров (по
/// одному на слой с адаптером). Внутри store каждый сегмент имеет
/// ровно один буфер состояния, разбитый по `AdapterSlice`.
///
/// # Инварианты (MIGRATION_PLAN.md §2)
///
///   * **I-6:** состояние живёт в отдельном store, optimizer его не
///     трогает.
///   * **I-5:** устройство хранения совпадает с устройством слоя —
///     store принимает `location` явно при `allocate_segment`.
///   * **Migrate** через тот же механизм, что и `ParamStore`:
///     `SmartDistributor::on_epoch_boundary`.
///
/// # Ленивость
///
/// Store может быть пустым (нет адаптеров с состоянием) — тогда
/// `num_buffers() == 0`, все методы доступа возвращают пустоту
/// безопасно. Это позволяет включать adapter infrastructure
/// постепенно: Фаза 2 (эта) — вхолостую; Фаза 5 — первый адаптер.
///
/// # Что store НЕ делает
///
///   * Не вызывает `MemoryExecutor::move_matrix_handle` — миграция
///     делегируется `SmartDistributor::on_epoch_boundary`, у которого
///     есть доступ к `MemoryOperatorV2`. Store хранит только
///     зафиксированное `location` и обновляется через `set_location`.
///
///   * Не трогает `grads` и `params` — это предмет `ParamStore`.
pub struct AdapterStateStore {
    /// Вектор буферов состояний. Индекс — `buffer_idx` в `AdapterSlice`.
    buffers: Vec<AdapterStateBuffer>,

    /// Глобальный менеджер памяти.
    memory: Arc<RwLock<MemoryExecutor>>,
}

impl AdapterStateStore {
    /// Создаёт пустой store.
    pub fn new(memory: Arc<RwLock<MemoryExecutor>>) -> Self {
        Self {
            buffers: Vec::new(),
            memory,
        }
    }

    /// Выделяет буфер состояния для нового сегмента.
    ///
    /// Параллель `ParamStore::allocate_segment`. Принимает размеры
    /// состояния (в элементах f32) для каждого адаптера сегмента.
    /// Возвращает `Vec<AdapterSlice>` — по одной записи на каждый
    /// переданный размер.
    ///
    /// Если суммарный размер равен нулю (нет адаптеров с состоянием
    /// либо все адаптеры без состояния), возвращает пустой вектор и
    /// новый буфер не создаётся.
    pub fn allocate_segment(
        &mut self,
        adapter_state_sizes: &[usize],
        location: MemoryDeviceKind,
    ) -> Vec<AdapterSlice> {
        let total_size: usize = adapter_state_sizes.iter().sum();
        if total_size == 0 {
            return Vec::new();
        }

        let state = {
            let mut mem = self.memory.write().unwrap();
            mem.acquire_matrix_handle(total_size, 1, location, BufferPriority::High)
                .expect("AdapterStateStore: failed to allocate state buffer")
        };

        let buffer_idx = self.buffers.len();
        self.buffers.push(AdapterStateBuffer::new(state, location));

        let mut start = 0usize;
        let mut slices = Vec::with_capacity(adapter_state_sizes.len());
        for &size in adapter_state_sizes {
            slices.push(AdapterSlice::new(buffer_idx, start, size));
            start += size;
        }
        slices
    }

    /// Возвращает дескриптор буфера состояния для заданного слайса.
    ///
    /// # Паника
    /// Паникует, если `slice.buffer_idx` выходит за пределы вектора буферов.
    #[inline]
    pub fn state_handle(&self, slice: &AdapterSlice) -> &MatrixBufferHandle {
        &self.buffers[slice.buffer_idx].state
    }

    /// Возвращает ссылку на `AdapterStateBuffer` по слайсу.
    #[inline]
    pub fn get_buffer(&self, slice: &AdapterSlice) -> &AdapterStateBuffer {
        &self.buffers[slice.buffer_idx]
    }

    /// Возвращает мутабельную ссылку на `AdapterStateBuffer` по слайсу.
    #[inline]
    pub fn get_buffer_mut(&mut self, slice: &AdapterSlice) -> &mut AdapterStateBuffer {
        &mut self.buffers[slice.buffer_idx]
    }

    /// Возвращает ссылку на `AdapterStateBuffer` по индексу.
    #[inline]
    pub fn get_buffer_by_idx(&self, idx: usize) -> &AdapterStateBuffer {
        &self.buffers[idx]
    }

    /// Возвращает мутабельную ссылку по индексу.
    #[inline]
    pub fn get_buffer_by_idx_mut(&mut self, idx: usize) -> &mut AdapterStateBuffer {
        &mut self.buffers[idx]
    }

    /// Количество сегментов (буферов) в хранилище.
    pub fn num_buffers(&self) -> usize {
        self.buffers.len()
    }

    /// `true`, если store пуст.
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }

    /// Суммарный размер всех состояний в элементах f32.
    pub fn total_state_size(&self) -> usize {
        self.buffers.iter().map(|b| b.len()).sum()
    }

    /// Обновляет зафиксированное в store устройство после успешной
    /// миграции.
    ///
    /// Вызывается из `SmartDistributor::on_epoch_boundary` — единственного
    /// владельца фактической миграции. Сам store не мигрирует.
    pub fn set_location(&mut self, buffer_idx: usize, location: MemoryDeviceKind) {
        if let Some(b) = self.buffers.get_mut(buffer_idx) {
            b.location = location;
        }
    }

    /// Обнуляет все буферы состояния. Работает только с CPU-буферами;
    /// если какой-то буфер на GPU, паникует.
    ///
    /// Полезно для сброса состояния между сессиями (например, при
    /// повторном использовании одного графа).
    pub fn zero_all(&mut self) {
        for b in &self.buffers {
            assert!(
                !b.state.is_gpu(),
                "AdapterStateStore::zero_all currently supports only CPU buffers"
            );
            let mut guard = b.state.write();
            let slice = guard
                .as_slice_mut()
                .expect("AdapterStateStore: state must be CPU");
            for v in slice.iter_mut() {
                *v = 0.0;
            }
        }
    }
}

// ============================================================================
// Юнит-тесты
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_manager::core::device_spec::DeviceSpec;
    use crate::compute_manager::operators_v2::gpu_v2::detect_gpus;

    /// Создаёт `MemoryExecutor` с зарегистрированным CPU и (если есть)
    /// GPU. Возвращает `(executor, has_gpu, gpu_device_id)`.
    fn make_executor() -> (Arc<RwLock<MemoryExecutor>>, bool, Option<usize>) {
        let mem = Arc::new(RwLock::new(MemoryExecutor::new()));
        mem.write()
            .unwrap()
            .register_compute_device(DeviceSpec::cpu(0, 4096, 1), None);
        mem.write().unwrap().set_self_arc(mem.clone());

        let gpus = detect_gpus();
        let has_gpu = gpus.is_some();
        let gpu_device_id = if has_gpu { Some(0usize) } else { None };

        if let Some(_) = gpu_device_id {
            // Регистрируем GPU-устройство в пуле памяти, но контекст
            // можно не создавать: тест миграции работает через
            // `move_matrix_handle`, которому нужен только `GpuContext`
            // в `MemoryExecutor`. Чтобы не поднимать Vulkan в тесте,
            // ограничимся тестом CPU-пути.
        }

        (mem, false, None) // GPU-путь в этом тесте пропускается: см. ниже
    }

    #[test]
    fn empty_store_is_inert() {
        let (mem, _, _) = make_executor();
        let store = AdapterStateStore::new(mem);
        assert_eq!(store.num_buffers(), 0);
        assert_eq!(store.total_state_size(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn allocate_zero_size_is_noop() {
        let (mem, _, _) = make_executor();
        let mut store = AdapterStateStore::new(mem);
        let slices = store.allocate_segment(&[0, 0], MemoryDeviceKind::HostRam);
        assert!(slices.is_empty());
        assert_eq!(store.num_buffers(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn allocate_and_read_state_cpu() {
        let (mem, _, _) = make_executor();
        let mut store = AdapterStateStore::new(mem);

        // Сегмент с двумя адаптерами: [4, 8] элементов.
        let slices = store.allocate_segment(&[4, 8], MemoryDeviceKind::HostRam);
        assert_eq!(slices.len(), 2);
        assert_eq!(slices[0].buffer_idx, 0);
        assert_eq!(slices[0].start, 0);
        assert_eq!(slices[0].len, 4);
        assert_eq!(slices[1].buffer_idx, 0);
        assert_eq!(slices[1].start, 4);
        assert_eq!(slices[1].len, 8);

        assert_eq!(store.num_buffers(), 1);
        assert_eq!(store.total_state_size(), 12);

        // Записываем данные и читаем обратно.
        let h = store.state_handle(&slices[0]).clone();
        h.write_range(0, &[1.0, 2.0, 3.0, 4.0]);

        let h2 = store.state_handle(&slices[1]).clone();
        h2.write_range(4, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);

        let v0 = h.read_range(0, 4);
        assert_eq!(v0, vec![1.0, 2.0, 3.0, 4.0]);

        let v1 = store.state_handle(&slices[1]).read_range(4, 8);
        assert_eq!(v1, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    }

    #[test]
    fn zero_all_resets_cpu_buffers() {
        let (mem, _, _) = make_executor();
        let mut store = AdapterStateStore::new(mem);
        let _ = store.allocate_segment(&[4], MemoryDeviceKind::HostRam);

        let h = store.state_handle(&AdapterSlice::new(0, 0, 4)).clone();
        h.write_range(0, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(h.read_range(0, 4), vec![1.0, 2.0, 3.0, 4.0]);

        store.zero_all();
        assert_eq!(h.read_range(0, 4), vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn set_location_updates_local_metadata() {
        let (mem, _, _) = make_executor();
        let mut store = AdapterStateStore::new(mem);
        let _ = store.allocate_segment(&[4], MemoryDeviceKind::HostRam);

        assert_eq!(
            store.get_buffer_by_idx(0).location,
            MemoryDeviceKind::HostRam
        );

        // Имитируем миграцию: фактически не двигаем буфер, только
        // обновляем метаданные (это ровно то, что делает
        // SmartDistributor::on_epoch_boundary после успешной миграции).
        store.set_location(0, MemoryDeviceKind::SsdCache);
        assert_eq!(
            store.get_buffer_by_idx(0).location,
            MemoryDeviceKind::SsdCache
        );
    }

    // Тест миграции между RAM и VRAM требует поднятого `GpuContext` и
    // регистрации VRAM-пула в `MemoryExecutor`. Пропускается здесь,
    // чтобы `cargo test` не поднимал Vulkan на CI-машинах без GPU.
    // Полноценная валидация миграции adapter-state идёт через
    // существующие примеры (см. §7 Фаза 2 «прогон всех примеров»).
    //
    // Фактически store НЕ управляет миграцией — он получает уже
    // мигрированный хендл через `SmartDistributor::on_epoch_boundary`.
    // Значит, корректность миграции adapter-state идентична
    // корректности миграции `ParamStore` и покрывается уже
    // существующими тестами `MemoryExecutor::move_matrix_handle`.
}
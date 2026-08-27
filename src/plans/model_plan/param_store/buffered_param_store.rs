// src/plans/model_plan/param_store/buffered_param_store.rs

use std::sync::{Arc, Mutex};

use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::compute_manager::memory_executor::executor::MemoryExecutor;
use crate::compute_manager::memory_executor::matrix_entry::MatrixStorage;
use crate::compute_manager::memory_executor::policy::BufferPriority;
use crate::compute_manager::memory_executor::types::MemoryDeviceKind;

use super::slice::ParamSlice;

/// Полноценное матричное хранилище параметров модели.
///
/// Все параметры хранятся в виде одного столбца `(total_params, 1)` внутри
/// управляемого `MatrixBufferHandle`. Это даёт следующие преимущества:
/// - единая интеграция с `MemoryExecutor` и `TempMatrixPool`;
/// - быстрый доступ к диапазонам и отдельным элементам через прямые блокировки;
/// - автоматическое расширение при выделении новых слоёв;
/// - совместимость с оптимизатором, работающим на `MatrixBufferHandle`.
pub struct BufferedParamStore {
    /// Основной буфер параметров, размер `(capacity, 1)`.
    pub(crate) params: MatrixBufferHandle,

    /// Буфер градиентов, размер `(capacity, 1)`.
    pub(crate) grads: MatrixBufferHandle,

    /// Состояние оптимизатора, размер `(capacity * state_size_per_param, 1)`.
    /// Если `state_size_per_param == 0`, состояние отсутствует.
    pub(crate) opt_state: Option<MatrixBufferHandle>,

    /// Логическое количество параметров (может быть меньше ёмкости).
    pub(crate) total_params: usize,

    /// Текущий размер состояния оптимизатора на один параметр.
    pub(crate) state_size_per_param: usize,

    /// Ссылка на менеджер памяти, используется для выделения и прямого доступа.
    memory: Arc<Mutex<MemoryExecutor>>,
}

impl BufferedParamStore {
    /// Создаёт новое хранилище параметров с указанной начальной ёмкостью.
    ///
    /// # Аргументы
    /// * `memory` – глобальный менеджер памяти.
    /// * `initial_capacity` – начальная ёмкость в количестве параметров.
    /// * `state_size_per_param` – размер состояния оптимизатора на один параметр
    ///   (например, 0 для SGD, 1 для Momentum, 2 для Adam).
    ///
    /// # Паника
    /// Паникует, если не удалось выделить память под параметры или градиенты.
    pub fn new(
        memory: Arc<Mutex<MemoryExecutor>>,
        initial_capacity: usize,
        state_size_per_param: usize,
    ) -> Self {
        let params = allocate_handle(&memory, initial_capacity, 1)
            .expect("BufferedParamStore: failed to allocate params handle");
        let grads = allocate_handle(&memory, initial_capacity, 1)
            .expect("BufferedParamStore: failed to allocate grads handle");

        let opt_state = if state_size_per_param > 0 {
            let total_state = initial_capacity * state_size_per_param;
            Some(
                allocate_handle(&memory, total_state, 1)
                    .expect("BufferedParamStore: failed to allocate optimizer state handle"),
            )
        } else {
            None
        };

        Self {
            params,
            grads,
            opt_state,
            total_params: 0, // логически пока нет параметров, ёмкость уже выделена
            state_size_per_param,
            memory,
        }
    }

    /// Выделяет непрерывный блок параметров заданной длины.
    ///
    /// При необходимости автоматически расширяет внутренние буферы.
    /// Возвращает [`ParamSlice`], описывающий выделенный диапазон.
    pub fn allocate(&mut self, len: usize) -> ParamSlice {
        let start = self.total_params;
        let new_total = start + len;

        if new_total > self.params.rows() {
            // Увеличиваем ёмкость с запасом.
            let new_capacity = (new_total)
                .max(self.params.rows() * 2)
                .max(1);
            self.resize(new_capacity);
        }

        self.total_params = new_total;
        ParamSlice::new(start, len)
    }

    /// Возвращает логическое количество параметров.
    #[inline]
    pub fn len(&self) -> usize {
        self.total_params
    }

    /// Возвращает текущую ёмкость (выделенное количество строк).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.params.rows()
    }

    /// Возвращает `true`, если параметры отсутствуют.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_params == 0
    }

    /// Читает значение одного параметра по глобальному индексу.
    pub fn get(&self, index: usize) -> f32 {
        assert!(index < self.total_params, "BufferedParamStore::get: index out of bounds");
        let data = read_range(&self.memory, &self.params, index, 1);
        data[0]
    }

    /// Записывает значение одного параметра по глобальному индексу.
    pub fn set(&mut self, index: usize, value: f32) {
        assert!(index < self.total_params, "BufferedParamStore::set: index out of bounds");
        write_range(&self.memory, &self.params, index, &[value]);
    }

    /// Копирует параметры из указанного слайса в предоставленный буфер.
    ///
    /// # Паника
    /// Паникует, если `dest.len() != slice.len`.
    pub fn get_slice(&self, slice: ParamSlice, dest: &mut [f32]) {
        assert_eq!(dest.len(), slice.len, "BufferedParamStore::get_slice: destination length mismatch");
        let data = read_range(&self.memory, &self.params, slice.start, slice.len);
        dest.copy_from_slice(&data);
    }

    /// Записывает значения из буфера в указанный слайс.
    ///
    /// # Паника
    /// Паникует, если `src.len() != slice.len`.
    pub fn set_slice(&mut self, slice: ParamSlice, src: &[f32]) {
        assert_eq!(src.len(), slice.len, "BufferedParamStore::set_slice: source length mismatch");
        write_range(&self.memory, &self.params, slice.start, src);
    }

    /// Возвращает все параметры в виде плоского вектора.
    pub fn get_all_params(&self) -> Vec<f32> {
        read_range(&self.memory, &self.params, 0, self.total_params)
    }

    /// Устанавливает все параметры из плоского вектора.
    ///
    /// # Паника
    /// Паникует, если `values.len() != self.len()`.
    pub fn set_all_params(&mut self, values: &[f32]) {
        assert_eq!(values.len(), self.total_params, "BufferedParamStore::set_all_params: length mismatch");
        write_range(&self.memory, &self.params, 0, values);
    }

    /// Возвращает ссылку на дескриптор параметров.
    ///
    /// Этот метод предназначен для использования оптимизатором и другими
    /// компонентами, которые работают напрямую с `MatrixBufferHandle`.
    #[inline]
    pub fn params_handle(&self) -> &MatrixBufferHandle {
        &self.params
    }

    /// Возвращает ссылку на дескриптор градиентов.
    #[inline]
    pub fn grads_handle(&self) -> &MatrixBufferHandle {
        &self.grads
    }

    /// Возвращает ссылку на дескриптор состояния оптимизатора, если он есть.
    #[inline]
    pub fn state_handle(&self) -> Option<&MatrixBufferHandle> {
        self.opt_state.as_ref()
    }

    /// Обнуляет градиенты.
    pub fn zero_grads(&mut self) {
        let zeros = vec![0.0f32; self.total_params];
        write_range(&self.memory, &self.grads, 0, &zeros);
    }

    /// Добавляет градиенты из плоского вектора к уже накопленным.
    ///
    /// # Паника
    /// Паникует, если `grads.len() != self.len()`.
    pub fn accumulate_grads_from_slice(&mut self, grads: &[f32]) {
        assert_eq!(grads.len(), self.total_params, "BufferedParamStore::accumulate_grads_from_slice: length mismatch");
        accumulate_grads(&self.memory, &self.grads, grads);
    }

    /// Копирует градиенты из плоского вектора в управляемый буфер,
    /// **заменяя** предыдущее содержимое.
    ///
    /// # Паника
    /// Паникует, если `grads.len() != self.len()`.
    pub fn copy_grads_from_slice(&mut self, grads: &[f32]) {
        assert_eq!(grads.len(), self.total_params, "BufferedParamStore::copy_grads_from_slice: length mismatch");
        write_range(&self.memory, &self.grads, 0, grads);
    }

    /// Копирует градиенты из управляемого буфера в плоский вектор.
    ///
    /// # Паника
    /// Паникует, если `out.len() != self.len()`.
    pub fn copy_grads_to_slice(&self, out: &mut [f32]) {
        assert_eq!(out.len(), self.total_params, "BufferedParamStore::copy_grads_to_slice: length mismatch");
        let data = read_range(&self.memory, &self.grads, 0, self.total_params);
        out.copy_from_slice(&data);
    }

    /// Гарантирует, что состояние оптимизатора выделено и имеет правильный размер.
    ///
    /// Если `state_size_per_param` изменился или буфер состояния меньше требуемого,
    /// старое состояние сохраняется (насколько возможно) в новом буфере.
    pub fn ensure_opt_state(&mut self, state_size_per_param: usize) {
        self.state_size_per_param = state_size_per_param;

        if state_size_per_param == 0 {
            self.opt_state = None;
            return;
        }

        let required = self.params.rows() * state_size_per_param;
        match &self.opt_state {
            Some(state) if state.rows() >= required => {}
            _ => {
                let old_state = self
                    .opt_state
                    .as_ref()
                    .map(|s| read_range(&self.memory, s, 0, s.rows()))
                    .unwrap_or_default();

                let new_state = allocate_handle(&self.memory, required, 1)
                    .expect("BufferedParamStore: failed to allocate optimizer state handle");
                write_range(&self.memory, &new_state, 0, &old_state);

                self.opt_state = Some(new_state);
            }
        }
    }

    /// Возвращает текущий размер состояния на один параметр.
    #[inline]
    pub fn state_size_per_param(&self) -> usize {
        self.state_size_per_param
    }

    /// Внутренний метод: увеличивает ёмкость всех буферов до `new_capacity`.
    ///
    /// В отличие от предыдущей версии, этот метод расширяет существующие CPU‑векторы
    /// на месте, избегая одновременного существования старых и новых буферов.
    /// Память резервируется в `MemoryPool` только для разницы.
    fn resize(&mut self, new_capacity: usize) {
        let old_capacity = self.params.rows();
        if new_capacity <= old_capacity {
            return;
        }

        let delta = new_capacity - old_capacity;
        // Вычисляем, сколько дополнительной памяти нужно для params и grads.
        let delta_params_grads = delta * 2;
        // Для состояния: если оно уже существует, расширяем на delta * state_size_per_param.
        let delta_state = if self.opt_state.is_some() {
            delta * self.state_size_per_param
        } else {
            0
        };
        let total_delta = delta_params_grads + delta_state;

        // Шаг 1: зарезервировать дополнительную память через публичный метод MemoryExecutor.
        {
            let mut mem = self.memory.lock().unwrap();
            mem.reserve_memory(MemoryDeviceKind::HostRam, total_delta)
                .expect("BufferedParamStore::resize: insufficient HostRam for expansion");

            // Расширяем params
            {
                let params_id = self.params.id();
                let entry = mem.get_matrix_entry_mut(params_id)
                    .expect("params entry not found");
                if let MatrixStorage::Cpu(data) = &mut entry.storage {
                    data.resize(new_capacity, 0.0);
                    entry.rows = new_capacity;
                } else {
                    panic!("BufferedParamStore::resize: params storage must be CPU");
                }
            }

            // Расширяем grads
            {
                let grads_id = self.grads.id();
                let entry = mem.get_matrix_entry_mut(grads_id)
                    .expect("grads entry not found");
                if let MatrixStorage::Cpu(data) = &mut entry.storage {
                    data.resize(new_capacity, 0.0);
                    entry.rows = new_capacity;
                } else {
                    panic!("BufferedParamStore::resize: grads storage must be CPU");
                }
            }
        }

        // Шаг 2: обработать opt_state.
        let required_state_len = new_capacity * self.state_size_per_param;
        if self.state_size_per_param > 0 {
            if let Some(state_handle) = &self.opt_state {
                // Расширяем существующее состояние.
                let state_id = state_handle.id();
                let mut mem = self.memory.lock().unwrap();
                let entry = mem.get_matrix_entry_mut(state_id)
                    .expect("opt_state entry not found");
                if let MatrixStorage::Cpu(data) = &mut entry.storage {
                    data.resize(required_state_len, 0.0);
                    entry.rows = required_state_len;
                } else {
                    panic!("BufferedParamStore::resize: opt_state storage must be CPU");
                }
            } else {
                // Создаём новое состояние.
                let new_state = allocate_handle(&self.memory, required_state_len, 1)
                    .expect("BufferedParamStore::resize: failed to allocate opt_state");
                self.opt_state = Some(new_state);
            }
        } else {
            self.opt_state = None;
        }
    }
}

// ============================================================================
// Вспомогательные функции для прямого доступа к данным
// ============================================================================

/// Выделяет новый `MatrixBufferHandle` заданного размера `(rows, 1)` в HostRam.
fn allocate_handle(
    memory: &Arc<Mutex<MemoryExecutor>>,
    rows: usize,
    cols: usize,
) -> Result<MatrixBufferHandle, String> {
    let mut mem = memory.lock().unwrap();
    mem.acquire_matrix_handle(
        rows,
        cols,
        MemoryDeviceKind::HostRam,
        BufferPriority::High,
    )
    .map_err(|e| format!("{:?}", e))
}

/// Читает диапазон элементов из CPU-хранилища.
///
/// Для CPU-буферов выполняется прямое копирование из `MatrixStorage::Cpu`
/// без промежуточного полного копирования всего буфера.
/// Для GPU/SSD выполняется fallback через полное чтение `MatrixBufferHandle::read`.
fn read_range(
    memory: &Arc<Mutex<MemoryExecutor>>,
    handle: &MatrixBufferHandle,
    start: usize,
    len: usize,
) -> Vec<f32> {
    // Пытаемся получить прямой доступ к CPU-данным.
    {
        let mem = memory.lock().unwrap();
        if let Some(entry) = mem.get_matrix_entry(handle.id()) {
            if let MatrixStorage::Cpu(data) = &entry.storage {
                assert!(
                    start + len <= data.len(),
                    "read_range: range out of bounds (start={}, len={}, total={})",
                    start,
                    len,
                    data.len()
                );
                return data[start..start + len].to_vec();
            }
        }
    }

    // Fallback для GPU/SSD или любых других хранилищ.
    let guard = handle.read();
    let full = guard
        .as_slice()
        .expect("read_range: expected CPU-compatible buffer")
        .to_vec();
    assert!(
        start + len <= full.len(),
        "read_range: range out of bounds in fallback"
    );
    full[start..start + len].to_vec()
}

/// Записывает данные в диапазон элементов CPU-хранилища.
///
/// Для CPU-буферов выполняется прямая запись в `MatrixStorage::Cpu`.
/// Для GPU/SSD выполняется fallback через полное чтение и полную запись.
fn write_range(
    memory: &Arc<Mutex<MemoryExecutor>>,
    handle: &MatrixBufferHandle,
    start: usize,
    data: &[f32],
) {
    // Пытаемся получить прямой доступ к CPU-данным.
    {
        let mut mem = memory.lock().unwrap();
        if let Some(entry) = mem.get_matrix_entry_mut(handle.id()) {
            if let MatrixStorage::Cpu(store) = &mut entry.storage {
                assert!(
                    start + data.len() <= store.len(),
                    "write_range: range out of bounds (start={}, len={}, total={})",
                    start,
                    data.len(),
                    store.len()
                );
                store[start..start + data.len()].copy_from_slice(data);
                return;
            }
        }
    }

    // Fallback для GPU/SSD: полное чтение-модификация-запись.
    let mut guard = handle.write();
    let full = guard
        .as_slice_mut()
        .expect("write_range: expected CPU-compatible buffer");
    assert!(
        start + data.len() <= full.len(),
        "write_range: range out of bounds in fallback"
    );
    full[start..start + data.len()].copy_from_slice(data);
}

/// Накопление градиентов без лишних копирований, когда хранилище CPU.
fn accumulate_grads(
    memory: &Arc<Mutex<MemoryExecutor>>,
    grads_handle: &MatrixBufferHandle,
    src: &[f32],
) {
    // Прямой путь для CPU.
    {
        let mut mem = memory.lock().unwrap();
        if let Some(entry) = mem.get_matrix_entry_mut(grads_handle.id()) {
            if let MatrixStorage::Cpu(data) = &mut entry.storage {
                assert!(
                    src.len() <= data.len(),
                    "accumulate_grads: source length exceeds buffer capacity"
                );
                for (d, s) in data.iter_mut().zip(src.iter()) {
                    *d += *s;
                }
                return;
            }
        }
    }

    // Fallback для GPU/SSD.
    let mut guard = grads_handle.write();
    let full = guard
        .as_slice_mut()
        .expect("accumulate_grads: expected CPU-compatible buffer");
    for (d, s) in full.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_manager::device_spec::DeviceSpec;
    use crate::compute_manager::memory_executor::MemoryExecutor;

    fn create_memory() -> Arc<Mutex<MemoryExecutor>> {
        let mem = Arc::new(Mutex::new(MemoryExecutor::new()));
        mem.lock()
            .unwrap()
            .register_compute_device(DeviceSpec::cpu(0, 1024, 1), None);
        mem.lock().unwrap().set_self_arc(mem.clone());
        mem
    }

    #[test]
    fn test_allocate_and_access() {
        let mem = create_memory();
        let mut store = BufferedParamStore::new(mem.clone(), 4, 0);

        let s1 = store.allocate(3);
        assert_eq!(s1.start, 0);
        assert_eq!(s1.len, 3);
        assert_eq!(store.len(), 3);

        store.set_slice(s1, &[1.0, 2.0, 3.0]);

        let mut out = [0.0; 3];
        store.get_slice(s1, &mut out);
        assert_eq!(out, [1.0, 2.0, 3.0]);

        assert_eq!(store.get(1), 2.0);

        let s2 = store.allocate(2);
        assert_eq!(s2.start, 3);
        assert_eq!(s2.len, 2);
        assert_eq!(store.len(), 5);

        store.set_slice(s2, &[4.0, 5.0]);
        assert_eq!(store.get_all_params(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_resize_triggers_expansion() {
        let mem = create_memory();
        let mut store = BufferedParamStore::new(mem.clone(), 2, 0);

        let s1 = store.allocate(2);
        store.set_slice(s1, &[1.0, 2.0]);
        assert_eq!(store.capacity(), 2);

        let s2 = store.allocate(3);
        assert!(store.capacity() >= 5);
        store.set_slice(s2, &[3.0, 4.0, 5.0]);

        assert_eq!(store.get_all_params(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_grads_and_accumulate() {
        let mem = create_memory();
        let mut store = BufferedParamStore::new(mem.clone(), 4, 0);
        store.allocate(4);

        store.zero_grads();
        store.accumulate_grads_from_slice(&[0.5, -0.5, 1.0, 2.0]);

        let mut grads = [0.0; 4];
        store.copy_grads_to_slice(&mut grads);
        assert_eq!(grads, [0.5, -0.5, 1.0, 2.0]);

        store.accumulate_grads_from_slice(&[0.1, 0.2, 0.3, 0.4]);
        store.copy_grads_to_slice(&mut grads);
        assert!((grads[0] - 0.6).abs() < 1e-6);
        assert!((grads[1] + 0.3).abs() < 1e-6);
        assert!((grads[2] - 1.3).abs() < 1e-6);
        assert!((grads[3] - 2.4).abs() < 1e-6);
    }

    #[test]
    fn test_opt_state_allocation() {
        let mem = create_memory();
        let mut store = BufferedParamStore::new(mem.clone(), 4, 0);
        store.allocate(4);
        store.ensure_opt_state(2);

        let state = store.state_handle().expect("state should be allocated");
        assert_eq!(state.rows(), store.capacity() * 2);
        assert_eq!(state.cols(), 1);
        assert_eq!(store.state_size_per_param(), 2);
    }
}
// src/plans/model_plan/param_store/param_store.rs

use std::sync::{Arc, RwLock};

use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::compute_manager::memory_executor::executor::MemoryExecutor;
use crate::compute_manager::memory_executor::policy::BufferPriority;
use crate::compute_manager::memory_executor::types::MemoryDeviceKind;

use super::param_buffer::ParamBuffer;
use super::slice::ParamSlice;

/// Менеджер параметров модели, организованных по сегментам.
///
/// Каждый сегмент модели (UniversalProcessor, Splitter, Combiner) имеет
/// собственный [`ParamBuffer`], в котором хранятся все параметры этого
/// сегмента. Градиенты и (опционально) состояние оптимизатора также
/// содержатся в этом же буфере или отдельном, но связанном.
///
/// Такая структура позволяет независимо перемещать параметры разных
/// сегментов между устройствами (CPU, GPU, SSD) в соответствии с
/// текущим размещением вычислений.
pub struct ParamStore {
    /// Вектор буферов параметров. Индекс соответствует `buffer_idx`
    /// в структуре [`ParamSlice`].
    buffers: Vec<ParamBuffer>,

    /// Глобальный менеджер памяти, используемый для выделения новых буферов.
    memory: Arc<RwLock<MemoryExecutor>>,
}

impl ParamStore {
    /// Создаёт пустое хранилище параметров.
    pub fn new(memory: Arc<RwLock<MemoryExecutor>>) -> Self {
        Self {
            buffers: Vec::new(),
            memory,
        }
    }

    /// Выделяет параметры для нового сегмента с заданными размерами слоёв.
    ///
    /// Создаёт один [`ParamBuffer`], содержащий параметры всех слоёв сегмента,
    /// и возвращает вектор [`ParamSlice`], по одному на каждый слой.
    /// Все слайсы ссылаются на один и тот же `buffer_idx`, но имеют разные
    /// смещения внутри буфера.
    ///
    /// # Аргументы
    /// * `layer_sizes` – количество параметров для каждого слоя сегмента.
    /// * `location` – устройство, на котором будут размещены параметры и градиенты.
    ///
    /// # Возвращает
    /// Вектор [`ParamSlice`] длиной `layer_sizes.len()`. Если суммарное
    /// количество параметров равно нулю (все слои без параметров), возвращается
    /// пустой вектор, и новый буфер не создаётся.
    pub fn allocate_segment(
        &mut self,
        layer_sizes: &[usize],
        location: MemoryDeviceKind,
    ) -> Vec<ParamSlice> {
        let total_params: usize = layer_sizes.iter().sum();
        if total_params == 0 {
            return Vec::new();
        }

        // Выделяем буферы параметров и градиентов одним блоком.
        let params = self
            .allocate_matrix_handle(total_params, 1, location, BufferPriority::High)
            .expect("ParamStore: failed to allocate params for segment");
        let grads = self
            .allocate_matrix_handle(total_params, 1, location, BufferPriority::High)
            .expect("ParamStore: failed to allocate grads for segment");

        let buffer_idx = self.buffers.len();
        self.buffers.push(ParamBuffer::new(params, grads, location));

        // Нарезаем слайсы для каждого слоя.
        let mut start = 0usize;
        let mut slices = Vec::with_capacity(layer_sizes.len());
        for &size in layer_sizes {
            slices.push(ParamSlice::new(buffer_idx, start, size));
            start += size;
        }
        slices
    }

    /// Возвращает дескриптор буфера параметров для заданного слайса.
    ///
    /// # Паника
    /// Паникует, если `slice.buffer_idx` выходит за пределы вектора буферов.
    #[inline]
    pub fn params_handle(&self, slice: &ParamSlice) -> &MatrixBufferHandle {
        &self.buffers[slice.buffer_idx].params
    }

    /// Возвращает дескриптор буфера градиентов для заданного слайса.
    #[inline]
    pub fn grads_handle(&self, slice: &ParamSlice) -> &MatrixBufferHandle {
        &self.buffers[slice.buffer_idx].grads
    }

    /// Возвращает дескриптор состояния оптимизатора, если оно выделено.
    #[inline]
    pub fn state_handle(&self, slice: &ParamSlice) -> Option<&MatrixBufferHandle> {
        self.buffers[slice.buffer_idx].opt_state.as_ref()
    }

    /// Гарантирует наличие состояния оптимизатора нужного размера для буфера,
    /// соответствующего слайсу.
    ///
    /// Если `state_size_per_param == 0`, состояние удаляется.
    /// В противном случае при отсутствии или недостаточном размере
    /// старое состояние заменяется новым, инициализированным нулями.
    ///
    /// # Аргументы
    /// * `slice` – слайс, определяющий целевой буфер.
    /// * `state_size_per_param` – требуемый размер состояния на один параметр.
    pub fn ensure_opt_state(&mut self, slice: &ParamSlice, state_size_per_param: usize) {
        if state_size_per_param == 0 {
            if let Some(buffer) = self.buffers.get_mut(slice.buffer_idx) {
                buffer.clear_opt_state();
            }
            return;
        }

        let (required, need_recreate, location) = {
            let buffer = &self.buffers[slice.buffer_idx];
            let required = buffer.params.rows() * state_size_per_param;
            let need_recreate = match &buffer.opt_state {
                Some(state) => state.rows() < required,
                None => true,
            };
            (required, need_recreate, buffer.location)
        };

        if need_recreate {
            let new_state = self
                .allocate_matrix_handle(required, 1, location, BufferPriority::Medium)
                .expect("ParamStore: failed to allocate optimizer state");
            let buffer = &mut self.buffers[slice.buffer_idx];
            buffer.set_opt_state(new_state);
        }
    }

    /// Возвращает ссылку на [`ParamBuffer`] по слайсу.
    #[inline]
    pub fn get_param_buffer(&self, slice: &ParamSlice) -> &ParamBuffer {
        &self.buffers[slice.buffer_idx]
    }

    /// Возвращает мутабельную ссылку на [`ParamBuffer`] по слайсу.
    #[inline]
    pub fn get_param_buffer_mut(&mut self, slice: &ParamSlice) -> &mut ParamBuffer {
        &mut self.buffers[slice.buffer_idx]
    }

    /// Возвращает ссылку на [`ParamBuffer`] по индексу буфера.
    #[inline]
    pub fn get_param_buffer_by_idx(&self, idx: usize) -> &ParamBuffer {
        &self.buffers[idx]
    }

    /// Возвращает общее количество параметров по всем сегментам.
    ///
    /// Суммирует `rows()` буферов параметров (так как каждый буфер хранит
    /// параметры в одном столбце).
    pub fn total_params(&self) -> usize {
        self.buffers.iter().map(|b| b.params.rows()).sum()
    }

    /// Возвращает количество сегментов (буферов) в хранилище.
    pub fn num_buffers(&self) -> usize {
        self.buffers.len()
    }

    /// Проверяет, есть ли хотя бы один буфер.
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }

    /// Обнуляет градиенты во всех буферах.
    ///
    /// Работает только для CPU‑буферов. Если какой‑либо буфер находится на GPU,
    /// метод запаникует.
    pub fn zero_grads(&mut self) {
        for buffer in &self.buffers {
            assert!(
                !buffer.grads.is_gpu(),
                "ParamStore::zero_grads currently supports only CPU buffers"
            );
            let mut guard = buffer.grads.write();
            let slice = guard.as_slice_mut().expect("grads must be CPU");
            for v in slice.iter_mut() {
                *v = 0.0;
            }
        }
    }

    /// Собирает все градиенты в единый вектор.
    ///
    /// Порядок соответствует порядку буферов в хранилище.
    /// Работает только для CPU‑буферов.
    pub fn copy_grads_to_slice(&self, out: &mut [f32]) {
        let total = self.total_params();
        assert_eq!(
            out.len(),
            total,
            "ParamStore::copy_grads_to_slice: output length {} does not match total params {}",
            out.len(),
            total
        );

        let mut offset = 0usize;
        for buffer in &self.buffers {
            let guard = buffer.grads.read();
            let src = guard.as_slice().expect("grads must be CPU");
            let len = src.len();
            out[offset..offset + len].copy_from_slice(src);
            offset += len;
        }
    }

    /// Устанавливает значения всех параметров из единого вектора.
    ///
    /// Порядок значений должен соответствовать порядку буферов и внутреннему
    /// расположению параметров. Работает только для CPU‑буферов.
    pub fn set_all_params(&mut self, values: &[f32]) {
        let total = self.total_params();
        assert_eq!(
            values.len(),
            total,
            "ParamStore::set_all_params: input length {} does not match total params {}",
            values.len(),
            total
        );

        let mut offset = 0usize;
        for buffer in &self.buffers {
            assert!(
                !buffer.params.is_gpu(),
                "ParamStore::set_all_params currently supports only CPU buffers"
            );
            let mut guard = buffer.params.write();
            let dst = guard.as_slice_mut().expect("params must be CPU");
            let len = dst.len();
            dst.copy_from_slice(&values[offset..offset + len]);
            offset += len;
        }
    }

    /// Возвращает все параметры в виде плоского вектора (CPU).
    pub fn get_all_params(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.total_params());
        for buffer in &self.buffers {
            let guard = buffer.params.read();
            let src = guard.as_slice().expect("params must be CPU");
            result.extend_from_slice(src);
        }
        result
    }

    // -----------------------------------------------------------------------
    // Приватные вспомогательные методы
    // -----------------------------------------------------------------------

    /// Выделяет новый управляемый буфер через `MemoryExecutor`.
    fn allocate_matrix_handle(
        &self,
        rows: usize,
        cols: usize,
        location: MemoryDeviceKind,
        priority: BufferPriority,
    ) -> Result<MatrixBufferHandle, String> {
        let mut mem = self.memory.write().unwrap();
        mem.acquire_matrix_handle(rows, cols, location, priority)
            .map_err(|e| format!("{:?}", e))
    }
}
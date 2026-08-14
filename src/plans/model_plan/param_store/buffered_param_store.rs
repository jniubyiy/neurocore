// src/plans/model_plan/param_store/buffered_param_store.rs

use std::sync::{Arc, Mutex};

use crate::compute_manager::memory_executor::policy::BufferPriority;
use crate::compute_manager::memory_executor::types::MemoryDeviceKind;
use crate::compute_manager::memory_executor::MemoryExecutor;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

/// Буферизованное хранилище параметров, градиентов и состояния оптимизатора,
/// использующее лёгкие дескрипторы [`MatrixBufferHandle`].
///
/// Все данные находятся в управляемой памяти `MemoryExecutor` и могут
/// участвовать в автоматическом перемещении между уровнями памяти.
pub struct BufferedParamStore {
    /// Параметры модели в виде вектор-столбца `(num_params, 1)`.
    pub(crate) params: MatrixBufferHandle,

    /// Градиенты параметров в виде вектор-столбца `(num_params, 1)`.
    pub(crate) grads: MatrixBufferHandle,

    /// Состояние оптимизатора (если цепочка требует память).
    /// Размер `(num_params * state_size_per_param, 1)`.
    /// В новом оптимизаторе состояния хранятся отдельно в `OptimizerExpr`,
    /// поэтому это поле может оставаться неиспользуемым, но сохранено
    /// для обратной совместимости.
    pub(crate) opt_state: Option<MatrixBufferHandle>,

    /// Количество параметров.
    pub(crate) num_params: usize,

    /// Суммарный размер состояния на один параметр.
    pub(crate) state_size_per_param: usize,

    /// Глобальный менеджер памяти.
    pub(crate) memory: Arc<Mutex<MemoryExecutor>>,
}

impl BufferedParamStore {
    /// Создаёт новое CPU‑хранилище параметров и градиентов на основе `MatrixBufferHandle`.
    ///
    /// # Аргументы
    /// * `memory` — `MemoryExecutor`, через который выделяется управляемая память.
    /// * `num_params` — количество параметров модели.
    /// * `state_size_per_param` — суммарный размер состояния оптимизатора
    ///   на один параметр (например, 1 для momentum, 2 для Adam, 0 для SGD).
    ///
    /// # Паника
    /// Паникует, если не удалось выделить память.
    pub fn new_cpu(
        memory: Arc<Mutex<MemoryExecutor>>,
        num_params: usize,
        state_size_per_param: usize,
    ) -> Self {
        let params = {
            let mut mem = memory.lock().unwrap();
            mem.acquire_matrix_handle(
                num_params,
                1,
                MemoryDeviceKind::HostRam,
                BufferPriority::High,
            )
            .expect("BufferedParamStore: failed to allocate params handle")
        };

        let grads = {
            let mut mem = memory.lock().unwrap();
            mem.acquire_matrix_handle(
                num_params,
                1,
                MemoryDeviceKind::HostRam,
                BufferPriority::High,
            )
            .expect("BufferedParamStore: failed to allocate grads handle")
        };

        let opt_state = if state_size_per_param > 0 {
            let total_state_elems = num_params * state_size_per_param;
            let mut mem = memory.lock().unwrap();
            Some(
                mem.acquire_matrix_handle(
                    total_state_elems,
                    1,
                    MemoryDeviceKind::HostRam,
                    BufferPriority::High,
                )
                .expect("BufferedParamStore: failed to allocate optimizer state handle"),
            )
        } else {
            None
        };

        Self {
            params,
            grads,
            opt_state,
            num_params,
            state_size_per_param,
            memory,
        }
    }

    /// Возвращает дескриптор параметров.
    #[inline]
    pub fn params_handle(&self) -> &MatrixBufferHandle {
        &self.params
    }

    /// Возвращает мутабельный дескриптор параметров.
    #[inline]
    pub fn params_handle_mut(&mut self) -> &mut MatrixBufferHandle {
        &mut self.params
    }

    /// Возвращает дескриптор градиентов.
    #[inline]
    pub fn grads_handle(&self) -> &MatrixBufferHandle {
        &self.grads
    }

    /// Возвращает мутабельный дескриптор градиентов.
    #[inline]
    pub fn grads_handle_mut(&mut self) -> &mut MatrixBufferHandle {
        &mut self.grads
    }

    /// Возвращает дескриптор состояния оптимизатора, если он есть.
    #[inline]
    pub fn state_handle(&self) -> Option<&MatrixBufferHandle> {
        self.opt_state.as_ref()
    }

    /// Возвращает мутабельный дескриптор состояния оптимизатора, если он есть.
    #[inline]
    pub fn state_handle_mut(&mut self) -> Option<&mut MatrixBufferHandle> {
        self.opt_state.as_mut()
    }

    /// Количество параметров.
    #[inline]
    pub fn num_params(&self) -> usize {
        self.num_params
    }

    /// Суммарный размер состояния на один параметр.
    #[inline]
    pub fn state_size_per_param(&self) -> usize {
        self.state_size_per_param
    }

    /// Копирует параметры из плоского среза в управляемый буфер.
    ///
    /// # Паника
    /// Паникует, если `params.len() != self.num_params`.
    pub fn copy_params_from_slice(&mut self, params: &[f32]) {
        assert_eq!(
            params.len(),
            self.num_params,
            "BufferedParamStore::copy_params_from_slice: length mismatch"
        );
        let mut guard = self.params.write();
        let dst = guard.as_slice_mut().expect("BufferedParamStore: params must be CPU");
        dst.copy_from_slice(params);
    }

    /// Копирует параметры из управляемого буфера в плоский срез.
    ///
    /// # Паника
    /// Паникует, если `out.len() != self.num_params`.
    pub fn copy_params_to_slice(&self, out: &mut [f32]) {
        assert_eq!(
            out.len(),
            self.num_params,
            "BufferedParamStore::copy_params_to_slice: length mismatch"
        );
        let guard = self.params.read();
        let src = guard.as_slice().expect("BufferedParamStore: params must be CPU");
        out.copy_from_slice(src);
    }

    /// Копирует градиенты из плоского среза в управляемый буфер.
    ///
    /// # Паника
    /// Паникует, если `grads.len() != self.num_params`.
    pub fn copy_grads_from_slice(&mut self, grads: &[f32]) {
        assert_eq!(
            grads.len(),
            self.num_params,
            "BufferedParamStore::copy_grads_from_slice: length mismatch"
        );
        let mut guard = self.grads.write();
        let dst = guard.as_slice_mut().expect("BufferedParamStore: grads must be CPU");
        dst.copy_from_slice(grads);
    }

    /// Копирует градиенты из управляемого буфера в плоский срез.
    ///
    /// # Паника
    /// Паникует, если `out.len() != self.num_params`.
    pub fn copy_grads_to_slice(&self, out: &mut [f32]) {
        assert_eq!(
            out.len(),
            self.num_params,
            "BufferedParamStore::copy_grads_to_slice: length mismatch"
        );
        let guard = self.grads.read();
        let src = guard.as_slice().expect("BufferedParamStore: grads must be CPU");
        out.copy_from_slice(src);
    }

    /// Обнуляет градиенты.
    #[inline]
    pub fn zero_grads(&mut self) {
        let mut guard = self.grads.write();
        let slice = guard.as_slice_mut().expect("BufferedParamStore: grads must be CPU");
        slice.fill(0.0);
    }

    /// Добавляет градиенты из плоского среза к уже накопленным.
    ///
    /// # Паника
    /// Паникует, если `grads.len() != self.num_params`.
    pub fn accumulate_grads_from_slice(&mut self, grads: &[f32]) {
        assert_eq!(
            grads.len(),
            self.num_params,
            "BufferedParamStore::accumulate_grads_from_slice: length mismatch"
        );
        let mut guard = self.grads.write();
        let dst = guard.as_slice_mut().expect("BufferedParamStore: grads must be CPU");
        for (d, &s) in dst.iter_mut().zip(grads.iter()) {
            *d += s;
        }
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
        mem
    }

    #[test]
    fn test_new_cpu() {
        let mem = create_memory();
        let store = BufferedParamStore::new_cpu(mem.clone(), 10, 0);
        assert_eq!(store.num_params(), 10);
        assert_eq!(store.params_handle().rows(), 10);
        assert_eq!(store.params_handle().cols(), 1);
        assert_eq!(store.grads_handle().rows(), 10);
        assert_eq!(store.grads_handle().cols(), 1);
        assert!(store.state_handle().is_none());
    }

    #[test]
    fn test_copy_params() {
        let mem = create_memory();
        let mut store = BufferedParamStore::new_cpu(mem.clone(), 5, 0);
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        store.copy_params_from_slice(&data);

        let mut out = [0.0; 5];
        store.copy_params_to_slice(&mut out);
        assert_eq!(out, data);
    }

    #[test]
    fn test_grads_zero_and_accumulate() {
        let mem = create_memory();
        let mut store = BufferedParamStore::new_cpu(mem.clone(), 4, 0);
        store.zero_grads();
        store.accumulate_grads_from_slice(&[0.5, -0.5, 1.0, 2.0]);

        let mut grads_out = [0.0; 4];
        store.copy_grads_to_slice(&mut grads_out);
        assert!((grads_out[0] - 0.5).abs() < 1e-6);
        assert!((grads_out[1] + 0.5).abs() < 1e-6);
        assert!((grads_out[2] - 1.0).abs() < 1e-6);
        assert!((grads_out[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_state_allocation() {
        let mem = create_memory();
        let store = BufferedParamStore::new_cpu(mem.clone(), 4, 2);
        let state = store.state_handle().expect("state should be allocated");
        assert_eq!(state.rows(), 8);
        assert_eq!(state.cols(), 1);
    }
}
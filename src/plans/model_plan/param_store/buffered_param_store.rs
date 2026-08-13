// src/plans/model_plan/param_store/buffered_param_store.rs

use std::sync::{Arc, Mutex};

use crate::compute_manager::matrix_buffer::MatrixBuffer;
use crate::compute_manager::memory_executor::policy::BufferPriority;
use crate::compute_manager::memory_executor::types::MemoryDeviceKind;
use crate::compute_manager::memory_executor::MemoryExecutor;

/// Буферизованное хранилище параметров, градиентов и состояния оптимизатора.
///
/// Все данные хранятся в управляемых [`MatrixBuffer`], которые регистрируются
/// в [`MemoryExecutor`] и могут участвовать в управлении памятью.
///
/// Это хранилище создаётся параллельно со старым `ParamStore` и не заменяет его
/// на данном этапе. Оно предназначено для постепенного перевода вычислений
/// на управляемую память.
pub struct BufferedParamStore {
    /// Параметры модели в виде вектор-столбца `(num_params, 1)`.
    pub(crate) params: MatrixBuffer,

    /// Градиенты параметров в виде вектор-столбца `(num_params, 1)`.
    pub(crate) grads: MatrixBuffer,

    /// Состояние оптимизатора, если цепочка требует память.
    /// Размер `(num_params * state_size_per_param, 1)`.
    pub(crate) opt_state: Option<MatrixBuffer>,

    /// Количество параметров.
    pub(crate) num_params: usize,

    /// Суммарный размер состояния на один параметр.
    pub(crate) state_size_per_param: usize,

    /// Глобальный менеджер памяти.
    pub(crate) memory: Arc<Mutex<MemoryExecutor>>,
}

impl BufferedParamStore {
    /// Создаёт новое CPU‑хранилище параметров и градиентов.
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
        let mut params = MatrixBuffer::new(&memory, num_params, 1)
            .expect("BufferedParamStore: failed to allocate params matrix");
        let mut grads = MatrixBuffer::new(&memory, num_params, 1)
            .expect("BufferedParamStore: failed to allocate grads matrix");

        // Регистрируем буферы в реестре MemoryExecutor
        {
            let mut mem = memory.lock().unwrap();
            let params_id = mem.register_matrix(
                num_params,
                1,
                MemoryDeviceKind::HostRam,
                BufferPriority::High,
            );
            let grads_id = mem.register_matrix(
                num_params,
                1,
                MemoryDeviceKind::HostRam,
                BufferPriority::High,
            );
            params.set_matrix_id(Some(params_id));
            grads.set_matrix_id(Some(grads_id));
        }

        let opt_state = if state_size_per_param > 0 {
            let total_state_elems = num_params * state_size_per_param;
            let mut state = MatrixBuffer::new(&memory, total_state_elems, 1)
                .expect("BufferedParamStore: failed to allocate optimizer state");
            let mut mem = memory.lock().unwrap();
            let state_id = mem.register_matrix(
                total_state_elems,
                1,
                MemoryDeviceKind::HostRam,
                BufferPriority::High,
            );
            state.set_matrix_id(Some(state_id));
            Some(state)
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

    /// Возвращает матрицу параметров.
    #[inline]
    pub fn params_matrix(&self) -> &MatrixBuffer {
        &self.params
    }

    /// Возвращает мутабельную матрицу параметров.
    #[inline]
    pub fn params_matrix_mut(&mut self) -> &mut MatrixBuffer {
        &mut self.params
    }

    /// Возвращает матрицу градиентов.
    #[inline]
    pub fn grads_matrix(&self) -> &MatrixBuffer {
        &self.grads
    }

    /// Возвращает мутабельную матрицу градиентов.
    #[inline]
    pub fn grads_matrix_mut(&mut self) -> &mut MatrixBuffer {
        &mut self.grads
    }

    /// Возвращает состояние оптимизатора, если оно есть.
    #[inline]
    pub fn state_matrix(&self) -> Option<&MatrixBuffer> {
        self.opt_state.as_ref()
    }

    /// Возвращает мутабельное состояние оптимизатора, если оно есть.
    #[inline]
    pub fn state_matrix_mut(&mut self) -> Option<&mut MatrixBuffer> {
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
        let mut mat = self.params.as_mat_mut();
        for i in 0..self.num_params {
            mat[(i, 0)] = params[i];
        }
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
        let mat = self.params.to_mat();
        for i in 0..self.num_params {
            out[i] = mat[(i, 0)];
        }
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
        let mut mat = self.grads.as_mat_mut();
        for i in 0..self.num_params {
            mat[(i, 0)] = grads[i];
        }
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
        let mat = self.grads.to_mat();
        for i in 0..self.num_params {
            out[i] = mat[(i, 0)];
        }
    }

    /// Обнуляет градиенты.
    #[inline]
    pub fn zero_grads(&mut self) {
        self.grads.fill(0.0);
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
        let mut mat = self.grads.as_mat_mut();
        for i in 0..self.num_params {
            mat[(i, 0)] += grads[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_manager::device_spec::DeviceSpec;
    use crate::compute_manager::memory_executor::MemoryExecutor;
    use crate::optimizer_plan::cubes::{ApplyUpdate, ScaleGradient};
    use crate::optimizer_plan::chain::OptimizerChain;

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
        assert_eq!(store.params.rows(), 10);
        assert_eq!(store.params.cols(), 1);
        assert_eq!(store.grads.rows(), 10);
        assert_eq!(store.grads.cols(), 1);
        assert!(store.opt_state.is_none());
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

        let grads = store.grads.to_mat();
        assert!((grads[(0, 0)] - 0.5).abs() < 1e-6);
        assert!((grads[(1, 0)] + 0.5).abs() < 1e-6);
        assert!((grads[(2, 0)] - 1.0).abs() < 1e-6);
        assert!((grads[(3, 0)] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_optimizer_step_buffered() {
        let mem = create_memory();
        let num_params = 4;
        let mut store = BufferedParamStore::new_cpu(mem.clone(), num_params, 0);

        store.copy_params_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        store.copy_grads_from_slice(&[0.1, 0.2, 0.3, 0.4]);

        let chain = OptimizerChain::new()
            .add(Box::new(ScaleGradient::new(0.1)))
            .add(Box::new(ApplyUpdate));

        // Состояния нет, поэтому создаём временное пустое состояние.
        let mut empty_state = MatrixBuffer::new(&mem, 0, 1).unwrap();
        chain.apply_all_buffered(&mut store.params, &mut store.grads, &mut empty_state);

        let updated = store.params.to_mat();
        assert!((updated[(0, 0)] - 0.99).abs() < 1e-6);
        assert!((updated[(1, 0)] - 1.98).abs() < 1e-6);
        assert!((updated[(2, 0)] - 2.97).abs() < 1e-6);
        assert!((updated[(3, 0)] - 3.96).abs() < 1e-6);
    }
}
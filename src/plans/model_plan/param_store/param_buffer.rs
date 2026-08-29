// src/plans/model_plan/param_store/param_buffer.rs

use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::compute_manager::memory_executor::types::MemoryDeviceKind;
use std::fmt;

/// Контейнер параметров одного сегмента модели.
///
/// Содержит дескрипторы управляемых буферов для параметров, градиентов
/// и (опционально) состояния оптимизатора, а также информацию о текущем
/// устройстве памяти, на котором размещены данные.
///
/// Все буферы, как правило, находятся на одном устройстве, что упрощает
/// выполнение вычислений и миграцию между CPU/GPU/SSD.
pub struct ParamBuffer {
    /// Параметры сегмента (размер `num_params × 1`).
    pub params: MatrixBufferHandle,

    /// Градиенты параметров (размер `num_params × 1`).
    pub grads: MatrixBufferHandle,

    /// Состояние оптимизатора (размер `num_params * state_size × 1`).
    /// `None`, если оптимизатор не требует состояния или оно ещё не выделено.
    pub opt_state: Option<MatrixBufferHandle>,

    /// Текущее устройство размещения всех буферов.
    pub location: MemoryDeviceKind,
}

impl ParamBuffer {
    /// Создаёт новый контейнер параметров с указанными буферами параметров
    /// и градиентов, размещёнными на заданном устройстве.
    ///
    /// Состояние оптимизатора изначально отсутствует.
    ///
    /// # Аргументы
    /// * `params` – дескриптор буфера параметров.
    /// * `grads`  – дескриптор буфера градиентов.
    /// * `location` – устройство памяти, на котором находятся оба буфера.
    pub fn new(
        params: MatrixBufferHandle,
        grads: MatrixBufferHandle,
        location: MemoryDeviceKind,
    ) -> Self {
        Self {
            params,
            grads,
            opt_state: None,
            location,
        }
    }

    /// Проверяет, выделено ли состояние оптимизатора.
    pub fn has_opt_state(&self) -> bool {
        self.opt_state.is_some()
    }

    /// Возвращает дескриптор состояния оптимизатора, если оно выделено.
    pub fn opt_state_handle(&self) -> Option<&MatrixBufferHandle> {
        self.opt_state.as_ref()
    }

    /// Возвращает дескриптор состояния оптимизатора для мутабельного доступа.
    pub fn opt_state_handle_mut(&mut self) -> Option<&mut MatrixBufferHandle> {
        self.opt_state.as_mut()
    }

    /// Устанавливает дескриптор состояния оптимизатора.
    pub fn set_opt_state(&mut self, handle: MatrixBufferHandle) {
        self.opt_state = Some(handle);
    }

    /// Очищает состояние оптимизатора (например, при смене его размера).
    pub fn clear_opt_state(&mut self) {
        self.opt_state = None;
    }
}

// Реализуем Debug вручную, чтобы не требовать Debug от MatrixBufferHandle и
// MemoryExecutor. Выводим только идентификаторы буферов и расположение.
impl fmt::Debug for ParamBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParamBuffer")
            .field("params_id", &self.params.id())
            .field("grads_id", &self.grads.id())
            .field(
                "opt_state_id",
                &self.opt_state.as_ref().map(|h| h.id()),
            )
            .field("location", &self.location)
            .finish()
    }
}
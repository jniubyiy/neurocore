// src/compute_manager/memory_executor/matrix_id.rs

use std::fmt;

/// Уникальный идентификатор управляемого матричного буфера.
///
/// Используется `MemoryExecutor` для отслеживания метаданных
/// и жизненного цикла `MatrixBuffer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatrixBufferId(pub usize);

impl fmt::Display for MatrixBufferId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mtx{}", self.0)
    }
}
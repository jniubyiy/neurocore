// src/compute_manager/matrix_buffer/weak_handle.rs

use std::sync::{Arc, RwLock};

use crate::compute_manager::memory_executor::executor::MemoryExecutor;
use crate::compute_manager::memory_executor::matrix_id::MatrixBufferId;

use super::handle::MatrixBufferHandle;

/// Слабая ссылка на управляемый матричный буфер.
pub struct WeakMatrixBufferHandle {
    pub(crate) id: MatrixBufferId,
    pub(crate) memory: Arc<RwLock<MemoryExecutor>>,
}

impl WeakMatrixBufferHandle {
    pub fn from_handle(handle: &MatrixBufferHandle) -> Self {
        Self {
            id: handle.id(),
            memory: handle.memory().clone(),
        }
    }

    #[inline]
    pub fn id(&self) -> MatrixBufferId {
        self.id
    }

    pub fn upgrade(&self) -> Option<MatrixBufferHandle> {
        let mut mem = self.memory.write().unwrap();
        let entry = mem.get_matrix_entry(self.id)?;

        if entry.ref_count == 0 {
            return None;
        }

        mem.increment_ref_count(self.id);
        Some(MatrixBufferHandle::new(self.id, self.memory.clone()))
    }

    pub fn is_alive(&self) -> bool {
        let mem = self.memory.read().unwrap();
        mem.get_matrix_entry(self.id)
            .map(|entry| entry.ref_count > 0)
            .unwrap_or(false)
    }
}

impl Clone for WeakMatrixBufferHandle {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            memory: self.memory.clone(),
        }
    }
}

impl std::fmt::Debug for WeakMatrixBufferHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeakMatrixBufferHandle")
            .field("id", &self.id)
            .finish()
    }
}
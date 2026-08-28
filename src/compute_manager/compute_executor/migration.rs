// src/compute_manager/compute_executor/migration.rs

use crate::compute_manager::memory_executor::MemoryExecutor;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::compute_manager::memory_executor::types::MemoryDeviceKind;
use crate::compute_manager::graph::types::Segment;
use crate::compute_manager::compute_executor::placement::SegmentPlacement;

/// Перемещает один буфер на целевое устройство хранения.
///
/// Использует `MemoryExecutor::move_matrix_handle` для переноса данных
/// между CPU, GPU и SSD. Если буфер уже находится на целевом устройстве,
/// операция не выполняется.
pub fn migrate_buffer_to_device(
    memory_executor: &mut MemoryExecutor,
    buffer: &MatrixBufferHandle,
    target_kind: MemoryDeviceKind,
) -> Result<(), String> {
    let current_kind = buffer.device_kind();
    if current_kind == target_kind {
        return Ok(());
    }

    memory_executor
        .move_matrix_handle(buffer.id(), target_kind)
        .map_err(|e| format!("Не удалось переместить буфер {} в {:?}: {:?}", buffer.id(), target_kind, e))
}

/// Перемещает несколько буферов на целевое устройство.
pub fn migrate_buffers_to_device(
    memory_executor: &mut MemoryExecutor,
    buffers: &[MatrixBufferHandle],
    target_kind: MemoryDeviceKind,
) -> Result<(), String> {
    for buf in buffers {
        migrate_buffer_to_device(memory_executor, buf, target_kind)?;
    }
    Ok(())
}

/// Выполняет миграцию данных для сегментов в соответствии с новым размещением.
///
/// В текущей архитектуре параметры модели хранятся в общем CPU‑буфере,
/// поэтому перенос на уровне сегментов не требуется. Функция оставлена
/// для будущих версий, когда у сегментов появятся собственные буферы.
pub fn migrate_segments(
    _segments: &[Segment],
    _new_placement: &[SegmentPlacement],
    _memory_executor: &std::sync::Arc<std::sync::Mutex<MemoryExecutor>>,
) {
    // В данной версии не выполняется: параметры общие, миграция не нужна.
}
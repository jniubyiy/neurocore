// src/plans/model_plan/adapter_store/buffer.rs

use crate::compute_manager::operators_v2::memory_v2::buffer::MatrixBufferHandle;
use crate::compute_manager::operators_v2::memory_v2::types::MemoryDeviceKind;
use std::fmt;

/// Контейнер состояния одного адаптера сегмента.
///
/// # Назначение
///
/// Адаптер слоя (трейт `GradientAdapter`, Фаза 3 плана) может иметь
/// персистентное состояние между шагами обучения: накопленные
/// статистики, счётчики, вспомогательные буферы и т. п. Это состояние
/// не является ни параметром модели, ни его градиентом — оно живёт
/// «рядом» и сохраняется/мигрирует вместе с параметрами сегмента.
///
/// Инвариант I-6 (MIGRATION_PLAN.md §2): это состояние — сестринское
/// к параметрам, но хранится в отдельном `AdapterStateStore`. Optimizer
/// его не трогает. Обновление — только через `GradientAdapter::apply`.
///
/// # Раскладка
///
/// На один сегмент — один буфер (вектор-столбец). Суммарный размер
/// равен сумме `state_size_per_param * num_params` всех адаптеров
/// сегмента. Если состояние адаптера составное (несколько векторов
/// разной семантики), адаптер сам решает, как упаковать их в один
/// непрерывный диапазон.
///
/// Если у сегмента нет ни одного адаптера с состоянием —
/// `AdapterStateStore::allocate_segment` не создаёт буфер вовсе
/// (см. `store.rs`).
pub struct AdapterStateBuffer {
    /// Буфер состояния (вектор-столбец).
    ///
    /// Всегда `rows × 1`. Если `rows == 0` — у адаптеров сегмента
    /// нет состояния (в текущей реализации такое не возникает:
    /// для пустого сегмента буфер вообще не создаётся; но
    /// поле остаётся на будущее).
    pub state: MatrixBufferHandle,

    /// Текущее устройство размещения.
    ///
    /// Держим локально, чтобы не читать из `MemoryExecutor` при каждом
    /// обращении к store. Синхронизируется с фактическим storage на
    /// момент `AdapterStateStore::allocate_segment` и обновляется через
    /// `AdapterStateStore::set_location` после успешной миграции.
    pub location: MemoryDeviceKind,
}

impl AdapterStateBuffer {
    /// Создаёт новый контейнер состояния.
    pub fn new(state: MatrixBufferHandle, location: MemoryDeviceKind) -> Self {
        Self { state, location }
    }

    /// Размер состояния в элементах f32.
    #[inline]
    pub fn len(&self) -> usize {
        self.state.rows() * self.state.cols()
    }

    /// `true`, если состояние пустое (нет элементов).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// AdapterStateBuffer не реализует Debug автоматически, потому что
// MatrixBufferHandle его не реализует. Пишем вручную — этого достаточно
// для диагностических сообщений.
impl fmt::Debug for AdapterStateBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdapterStateBuffer")
            .field("state_id", &self.state.id())
            .field("len", &self.len())
            .field("location", &self.location)
            .finish()
    }
}
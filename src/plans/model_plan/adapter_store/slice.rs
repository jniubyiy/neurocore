// src/plans/model_plan/adapter_store/slice.rs

/// Дескриптор непрерывного участка состояния адаптера внутри
/// конкретного буфера `AdapterStateStore`.
///
/// Параллель `ParamSlice`: не владеет данными, а только описывает
/// расположение. Формат идентичен — это упрощает совместную работу
/// `ParamStore` и `AdapterStateStore` в графе и распределителе.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AdapterSlice {
    /// Индекс буфера в `AdapterStateStore::buffers`.
    pub buffer_idx: usize,
    /// Начальный индекс (смещение) внутри выбранного буфера.
    pub start: usize,
    /// Длина участка (количество элементов f32).
    pub len: usize,
}

impl AdapterSlice {
    /// Создаёт новый дескриптор участка состояния адаптера.
    #[inline]
    pub fn new(buffer_idx: usize, start: usize, len: usize) -> Self {
        Self { buffer_idx, start, len }
    }

    /// Возвращает конечный индекс (исключительный).
    #[inline]
    pub fn end(&self) -> usize {
        self.start + self.len
    }

    /// Проверяет, пуст ли участок.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Проверяет, содержится ли данный индекс внутри участка.
    #[inline]
    pub fn contains(&self, index: usize) -> bool {
        index >= self.start && index < self.end()
    }

    /// Возвращает индекс буфера.
    #[inline]
    pub fn buffer_idx(&self) -> usize {
        self.buffer_idx
    }
}
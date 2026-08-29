// src/plans/model_plan/param_store/slice.rs

/// Дескриптор непрерывного участка параметров внутри конкретного буфера.
///
/// Теперь параметры хранятся в нескольких буферах (по одному на сегмент),
/// поэтому слайс содержит индекс буфера (`buffer_idx`) и смещение (`start`)
/// внутри этого буфера.
///
/// Не владеет данными, а только описывает их расположение.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamSlice {
    /// Индекс буфера в `ParamStore::buffers`.
    pub buffer_idx: usize,
    /// Начальный индекс (смещение) внутри выбранного буфера.
    pub start: usize,
    /// Длина участка (количество параметров).
    pub len: usize,
}

impl ParamSlice {
    /// Создаёт новый дескриптор участка параметров.
    ///
    /// # Аргументы
    /// * `buffer_idx` – индекс буфера в хранилище.
    /// * `start` – начальный индекс внутри буфера.
    /// * `len` – количество параметров в участке.
    #[inline]
    pub fn new(buffer_idx: usize, start: usize, len: usize) -> Self {
        Self { buffer_idx, start, len }
    }

    /// Возвращает конечный индекс (исключительный) внутри буфера.
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
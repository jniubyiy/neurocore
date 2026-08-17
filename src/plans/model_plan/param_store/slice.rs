// src/plans/model_plan/param_store/slice.rs

/// Дескриптор непрерывного участка параметров.
///
/// Используется для указания диапазона индексов в общем хранилище параметров.
/// Не владеет данными, а только описывает их расположение.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamSlice {
    /// Начальный индекс в глобальном массиве параметров.
    pub start: usize,
    /// Длина участка (количество параметров).
    pub len: usize,
}

impl ParamSlice {
    /// Создаёт новый дескриптор участка параметров.
    ///
    /// # Аргументы
    /// * `start` – начальный индекс.
    /// * `len` – количество параметров в участке.
    #[inline]
    pub fn new(start: usize, len: usize) -> Self {
        Self { start, len }
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
}
// src/layers/memory/memory.rs

use std::sync::Mutex;

use crate::layers::UniversalLayer;

/// Состояние слоя Memory для CPU-вычислительного пути.
///
/// Хранит по два якоря на каждый признак:
/// - `min_cells[c]` — минимальное наблюдаемое значение признака `c`;
/// - `max_cells[c]` — максимальное наблюдаемое значение признака `c`.
///
/// Якоря обновляются плавно с коэффициентом `alpha`: при наблюдении
/// значения за пределами текущего диапазона соответствующий якорь
/// сдвигается на долю `alpha` от разрыва. При попадании значения внутрь
/// диапазона обновляются оба якоря (сдвигаются навстречу значению).
///
/// Флаг `initialized` указывает, была ли выполнена первичная инициализация
/// якорей по первому образцу батча. До инициализации значения `min_cells`
/// и `max_cells` не используются и равны нулю.
pub(crate) struct MemoryState {
    pub(crate) min_cells: Vec<f32>,
    pub(crate) max_cells: Vec<f32>,
    pub(crate) initialized: bool,
}

/// Слой Memory.
///
/// Слой выполняет сжатие активаций к ближайшему из двух обучаемых «якорей»
/// (минимальному или максимальному), наблюдаемых для каждого признака.
/// Используется для адаптивного сглаживания и очистки сигнала от шума.
///
/// Формула для каждого элемента `x`:
///   `closest = argmin_{a ∈ {min, max}} |x - a|`
///   `y = x + alpha * (closest - x)`
///
/// Якоря обновляются во время прямого прохода:
/// - если `x > max`, то `max ← max + alpha * (x - max)`;
/// - если `x < min`, то `min ← min + alpha * (x - min)`;
/// - иначе `min ← min + alpha * (x - min)` и `max ← max + alpha * (x - max)`.
///
/// Вход и выход имеют одинаковую размерность `features`.
/// Слой не имеет обучаемых параметров в общем `ParamStore` —
/// якоря хранятся внутри самого слоя как часть его вычислительного состояния.
pub struct Memory {
    pub(crate) features: usize,
    pub alpha: f32,
    pub(crate) state: Mutex<MemoryState>,
}

impl Memory {
    /// Создаёт новый слой Memory.
    ///
    /// # Аргументы
    /// * `in_features` — количество входных признаков.
    /// * `out_features` — количество выходных признаков. Должно совпадать с `in_features`.
    ///
    /// # Паника
    /// Паникует, если `in_features != out_features`.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        assert_eq!(
            in_features, out_features,
            "Memory: in_features must equal out_features"
        );
        Self {
            features: in_features,
            alpha: 0.1,
            state: Mutex::new(MemoryState {
                min_cells: vec![0.0; in_features],
                max_cells: vec![0.0; in_features],
                initialized: false,
            }),
        }
    }
}

impl UniversalLayer for Memory {
    fn as_memory(&self) -> Option<&Memory> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        0
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}
// src/layers/memory/memory.rs

use crate::layers::UniversalLayer;

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
/// Слой не имеет обучаемых параметров в общем `ParamStore`.
///
/// Состояние (якоря min_cells/max_cells) не хранится в структуре слоя —
/// оно создаётся per-chunk в `forward_buffered` и передаётся через
/// `BufferedContext::Memory`.
pub struct Memory {
    pub(crate) features: usize,
    pub alpha: f32,
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
// src/losses/cross_entropy/mod.rs

use std::any::Any;
use crate::losses::ElemCube;

/// Кросс-энтропия с логитами.
///
/// Принимает матрицу размера `(batch, num_classes + 1)`, где первые `num_classes`
/// столбцов — логиты (предсказания модели), а последний столбец содержит
/// индекс правильного класса (как `f32`, который приводится к `usize`).
///
/// Возвращает матрицу `(batch, 1)` со значениями потерь для каждого сэмпла.
pub struct CrossEntropyWithLogits {
    pub num_classes: usize,
}

impl CrossEntropyWithLogits {
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }
}

impl ElemCube for CrossEntropyWithLogits {
    fn in_features(&self) -> usize {
        self.num_classes + 1   // логиты + индекс класса
    }

    fn out_features(&self) -> usize {
        1
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// src/losses/sum_columns/mod.rs

use std::any::Any;
use crate::losses::ElemCube;

/// Суммирует все столбцы входной матрицы, превращая её в вектор (batch, 1).
#[derive(Debug)]
pub struct SumColumns;

impl ElemCube for SumColumns {
    fn in_features(&self) -> usize { 0 }
    fn out_features(&self) -> usize { 1 }
    fn as_any(&self) -> &dyn Any { self }
}

mod cpu;

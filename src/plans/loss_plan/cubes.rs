// src/plans/loss_plan/cubes.rs

use std::any::Any;
use std::fmt::Debug;
use faer::Mat;

/// Элементарный кубик функции потерь (матричная версия).
pub trait ElemCube: Any + Send + Sync + Debug {
    fn in_features(&self) -> usize;
    fn out_features(&self) -> usize;
    fn forward_batch(&self, input: &Mat<f32>) -> Mat<f32>;
    fn backward_batch(
        &self,
        input: &Mat<f32>,
        output_cache: &Mat<f32>,
        grad_out: &Mat<f32>,
    ) -> Mat<f32>;
    fn as_any(&self) -> &dyn Any;
}

// ----------------------------------------------------------------
// Простейшие кубики, поддерживающие векторные признаки
// ----------------------------------------------------------------

#[derive(Debug)]
pub struct Sub {
    /// Количество признаков предсказания (равно количеству признаков цели)
    features: usize,
}

impl Sub {
    pub fn new(pred_features: usize) -> Self {
        assert!(pred_features > 0, "Sub: pred_features must be positive");
        Self { features: pred_features }
    }
}

impl Default for Sub {
    fn default() -> Self {
        Self { features: 1 } // обратная совместимость
    }
}

impl ElemCube for Sub {
    fn in_features(&self) -> usize {
        2 * self.features
    }

    fn out_features(&self) -> usize {
        self.features
    }

    fn forward_batch(&self, input: &Mat<f32>) -> Mat<f32> {
        let batch = input.nrows();
        let f = self.features;
        Mat::from_fn(batch, f, |i, j| {
            input[(i, j)] - input[(i, j + f)]
        })
    }

    fn backward_batch(
        &self,
        _input: &Mat<f32>,
        _cache: &Mat<f32>,
        grad_out: &Mat<f32>,
    ) -> Mat<f32> {
        let batch = grad_out.nrows();
        let f = self.features;
        Mat::from_fn(batch, 2 * f, |i, j| {
            let g = grad_out[(i, j % f)];
            if j < f { g } else { -g }
        })
    }

    fn as_any(&self) -> &dyn Any { self }
}

#[derive(Debug)]
pub struct Square;

impl ElemCube for Square {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }

    fn forward_batch(&self, input: &Mat<f32>) -> Mat<f32> {
        input.map(|x| x * x)
    }

    fn backward_batch(
        &self,
        input: &Mat<f32>,
        _cache: &Mat<f32>,
        grad_out: &Mat<f32>,
    ) -> Mat<f32> {
        let batch = grad_out.nrows();
        let cols = input.ncols();
        Mat::from_fn(batch, cols, |i, j| {
            2.0 * input[(i, j)] * grad_out[(i, j)]
        })
    }

    fn as_any(&self) -> &dyn Any { self }
}

/// Суммирует все столбцы в каждой строке, превращая (batch, features) в (batch, 1)
#[derive(Debug)]
pub struct SumColumns;

impl ElemCube for SumColumns {
    fn in_features(&self) -> usize { 0 } // фактически не важно, т.к. применяется к матрице с произвольным числом столбцов
    fn out_features(&self) -> usize { 1 }

    fn forward_batch(&self, input: &Mat<f32>) -> Mat<f32> {
        let batch = input.nrows();
        let cols = input.ncols();
        Mat::from_fn(batch, 1, |i, _| {
            let mut sum = 0.0;
            for j in 0..cols {
                sum += input[(i, j)];
            }
            sum
        })
    }

    fn backward_batch(
        &self,
        _input: &Mat<f32>,
        _cache: &Mat<f32>,
        grad_out: &Mat<f32>,
    ) -> Mat<f32> {
        let batch = grad_out.nrows();
        let cols = _input.ncols(); // нужно знать исходное число столбцов; передадим его через input
        let mut grad = Mat::zeros(batch, cols);
        for i in 0..batch {
            let g = grad_out[(i, 0)];
            for j in 0..cols {
                grad[(i, j)] = g;
            }
        }
        grad
    }

    fn as_any(&self) -> &dyn Any { self }
}

// Остальные кубики оставлены без изменений, так как они работают с матрицами
// и будут применяться после SumColumns, когда матрица уже стала (batch,1).

#[derive(Debug)]
pub struct Log;
impl ElemCube for Log {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }

    fn forward_batch(&self, input: &Mat<f32>) -> Mat<f32> {
        input.map(|x| x.ln())
    }

    fn backward_batch(
        &self,
        input: &Mat<f32>,
        _cache: &Mat<f32>,
        grad_out: &Mat<f32>,
    ) -> Mat<f32> {
        let batch = grad_out.nrows();
        Mat::from_fn(batch, 1, |i, _| {
            grad_out[(i, 0)] / input[(i, 0)]
        })
    }
    fn as_any(&self) -> &dyn Any { self }
}

#[derive(Debug)]
pub struct Neg;
impl ElemCube for Neg {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }

    fn forward_batch(&self, input: &Mat<f32>) -> Mat<f32> {
        -input
    }

    fn backward_batch(
        &self,
        _input: &Mat<f32>,
        _cache: &Mat<f32>,
        grad_out: &Mat<f32>,
    ) -> Mat<f32> {
        -grad_out
    }
    fn as_any(&self) -> &dyn Any { self }
}

#[derive(Debug)]
pub struct Mul {
    /// Количество признаков предсказания (равно количеству признаков цели)
    features: usize,
}

impl Mul {
    pub fn new(pred_features: usize) -> Self {
        assert!(pred_features > 0, "Mul: pred_features must be positive");
        Self { features: pred_features }
    }
}

impl Default for Mul {
    fn default() -> Self {
        Self { features: 1 }
    }
}

impl ElemCube for Mul {
    fn in_features(&self) -> usize {
        2 * self.features
    }

    fn out_features(&self) -> usize {
        self.features
    }

    fn forward_batch(&self, input: &Mat<f32>) -> Mat<f32> {
        let batch = input.nrows();
        let f = self.features;
        Mat::from_fn(batch, f, |i, j| {
            input[(i, j)] * input[(i, j + f)]
        })
    }

    fn backward_batch(
        &self,
        input: &Mat<f32>,
        _cache: &Mat<f32>,
        grad_out: &Mat<f32>,
    ) -> Mat<f32> {
        let batch = grad_out.nrows();
        let f = self.features;
        Mat::from_fn(batch, 2 * f, |i, j| {
            let g = grad_out[(i, j % f)];
            if j < f {
                g * input[(i, j + f)]
            } else {
                g * input[(i, j - f)]
            }
        })
    }

    fn as_any(&self) -> &dyn Any { self }
}

#[derive(Debug)]
pub struct Abs;
impl ElemCube for Abs {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }

    fn forward_batch(&self, input: &Mat<f32>) -> Mat<f32> {
        input.map(|x| x.abs())
    }

    fn backward_batch(
        &self,
        input: &Mat<f32>,
        _cache: &Mat<f32>,
        grad_out: &Mat<f32>,
    ) -> Mat<f32> {
        let batch = grad_out.nrows();
        let cols = input.ncols();
        Mat::from_fn(batch, cols, |i, j| {
            let x = input[(i, j)];
            let g = grad_out[(i, j)];
            if x > 0.0 { g } else if x < 0.0 { -g } else { 0.0 }
        })
    }
    fn as_any(&self) -> &dyn Any { self }
}

#[derive(Debug)]
pub struct AddScalar(pub f32);
impl ElemCube for AddScalar {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }

    fn forward_batch(&self, input: &Mat<f32>) -> Mat<f32> {
        input.map(|x| x + self.0)
    }

    fn backward_batch(
        &self,
        _input: &Mat<f32>,
        _cache: &Mat<f32>,
        grad_out: &Mat<f32>,
    ) -> Mat<f32> {
        grad_out.to_owned()
    }
    fn as_any(&self) -> &dyn Any { self }
}

#[derive(Debug)]
pub struct Log1p;
impl ElemCube for Log1p {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }

    fn forward_batch(&self, input: &Mat<f32>) -> Mat<f32> {
        input.map(|x| (x + 1.0).ln())
    }

    fn backward_batch(
        &self,
        input: &Mat<f32>,
        _cache: &Mat<f32>,
        grad_out: &Mat<f32>,
    ) -> Mat<f32> {
        let batch = grad_out.nrows();
        let cols = input.ncols();
        Mat::from_fn(batch, cols, |i, j| {
            grad_out[(i, j)] / (1.0 + input[(i, j)])
        })
    }
    fn as_any(&self) -> &dyn Any { self }
}

#[derive(Debug)]
pub struct AbsDiff {
    features: usize,
}

impl AbsDiff {
    pub fn new(pred_features: usize) -> Self {
        assert!(pred_features > 0, "AbsDiff: pred_features must be positive");
        Self { features: pred_features }
    }
}

impl Default for AbsDiff {
    fn default() -> Self {
        Self { features: 1 }
    }
}

impl ElemCube for AbsDiff {
    fn in_features(&self) -> usize {
        2 * self.features
    }

    fn out_features(&self) -> usize {
        self.features
    }

    fn forward_batch(&self, input: &Mat<f32>) -> Mat<f32> {
        let batch = input.nrows();
        let f = self.features;
        Mat::from_fn(batch, f, |i, j| {
            (input[(i, j)] - input[(i, j + f)]).abs()
        })
    }

    fn backward_batch(
        &self,
        input: &Mat<f32>,
        _cache: &Mat<f32>,
        grad_out: &Mat<f32>,
    ) -> Mat<f32> {
        let batch = grad_out.nrows();
        let f = self.features;
        Mat::from_fn(batch, 2 * f, |i, j| {
            let diff = input[(i, j % f)] - input[(i, j % f + f)];
            let g = grad_out[(i, j % f)];
            let grad = if diff > 0.0 { g } else if diff < 0.0 { -g } else { 0.0 };
            if j < f { grad } else { -grad }
        })
    }

    fn as_any(&self) -> &dyn Any { self }
}
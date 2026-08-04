// src/loss_plan/cubes.rs

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
// Простейшие кубики (Debug реализован через derive)
// ----------------------------------------------------------------

#[derive(Debug)]
pub struct Sub;
impl ElemCube for Sub {
    fn in_features(&self) -> usize { 2 }
    fn out_features(&self) -> usize { 1 }

    fn forward_batch(&self, input: &Mat<f32>) -> Mat<f32> {
        let a = input.subcols(0, 1);
        let b = input.subcols(1, 1);
        &a - &b
    }

    fn backward_batch(
        &self,
        _input: &Mat<f32>,
        _cache: &Mat<f32>,
        grad_out: &Mat<f32>,
    ) -> Mat<f32> {
        let batch = grad_out.nrows();
        Mat::from_fn(batch, 2, |i, j| {
            let g = grad_out[(i, 0)];
            if j == 0 { g } else { -g }
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
        Mat::from_fn(batch, 1, |i, _| {
            2.0 * input[(i, 0)] * grad_out[(i, 0)]
        })
    }
    fn as_any(&self) -> &dyn Any { self }
}

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
pub struct Mul;
impl ElemCube for Mul {
    fn in_features(&self) -> usize { 2 }
    fn out_features(&self) -> usize { 1 }

    fn forward_batch(&self, input: &Mat<f32>) -> Mat<f32> {
        let batch = input.nrows();
        Mat::from_fn(batch, 1, |i, _| {
            input[(i, 0)] * input[(i, 1)]
        })
    }

    fn backward_batch(
        &self,
        input: &Mat<f32>,
        _cache: &Mat<f32>,
        grad_out: &Mat<f32>,
    ) -> Mat<f32> {
        let batch = grad_out.nrows();
        Mat::from_fn(batch, 2, |i, j| {
            let g = grad_out[(i, 0)];
            if j == 0 { g * input[(i, 1)] } else { g * input[(i, 0)] }
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
        Mat::from_fn(batch, 1, |i, _| {
            let g = grad_out[(i, 0)];
            let x = input[(i, 0)];
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
        Mat::from_fn(batch, 1, |i, _| {
            grad_out[(i, 0)] / (1.0 + input[(i, 0)])
        })
    }
    fn as_any(&self) -> &dyn Any { self }
}

#[derive(Debug)]
pub struct AbsDiff;
impl ElemCube for AbsDiff {
    fn in_features(&self) -> usize { 2 }
    fn out_features(&self) -> usize { 1 }

    fn forward_batch(&self, input: &Mat<f32>) -> Mat<f32> {
        let batch = input.nrows();
        Mat::from_fn(batch, 1, |i, _| {
            (input[(i, 0)] - input[(i, 1)]).abs()
        })
    }

    fn backward_batch(
        &self,
        input: &Mat<f32>,
        _cache: &Mat<f32>,
        grad_out: &Mat<f32>,
    ) -> Mat<f32> {
        let batch = grad_out.nrows();
        Mat::from_fn(batch, 2, |i, j| {
            let diff = input[(i, 0)] - input[(i, 1)];
            let g = grad_out[(i, 0)];
            let grad = if diff > 0.0 { g } else if diff < 0.0 { -g } else { 0.0 };
            if j == 0 { grad } else { -grad }
        })
    }
    fn as_any(&self) -> &dyn Any { self }
}
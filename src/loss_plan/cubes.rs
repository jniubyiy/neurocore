// src/loss_plan/cubes.rs

use faer::Mat;

/// Элементарный кубик функции потерь (матричная версия).
///
/// Каждый кубик принимает на вход матрицу размера `(batch, in_features)`
/// и возвращает матрицу размера `(batch, out_features)`.
/// Прямой проход вычисляет выход, обратный — градиенты по входам,
/// используя переданные матрицы.
pub trait ElemCube: Send + Sync {
    /// Число столбцов входной матрицы (признаки на одну задачу).
    fn in_features(&self) -> usize;

    /// Число столбцов выходной матрицы.
    fn out_features(&self) -> usize;

    /// Прямой проход: `input` размер `(batch, in_features)` → `(batch, out_features)`.
    fn forward_batch(&self, input: &Mat<f32>) -> Mat<f32>;

    /// Обратный проход:
    /// `input` – входная матрица (такая же, как при прямом проходе),
    /// `output_cache` – результат прямого прохода (может игнорироваться),
    /// `grad_out` – пришедший градиент `(batch, out_features)`.
    /// Возвращает градиент по входу `(batch, in_features)`.
    fn backward_batch(
        &self,
        input: &Mat<f32>,
        output_cache: &Mat<f32>,
        grad_out: &Mat<f32>,
    ) -> Mat<f32>;
}

// ----------------------------------------------------------------
// Простейшие кубики
// ----------------------------------------------------------------

/// Вычитание двух чисел: `input[:,0] - input[:,1]`.
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
}

/// Квадрат числа: `x²`.
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
}

/// Натуральный логарифм: `ln(x)`.
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
}

/// Унарный минус: `-x`.
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
}

/// Умножение двух чисел: `input[:,0] * input[:,1]`.
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
}

/// Абсолютное значение: `|x|`.
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
}

/// Прибавление константы: `x + scalar`.
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
}

/// `ln(1 + x)`.
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
}

/// Модуль разности двух чисел: `|a - b|`.
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
}
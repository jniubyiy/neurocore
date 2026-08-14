// src/plans/loss_plan/cubes.rs

use std::any::Any;
use std::fmt::Debug;
use faer::Mat;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

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

/// Буферизованный элементарный кубик функции потерь.
/// Работает с управляемыми буферами `MatrixBufferHandle` (CPU).
pub trait BufferedElemCube: Send + Sync + Debug {
    fn in_features(&self) -> usize;
    fn out_features(&self) -> usize;
    fn forward_buffered(&self, input: &MatrixBufferHandle, output: &mut MatrixBufferHandle);
    fn backward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output_cache: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        grad_in: &mut MatrixBufferHandle,
    );
}

// ----------------------------------------------------------------
// Sub
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

impl BufferedElemCube for Sub {
    fn in_features(&self) -> usize {
        2 * self.features
    }

    fn out_features(&self) -> usize {
        self.features
    }

    fn forward_buffered(&self, input: &MatrixBufferHandle, output: &mut MatrixBufferHandle) {
        let rows = input.rows();
        let f = self.features;

        let src_guard = input.read();
        let src = src_guard.as_slice().expect("Sub forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("Sub forward: expected CPU buffer");

        debug_assert_eq!(src.len(), rows * 2 * f);
        debug_assert_eq!(dst.len(), rows * f);

        for r in 0..rows {
            for c in 0..f {
                dst[c * rows + r] = src[c * rows + r] - src[(c + f) * rows + r];
            }
        }
    }

    fn backward_buffered(
        &self,
        _input: &MatrixBufferHandle,
        _output_cache: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        grad_in: &mut MatrixBufferHandle,
    ) {
        let rows = grad_out.rows();
        let f = self.features;

        let go_guard = grad_out.read();
        let go = go_guard.as_slice().expect("Sub backward: expected CPU buffer");

        let mut gi_guard = grad_in.write();
        let gi = gi_guard.as_slice_mut().expect("Sub backward: expected CPU buffer");

        debug_assert_eq!(go.len(), rows * f);
        debug_assert_eq!(gi.len(), rows * 2 * f);

        for r in 0..rows {
            for c in 0..f {
                let g = go[c * rows + r];
                gi[c * rows + r] = g;
                gi[(c + f) * rows + r] = -g;
            }
        }
    }
}

// ----------------------------------------------------------------
// Square
// ----------------------------------------------------------------

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

impl BufferedElemCube for Square {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }

    fn forward_buffered(&self, input: &MatrixBufferHandle, output: &mut MatrixBufferHandle) {
        let src_guard = input.read();
        let src = src_guard.as_slice().expect("Square forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("Square forward: expected CPU buffer");

        debug_assert_eq!(src.len(), dst.len());

        for (o, &x) in dst.iter_mut().zip(src.iter()) {
            *o = x * x;
        }
    }

    fn backward_buffered(
        &self,
        input: &MatrixBufferHandle,
        _output_cache: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        grad_in: &mut MatrixBufferHandle,
    ) {
        let x_guard = input.read();
        let x = x_guard.as_slice().expect("Square backward: expected CPU buffer");

        let go_guard = grad_out.read();
        let go = go_guard.as_slice().expect("Square backward: expected CPU buffer");

        let mut gi_guard = grad_in.write();
        let gi = gi_guard.as_slice_mut().expect("Square backward: expected CPU buffer");

        debug_assert_eq!(x.len(), go.len());
        debug_assert_eq!(x.len(), gi.len());

        for i in 0..x.len() {
            gi[i] = 2.0 * x[i] * go[i];
        }
    }
}

// ----------------------------------------------------------------
// SumColumns
// ----------------------------------------------------------------

#[derive(Debug)]
pub struct SumColumns;

impl ElemCube for SumColumns {
    fn in_features(&self) -> usize { 0 }
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
        let cols = _input.ncols();
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

impl BufferedElemCube for SumColumns {
    fn in_features(&self) -> usize { 0 }
    fn out_features(&self) -> usize { 1 }

    fn forward_buffered(&self, input: &MatrixBufferHandle, output: &mut MatrixBufferHandle) {
        let rows = input.rows();
        let cols = input.cols();

        let src_guard = input.read();
        let src = src_guard.as_slice().expect("SumColumns forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("SumColumns forward: expected CPU buffer");

        debug_assert_eq!(src.len(), rows * cols);
        debug_assert_eq!(dst.len(), rows);

        for r in 0..rows {
            let mut sum = 0.0;
            for c in 0..cols {
                sum += src[c * rows + r];
            }
            dst[r] = sum;
        }
    }

    fn backward_buffered(
        &self,
        input: &MatrixBufferHandle,
        _output_cache: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        grad_in: &mut MatrixBufferHandle,
    ) {
        let rows = grad_out.rows();
        let cols = input.cols();

        let go_guard = grad_out.read();
        let go = go_guard.as_slice().expect("SumColumns backward: expected CPU buffer");

        let mut gi_guard = grad_in.write();
        let gi = gi_guard.as_slice_mut().expect("SumColumns backward: expected CPU buffer");

        debug_assert_eq!(go.len(), rows);
        debug_assert_eq!(gi.len(), rows * cols);

        for r in 0..rows {
            let g = go[r];
            for c in 0..cols {
                gi[c * rows + r] = g;
            }
        }
    }
}

// ----------------------------------------------------------------
// Log
// ----------------------------------------------------------------

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

impl BufferedElemCube for Log {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }

    fn forward_buffered(&self, input: &MatrixBufferHandle, output: &mut MatrixBufferHandle) {
        let src_guard = input.read();
        let src = src_guard.as_slice().expect("Log forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("Log forward: expected CPU buffer");

        debug_assert_eq!(src.len(), dst.len());

        for (o, &x) in dst.iter_mut().zip(src.iter()) {
            *o = x.ln();
        }
    }

    fn backward_buffered(
        &self,
        input: &MatrixBufferHandle,
        _output_cache: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        grad_in: &mut MatrixBufferHandle,
    ) {
        let x_guard = input.read();
        let x = x_guard.as_slice().expect("Log backward: expected CPU buffer");

        let go_guard = grad_out.read();
        let go = go_guard.as_slice().expect("Log backward: expected CPU buffer");

        let mut gi_guard = grad_in.write();
        let gi = gi_guard.as_slice_mut().expect("Log backward: expected CPU buffer");

        debug_assert_eq!(x.len(), go.len());
        debug_assert_eq!(x.len(), gi.len());

        for i in 0..x.len() {
            gi[i] = go[i] / x[i];
        }
    }
}

// ----------------------------------------------------------------
// Neg
// ----------------------------------------------------------------

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

impl BufferedElemCube for Neg {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }

    fn forward_buffered(&self, input: &MatrixBufferHandle, output: &mut MatrixBufferHandle) {
        let src_guard = input.read();
        let src = src_guard.as_slice().expect("Neg forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("Neg forward: expected CPU buffer");

        debug_assert_eq!(src.len(), dst.len());

        for (o, &x) in dst.iter_mut().zip(src.iter()) {
            *o = -x;
        }
    }

    fn backward_buffered(
        &self,
        _input: &MatrixBufferHandle,
        _output_cache: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        grad_in: &mut MatrixBufferHandle,
    ) {
        let go_guard = grad_out.read();
        let go = go_guard.as_slice().expect("Neg backward: expected CPU buffer");

        let mut gi_guard = grad_in.write();
        let gi = gi_guard.as_slice_mut().expect("Neg backward: expected CPU buffer");

        debug_assert_eq!(go.len(), gi.len());

        for (o, &g) in gi.iter_mut().zip(go.iter()) {
            *o = -g;
        }
    }
}

// ----------------------------------------------------------------
// Mul
// ----------------------------------------------------------------

#[derive(Debug)]
pub struct Mul {
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

impl BufferedElemCube for Mul {
    fn in_features(&self) -> usize {
        2 * self.features
    }

    fn out_features(&self) -> usize {
        self.features
    }

    fn forward_buffered(&self, input: &MatrixBufferHandle, output: &mut MatrixBufferHandle) {
        let rows = input.rows();
        let f = self.features;

        let src_guard = input.read();
        let src = src_guard.as_slice().expect("Mul forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("Mul forward: expected CPU buffer");

        debug_assert_eq!(src.len(), rows * 2 * f);
        debug_assert_eq!(dst.len(), rows * f);

        for r in 0..rows {
            for c in 0..f {
                dst[c * rows + r] = src[c * rows + r] * src[(c + f) * rows + r];
            }
        }
    }

    fn backward_buffered(
        &self,
        input: &MatrixBufferHandle,
        _output_cache: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        grad_in: &mut MatrixBufferHandle,
    ) {
        let rows = grad_out.rows();
        let f = self.features;

        let src_guard = input.read();
        let src = src_guard.as_slice().expect("Mul backward: expected CPU buffer");

        let go_guard = grad_out.read();
        let go = go_guard.as_slice().expect("Mul backward: expected CPU buffer");

        let mut gi_guard = grad_in.write();
        let gi = gi_guard.as_slice_mut().expect("Mul backward: expected CPU buffer");

        debug_assert_eq!(src.len(), rows * 2 * f);
        debug_assert_eq!(go.len(), rows * f);
        debug_assert_eq!(gi.len(), rows * 2 * f);

        for r in 0..rows {
            for c in 0..f {
                let g = go[c * rows + r];
                gi[c * rows + r] = g * src[(c + f) * rows + r];
                gi[(c + f) * rows + r] = g * src[c * rows + r];
            }
        }
    }
}

// ----------------------------------------------------------------
// Abs
// ----------------------------------------------------------------

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

impl BufferedElemCube for Abs {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }

    fn forward_buffered(&self, input: &MatrixBufferHandle, output: &mut MatrixBufferHandle) {
        let src_guard = input.read();
        let src = src_guard.as_slice().expect("Abs forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("Abs forward: expected CPU buffer");

        debug_assert_eq!(src.len(), dst.len());

        for (o, &x) in dst.iter_mut().zip(src.iter()) {
            *o = x.abs();
        }
    }

    fn backward_buffered(
        &self,
        input: &MatrixBufferHandle,
        _output_cache: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        grad_in: &mut MatrixBufferHandle,
    ) {
        let x_guard = input.read();
        let x = x_guard.as_slice().expect("Abs backward: expected CPU buffer");

        let go_guard = grad_out.read();
        let go = go_guard.as_slice().expect("Abs backward: expected CPU buffer");

        let mut gi_guard = grad_in.write();
        let gi = gi_guard.as_slice_mut().expect("Abs backward: expected CPU buffer");

        debug_assert_eq!(x.len(), go.len());
        debug_assert_eq!(x.len(), gi.len());

        for i in 0..x.len() {
            if x[i] > 0.0 {
                gi[i] = go[i];
            } else if x[i] < 0.0 {
                gi[i] = -go[i];
            } else {
                gi[i] = 0.0;
            }
        }
    }
}

// ----------------------------------------------------------------
// AddScalar
// ----------------------------------------------------------------

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

impl BufferedElemCube for AddScalar {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }

    fn forward_buffered(&self, input: &MatrixBufferHandle, output: &mut MatrixBufferHandle) {
        let scalar = self.0;
        let src_guard = input.read();
        let src = src_guard.as_slice().expect("AddScalar forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("AddScalar forward: expected CPU buffer");

        debug_assert_eq!(src.len(), dst.len());

        for (o, &x) in dst.iter_mut().zip(src.iter()) {
            *o = x + scalar;
        }
    }

    fn backward_buffered(
        &self,
        _input: &MatrixBufferHandle,
        _output_cache: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        grad_in: &mut MatrixBufferHandle,
    ) {
        let go_guard = grad_out.read();
        let go = go_guard.as_slice().expect("AddScalar backward: expected CPU buffer");

        let mut gi_guard = grad_in.write();
        let gi = gi_guard.as_slice_mut().expect("AddScalar backward: expected CPU buffer");

        debug_assert_eq!(go.len(), gi.len());
        gi.copy_from_slice(go);
    }
}

// ----------------------------------------------------------------
// Log1p
// ----------------------------------------------------------------

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

impl BufferedElemCube for Log1p {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }

    fn forward_buffered(&self, input: &MatrixBufferHandle, output: &mut MatrixBufferHandle) {
        let src_guard = input.read();
        let src = src_guard.as_slice().expect("Log1p forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("Log1p forward: expected CPU buffer");

        debug_assert_eq!(src.len(), dst.len());

        for (o, &x) in dst.iter_mut().zip(src.iter()) {
            *o = (x + 1.0).ln();
        }
    }

    fn backward_buffered(
        &self,
        input: &MatrixBufferHandle,
        _output_cache: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        grad_in: &mut MatrixBufferHandle,
    ) {
        let x_guard = input.read();
        let x = x_guard.as_slice().expect("Log1p backward: expected CPU buffer");

        let go_guard = grad_out.read();
        let go = go_guard.as_slice().expect("Log1p backward: expected CPU buffer");

        let mut gi_guard = grad_in.write();
        let gi = gi_guard.as_slice_mut().expect("Log1p backward: expected CPU buffer");

        debug_assert_eq!(x.len(), go.len());
        debug_assert_eq!(x.len(), gi.len());

        for i in 0..x.len() {
            gi[i] = go[i] / (1.0 + x[i]);
        }
    }
}

// ----------------------------------------------------------------
// AbsDiff
// ----------------------------------------------------------------

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

impl BufferedElemCube for AbsDiff {
    fn in_features(&self) -> usize {
        2 * self.features
    }

    fn out_features(&self) -> usize {
        self.features
    }

    fn forward_buffered(&self, input: &MatrixBufferHandle, output: &mut MatrixBufferHandle) {
        let rows = input.rows();
        let f = self.features;

        let src_guard = input.read();
        let src = src_guard.as_slice().expect("AbsDiff forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("AbsDiff forward: expected CPU buffer");

        debug_assert_eq!(src.len(), rows * 2 * f);
        debug_assert_eq!(dst.len(), rows * f);

        for r in 0..rows {
            for c in 0..f {
                dst[c * rows + r] = (src[c * rows + r] - src[(c + f) * rows + r]).abs();
            }
        }
    }

    fn backward_buffered(
        &self,
        input: &MatrixBufferHandle,
        _output_cache: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        grad_in: &mut MatrixBufferHandle,
    ) {
        let rows = grad_out.rows();
        let f = self.features;

        let src_guard = input.read();
        let src = src_guard.as_slice().expect("AbsDiff backward: expected CPU buffer");

        let go_guard = grad_out.read();
        let go = go_guard.as_slice().expect("AbsDiff backward: expected CPU buffer");

        let mut gi_guard = grad_in.write();
        let gi = gi_guard.as_slice_mut().expect("AbsDiff backward: expected CPU buffer");

        debug_assert_eq!(src.len(), rows * 2 * f);
        debug_assert_eq!(go.len(), rows * f);
        debug_assert_eq!(gi.len(), rows * 2 * f);

        for r in 0..rows {
            for c in 0..f {
                let diff = src[c * rows + r] - src[(c + f) * rows + r];
                let g = go[c * rows + r];
                let grad = if diff > 0.0 { g } else if diff < 0.0 { -g } else { 0.0 };
                gi[c * rows + r] = grad;
                gi[(c + f) * rows + r] = -grad;
            }
        }
    }
}

// ----------------------------------------------------------------
// CrossEntropyWithLogits (буферизованная реализация находится в cross_entropy.rs)
// ----------------------------------------------------------------

// Внимание: чтобы не создавать циклическую зависимость, реализация для
// CrossEntropyWithLogits находится в файле cross_entropy.rs.
// В этом файле мы оставляем только те кубики, которые определены здесь.
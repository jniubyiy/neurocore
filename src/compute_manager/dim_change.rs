// src/compute_manager/dim_change.rs

use crate::tensor::{Tensor2D, Tensor3D, Tensor4D, Tensor5D};

use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};

#[derive(Clone, Debug)]
pub enum DynamicTensor {
    Dim1(Tensor2D),
    Dim2(Tensor3D),
    Dim3(Tensor4D),
    Dim4(Tensor5D),
}

impl DynamicTensor {
    pub fn batch_size(&self) -> usize {
        match self {
            DynamicTensor::Dim1(t) => t.dim1,
            DynamicTensor::Dim2(t) => t.dim1,
            DynamicTensor::Dim3(t) => t.dim1,
            DynamicTensor::Dim4(t) => t.dim1,
        }
    }

    pub fn sample(&self, idx: usize) -> DynamicTensor {
        match self {
            DynamicTensor::Dim1(t) => DynamicTensor::Dim1(Tensor2D::new(vec![t.data[idx].clone()])),
            DynamicTensor::Dim2(t) => DynamicTensor::Dim2(Tensor3D::new(vec![t.data[idx].clone()])),
            DynamicTensor::Dim3(t) => DynamicTensor::Dim3(Tensor4D::new(vec![t.data[idx].clone()])),
            DynamicTensor::Dim4(t) => DynamicTensor::Dim4(Tensor5D::new(vec![t.data[idx].clone()])),
        }
    }

    pub fn features(&self) -> usize {
        match self {
            DynamicTensor::Dim1(t) => t.dim2,
            DynamicTensor::Dim2(t) => t.dim2 * t.dim3,
            DynamicTensor::Dim3(t) => t.dim2 * t.dim3 * t.dim4,
            DynamicTensor::Dim4(t) => t.dim2 * t.dim3 * t.dim4 * t.dim5,
        }
    }

    pub fn to_flat(&self) -> Vec<f32> {
        let mut buf = Vec::new();
        self.write_to_flat(&mut buf);
        buf
    }

    pub fn write_to_flat(&self, buf: &mut Vec<f32>) {
        buf.clear();
        match self {
            DynamicTensor::Dim1(t) => {
                buf.reserve(t.dim1 * t.dim2);
                for row in &t.data { buf.extend_from_slice(row); }
            }
            DynamicTensor::Dim2(t) => {
                let cap = t.dim1 * t.dim2 * t.dim3;
                buf.reserve(cap);
                for plane in &t.data { for row in plane { buf.extend_from_slice(row); } }
            }
            DynamicTensor::Dim3(t) => {
                let cap = t.dim1 * t.dim2 * t.dim3 * t.dim4;
                buf.reserve(cap);
                for vol in &t.data { for plane in vol { for row in plane { buf.extend_from_slice(row); } } }
            }
            DynamicTensor::Dim4(t) => {
                let cap = t.dim1 * t.dim2 * t.dim3 * t.dim4 * t.dim5;
                buf.reserve(cap);
                for hyper in &t.data { for vol in hyper { for plane in vol { for row in plane { buf.extend_from_slice(row); } } } }
            }
        }
    }

    pub fn from_flat(shape: &DynamicTensor, data: Vec<f32>) -> DynamicTensor {
        let mut dest = shape.clone();
        Self::from_flat_into(shape, &data, &mut dest);
        dest
    }

    pub fn from_flat_into(shape: &DynamicTensor, data: &[f32], dest: &mut DynamicTensor) {
        match (shape, dest) {
            (DynamicTensor::Dim1(orig), DynamicTensor::Dim1(ref mut t)) => {
                let features = orig.dim2;
                assert_eq!(data.len(), orig.dim1 * features);
                for (r, row) in t.data.iter_mut().enumerate() {
                    let start = r * features;
                    row.copy_from_slice(&data[start..start + features]);
                }
            }
            (DynamicTensor::Dim2(orig), DynamicTensor::Dim2(ref mut t)) => {
                let features = orig.dim3;
                assert_eq!(data.len(), orig.dim1 * orig.dim2 * features);
                let mut offset = 0;
                for plane in t.data.iter_mut() {
                    for row in plane.iter_mut() {
                        row.copy_from_slice(&data[offset..offset + features]);
                        offset += features;
                    }
                }
            }
            (DynamicTensor::Dim3(orig), DynamicTensor::Dim3(ref mut t)) => {
                let features = orig.dim4;
                assert_eq!(data.len(), orig.dim1 * orig.dim2 * orig.dim3 * features);
                let mut offset = 0;
                for vol in t.data.iter_mut() {
                    for plane in vol.iter_mut() {
                        for row in plane.iter_mut() {
                            row.copy_from_slice(&data[offset..offset + features]);
                            offset += features;
                        }
                    }
                }
            }
            (DynamicTensor::Dim4(orig), DynamicTensor::Dim4(ref mut t)) => {
                let features = orig.dim5;
                assert_eq!(data.len(), orig.dim1 * orig.dim2 * orig.dim3 * orig.dim4 * features);
                let mut offset = 0;
                for hyper in t.data.iter_mut() {
                    for vol in hyper.iter_mut() {
                        for plane in vol.iter_mut() {
                            for row in plane.iter_mut() {
                                row.copy_from_slice(&data[offset..offset + features]);
                                offset += features;
                            }
                        }
                    }
                }
            }
            _ => panic!("Shape mismatch in from_flat_into"),
        }
    }
}

// ------------------ Буферизованные версии (MatrixBufferHandle) ------------------

/// Версия `unsqueeze_mat` с использованием [`MatrixBufferHandle`] и пула [`TempMatrixPool`].
pub fn unsqueeze_mat_buffered_handle(
    pool: &mut TempMatrixPool,
    input: MatrixBufferHandle,
    target_dims: &[usize],
) -> MatrixBufferHandle {
    let batch = input.rows();
    let features = input.cols();
    let total_new: usize = target_dims.iter().product();
    assert_eq!(features, total_new, "unsqueeze_mat_buffered_handle: features mismatch");

    let last_dim = target_dims[target_dims.len() - 1];
    let remaining_product: usize = target_dims[..target_dims.len()-1].iter().product();
    let new_rows = batch * remaining_product;
    let new_cols = last_dim;

    let output = pool.acquire(new_rows, new_cols);

    {
        let ids = [input.id(), output.id()];
        input.memory().lock().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let src: &[f32] = &*first[0];
            let dst: &mut [f32] = &mut *rest[0];
            let mut idx = 0;
            for c in 0..features {
                for r in 0..batch {
                    let src_idx = c * batch + r;
                    let dst_r = idx / new_cols;
                    let dst_c = idx % new_cols;
                    let dst_idx = dst_c * new_rows + dst_r;
                    dst[dst_idx] = src[src_idx];
                    idx += 1;
                }
            }
        });
    }

    pool.release(input);

    output
}

/// Версия `reduce_mat` с использованием [`MatrixBufferHandle`] и пула [`TempMatrixPool`].
pub fn reduce_mat_buffered_handle(
    pool: &mut TempMatrixPool,
    input: MatrixBufferHandle,
    target_dims: &[usize],
) -> MatrixBufferHandle {
    let input_rows = input.rows();
    let input_cols = input.cols();
    let total = input_rows * input_cols;

    let remaining_product: usize = target_dims[..target_dims.len()-1].iter().product();
    let batch = input_rows / remaining_product;
    let new_rows = batch;
    let new_cols = total / new_rows;

    assert_eq!(total, new_rows * new_cols, "reduce_mat_buffered_handle: element count mismatch");

    let output = pool.acquire(new_rows, new_cols);

    {
        let ids = [input.id(), output.id()];
        input.memory().lock().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let src: &[f32] = &*first[0];
            let dst: &mut [f32] = &mut *rest[0];
            let mut idx = 0;
            for c in 0..input_cols {
                for r in 0..input_rows {
                    let src_idx = c * input_rows + r;
                    let dst_r = idx / new_cols;
                    let dst_c = idx % new_cols;
                    let dst_idx = dst_c * new_rows + dst_r;
                    dst[dst_idx] = src[src_idx];
                    idx += 1;
                }
            }
        });
    }

    pool.release(input);

    output
}
// src/compute_manager/dim_change.rs

use crate::tensor::{Tensor2D, Tensor3D, Tensor4D, Tensor5D};
use faer::Mat;

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
            DynamicTensor::Dim2(t) => t.dim3,
            DynamicTensor::Dim3(t) => t.dim4,
            DynamicTensor::Dim4(t) => t.dim5,
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

// ------------------ Вспомогательные функции ------------------

fn reshape_matrix(src: &Mat<f32>, new_rows: usize, new_cols: usize) -> Mat<f32> {
    let total = src.nrows() * src.ncols();
    assert_eq!(total, new_rows * new_cols,
        "reshape_matrix: total element count mismatch");
    let mut dst = Mat::zeros(new_rows, new_cols);
    let mut idx = 0;
    for c in 0..src.ncols() {
        for r in 0..src.nrows() {
            let dst_r = idx / new_cols;
            let dst_c = idx % new_cols;
            dst[(dst_r, dst_c)] = src[(r, c)];
            idx += 1;
        }
    }
    dst
}

// ------------------ Тензорные версии ------------------

pub fn unsqueeze_to(tensor: DynamicTensor, target_dims: Vec<usize>) -> DynamicTensor {
    match tensor {
        DynamicTensor::Dim1(t) => {
            let x_mat = crate::linalg::tensor2d_to_faer(&t);
            let batch = t.dim1;
            assert_eq!(target_dims.len(), 2);
            let d1 = target_dims[0];
            let d2 = target_dims[1];
            assert_eq!(x_mat.ncols(), d1 * d2);
            DynamicTensor::Dim2(crate::linalg::faer_to_tensor3d(&x_mat, batch, d1, d2))
        }
        DynamicTensor::Dim2(t) => {
            let x_mat = crate::linalg::tensor3d_to_faer(&t);
            let batch = t.dim1;
            assert_eq!(target_dims.len(), 3);
            let d1 = target_dims[0];
            let d2 = target_dims[1];
            let d3 = target_dims[2];
            assert_eq!(x_mat.ncols(), d1 * d2 * d3);
            DynamicTensor::Dim3(crate::linalg::faer_to_tensor4d(&x_mat, batch, d1, d2, d3))
        }
        DynamicTensor::Dim3(t) => {
            let x_mat = crate::linalg::tensor4d_to_faer(&t);
            let batch = t.dim1;
            assert_eq!(target_dims.len(), 4);
            let d1 = target_dims[0];
            let d2 = target_dims[1];
            let d3 = target_dims[2];
            let d4 = target_dims[3];
            assert_eq!(x_mat.ncols(), d1 * d2 * d3 * d4);
            DynamicTensor::Dim4(crate::linalg::faer_to_tensor5d(&x_mat, batch, d1, d2, d3, d4))
        }
        DynamicTensor::Dim4(_) => panic!("Cannot unsqueeze a 4D tensor (max)"),
    }
}

pub fn reduce_to(tensor: DynamicTensor, target_dims: Vec<usize>) -> DynamicTensor {
    match tensor {
        DynamicTensor::Dim2(t) => {
            let x_mat = crate::linalg::tensor3d_to_faer(&t);
            let batch = t.dim1;
            let new_rows = batch;
            let new_cols = x_mat.nrows() * x_mat.ncols() / new_rows;
            let y_mat = reshape_matrix(&x_mat, new_rows, new_cols);
            DynamicTensor::Dim1(crate::linalg::faer_to_tensor2d(&y_mat))
        }
        DynamicTensor::Dim3(t) => {
            let x_mat = crate::linalg::tensor4d_to_faer(&t);
            let batch = t.dim1;
            let new_rows = batch;
            let new_cols = x_mat.nrows() * x_mat.ncols() / new_rows;
            let y_mat = reshape_matrix(&x_mat, new_rows, new_cols);
            assert_eq!(target_dims.len(), 2);
            let d1 = target_dims[0];
            let d2 = target_dims[1];
            DynamicTensor::Dim2(crate::linalg::faer_to_tensor3d(&y_mat, batch, d1, d2))
        }
        DynamicTensor::Dim4(t) => {
            let x_mat = crate::linalg::tensor5d_to_faer(&t);
            let batch = t.dim1;
            let new_rows = batch;
            let new_cols = x_mat.nrows() * x_mat.ncols() / new_rows;
            let y_mat = reshape_matrix(&x_mat, new_rows, new_cols);
            assert_eq!(target_dims.len(), 3);
            let d1 = target_dims[0];
            let d2 = target_dims[1];
            let d3 = target_dims[2];
            DynamicTensor::Dim3(crate::linalg::faer_to_tensor4d(&y_mat, batch, d1, d2, d3))
        }
        DynamicTensor::Dim1(_) => panic!("Cannot reduce a 1D tensor"),
    }
}

// ------------------ Матричные версии ------------------

pub fn unsqueeze_mat(
    mat: &Mat<f32>,
    target_dims: &[usize],
) -> Mat<f32> {
    let batch = mat.nrows();
    let features = mat.ncols();
    let total_new = target_dims.iter().product::<usize>();
    assert_eq!(features, total_new, "unsqueeze_mat: features mismatch");

    let last_dim = target_dims[target_dims.len() - 1];
    let remaining_product: usize = target_dims[..target_dims.len()-1].iter().product();
    let new_rows = batch * remaining_product;
    let new_cols = last_dim;

    reshape_matrix(mat, new_rows, new_cols)
}

pub fn reduce_mat(
    mat: &Mat<f32>,
    target_dims: &[usize],
) -> Mat<f32> {
    let total = mat.nrows() * mat.ncols();
    let remaining_product: usize = target_dims[..target_dims.len()-1].iter().product();
    let batch = mat.nrows() / remaining_product;
    let new_rows = batch;
    let new_cols = total / new_rows;

    assert_eq!(total, new_rows * new_cols, "reduce_mat: element count mismatch");
    reshape_matrix(mat, new_rows, new_cols)
}
// src/linalg/mod.rs
use faer::Mat;
use crate::tensor::{Tensor2D, Tensor3D, Tensor4D, Tensor5D};

// 2D (Dim1) – уже имеет формат (batch, features), без изменений
pub fn tensor2d_to_faer(t: &Tensor2D) -> Mat<f32> {
    let rows = t.dim1;
    let cols = t.dim2;
    let mut flat = Vec::with_capacity(rows * cols);
    for r in 0..rows { flat.extend_from_slice(&t.data[r]); }
    Mat::from_fn(rows, cols, |r, c| flat[r * cols + c])
}
pub fn faer_to_tensor2d(m: &Mat<f32>) -> Tensor2D {
    let rows = m.nrows();
    let cols = m.ncols();
    let mut data = Vec::with_capacity(rows);
    for r in 0..rows { data.push((0..cols).map(|c| m[(r, c)]).collect()); }
    Tensor2D::new(data)
}

// 3D (Dim2) – flatten всех осей, кроме batch
pub fn tensor3d_to_faer(t: &Tensor3D) -> Mat<f32> {
    let (d1, d2, d3) = (t.dim1, t.dim2, t.dim3);
    let features = d2 * d3;
    let mut flat = Vec::with_capacity(d1 * features);
    for i in 0..d1 {
        for j in 0..d2 {
            flat.extend_from_slice(&t.data[i][j]);
        }
    }
    Mat::from_fn(d1, features, |r, c| flat[r * features + c])
}
pub fn faer_to_tensor3d(m: &Mat<f32>, d1: usize, d2: usize, d3: usize) -> Tensor3D {
    assert_eq!(m.nrows(), d1);
    assert_eq!(m.ncols(), d2 * d3);
    let mut data = Vec::with_capacity(d1);
    for i in 0..d1 {
        let mut slice = Vec::with_capacity(d2);
        for j in 0..d2 {
            let start = j * d3;
            let row: Vec<f32> = (0..d3).map(|c| m[(i, start + c)]).collect();
            slice.push(row);
        }
        data.push(slice);
    }
    Tensor3D::new(data)
}

// 4D (Dim3) – flatten всех осей, кроме batch
pub fn tensor4d_to_faer(t: &Tensor4D) -> Mat<f32> {
    let (d1, d2, d3, d4) = (t.dim1, t.dim2, t.dim3, t.dim4);
    let features = d2 * d3 * d4;
    let mut flat = Vec::with_capacity(d1 * features);
    for i in 0..d1 {
        for j in 0..d2 {
            for k in 0..d3 {
                flat.extend_from_slice(&t.data[i][j][k]);
            }
        }
    }
    Mat::from_fn(d1, features, |r, c| flat[r * features + c])
}
pub fn faer_to_tensor4d(m: &Mat<f32>, d1: usize, d2: usize, d3: usize, d4: usize) -> Tensor4D {
    assert_eq!(m.nrows(), d1);
    assert_eq!(m.ncols(), d2 * d3 * d4);
    let mut data = Vec::with_capacity(d1);
    for i in 0..d1 {
        let mut vol = Vec::with_capacity(d2);
        for j in 0..d2 {
            let mut plane = Vec::with_capacity(d3);
            for k in 0..d3 {
                let start = (j * d3 + k) * d4;
                let row: Vec<f32> = (0..d4).map(|c| m[(i, start + c)]).collect();
                plane.push(row);
            }
            vol.push(plane);
        }
        data.push(vol);
    }
    Tensor4D::new(data)
}

// 5D (Dim4) – flatten всех осей, кроме batch
pub fn tensor5d_to_faer(t: &Tensor5D) -> Mat<f32> {
    let (d1, d2, d3, d4, d5) = (t.dim1, t.dim2, t.dim3, t.dim4, t.dim5);
    let features = d2 * d3 * d4 * d5;
    let mut flat = Vec::with_capacity(d1 * features);
    for i in 0..d1 {
        for j in 0..d2 {
            for k in 0..d3 {
                for l in 0..d4 {
                    flat.extend_from_slice(&t.data[i][j][k][l]);
                }
            }
        }
    }
    Mat::from_fn(d1, features, |r, c| flat[r * features + c])
}
pub fn faer_to_tensor5d(m: &Mat<f32>, d1: usize, d2: usize, d3: usize, d4: usize, d5: usize) -> Tensor5D {
    assert_eq!(m.nrows(), d1);
    assert_eq!(m.ncols(), d2 * d3 * d4 * d5);
    let mut data = Vec::with_capacity(d1);
    for i in 0..d1 {
        let mut hyper = Vec::with_capacity(d2);
        for j in 0..d2 {
            let mut vol = Vec::with_capacity(d3);
            for k in 0..d3 {
                let mut plane = Vec::with_capacity(d4);
                for l in 0..d4 {
                    let start = ((j * d3 + k) * d4 + l) * d5;
                    let row: Vec<f32> = (0..d5).map(|c| m[(i, start + c)]).collect();
                    plane.push(row);
                }
                vol.push(plane);
            }
            hyper.push(vol);
        }
        data.push(hyper);
    }
    Tensor5D::new(data)
}
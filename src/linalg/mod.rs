use faer::Mat;
use crate::tensor::{Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D};

// ---------- 1D ----------
pub fn tensor1d_to_faer(t: &Tensor1D) -> Mat<f32> {
    Mat::from_fn(1, t.len(), |_, j| t.data[j])
}
pub fn faer_to_tensor1d(m: &Mat<f32>) -> Tensor1D {
    assert_eq!(m.nrows(), 1);
    Tensor1D::new((0..m.ncols()).map(|j| m[(0, j)]).collect())
}

// ---------- 2D ----------
pub fn tensor2d_to_faer(t: &Tensor2D) -> Mat<f32> {
    let rows = t.rows;
    let cols = t.cols;
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

// ---------- 3D ----------
pub fn tensor3d_to_faer(t: &Tensor3D) -> Mat<f32> {
    let (depth, rows, cols) = (t.depth, t.rows, t.cols);
    let total = depth * rows;
    let mut flat = Vec::with_capacity(total * cols);
    for d in 0..depth { for r in 0..rows { flat.extend_from_slice(&t.data[d][r]); } }
    Mat::from_fn(total, cols, |r, c| flat[r * cols + c])
}
pub fn faer_to_tensor3d(m: &Mat<f32>, depth: usize, rows: usize, cols: usize) -> Tensor3D {
    assert_eq!(m.nrows(), depth * rows);
    assert_eq!(m.ncols(), cols);
    let mut data = Vec::with_capacity(depth);
    for d in 0..depth {
        let mut slice = Vec::with_capacity(rows);
        for r in 0..rows {
            let offset = d * rows + r;
            slice.push((0..cols).map(|c| m[(offset, c)]).collect());
        }
        data.push(slice);
    }
    Tensor3D::new(data)
}

// ---------- 4D ----------
pub fn tensor4d_to_faer(t: &Tensor4D) -> Mat<f32> {
    let (dim1, depth, rows, cols) = (t.dim1, t.depth, t.rows, t.cols);
    let total = dim1 * depth * rows;
    let mut flat = Vec::with_capacity(total * cols);
    for d1 in 0..dim1 {
        for d in 0..depth {
            for r in 0..rows {
                flat.extend_from_slice(&t.data[d1][d][r]);
            }
        }
    }
    Mat::from_fn(total, cols, |r, c| flat[r * cols + c])
}
pub fn faer_to_tensor4d(m: &Mat<f32>, dim1: usize, depth: usize, rows: usize, cols: usize) -> Tensor4D {
    assert_eq!(m.nrows(), dim1 * depth * rows);
    assert_eq!(m.ncols(), cols);
    let mut data = Vec::with_capacity(dim1);
    for d1 in 0..dim1 {
        let mut slice3d = Vec::with_capacity(depth);
        for d in 0..depth {
            let mut slice2d = Vec::with_capacity(rows);
            for r in 0..rows {
                let offset = d1 * (depth * rows) + d * rows + r;
                slice2d.push((0..cols).map(|c| m[(offset, c)]).collect());
            }
            slice3d.push(slice2d);
        }
        data.push(slice3d);
    }
    Tensor4D::new(data)
}

// ---------- 5D ----------
pub fn tensor5d_to_faer(t: &Tensor5D) -> Mat<f32> {
    let (outer, dim1, depth, rows, cols) = (t.outer, t.dim1, t.depth, t.rows, t.cols);
    let total = outer * dim1 * depth * rows;
    let mut flat = Vec::with_capacity(total * cols);
    for o in 0..outer {
        for d1 in 0..dim1 {
            for d in 0..depth {
                for r in 0..rows {
                    flat.extend_from_slice(&t.data[o][d1][d][r]);
                }
            }
        }
    }
    Mat::from_fn(total, cols, |r, c| flat[r * cols + c])
}
pub fn faer_to_tensor5d(m: &Mat<f32>, outer: usize, dim1: usize, depth: usize, rows: usize, cols: usize) -> Tensor5D {
    assert_eq!(m.nrows(), outer * dim1 * depth * rows);
    assert_eq!(m.ncols(), cols);
    let mut data = Vec::with_capacity(outer);
    for o in 0..outer {
        let mut slice4d = Vec::with_capacity(dim1);
        for d1 in 0..dim1 {
            let mut slice3d = Vec::with_capacity(depth);
            for d in 0..depth {
                let mut slice2d = Vec::with_capacity(rows);
                for r in 0..rows {
                    let offset = o * (dim1 * depth * rows) + d1 * (depth * rows) + d * rows + r;
                    slice2d.push((0..cols).map(|c| m[(offset, c)]).collect());
                }
                slice3d.push(slice2d);
            }
            slice4d.push(slice3d);
        }
        data.push(slice4d);
    }
    Tensor5D::new(data)
}
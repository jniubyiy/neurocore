use faer::Mat;
use crate::tensor::{Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D};

// 1D
pub fn tensor1d_to_faer(t: &Tensor1D) -> Mat<f32> {
    Mat::from_fn(1, t.dim1(), |_, j| t.data[j])
}
pub fn faer_to_tensor1d(m: &Mat<f32>) -> Tensor1D {
    assert_eq!(m.nrows(), 1);
    Tensor1D::new((0..m.ncols()).map(|j| m[(0, j)]).collect())
}

// 2D
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

// 3D
pub fn tensor3d_to_faer(t: &Tensor3D) -> Mat<f32> {
    let (dim1, dim2, dim3) = (t.dim1, t.dim2, t.dim3);
    let total = dim1 * dim2;
    let mut flat = Vec::with_capacity(total * dim3);
    for i in 0..dim1 { for j in 0..dim2 { flat.extend_from_slice(&t.data[i][j]); } }
    Mat::from_fn(total, dim3, |r, c| flat[r * dim3 + c])
}
pub fn faer_to_tensor3d(m: &Mat<f32>, dim1: usize, dim2: usize, dim3: usize) -> Tensor3D {
    assert_eq!(m.nrows(), dim1 * dim2);
    assert_eq!(m.ncols(), dim3);
    let mut data = Vec::with_capacity(dim1);
    for i in 0..dim1 {
        let mut slice = Vec::with_capacity(dim2);
        for j in 0..dim2 {
            let offset = i * dim2 + j;
            slice.push((0..dim3).map(|c| m[(offset, c)]).collect());
        }
        data.push(slice);
    }
    Tensor3D::new(data)
}

// 4D
pub fn tensor4d_to_faer(t: &Tensor4D) -> Mat<f32> {
    let (dim1, dim2, dim3, dim4) = (t.dim1, t.dim2, t.dim3, t.dim4);
    let total = dim1 * dim2 * dim3;
    let mut flat = Vec::with_capacity(total * dim4);
    for i in 0..dim1 {
        for j in 0..dim2 {
            for k in 0..dim3 {
                flat.extend_from_slice(&t.data[i][j][k]);
            }
        }
    }
    Mat::from_fn(total, dim4, |r, c| flat[r * dim4 + c])
}
pub fn faer_to_tensor4d(m: &Mat<f32>, dim1: usize, dim2: usize, dim3: usize, dim4: usize) -> Tensor4D {
    assert_eq!(m.nrows(), dim1 * dim2 * dim3);
    assert_eq!(m.ncols(), dim4);
    let mut data = Vec::with_capacity(dim1);
    for i in 0..dim1 {
        let mut slice3d = Vec::with_capacity(dim2);
        for j in 0..dim2 {
            let mut slice2d = Vec::with_capacity(dim3);
            for k in 0..dim3 {
                let offset = i * (dim2 * dim3) + j * dim3 + k;
                slice2d.push((0..dim4).map(|c| m[(offset, c)]).collect());
            }
            slice3d.push(slice2d);
        }
        data.push(slice3d);
    }
    Tensor4D::new(data)
}

// 5D
pub fn tensor5d_to_faer(t: &Tensor5D) -> Mat<f32> {
    let (dim1, dim2, dim3, dim4, dim5) = (t.dim1, t.dim2, t.dim3, t.dim4, t.dim5);
    let total = dim1 * dim2 * dim3 * dim4;
    let mut flat = Vec::with_capacity(total * dim5);
    for i in 0..dim1 {
        for j in 0..dim2 {
            for k in 0..dim3 {
                for l in 0..dim4 {
                    flat.extend_from_slice(&t.data[i][j][k][l]);
                }
            }
        }
    }
    Mat::from_fn(total, dim5, |r, c| flat[r * dim5 + c])
}
pub fn faer_to_tensor5d(m: &Mat<f32>, dim1: usize, dim2: usize, dim3: usize, dim4: usize, dim5: usize) -> Tensor5D {
    assert_eq!(m.nrows(), dim1 * dim2 * dim3 * dim4);
    assert_eq!(m.ncols(), dim5);
    let mut data = Vec::with_capacity(dim1);
    for i in 0..dim1 {
        let mut slice4d = Vec::with_capacity(dim2);
        for j in 0..dim2 {
            let mut slice3d = Vec::with_capacity(dim3);
            for k in 0..dim3 {
                let mut slice2d = Vec::with_capacity(dim4);
                for l in 0..dim4 {
                    let offset = i * (dim2 * dim3 * dim4) + j * (dim3 * dim4) + k * dim4 + l;
                    slice2d.push((0..dim5).map(|c| m[(offset, c)]).collect());
                }
                slice3d.push(slice2d);
            }
            slice4d.push(slice3d);
        }
        data.push(slice4d);
    }
    Tensor5D::new(data)
}
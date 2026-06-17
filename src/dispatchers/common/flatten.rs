// Временно отключаем, т.к. Jacobian2D недоступен
/*
use crate::tensor::Tensor2D;
use crate::jacobian::Jacobian2D;

pub fn flatten_tensor(t: &Tensor2D) -> Vec<f32> {
    let mut v = Vec::with_capacity(t.rows * t.cols);
    for r in 0..t.rows { v.extend_from_slice(&t.data[r]); }
    v
}

pub fn flatten_jacobian(j: &Jacobian2D) -> Vec<f32> {
    let mut v = Vec::with_capacity(j.rows * j.out_features * j.num_params);
    for r in 0..j.rows {
        for c in 0..j.out_features {
            v.extend_from_slice(&j.data[r][c]);
        }
    }
    v
}

pub fn unflatten_tensor(flat: &[f32], rows: usize, cols: usize) -> Tensor2D {
    let mut data = Vec::with_capacity(rows);
    for r in 0..rows { data.push(flat[r*cols..(r+1)*cols].to_vec()); }
    Tensor2D::new(data)
}

pub fn unflatten_jacobian(flat: &[f32], rows: usize, out_features: usize, num_params: usize) -> Jacobian2D {
    let mut j = Jacobian2D::new(rows, out_features, num_params);
    for r in 0..rows {
        for c in 0..out_features {
            let start = (r * out_features + c) * num_params;
            for p in 0..num_params {
                j.data[r][c][p] = flat[start + p];
            }
        }
    }
    j
}
*/

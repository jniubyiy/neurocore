use crate::tensor::{Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D};

// ---------- 1D ----------
pub fn mse_loss(pred: &Tensor1D, target: &Tensor1D) -> (f32, Tensor1D) {
    let n = pred.dim1() as f32;
    assert_eq!(pred.dim1(), target.dim1(), "MSE 1D: size mismatch");
    let diff: Vec<f32> = pred.data.iter().zip(target.data.iter()).map(|(p, t)| p - t).collect();
    let loss = diff.iter().map(|d| d * d).sum::<f32>() / n;
    let delta: Vec<f32> = diff.iter().map(|d| 2.0 * d / n).collect();
    (loss, Tensor1D::new(delta))
}

pub fn mae_loss(pred: &Tensor1D, target: &Tensor1D) -> (f32, Tensor1D) {
    let n = pred.dim1() as f32;
    let diff: Vec<f32> = pred.data.iter().zip(target.data.iter()).map(|(p, t)| p - t).collect();
    let loss = diff.iter().map(|d| d.abs()).sum::<f32>() / n;
    let delta: Vec<f32> = diff.iter().map(|d| d.signum() / n).collect();
    (loss, Tensor1D::new(delta))
}

pub fn cross_entropy_loss(pred: &Tensor1D, target: &Tensor1D) -> (f32, Tensor1D) {
    let class = target.data[0] as usize;
    let n = pred.dim1();
    let max_val = pred.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = pred.data.iter().map(|&x| (x - max_val).exp()).collect();
    let sum = exps.iter().sum::<f32>();
    let softmax: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
    let loss = -softmax[class].ln();
    let delta: Vec<f32> = (0..n).map(|i| if i == class { softmax[i] - 1.0 } else { softmax[i] }).collect();
    (loss, Tensor1D::new(delta))
}

// ---------- 2D ----------
pub fn mse_loss_2d(pred: &Tensor2D, target: &Tensor2D) -> (f32, Tensor2D) {
    let n = (pred.dim1 * pred.dim2) as f32;
    let mut loss = 0.0;
    let mut delta = vec![vec![0.0; pred.dim2]; pred.dim1];
    for i in 0..pred.dim1 {
        for j in 0..pred.dim2 {
            let diff = pred.data[i][j] - target.data[i][j];
            loss += diff * diff;
            delta[i][j] = 2.0 * diff / n;
        }
    }
    (loss / n, Tensor2D::new(delta))
}

pub fn mae_loss_2d(pred: &Tensor2D, target: &Tensor2D) -> (f32, Tensor2D) {
    let n = (pred.dim1 * pred.dim2) as f32;
    let mut loss = 0.0;
    let mut delta = vec![vec![0.0; pred.dim2]; pred.dim1];
    for i in 0..pred.dim1 {
        for j in 0..pred.dim2 {
            let diff = pred.data[i][j] - target.data[i][j];
            loss += diff.abs();
            delta[i][j] = diff.signum() / n;
        }
    }
    (loss / n, Tensor2D::new(delta))
}

pub fn cross_entropy_loss_2d(logits: &Tensor2D, target: &Tensor2D) -> (f32, Tensor2D) {
    let rows = logits.dim1;
    let cols = logits.dim2;
    let mut loss = 0.0;
    let mut delta = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        let class = target.data[i][0] as usize;
        let max_val = logits.data[i].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exps = vec![0.0; cols];
        let mut sum = 0.0;
        for c in 0..cols {
            exps[c] = (logits.data[i][c] - max_val).exp();
            sum += exps[c];
        }
        loss -= (exps[class] / sum).ln();
        for c in 0..cols {
            let sm = exps[c] / sum;
            delta[i][c] = if c == class { sm - 1.0 } else { sm };
        }
    }
    (loss / rows as f32, Tensor2D::new(delta))
}

// ---------- 3D ----------
pub fn mse_loss_3d(pred: &Tensor3D, target: &Tensor3D) -> (f32, Tensor3D) {
    let n = (pred.dim1 * pred.dim2 * pred.dim3) as f32;
    let mut loss = 0.0;
    let mut delta = vec![vec![vec![0.0; pred.dim3]; pred.dim2]; pred.dim1];
    for i in 0..pred.dim1 {
        for j in 0..pred.dim2 {
            for k in 0..pred.dim3 {
                let diff = pred.data[i][j][k] - target.data[i][j][k];
                loss += diff * diff;
                delta[i][j][k] = 2.0 * diff / n;
            }
        }
    }
    (loss / n, Tensor3D::new(delta))
}

pub fn mae_loss_3d(pred: &Tensor3D, target: &Tensor3D) -> (f32, Tensor3D) {
    let n = (pred.dim1 * pred.dim2 * pred.dim3) as f32;
    let mut loss = 0.0;
    let mut delta = vec![vec![vec![0.0; pred.dim3]; pred.dim2]; pred.dim1];
    for i in 0..pred.dim1 {
        for j in 0..pred.dim2 {
            for k in 0..pred.dim3 {
                let diff = pred.data[i][j][k] - target.data[i][j][k];
                loss += diff.abs();
                delta[i][j][k] = diff.signum() / n;
            }
        }
    }
    (loss / n, Tensor3D::new(delta))
}

pub fn cross_entropy_loss_3d(logits: &Tensor3D, target: &Tensor3D) -> (f32, Tensor3D) {
    let rows = logits.dim1 * logits.dim2;
    let cols = logits.dim3;
    let mut loss = 0.0;
    let mut delta = vec![vec![vec![0.0; cols]; logits.dim2]; logits.dim1];
    for i in 0..logits.dim1 {
        for j in 0..logits.dim2 {
            let class = target.data[i][j][0] as usize;
            let max_val = logits.data[i][j].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exps = vec![0.0; cols];
            let mut sum = 0.0;
            for c in 0..cols {
                exps[c] = (logits.data[i][j][c] - max_val).exp();
                sum += exps[c];
            }
            loss -= (exps[class] / sum).ln();
            for c in 0..cols {
                let sm = exps[c] / sum;
                delta[i][j][c] = if c == class { sm - 1.0 } else { sm };
            }
        }
    }
    (loss / rows as f32, Tensor3D::new(delta))
}

// ---------- 4D ----------
pub fn mse_loss_4d(pred: &Tensor4D, target: &Tensor4D) -> (f32, Tensor4D) {
    let n = (pred.dim1 * pred.dim2 * pred.dim3 * pred.dim4) as f32;
    let mut loss = 0.0;
    let mut delta = vec![vec![vec![vec![0.0; pred.dim4]; pred.dim3]; pred.dim2]; pred.dim1];
    for i in 0..pred.dim1 {
        for j in 0..pred.dim2 {
            for k in 0..pred.dim3 {
                for l in 0..pred.dim4 {
                    let diff = pred.data[i][j][k][l] - target.data[i][j][k][l];
                    loss += diff * diff;
                    delta[i][j][k][l] = 2.0 * diff / n;
                }
            }
        }
    }
    (loss / n, Tensor4D::new(delta))
}

pub fn mae_loss_4d(pred: &Tensor4D, target: &Tensor4D) -> (f32, Tensor4D) {
    let n = (pred.dim1 * pred.dim2 * pred.dim3 * pred.dim4) as f32;
    let mut loss = 0.0;
    let mut delta = vec![vec![vec![vec![0.0; pred.dim4]; pred.dim3]; pred.dim2]; pred.dim1];
    for i in 0..pred.dim1 {
        for j in 0..pred.dim2 {
            for k in 0..pred.dim3 {
                for l in 0..pred.dim4 {
                    let diff = pred.data[i][j][k][l] - target.data[i][j][k][l];
                    loss += diff.abs();
                    delta[i][j][k][l] = diff.signum() / n;
                }
            }
        }
    }
    (loss / n, Tensor4D::new(delta))
}

pub fn cross_entropy_loss_4d(logits: &Tensor4D, target: &Tensor4D) -> (f32, Tensor4D) {
    let rows = logits.dim1 * logits.dim2 * logits.dim3;
    let cols = logits.dim4;
    let mut loss = 0.0;
    let mut delta = vec![vec![vec![vec![0.0; cols]; logits.dim3]; logits.dim2]; logits.dim1];
    for i in 0..logits.dim1 {
        for j in 0..logits.dim2 {
            for k in 0..logits.dim3 {
                let class = target.data[i][j][k][0] as usize;
                let max_val = logits.data[i][j][k].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exps = vec![0.0; cols];
                let mut sum = 0.0;
                for c in 0..cols {
                    exps[c] = (logits.data[i][j][k][c] - max_val).exp();
                    sum += exps[c];
                }
                loss -= (exps[class] / sum).ln();
                for c in 0..cols {
                    let sm = exps[c] / sum;
                    delta[i][j][k][c] = if c == class { sm - 1.0 } else { sm };
                }
            }
        }
    }
    (loss / rows as f32, Tensor4D::new(delta))
}

// ---------- 5D ----------
pub fn mse_loss_5d(pred: &Tensor5D, target: &Tensor5D) -> (f32, Tensor5D) {
    let n = (pred.dim1 * pred.dim2 * pred.dim3 * pred.dim4 * pred.dim5) as f32;
    let mut loss = 0.0;
    let mut delta = vec![vec![vec![vec![vec![0.0; pred.dim5]; pred.dim4]; pred.dim3]; pred.dim2]; pred.dim1];
    for i in 0..pred.dim1 {
        for j in 0..pred.dim2 {
            for k in 0..pred.dim3 {
                for l in 0..pred.dim4 {
                    for m in 0..pred.dim5 {
                        let diff = pred.data[i][j][k][l][m] - target.data[i][j][k][l][m];
                        loss += diff * diff;
                        delta[i][j][k][l][m] = 2.0 * diff / n;
                    }
                }
            }
        }
    }
    (loss / n, Tensor5D::new(delta))
}

pub fn mae_loss_5d(pred: &Tensor5D, target: &Tensor5D) -> (f32, Tensor5D) {
    let n = (pred.dim1 * pred.dim2 * pred.dim3 * pred.dim4 * pred.dim5) as f32;
    let mut loss = 0.0;
    let mut delta = vec![vec![vec![vec![vec![0.0; pred.dim5]; pred.dim4]; pred.dim3]; pred.dim2]; pred.dim1];
    for i in 0..pred.dim1 {
        for j in 0..pred.dim2 {
            for k in 0..pred.dim3 {
                for l in 0..pred.dim4 {
                    for m in 0..pred.dim5 {
                        let diff = pred.data[i][j][k][l][m] - target.data[i][j][k][l][m];
                        loss += diff.abs();
                        delta[i][j][k][l][m] = diff.signum() / n;
                    }
                }
            }
        }
    }
    (loss / n, Tensor5D::new(delta))
}

pub fn cross_entropy_loss_5d(logits: &Tensor5D, target: &Tensor5D) -> (f32, Tensor5D) {
    let rows = logits.dim1 * logits.dim2 * logits.dim3 * logits.dim4;
    let cols = logits.dim5;
    let mut loss = 0.0;
    let mut delta = vec![vec![vec![vec![vec![0.0; cols]; logits.dim4]; logits.dim3]; logits.dim2]; logits.dim1];
    for i in 0..logits.dim1 {
        for j in 0..logits.dim2 {
            for k in 0..logits.dim3 {
                for l in 0..logits.dim4 {
                    let class = target.data[i][j][k][l][0] as usize;
                    let max_val = logits.data[i][j][k][l].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut exps = vec![0.0; cols];
                    let mut sum = 0.0;
                    for c in 0..cols {
                        exps[c] = (logits.data[i][j][k][l][c] - max_val).exp();
                        sum += exps[c];
                    }
                    loss -= (exps[class] / sum).ln();
                    for c in 0..cols {
                        let sm = exps[c] / sum;
                        delta[i][j][k][l][c] = if c == class { sm - 1.0 } else { sm };
                    }
                }
            }
        }
    }
    (loss / rows as f32, Tensor5D::new(delta))
}
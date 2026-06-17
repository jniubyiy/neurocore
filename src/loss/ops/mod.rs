use crate::tensor::{Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D};

// ---------- 1D ----------
pub fn mse_loss(pred: &Tensor1D, target: &Tensor1D) -> (f32, Tensor1D) {
    let n = pred.len();
    assert_eq!(n, target.len(), "MSE: pred and target must have same length");
    let diff: Vec<f32> = pred.data.iter().zip(target.data.iter()).map(|(p, t)| p - t).collect();
    let loss = diff.iter().map(|d| d * d).sum::<f32>() / n as f32;
    let delta: Vec<f32> = diff.iter().map(|d| 2.0 * d / n as f32).collect();
    (loss, Tensor1D::new(delta))
}

pub fn mae_loss(pred: &Tensor1D, target: &Tensor1D) -> (f32, Tensor1D) {
    let n = pred.len();
    assert_eq!(n, target.len(), "MAE: pred and target must have same length");
    let diff: Vec<f32> = pred.data.iter().zip(target.data.iter()).map(|(p, t)| p - t).collect();
    let loss = diff.iter().map(|d| d.abs()).sum::<f32>() / n as f32;
    let delta: Vec<f32> = diff.iter().map(|d| d.signum() / n as f32).collect();
    (loss, Tensor1D::new(delta))
}

pub fn cross_entropy_loss(pred: &Tensor1D, target: &Tensor1D) -> (f32, Tensor1D) {
    let class = target.data[0] as usize;
    let n = pred.len();
    assert!(class < n, "CrossEntropy: class index out of bounds");

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
    let n = (pred.rows * pred.cols) as f32;
    let mut loss = 0.0;
    let mut delta = vec![vec![0.0; pred.cols]; pred.rows];
    for r in 0..pred.rows {
        for c in 0..pred.cols {
            let diff = pred.data[r][c] - target.data[r][c];
            loss += diff * diff;
            delta[r][c] = 2.0 * diff / n;
        }
    }
    (loss / n, Tensor2D::new(delta))
}

pub fn mae_loss_2d(pred: &Tensor2D, target: &Tensor2D) -> (f32, Tensor2D) {
    let n = (pred.rows * pred.cols) as f32;
    let mut loss = 0.0;
    let mut delta = vec![vec![0.0; pred.cols]; pred.rows];
    for r in 0..pred.rows {
        for c in 0..pred.cols {
            let diff = pred.data[r][c] - target.data[r][c];
            loss += diff.abs();
            delta[r][c] = diff.signum() / n;
        }
    }
    (loss / n, Tensor2D::new(delta))
}

pub fn cross_entropy_loss_2d(logits: &Tensor2D, target: &Tensor2D) -> (f32, Tensor2D) {
    let rows = logits.rows;
    let cols = logits.cols;
    let mut loss = 0.0;
    let mut delta = vec![vec![0.0; cols]; rows];
    for r in 0..rows {
        let class = target.data[r][0] as usize;
        let max_val = logits.data[r].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exps = vec![0.0; cols];
        let mut sum = 0.0;
        for c in 0..cols {
            exps[c] = (logits.data[r][c] - max_val).exp();
            sum += exps[c];
        }
        loss -= (exps[class] / sum).ln();
        for c in 0..cols {
            let sm = exps[c] / sum;
            delta[r][c] = if c == class { sm - 1.0 } else { sm };
        }
    }
    (loss / rows as f32, Tensor2D::new(delta))
}

// ---------- 3D ----------
pub fn mse_loss_3d(pred: &Tensor3D, target: &Tensor3D) -> (f32, Tensor3D) {
    let n = (pred.depth * pred.rows * pred.cols) as f32;
    let mut loss = 0.0;
    let mut delta = vec![vec![vec![0.0; pred.cols]; pred.rows]; pred.depth];
    for d in 0..pred.depth {
        for r in 0..pred.rows {
            for c in 0..pred.cols {
                let diff = pred.data[d][r][c] - target.data[d][r][c];
                loss += diff * diff;
                delta[d][r][c] = 2.0 * diff / n;
            }
        }
    }
    (loss / n, Tensor3D::new(delta))
}

pub fn mae_loss_3d(pred: &Tensor3D, target: &Tensor3D) -> (f32, Tensor3D) {
    let n = (pred.depth * pred.rows * pred.cols) as f32;
    let mut loss = 0.0;
    let mut delta = vec![vec![vec![0.0; pred.cols]; pred.rows]; pred.depth];
    for d in 0..pred.depth {
        for r in 0..pred.rows {
            for c in 0..pred.cols {
                let diff = pred.data[d][r][c] - target.data[d][r][c];
                loss += diff.abs();
                delta[d][r][c] = diff.signum() / n;
            }
        }
    }
    (loss / n, Tensor3D::new(delta))
}

pub fn cross_entropy_loss_3d(logits: &Tensor3D, target: &Tensor3D) -> (f32, Tensor3D) {
    let rows = logits.depth * logits.rows;
    let cols = logits.cols;
    let mut loss = 0.0;
    let mut delta = vec![vec![vec![0.0; cols]; logits.rows]; logits.depth];
    for d in 0..logits.depth {
        for r in 0..logits.rows {
            let class = target.data[d][r][0] as usize;
            let max_val = logits.data[d][r].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exps = vec![0.0; cols];
            let mut sum = 0.0;
            for c in 0..cols {
                exps[c] = (logits.data[d][r][c] - max_val).exp();
                sum += exps[c];
            }
            loss -= (exps[class] / sum).ln();
            for c in 0..cols {
                let sm = exps[c] / sum;
                delta[d][r][c] = if c == class { sm - 1.0 } else { sm };
            }
        }
    }
    (loss / rows as f32, Tensor3D::new(delta))
}

// ---------- 4D ----------
pub fn mse_loss_4d(pred: &Tensor4D, target: &Tensor4D) -> (f32, Tensor4D) {
    let n = (pred.dim1 * pred.depth * pred.rows * pred.cols) as f32;
    let mut loss = 0.0;
    let mut delta = vec![vec![vec![vec![0.0; pred.cols]; pred.rows]; pred.depth]; pred.dim1];
    for d1 in 0..pred.dim1 {
        for d in 0..pred.depth {
            for r in 0..pred.rows {
                for c in 0..pred.cols {
                    let diff = pred.data[d1][d][r][c] - target.data[d1][d][r][c];
                    loss += diff * diff;
                    delta[d1][d][r][c] = 2.0 * diff / n;
                }
            }
        }
    }
    (loss / n, Tensor4D::new(delta))
}

pub fn mae_loss_4d(pred: &Tensor4D, target: &Tensor4D) -> (f32, Tensor4D) {
    let n = (pred.dim1 * pred.depth * pred.rows * pred.cols) as f32;
    let mut loss = 0.0;
    let mut delta = vec![vec![vec![vec![0.0; pred.cols]; pred.rows]; pred.depth]; pred.dim1];
    for d1 in 0..pred.dim1 {
        for d in 0..pred.depth {
            for r in 0..pred.rows {
                for c in 0..pred.cols {
                    let diff = pred.data[d1][d][r][c] - target.data[d1][d][r][c];
                    loss += diff.abs();
                    delta[d1][d][r][c] = diff.signum() / n;
                }
            }
        }
    }
    (loss / n, Tensor4D::new(delta))
}

pub fn cross_entropy_loss_4d(logits: &Tensor4D, target: &Tensor4D) -> (f32, Tensor4D) {
    let rows = logits.dim1 * logits.depth * logits.rows;
    let cols = logits.cols;
    let mut loss = 0.0;
    let mut delta = vec![vec![vec![vec![0.0; cols]; logits.rows]; logits.depth]; logits.dim1];
    for d1 in 0..logits.dim1 {
        for d in 0..logits.depth {
            for r in 0..logits.rows {
                let class = target.data[d1][d][r][0] as usize;
                let max_val = logits.data[d1][d][r].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exps = vec![0.0; cols];
                let mut sum = 0.0;
                for c in 0..cols {
                    exps[c] = (logits.data[d1][d][r][c] - max_val).exp();
                    sum += exps[c];
                }
                loss -= (exps[class] / sum).ln();
                for c in 0..cols {
                    let sm = exps[c] / sum;
                    delta[d1][d][r][c] = if c == class { sm - 1.0 } else { sm };
                }
            }
        }
    }
    (loss / rows as f32, Tensor4D::new(delta))
}

// ---------- 5D ----------
pub fn mse_loss_5d(pred: &Tensor5D, target: &Tensor5D) -> (f32, Tensor5D) {
    let n = (pred.outer * pred.dim1 * pred.depth * pred.rows * pred.cols) as f32;
    let mut loss = 0.0;
    let mut delta = vec![vec![vec![vec![vec![0.0; pred.cols]; pred.rows]; pred.depth]; pred.dim1]; pred.outer];
    for o in 0..pred.outer {
        for d1 in 0..pred.dim1 {
            for d in 0..pred.depth {
                for r in 0..pred.rows {
                    for c in 0..pred.cols {
                        let diff = pred.data[o][d1][d][r][c] - target.data[o][d1][d][r][c];
                        loss += diff * diff;
                        delta[o][d1][d][r][c] = 2.0 * diff / n;
                    }
                }
            }
        }
    }
    (loss / n, Tensor5D::new(delta))
}

pub fn mae_loss_5d(pred: &Tensor5D, target: &Tensor5D) -> (f32, Tensor5D) {
    let n = (pred.outer * pred.dim1 * pred.depth * pred.rows * pred.cols) as f32;
    let mut loss = 0.0;
    let mut delta = vec![vec![vec![vec![vec![0.0; pred.cols]; pred.rows]; pred.depth]; pred.dim1]; pred.outer];
    for o in 0..pred.outer {
        for d1 in 0..pred.dim1 {
            for d in 0..pred.depth {
                for r in 0..pred.rows {
                    for c in 0..pred.cols {
                        let diff = pred.data[o][d1][d][r][c] - target.data[o][d1][d][r][c];
                        loss += diff.abs();
                        delta[o][d1][d][r][c] = diff.signum() / n;
                    }
                }
            }
        }
    }
    (loss / n, Tensor5D::new(delta))
}

pub fn cross_entropy_loss_5d(logits: &Tensor5D, target: &Tensor5D) -> (f32, Tensor5D) {
    let rows = logits.outer * logits.dim1 * logits.depth * logits.rows;
    let cols = logits.cols;
    let mut loss = 0.0;
    let mut delta = vec![vec![vec![vec![vec![0.0; cols]; logits.rows]; logits.depth]; logits.dim1]; logits.outer];
    for o in 0..logits.outer {
        for d1 in 0..logits.dim1 {
            for d in 0..logits.depth {
                for r in 0..logits.rows {
                    let class = target.data[o][d1][d][r][0] as usize;
                    let max_val = logits.data[o][d1][d][r].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut exps = vec![0.0; cols];
                    let mut sum = 0.0;
                    for c in 0..cols {
                        exps[c] = (logits.data[o][d1][d][r][c] - max_val).exp();
                        sum += exps[c];
                    }
                    loss -= (exps[class] / sum).ln();
                    for c in 0..cols {
                        let sm = exps[c] / sum;
                        delta[o][d1][d][r][c] = if c == class { sm - 1.0 } else { sm };
                    }
                }
            }
        }
    }
    (loss / rows as f32, Tensor5D::new(delta))
}
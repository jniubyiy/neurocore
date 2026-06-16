use crate::tensor::Tensor1D;
use crate::jacobian::Jacobian;
use crate::tensor::Tensor2D;
use crate::jacobian::Jacobian2D;
use crate::tensor::Tensor3D;
use crate::jacobian::Jacobian3D;
use crate::tensor::Tensor4D;
use crate::jacobian::Jacobian4D;
use crate::tensor::Tensor5D;
use crate::jacobian::Jacobian5D;
use super::DimReduce;
use crate::logging::Logger;

pub struct ReduceMean {
    axis: usize,
}

impl ReduceMean {
    pub fn new(axis: usize) -> Self {
        ReduceMean { axis }
    }
}

// 2D -> 1D
impl DimReduce<Tensor2D, Jacobian2D, Tensor1D, Jacobian> for ReduceMean {
    fn reduce(&self, input: &Tensor2D, j_input: &Jacobian2D) -> (Tensor1D, Jacobian) {
        let logger = Logger::new();
        logger.trace("ReduceMean 2D -> 1D");
        assert!(self.axis < 2);
        let (rows, cols) = (input.rows, input.cols);
        let p = j_input.num_params;
        if self.axis == 0 {
            let mut out_data = vec![0.0; cols];
            let mut j_out = Jacobian::new(cols, p);
            for r in 0..rows {
                for c in 0..cols {
                    out_data[c] += input.data[r][c];
                    for j in 0..p {
                        j_out.data[c][j] += j_input.data[r][c][j];
                    }
                }
            }
            let n = rows as f32;
            for c in 0..cols {
                out_data[c] /= n;
                for j in 0..p { j_out.data[c][j] /= n; }
            }
            (Tensor1D::new(out_data), j_out)
        } else {
            let mut out_data = vec![0.0; rows];
            let mut j_out = Jacobian::new(rows, p);
            for r in 0..rows {
                for c in 0..cols {
                    out_data[r] += input.data[r][c];
                    for j in 0..p {
                        j_out.data[r][j] += j_input.data[r][c][j];
                    }
                }
            }
            let n = cols as f32;
            for r in 0..rows {
                out_data[r] /= n;
                for j in 0..p { j_out.data[r][j] /= n; }
            }
            (Tensor1D::new(out_data), j_out)
        }
    }

    fn param_count(&self) -> usize { 0 }
    fn update_params(&mut self, _lr: f32, _grad: &[f32]) {}
    fn get_params(&self) -> Vec<f32> { vec![] }
    fn set_params(&mut self, _values: &[f32]) {}
}

// 3D -> 2D
impl DimReduce<Tensor3D, Jacobian3D, Tensor2D, Jacobian2D> for ReduceMean {
    fn reduce(&self, input: &Tensor3D, j_input: &Jacobian3D) -> (Tensor2D, Jacobian2D) {
        let logger = Logger::new();
        logger.trace("ReduceMean 3D -> 2D");
        assert_eq!(self.axis, 0, "ReduceMean 3D->2D only supports axis=0 (depth)");
        let depth = input.depth;
        let rows = input.rows;
        let cols = input.cols;
        let p = j_input.num_params;
        let mut out_data = vec![vec![0.0; cols]; rows];
        let mut j_out = Jacobian2D::new(rows, cols, p);
        for d in 0..depth {
            for r in 0..rows {
                for c in 0..cols {
                    out_data[r][c] += input.data[d][r][c];
                    for j in 0..p {
                        j_out.data[r][c][j] += j_input.data[d][r][c][j];
                    }
                }
            }
        }
        let n = depth as f32;
        for r in 0..rows {
            for c in 0..cols {
                out_data[r][c] /= n;
                for j in 0..p { j_out.data[r][c][j] /= n; }
            }
        }
        (Tensor2D::new(out_data), j_out)
    }

    fn param_count(&self) -> usize { 0 }
    fn update_params(&mut self, _lr: f32, _grad: &[f32]) {}
    fn get_params(&self) -> Vec<f32> { vec![] }
    fn set_params(&mut self, _values: &[f32]) {}
}

// 4D -> 3D
impl DimReduce<Tensor4D, Jacobian4D, Tensor3D, Jacobian3D> for ReduceMean {
    fn reduce(&self, input: &Tensor4D, j_input: &Jacobian4D) -> (Tensor3D, Jacobian3D) {
        let logger = Logger::new();
        logger.trace("ReduceMean 4D -> 3D");
        assert_eq!(self.axis, 0, "ReduceMean 4D->3D only supports axis=0 (dim1)");
        let dim1 = input.dim1;
        let depth = input.depth;
        let rows = input.rows;
        let cols = input.cols;
        let p = j_input.num_params;
        let mut out = vec![vec![vec![0.0; cols]; rows]; depth];
        let mut j_out = Jacobian3D::new(depth, rows, cols, p);
        for d1 in 0..dim1 {
            for d in 0..depth {
                for r in 0..rows {
                    for c in 0..cols {
                        out[d][r][c] += input.data[d1][d][r][c];
                        for j in 0..p {
                            j_out.data[d][r][c][j] += j_input.data[d1][d][r][c][j];
                        }
                    }
                }
            }
        }
        let n = dim1 as f32;
        for d in 0..depth {
            for r in 0..rows {
                for c in 0..cols {
                    out[d][r][c] /= n;
                    for j in 0..p { j_out.data[d][r][c][j] /= n; }
                }
            }
        }
        (Tensor3D::new(out), j_out)
    }

    fn param_count(&self) -> usize { 0 }
    fn update_params(&mut self, _lr: f32, _grad: &[f32]) {}
    fn get_params(&self) -> Vec<f32> { vec![] }
    fn set_params(&mut self, _values: &[f32]) {}
}

// 5D -> 4D
impl DimReduce<Tensor5D, Jacobian5D, Tensor4D, Jacobian4D> for ReduceMean {
    fn reduce(&self, input: &Tensor5D, j_input: &Jacobian5D) -> (Tensor4D, Jacobian4D) {
        let logger = Logger::new();
        logger.trace("ReduceMean 5D -> 4D");
        assert_eq!(self.axis, 0, "ReduceMean 5D->4D only supports axis=0 (outer)");
        let outer = input.outer;
        let dim1 = input.dim1;
        let depth = input.depth;
        let rows = input.rows;
        let cols = input.cols;
        let p = j_input.num_params;
        let mut out = vec![vec![vec![vec![0.0; cols]; rows]; depth]; dim1];
        let mut j_out = Jacobian4D::new(dim1, depth, rows, cols, p);
        for o in 0..outer {
            for d1 in 0..dim1 {
                for d in 0..depth {
                    for r in 0..rows {
                        for c in 0..cols {
                            out[d1][d][r][c] += input.data[o][d1][d][r][c];
                            for j in 0..p {
                                j_out.data[d1][d][r][c][j] += j_input.data[o][d1][d][r][c][j];
                            }
                        }
                    }
                }
            }
        }
        let n = outer as f32;
        for d1 in 0..dim1 {
            for d in 0..depth {
                for r in 0..rows {
                    for c in 0..cols {
                        out[d1][d][r][c] /= n;
                        for j in 0..p { j_out.data[d1][d][r][c][j] /= n; }
                    }
                }
            }
        }
        (Tensor4D::new(out), j_out)
    }

    fn param_count(&self) -> usize { 0 }
    fn update_params(&mut self, _lr: f32, _grad: &[f32]) {}
    fn get_params(&self) -> Vec<f32> { vec![] }
    fn set_params(&mut self, _values: &[f32]) {}
}






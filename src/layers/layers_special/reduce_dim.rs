use crate::tensor::{Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D};
use super::DimReduce;

pub struct ReduceMean {
    axis: usize,
}

impl ReduceMean {
    pub fn new(axis: usize) -> Self {
        ReduceMean { axis }
    }
}

// 2D -> 1D
impl DimReduce<Tensor2D, Tensor1D> for ReduceMean {
    fn reduce(&self, input: &Tensor2D) -> Tensor1D {
        assert!(self.axis < 2);
        let (rows, cols) = (input.rows, input.cols);
        if self.axis == 0 {
            let mut out_data = vec![0.0; cols];
            for r in 0..rows {
                for c in 0..cols {
                    out_data[c] += input.data[r][c];
                }
            }
            let n = rows as f32;
            for c in 0..cols { out_data[c] /= n; }
            Tensor1D::new(out_data)
        } else {
            let mut out_data = vec![0.0; rows];
            for r in 0..rows {
                for c in 0..cols {
                    out_data[r] += input.data[r][c];
                }
            }
            let n = cols as f32;
            for r in 0..rows { out_data[r] /= n; }
            Tensor1D::new(out_data)
        }
    }
}

// 3D -> 2D (усреднение по глубине, axis=0)
impl DimReduce<Tensor3D, Tensor2D> for ReduceMean {
    fn reduce(&self, input: &Tensor3D) -> Tensor2D {
        assert_eq!(self.axis, 0, "ReduceMean 3D->2D only supports axis=0 (depth)");
        let depth = input.depth;
        let rows = input.rows;
        let cols = input.cols;
        let mut out_data = vec![vec![0.0; cols]; rows];
        for d in 0..depth {
            for r in 0..rows {
                for c in 0..cols {
                    out_data[r][c] += input.data[d][r][c];
                }
            }
        }
        let n = depth as f32;
        for r in 0..rows {
            for c in 0..cols { out_data[r][c] /= n; }
        }
        Tensor2D::new(out_data)
    }
}

// 4D -> 3D (усреднение по dim1, axis=0)
impl DimReduce<Tensor4D, Tensor3D> for ReduceMean {
    fn reduce(&self, input: &Tensor4D) -> Tensor3D {
        assert_eq!(self.axis, 0, "ReduceMean 4D->3D only supports axis=0 (dim1)");
        let dim1 = input.dim1;
        let depth = input.depth;
        let rows = input.rows;
        let cols = input.cols;
        let mut out = vec![vec![vec![0.0; cols]; rows]; depth];
        for d1 in 0..dim1 {
            for d in 0..depth {
                for r in 0..rows {
                    for c in 0..cols {
                        out[d][r][c] += input.data[d1][d][r][c];
                    }
                }
            }
        }
        let n = dim1 as f32;
        for d in 0..depth {
            for r in 0..rows {
                for c in 0..cols { out[d][r][c] /= n; }
            }
        }
        Tensor3D::new(out)
    }
}

// 5D -> 4D (усреднение по outer, axis=0)
impl DimReduce<Tensor5D, Tensor4D> for ReduceMean {
    fn reduce(&self, input: &Tensor5D) -> Tensor4D {
        assert_eq!(self.axis, 0, "ReduceMean 5D->4D only supports axis=0 (outer)");
        let outer = input.outer;
        let dim1 = input.dim1;
        let depth = input.depth;
        let rows = input.rows;
        let cols = input.cols;
        let mut out = vec![vec![vec![vec![0.0; cols]; rows]; depth]; dim1];
        for o in 0..outer {
            for d1 in 0..dim1 {
                for d in 0..depth {
                    for r in 0..rows {
                        for c in 0..cols {
                            out[d1][d][r][c] += input.data[o][d1][d][r][c];
                        }
                    }
                }
            }
        }
        let n = outer as f32;
        for d1 in 0..dim1 {
            for d in 0..depth {
                for r in 0..rows {
                    for c in 0..cols { out[d1][d][r][c] /= n; }
                }
            }
        }
        Tensor4D::new(out)
    }
}






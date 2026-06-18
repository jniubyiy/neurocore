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
        let (dim1, dim2) = (input.dim1, input.dim2);
        if self.axis == 0 {
            let mut out_data = vec![0.0; dim2];
            for i in 0..dim1 {
                for j in 0..dim2 {
                    out_data[j] += input.data[i][j];
                }
            }
            let n = dim1 as f32;
            for c in 0..dim2 { out_data[c] /= n; }
            Tensor1D::new(out_data)
        } else {
            let mut out_data = vec![0.0; dim1];
            for i in 0..dim1 {
                for j in 0..dim2 {
                    out_data[i] += input.data[i][j];
                }
            }
            let n = dim2 as f32;
            for r in 0..dim1 { out_data[r] /= n; }
            Tensor1D::new(out_data)
        }
    }
}

// 3D -> 2D (усреднение по dim1, axis=0)
impl DimReduce<Tensor3D, Tensor2D> for ReduceMean {
    fn reduce(&self, input: &Tensor3D) -> Tensor2D {
        assert_eq!(self.axis, 0, "ReduceMean 3D->2D only supports axis=0 (dim1)");
        let dim1 = input.dim1;
        let dim2 = input.dim2;
        let dim3 = input.dim3;
        let mut out_data = vec![vec![0.0; dim3]; dim2];
        for i in 0..dim1 {
            for j in 0..dim2 {
                for k in 0..dim3 {
                    out_data[j][k] += input.data[i][j][k];
                }
            }
        }
        let n = dim1 as f32;
        for j in 0..dim2 {
            for k in 0..dim3 { out_data[j][k] /= n; }
        }
        Tensor2D::new(out_data)
    }
}

// 4D -> 3D (усреднение по dim1, axis=0)
impl DimReduce<Tensor4D, Tensor3D> for ReduceMean {
    fn reduce(&self, input: &Tensor4D) -> Tensor3D {
        assert_eq!(self.axis, 0, "ReduceMean 4D->3D only supports axis=0 (dim1)");
        let dim1 = input.dim1;
        let dim2 = input.dim2;
        let dim3 = input.dim3;
        let dim4 = input.dim4;
        let mut out = vec![vec![vec![0.0; dim4]; dim3]; dim2];
        for i in 0..dim1 {
            for j in 0..dim2 {
                for k in 0..dim3 {
                    for l in 0..dim4 {
                        out[j][k][l] += input.data[i][j][k][l];
                    }
                }
            }
        }
        let n = dim1 as f32;
        for j in 0..dim2 {
            for k in 0..dim3 {
                for l in 0..dim4 { out[j][k][l] /= n; }
            }
        }
        Tensor3D::new(out)
    }
}

// 5D -> 4D (усреднение по dim1, axis=0)
impl DimReduce<Tensor5D, Tensor4D> for ReduceMean {
    fn reduce(&self, input: &Tensor5D) -> Tensor4D {
        assert_eq!(self.axis, 0, "ReduceMean 5D->4D only supports axis=0 (dim1)");
        let dim1 = input.dim1;
        let dim2 = input.dim2;
        let dim3 = input.dim3;
        let dim4 = input.dim4;
        let dim5 = input.dim5;
        let mut out = vec![vec![vec![vec![0.0; dim5]; dim4]; dim3]; dim2];
        for i in 0..dim1 {
            for j in 0..dim2 {
                for k in 0..dim3 {
                    for l in 0..dim4 {
                        for m in 0..dim5 {
                            out[j][k][l][m] += input.data[i][j][k][l][m];
                        }
                    }
                }
            }
        }
        let n = dim1 as f32;
        for j in 0..dim2 {
            for k in 0..dim3 {
                for l in 0..dim4 {
                    for m in 0..dim5 { out[j][k][l][m] /= n; }
                }
            }
        }
        Tensor4D::new(out)
    }
}






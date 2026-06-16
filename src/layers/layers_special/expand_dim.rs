use crate::tensor::Tensor1D;
use crate::jacobian::Jacobian;
use crate::tensor::Tensor2D;
use crate::jacobian::Jacobian2D;
use crate::tensor::Tensor3D;
use crate::jacobian::Jacobian3D;
use super::DimExpand;
use crate::logging::Logger;

pub struct Unsqueeze {
    axis: usize,
}

impl Unsqueeze {
    pub fn new(axis: usize) -> Self { Unsqueeze { axis } }
}

// 1D -> 2D
impl DimExpand<Tensor1D, Jacobian, Tensor2D, Jacobian2D> for Unsqueeze {
    fn expand(&self, input: &Tensor1D, j_input: &Jacobian) -> (Tensor2D, Jacobian2D) {
        let logger = Logger::new();
        logger.trace(&format!("Unsqueeze 1D->2D axis={}", self.axis));
        let len = input.len();
        let p = j_input.num_params;
        if self.axis == 0 {
            let data = vec![input.data.clone()];
            let mut j_out = Jacobian2D::new(1, len, p);
            for c in 0..len { j_out.data[0][c] = j_input.data[c].clone(); }
            (Tensor2D::new(data), j_out)
        } else {
            let mut data = Vec::with_capacity(len);
            for i in 0..len { data.push(vec![input.data[i]]); }
            let mut j_out = Jacobian2D::new(len, 1, p);
            for r in 0..len { j_out.data[r][0] = j_input.data[r].clone(); }
            (Tensor2D::new(data), j_out)
        }
    }
    fn param_count(&self) -> usize { 0 }
    fn update_params(&mut self, _lr: f32, _grad: &[f32]) {}
    fn get_params(&self) -> Vec<f32> { vec![] }
    fn set_params(&mut self, _values: &[f32]) {}
}

// 2D -> 3D
impl DimExpand<Tensor2D, Jacobian2D, Tensor3D, Jacobian3D> for Unsqueeze {
    fn expand(&self, input: &Tensor2D, j_input: &Jacobian2D) -> (Tensor3D, Jacobian3D) {
        let logger = Logger::new();
        logger.trace(&format!("Unsqueeze 2D->3D axis={}", self.axis));
        let rows = input.rows;
        let cols = input.cols;
        let p = j_input.num_params;
        if self.axis == 0 {
            let mut j_out = Jacobian3D::new(1, rows, cols, p);
            for r in 0..rows {
                for c in 0..cols {
                    j_out.data[0][r][c] = j_input.data[r][c].clone();
                }
            }
            (Tensor3D::new(vec![input.data.clone()]), j_out)
        } else {
            let mut out = Vec::with_capacity(rows);
            let mut j_out = Jacobian3D::new(rows, 1, cols, p);
            for r in 0..rows {
                out.push(vec![input.data[r].clone()]);
                for c in 0..cols {
                    j_out.data[r][0][c] = j_input.data[r][c].clone();
                }
            }
            (Tensor3D::new(out), j_out)
        }
    }
    fn param_count(&self) -> usize { 0 }
    fn update_params(&mut self, _lr: f32, _grad: &[f32]) {}
    fn get_params(&self) -> Vec<f32> { vec![] }
    fn set_params(&mut self, _values: &[f32]) {}
}






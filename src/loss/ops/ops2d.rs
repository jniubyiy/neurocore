use crate::tensor::Tensor2D;
use crate::jacobian::Jacobian2D;
use super::{LossInput, LossJacobian};

impl LossInput for Tensor2D {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn rows(&self) -> usize { self.rows }
    fn cols(&self) -> usize { self.cols }
    fn to_flat(&self) -> Vec<f32> { self.data.concat() }
    fn zero_clone(&self) -> Box<dyn LossInput> {
        Box::new(Tensor2D::zeros(self.rows, self.cols))
    }
    fn fill_from_flat(&mut self, data: &[f32]) {
        assert_eq!(data.len(), self.rows * self.cols, "Tensor2D::fill_from_flat size mismatch");
        for r in 0..self.rows {
            let start = r * self.cols;
            self.data[r].copy_from_slice(&data[start..start + self.cols]);
        }
    }
}

impl LossJacobian for Jacobian2D {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn params(&self) -> usize { self.num_params }
    fn rows(&self) -> usize { self.rows }
    fn cols(&self) -> usize { self.out_features }

    fn to_flat(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.rows * self.out_features * self.num_params);
        for row in &self.data {
            for col in row {
                flat.extend_from_slice(col);
            }
        }
        flat
    }

    fn zero_clone(&self) -> Box<dyn LossJacobian> {
        Box::new(Jacobian2D::new(self.rows, self.out_features, self.num_params))
    }

    fn fill_from_flat(&mut self, data: &[f32]) {
        assert_eq!(data.len(), self.rows * self.out_features * self.num_params,
            "Jacobian2D::fill_from_flat size mismatch");
        let mut idx = 0;
        for r in 0..self.rows {
            for c in 0..self.out_features {
                for p in 0..self.num_params {
                    self.data[r][c][p] = data[idx];
                    idx += 1;
                }
            }
        }
    }
}
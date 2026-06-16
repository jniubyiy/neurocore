use crate::tensor::Tensor1D;
use crate::jacobian::Jacobian;
use super::{LossInput, LossJacobian};

impl LossInput for Tensor1D {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn rows(&self) -> usize { 1 }
    fn cols(&self) -> usize { self.len() }
    fn to_flat(&self) -> Vec<f32> { self.data.clone() }
    fn zero_clone(&self) -> Box<dyn LossInput> {
        Box::new(Tensor1D::zeros(self.cols()))
    }
    fn fill_from_flat(&mut self, data: &[f32]) {
        assert_eq!(data.len(), self.cols(), "Tensor1D::fill_from_flat size mismatch");
        self.data.copy_from_slice(data);
    }
}

impl LossJacobian for Jacobian {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn rows(&self) -> usize { 1 }
    fn cols(&self) -> usize { self.out_features }
    fn params(&self) -> usize { self.num_params }

    fn to_flat(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.out_features * self.num_params);
        for row in &self.data {
            flat.extend_from_slice(row);
        }
        flat
    }

    fn zero_clone(&self) -> Box<dyn LossJacobian> {
        Box::new(Jacobian::new(self.out_features, self.num_params))
    }

    fn fill_from_flat(&mut self, data: &[f32]) {
        assert_eq!(data.len(), self.out_features * self.num_params,
            "Jacobian::fill_from_flat size mismatch");
        for i in 0..self.out_features {
            let start = i * self.num_params;
            self.data[i].copy_from_slice(&data[start..start + self.num_params]);
        }
    }
}
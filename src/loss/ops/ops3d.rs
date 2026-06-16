use crate::tensor::Tensor3D;
use crate::jacobian::Jacobian3D;
use super::{LossInput, LossJacobian};

impl LossInput for Tensor3D {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn rows(&self) -> usize { self.depth * self.rows }
    fn cols(&self) -> usize { self.cols }
    fn to_flat(&self) -> Vec<f32> {
        self.data.iter().flat_map(|d| d.iter().flat_map(|r| r.iter().copied())).collect()
    }
    fn zero_clone(&self) -> Box<dyn LossInput> {
        Box::new(Tensor3D::zeros(self.depth, self.rows, self.cols))
    }
    fn fill_from_flat(&mut self, data: &[f32]) {
        assert_eq!(data.len(), self.depth * self.rows * self.cols, "Tensor3D::fill_from_flat size mismatch");
        let mut idx = 0;
        for d in 0..self.depth {
            for r in 0..self.rows {
                for c in 0..self.cols {
                    self.data[d][r][c] = data[idx];
                    idx += 1;
                }
            }
        }
    }
}

impl LossJacobian for Jacobian3D {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn params(&self) -> usize { self.num_params }
    fn rows(&self) -> usize { self.depth * self.rows }
    fn cols(&self) -> usize { self.out_features }

    fn to_flat(&self) -> Vec<f32> {
        let mut flat = Vec::new();
        for d in 0..self.depth {
            for r in 0..self.rows {
                for c in 0..self.out_features {
                    flat.extend_from_slice(&self.data[d][r][c]);
                }
            }
        }
        flat
    }

    fn zero_clone(&self) -> Box<dyn LossJacobian> {
        Box::new(Jacobian3D::new(self.depth, self.rows, self.out_features, self.num_params))
    }

    fn fill_from_flat(&mut self, data: &[f32]) {
        let expected = self.depth * self.rows * self.out_features * self.num_params;
        assert_eq!(data.len(), expected, "Jacobian3D::fill_from_flat size mismatch");
        let mut idx = 0;
        for d in 0..self.depth {
            for r in 0..self.rows {
                for c in 0..self.out_features {
                    for p in 0..self.num_params {
                        self.data[d][r][c][p] = data[idx];
                        idx += 1;
                    }
                }
            }
        }
    }
}
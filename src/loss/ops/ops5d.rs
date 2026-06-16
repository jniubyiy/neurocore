use crate::tensor::Tensor5D;
use crate::jacobian::Jacobian5D;
use super::{LossInput, LossJacobian};

impl LossInput for Tensor5D {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn rows(&self) -> usize { self.outer * self.dim1 * self.depth * self.rows }
    fn cols(&self) -> usize { self.cols }
    fn to_flat(&self) -> Vec<f32> {
        self.data.iter()
            .flat_map(|o| o.iter().flat_map(|d1| d1.iter().flat_map(|d| d.iter().flat_map(|r| r.iter().copied()))))
            .collect()
    }
    fn zero_clone(&self) -> Box<dyn LossInput> {
        Box::new(Tensor5D::zeros(self.outer, self.dim1, self.depth, self.rows, self.cols))
    }
    fn fill_from_flat(&mut self, data: &[f32]) {
        let total = self.outer * self.dim1 * self.depth * self.rows * self.cols;
        assert_eq!(data.len(), total, "Tensor5D::fill_from_flat size mismatch");
        let mut idx = 0;
        for o in 0..self.outer {
            for d1 in 0..self.dim1 {
                for d in 0..self.depth {
                    for r in 0..self.rows {
                        for c in 0..self.cols {
                            self.data[o][d1][d][r][c] = data[idx];
                            idx += 1;
                        }
                    }
                }
            }
        }
    }
}

impl LossJacobian for Jacobian5D {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn params(&self) -> usize { self.num_params }
    fn rows(&self) -> usize { self.outer * self.dim1 * self.depth * self.rows }
    fn cols(&self) -> usize { self.out_features }

    fn to_flat(&self) -> Vec<f32> {
        let mut flat = Vec::new();
        for o in 0..self.outer {
            for d1 in 0..self.dim1 {
                for d in 0..self.depth {
                    for r in 0..self.rows {
                        for c in 0..self.out_features {
                            flat.extend_from_slice(&self.data[o][d1][d][r][c]);
                        }
                    }
                }
            }
        }
        flat
    }

    fn zero_clone(&self) -> Box<dyn LossJacobian> {
        Box::new(Jacobian5D::new(self.outer, self.dim1, self.depth, self.rows, self.out_features, self.num_params))
    }

    fn fill_from_flat(&mut self, data: &[f32]) {
        let total = self.outer * self.dim1 * self.depth * self.rows * self.out_features * self.num_params;
        assert_eq!(data.len(), total, "Jacobian5D::fill_from_flat size mismatch");
        let mut idx = 0;
        for o in 0..self.outer {
            for d1 in 0..self.dim1 {
                for d in 0..self.depth {
                    for r in 0..self.rows {
                        for c in 0..self.out_features {
                            for p in 0..self.num_params {
                                self.data[o][d1][d][r][c][p] = data[idx];
                                idx += 1;
                            }
                        }
                    }
                }
            }
        }
    }
}
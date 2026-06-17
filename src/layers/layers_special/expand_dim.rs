use crate::tensor::{Tensor1D, Tensor2D, Tensor3D};
use super::DimExpand;

pub struct Unsqueeze {
    axis: usize,
}

impl Unsqueeze {
    pub fn new(axis: usize) -> Self {
        Unsqueeze { axis }
    }
}

// 1D -> 2D
impl DimExpand<Tensor1D, Tensor2D> for Unsqueeze {
    fn expand(&self, input: &Tensor1D) -> Tensor2D {
        let len = input.len();
        if self.axis == 0 {
            Tensor2D::new(vec![input.data.clone()])
        } else {
            let mut data = Vec::with_capacity(len);
            for i in 0..len {
                data.push(vec![input.data[i]]);
            }
            Tensor2D::new(data)
        }
    }
}

// 2D -> 3D
impl DimExpand<Tensor2D, Tensor3D> for Unsqueeze {
    fn expand(&self, input: &Tensor2D) -> Tensor3D {
        let rows = input.rows;
        if self.axis == 0 {
            Tensor3D::new(vec![input.data.clone()])
        } else {
            let mut out = Vec::with_capacity(rows);
            for r in 0..rows {
                out.push(vec![input.data[r].clone()]);
            }
            Tensor3D::new(out)
        }
    }
}






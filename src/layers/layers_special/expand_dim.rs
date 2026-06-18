use crate::tensor::{Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D};
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
        let len = input.dim1();
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
        let dim1 = input.dim1;
        if self.axis == 0 {
            Tensor3D::new(vec![input.data.clone()])
        } else {
            let mut out = Vec::with_capacity(dim1);
            for i in 0..dim1 {
                out.push(vec![input.data[i].clone()]);
            }
            Tensor3D::new(out)
        }
    }
}

// 3D -> 4D
impl DimExpand<Tensor3D, Tensor4D> for Unsqueeze {
    fn expand(&self, input: &Tensor3D) -> Tensor4D {
        if self.axis == 0 {
            Tensor4D::new(vec![input.data.clone()])
        } else {
            panic!("Unsqueeze 3D->4D only supports axis=0 for now");
        }
    }
}

// 4D -> 5D
impl DimExpand<Tensor4D, Tensor5D> for Unsqueeze {
    fn expand(&self, input: &Tensor4D) -> Tensor5D {
        if self.axis == 0 {
            Tensor5D::new(vec![input.data.clone()])
        } else {
            panic!("Unsqueeze 4D->5D only supports axis=0 for now");
        }
    }
}






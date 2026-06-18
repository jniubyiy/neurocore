// src/dispatchers/auto_model/dim_change.rs

use crate::tensor::{Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D};
use crate::layers::layers_special::{DimExpand, DimReduce};

#[derive(Clone)]
pub enum DynamicTensor {
    Dim1(Tensor1D),
    Dim2(Tensor2D),
    Dim3(Tensor3D),
    Dim4(Tensor4D),
    Dim5(Tensor5D),
}

pub fn unsqueeze(tensor: DynamicTensor, axis: usize) -> DynamicTensor {
    match tensor {
        DynamicTensor::Dim1(t) => DynamicTensor::Dim2(crate::layers::Unsqueeze::new(axis).expand(&t)),
        DynamicTensor::Dim2(t) => DynamicTensor::Dim3(crate::layers::Unsqueeze::new(axis).expand(&t)),
        DynamicTensor::Dim3(t) => DynamicTensor::Dim4(crate::layers::Unsqueeze::new(axis).expand(&t)),
        DynamicTensor::Dim4(t) => DynamicTensor::Dim5(crate::layers::Unsqueeze::new(axis).expand(&t)),
        DynamicTensor::Dim5(_) => panic!("Cannot unsqueeze a 5D tensor"),
    }
}

pub fn unsqueeze_backward(tensor: DynamicTensor, axis: usize) -> DynamicTensor {
    match tensor {
        DynamicTensor::Dim2(t) => DynamicTensor::Dim1(crate::layers::ReduceMean::new(axis).reduce(&t)),
        DynamicTensor::Dim3(t) => DynamicTensor::Dim2(crate::layers::ReduceMean::new(axis).reduce(&t)),
        DynamicTensor::Dim4(t) => DynamicTensor::Dim3(crate::layers::ReduceMean::new(axis).reduce(&t)),
        DynamicTensor::Dim5(t) => DynamicTensor::Dim4(crate::layers::ReduceMean::new(axis).reduce(&t)),
        _ => panic!("Invalid tensor dimension for unsqueeze_backward"),
    }
}

pub fn reduce_mean(tensor: DynamicTensor, axis: usize) -> DynamicTensor {
    match tensor {
        DynamicTensor::Dim2(t) => DynamicTensor::Dim1(crate::layers::ReduceMean::new(axis).reduce(&t)),
        DynamicTensor::Dim3(t) => DynamicTensor::Dim2(crate::layers::ReduceMean::new(axis).reduce(&t)),
        DynamicTensor::Dim4(t) => DynamicTensor::Dim3(crate::layers::ReduceMean::new(axis).reduce(&t)),
        DynamicTensor::Dim5(t) => DynamicTensor::Dim4(crate::layers::ReduceMean::new(axis).reduce(&t)),
        DynamicTensor::Dim1(_) => panic!("Cannot reduce a 1D tensor"),
    }
}

pub fn reduce_mean_backward(tensor: DynamicTensor, axis: usize) -> DynamicTensor {
    match tensor {
        DynamicTensor::Dim1(t) => DynamicTensor::Dim2(crate::layers::Unsqueeze::new(axis).expand(&t)),
        DynamicTensor::Dim2(t) => DynamicTensor::Dim3(crate::layers::Unsqueeze::new(axis).expand(&t)),
        DynamicTensor::Dim3(t) => DynamicTensor::Dim4(crate::layers::Unsqueeze::new(axis).expand(&t)),
        DynamicTensor::Dim4(t) => DynamicTensor::Dim5(crate::layers::Unsqueeze::new(axis).expand(&t)),
        _ => panic!("Invalid tensor dimension for reduce_mean_backward"),
    }
}
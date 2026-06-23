// src/compute_manager/dim_change.rs

use crate::tensor::{Tensor2D, Tensor3D, Tensor4D, Tensor5D};
use crate::layers::layers_special::reduce_dim::ReduceMean;
use crate::layers::layers_special::expand_dim::Unsqueeze;
use crate::layers::layers_special::DimExpand;
use crate::layers::layers_special::DimReduce;

#[derive(Clone)]
pub enum DynamicTensor {
    Dim1(Tensor2D),   // батч + 1 ось
    Dim2(Tensor3D),   // батч + 2 оси
    Dim3(Tensor4D),   // батч + 3 оси
    Dim4(Tensor5D),   // батч + 4 оси
}

impl DynamicTensor {
    /// Размер батча (первая размерность).
    pub fn batch_size(&self) -> usize {
        match self {
            DynamicTensor::Dim1(t) => t.dim1,
            DynamicTensor::Dim2(t) => t.dim1,
            DynamicTensor::Dim3(t) => t.dim1,
            DynamicTensor::Dim4(t) => t.dim1,
        }
    }

    /// Извлекает один образец (получается тензор с батчем 1? Или чистый вектор? Поскольку у нас нет тензоров без батча,
    /// образец – это тоже тензор с батчем 1 того же типа. Уменьшать размерность не нужно, просто берем строку.
    pub fn sample(&self, idx: usize) -> DynamicTensor {
        match self {
            DynamicTensor::Dim1(t) => {
                DynamicTensor::Dim1(Tensor2D::new(vec![t.data[idx].clone()]))
            }
            DynamicTensor::Dim2(t) => {
                DynamicTensor::Dim2(Tensor3D::new(vec![t.data[idx].clone()]))
            }
            DynamicTensor::Dim3(t) => {
                DynamicTensor::Dim3(Tensor4D::new(vec![t.data[idx].clone()]))
            }
            DynamicTensor::Dim4(t) => {
                DynamicTensor::Dim4(Tensor5D::new(vec![t.data[idx].clone()]))
            }
        }
    }

    /// Количество элементов в последней оси (признаки).
    pub fn features(&self) -> usize {
        match self {
            DynamicTensor::Dim1(t) => t.dim2,
            DynamicTensor::Dim2(t) => t.dim3,
            DynamicTensor::Dim3(t) => t.dim4,
            DynamicTensor::Dim4(t) => t.dim5,
        }
    }

    /// Сериализует все элементы тензора в один плоский вектор (batch-major).
    pub fn to_flat(&self) -> Vec<f32> {
        let mut buf = Vec::new();
        self.write_to_flat(&mut buf);
        buf
    }

    pub fn write_to_flat(&self, buf: &mut Vec<f32>) {
        buf.clear();
        match self {
            DynamicTensor::Dim1(t) => {
                buf.reserve(t.dim1 * t.dim2);
                for row in &t.data { buf.extend_from_slice(row); }
            }
            DynamicTensor::Dim2(t) => {
                let cap = t.dim1 * t.dim2 * t.dim3;
                buf.reserve(cap);
                for plane in &t.data { for row in plane { buf.extend_from_slice(row); } }
            }
            DynamicTensor::Dim3(t) => {
                let cap = t.dim1 * t.dim2 * t.dim3 * t.dim4;
                buf.reserve(cap);
                for vol in &t.data { for plane in vol { for row in plane { buf.extend_from_slice(row); } } }
            }
            DynamicTensor::Dim4(t) => {
                let cap = t.dim1 * t.dim2 * t.dim3 * t.dim4 * t.dim5;
                buf.reserve(cap);
                for hyper in &t.data { for vol in hyper { for plane in vol { for row in plane { buf.extend_from_slice(row); } } } }
            }
        }
    }

    pub fn from_flat(shape: &DynamicTensor, data: Vec<f32>) -> DynamicTensor {
        let mut dest = shape.clone();
        Self::from_flat_into(shape, &data, &mut dest);
        dest
    }

    pub fn from_flat_into(shape: &DynamicTensor, data: &[f32], dest: &mut DynamicTensor) {
        match (shape, dest) {
            (DynamicTensor::Dim1(orig), DynamicTensor::Dim1(ref mut t)) => {
                let features = orig.dim2;
                assert_eq!(data.len(), orig.dim1 * features);
                for (r, row) in t.data.iter_mut().enumerate() {
                    let start = r * features;
                    row.copy_from_slice(&data[start..start + features]);
                }
            }
            (DynamicTensor::Dim2(orig), DynamicTensor::Dim2(ref mut t)) => {
                let features = orig.dim3;
                assert_eq!(data.len(), orig.dim1 * orig.dim2 * features);
                let mut offset = 0;
                for plane in t.data.iter_mut() {
                    for row in plane.iter_mut() {
                        row.copy_from_slice(&data[offset..offset + features]);
                        offset += features;
                    }
                }
            }
            (DynamicTensor::Dim3(orig), DynamicTensor::Dim3(ref mut t)) => {
                let features = orig.dim4;
                assert_eq!(data.len(), orig.dim1 * orig.dim2 * orig.dim3 * features);
                let mut offset = 0;
                for vol in t.data.iter_mut() {
                    for plane in vol.iter_mut() {
                        for row in plane.iter_mut() {
                            row.copy_from_slice(&data[offset..offset + features]);
                            offset += features;
                        }
                    }
                }
            }
            (DynamicTensor::Dim4(orig), DynamicTensor::Dim4(ref mut t)) => {
                let features = orig.dim5;
                assert_eq!(data.len(), orig.dim1 * orig.dim2 * orig.dim3 * orig.dim4 * features);
                let mut offset = 0;
                for hyper in t.data.iter_mut() {
                    for vol in hyper.iter_mut() {
                        for plane in vol.iter_mut() {
                            for row in plane.iter_mut() {
                                row.copy_from_slice(&data[offset..offset + features]);
                                offset += features;
                            }
                        }
                    }
                }
            }
            _ => panic!("Shape mismatch in from_flat_into"),
        }
    }
}

// ------------------ Функции reshape ------------------

pub fn unsqueeze_to(tensor: DynamicTensor, target_dims: Vec<usize>) -> DynamicTensor {
    match tensor {
        DynamicTensor::Dim1(t) => {
            let expander = Unsqueeze::with_target_dims(target_dims);
            DynamicTensor::Dim2(expander.expand(&t))
        }
        DynamicTensor::Dim2(t) => {
            let expander = Unsqueeze::with_target_dims(target_dims);
            DynamicTensor::Dim3(expander.expand(&t))
        }
        DynamicTensor::Dim3(t) => {
            let expander = Unsqueeze::with_target_dims(target_dims);
            DynamicTensor::Dim4(expander.expand(&t))
        }
        DynamicTensor::Dim4(_) => panic!("Cannot unsqueeze a 4D tensor (max)"),
    }
}

pub fn reduce_to(tensor: DynamicTensor, target_dims: Vec<usize>) -> DynamicTensor {
    match tensor {
        DynamicTensor::Dim2(t) => {
            let reducer = ReduceMean::with_target_dims(target_dims);
            DynamicTensor::Dim1(reducer.reduce(&t))
        }
        DynamicTensor::Dim3(t) => {
            let reducer = ReduceMean::with_target_dims(target_dims);
            DynamicTensor::Dim2(reducer.reduce(&t))
        }
        DynamicTensor::Dim4(t) => {
            let reducer = ReduceMean::with_target_dims(target_dims);
            DynamicTensor::Dim3(reducer.reduce(&t))
        }
        DynamicTensor::Dim1(_) => panic!("Cannot reduce a 1D tensor"),
    }
}
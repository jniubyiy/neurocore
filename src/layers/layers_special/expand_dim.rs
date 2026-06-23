// src/layers/layers_special/expand_dim.rs
// Повышение размерности (reshape вверх). Батч сохраняется.
// Все элементы после батча собираются в плоский массив и перепаковываются в target_dims.

use crate::compute_manager::dim_change::{self, DynamicTensor};
use crate::compute_manager::graph::types::DynamicContext;
use crate::model_plan::param_store::ParamSlice;
use crate::layers::UniversalLayer;
use crate::tensor::{Tensor2D, Tensor3D, Tensor4D, Tensor5D};
use super::DimExpand;

pub struct Unsqueeze {
    /// Размеры осей после батча в результирующем тензоре.
    /// Длина вектора = новая размерность Dim.
    pub target_dims: Vec<usize>,
}

impl Unsqueeze {
    pub fn with_target_dims(target_dims: Vec<usize>) -> Self {
        Self { target_dims }
    }
}

// 2D -> 3D
impl DimExpand<Tensor2D, Tensor3D> for Unsqueeze {
    fn expand(&self, input: &Tensor2D) -> Tensor3D {
        let batch = input.dim1;
        let flat_len = input.dim2;
        assert_eq!(self.target_dims.len(), 2);
        let d1 = self.target_dims[0];
        let d2 = self.target_dims[1];
        assert_eq!(d1 * d2, flat_len);
        let mut data = Vec::with_capacity(batch);
        for b in 0..batch {
            let row = &input.data[b];
            let mut plane = Vec::with_capacity(d1);
            for i in 0..d1 {
                plane.push(row[i * d2..(i + 1) * d2].to_vec());
            }
            data.push(plane);
        }
        Tensor3D::new(data)
    }
}

// 3D -> 4D
impl DimExpand<Tensor3D, Tensor4D> for Unsqueeze {
    fn expand(&self, input: &Tensor3D) -> Tensor4D {
        let batch = input.dim1;
        let d1_old = input.dim2;
        let d2_old = input.dim3;
        assert_eq!(self.target_dims.len(), 3);
        let d1 = self.target_dims[0];
        let d2 = self.target_dims[1];
        let d3 = self.target_dims[2];
        let total = d1_old * d2_old;
        assert_eq!(d1 * d2 * d3, total);
        let mut data = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut flat = Vec::with_capacity(total);
            for i in 0..d1_old {
                flat.extend_from_slice(&input.data[b][i]);
            }
            let mut volume = Vec::with_capacity(d1);
            let mut offset = 0;
            for _ in 0..d1 {
                let mut plane = Vec::with_capacity(d2);
                for _ in 0..d2 {
                    plane.push(flat[offset..offset + d3].to_vec());
                    offset += d3;
                }
                volume.push(plane);
            }
            data.push(volume);
        }
        Tensor4D::new(data)
    }
}

// 4D -> 5D
impl DimExpand<Tensor4D, Tensor5D> for Unsqueeze {
    fn expand(&self, input: &Tensor4D) -> Tensor5D {
        let batch = input.dim1;
        let d1_old = input.dim2;
        let d2_old = input.dim3;
        let d3_old = input.dim4;
        assert_eq!(self.target_dims.len(), 4);
        let d1 = self.target_dims[0];
        let d2 = self.target_dims[1];
        let d3 = self.target_dims[2];
        let d4 = self.target_dims[3];
        let total = d1_old * d2_old * d3_old;
        assert_eq!(d1 * d2 * d3 * d4, total);
        let mut data = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut flat = Vec::with_capacity(total);
            for i in 0..d1_old {
                for j in 0..d2_old {
                    flat.extend_from_slice(&input.data[b][i][j]);
                }
            }
            let mut hyper = Vec::with_capacity(d1);
            let mut offset = 0;
            for _ in 0..d1 {
                let mut volume = Vec::with_capacity(d2);
                for _ in 0..d2 {
                    let mut plane = Vec::with_capacity(d3);
                    for _ in 0..d3 {
                        plane.push(flat[offset..offset + d4].to_vec());
                        offset += d4;
                    }
                    volume.push(plane);
                }
                hyper.push(volume);
            }
            data.push(hyper);
        }
        Tensor5D::new(data)
    }
}

impl UniversalLayer for Unsqueeze {
    fn forward(
        &self,
        input: &DynamicTensor,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (DynamicTensor, DynamicContext) {
        let out = dim_change::unsqueeze_to(input.clone(), self.target_dims.clone());
        let dummy_ctx = DynamicContext::Ctx1D(
            crate::layers::context1d::LayerContext1D::Linear {
                input: Tensor2D::zeros(1, 0),
            },
        );
        (out, dummy_ctx)
    }

    fn backward(
        &self,
        _ctx: &DynamicContext,
        delta: &DynamicTensor,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (DynamicTensor, Vec<f32>) {
        let grad_input = dim_change::reduce_to(delta.clone(), self.target_dims.clone());
        (grad_input, vec![])
    }

    fn param_len(&self) -> usize { 0 }
    fn input_features(&self) -> usize { 0 }
    fn output_features(&self) -> usize { 0 }

    fn total_tasks(&self, _input: &DynamicTensor) -> usize { 0 }
    fn execute_tasks(
        &self,
        _input: &DynamicTensor,
        _output: &mut DynamicTensor,
        _task_offset: usize,
        _task_count: usize,
        _params: &[f32],
        _slice: &ParamSlice,
    ) {}
    fn create_sample_context(
        &self,
        _input_sample: &DynamicTensor,
        _output_sample: &DynamicTensor,
    ) -> DynamicContext {
        DynamicContext::Ctx1D(
            crate::layers::context1d::LayerContext1D::Linear {
                input: Tensor2D::zeros(1, 0),
            },
        )
    }
    fn output_tensor_shape(&self, input: &DynamicTensor) -> DynamicTensor {
        dim_change::unsqueeze_to(input.clone(), self.target_dims.clone())
    }
}






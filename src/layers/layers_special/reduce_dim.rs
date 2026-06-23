// src/layers/layers_special/reduce_dim.rs
// Понижение размерности (reshape вниз). Батч сохраняется.
// Все элементы после батча собираются в плоский массив и перепаковываются в target_dims.

use crate::compute_manager::dim_change::{self, DynamicTensor};
use crate::compute_manager::graph::types::DynamicContext;
use crate::model_plan::param_store::ParamSlice;
use crate::layers::UniversalLayer;
use crate::tensor::{Tensor2D, Tensor3D, Tensor4D, Tensor5D};
use super::DimReduce;

pub struct ReduceMean {
    /// Размеры осей после батча в результирующем тензоре.
    /// Длина вектора = результирующая размерность Dim.
    pub target_dims: Vec<usize>,
}

impl ReduceMean {
    pub fn with_target_dims(target_dims: Vec<usize>) -> Self {
        Self { target_dims }
    }
}

// 3D -> 2D (Dim2 -> Dim1)
impl DimReduce<Tensor3D, Tensor2D> for ReduceMean {
    fn reduce(&self, input: &Tensor3D) -> Tensor2D {
        let batch = input.dim1;
        let d1_old = input.dim2;
        let d2_old = input.dim3;
        assert_eq!(self.target_dims.len(), 1, "ReduceMean 3D->2D: target_dims должен содержать 1 число (новую последнюю ось)");
        let new_len = self.target_dims[0];
        assert_eq!(d1_old * d2_old, new_len, "Произведение старых размеров не совпадает с целевым размером");
        let mut data = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut flat = Vec::with_capacity(new_len);
            for i in 0..d1_old {
                flat.extend_from_slice(&input.data[b][i]);
            }
            data.push(flat);
        }
        Tensor2D::new(data)
    }
}

// 4D -> 3D (Dim3 -> Dim2)
impl DimReduce<Tensor4D, Tensor3D> for ReduceMean {
    fn reduce(&self, input: &Tensor4D) -> Tensor3D {
        let batch = input.dim1;
        let d1_old = input.dim2;
        let d2_old = input.dim3;
        let d3_old = input.dim4;
        assert_eq!(self.target_dims.len(), 2, "ReduceMean 4D->3D: target_dims должен содержать 2 числа [new_d1, new_d2]");
        let new_d1 = self.target_dims[0];
        let new_d2 = self.target_dims[1];
        let total = d1_old * d2_old * d3_old;
        assert_eq!(new_d1 * new_d2, total, "Произведение целевых размеров не совпадает с общим количеством элементов на батч");
        let mut data = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut flat = Vec::with_capacity(total);
            for i in 0..d1_old {
                for j in 0..d2_old {
                    flat.extend_from_slice(&input.data[b][i][j]);
                }
            }
            // упаковываем в new_d1 строк длиной new_d2
            let mut volume = Vec::with_capacity(new_d1);
            let mut idx = 0;
            for _ in 0..new_d1 {
                volume.push(flat[idx..idx + new_d2].to_vec());
                idx += new_d2;
            }
            data.push(volume);
        }
        Tensor3D::new(data)
    }
}

// 5D -> 4D (Dim4 -> Dim3)
impl DimReduce<Tensor5D, Tensor4D> for ReduceMean {
    fn reduce(&self, input: &Tensor5D) -> Tensor4D {
        let batch = input.dim1;
        let d1_old = input.dim2;
        let d2_old = input.dim3;
        let d3_old = input.dim4;
        let d4_old = input.dim5;
        assert_eq!(self.target_dims.len(), 3, "ReduceMean 5D->4D: target_dims должен содержать 3 числа [new_d1, new_d2, new_d3]");
        let new_d1 = self.target_dims[0];
        let new_d2 = self.target_dims[1];
        let new_d3 = self.target_dims[2];
        let total = d1_old * d2_old * d3_old * d4_old;
        assert_eq!(new_d1 * new_d2 * new_d3, total, "Произведение целевых размеров не совпадает с общим количеством элементов на батч");
        let mut data = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut flat = Vec::with_capacity(total);
            for i in 0..d1_old {
                for j in 0..d2_old {
                    for k in 0..d3_old {
                        flat.extend_from_slice(&input.data[b][i][j][k]);
                    }
                }
            }
            let mut hyper = Vec::with_capacity(new_d1);
            let mut idx = 0;
            for _ in 0..new_d1 {
                let mut volume = Vec::with_capacity(new_d2);
                for _ in 0..new_d2 {
                    volume.push(flat[idx..idx + new_d3].to_vec());
                    idx += new_d3;
                }
                hyper.push(volume);
            }
            data.push(hyper);
        }
        Tensor4D::new(data)
    }
}

// ================== UniversalLayer ==================
impl UniversalLayer for ReduceMean {
    fn forward(
        &self,
        input: &DynamicTensor,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (DynamicTensor, DynamicContext) {
        let out = dim_change::reduce_to(input.clone(), self.target_dims.clone());
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
        let grad_input = dim_change::unsqueeze_to(delta.clone(), self.target_dims.clone());
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
        dim_change::reduce_to(input.clone(), self.target_dims.clone())
    }
}






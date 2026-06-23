// src/layers/combiner_connector/combiner_connector.rs

use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::graph::types::DynamicContext;
use crate::model_plan::param_store::ParamSlice;
use crate::tensor::{Tensor2D, Tensor3D, Tensor4D, Tensor5D};

pub(crate) mod dim {
    use crate::tensor::{Tensor2D, Tensor3D, Tensor4D, Tensor5D};
    use crate::model_plan::param_store::ParamSlice;
    use crate::layers::context1d::{Layer, LayerContext1D};
    use crate::layers::context2d::{Layer2D, LayerContext as LayerContext2D};
    use crate::layers::context3d::{Layer3D, LayerContext3D};
    use crate::layers::context4d::{Layer4D, LayerContext4D};

    // ======================== Dim1 (одна ось после батча, работает с Tensor2D) ========================
    pub struct CombinerConnector2D {
        pub input_dims: Vec<usize>,
        pub output_dim: usize,
    }

    impl Layer for CombinerConnector2D {
        fn input_dim1s(&self) -> Vec<usize> { self.input_dims.clone() }
        fn output_dim1s(&self) -> Vec<usize> { vec![self.output_dim] }

        fn forward_into(
            &self,
            inputs: &[Tensor2D],
            _params: &[f32],
            _slice: &ParamSlice,
            out_bufs: &mut [Vec<f32>],
        ) -> Vec<LayerContext1D> {
            assert_eq!(inputs.len(), self.input_dims.len());
            assert_eq!(out_bufs.len(), 1);
            let batch = inputs[0].dim1;
            for (i, input) in inputs.iter().enumerate() {
                assert_eq!(input.dim1, batch);
                assert_eq!(input.dim2, self.input_dims[i]);
            }
            let out = &mut out_bufs[0];
            assert_eq!(out.len(), self.output_dim);

            for r in 0..batch {
                let mut offset = 0;
                for input in inputs.iter() {
                    let len = input.dim2;
                    out[offset..offset + len].copy_from_slice(&input.data[r]);
                    offset += len;
                }
            }

            vec![LayerContext1D::CombinerConnector { inputs: inputs.to_vec() }]
        }

        fn backward(
            &self,
            ctxs: &[LayerContext1D],
            deltas: &[Tensor2D],
            _params: &[f32],
            _slice: &ParamSlice,
        ) -> (Vec<Tensor2D>, Vec<f32>) {
            assert_eq!(deltas.len(), 1);
            let delta = &deltas[0].data;
            let inputs = match &ctxs[0] {
                LayerContext1D::CombinerConnector { inputs } => inputs,
                _ => panic!("Expected CombinerConnector context"),
            };
            assert_eq!(inputs.len(), self.input_dims.len());
            let mut in_grads = Vec::new();
            let mut offset = 0;
            for input in inputs {
                let len = input.dim2;
                let d_i = delta[offset..offset + len].to_vec();
                in_grads.push(Tensor2D::new(d_i)); // исправлено: d_i - это уже Vec<Vec<f32>>
                offset += len;
            }
            (in_grads, vec![])
        }

        fn param_len(&self) -> usize { 0 }
    }

    // ======================== Dim2 (две оси после батча, работает с Tensor3D) ========================
    pub struct CombinerConnector3D {
        pub input_dims: Vec<usize>,
        pub output_dim: usize,
    }

    impl Layer2D for CombinerConnector3D {
        fn input_dims(&self) -> Vec<usize> { self.input_dims.clone() }
        fn output_dims(&self) -> Vec<usize> { vec![self.output_dim] }

        fn forward_into(
            &self,
            inputs: &[Tensor3D],
            _params: &[f32],
            _slice: &ParamSlice,
            out_bufs: &mut [Vec<Vec<Vec<f32>>>],
        ) -> Vec<LayerContext2D> {
            assert_eq!(inputs.len(), self.input_dims.len());
            assert_eq!(out_bufs.len(), 1);
            let batch = inputs[0].dim1;
            let dim2  = inputs[0].dim2;
            for (i, input) in inputs.iter().enumerate() {
                assert_eq!(input.dim1, batch);
                assert_eq!(input.dim2, dim2);
                assert_eq!(input.dim3, self.input_dims[i]);
            }
            let out = &mut out_bufs[0];
            assert_eq!(out.len(), batch);
            assert_eq!(out[0].len(), dim2);
            assert_eq!(out[0][0].len(), self.output_dim);

            for i in 0..batch {
                for j in 0..dim2 {
                    let mut offset = 0;
                    for input in inputs.iter() {
                        let len = input.dim3;
                        out[i][j][offset..offset + len].copy_from_slice(&input.data[i][j]);
                        offset += len;
                    }
                }
            }

            vec![LayerContext2D::CombinerConnector { inputs: inputs.to_vec() }]
        }

        fn backward(
            &self,
            ctxs: &[LayerContext2D],
            deltas: &[Tensor3D],
            _params: &[f32],
            _slice: &ParamSlice,
        ) -> (Vec<Tensor3D>, Vec<f32>) {
            assert_eq!(ctxs.len(), 1);
            assert_eq!(deltas.len(), 1);
            let inputs = match &ctxs[0] {
                LayerContext2D::CombinerConnector { inputs } => inputs,
                _ => panic!("Expected CombinerConnector context"),
            };
            let delta = &deltas[0];
            let batch = delta.dim1;
            let dim2  = delta.dim2;
            let mut in_grads = Vec::new();
            let mut offset = 0;
            for input in inputs {
                let len = input.dim3;
                let mut grad = vec![vec![vec![0.0; len]; dim2]; batch];
                for i in 0..batch {
                    for j in 0..dim2 {
                        grad[i][j].copy_from_slice(&delta.data[i][j][offset..offset + len]);
                    }
                }
                in_grads.push(Tensor3D::new(grad));
                offset += len;
            }
            (in_grads, vec![])
        }

        fn param_len(&self) -> usize { 0 }
    }

    // ======================== Dim3 (три оси после батча, работает с Tensor4D) ========================
    pub struct CombinerConnector4D {
        pub input_dims: Vec<usize>,
        pub output_dim: usize,
    }

    impl Layer3D for CombinerConnector4D {
        fn input_dims(&self) -> Vec<usize> { self.input_dims.clone() }
        fn output_dims(&self) -> Vec<usize> { vec![self.output_dim] }

        fn forward_into(
            &self,
            inputs: &[Tensor4D],
            _params: &[f32],
            _slice: &ParamSlice,
            out_bufs: &mut [Vec<Vec<Vec<Vec<f32>>>>],
        ) -> Vec<LayerContext3D> {
            assert_eq!(inputs.len(), self.input_dims.len());
            assert_eq!(out_bufs.len(), 1);
            let batch = inputs[0].dim1;
            let dim2  = inputs[0].dim2;
            let dim3  = inputs[0].dim3;
            for (i, input) in inputs.iter().enumerate() {
                assert_eq!(input.dim1, batch);
                assert_eq!(input.dim2, dim2);
                assert_eq!(input.dim3, dim3);
                assert_eq!(input.dim4, self.input_dims[i]);
            }
            let out = &mut out_bufs[0];
            assert_eq!(out.len(), batch);
            assert_eq!(out[0].len(), dim2);
            assert_eq!(out[0][0].len(), dim3);
            assert_eq!(out[0][0][0].len(), self.output_dim);

            for i in 0..batch {
                for j in 0..dim2 {
                    for k in 0..dim3 {
                        let mut offset = 0;
                        for input in inputs.iter() {
                            let len = input.dim4;
                            out[i][j][k][offset..offset + len]
                                .copy_from_slice(&input.data[i][j][k]);
                            offset += len;
                        }
                    }
                }
            }

            vec![LayerContext3D::CombinerConnector { inputs: inputs.to_vec() }]
        }

        fn backward(
            &self,
            ctxs: &[LayerContext3D],
            deltas: &[Tensor4D],
            _params: &[f32],
            _slice: &ParamSlice,
        ) -> (Vec<Tensor4D>, Vec<f32>) {
            assert_eq!(ctxs.len(), 1);
            assert_eq!(deltas.len(), 1);
            let inputs = match &ctxs[0] {
                LayerContext3D::CombinerConnector { inputs } => inputs,
                _ => panic!("Expected CombinerConnector context"),
            };
            let delta = &deltas[0];
            let batch = delta.dim1;
            let dim2  = delta.dim2;
            let dim3  = delta.dim3;
            let mut in_grads = Vec::new();
            let mut offset = 0;
            for input in inputs {
                let len = input.dim4;
                let mut grad = vec![vec![vec![vec![0.0; len]; dim3]; dim2]; batch];
                for i in 0..batch {
                    for j in 0..dim2 {
                        for k in 0..dim3 {
                            grad[i][j][k]
                                .copy_from_slice(&delta.data[i][j][k][offset..offset + len]);
                        }
                    }
                }
                in_grads.push(Tensor4D::new(grad));
                offset += len;
            }
            (in_grads, vec![])
        }

        fn param_len(&self) -> usize { 0 }
    }

    // ======================== Dim4 (четыре оси после батча, работает с Tensor5D) ========================
    pub struct CombinerConnector5D {
        pub input_dims: Vec<usize>,
        pub output_dim: usize,
    }

    impl Layer4D for CombinerConnector5D {
        fn input_dims(&self) -> Vec<usize> { self.input_dims.clone() }
        fn output_dims(&self) -> Vec<usize> { vec![self.output_dim] }

        fn forward_into(
            &self,
            inputs: &[Tensor5D],
            _params: &[f32],
            _slice: &ParamSlice,
            out_bufs: &mut [Vec<Vec<Vec<Vec<Vec<f32>>>>>],
        ) -> Vec<LayerContext4D> {
            assert_eq!(inputs.len(), self.input_dims.len());
            assert_eq!(out_bufs.len(), 1);
            let batch = inputs[0].dim1;
            let dim2  = inputs[0].dim2;
            let dim3  = inputs[0].dim3;
            let dim4  = inputs[0].dim4;
            for (i, input) in inputs.iter().enumerate() {
                assert_eq!(input.dim1, batch);
                assert_eq!(input.dim2, dim2);
                assert_eq!(input.dim3, dim3);
                assert_eq!(input.dim4, dim4);
                assert_eq!(input.dim5, self.input_dims[i]);
            }
            let out = &mut out_bufs[0];
            assert_eq!(out.len(), batch);
            assert_eq!(out[0].len(), dim2);
            assert_eq!(out[0][0].len(), dim3);
            assert_eq!(out[0][0][0].len(), dim4);
            assert_eq!(out[0][0][0][0].len(), self.output_dim);

            for i in 0..batch {
                for j in 0..dim2 {
                    for k in 0..dim3 {
                        for l in 0..dim4 {
                            let mut offset = 0;
                            for input in inputs.iter() {
                                let len = input.dim5;
                                out[i][j][k][l][offset..offset + len]
                                    .copy_from_slice(&input.data[i][j][k][l]);
                                offset += len;
                            }
                        }
                    }
                }
            }

            vec![LayerContext4D::CombinerConnector { inputs: inputs.to_vec() }]
        }

        fn backward(
            &self,
            ctxs: &[LayerContext4D],
            deltas: &[Tensor5D],
            _params: &[f32],
            _slice: &ParamSlice,
        ) -> (Vec<Tensor5D>, Vec<f32>) {
            assert_eq!(ctxs.len(), 1);
            assert_eq!(deltas.len(), 1);
            let inputs = match &ctxs[0] {
                LayerContext4D::CombinerConnector { inputs } => inputs,
                _ => panic!("Expected CombinerConnector context"),
            };
            let delta = &deltas[0];
            let batch = delta.dim1;
            let dim2  = delta.dim2;
            let dim3  = delta.dim3;
            let dim4  = delta.dim4;
            let mut in_grads = Vec::new();
            let mut offset = 0;
            for input in inputs {
                let len = input.dim5;
                let mut grad = vec![vec![vec![vec![vec![0.0; len]; dim4]; dim3]; dim2]; batch];
                for i in 0..batch {
                    for j in 0..dim2 {
                        for k in 0..dim3 {
                            for l in 0..dim4 {
                                grad[i][j][k][l]
                                    .copy_from_slice(&delta.data[i][j][k][l][offset..offset + len]);
                            }
                        }
                    }
                }
                in_grads.push(Tensor5D::new(grad));
                offset += len;
            }
            (in_grads, vec![])
        }

        fn param_len(&self) -> usize { 0 }
    }
}

// ====================== Публичный универсальный CombinerConnector ======================
pub struct CombinerConnector {
    input_dims: Vec<usize>,
    output_dim: usize,
}

impl CombinerConnector {
    pub fn new(input_dims: Vec<usize>) -> Self {
        let output_dim = input_dims.iter().sum();
        Self { input_dims, output_dim }
    }

    pub fn input_dims(&self) -> &[usize] { &self.input_dims }
    pub fn output_dim(&self) -> usize { self.output_dim }

    pub fn forward(
        &self,
        inputs: &[DynamicTensor],
    ) -> (DynamicTensor, DynamicContext) {
        if inputs.is_empty() {
            panic!("CombinerConnector требует хотя бы один вход");
        }
        match &inputs[0] {
            DynamicTensor::Dim1(_) => {
                let tensors: Vec<Tensor2D> = inputs.iter().map(|d| match d {
                    DynamicTensor::Dim1(t) => t.clone(),
                    _ => panic!("Все входы должны быть Dim1"),
                }).collect();
                let l = dim::CombinerConnector2D { input_dims: self.input_dims.clone(), output_dim: self.output_dim };
                use crate::layers::context1d::Layer;
                let (mut outs, ctxs) = l.forward(&tensors, &[], &ParamSlice::new(0, 0));
                (DynamicTensor::Dim1(outs.remove(0)), DynamicContext::Ctx1D(ctxs.into_iter().next().unwrap()))
            }
            DynamicTensor::Dim2(_) => {
                let tensors: Vec<Tensor3D> = inputs.iter().map(|d| match d {
                    DynamicTensor::Dim2(t) => t.clone(),
                    _ => panic!("Все входы должны быть Dim2"),
                }).collect();
                let l = dim::CombinerConnector3D { input_dims: self.input_dims.clone(), output_dim: self.output_dim };
                use crate::layers::context2d::Layer2D;
                let (mut outs, ctxs) = l.forward(&tensors, &[], &ParamSlice::new(0, 0));
                (DynamicTensor::Dim2(outs.remove(0)), DynamicContext::Ctx2D(ctxs.into_iter().next().unwrap()))
            }
            DynamicTensor::Dim3(_) => {
                let tensors: Vec<Tensor4D> = inputs.iter().map(|d| match d {
                    DynamicTensor::Dim3(t) => t.clone(),
                    _ => panic!("Все входы должны быть Dim3"),
                }).collect();
                let l = dim::CombinerConnector4D { input_dims: self.input_dims.clone(), output_dim: self.output_dim };
                use crate::layers::context3d::Layer3D;
                let (mut outs, ctxs) = l.forward(&tensors, &[], &ParamSlice::new(0, 0));
                (DynamicTensor::Dim3(outs.remove(0)), DynamicContext::Ctx3D(ctxs.into_iter().next().unwrap()))
            }
            DynamicTensor::Dim4(_) => {
                let tensors: Vec<Tensor5D> = inputs.iter().map(|d| match d {
                    DynamicTensor::Dim4(t) => t.clone(),
                    _ => panic!("Все входы должны быть Dim4"),
                }).collect();
                let l = dim::CombinerConnector5D { input_dims: self.input_dims.clone(), output_dim: self.output_dim };
                use crate::layers::context4d::Layer4D;
                let (mut outs, ctxs) = l.forward(&tensors, &[], &ParamSlice::new(0, 0));
                (DynamicTensor::Dim4(outs.remove(0)), DynamicContext::Ctx4D(ctxs.into_iter().next().unwrap()))
            }
        }
    }

    pub fn backward(
        &self,
        ctx: &DynamicContext,
        delta: &DynamicTensor,
    ) -> (Vec<DynamicTensor>, Vec<f32>) {
        match (ctx, delta) {
            (DynamicContext::Ctx1D(c), DynamicTensor::Dim1(d)) => {
                let l = dim::CombinerConnector2D { input_dims: self.input_dims.clone(), output_dim: self.output_dim };
                use crate::layers::context1d::Layer;
                let (in_deltas, _) = l.backward(&[c.clone()], &[d.clone()], &[], &ParamSlice::new(0, 0));
                let dyn_ins = in_deltas.into_iter().map(DynamicTensor::Dim1).collect();
                (dyn_ins, vec![])
            }
            (DynamicContext::Ctx2D(c), DynamicTensor::Dim2(d)) => {
                let l = dim::CombinerConnector3D { input_dims: self.input_dims.clone(), output_dim: self.output_dim };
                use crate::layers::context2d::Layer2D;
                let (in_deltas, _) = l.backward(&[c.clone()], &[d.clone()], &[], &ParamSlice::new(0, 0));
                let dyn_ins = in_deltas.into_iter().map(DynamicTensor::Dim2).collect();
                (dyn_ins, vec![])
            }
            (DynamicContext::Ctx3D(c), DynamicTensor::Dim3(d)) => {
                let l = dim::CombinerConnector4D { input_dims: self.input_dims.clone(), output_dim: self.output_dim };
                use crate::layers::context3d::Layer3D;
                let (in_deltas, _) = l.backward(&[c.clone()], &[d.clone()], &[], &ParamSlice::new(0, 0));
                let dyn_ins = in_deltas.into_iter().map(DynamicTensor::Dim3).collect();
                (dyn_ins, vec![])
            }
            (DynamicContext::Ctx4D(c), DynamicTensor::Dim4(d)) => {
                let l = dim::CombinerConnector5D { input_dims: self.input_dims.clone(), output_dim: self.output_dim };
                use crate::layers::context4d::Layer4D;
                let (in_deltas, _) = l.backward(&[c.clone()], &[d.clone()], &[], &ParamSlice::new(0, 0));
                let dyn_ins = in_deltas.into_iter().map(DynamicTensor::Dim4).collect();
                (dyn_ins, vec![])
            }
            _ => panic!("Несовпадение размерностей контекста и градиента"),
        }
    }

    pub fn param_len(&self) -> usize { 0 }
}
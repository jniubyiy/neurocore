// src/layers/learnable_mish/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::learnable_mish::LearnableMish;

impl UniversalLayerBuffered for LearnableMish {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
    ) {
        let rows = input.rows();
        let cols = input.cols();
        debug_assert_eq!(cols, self.features);
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "LearnableMish: parameter slice out of bounds"
        );

        let ids = [input.id(), output.id(), params.id()];
        input.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let y: &mut [f32] = &mut *second[0];
            let p: &[f32] = &*rest[0];

            let lambda = p[slice.start];

            for i in 0..x.len() {
                let x_val = x[i];
                let sp = (1.0 + x_val.exp()).ln(); // softplus
                let tanh_sp = (lambda * sp).tanh();
                y[i] = x_val * tanh_sp;
            }
        });
    }

    fn backward_buffered(
        &self,
        ctx: &DynamicContext,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
        grad_params: &MatrixBufferHandle,
    ) {
        let DynamicContext::Buffered(bc) = ctx;
        let input_handle = match bc {
            BufferedContext::LearnableMish { input } => input,
            _ => panic!("Expected LearnableMish context"),
        };

        let rows = grad_output.rows();
        let cols = grad_output.cols();
        debug_assert_eq!(cols, self.features);
        debug_assert_eq!(rows, input_handle.rows());
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "LearnableMish backward: parameter slice out of bounds"
        );
        debug_assert!(
            slice.start + self.param_len() <= grad_params.rows() * grad_params.cols(),
            "LearnableMish backward: grad parameter slice out of bounds"
        );

        let ids = [
            input_handle.id(),
            grad_output.id(),
            grad_input.id(),
            params.id(),
            grad_params.id(),
        ];
        input_handle
            .memory()
            .write()
            .unwrap()
            .with_cpu_slices_mut(&ids, |slices| {
                let (first, rest) = slices.split_at_mut(1);
                let x: &[f32] = &*first[0];
                let (second, rest) = rest.split_at_mut(1);
                let go: &[f32] = &*second[0];
                let (third, rest) = rest.split_at_mut(1);
                let gi: &mut [f32] = &mut *third[0];
                let (fourth, rest) = rest.split_at_mut(1);
                let p: &[f32] = &*fourth[0];
                let gp: &mut [f32] = &mut *rest[0];

                let lambda = p[slice.start];
                let mut grad_lambda = 0.0f32;

                for i in 0..x.len() {
                    let x_val = x[i];
                    let sp = (1.0 + x_val.exp()).ln();
                    let tanh_sp = (lambda * sp).tanh();
                    let dtanh = 1.0 - tanh_sp * tanh_sp;
                    let sigmoid = 1.0 / (1.0 + (-x_val).exp());

                    // градиент по входу
                    let dx = tanh_sp + x_val * dtanh * lambda * sigmoid;
                    gi[i] = go[i] * dx;

                    // градиент по lambda
                    grad_lambda += go[i] * x_val * dtanh * sp;
                }

                // записываем градиент по lambda (накопленный по всем элементам)
                gp[slice.start] = grad_lambda;
            });
    }

    fn param_len(&self) -> usize {
        1
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}
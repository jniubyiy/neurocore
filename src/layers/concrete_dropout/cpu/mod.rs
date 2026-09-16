// src/layers/concrete_dropout/cpu/mod.rs

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::concrete_dropout::ConcreteDropout;

impl UniversalLayerBuffered for ConcreteDropout {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
        pool: &mut TempMatrixPool,
    ) -> BufferedContext {
        let rows = input.rows();
        let cols = input.cols();
        let total = rows * cols;
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "ConcreteDropout: parameter slice out of bounds"
        );

        // Per-chunk буфер для аргументов сигмоиды.
        // Форма — вектор-столбец длины total.
        let arg_handle = pool.acquire(total, 1);

        // Всё чтение/запись выполняем одним срезом.
        let ids = [
            input.id(),
            output.id(),
            params.id(),
            arg_handle.id(),
        ];
        input
            .memory()
            .write()
            .unwrap()
            .with_cpu_slices_mut(&ids, |slices| {
                let (first, rest) = slices.split_at_mut(1);
                let x: &[f32] = &*first[0];

                let (second, rest) = rest.split_at_mut(1);
                let y: &mut [f32] = &mut *second[0];

                let (third, rest) = rest.split_at_mut(1);
                let p: &[f32] = &*third[0];

                let (fourth, _) = rest.split_at_mut(1);
                let arg: &mut [f32] = &mut *fourth[0];

                let logit_p = p[slice.start];
                let temp = self.temperature;
                let mut rng = StdRng::seed_from_u64(self.seed);
                let eps = 1e-8f32;

                for i in 0..total {
                    let u: f32 = rng.gen();
                    let log_u = (u + eps).ln();
                    let log_1mu = (1.0 - u + eps).ln();
                    let a = (logit_p + log_u - log_1mu) / temp;
                    let z = 1.0 / (1.0 + (-a).exp());
                    arg[i] = a;
                    y[i] = x[i] * z;
                }
            });

        BufferedContext::ConcreteDropout {
            input: input.clone(),
            arg: arg_handle,
        }
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
        let (input_handle, arg_handle) = match bc {
            BufferedContext::ConcreteDropout { input, arg } => (input, arg),
            _ => panic!("Expected ConcreteDropout Buffered context"),
        };

        let rows = grad_output.rows();
        let cols = grad_output.cols();
        let total = rows * cols;
        debug_assert_eq!(rows, input_handle.rows());
        debug_assert_eq!(cols, input_handle.cols());
        debug_assert_eq!(total, arg_handle.rows() * arg_handle.cols());
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "ConcreteDropout backward: parameter slice out of bounds"
        );
        debug_assert!(
            slice.start + self.param_len() <= grad_params.rows() * grad_params.cols(),
            "ConcreteDropout backward: grad parameter slice out of bounds"
        );

        let ids = [
            input_handle.id(),
            grad_output.id(),
            grad_input.id(),
            params.id(),
            grad_params.id(),
            arg_handle.id(),
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
                let _p: &[f32] = &*fourth[0];

                let (fifth, rest) = rest.split_at_mut(1);
                let gp: &mut [f32] = &mut *fifth[0];

                let (sixth, _) = rest.split_at_mut(1);
                let arg: &[f32] = &*sixth[0];

                let temp = self.temperature;
                let mut grad_logit_p = 0.0f32;

                for i in 0..total {
                    let a = arg[i];
                    let sigmoid = 1.0 / (1.0 + (-a).exp());
                    let dsigmoid = sigmoid * (1.0 - sigmoid);

                    // Градиент по входу.
                    gi[i] = go[i] * sigmoid;

                    // Градиент по logit_p.
                    grad_logit_p += go[i] * x[i] * dsigmoid / temp;
                }

                // Записываем градиент по logit_p.
                gp[slice.start] = grad_logit_p;
            });
    }

    fn param_len(&self) -> usize {
        1
    }

    fn input_features(&self) -> usize {
        0
    }

    fn output_features(&self) -> usize {
        0
    }
}
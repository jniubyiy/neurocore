// src/compute_manager/gpu/processor/layers/activation.rs
//
// Категория «активации и поэлементные функции»:
//   ReLU, Sigmoid, Tanh, LeakyReLU, Softmax, Identity,
//   DualSlopeReLU, LearnableMish, LearnableSoftplus,
//   AdaptivePerFeatureActivation.

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::view::MatrixBufferView;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;

pub fn forward(
    gpu: &GpuCompute,
    layer: &dyn UniversalLayer,
    input: &MatrixBufferHandle,
    params_handle: &MatrixBufferHandle,
    slice: &ParamSlice,
) -> Option<(MatrixBufferHandle, DynamicContext)> {
    // ---- ReLU ----
    if layer.as_relu().is_some() {
        let out = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());
        gpu.run_relu_forward_buffered_handle(input, &out);
        let ctx = DynamicContext::Buffered(BufferedContext::ReLU {
            input: input.clone(),
        });
        return Some((out, ctx));
    }

    // ---- Sigmoid ----
    if layer.as_sigmoid().is_some() {
        let out = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());
        gpu.run_sigmoid_forward_buffered_handle(input, &out);
        let ctx = DynamicContext::Buffered(BufferedContext::Sigmoid {
            output: out.clone(),
        });
        return Some((out, ctx));
    }

    // ---- Tanh ----
    if layer.as_tanh().is_some() {
        let out = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());
        gpu.run_tanh_forward_buffered_handle(input, &out);
        let ctx = DynamicContext::Buffered(BufferedContext::Tanh {
            output: out.clone(),
        });
        return Some((out, ctx));
    }

    // ---- LeakyReLU ----
    if let Some(leaky) = layer.as_leaky_relu() {
        let out = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());
        gpu.run_leaky_relu_forward_buffered_handle(input, &out, leaky.alpha);
        let ctx = DynamicContext::Buffered(BufferedContext::LeakyReLU {
            input: input.clone(),
        });
        return Some((out, ctx));
    }

    // ---- Softmax ----
    if layer.as_softmax().is_some() {
        let out = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());
        gpu.run_softmax_forward_buffered_handle(input, &out);
        let ctx = DynamicContext::Buffered(BufferedContext::Softmax {
            output: out.clone(),
        });
        return Some((out, ctx));
    }

    // ---- Identity ----
    if layer.as_identity().is_some() {
        let out = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());
        gpu.run_identity_forward_buffered_handle(input, &out);
        let ctx = DynamicContext::Buffered(BufferedContext::Identity {
            input: input.clone(),
        });
        return Some((out, ctx));
    }

    // ---- DualSlopeReLU ----
    if let Some(dslope) = layer.as_dual_slope_relu() {
        let features = dslope.features;
        let alpha_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, features);
        let beta_view = MatrixBufferView::new(
            params_handle.clone(),
            slice.start + features,
            features,
        );
        let out = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());
        gpu.run_dual_slope_relu_forward_buffered_handle(
            input,
            &alpha_view,
            &beta_view,
            &out,
        );
        let ctx = DynamicContext::Buffered(BufferedContext::DualSlopeReLU {
            input: input.clone(),
        });
        return Some((out, ctx));
    }

    // ---- LearnableMish ----
    if let Some(mish) = layer.as_learnable_mish() {
        let features = mish.features;
        let lambda_view = MatrixBufferView::new(params_handle.clone(), slice.start, 1);
        let out = gpu.allocate_gpu_matrix_handle(input.rows(), features);
        gpu.run_learnable_mish_forward_buffered_handle(input, &lambda_view, &out);
        let ctx = DynamicContext::Buffered(BufferedContext::LearnableMish {
            input: input.clone(),
        });
        return Some((out, ctx));
    }

    // ---- LearnableSoftplus ----
    if let Some(softplus) = layer.as_learnable_softplus() {
        let features = softplus.features;
        let beta_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, features);
        let theta_view = MatrixBufferView::new(
            params_handle.clone(),
            slice.start + features,
            features,
        );
        let out = gpu.allocate_gpu_matrix_handle(input.rows(), features);
        gpu.run_learnable_softplus_forward_buffered_handle(
            input,
            &beta_view,
            &theta_view,
            &out,
        );
        let ctx = DynamicContext::Buffered(BufferedContext::LearnableSoftplus {
            input: input.clone(),
        });
        return Some((out, ctx));
    }

    // ---- AdaptivePerFeatureActivation ----
    if let Some(adaptive) = layer.as_adaptive_activation() {
        let in_features = adaptive.in_features;
        let num_activations = adaptive.num_activations;
        let params_len = in_features * num_activations;
        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
        let out = gpu.allocate_gpu_matrix_handle(input.rows(), in_features);
        gpu.run_adaptive_activation_forward_buffered_handle(
            input,
            &params_view,
            num_activations,
            &out,
        );
        let ctx = DynamicContext::Buffered(BufferedContext::AdaptiveActivation {
            input: input.clone(),
        });
        return Some((out, ctx));
    }

    None
}

pub fn backward(
    gpu: &GpuCompute,
    layer: &dyn UniversalLayer,
    ctx: &DynamicContext,
    grad_output: &MatrixBufferHandle,
    params_handle: &MatrixBufferHandle,
    slice: &ParamSlice,
    grad_params_handle: &MatrixBufferHandle,
) -> Option<MatrixBufferHandle> {
    // ---- ReLU ----
    if layer.as_relu().is_some() {
        let DynamicContext::Buffered(bc) = ctx;
        let input_handle = match bc {
            BufferedContext::ReLU { input } => input.clone(),
            _ => panic!("Expected ReLU Buffered context"),
        };
        let gi = gpu.allocate_gpu_matrix_handle(grad_output.rows(), grad_output.cols());
        gpu.run_relu_backward_buffered_handle(&input_handle, grad_output, &gi);
        return Some(gi);
    }

    // ---- Sigmoid ----
    if layer.as_sigmoid().is_some() {
        let DynamicContext::Buffered(bc) = ctx;
        let output_handle = match bc {
            BufferedContext::Sigmoid { output } => output.clone(),
            _ => panic!("Expected Sigmoid Buffered context"),
        };
        let gi = gpu.allocate_gpu_matrix_handle(grad_output.rows(), grad_output.cols());
        gpu.run_sigmoid_backward_buffered_handle(&output_handle, grad_output, &gi);
        return Some(gi);
    }

    // ---- Tanh ----
    if layer.as_tanh().is_some() {
        let DynamicContext::Buffered(bc) = ctx;
        let output_handle = match bc {
            BufferedContext::Tanh { output } => output.clone(),
            _ => panic!("Expected Tanh Buffered context"),
        };
        let gi = gpu.allocate_gpu_matrix_handle(grad_output.rows(), grad_output.cols());
        gpu.run_tanh_backward_buffered_handle(&output_handle, grad_output, &gi);
        return Some(gi);
    }

    // ---- LeakyReLU ----
    if let Some(leaky) = layer.as_leaky_relu() {
        let DynamicContext::Buffered(bc) = ctx;
        let input_handle = match bc {
            BufferedContext::LeakyReLU { input } => input.clone(),
            _ => panic!("Expected LeakyReLU Buffered context"),
        };
        let gi = gpu.allocate_gpu_matrix_handle(grad_output.rows(), grad_output.cols());
        gpu.run_leaky_relu_backward_buffered_handle(
            &input_handle,
            grad_output,
            &gi,
            leaky.alpha,
        );
        return Some(gi);
    }

    // ---- Softmax ----
    if layer.as_softmax().is_some() {
        let DynamicContext::Buffered(bc) = ctx;
        let output_handle = match bc {
            BufferedContext::Softmax { output } => output.clone(),
            _ => panic!("Expected Softmax Buffered context"),
        };
        let gi = gpu.allocate_gpu_matrix_handle(grad_output.rows(), grad_output.cols());
        gpu.run_softmax_backward_buffered_handle(&output_handle, grad_output, &gi);
        return Some(gi);
    }

    // ---- Identity ----
    if layer.as_identity().is_some() {
        let gi = gpu.allocate_gpu_matrix_handle(grad_output.rows(), grad_output.cols());
        gpu.run_identity_backward_buffered_handle(grad_output, &gi);
        return Some(gi);
    }

    // ---- DualSlopeReLU ----
    if let Some(dslope) = layer.as_dual_slope_relu() {
        let features = dslope.features;
        let DynamicContext::Buffered(bc) = ctx;
        let input_handle = match bc {
            BufferedContext::DualSlopeReLU { input } => input.clone(),
            _ => panic!("Expected DualSlopeReLU Buffered context"),
        };
        let alpha_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, features);
        let beta_view = MatrixBufferView::new(
            params_handle.clone(),
            slice.start + features,
            features,
        );
        let grad_alpha_view =
            MatrixBufferView::new(grad_params_handle.clone(), slice.start, features);
        let grad_beta_view = MatrixBufferView::new(
            grad_params_handle.clone(),
            slice.start + features,
            features,
        );
        let gi = gpu.allocate_gpu_matrix_handle(grad_output.rows(), grad_output.cols());
        gpu.run_dual_slope_relu_backward_buffered_handle(
            &input_handle,
            grad_output,
            &alpha_view,
            &beta_view,
            &gi,
            &grad_alpha_view,
            &grad_beta_view,
        );
        return Some(gi);
    }

    // ---- LearnableMish ----
    if let Some(mish) = layer.as_learnable_mish() {
        let features = mish.features;
        let DynamicContext::Buffered(bc) = ctx;
        let input_handle = match bc {
            BufferedContext::LearnableMish { input } => input.clone(),
            _ => panic!("Expected LearnableMish Buffered context"),
        };
        let lambda_view = MatrixBufferView::new(params_handle.clone(), slice.start, 1);
        let grad_lambda_view =
            MatrixBufferView::new(grad_params_handle.clone(), slice.start, 1);
        let gi = gpu.allocate_gpu_matrix_handle(grad_output.rows(), features);
        gpu.run_learnable_mish_backward_buffered_handle(
            &input_handle,
            grad_output,
            &lambda_view,
            &gi,
            &grad_lambda_view,
        );
        return Some(gi);
    }

    // ---- LearnableSoftplus ----
    if let Some(softplus) = layer.as_learnable_softplus() {
        let features = softplus.features;
        let DynamicContext::Buffered(bc) = ctx;
        let input_handle = match bc {
            BufferedContext::LearnableSoftplus { input } => input.clone(),
            _ => panic!("Expected LearnableSoftplus Buffered context"),
        };
        let beta_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, features);
        let theta_view = MatrixBufferView::new(
            params_handle.clone(),
            slice.start + features,
            features,
        );
        let grad_beta_view =
            MatrixBufferView::new(grad_params_handle.clone(), slice.start, features);
        let grad_theta_view = MatrixBufferView::new(
            grad_params_handle.clone(),
            slice.start + features,
            features,
        );
        let gi = gpu.allocate_gpu_matrix_handle(grad_output.rows(), features);
        gpu.run_learnable_softplus_backward_buffered_handle(
            &input_handle,
            grad_output,
            &beta_view,
            &theta_view,
            &gi,
            &grad_beta_view,
            &grad_theta_view,
        );
        return Some(gi);
    }

    // ---- AdaptivePerFeatureActivation ----
    if let Some(adaptive) = layer.as_adaptive_activation() {
        let in_features = adaptive.in_features;
        let num_activations = adaptive.num_activations;
        let params_len = in_features * num_activations;

        let DynamicContext::Buffered(bc) = ctx;
        let input_handle = match bc {
            BufferedContext::AdaptiveActivation { input } => input.clone(),
            _ => panic!("Expected AdaptiveActivation Buffered context"),
        };

        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
        let grad_params_view =
            MatrixBufferView::new(grad_params_handle.clone(), slice.start, params_len);

        let gi = gpu.allocate_gpu_matrix_handle(grad_output.rows(), in_features);
        gpu.run_adaptive_activation_backward_buffered_handle(
            &input_handle,
            grad_output,
            &params_view,
            num_activations,
            &gi,
            &grad_params_view,
        );
        return Some(gi);
    }

    None
}
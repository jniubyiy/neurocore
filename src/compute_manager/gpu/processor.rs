// src/compute_manager/gpu/processor.rs

use faer::Mat;
use crate::compute_manager::graph::types::DynamicContext;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;
use super::compute::GpuCompute;

pub fn process_forward_gpu(
    gpu_compute: &GpuCompute,
    layers: &[Box<dyn UniversalLayer>],
    slices: &[ParamSlice],
    params: &[f32],
    input: &Mat<f32>,
) -> (Mat<f32>, Vec<DynamicContext>) {
    let mut current = input.clone();
    let mut ctxs = Vec::with_capacity(layers.len());

    for (layer, slice) in layers.iter().zip(slices.iter()) {
        if let Some(linear) = layer.as_linear() {
            let input_tensor = crate::linalg::faer_to_tensor2d(&current);
            let (weight, bias) = linear.get_weight_matrix_and_bias(params, slice);
            current = gpu_compute.run_linear_forward(&current, &weight, &bias);
            ctxs.push(DynamicContext::Ctx1D(
                crate::layers::context1d::LayerContext1D::Linear { input: input_tensor },
            ));
        } else if let Some(_) = layer.as_relu() {
            let input_tensor = crate::linalg::faer_to_tensor2d(&current);
            current = gpu_compute.run_relu_forward(&current);
            ctxs.push(DynamicContext::Ctx1D(
                crate::layers::context1d::LayerContext1D::ReLU { input: input_tensor },
            ));
        } else if let Some(_) = layer.as_sigmoid() {
            current = gpu_compute.run_sigmoid_forward(&current);
            let output_tensor = crate::linalg::faer_to_tensor2d(&current);
            ctxs.push(DynamicContext::Ctx1D(
                crate::layers::context1d::LayerContext1D::Sigmoid { output: output_tensor },
            ));
        } else if let Some(_) = layer.as_tanh() {
            current = gpu_compute.run_tanh_forward(&current);
            let output_tensor = crate::linalg::faer_to_tensor2d(&current);
            ctxs.push(DynamicContext::Ctx1D(
                crate::layers::context1d::LayerContext1D::Tanh { output: output_tensor },
            ));
        } else if let Some(leaky) = layer.as_leaky_relu() {
            let input_tensor = crate::linalg::faer_to_tensor2d(&current);
            current = gpu_compute.run_leaky_relu_forward(&current, leaky.alpha);
            ctxs.push(DynamicContext::Ctx1D(
                crate::layers::context1d::LayerContext1D::LeakyReLU { input: input_tensor },
            ));
        } else if layer.as_identity().is_some() {
            let (out, ctx) = layer.forward_mat(&current, params, slice);
            current = out;
            ctxs.push(ctx);
        } else {
            let (out, ctx) = layer.forward_mat(&current, params, slice);
            current = out;
            ctxs.push(ctx);
        }
    }

    (current, ctxs)
}

pub fn process_backward_gpu(
    gpu_compute: &GpuCompute,
    layers: &[Box<dyn UniversalLayer>],
    slices: &[ParamSlice],
    contexts: &[DynamicContext],
    params: &[f32],
    grad_output: &Mat<f32>,
    total_grad: &mut [f32],
) -> Mat<f32> {
    let num_layers = layers.len();
    assert_eq!(contexts.len(), num_layers);
    assert_eq!(slices.len(), num_layers);

    let mut current_grad = grad_output.clone();

    for idx in (0..num_layers).rev() {
        let layer = &layers[idx];
        let slice = &slices[idx];
        let ctx = &contexts[idx];

        if let Some(linear) = layer.as_linear() {
            let (weight, _) = linear.get_weight_matrix_and_bias(params, slice);
            let input_tensor = match ctx {
                DynamicContext::Ctx1D(c) => match c {
                    crate::layers::context1d::LayerContext1D::Linear { input } => input,
                    _ => panic!("Expected Linear context"),
                },
                _ => panic!("Expected Ctx1D"),
            };
            let input_mat = crate::linalg::tensor2d_to_faer(input_tensor);
            let (dx, dw, db) = gpu_compute.run_linear_backward(&input_mat, &weight, &current_grad);
            current_grad = dx;

            let in_feat = linear.input_features();
            let out_feat = linear.output_features();
            let w_start = slice.start;
            for r in 0..out_feat {
                for c in 0..in_feat {
                    total_grad[w_start + r * in_feat + c] += dw[(r, c)];
                }
            }
            let b_start = w_start + in_feat * out_feat;
            for (i, &v) in db.iter().enumerate() {
                total_grad[b_start + i] += v;
            }
        } else if let Some(_) = layer.as_relu() {
            let input_tensor = match ctx {
                DynamicContext::Ctx1D(c) => match c {
                    crate::layers::context1d::LayerContext1D::ReLU { input } => input,
                    _ => panic!(),
                },
                _ => panic!(),
            };
            let input_mat = crate::linalg::tensor2d_to_faer(input_tensor);
            current_grad = gpu_compute.run_relu_backward(&input_mat, &current_grad);
        } else if let Some(_) = layer.as_sigmoid() {
            let output_tensor = match ctx {
                DynamicContext::Ctx1D(c) => match c {
                    crate::layers::context1d::LayerContext1D::Sigmoid { output } => output,
                    _ => panic!(),
                },
                _ => panic!(),
            };
            let output_mat = crate::linalg::tensor2d_to_faer(output_tensor);
            current_grad = gpu_compute.run_sigmoid_backward(&output_mat, &current_grad);
        } else if let Some(_) = layer.as_tanh() {
            let output_tensor = match ctx {
                DynamicContext::Ctx1D(c) => match c {
                    crate::layers::context1d::LayerContext1D::Tanh { output } => output,
                    _ => panic!(),
                },
                _ => panic!(),
            };
            let output_mat = crate::linalg::tensor2d_to_faer(output_tensor);
            current_grad = gpu_compute.run_tanh_backward(&output_mat, &current_grad);
        } else if let Some(leaky) = layer.as_leaky_relu() {
            let input_tensor = match ctx {
                DynamicContext::Ctx1D(c) => match c {
                    crate::layers::context1d::LayerContext1D::LeakyReLU { input } => input,
                    _ => panic!(),
                },
                _ => panic!(),
            };
            let input_mat = crate::linalg::tensor2d_to_faer(input_tensor);
            current_grad = gpu_compute.run_leaky_relu_backward(&input_mat, &current_grad, leaky.alpha);
        } else if let Some(_) = layer.as_softmax() {
            let output_tensor = match ctx {
                DynamicContext::Ctx1D(c) => match c {
                    crate::layers::context1d::LayerContext1D::Softmax { output } => output,
                    _ => panic!(),
                },
                _ => panic!(),
            };
            let output_mat = crate::linalg::tensor2d_to_faer(output_tensor);
            current_grad = gpu_compute.run_softmax_backward(&output_mat, &current_grad);
        } else if let Some(_) = layer.as_memory() {
            let memory = layer.as_memory().unwrap();
            let _input_tensor = match ctx {
                DynamicContext::Ctx1D(c) => match c {
                    crate::layers::context1d::LayerContext1D::Memory { input } => input,
                    _ => panic!(),
                },
                _ => panic!(),
            };
            current_grad = gpu_compute.run_memory_backward(&current_grad, memory.alpha);
        } else if let Some(softsparse) = layer.as_soft_sparse_gate() {
            let input_tensor = match ctx {
                DynamicContext::Ctx1D(c) => match c {
                    crate::layers::context1d::LayerContext1D::SoftSparseGate { input } => input,
                    _ => panic!(),
                },
                _ => panic!(),
            };
            let input_mat = crate::linalg::tensor2d_to_faer(input_tensor);
            let thresholds = &params[slice.start..slice.start + softsparse.in_features];
            let (dx, d_thr) = gpu_compute.run_softsparse_backward(
                &input_mat,
                &current_grad,
                thresholds,
                softsparse.temperature,
            );
            current_grad = dx;
            for (i, &g) in d_thr.iter().enumerate() {
                total_grad[slice.start + i] += g;
            }
        } else if let Some(softkeep) = layer.as_soft_keep_gate() {
            let input_tensor = match ctx {
                DynamicContext::Ctx1D(c) => match c {
                    crate::layers::context1d::LayerContext1D::SoftKeepGate { input } => input,
                    _ => panic!(),
                },
                _ => panic!(),
            };
            let input_mat = crate::linalg::tensor2d_to_faer(input_tensor);
            let thresholds = &params[slice.start..slice.start + softkeep.in_features];
            let (dx, d_thr) = gpu_compute.run_softkeep_backward(
                &input_mat,
                &current_grad,
                thresholds,
                softkeep.temperature,
            );
            current_grad = dx;
            for (i, &g) in d_thr.iter().enumerate() {
                total_grad[slice.start + i] += g;
            }
        } else if let Some(dual) = layer.as_dual_anchor() {
            let input_tensor = match ctx {
                DynamicContext::Ctx1D(c) => match c {
                    crate::layers::context1d::LayerContext1D::DualAnchor1D { input } => input,
                    _ => panic!(),
                },
                _ => panic!(),
            };
            let input_mat = crate::linalg::tensor2d_to_faer(input_tensor);
            let features = dual.features;
            let min_vals = &params[slice.start..slice.start + features];
            let max_vals = &params[slice.start + features..slice.start + 2 * features];
            let alpha = params[slice.start + 2 * features];
            let (dx, grad) = gpu_compute.run_dualanchor_backward(
                &input_mat,
                &current_grad,
                min_vals,
                max_vals,
                alpha,
            );
            current_grad = dx;
            for (i, &g) in grad.iter().enumerate() {
                total_grad[slice.start + i] += g;
            }
        } else {
            let (dx, layer_grad) = layer.backward_mat(ctx, &current_grad, params, slice);
            current_grad = dx;
            for (i, &g) in layer_grad.iter().enumerate() {
                total_grad[slice.start + i] += g;
            }
        }
    }

    current_grad
}
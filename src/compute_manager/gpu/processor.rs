// src/compute_manager/gpu/processor.rs

use faer::Mat;
use crate::compute_manager::graph::types::DynamicContext;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;
use super::compute::GpuCompute;

/// Выполняет прямой проход батча через цепочку слоёв на GPU.
/// Для каждого слоя сохраняется корректный контекст (входной тензор **до** вычисления).
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
            // Сохраняем входной тензор ДО вычисления
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
            let output_tensor = crate::linalg::faer_to_tensor2d(&current); // для сигмоиды контекст хранит выход
            current = gpu_compute.run_sigmoid_forward(&current);
            ctxs.push(DynamicContext::Ctx1D(
                crate::layers::context1d::LayerContext1D::Sigmoid { output: output_tensor },
            ));
            // Примечание: сигмоида в backward использует выход, а не вход, поэтому сохраняем выход ДО вычисления.
            // Здесь мы сохраняем текущий current как выход предыдущего слоя, что является входом для сигмоиды.
            // Но для backward сигмоиде нужен её собственный выход, который будет вычислен. Поэтому мы должны сохранить выход после вычисления.
            // Исправляем: сначала вычисляем, потом сохраняем выход.
            // Перепишем блок для сигмоиды:
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
            // Fallback на CPU для остальных слоёв
            let (out, ctx) = layer.forward_mat(&current, params, slice);
            current = out;
            ctxs.push(ctx);
        }
    }

    (current, ctxs)
}
// src/compute_manager/gpu/processor.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::view::MatrixBufferView;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;

use super::compute::GpuCompute;

/// Прямой проход на GPU с использованием MatrixBufferHandle.
/// Вход и выход — GPU-дескрипторы. Контексты создаются как Buffered.
///
/// Параметры сегмента уже должны находиться на GPU (в `params_handle`).
/// Доступ к отдельным слоям осуществляется через `MatrixBufferView`,
/// который представляет собой непрерывный диапазон внутри буфера.
pub fn process_forward_gpu_buffered(
    gpu_compute: &GpuCompute,
    layers: &[Box<dyn UniversalLayer>],
    slices: &[ParamSlice],
    params_handle: &MatrixBufferHandle,
    input: MatrixBufferHandle,
) -> (MatrixBufferHandle, Vec<DynamicContext>) {
    assert!(input.is_gpu(), "Input must be GPU handle");
    assert!(
        params_handle.is_gpu() || params_handle.rows() == 0,
        "process_forward_gpu_buffered: params must be GPU or empty for parameterless segment"
    );

    let mut current = input;
    let mut ctxs = Vec::with_capacity(layers.len());
    let mut memory_idx = 0usize; // счётчик слоёв Memory

    for (layer, slice) in layers.iter().zip(slices.iter()) {
        // ================ Существующие слои ================
        if let Some(linear) = layer.as_linear() {
            let in_feat = linear.input_features();
            let out_feat = linear.output_features();
            let w_start = slice.start;
            let b_start = w_start + in_feat * out_feat;

            let weight_view = MatrixBufferView::with_shape(
                params_handle.clone(),
                w_start,
                in_feat * out_feat,
                out_feat,
                in_feat,
            );
            let bias_view = MatrixBufferView::with_shape(
                params_handle.clone(),
                b_start,
                out_feat,
                1,
                out_feat,
            );

            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), out_feat);
            gpu_compute.run_linear_forward_buffered_handle(
                &current,
                &weight_view,
                &bias_view,
                &out_handle,
            );

            ctxs.push(DynamicContext::Buffered(BufferedContext::Linear {
                input: current.clone(),
            }));
            current = out_handle;
        } else if layer.as_relu().is_some() {
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_relu_forward_buffered_handle(&current, &out_handle);
            ctxs.push(DynamicContext::Buffered(BufferedContext::ReLU {
                input: current.clone(),
            }));
            current = out_handle;
        } else if layer.as_sigmoid().is_some() {
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_sigmoid_forward_buffered_handle(&current, &out_handle);
            ctxs.push(DynamicContext::Buffered(BufferedContext::Sigmoid {
                output: out_handle.clone(),
            }));
            current = out_handle;
        } else if layer.as_tanh().is_some() {
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_tanh_forward_buffered_handle(&current, &out_handle);
            ctxs.push(DynamicContext::Buffered(BufferedContext::Tanh {
                output: out_handle.clone(),
            }));
            current = out_handle;
        } else if let Some(leaky) = layer.as_leaky_relu() {
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_leaky_relu_forward_buffered_handle(&current, &out_handle, leaky.alpha);
            ctxs.push(DynamicContext::Buffered(BufferedContext::LeakyReLU {
                input: current.clone(),
            }));
            current = out_handle;
        } else if layer.as_softmax().is_some() {
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_softmax_forward_buffered_handle(&current, &out_handle);
            ctxs.push(DynamicContext::Buffered(BufferedContext::Softmax {
                output: out_handle.clone(),
            }));
            current = out_handle;
        } else if layer.as_identity().is_some() {
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_identity_forward_buffered_handle(&current, &out_handle);
            ctxs.push(DynamicContext::Buffered(BufferedContext::Identity {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(memory) = layer.as_memory() {
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_memory_forward_buffered_handle(
                &current,
                &out_handle,
                memory.alpha,
                memory_idx,
            );
            memory_idx += 1;
            ctxs.push(DynamicContext::Buffered(BufferedContext::Memory {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(soft_sparse) = layer.as_soft_sparse_gate() {
            let features = soft_sparse.in_features;
            let thresholds_view = MatrixBufferView::new(
                params_handle.clone(),
                slice.start,
                features,
            );
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_softsparse_forward_buffered_handle(
                &current,
                &thresholds_view,
                soft_sparse.temperature,
                &out_handle,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::SoftSparseGate {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(soft_keep) = layer.as_soft_keep_gate() {
            let features = soft_keep.in_features;
            let thresholds_view = MatrixBufferView::new(
                params_handle.clone(),
                slice.start,
                features,
            );
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_softkeep_forward_buffered_handle(
                &current,
                &thresholds_view,
                soft_keep.temperature,
                &out_handle,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::SoftKeepGate {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(dual) = layer.as_dual_anchor() {
            let features = dual.features;
            let min_view = MatrixBufferView::new(params_handle.clone(), slice.start, features);
            let max_view = MatrixBufferView::new(
                params_handle.clone(),
                slice.start + features,
                features,
            );
            let alpha_view = MatrixBufferView::new(
                params_handle.clone(),
                slice.start + 2 * features,
                1,
            );
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_dualanchor_forward_buffered_handle(
                &current,
                &min_view,
                &max_view,
                &alpha_view,
                &out_handle,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::DualAnchor1D {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(adaptive) = layer.as_adaptive_activation() {
            let in_features = adaptive.in_features;
            let num_activations = adaptive.num_activations;
            let params_len = in_features * num_activations;
            let params_view = MatrixBufferView::new(
                params_handle.clone(),
                slice.start,
                params_len,
            );
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), in_features);
            gpu_compute.run_adaptive_activation_forward_buffered_handle(
                &current,
                &params_view,
                num_activations,
                &out_handle,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::AdaptiveActivation {
                input: current.clone(),
            }));
            current = out_handle;
        }
        // ================ Новые слои ================
        else if let Some(dslope) = layer.as_dual_slope_relu() {
            let features = dslope.features;
            let alpha_view = MatrixBufferView::new(params_handle.clone(), slice.start, features);
            let beta_view = MatrixBufferView::new(params_handle.clone(), slice.start + features, features);
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_dual_slope_relu_forward_buffered_handle(
                &current,
                &alpha_view,
                &beta_view,
                &out_handle,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::DualSlopeReLU {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(mish) = layer.as_learnable_mish() {
            let features = mish.features;
            let lambda_view = MatrixBufferView::new(params_handle.clone(), slice.start, 1);
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), features);
            gpu_compute.run_learnable_mish_forward_buffered_handle(
                &current,
                &lambda_view,
                &out_handle,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::LearnableMish {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(softplus) = layer.as_learnable_softplus() {
            let features = softplus.features;
            let beta_view = MatrixBufferView::new(params_handle.clone(), slice.start, features);
            let theta_view = MatrixBufferView::new(params_handle.clone(), slice.start + features, features);
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), features);
            gpu_compute.run_learnable_softplus_forward_buffered_handle(
                &current,
                &beta_view,
                &theta_view,
                &out_handle,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::LearnableSoftplus {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(rms) = layer.as_rms_norm_learnable_eps() {
            let features = rms.features;
            let params_len = 2 * features;
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), features);
            gpu_compute.run_rms_norm_learnable_eps_forward_buffered_handle(
                &current,
                &params_view,
                &out_handle,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::RMSNormWithLearnableEpsilon {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(adrop) = layer.as_adaptive_dropout() {
            let features = adrop.features;
            let params_len = 2 * features;
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
            let batch = current.rows();
            let mask_out = gpu_compute.allocate_gpu_matrix_handle(batch, features);
            let arg_out = gpu_compute.allocate_gpu_matrix_handle(batch, features);
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(batch, features);
            let seed = adrop.seed as u32; // Приведение u64 -> u32
            gpu_compute.run_adaptive_dropout_forward_buffered_handle(
                &current,
                &params_view,
                &out_handle,
                &mask_out,
                &arg_out,
                seed,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::AdaptiveDropout {
                input: current.clone(),
                mask: mask_out,
                arg: arg_out,
            }));
            current = out_handle;
        } else if let Some(fusion) = layer.as_feature_fusion() {
            let in_features = fusion.in_features;
            let out_features = fusion.out_features;
            let params_len = out_features * (in_features + 1);
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), out_features);
            gpu_compute.run_feature_fusion_forward_buffered_handle(
                &current,
                &params_view,
                &out_handle,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::FeatureFusion {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(sfs) = layer.as_sparse_feature_selection_gate() {
            let features = sfs.features;
            let params_len = features + 1;
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), features);
            gpu_compute.run_sparse_feature_selection_forward_buffered_handle(
                &current,
                &params_view,
                &out_handle,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::SparseFeatureSelectionGate {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(kan) = layer.as_multi_resolution_kan_linear() {
            let in_features = kan.in_features;
            let out_features = kan.out_features;
            let params_len = in_features * out_features * 12 + out_features;
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), out_features);
            gpu_compute.run_multi_resolution_kan_linear_forward_buffered_handle(
                &current,
                &params_view,
                &out_handle,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::MultiResolutionKANLinear {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(adnorm) = layer.as_adaptive_normalization() {
            let features = adnorm.features;
            let params_len = 7 * features;
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), features);
            gpu_compute.run_adaptive_norm_forward_buffered_handle(
                &current,
                &params_view,
                &out_handle,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::AdaptiveNormalization {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(bn) = layer.as_batch_renorm() {
            let features = bn.features;
            let params_len = 4 * features;
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), features);
            gpu_compute.run_batch_renorm_forward_buffered_handle(
                &current,
                &params_view,
                &out_handle,
            );
            // В GPU статистики пересчитываются в backward, поэтому храним заглушки
            ctxs.push(DynamicContext::Buffered(BufferedContext::BatchRenorm {
                input: current.clone(),
                mean: Vec::new(),
                var: Vec::new(),
                use_batch_stats: true,
            }));
            current = out_handle;
        } else if let Some(cdrop) = layer.as_concrete_dropout() {
            let temperature = cdrop.temperature;
            let logit_view = MatrixBufferView::new(params_handle.clone(), slice.start, 1);
            let batch = current.rows();
            let arg_out = gpu_compute.allocate_gpu_matrix_handle(batch * current.cols(), 1);
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(batch, current.cols());
            let seed = cdrop.seed as u32; // Приведение u64 -> u32
            gpu_compute.run_concrete_dropout_forward_buffered_handle(
                &current,
                &logit_view,
                temperature,
                &out_handle,
                &arg_out,
                seed,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::ConcreteDropout {
                input: current.clone(),
                arg: arg_out,
            }));
            current = out_handle;
        } else if let Some(ind) = layer.as_ind_rnn() {
            let seq_len = ind.seq_len;
            let input_dim = ind.input_dim;
            let batch = current.rows();
            let h_all = gpu_compute.allocate_gpu_matrix_handle(batch * seq_len * input_dim, 1);
            let params_len = input_dim * input_dim + 2 * input_dim;
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(batch, seq_len * input_dim);
            gpu_compute.run_ind_rnn_forward_buffered_handle(
                &current,
                &params_view,
                &out_handle,
                seq_len,
                input_dim,
                &h_all,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::IndRNN {
                input: current.clone(),
                h_all,
            }));
            current = out_handle;
        } else if let Some(mamba) = layer.as_mamba() {
            let seq_len = mamba.seq_len;
            let input_dim = mamba.input_dim;
            let state_dim = mamba.state_dim;
            let batch = current.rows();
            let h_all = gpu_compute.allocate_gpu_matrix_handle(batch * seq_len * state_dim, 1);
            let params_len = state_dim * state_dim + state_dim * input_dim + input_dim * state_dim + 2;
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(batch, seq_len * input_dim);
            gpu_compute.run_mamba_forward_buffered_handle(
                &current,
                &params_view,
                &out_handle,
                seq_len,
                input_dim,
                state_dim,
                &h_all,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::Mamba {
                input: current.clone(),
                h_all,
            }));
            current = out_handle;
        } else if let Some(lin_att) = layer.as_linear_attention() {
            let seq_len = lin_att.seq_len;
            let d_model = lin_att.d_model;
            let batch = current.rows();
            let total_tokens = batch * seq_len;

            let q_raw = gpu_compute.allocate_gpu_matrix_handle(total_tokens, d_model);
            let k_raw = gpu_compute.allocate_gpu_matrix_handle(total_tokens, d_model);
            let v_raw = gpu_compute.allocate_gpu_matrix_handle(total_tokens, d_model);
            let q_phi = gpu_compute.allocate_gpu_matrix_handle(total_tokens, d_model);
            let k_phi = gpu_compute.allocate_gpu_matrix_handle(total_tokens, d_model);
            let kv = gpu_compute.allocate_gpu_matrix_handle(d_model * d_model, 1);
            let z = gpu_compute.allocate_gpu_matrix_handle(d_model, 1);

            let param_len = lin_att.param_len();
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, param_len);

            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), lin_att.output_features());

            gpu_compute.run_linear_attention_forward_buffered_handle_with_dims(
                &current,
                &params_view,
                &out_handle,
                seq_len,
                d_model,
                &q_raw,
                &k_raw,
                &v_raw,
                &q_phi,
                &k_phi,
                &kv,
                &z,
            );

            ctxs.push(DynamicContext::Buffered(BufferedContext::LinearAttention {
                input: current.clone(),
                q_raw: Some(q_raw),
                k_raw: Some(k_raw),
                v_raw: Some(v_raw),
                q_phi: Some(q_phi),
                k_phi: Some(k_phi),
                kv: Some(kv),
                z: Some(z),
            }));
            current = out_handle;
        } else if let Some(rel_att) = layer.as_relative_position_attention() {
            let seq_len = rel_att.seq_len;
            let d_model = rel_att.d_model;
            let batch = current.rows();
            let total_tokens = batch * seq_len;

            let q = gpu_compute.allocate_gpu_matrix_handle(total_tokens, d_model);
            let k = gpu_compute.allocate_gpu_matrix_handle(total_tokens, d_model);
            let v = gpu_compute.allocate_gpu_matrix_handle(total_tokens, d_model);
            let scores = gpu_compute.allocate_gpu_matrix_handle(total_tokens, seq_len);
            let weights = gpu_compute.allocate_gpu_matrix_handle(total_tokens, seq_len);

            let param_len = rel_att.param_len();
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, param_len);

            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), rel_att.output_features());

            gpu_compute.run_relative_position_attention_forward_buffered_handle(
                &current,
                &params_view,
                &out_handle,
                seq_len,
                d_model,
                &q,
                &k,
                &v,
                &scores,
                &weights,
            );

            ctxs.push(DynamicContext::Buffered(BufferedContext::RelativePositionAttention {
                input: current.clone(),
                q: Some(q),
                k: Some(k),
                v: Some(v),
                scores: Some(scores),
                weights: Some(weights),
            }));
            current = out_handle;
        } else if let Some(sn) = layer.as_spectral_norm_linear() {
            let in_feat = sn.in_features;
            let out_feat = sn.out_features;
            let params_len = in_feat * out_feat + out_feat + 1;
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
            // Временные GPU буферы для состояния (u, v, sigma)
            let u_state = gpu_compute.allocate_gpu_matrix_handle(in_feat, 1);
            let v_state = gpu_compute.allocate_gpu_matrix_handle(out_feat, 1);
            let sigma_state = gpu_compute.allocate_gpu_matrix_handle(1, 1);
            // Загружаем начальные значения (можно инициализировать единицами)
            gpu_compute.fill_gpu_handle(&u_state, 1.0);
            gpu_compute.fill_gpu_handle(&v_state, 1.0);
            gpu_compute.fill_gpu_handle(&sigma_state, 1.0);

            // Извлекаем scale из параметров (последний элемент)
            let scale_view = MatrixBufferView::new(
                params_handle.clone(),
                slice.start + in_feat * out_feat + out_feat,
                1,
            );
            let scale_cpu_handle = gpu_compute.download_gpu_handle_to_cpu_handle(scale_view.parent_handle());
            let scale = {
                let guard = scale_cpu_handle.read();
                guard.as_slice().unwrap()[scale_view.offset_elements()]
            };

            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), out_feat);
            gpu_compute.run_spectral_norm_linear_forward_buffered_handle(
                &current,
                &params_view,
                scale,
                &out_handle,
                &u_state,
                &v_state,
                &sigma_state,
            );

            // После forward получаем sigma с GPU и сохраняем в слое
            let sigma_vec = gpu_compute.download_gpu_handle_to_vec(&sigma_state);
            let sigma = sigma_vec[0];
            sn.set_last_sigma(sigma);

            ctxs.push(DynamicContext::Buffered(BufferedContext::SpectralNormLinear {
                input: current.clone(),
            }));
            current = out_handle;
        } else {
            panic!(
                "Unsupported layer in GPU buffered forward: {:?}",
                std::any::type_name_of_val(layer.as_ref())
            );
        }
    }

    (current, ctxs)
}

/// Обратный проход на GPU с использованием MatrixBufferHandle.
/// Входной градиент — GPU-дескриптор, выходной градиент — GPU-дескриптор.
/// Градиенты параметров записываются напрямую в `grad_params_handle` (GPU).
pub fn process_backward_gpu_buffered(
    gpu_compute: &GpuCompute,
    layers: &[Box<dyn UniversalLayer>],
    slices: &[ParamSlice],
    contexts: &[DynamicContext],
    params_handle: &MatrixBufferHandle,
    grad_output: MatrixBufferHandle,
    grad_params_handle: &MatrixBufferHandle,
) -> MatrixBufferHandle {
    assert!(grad_output.is_gpu(), "Grad output must be GPU handle");
    assert!(
        grad_params_handle.is_gpu() || grad_params_handle.rows() == 0,
        "grad_params_handle must be GPU or empty"
    );

    // ========================================================================
    // КРИТИЧНО: обнуляем буфер градиентов параметров перед backward.
    //
    // GPU-шейдеры слоёв (linear_bwd.comp, learnable_softplus_bwd.comp и все
    // остальные *_bwd.comp, у которых есть параметры) накапливают градиенты
    // по параметрам через ATOMIC_ADD_FLOAT (CAS-loop). Это необходимо для
    // корректной работы внутри одного шейдера, где много потоков пишут в
    // одни и те же ячейки.
    //
    // Однако если буфер grad_params не обнулить перед вызовом backward, то
    // атомарные сложения будут добавляться к градиентам от ПРЕДЫДУЩЕГО
    // backward. Это приводит к экспоненциальному накоплению:
    //   grad(batch=1) = grad(0) + grad(1)
    //   grad(batch=2) = grad(0) + grad(1) + grad(2)
    //   ...
    //
    // CPU-ветка от этой проблемы свободна: CPU-реализации слоёв пишут
    // градиенты параметров напрямую (`gp[i] = sum`), без accumulation,
    // тем самым неявно перезаписывая предыдущие значения.
    //
    // Явное обнуление здесь эквивалентно перезаписи и делает семантику
    // GPU-пути согласованной с CPU-путём.
    //
    // Это было подтверждено логами: V3 GPU на батче 1 даёт
    //   grad = grad(batch=0) + grad(batch=1),
    // что точно совпадает по сумме, а не по отдельным значениям.
    // ========================================================================
    if grad_params_handle.is_gpu()
        && grad_params_handle.rows() * grad_params_handle.cols() > 0
    {
        gpu_compute.fill_gpu_handle(grad_params_handle, 0.0);
    }

    let num_layers = layers.len();
    assert_eq!(contexts.len(), num_layers);
    assert_eq!(slices.len(), num_layers);

    let mut current_grad = grad_output;

    for idx in (0..num_layers).rev() {
        let layer = &layers[idx];
        let slice = &slices[idx];
        let ctx = &contexts[idx];

        // ================ Существующие слои ================
        if let Some(linear) = layer.as_linear() {
            let in_feat = linear.input_features();
            let out_feat = linear.output_features();
            let w_start = slice.start;
            let b_start = w_start + in_feat * out_feat;

            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::Linear { input } => input.clone(),
                _ => panic!("Expected Linear Buffered context"),
            };

            let weight_view = MatrixBufferView::with_shape(
                params_handle.clone(),
                w_start,
                in_feat * out_feat,
                out_feat,
                in_feat,
            );
            let grad_weight_view = MatrixBufferView::with_shape(
                grad_params_handle.clone(),
                w_start,
                in_feat * out_feat,
                out_feat,
                in_feat,
            );
            let grad_bias_view = MatrixBufferView::with_shape(
                grad_params_handle.clone(),
                b_start,
                out_feat,
                1,
                out_feat,
            );

            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), in_feat);
            gpu_compute.run_linear_backward_buffered_handle(
                &input_handle,
                &weight_view,
                &current_grad,
                &grad_input_handle,
                &grad_weight_view,
                &grad_bias_view,
            );
            current_grad = grad_input_handle;
        } else if layer.as_relu().is_some() {
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::ReLU { input } => input.clone(),
                _ => panic!("Expected ReLU Buffered context"),
            };
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_relu_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &grad_input_handle,
            );
            current_grad = grad_input_handle;
        } else if layer.as_sigmoid().is_some() {
            let DynamicContext::Buffered(bc) = ctx;
            let output_handle = match bc {
                BufferedContext::Sigmoid { output } => output.clone(),
                _ => panic!("Expected Sigmoid Buffered context"),
            };
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_sigmoid_backward_buffered_handle(
                &output_handle,
                &current_grad,
                &grad_input_handle,
            );
            current_grad = grad_input_handle;
        } else if layer.as_tanh().is_some() {
            let DynamicContext::Buffered(bc) = ctx;
            let output_handle = match bc {
                BufferedContext::Tanh { output } => output.clone(),
                _ => panic!("Expected Tanh Buffered context"),
            };
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_tanh_backward_buffered_handle(
                &output_handle,
                &current_grad,
                &grad_input_handle,
            );
            current_grad = grad_input_handle;
        } else if let Some(leaky) = layer.as_leaky_relu() {
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::LeakyReLU { input } => input.clone(),
                _ => panic!("Expected LeakyReLU Buffered context"),
            };
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_leaky_relu_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &grad_input_handle,
                leaky.alpha,
            );
            current_grad = grad_input_handle;
        } else if layer.as_softmax().is_some() {
            let DynamicContext::Buffered(bc) = ctx;
            let output_handle = match bc {
                BufferedContext::Softmax { output } => output.clone(),
                _ => panic!("Expected Softmax Buffered context"),
            };
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_softmax_backward_buffered_handle(
                &output_handle,
                &current_grad,
                &grad_input_handle,
            );
            current_grad = grad_input_handle;
        } else if layer.as_identity().is_some() {
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_identity_backward_buffered_handle(
                &current_grad,
                &grad_input_handle,
            );
            current_grad = grad_input_handle;
        } else if let Some(memory) = layer.as_memory() {
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_memory_backward_buffered_handle(
                &current_grad,
                &grad_input_handle,
                memory.alpha,
            );
            current_grad = grad_input_handle;
        } else if let Some(soft_sparse) = layer.as_soft_sparse_gate() {
            let features = soft_sparse.in_features;
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::SoftSparseGate { input } => input.clone(),
                _ => panic!("Expected SoftSparseGate Buffered context"),
            };
            let thresholds_view = MatrixBufferView::new(params_handle.clone(), slice.start, features);
            let grad_thresh_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, features);
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_softsparse_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &thresholds_view,
                soft_sparse.temperature,
                &grad_input_handle,
                &grad_thresh_view,
            );
            current_grad = grad_input_handle;
        } else if let Some(soft_keep) = layer.as_soft_keep_gate() {
            let features = soft_keep.in_features;
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::SoftKeepGate { input } => input.clone(),
                _ => panic!("Expected SoftKeepGate Buffered context"),
            };
            let thresholds_view = MatrixBufferView::new(params_handle.clone(), slice.start, features);
            let grad_thresh_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, features);
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_softkeep_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &thresholds_view,
                soft_keep.temperature,
                &grad_input_handle,
                &grad_thresh_view,
            );
            current_grad = grad_input_handle;
        } else if let Some(dual) = layer.as_dual_anchor() {
            let features = dual.features;
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::DualAnchor1D { input } => input.clone(),
                _ => panic!("Expected DualAnchor1D Buffered context"),
            };
            let min_view = MatrixBufferView::new(params_handle.clone(), slice.start, features);
            let max_view = MatrixBufferView::new(params_handle.clone(), slice.start + features, features);
            let alpha_view = MatrixBufferView::new(params_handle.clone(), slice.start + 2 * features, 1);

            // Единый view на весь блок градиентов параметров слоя:
            // [grad_min (features), grad_max (features), grad_alpha (1)].
            let grad_params_view = MatrixBufferView::new(
                grad_params_handle.clone(),
                slice.start,
                2 * features + 1,
            );

            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_dualanchor_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &min_view,
                &max_view,
                &alpha_view,
                &grad_input_handle,
                &grad_params_view,
            );
            current_grad = grad_input_handle;
        } else if let Some(adaptive) = layer.as_adaptive_activation() {
            let in_features = adaptive.in_features;
            let num_activations = adaptive.num_activations;
            let params_len = in_features * num_activations;

            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::AdaptiveActivation { input } => input.clone(),
                _ => panic!("Expected AdaptiveActivation Buffered context"),
            };

            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
            let grad_params_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, params_len);

            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), in_features);
            gpu_compute.run_adaptive_activation_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &params_view,
                num_activations,
                &grad_input_handle,
                &grad_params_view,
            );
            current_grad = grad_input_handle;
        }
        // ================ Новые слои ================
        else if let Some(dslope) = layer.as_dual_slope_relu() {
            let features = dslope.features;
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::DualSlopeReLU { input } => input.clone(),
                _ => panic!("Expected DualSlopeReLU Buffered context"),
            };
            let alpha_view = MatrixBufferView::new(params_handle.clone(), slice.start, features);
            let beta_view = MatrixBufferView::new(params_handle.clone(), slice.start + features, features);
            let grad_alpha_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, features);
            let grad_beta_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start + features, features);
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_dual_slope_relu_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &alpha_view,
                &beta_view,
                &grad_input_handle,
                &grad_alpha_view,
                &grad_beta_view,
            );
            current_grad = grad_input_handle;
        } else if let Some(mish) = layer.as_learnable_mish() {
            let features = mish.features;
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::LearnableMish { input } => input.clone(),
                _ => panic!("Expected LearnableMish Buffered context"),
            };
            let lambda_view = MatrixBufferView::new(params_handle.clone(), slice.start, 1);
            let grad_lambda_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, 1);
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), features);
            gpu_compute.run_learnable_mish_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &lambda_view,
                &grad_input_handle,
                &grad_lambda_view,
            );
            current_grad = grad_input_handle;
        } else if let Some(softplus) = layer.as_learnable_softplus() {
            let features = softplus.features;
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::LearnableSoftplus { input } => input.clone(),
                _ => panic!("Expected LearnableSoftplus Buffered context"),
            };
            let beta_view = MatrixBufferView::new(params_handle.clone(), slice.start, features);
            let theta_view = MatrixBufferView::new(params_handle.clone(), slice.start + features, features);
            let grad_beta_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, features);
            let grad_theta_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start + features, features);
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), features);
            gpu_compute.run_learnable_softplus_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &beta_view,
                &theta_view,
                &grad_input_handle,
                &grad_beta_view,
                &grad_theta_view,
            );
            current_grad = grad_input_handle;
        } else if let Some(rms) = layer.as_rms_norm_learnable_eps() {
            let features = rms.features;
            let params_len = 2 * features;
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::RMSNormWithLearnableEpsilon { input } => input.clone(),
                _ => panic!("Expected RMSNormWithLearnableEpsilon Buffered context"),
            };
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
            let grad_params_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, params_len);
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), features);
            gpu_compute.run_rms_norm_learnable_eps_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &params_view,
                &grad_input_handle,
                &grad_params_view,
            );
            current_grad = grad_input_handle;
        } else if let Some(adrop) = layer.as_adaptive_dropout() {
            let features = adrop.features;
            let params_len = 2 * features;
            let DynamicContext::Buffered(bc) = ctx;
            let (input_handle, mask_handle, arg_handle) = match bc {
                BufferedContext::AdaptiveDropout { input, mask, arg } => (input.clone(), mask.clone(), arg.clone()),
                _ => panic!("Expected AdaptiveDropout Buffered context"),
            };
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
            let grad_params_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, params_len);
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), features);
            gpu_compute.run_adaptive_dropout_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &params_view,
                &mask_handle,
                &arg_handle,
                &grad_input_handle,
                &grad_params_view,
            );
            current_grad = grad_input_handle;
        } else if let Some(fusion) = layer.as_feature_fusion() {
            let in_features = fusion.in_features;
            let out_features = fusion.out_features;
            let params_len = out_features * (in_features + 1);
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::FeatureFusion { input } => input.clone(),
                _ => panic!("Expected FeatureFusion Buffered context"),
            };
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
            let grad_params_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, params_len);
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), in_features);
            gpu_compute.run_feature_fusion_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &params_view,
                &grad_input_handle,
                &grad_params_view,
            );
            current_grad = grad_input_handle;
        } else if let Some(sfs) = layer.as_sparse_feature_selection_gate() {
            let features = sfs.features;
            let params_len = features + 1;
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::SparseFeatureSelectionGate { input } => input.clone(),
                _ => panic!("Expected SparseFeatureSelectionGate Buffered context"),
            };
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
            let grad_params_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, params_len);
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), features);
            gpu_compute.run_sparse_feature_selection_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &params_view,
                &grad_input_handle,
                &grad_params_view,
            );
            current_grad = grad_input_handle;
        } else if let Some(kan) = layer.as_multi_resolution_kan_linear() {
            let in_features = kan.in_features;
            let out_features = kan.out_features;
            let params_len = in_features * out_features * 12 + out_features;
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::MultiResolutionKANLinear { input } => input.clone(),
                _ => panic!("Expected MultiResolutionKANLinear Buffered context"),
            };
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
            let grad_params_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, params_len);
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), in_features);
            gpu_compute.run_multi_resolution_kan_linear_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &params_view,
                &grad_input_handle,
                &grad_params_view,
            );
            current_grad = grad_input_handle;
        } else if let Some(adnorm) = layer.as_adaptive_normalization() {
            let features = adnorm.features;
            let params_len = 7 * features;
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::AdaptiveNormalization { input } => input.clone(),
                _ => panic!("Expected AdaptiveNormalization Buffered context"),
            };
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
            let grad_params_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, params_len);
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), features);
            gpu_compute.run_adaptive_norm_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &params_view,
                &grad_input_handle,
                &grad_params_view,
            );
            current_grad = grad_input_handle;
        } else if let Some(bn) = layer.as_batch_renorm() {
            let features = bn.features;
            let params_len = 4 * features;
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::BatchRenorm { input, .. } => input.clone(),
                _ => panic!("Expected BatchRenorm Buffered context"),
            };
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
            let grad_params_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, params_len);
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), features);
            gpu_compute.run_batch_renorm_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &params_view,
                &grad_input_handle,
                &grad_params_view,
            );
            current_grad = grad_input_handle;
        } else if let Some(cdrop) = layer.as_concrete_dropout() {
            let temperature = cdrop.temperature;
            let DynamicContext::Buffered(bc) = ctx;
            let (input_handle, arg_handle) = match bc {
                BufferedContext::ConcreteDropout { input, arg } => (input.clone(), arg.clone()),
                _ => panic!("Expected ConcreteDropout Buffered context"),
            };
            let logit_view = MatrixBufferView::new(params_handle.clone(), slice.start, 1);
            let grad_logit_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, 1);
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_concrete_dropout_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &logit_view,
                temperature,
                &arg_handle,
                &grad_input_handle,
                &grad_logit_view,
            );
            current_grad = grad_input_handle;
        } else if let Some(ind) = layer.as_ind_rnn() {
            let seq_len = ind.seq_len;
            let input_dim = ind.input_dim;
            let DynamicContext::Buffered(bc) = ctx;
            let (input_handle, h_all_handle) = match bc {
                BufferedContext::IndRNN { input, h_all } => (input.clone(), h_all.clone()),
                _ => panic!("Expected IndRNN Buffered context"),
            };
            let params_len = input_dim * input_dim + 2 * input_dim;
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
            let grad_params_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, params_len);
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), seq_len * input_dim);
            gpu_compute.run_ind_rnn_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &params_view,
                &grad_input_handle,
                &grad_params_view,
                seq_len,
                input_dim,
                &h_all_handle,
            );
            current_grad = grad_input_handle;
        } else if let Some(mamba) = layer.as_mamba() {
            let seq_len = mamba.seq_len;
            let input_dim = mamba.input_dim;
            let state_dim = mamba.state_dim;
            let DynamicContext::Buffered(bc) = ctx;
            let (input_handle, h_all_handle) = match bc {
                BufferedContext::Mamba { input, h_all } => (input.clone(), h_all.clone()),
                _ => panic!("Expected Mamba Buffered context"),
            };
            let params_len = state_dim * state_dim + state_dim * input_dim + input_dim * state_dim + 2;
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
            let grad_params_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, params_len);
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), seq_len * input_dim);
            gpu_compute.run_mamba_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &params_view,
                &grad_input_handle,
                &grad_params_view,
                seq_len,
                input_dim,
                state_dim,
                &h_all_handle,
            );
            current_grad = grad_input_handle;
        } else if let Some(lin_att) = layer.as_linear_attention() {
            let seq_len = lin_att.seq_len;
            let d_model = lin_att.d_model;
            let DynamicContext::Buffered(bc) = ctx;
            let (input_handle, q_raw, k_raw, v_raw, q_phi, k_phi, kv, z) = match bc {
                BufferedContext::LinearAttention { input, q_raw, k_raw, v_raw, q_phi, k_phi, kv, z } => {
                    (input.clone(), q_raw.clone().unwrap(), k_raw.clone().unwrap(), v_raw.clone().unwrap(), q_phi.clone().unwrap(), k_phi.clone().unwrap(), kv.clone().unwrap(), z.clone().unwrap())
                },
                _ => panic!("Expected LinearAttention Buffered context"),
            };

            let param_len = lin_att.param_len();
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, param_len);
            let grad_params_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, param_len);

            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), lin_att.input_features());

            gpu_compute.run_linear_attention_backward_buffered_handle_with_dims(
                &input_handle,
                &current_grad,
                &params_view,
                &grad_input_handle,
                &grad_params_view,
                seq_len,
                d_model,
                &q_raw,
                &k_raw,
                &v_raw,
                &q_phi,
                &k_phi,
                &kv,
                &z,
            );
            current_grad = grad_input_handle;
        } else if let Some(rel_att) = layer.as_relative_position_attention() {
            let seq_len = rel_att.seq_len;
            let d_model = rel_att.d_model;
            let DynamicContext::Buffered(bc) = ctx;
            let (input_handle, q, k, v, scores, weights) = match bc {
                BufferedContext::RelativePositionAttention { input, q, k, v, scores, weights } => {
                    (input.clone(), q.clone().unwrap(), k.clone().unwrap(), v.clone().unwrap(), scores.clone().unwrap(), weights.clone().unwrap())
                },
                _ => panic!("Expected RelativePositionAttention Buffered context"),
            };

            let param_len = rel_att.param_len();
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, param_len);
            let grad_params_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, param_len);

            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), rel_att.input_features());

            gpu_compute.run_relative_position_attention_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &params_view,
                &grad_input_handle,
                &grad_params_view,
                seq_len,
                d_model,
                &q,
                &k,
                &v,
                &scores,
                &weights,
            );
            current_grad = grad_input_handle;
        } else if let Some(sn) = layer.as_spectral_norm_linear() {
            let in_feat = sn.in_features;
            let out_feat = sn.out_features;
            let params_len = in_feat * out_feat + out_feat + 1;
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::SpectralNormLinear { input } => input.clone(),
                _ => panic!("Expected SpectralNormLinear Buffered context"),
            };
            let params_view = MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
            let grad_params_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, params_len);
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), in_feat);

            // Извлекаем scale из параметров
            let scale_view = MatrixBufferView::new(
                params_handle.clone(),
                slice.start + in_feat * out_feat + out_feat,
                1,
            );
            let scale_cpu_handle = gpu_compute.download_gpu_handle_to_cpu_handle(scale_view.parent_handle());
            let scale = {
                let guard = scale_cpu_handle.read();
                guard.as_slice().unwrap()[scale_view.offset_elements()]
            };

            let sigma = sn.get_last_sigma();

            gpu_compute.run_spectral_norm_linear_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &params_view,
                &grad_input_handle,
                &grad_params_view,
                scale,
                sigma,
            );
            current_grad = grad_input_handle;
        } else {
            panic!(
                "Unsupported layer in GPU buffered backward: {:?}",
                std::any::type_name_of_val(layer.as_ref())
            );
        }
    }

    current_grad
}
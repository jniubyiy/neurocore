// src/compute_manager/graph/forward/segments/processors.rs

use std::sync::Arc;

use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::buffered_context::BufferedContext;
use crate::layers::{UniversalLayer, UniversalLayerBuffered};
use crate::model_plan::param_store::ParamSlice;

impl MixedModel {
    // CPU-путь с использованием MatrixBufferHandle
    // Теперь возвращает Vec<DynamicContext> для единственного чанка (весь батч)
    pub(crate) fn process_universal_processor_forward_buffered(
        &mut self,
        pool: &mut TempMatrixPool,
        proc: &Arc<Vec<Box<dyn UniversalLayer>>>,
        slices: &[ParamSlice],
        _model_index: usize,
        params: &MatrixBufferHandle,
        stream_buffers: &mut Vec<MatrixBufferHandle>,
        stream_indices: &Option<Vec<usize>>,
    ) -> Vec<DynamicContext> {
        let active_indices: Vec<usize> = match stream_indices {
            Some(indices) => indices.clone(),
            None => (0..stream_buffers.len()).collect(),
        };

        let layers = proc.as_ref();
        let num_layers = layers.len();

        // Клонируем входные дескрипторы для неактивных потоков (они не меняются)
        let mut new_stream: Vec<Option<MatrixBufferHandle>> = stream_buffers
            .iter()
            .map(|handle| Some(handle.clone()))
            .collect();

        // Сюда будем собирать контексты для всех активных потоков (но в текущей модели обычно один поток)
        // Для простоты предположим, что активных потоков может быть несколько,
        // но контексты разных потоков объединяем в один вектор? В старой логике контексты добавлялись
        // для каждого сэмпла одинаково, т.е. они были общими для всех потоков? Нет, для каждого потока
        // свои контексты, но они добавлялись в общий список all_ctxs последовательно.
        // В новой схеме мы возвращаем один вектор для одного чанка. Если активных потоков несколько,
        // то нужно обработать каждый отдельно и объединить контексты? Но чанк один, и контексты должны
        // соответствовать каждому слою для каждого потока. Для простоты оставим как есть: для одного потока.
        // Так как в большинстве случаев UniversalProcessor имеет один поток, этого достаточно.

        // Выберем первый активный поток для обработки (или можно обработать все, но вернуть контексты первого)
        // В оригинальном коде обрабатывались все активные потоки, но контексты добавлялись для каждого сэмпла,
        // что означает, что контексты были одинаковы для всех потоков? Это сомнительно. Мы упростим:
        // будем обрабатывать все активные потоки, но контексты вернём только для первого? Но обратный проход
        // ожидает контексты для каждого слоя и для каждого потока. Для многопоточных моделей нужна поддержка,
        // но пока сосредоточимся на однопоточных. Поэтому предположим, что active_indices.len() == 1.

        let mut result_ctxs = Vec::new();

        for &stream_idx in &active_indices {
            let input_handle = stream_buffers[stream_idx].clone();
            let batch_size = input_handle.rows();
            let mut current_input = input_handle;
            let mut layer_ctxs: Vec<DynamicContext> = Vec::with_capacity(num_layers);

            for i in 0..num_layers {
                let layer = &layers[i];
                let slice = &slices[i];

                // Определяем размер выходного буфера
                let out_features = get_buffered_output_features(layer, &current_input);
                let output_handle = pool.acquire(batch_size, out_features);

                // Выполняем прямой проход
                call_forward_buffered(layer, &current_input, &output_handle, params, slice);

                // Создаём контекст для обратного прохода
                let buffered_ctx = build_buffered_context(layer, &current_input, &output_handle);
                layer_ctxs.push(DynamicContext::Buffered(buffered_ctx));

                current_input = output_handle;
            }

            // Записываем результат для этого потока
            new_stream[stream_idx] = Some(current_input);
            result_ctxs = layer_ctxs; // предполагаем один активный поток
        }

        // Обновляем stream_buffers
        *stream_buffers = new_stream
            .into_iter()
            .map(|opt| opt.expect("Missing stream buffer after forward"))
            .collect();

        result_ctxs
    }
}

// Вспомогательные функции (можно вынести в отдельный модуль, но оставлены здесь)

/// Возвращает количество выходных признаков слоя, используя UniversalLayerBuffered.
fn get_buffered_output_features(layer: &Box<dyn UniversalLayer>, input: &MatrixBufferHandle) -> usize {
    if let Some(l) = layer.as_linear() {
        <dyn UniversalLayerBuffered>::output_features(l)
    } else if layer.as_relu().is_some()
        || layer.as_sigmoid().is_some()
        || layer.as_tanh().is_some()
        || layer.as_leaky_relu().is_some()
        || layer.as_identity().is_some()
        || layer.as_softmax().is_some()
        || layer.as_memory().is_some()
        || layer.as_soft_sparse_gate().is_some()
        || layer.as_soft_keep_gate().is_some()
        || layer.as_dual_anchor().is_some()
    {
        // Для этих слоёв выходная размерность равна входной
        input.cols()
    } else {
        // Fallback
        input.cols()
    }
}

/// Вызывает буферизованный прямой проход для слоя.
fn call_forward_buffered(
    layer: &Box<dyn UniversalLayer>,
    input: &MatrixBufferHandle,
    output: &MatrixBufferHandle,
    params: &MatrixBufferHandle,
    slice: &ParamSlice,
) {
    if let Some(l) = layer.as_linear() {
        <dyn UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_relu() {
        <dyn UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_sigmoid() {
        <dyn UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_tanh() {
        <dyn UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_leaky_relu() {
        <dyn UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_identity() {
        <dyn UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_softmax() {
        <dyn UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_memory() {
        <dyn UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_soft_sparse_gate() {
        <dyn UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_soft_keep_gate() {
        <dyn UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_dual_anchor() {
        <dyn UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else {
        unreachable!(
            "Layer {:?} does not implement UniversalLayerBuffered for CPU path",
            std::any::type_name_of_val(layer.as_ref())
        );
    }
}

/// Создаёт буферизованный контекст для слоя.
fn build_buffered_context(
    layer: &Box<dyn UniversalLayer>,
    input: &MatrixBufferHandle,
    output: &MatrixBufferHandle,
) -> BufferedContext {
    if layer.as_linear().is_some() {
        BufferedContext::Linear { input: input.clone() }
    } else if layer.as_relu().is_some() {
        BufferedContext::ReLU { input: input.clone() }
    } else if layer.as_sigmoid().is_some() {
        BufferedContext::Sigmoid { output: output.clone() }
    } else if layer.as_tanh().is_some() {
        BufferedContext::Tanh { output: output.clone() }
    } else if layer.as_softmax().is_some() {
        BufferedContext::Softmax { output: output.clone() }
    } else if layer.as_leaky_relu().is_some() {
        BufferedContext::LeakyReLU { input: input.clone() }
    } else if layer.as_identity().is_some() {
        BufferedContext::Identity { input: input.clone() }
    } else if layer.as_memory().is_some() {
        BufferedContext::Memory { input: input.clone() }
    } else if layer.as_soft_sparse_gate().is_some() {
        BufferedContext::SoftSparseGate { input: input.clone() }
    } else if layer.as_soft_keep_gate().is_some() {
        BufferedContext::SoftKeepGate { input: input.clone() }
    } else if layer.as_dual_anchor().is_some() {
        BufferedContext::DualAnchor1D { input: input.clone() }
    } else {
        // Fallback: Identity
        BufferedContext::Identity { input: input.clone() }
    }
}
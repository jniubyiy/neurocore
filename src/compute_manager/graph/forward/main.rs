// src/compute_manager/graph/forward/main.rs

use faer::Mat;
use crate::compute_manager::dim_change;
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::{DynamicContext, Segment};

impl MixedModel {
    /// Прямой матричный проход с множественными входами и выходами.
    /// Вход: срез матриц (по одной на каждый входной поток).
    /// Выход: вектор выходных матриц и контексты (один набор контекстов для всего батча).
    pub fn forward_mat_multi(
        &self,
        inputs: &[Mat<f32>],
    ) -> (Vec<Mat<f32>>, Vec<Vec<DynamicContext>>) {
        assert_eq!(
            inputs.len(),
            self.input_stream_count,
            "forward_mat_multi: expected {} inputs, got {}",
            self.input_stream_count,
            inputs.len()
        );

        let batch_size = if let Some(first) = inputs.first() {
            first.nrows()
        } else {
            return (Vec::new(), Vec::new());
        };
        for mat in inputs {
            assert_eq!(
                mat.nrows(),
                batch_size,
                "All input matrices must have the same number of rows (batch size)"
            );
        }

        // Начальные потоки матриц (каждый элемент — матрица для одного потока)
        let mut stream_matrices: Vec<Mat<f32>> = inputs.to_vec();
        // Контексты для каждого сэмпла в батче (пока пустые)
        let mut all_ctxs: Vec<Vec<DynamicContext>> = vec![Vec::new(); batch_size];

        // Исполняем сегменты графа
        for (seg_index, seg) in self.segments.iter().enumerate() {
            match seg {
                Segment::Unsqueeze(target_dims) => {
                    for mat in &mut stream_matrices {
                        *mat = dim_change::unsqueeze_mat(mat, target_dims);
                    }
                }
                Segment::ReduceMean(target_dims) => {
                    for mat in &mut stream_matrices {
                        *mat = dim_change::reduce_mat(mat, target_dims);
                    }
                }
                Segment::UniversalProcessor(proc, slices, stream_indices) => {
                    let params = self.store.lock().unwrap().all_params();
                    self.process_universal_processor_forward(
                        proc,
                        slices,
                        seg_index,
                        &params,
                        &mut stream_matrices,
                        &mut all_ctxs,
                        stream_indices,
                    );
                }
                Segment::SplitterConnector { dim_a, dim_b } => {
                    self.process_splitter_connector_forward(
                        *dim_a,
                        *dim_b,
                        batch_size,
                        &mut stream_matrices,
                        &mut all_ctxs,
                    );
                }
                Segment::CombinerConnector { input_dims, .. } => {
                    self.process_combiner_connector_forward(
                        input_dims.clone(),
                        batch_size,
                        &mut stream_matrices,
                        &mut all_ctxs,
                    );
                }
                Segment::Splitter {
                    input_dim,
                    output_dims,
                    slice,
                } => {
                    self.process_splitter_forward(
                        *input_dim,
                        output_dims.clone(),
                        *slice,
                        batch_size,
                        &mut stream_matrices,
                        &mut all_ctxs,
                    );
                }
                Segment::Combiner {
                    input_dim,
                    output_dim,
                    slice,
                } => {
                    self.process_combiner_forward(
                        *input_dim,
                        *output_dim,
                        *slice,
                        batch_size,
                        &mut stream_matrices,
                        &mut all_ctxs,
                    );
                }
            }
        }

        assert_eq!(
            stream_matrices.len(),
            self.output_stream_count,
            "forward_mat_multi: output stream count mismatch"
        );

        (stream_matrices, all_ctxs)
    }

    /// Обычный матричный проход (один вход – один выход).
    /// Оставлен для обратной совместимости.
    pub fn forward_mat(
        &self,
        input: &Mat<f32>,
    ) -> (Mat<f32>, Vec<Vec<DynamicContext>>) {
        let (outs, ctxs) = self.forward_mat_multi(&[input.clone()]);
        assert_eq!(outs.len(), 1);
        (outs.into_iter().next().unwrap(), ctxs)
    }
}
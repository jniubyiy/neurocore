// src/compute_manager/graph/forward/main.rs

use faer::Mat;
use crate::compute_manager::dim_change::DynamicTensor;
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
        assert_eq!(inputs.len(), self.input_stream_count,
            "forward_mat_multi: expected {} inputs, got {}", self.input_stream_count, inputs.len());

        let batch_size = if let Some(first) = inputs.first() {
            first.nrows()
        } else {
            return (Vec::new(), Vec::new());
        };
        for mat in inputs {
            assert_eq!(mat.nrows(), batch_size,
                "All input matrices must have the same number of rows (batch size)");
        }

        // Строим начальные потоки: для каждой входной матрицы создаём поток образцов Dim1
        let mut streams: Vec<Vec<DynamicTensor>> = Vec::with_capacity(inputs.len());
        for mat in inputs {
            let cols = mat.ncols();
            let mut stream = Vec::with_capacity(batch_size);
            for r in 0..batch_size {
                let row: Vec<f32> = (0..cols).map(|c| mat[(r, c)]).collect();
                stream.push(DynamicTensor::Dim1(crate::tensor::Tensor2D::new(vec![row])));
            }
            streams.push(stream);
        }

        let mut all_ctxs: Vec<Vec<DynamicContext>> = vec![Vec::new(); batch_size];

        // Исполняем сегменты графа
        for seg in self.segments.iter() {
            match seg {
                Segment::Unsqueeze(target_dims) => {
                    self.process_unsqueeze_forward_mat(&mut streams, target_dims);
                }
                Segment::ReduceMean(target_dims) => {
                    self.process_reduce_mean_forward_mat(&mut streams, target_dims);
                }
                Segment::UniversalProcessor(proc, slices, stream_indices) => {
                    let params = self.store.lock().unwrap().all_params();
                    self.process_universal_processor_forward(
                        proc, slices, 0, &params, &mut streams, &mut all_ctxs, stream_indices,
                    );
                }
                Segment::SplitterConnector { dim_a, dim_b } => {
                    self.process_splitter_connector_forward(
                        *dim_a, *dim_b, batch_size, &mut streams, &mut all_ctxs,
                    );
                }
                Segment::CombinerConnector { input_dims, .. } => {
                    self.process_combiner_connector_forward(
                        input_dims.clone(), batch_size, &mut streams, &mut all_ctxs,
                    );
                }
                Segment::Splitter { input_dim, output_dims, slice } => {
                    self.process_splitter_forward(
                        *input_dim, output_dims.clone(), *slice, batch_size, &mut streams, &mut all_ctxs,
                    );
                }
                Segment::Combiner { input_dim, output_dim, slice } => {
                    self.process_combiner_forward(
                        *input_dim, *output_dim, *slice, batch_size, &mut streams, &mut all_ctxs,
                    );
                }
            }
        }

        assert_eq!(streams.len(), self.output_stream_count,
            "forward_mat_multi: output stream count mismatch");

        // Преобразуем выходные потоки обратно в матрицы
        let out_mats: Vec<Mat<f32>> = streams.iter()
            .map(|stream| MixedModel::samples_to_mat(stream))
            .collect();

        (out_mats, all_ctxs)
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

    // ─────────────────────────────────────────────────
    // Вспомогательные методы обработки сегментов
    // ─────────────────────────────────────────────────
    fn process_unsqueeze_forward_mat(
        &self,
        streams: &mut Vec<Vec<DynamicTensor>>,
        target_dims: &[usize],
    ) {
        for sample in streams.iter_mut().flatten() {
            if let DynamicTensor::Dim1(t) = sample {
                let mat = crate::linalg::tensor2d_to_faer(t);
                let new_mat = crate::compute_manager::dim_change::unsqueeze_mat(&mat, target_dims);
                *sample = DynamicTensor::Dim1(crate::linalg::faer_to_tensor2d(&new_mat));
            } else {
                panic!("Unsqueeze requires Dim1 input");
            }
        }
    }

    fn process_reduce_mean_forward_mat(
        &self,
        streams: &mut Vec<Vec<DynamicTensor>>,
        target_dims: &[usize],
    ) {
        for sample in streams.iter_mut().flatten() {
            if let DynamicTensor::Dim1(t) = sample {
                let mat = crate::linalg::tensor2d_to_faer(t);
                let new_mat = crate::compute_manager::dim_change::reduce_mat(&mat, target_dims);
                *sample = DynamicTensor::Dim1(crate::linalg::faer_to_tensor2d(&new_mat));
            } else {
                panic!("ReduceMean requires Dim1 input");
            }
        }
    }
}
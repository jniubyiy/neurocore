use crate::compute_manager::graph::types::DynamicContext;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;
use crate::linalg;
use faer::Mat;
use std::sync::Mutex;

pub struct Memory {
    features: usize,
    pub alpha: f32,
    cells: Mutex<Vec<f32>>,
}

impl Memory {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        assert_eq!(in_features, out_features,
            "Memory: in_features must equal out_features");
        let mut cells = Vec::with_capacity(2 * in_features);
        cells.resize(in_features, f32::MAX);
        cells.resize(2 * in_features, f32::MIN);
        Self {
            features: in_features,
            alpha: 0.1,
            cells: Mutex::new(cells),
        }
    }

    fn forward_mat_impl(&self, input: &Mat<f32>) -> Mat<f32> {
        let batch = input.nrows();
        let features = self.features;
        let mut output = Mat::zeros(batch, features);
        let mut cells = self.cells.lock().unwrap();

        for r in 0..batch {
            for c in 0..features {
                let x = input[(r, c)];
                let min_idx = c;
                let max_idx = features + c;
                let min_val = cells[min_idx];
                let max_val = cells[max_idx];

                let d_min = (x - min_val).abs();
                let d_max = (x - max_val).abs();
                let closest = if d_min <= d_max { min_val } else { max_val };
                output[(r, c)] = x + self.alpha * (closest - x);

                if x > max_val {
                    cells[max_idx] += self.alpha * (x - max_val);
                } else if x < min_val {
                    cells[min_idx] += self.alpha * (x - min_val);
                } else {
                    cells[min_idx] += self.alpha * (x - min_val);
                    cells[max_idx] += self.alpha * (x - max_val);
                }
            }
        }
        output
    }
}

impl UniversalLayer for Memory {
    fn forward_mat(
        &self,
        input: &Mat<f32>,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Mat<f32>, DynamicContext) {
        let output = self.forward_mat_impl(input);
        let input_tensor = linalg::faer_to_tensor2d(input);
        let ctx = DynamicContext::Ctx1D(
            crate::layers::context1d::LayerContext1D::Memory { input: input_tensor },
        );
        (output, ctx)
    }

    fn backward_mat(
        &self,
        _ctx: &DynamicContext,
        delta: &Mat<f32>,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Mat<f32>, Vec<f32>) {
        let factor = 1.0 - self.alpha;
        let dx = delta.map(|v| v * factor);
        (dx, vec![])
    }

    fn param_len(&self) -> usize { 0 }
    fn input_features(&self) -> usize { self.features }
    fn output_features(&self) -> usize { self.features }

    fn total_tasks(&self, batch_size: usize) -> usize { batch_size }

    fn execute_tasks(
        &self,
        input: &Mat<f32>,
        output: &mut Mat<f32>,
        task_offset: usize,
        task_count: usize,
        _params: &[f32],
        _slice: &ParamSlice,
    ) {
        let chunk = input.submatrix(task_offset, 0, task_count, self.features);
        let out_chunk = self.forward_mat_impl(&chunk.to_owned());
        for r in 0..task_count {
            for c in 0..self.features {
                output[(task_offset + r, c)] = out_chunk[(r, c)];
            }
        }
    }

    fn create_sample_context(
        &self,
        input_sample: &Mat<f32>,
        _output_sample: &Mat<f32>,
    ) -> DynamicContext {
        let t = linalg::faer_to_tensor2d(input_sample);
        DynamicContext::Ctx1D(
            crate::layers::context1d::LayerContext1D::Memory { input: t },
        )
    }

    fn output_mat_shape(&self, batch_size: usize) -> Mat<f32> {
        Mat::zeros(batch_size, self.features)
    }
}
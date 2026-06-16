use std::time::Instant;
use crate::layers::{Layer2D, Layer};
use crate::tensor::{Tensor2D, Tensor1D};
use crate::jacobian::{Jacobian2D, Jacobian};
use crate::dispatchers::common::task::RangeTask;

// --- SchedulePredictor ---
pub struct SchedulePredictor {
    model: LinearLayer,
}

impl SchedulePredictor {
    pub fn new() -> Self {
        let mut model = LinearLayer::new(5, 1, 0);
        model.set_params(&vec![0.1, 0.1, 0.1, 0.1, 0.1, 1.0]);
        SchedulePredictor { model }
    }

    pub fn predict_num_tasks(&self, features: &LayerFeatures) -> usize {
        let input = Tensor1D::new(vec![
            features.total_rows,
            features.in_features,
            features.out_features,
            features.is_linear,
            features.num_cores,
        ]);
        let jac = Jacobian::new(5, self.model.param_count());
        let (output, _) = self.model.forward(&input, &jac);
        output.data[0].max(1.0).round() as usize
    }

    pub fn train_step(&mut self, features: &LayerFeatures, target_tasks: f32, lr: f32) {
        let input = Tensor1D::new(vec![
            features.total_rows, features.in_features, features.out_features,
            features.is_linear, features.num_cores,
        ]);
        let jac = Jacobian::new(5, self.model.param_count());
        let (output, _) = self.model.forward(&input, &jac);
        let error = output.data[0] - target_tasks;
        let mut grad = vec![0.0f32; self.model.param_count()];
        for i in 0..5 { grad[i] = error * input.data[i]; }
        grad[5] = error;
        self.model.update_params(lr, &grad);
    }
}

#[derive(Debug, Clone)]
pub struct LayerFeatures {
    pub total_rows: f32,
    pub in_features: f32,
    pub out_features: f32,
    pub is_linear: f32,
    pub num_cores: f32,
}

// --- ProfileCollector ---
pub struct ProfileCollector {
    predictor: SchedulePredictor,
    samples: Vec<(LayerFeatures, f32)>,
}

impl ProfileCollector {
    pub fn new() -> Self {
        ProfileCollector { predictor: SchedulePredictor::new(), samples: Vec::new() }
    }

    pub fn profile_layer(
        &mut self,
        layer: &dyn Layer2D,
        input: &Tensor2D,
        j_input: &Jacobian2D,
        in_features: usize,
        out_features: usize,
        is_linear: bool,
        num_cores: usize,
    ) {
        let total_rows = input.rows;
        let total_params = j_input.params;
        let candidates = vec![1, 2, num_cores.min(8)];
        let mut best_tasks = 1;
        let mut best_time = f64::MAX;
        for num_tasks in candidates {
            let tasks = Self::make_tasks(total_rows, out_features, num_tasks);
            let mut flat_out = vec![0.0f32; total_rows * out_features];
            let mut flat_jac = vec![0.0f32; total_rows * out_features * total_params];
            let start = Instant::now();
            for task in &tasks {
                layer.execute_range(
                    input, j_input,
                    &mut flat_out, &mut flat_jac,
                    task.row_start, task.row_end,
                    task.col_start, task.col_end,
                    total_params,
                );
            }
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed < best_time {
                best_time = elapsed;
                best_tasks = num_tasks;
            }
        }
        let features = LayerFeatures {
            total_rows: total_rows as f32,
            in_features: in_features as f32,
            out_features: out_features as f32,
            is_linear: if is_linear { 1.0 } else { 0.0 },
            num_cores: num_cores as f32,
        };
        self.samples.push((features, best_tasks as f32));
    }

    pub fn train_predictor(&mut self, lr: f32) {
        for (features, target) in &self.samples {
            self.predictor.train_step(features, *target, lr);
        }
    }

    pub fn into_predictor(self) -> SchedulePredictor {
        self.predictor
    }

    fn make_tasks(total_rows: usize, out_features: usize, num_tasks: usize) -> Vec<RangeTask> {
        let rows_per = (total_rows as f64 / num_tasks as f64).ceil() as usize;
        (0..num_tasks).map(|i| {
            let rs = i * rows_per;
            let re = (rs + rows_per).min(total_rows);
            RangeTask { row_start: rs, row_end: re, col_start: 0, col_end: out_features }
        }).collect()
    }
}
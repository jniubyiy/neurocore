use std::sync::Arc;
use std::thread;
use crate::tensor::Tensor3D;
use crate::jacobian::Jacobian3D;
use crate::loss::ops::{LossInput, LossJacobian};
use crate::loss_plan::BuiltLoss;
use crate::dispatchers::common::model_trait::LossDispatch;

pub struct TrainedLoss3D {
    num_workers: usize,
}

impl TrainedLoss3D {
    pub fn new(num_threads: usize) -> Self {
        TrainedLoss3D { num_workers: num_threads.max(1) }
    }
}

impl LossDispatch for TrainedLoss3D {
    fn compute_loss(
        &self,
        pred: &dyn LossInput,
        target: &dyn LossInput,
        j_pred: &dyn LossJacobian,
        built_loss: &BuiltLoss,
    ) -> (f32, Vec<f32>) {
        let pred = pred.as_any().downcast_ref::<Tensor3D>().unwrap();
        let target = target.as_any().downcast_ref::<Tensor3D>().unwrap();
        let j_pred = j_pred.as_any().downcast_ref::<Jacobian3D>().unwrap();

        let depth = pred.depth;
        let rows = pred.rows;
        let cols = pred.cols;
        let p = j_pred.num_params;                      // <-- исправлено
        if depth == 0 || rows == 0 || cols == 0 {
            return (0.0, vec![0.0; p]);
        }

        let num_threads = self.num_workers.min(depth);
        let depth_per_chunk = (depth + num_threads - 1) / num_threads;

        let pred_data = Arc::new(pred.data.clone());
        let target_data = Arc::new(target.data.clone());
        let j_pred_data = Arc::new(j_pred.data.clone());
        let built_loss = Arc::clone(&built_loss.forward);

        let mut handles = Vec::with_capacity(num_threads);
        for tid in 0..num_threads {
            let start_d = tid * depth_per_chunk;
            let end_d = (start_d + depth_per_chunk).min(depth);
            if start_d >= depth { break; }
            let chunk_depth = end_d - start_d;

            let pred_data = Arc::clone(&pred_data);
            let target_data = Arc::clone(&target_data);
            let j_pred_data = Arc::clone(&j_pred_data);
            let built_loss = Arc::clone(&built_loss);

            let handle = thread::spawn(move || {
                let pred_chunk = Tensor3D::new(pred_data[start_d..end_d].to_vec());
                let target_chunk = Tensor3D::new(target_data[start_d..end_d].to_vec());
                let mut j_chunk = Jacobian3D::new(chunk_depth, rows, cols, p);
                for d in 0..chunk_depth {
                    for r in 0..rows {
                        for c in 0..cols {
                            for k in 0..p {
                                j_chunk.data[d][r][c][k] = j_pred_data[start_d + d][r][c][k];
                            }
                        }
                    }
                }
                let (loss_chunk, grad_chunk) = built_loss(&pred_chunk, &target_chunk, &j_chunk);
                let scale = chunk_depth as f32;
                (loss_chunk * scale,
                 grad_chunk.iter().map(|g| g * scale).collect::<Vec<_>>())
            });
            handles.push(handle);
        }

        let mut total_loss = 0.0f32;
        let mut total_grad = vec![0.0f32; p];
        for h in handles {
            let (loss_sum, grad_sum) = h.join().unwrap();
            total_loss += loss_sum;
            for i in 0..p { total_grad[i] += grad_sum[i]; }
        }
        total_loss /= depth as f32;
        for g in &mut total_grad { *g /= depth as f32; }
        (total_loss, total_grad)
    }

    fn num_workers(&self) -> usize { self.num_workers }
}
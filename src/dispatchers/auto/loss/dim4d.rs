use std::sync::Arc;
use std::thread;
use crate::tensor::Tensor4D;
use crate::jacobian::Jacobian4D;
use crate::loss::ops::{LossInput, LossJacobian};
use crate::loss_plan::BuiltLoss;
use crate::dispatchers::common::model_trait::LossDispatch;

pub struct AutoLoss4D {
    num_workers: usize,
}

impl AutoLoss4D {
    pub fn new(num_threads: usize) -> Self {
        AutoLoss4D { num_workers: num_threads.max(1) }
    }
}

impl LossDispatch for AutoLoss4D {
    fn compute_loss(
        &self,
        pred: &dyn LossInput,
        target: &dyn LossInput,
        j_pred: &dyn LossJacobian,
        built_loss: &BuiltLoss,
    ) -> (f32, Vec<f32>) {
        let pred = pred.as_any().downcast_ref::<Tensor4D>()
            .expect("AutoLoss4D: pred must be Tensor4D");
        let target = target.as_any().downcast_ref::<Tensor4D>()
            .expect("AutoLoss4D: target must be Tensor4D");
        let j_pred = j_pred.as_any().downcast_ref::<Jacobian4D>()
            .expect("AutoLoss4D: j_pred must be Jacobian4D");

        let dim1 = pred.dim1;
        let depth = pred.depth;
        let rows = pred.rows;
        let cols = pred.cols;
        let p = j_pred.num_params;                      // <-- исправлено
        if dim1 == 0 || depth == 0 || rows == 0 || cols == 0 {
            return (0.0, vec![0.0; p]);
        }

        let num_threads = self.num_workers.min(dim1);
        let dim1_per_chunk = (dim1 + num_threads - 1) / num_threads;

        let pred_data = Arc::new(pred.data.clone());
        let target_data = Arc::new(target.data.clone());
        let j_pred_data = Arc::new(j_pred.data.clone());
        let built_loss = Arc::clone(&built_loss.forward);

        let mut handles = Vec::with_capacity(num_threads);

        for tid in 0..num_threads {
            let start_d1 = tid * dim1_per_chunk;
            let end_d1 = (start_d1 + dim1_per_chunk).min(dim1);
            if start_d1 >= dim1 { break; }
            let chunk_dim1 = end_d1 - start_d1;

            let pred_data = Arc::clone(&pred_data);
            let target_data = Arc::clone(&target_data);
            let j_pred_data = Arc::clone(&j_pred_data);
            let built_loss = Arc::clone(&built_loss);

            let handle = thread::spawn(move || {
                let pred_chunk = Tensor4D::new(pred_data[start_d1..end_d1].to_vec());
                let target_chunk = Tensor4D::new(target_data[start_d1..end_d1].to_vec());
                let mut j_chunk = Jacobian4D::new(chunk_dim1, depth, rows, cols, p);
                for d1 in 0..chunk_dim1 {
                    for d in 0..depth {
                        for r in 0..rows {
                            for c in 0..cols {
                                for k in 0..p {
                                    j_chunk.data[d1][d][r][c][k] = j_pred_data[start_d1 + d1][d][r][c][k];
                                }
                            }
                        }
                    }
                }

                let (loss_chunk, grad_chunk) = built_loss(&pred_chunk, &target_chunk, &j_chunk);
                let scale = chunk_dim1 as f32;
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
        total_loss /= dim1 as f32;
        for g in &mut total_grad { *g /= dim1 as f32; }
        (total_loss, total_grad)
    }

    fn num_workers(&self) -> usize { self.num_workers }
}
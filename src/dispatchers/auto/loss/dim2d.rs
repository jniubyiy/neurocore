use std::sync::Arc;
use std::thread;
use crate::tensor::Tensor2D;
use crate::jacobian::Jacobian2D;
use crate::loss::ops::{LossInput, LossJacobian};
use crate::loss_plan::BuiltLoss;
use crate::dispatchers::common::model_trait::LossDispatch;

pub struct AutoLoss2D {
    num_workers: usize,
}

impl AutoLoss2D {
    pub fn new(num_threads: usize) -> Self {
        AutoLoss2D { num_workers: num_threads.max(1) }
    }
}

impl LossDispatch for AutoLoss2D {
    fn compute_loss(
        &self,
        pred: &dyn LossInput,
        target: &dyn LossInput,
        j_pred: &dyn LossJacobian,
        built_loss: &BuiltLoss,
    ) -> (f32, Vec<f32>) {
        let pred = pred.as_any().downcast_ref::<Tensor2D>()
            .expect("AutoLoss2D: pred must be Tensor2D");
        let target = target.as_any().downcast_ref::<Tensor2D>()
            .expect("AutoLoss2D: target must be Tensor2D");
        let j_pred = j_pred.as_any().downcast_ref::<Jacobian2D>()
            .expect("AutoLoss2D: j_pred must be Jacobian2D");

        let rows = pred.rows;
        let cols = pred.cols;
        let p = j_pred.num_params;          // <-- исправлено
        if rows == 0 || cols == 0 {
            return (0.0, vec![0.0; p]);
        }

        let num_threads = self.num_workers.min(rows);
        let rows_per_chunk = (rows + num_threads - 1) / num_threads;

        let pred_data = Arc::new(pred.data.clone());
        let target_data = Arc::new(target.data.clone());
        let j_pred_data = Arc::new(j_pred.data.clone());
        let built_loss = Arc::clone(&built_loss.forward);

        let mut handles = Vec::with_capacity(num_threads);

        for tid in 0..num_threads {
            let start_row = tid * rows_per_chunk;
            let end_row = (start_row + rows_per_chunk).min(rows);
            if start_row >= rows { break; }
            let chunk_rows = end_row - start_row;

            let pred_data = Arc::clone(&pred_data);
            let target_data = Arc::clone(&target_data);
            let j_pred_data = Arc::clone(&j_pred_data);
            let built_loss = Arc::clone(&built_loss);

            let handle = thread::spawn(move || {
                let pred_chunk = Tensor2D::new(pred_data[start_row..end_row].to_vec());
                let target_chunk = Tensor2D::new(target_data[start_row..end_row].to_vec());
                let mut j_chunk = Jacobian2D::new(chunk_rows, cols, p);
                for r in 0..chunk_rows {
                    for c in 0..cols {
                        for k in 0..p {
                            j_chunk.data[r][c][k] = j_pred_data[start_row + r][c][k];
                        }
                    }
                }
                let (loss_chunk, grad_chunk) = built_loss(&pred_chunk, &target_chunk, &j_chunk);
                // Масштабируем на число строк чанка
                let scale = chunk_rows as f32;
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
            for i in 0..p {
                total_grad[i] += grad_sum[i];
            }
        }
        total_loss /= rows as f32;
        for g in &mut total_grad { *g /= rows as f32; }
        (total_loss, total_grad)
    }

    fn num_workers(&self) -> usize { self.num_workers }
}
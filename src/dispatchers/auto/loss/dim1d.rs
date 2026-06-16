use std::sync::Arc;
use std::thread;
use crate::tensor::Tensor1D;
use crate::jacobian::Jacobian;
use crate::loss::ops::{LossInput, LossJacobian};
use crate::loss_plan::BuiltLoss;
use crate::dispatchers::common::model_trait::LossDispatch;

pub struct AutoLoss1D {
    num_workers: usize,
}

impl AutoLoss1D {
    pub fn new(num_threads: usize) -> Self {
        AutoLoss1D { num_workers: num_threads.max(1) }
    }
}

impl LossDispatch for AutoLoss1D {
    fn compute_loss(
        &self,
        pred: &dyn LossInput,
        target: &dyn LossInput,
        j_pred: &dyn LossJacobian,
        built_loss: &BuiltLoss,
    ) -> (f32, Vec<f32>) {
        let pred = pred.as_any().downcast_ref::<Tensor1D>()
            .expect("AutoLoss1D: pred must be Tensor1D");
        let target = target.as_any().downcast_ref::<Tensor1D>()
            .expect("AutoLoss1D: target must be Tensor1D");
        let j_pred = j_pred.as_any().downcast_ref::<Jacobian>()
            .expect("AutoLoss1D: j_pred must be Jacobian");

        let n = pred.len();
        let p = j_pred.cols();
        if n == 0 {
            return (0.0, vec![0.0; p]);
        }

        let num_threads = self.num_workers.min(n);
        let chunk_size = (n + num_threads - 1) / num_threads; // ceil

        let pred_data = Arc::new(pred.data.clone());
        let target_data = Arc::new(target.data.clone());
        let j_pred_data = Arc::new(j_pred.data.clone());
        let built_loss = Arc::clone(&built_loss.forward);

        let mut handles = Vec::with_capacity(num_threads);

        for tid in 0..num_threads {
            let start = tid * chunk_size;
            let end = (start + chunk_size).min(n);
            if start >= n { break; }
            let chunk_len = end - start;

            let pred_data = Arc::clone(&pred_data);
            let target_data = Arc::clone(&target_data);
            let j_pred_data = Arc::clone(&j_pred_data);
            let built_loss = Arc::clone(&built_loss);

            let handle = thread::spawn(move || {
                let pred_chunk = Tensor1D::new(pred_data[start..end].to_vec());
                let target_chunk = Tensor1D::new(target_data[start..end].to_vec());
                let mut j_chunk = Jacobian::new(chunk_len, p);
                for i in 0..chunk_len {
                    j_chunk.data[i] = j_pred_data[start + i].clone();
                }
                let (loss_chunk, grad_chunk) = built_loss(&pred_chunk, &target_chunk, &j_chunk);
                // Масштабируем обратно к сумме, чтобы потом правильно усреднить
                (loss_chunk * chunk_len as f32,
                 grad_chunk.iter().map(|g| g * chunk_len as f32).collect::<Vec<_>>())
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
        total_loss /= n as f32;
        for g in &mut total_grad { *g /= n as f32; }
        (total_loss, total_grad)
    }

    fn num_workers(&self) -> usize { self.num_workers }
}
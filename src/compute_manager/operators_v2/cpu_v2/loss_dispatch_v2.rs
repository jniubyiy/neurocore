// src/compute_manager/operators_v2/cpu_v2/loss_dispatch_v2.rs
//
// Диспетчер CPU-вычисления функции потерь.
//
// Использует существующий `compute_loss_mat_buffered` из старого
// `plans::loss_plan::execution`. Функция работает только с CPU-буферами;
// если pred или target — на GPU, вызывающий (распределитель) должен был
// отправить job в GpuOperatorV2.

use std::sync::{Arc, Mutex};

use crate::compute_manager::jobs_v2::{JobResult, LossJob};
use crate::compute_manager::operators_v2::memory_v2::buffer::TempMatrixPool;
use crate::loss_plan::compute_loss_mat_buffered;

/// Исполняет `LossJob` на CPU.
pub fn execute_loss(job: LossJob, pool: Arc<Mutex<TempMatrixPool>>) -> JobResult {
    if job.pred.is_gpu() || job.target.is_gpu() {
        return JobResult::Failed(
            "CpuOperatorV2::loss: pred or target is on GPU (expected CPU)".to_string(),
        );
    }

    let mut p = pool.lock().unwrap();
    let (value, grad_pred) =
        compute_loss_mat_buffered(&job.expr, &job.pred, &job.target, &mut *p);

    JobResult::Loss { value, grad_pred }
}
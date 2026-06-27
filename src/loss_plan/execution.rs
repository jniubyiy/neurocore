// src/loss_plan/execution.rs

use std::sync::{Arc, mpsc};

use faer::Mat;
use crate::compute_manager::cpu::{Scheduler, WorkerPool};
use super::expr::LossExpr;

pub fn compute_loss_mat(
    expr: &Arc<LossExpr>,
    pred: &Mat<f32>,
    target: &Mat<f32>,
    scheduler: &mut Scheduler,
    pool: &WorkerPool,
) -> (f32, Mat<f32>) {
    let pred_feat = expr.pred_features();
    let target_feat = expr.target_features();
    let in_features = pred_feat + target_feat;

    // Разворачиваем матрицы в плоские векторы поэлементно
    let flat_pred: Vec<f32> = (0..pred.nrows())
        .flat_map(|r| (0..pred_feat).map(move |c| pred[(r, c)]))
        .collect();
    let flat_target: Vec<f32> = (0..target.nrows())
        .flat_map(|r| (0..target_feat).map(move |c| target[(r, c)]))
        .collect();

    let total_tasks = flat_pred.len() / pred_feat;

    // Формируем матрицу задач, каждая строка: [pred_i, target_i]
    let full_input = Mat::from_fn(total_tasks, in_features, |i, j| {
        if j < pred_feat {
            flat_pred[i * pred_feat + j]
        } else {
            let t_idx = j - pred_feat;
            flat_target[i * target_feat + t_idx]
        }
    });

    let assignment = scheduler.plan_chunks_assignment(total_tasks);
    let full_input = Arc::new(full_input);
    let (tx, rx) = mpsc::channel();
    let mut tasks_sent = 0;

    for (_worker_id, ranges) in assignment.iter().enumerate() {
        if ranges.is_empty() { continue; }
        let ranges = ranges.clone();
        let full_input = Arc::clone(&full_input);
        let expr = Arc::clone(expr);
        let tx = tx.clone();
        pool.execute(move || {
            let mut loss_contrib = Vec::new();
            let mut grad_contrib = Vec::new();
            for (start, count) in &ranges {
                let chunk = full_input.submatrix(*start, 0, *count, in_features).to_owned();
                let (loss_vec, intermediates) = expr.forward_chunk(&chunk);
                let grad_loss = vec![1.0f32; *count];
                let grad_mat = expr.backward_chunk(&intermediates, &grad_loss);
                loss_contrib.extend_from_slice(&loss_vec);
                for i in 0..*count {
                    for j in 0..in_features {
                        grad_contrib.push(grad_mat[(i, j)]);
                    }
                }
            }
            tx.send((loss_contrib, grad_contrib)).ok();
        });
        tasks_sent += 1;
    }
    pool.wait_all();

    let mut all_loss = Vec::with_capacity(total_tasks);
    let mut all_grad = Vec::with_capacity(total_tasks * in_features);
    for _ in 0..tasks_sent {
        if let Ok((loss, grad)) = rx.recv() {
            all_loss.extend_from_slice(&loss);
            all_grad.extend_from_slice(&grad);
        }
    }

    let loss = expr.aggregate_loss(&all_loss);
    let grad_flat = expr.aggregate_grad(&all_grad);

    // Восстанавливаем градиент по pred в исходной матричной форме
    let mut grad_pred = Mat::zeros(pred.nrows(), pred.ncols());
    for i in 0..total_tasks {
        let start = i * in_features;
        let row = i / pred_feat;
        let col = i % pred_feat;
        grad_pred[(row, col)] = grad_flat[start]; // первый компонент градиента – по pred
    }

    (loss, grad_pred)
}
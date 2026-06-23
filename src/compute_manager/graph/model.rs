// src/compute_manager/graph/model.rs

use std::sync::{Arc, Mutex};

use crate::compute_manager::cpu::{Scheduler, WorkerPool};
use crate::compute_manager::cpu::scheduler::LayerInfo;
use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::graph::types::Segment;
use crate::loss_plan::LossExpr;
use crate::model_plan::param_store::ParamStore;
use crate::optimizer_plan::{OptimizerExpr, OptimizerChain};

pub struct MixedModel {
    pub(crate) segments: Vec<Segment>,
    pub(crate) store: Arc<Mutex<ParamStore>>,
    pub(crate) pool: Arc<WorkerPool>,
    pub(crate) scheduler: Mutex<Scheduler>,
    #[allow(dead_code)]
    pub(crate) layer_infos: Vec<Vec<LayerInfo>>,

    // Буферы для повторного использования в compute_loss_with_expr
    pub(crate) flat_pred_buf: Mutex<Vec<f32>>,
    pub(crate) flat_target_buf: Mutex<Vec<f32>>,
}

impl MixedModel {
    pub fn num_workers(&self) -> usize {
        self.scheduler.lock().unwrap().num_workers()
    }

    pub fn param_store(&self) -> &Arc<Mutex<ParamStore>> {
        &self.store
    }

    /// Создаёт OptimizerExpr, привязанный к текущему числу параметров модели.
    pub fn create_optimizer(&self, chain: OptimizerChain) -> OptimizerExpr {
        let num_params = self.store.lock().unwrap().len();
        OptimizerExpr::new(num_params, chain)
    }

    /// Применяет градиенты с помощью заданного оптимизатора.
    pub fn update_params_with_optimizer(&mut self, optimizer: &mut OptimizerExpr, grads: &[f32]) {
        let mut store = self.store.lock().unwrap();
        let mut params = store.all_params_vec();
        optimizer.step(&mut params, grads);
        store.set_all_params(&params);
    }

    pub fn compute_loss_with_expr(
        &self,
        expr: Arc<LossExpr>,
        pred: &DynamicTensor,
        target: &DynamicTensor,
    ) -> (f32, DynamicTensor) {
        let total_tasks = expr.num_tasks();
        let in_size = expr.task_input_size();
        let pred_feat = expr.pred_features();
        let target_feat = expr.target_features();

        let mut flat_pred = self.flat_pred_buf.lock().unwrap();
        pred.write_to_flat(&mut flat_pred);
        let mut flat_target = self.flat_target_buf.lock().unwrap();
        target.write_to_flat(&mut flat_target);

        // Универсальная сборка входного вектора для функции потерь
        let mut flat_input = Vec::with_capacity(total_tasks * in_size);
        for i in 0..total_tasks {
            let pred_start = i * pred_feat;
            let target_start = i * target_feat;
            flat_input.extend_from_slice(&flat_pred[pred_start..pred_start + pred_feat]);
            flat_input.extend_from_slice(&flat_target[target_start..target_start + target_feat]);
        }

        let mut scheduler = self.scheduler.lock().unwrap();
        let assignment = scheduler.plan_chunks_assignment(total_tasks);
        drop(scheduler);

        let (tx, rx) = std::sync::mpsc::channel();
        let flat_input = Arc::new(flat_input);
        let mut tasks_sent = 0;

        for (_worker_id, ranges) in assignment.iter().enumerate() {
            if ranges.is_empty() {
                continue;
            }
            let ranges = ranges.clone();
            let flat_input = Arc::clone(&flat_input);
            let expr = Arc::clone(&expr);
            let tx = tx.clone();

            self.pool.execute(move || {
                let mut loss_contrib = Vec::new();
                let mut grad_contrib = Vec::new();

                for (start, count) in &ranges {
                    let start = *start;
                    let count = *count;
                    let chunk_inputs = &flat_input[start * in_size..(start + count) * in_size];
                    let mut out_loss = vec![0.0; count];
                    let mut grad_pred = vec![0.0; count * in_size];

                    expr.forward_chunk(start, count, chunk_inputs, &mut out_loss);
                    let grad_loss = vec![1.0; count];
                    expr.backward_chunk(
                        start,
                        count,
                        chunk_inputs,
                        &out_loss,
                        &grad_loss,
                        &mut grad_pred,
                    );

                    loss_contrib.extend_from_slice(&out_loss);
                    grad_contrib.extend_from_slice(&grad_pred);
                }
                tx.send((loss_contrib, grad_contrib)).ok();
            });
            tasks_sent += 1;
        }

        self.pool.wait_all();

        let mut all_loss = Vec::with_capacity(total_tasks);
        let mut all_grad = Vec::with_capacity(total_tasks * in_size);
        for _ in 0..tasks_sent {
            if let Ok((loss, grad)) = rx.recv() {
                all_loss.extend_from_slice(&loss);
                all_grad.extend_from_slice(&grad);
            }
        }

        let loss = expr.aggregate_loss(&all_loss);
        let grad_flat = expr.aggregate_grad(&all_grad);

        // Извлекаем градиент только для pred-части (первые pred_feat элементов каждого таска)
        let mut grad_only = Vec::with_capacity(total_tasks * pred_feat);
        for i in 0..total_tasks {
            let start = i * in_size;
            grad_only.extend_from_slice(&grad_flat[start..start + pred_feat]);
        }
        let grad_dyn = DynamicTensor::from_flat(pred, grad_only);

        (loss, grad_dyn)
    }
}
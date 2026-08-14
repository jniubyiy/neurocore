// src/plans/loss_plan/expr.rs

use std::sync::Arc;

use super::chain::ElementChain;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};

/// Способ агрегирования значений потерь по задачам.
#[derive(Debug, Clone, Copy)]
pub enum Aggregation {
    /// Суммировать значения потерь по всем задачам.
    Sum,
    /// Усреднить значения потерь (разделить на количество задач).
    Mean,
}

/// Выражение функции потерь, построенное на цепочке элементарных кубиков.
///
/// Принимает матрицы размера `(batch, pred_features + target_features)`,
/// где `batch` — количество сэмплов в батче (значение `total_tasks`).
pub struct LossExpr {
    chain: Arc<ElementChain>,
    aggregation: Aggregation,
    /// Количество сэмплов в батче (размер `batch`)
    total_tasks: usize,
    /// Число признаков предсказания на один сэмпл
    pred_features: usize,
    /// Число признаков целевой переменной на один сэмпл
    target_features: usize,
}

impl LossExpr {
    /// Создаёт новое выражение потерь.
    pub fn new(
        chain: Arc<ElementChain>,
        aggregation: Aggregation,
        total_tasks: usize,
        pred_features: usize,
        target_features: usize,
    ) -> Self {
        Self {
            chain,
            aggregation,
            total_tasks,
            pred_features,
            target_features,
        }
    }

    /// Количество сэмплов в батче.
    pub fn num_tasks(&self) -> usize {
        self.total_tasks
    }

    /// Размер входной матрицы для одного сэмпла.
    pub fn task_input_size(&self) -> usize {
        self.pred_features + self.target_features
    }

    /// Количество признаков предсказания.
    pub fn pred_features(&self) -> usize {
        self.pred_features
    }

    /// Количество признаков целевой переменной.
    pub fn target_features(&self) -> usize {
        self.target_features
    }

    /// Ссылка на цепочку кубиков.
    pub fn chain(&self) -> &ElementChain {
        &self.chain
    }

    /// Тип агрегации.
    pub fn aggregation(&self) -> Aggregation {
        self.aggregation
    }

    /// Агрегирует значения потерь.
    pub fn aggregate_loss(&self, loss_parts: &[f32]) -> f32 {
        let sum: f32 = loss_parts.iter().sum();
        let n = self.total_tasks as f32;
        match self.aggregation {
            Aggregation::Sum => sum,
            Aggregation::Mean => sum / n,
        }
    }

    /// Агрегирует градиенты.
    pub fn aggregate_grad(&self, grad_parts: &[f32]) -> Vec<f32> {
        let n = self.total_tasks as f32;
        match self.aggregation {
            Aggregation::Sum => grad_parts.to_vec(),
            Aggregation::Mean => grad_parts.iter().map(|g| g / n).collect(),
        }
    }

    // ===================================================================
    // БУФЕРИЗОВАННЫЕ МЕТОДЫ (MatrixBufferHandle + TempMatrixPool)
    // ===================================================================

    /// Прямой проход с управляемыми дескрипторами.
    pub fn forward_chunk_buffered(
        &self,
        chunk_input: &MatrixBufferHandle,
        pool: &mut TempMatrixPool,
    ) -> (Vec<f32>, Vec<(MatrixBufferHandle, MatrixBufferHandle)>) {
        let (out_mat, intermediates) = self.chain.forward_batch_buffered(chunk_input, pool);
        let batch = out_mat.rows();

        // Извлекаем значения потерь (out_mat имеет размер (batch, 1))
        let out_guard = out_mat.read();
        let out_slice = out_guard.as_slice().expect("Loss output must be CPU");
        let loss_vec: Vec<f32> = (0..batch).map(|i| out_slice[i]).collect();
        drop(out_guard);

        // Финальный выход не нужен после извлечения loss – возвращаем в пул
        pool.release(out_mat);

        (loss_vec, intermediates)
    }

    /// Обратный проход с управляемыми дескрипторами.
    pub fn backward_chunk_buffered(
        &self,
        intermediates: &[(MatrixBufferHandle, MatrixBufferHandle)],
        grad_loss: &[f32],
        pool: &mut TempMatrixPool,
    ) -> MatrixBufferHandle {
        let batch = intermediates.first()
            .map(|(inp, _)| inp.rows())
            .unwrap_or(0);
        assert_eq!(batch, grad_loss.len(),
            "backward_chunk_buffered: длина grad_loss должна совпадать с размером батча");

        // Создаём буфер градиента по выходу цепочки (batch, 1)
        let grad_out = pool.acquire(batch, 1);
        {
            let mut grad_out_guard = grad_out.write();
            let grad_out_slice = grad_out_guard.as_slice_mut().expect("Grad out must be CPU");
            for i in 0..batch {
                grad_out_slice[i] = grad_loss[i];
            }
        }

        let grad_in = self.chain.backward_batch_buffered(intermediates, &grad_out, pool);

        // Возвращаем временный grad_out в пул
        pool.release(grad_out);

        grad_in
    }
}
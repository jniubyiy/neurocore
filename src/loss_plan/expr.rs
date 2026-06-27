// src/loss_plan/expr.rs

use faer::Mat;
use super::chain::ElementChain;

/// Способ агрегирования значений потерь по задачам.
pub enum Aggregation {
    /// Суммировать значения потерь по всем задачам.
    Sum,
    /// Усреднить значения потерь (разделить на количество задач).
    Mean,
}

/// Выражение функции потерь, построенное на цепочке элементарных кубиков.
pub struct LossExpr {
    chain: ElementChain,
    aggregation: Aggregation,
    total_tasks: usize,
    pred_features: usize,
    target_features: usize,
}

impl LossExpr {
    /// Создаёт новое выражение потерь.
    ///
    /// # Аргументы
    /// * `chain` — цепочка кубиков, преобразующая вход в значение потерь.
    /// * `aggregation` — способ агрегирования потерь по отдельным задачам.
    /// * `total_tasks` — общее количество задач (например, элементов в батче).
    /// * `pred_features` — количество признаков в предсказании на одну задачу.
    /// * `target_features` — количество признаков в целевой переменной на одну задачу.
    pub fn new(
        chain: ElementChain,
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

    /// Количество задач.
    pub fn num_tasks(&self) -> usize {
        self.total_tasks
    }

    /// Размер входной матрицы для одной задачи (число столбцов).
    pub fn task_input_size(&self) -> usize {
        self.chain.task_input_size()
    }

    /// Количество признаков предсказания на задачу.
    pub fn pred_features(&self) -> usize {
        self.pred_features
    }

    /// Количество признаков целевой переменной на задачу.
    pub fn target_features(&self) -> usize {
        self.target_features
    }

    /// Выполняет прямой проход для чанка задач.
    ///
    /// * `chunk_input` — матрица размера `(chunk_size, task_input_size())`,
    ///   где каждая строка содержит признаки предсказания и целевой переменной.
    ///
    /// Возвращает кортеж:
    /// * вектор значений потерь длиной `chunk_size`,
    /// * вектор промежуточных результатов для каждого кубика
    ///   (пары `(вход_кубика, выход_кубика)`) — необходим для обратного прохода.
    pub fn forward_chunk(
        &self,
        chunk_input: &Mat<f32>,
    ) -> (Vec<f32>, Vec<(Mat<f32>, Mat<f32>)>) {
        let (out_mat, intermediates) = self.chain.forward_batch(chunk_input);
        // out_mat имеет размер (chunk_size, 1) – так как последний кубик должен иметь out_features = 1
        let loss_vec: Vec<f32> = (0..out_mat.nrows())
            .map(|i| out_mat[(i, 0)])
            .collect();
        (loss_vec, intermediates)
    }

    /// Выполняет обратный проход для чанка задач.
    ///
    /// * `intermediates` — кэш, полученный из `forward_chunk`.
    /// * `grad_loss` — градиент по значениям потерь (обычно единицы), длина `chunk_size`.
    ///
    /// Возвращает матрицу градиентов по входу размером `(chunk_size, task_input_size())`.
    pub fn backward_chunk(
        &self,
        intermediates: &[(Mat<f32>, Mat<f32>)],
        grad_loss: &[f32],
    ) -> Mat<f32> {
        let batch = intermediates.first()
            .map(|(inp, _)| inp.nrows())
            .unwrap_or(0);
        assert_eq!(batch, grad_loss.len(),
            "backward_chunk: длина grad_loss должна совпадать с размером батча");

        // Превращаем вектор градиентов в матрицу-столбец (batch, 1)
        let grad_out = Mat::from_fn(batch, 1, |i, _| grad_loss[i]);
        self.chain.backward_batch(intermediates, &grad_out)
    }

    /// Вычисляет итоговое значение потерь путём агрегации значений по отдельным задачам.
    pub fn aggregate_loss(&self, loss_parts: &[f32]) -> f32 {
        let sum: f32 = loss_parts.iter().sum();
        let n = self.total_tasks as f32;
        match self.aggregation {
            Aggregation::Sum => sum,
            Aggregation::Mean => sum / n,
        }
    }

    /// Вычисляет агрегированный градиент по входным данным.
    ///
    /// Принимает плоский вектор `grad_parts`, где для каждой задачи идут градиенты
    /// по её входным признакам (сначала pred_features, затем target_features).
    /// Возвращает такой же плоский вектор после применения агрегации.
    pub fn aggregate_grad(&self, grad_parts: &[f32]) -> Vec<f32> {
        let n = self.total_tasks as f32;
        match self.aggregation {
            Aggregation::Sum => grad_parts.to_vec(),
            Aggregation::Mean => grad_parts.iter().map(|g| g / n).collect(),
        }
    }
}
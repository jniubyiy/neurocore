// src/plans/loss_plan/expr.rs

use faer::Mat;
use std::sync::Arc;
use super::chain::ElementChain;

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
    ///
    /// # Аргументы
    /// * `chain` – цепочка элементарных кубиков (первый кубик должен принимать
    ///   `pred_features + target_features` столбцов).
    /// * `aggregation` – способ агрегирования.
    /// * `total_tasks` – размер батча (`batch`).
    /// * `pred_features` – количество выходных признаков модели на один сэмпл.
    /// * `target_features` – количество целевых признаков на один сэмпл
    ///   (обычно равно `pred_features`).
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

    /// Количество сэмплов в батче (размер `batch`).
    pub fn num_tasks(&self) -> usize {
        self.total_tasks
    }

    /// Размер входной матрицы для одного сэмпла (число столбцов).
    /// Равно `pred_features + target_features`.
    pub fn task_input_size(&self) -> usize {
        self.pred_features + self.target_features
    }

    /// Количество признаков предсказания на сэмпл.
    pub fn pred_features(&self) -> usize {
        self.pred_features
    }

    /// Количество признаков целевой переменной на сэмпл.
    pub fn target_features(&self) -> usize {
        self.target_features
    }

    /// Получить ссылку на внутреннюю цепочку кубиков.
    pub fn chain(&self) -> &ElementChain {
        &self.chain
    }

    /// Возвращает тип агрегации потерь.
    pub fn aggregation(&self) -> Aggregation {
        self.aggregation
    }

    /// Выполняет прямой проход для всего батча.
    ///
    /// # Аргументы
    /// * `chunk_input` – матрица размера `(batch, pred_features + target_features)`,
    ///   где каждая строка содержит признаки предсказания и цели для одного сэмпла.
    ///
    /// # Возвращает
    /// * вектор значений потерь (длина равна `batch`),
    /// * вектор промежуточных результатов кубиков для обратного прохода.
    pub fn forward_chunk(
        &self,
        chunk_input: &Mat<f32>,
    ) -> (Vec<f32>, Vec<(Mat<f32>, Mat<f32>)>) {
        let (out_mat, intermediates) = self.chain.forward_batch(chunk_input);
        let loss_vec: Vec<f32> = (0..out_mat.nrows())
            .map(|i| out_mat[(i, 0)])
            .collect();
        (loss_vec, intermediates)
    }

    /// Выполняет обратный проход для всего батча.
    ///
    /// # Аргументы
    /// * `intermediates` – промежуточные результаты прямого прохода.
    /// * `grad_loss` – градиент по значениям потерь (вектор длины `batch`).
    ///
    /// # Возвращает
    /// матрицу градиентов по входным данным размера `(batch, pred_features + target_features)`.
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

        let grad_out = Mat::from_fn(batch, 1, |i, _| grad_loss[i]);
        self.chain.backward_batch(intermediates, &grad_out)
    }

    /// Вычисляет итоговое значение потерь путём агрегации значений по отдельным сэмплам.
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
    /// `grad_parts` – плоский вектор, содержащий все элементы матрицы градиентов
    /// (полученной из `backward_chunk`) в row-major порядке.
    pub fn aggregate_grad(&self, grad_parts: &[f32]) -> Vec<f32> {
        let n = self.total_tasks as f32;
        match self.aggregation {
            Aggregation::Sum => grad_parts.to_vec(),
            Aggregation::Mean => grad_parts.iter().map(|g| g / n).collect(),
        }
    }
}
// src/loss_plan/chain.rs

use faer::Mat;
use super::cubes::ElemCube;

/// Цепочка элементарных кубиков, выполняющая последовательное преобразование над батчем.
pub struct ElementChain {
    cubes: Vec<Box<dyn ElemCube>>,
}

impl ElementChain {
    /// Создаёт пустую цепочку.
    pub fn new() -> Self {
        ElementChain { cubes: Vec::new() }
    }

    /// Добавляет кубик в конец цепочки.
    pub fn add(mut self, cube: Box<dyn ElemCube>) -> Self {
        self.cubes.push(cube);
        self
    }

    /// Возвращает размер входной матрицы цепочки (число столбцов первого кубика).
    /// Если цепочка пуста, возвращает 0.
    pub fn task_input_size(&self) -> usize {
        self.cubes.first().map(|c| c.in_features()).unwrap_or(0)
    }

    /// Возвращает ссылку на срез кубиков (например, для отладки).
    pub fn cubes(&self) -> &[Box<dyn ElemCube>] {
        &self.cubes
    }

    /// Выполняет полный прямой проход по всей цепочке над одним батчем.
    ///
    /// Принимает матрицу `input` размером `(batch, task_input_size())`.
    /// Возвращает кортеж:
    /// * итоговая матрица `(batch, out_features последнего кубика)`,
    /// * вектор промежуточных результатов в формате `(вход_кубика, выход_кубика)` для каждого кубика.
    ///   Это необходимо для последующего обратного прохода.
    pub fn forward_batch(&self, input: &Mat<f32>) -> (Mat<f32>, Vec<(Mat<f32>, Mat<f32>)>) {
        let mut intermediates = Vec::with_capacity(self.cubes.len());
        let mut current = input.clone();
        for cube in &self.cubes {
            let out = cube.forward_batch(&current);
            intermediates.push((current.clone(), out.clone()));
            current = out;
        }
        (current, intermediates)
    }

    /// Выполняет обратный проход по всей цепочке, используя сохранённые промежуточные значения.
    ///
    /// * `intermediates` — результат `forward_batch` (вектор пар (вход, выход) для каждого кубика),
    /// * `grad_out` — градиент по выходу цепочки, матрица `(batch, out_features последнего кубика)`.
    ///
    /// Возвращает градиент по входу цепочки — матрица `(batch, task_input_size())`.
    pub fn backward_batch(
        &self,
        intermediates: &[(Mat<f32>, Mat<f32>)],
        grad_out: &Mat<f32>,
    ) -> Mat<f32> {
        assert_eq!(intermediates.len(), self.cubes.len(),
            "ElementChain::backward_batch: количество промежуточных результатов не совпадает с числом кубиков");

        let mut grad = grad_out.clone();
        // Идём по кубикам в обратном порядке
        for (cube, (inp, outp)) in self.cubes.iter().zip(intermediates.iter()).rev() {
            grad = cube.backward_batch(inp, outp, &grad);
        }
        grad
    }
}
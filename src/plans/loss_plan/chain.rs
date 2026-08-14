// src/plans/loss_plan/chain.rs

use faer::Mat;
use super::cubes::ElemCube;
use super::cubes::BufferedElemCube;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};

/// Цепочка элементарных кубиков, выполняющая последовательное преобразование над батчем.
#[derive(Debug)]
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

    /// Выполняет полный прямой проход по всей цепочке над одним батчем (матричная версия).
    ///
    /// Принимает матрицу `input` размером `(batch, task_input_size())`.
    /// Возвращает кортеж:
    /// * итоговая матрица `(batch, out_features последнего кубика)`,
    /// * вектор промежуточных результатов в формате `(вход_кубика, выход_кубика)` для каждого кубика.
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

    /// Выполняет обратный проход по всей цепочке, используя сохранённые промежуточные значения (матричная версия).
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

    // ===================================================================
    // БУФЕРИЗОВАННЫЕ МЕТОДЫ (MatrixBufferHandle + TempMatrixPool)
    // ===================================================================

    /// Прямой проход по цепочке с использованием управляемых дескрипторов.
    ///
    /// Принимает входной `MatrixBufferHandle` (CPU) и пул для выделения промежуточных буферов.
    /// Возвращает итоговый буфер и вектор пар `(вход, выход)` для обратного прохода.
    /// Все промежуточные данные выделяются через пул и управляются `MemoryExecutor`.
    pub fn forward_batch_buffered(
        &self,
        input: &MatrixBufferHandle,
        pool: &mut TempMatrixPool,
    ) -> (MatrixBufferHandle, Vec<(MatrixBufferHandle, MatrixBufferHandle)>) {
        let batch = input.rows();
        let mut intermediates = Vec::with_capacity(self.cubes.len());

        // Начальный буфер — копия входа (чтобы сохранить данные для обратного прохода)
        let mut current = clone_handle_data(pool, input);

        for cube in &self.cubes {
            // Определяем размеры выходного буфера.
            let out_rows = current.rows();
            let out_cols = if cube_preserves_shape(cube.as_ref()) {
                current.cols()
            } else {
                cube.out_features()
            };
            let mut out = pool.acquire(out_rows, out_cols);

            // Вызываем буферизованный прямой проход для конкретного кубика
            let buffered_cube = cube_as_buffered(cube);
            buffered_cube.forward_buffered(&current, &mut out);

            // Сохраняем копии входа и выхода для обратного прохода.
            // Важно: создаём новые буферы с копией данных, так как `current` и `out`
            // могут быть изменены в следующих итерациях.
            intermediates.push((
                clone_handle_data(pool, &current),
                clone_handle_data(pool, &out),
            ));

            current = out;
        }

        (current, intermediates)
    }

    /// Обратный проход по цепочке с использованием управляемых дескрипторов.
    ///
    /// Принимает промежуточные результаты (из `forward_batch_buffered`),
    /// градиент по выходу цепочки и пул.
    /// Возвращает градиент по входу цепочки.
    pub fn backward_batch_buffered(
        &self,
        intermediates: &[(MatrixBufferHandle, MatrixBufferHandle)],
        grad_out: &MatrixBufferHandle,
        pool: &mut TempMatrixPool,
    ) -> MatrixBufferHandle {
        assert_eq!(intermediates.len(), self.cubes.len(),
            "ElementChain::backward_batch_buffered: количество промежуточных результатов не совпадает с числом кубиков");

        // Начальный градиент — копия grad_out (чтобы не изменять оригинал)
        let mut grad = clone_handle_data(pool, grad_out);

        // Идём по кубикам в обратном порядке
        for (cube, (inp, outp)) in self.cubes.iter().zip(intermediates.iter()).rev() {
            // Входной градиент имеет размерность входа кубика
            let mut grad_in = pool.acquire(inp.rows(), inp.cols());

            let buffered_cube = cube_as_buffered(cube);
            buffered_cube.backward_buffered(inp, outp, &grad, &mut grad_in);

            // Освобождаем предыдущий градиент (возвращаем в пул)
            pool.release(grad);
            grad = grad_in;
        }

        grad
    }
}

// ---------------------------------------------------------------------------
// Вспомогательные функции
// ---------------------------------------------------------------------------

/// Создаёт копию данных из исходного handle через пул.
/// Работает только для CPU-буферов (для GPU не реализовано).
fn clone_handle_data(pool: &mut TempMatrixPool, src: &MatrixBufferHandle) -> MatrixBufferHandle {
    assert!(!src.is_gpu(), "clone_handle_data does not support GPU buffers");
    let rows = src.rows();
    let cols = src.cols();
    let mut copy = pool.acquire(rows, cols);
    {
        let src_guard = src.read();
        let src_slice = src_guard.as_slice().expect("Source must be CPU");
        let mut dst_guard = copy.write();
        let dst_slice = dst_guard.as_slice_mut().expect("Destination must be CPU");
        assert_eq!(src_slice.len(), dst_slice.len());
        dst_slice.copy_from_slice(src_slice);
    }
    copy
}

/// Преобразует `&Box<dyn ElemCube>` в `&dyn BufferedElemCube`.
/// Использует downcasting к конкретным типам, которые реализуют оба трейта.
fn cube_as_buffered(cube: &Box<dyn ElemCube>) -> &dyn BufferedElemCube {
    use super::cubes::{Sub, Square, SumColumns, Log, Neg, Mul, Abs, AddScalar, Log1p, AbsDiff};
    use super::cross_entropy::CrossEntropyWithLogits;

    if let Some(c) = cube.as_any().downcast_ref::<Sub>() { return c; }
    if let Some(c) = cube.as_any().downcast_ref::<Square>() { return c; }
    if let Some(c) = cube.as_any().downcast_ref::<SumColumns>() { return c; }
    if let Some(c) = cube.as_any().downcast_ref::<Log>() { return c; }
    if let Some(c) = cube.as_any().downcast_ref::<Neg>() { return c; }
    if let Some(c) = cube.as_any().downcast_ref::<Mul>() { return c; }
    if let Some(c) = cube.as_any().downcast_ref::<Abs>() { return c; }
    if let Some(c) = cube.as_any().downcast_ref::<AddScalar>() { return c; }
    if let Some(c) = cube.as_any().downcast_ref::<Log1p>() { return c; }
    if let Some(c) = cube.as_any().downcast_ref::<AbsDiff>() { return c; }
    if let Some(c) = cube.as_any().downcast_ref::<CrossEntropyWithLogits>() { return c; }

    panic!("Cube does not implement BufferedElemCube: {:?}", std::any::type_name_of_val(cube.as_ref()))
}

/// Возвращает `true`, если кубик сохраняет размерность (поэлементные операции).
fn cube_preserves_shape(cube: &dyn ElemCube) -> bool {
    use super::cubes::{Square, Abs, Neg, Log, Log1p, AddScalar};
    cube.as_any().downcast_ref::<Square>().is_some()
        || cube.as_any().downcast_ref::<Abs>().is_some()
        || cube.as_any().downcast_ref::<Neg>().is_some()
        || cube.as_any().downcast_ref::<Log>().is_some()
        || cube.as_any().downcast_ref::<Log1p>().is_some()
        || cube.as_any().downcast_ref::<AddScalar>().is_some()
}
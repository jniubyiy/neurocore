// src/loss_plan/gpu_exec.rs

use faer::Mat;
use crate::compute_manager::gpu::compute::GpuCompute;
use crate::loss_plan::CrossEntropyWithLogits;
use super::cubes::*;
use super::expr::LossExpr;

/// Выполняет вычисление потерь и градиентов на GPU.
pub fn compute_loss_gpu(
    gpu: &GpuCompute,
    expr: &LossExpr,
    pred: &Mat<f32>,
    target: &Mat<f32>,
) -> (f32, Mat<f32>) {
    let chain = expr.chain();
    let pred_feat = expr.pred_features();
    let target_feat = expr.target_features();
    let in_features = pred_feat + target_feat;
    let total_tasks = expr.num_tasks();   // <-- берём из выражения, а не pred.nrows()

    // Разворачиваем pred и target в плоские векторы по элементам
    let flat_pred: Vec<f32> = (0..pred.nrows())
        .flat_map(|r| (0..pred_feat).map(move |c| pred[(r, c)]))
        .collect();
    let flat_target: Vec<f32> = (0..target.nrows())
        .flat_map(|r| (0..target_feat).map(move |c| target[(r, c)]))
        .collect();

    // Формируем матрицу задач: каждая строка – [pred_i, target_i]
    let full_input = Mat::from_fn(total_tasks, in_features, |i, j| {
        if j < pred_feat {
            flat_pred[i * pred_feat + j]
        } else {
            let t_idx = j - pred_feat;
            flat_target[i * target_feat + t_idx]
        }
    });

    // Прямой проход цепочки кубиков на GPU
    let mut current = full_input.clone();
    let mut intermediates: Vec<(Mat<f32>, Mat<f32>)> = Vec::with_capacity(chain.cubes().len());

    for cube in chain.cubes() {
        let input_for_cube = current.clone();
        let output = run_cube_forward_gpu(cube.as_ref(), gpu, &current);
        intermediates.push((input_for_cube, output.clone()));
        current = output;
    }

    // current имеет размер (batch, 1) — значения потерь
    let loss_vec: Vec<f32> = (0..total_tasks).map(|i| current[(i, 0)]).collect();
    let loss = expr.aggregate_loss(&loss_vec);

    // Подготавливаем градиент по потерям с учётом агрегации
    let grad_scale = match expr.aggregation() {
        super::expr::Aggregation::Sum => 1.0f32,
        super::expr::Aggregation::Mean => 1.0f32 / total_tasks as f32,
    };
    let grad_loss_mat = Mat::from_fn(total_tasks, 1, |_i, _j| grad_scale);

    // Обратный проход цепочки на GPU
    let mut grad = grad_loss_mat;
    for (cube, (inp, _outp)) in chain.cubes().iter().zip(intermediates.iter()).rev() {
        grad = run_cube_backward_gpu(cube.as_ref(), gpu, inp, &grad);
    }

    // grad имеет размер (total_tasks, in_features).
    // Восстанавливаем градиент по pred в исходной матричной форме (pred.nrows(), pred.ncols())
    let mut grad_pred = Mat::zeros(pred.nrows(), pred.ncols());
    for i in 0..total_tasks {
        let start = i * in_features;
        let row = i / pred_feat;      // pred_feat может быть меньше pred.ncols(), но для данной задачи это 1
        let col = i % pred_feat;
        grad_pred[(row, col)] = grad[(i, 0)];   // первый компонент градиента – по pred
    }

    (loss, grad_pred)
}

/// Запуск прямого прохода одного кубика на GPU.
fn run_cube_forward_gpu(cube: &dyn ElemCube, gpu: &GpuCompute, input: &Mat<f32>) -> Mat<f32> {
    if let Some(_) = cube.as_any().downcast_ref::<Sub>() {
        let a_col = input.subcols(0, 1).to_owned();
        let b_col = input.subcols(1, 1).to_owned();
        return gpu.run_sub_forward(&a_col, &b_col);
    } else if let Some(_) = cube.as_any().downcast_ref::<Square>() {
        return gpu.run_square_forward(input);
    } else if let Some(_) = cube.as_any().downcast_ref::<Abs>() {
        return gpu.run_abs_forward(input);
    } else if let Some(_) = cube.as_any().downcast_ref::<Log1p>() {
        return gpu.run_log1p_forward(input);
    } else if let Some(_) = cube.as_any().downcast_ref::<AbsDiff>() {
        let a_col = input.subcols(0, 1).to_owned();
        let b_col = input.subcols(1, 1).to_owned();
        return gpu.run_absdiff_forward(&a_col, &b_col);
    } else if let Some(_) = cube.as_any().downcast_ref::<Log>() {
        return gpu.run_log_forward(input);
    } else if let Some(_) = cube.as_any().downcast_ref::<Neg>() {
        return gpu.run_neg_forward(input);
    } else if let Some(_) = cube.as_any().downcast_ref::<Mul>() {
        let a_col = input.subcols(0, 1).to_owned();
        let b_col = input.subcols(1, 1).to_owned();
        return gpu.run_mul_forward(&a_col, &b_col);
    } else if let Some(addscalar) = cube.as_any().downcast_ref::<AddScalar>() {
        return gpu.run_addscalar_forward(input, addscalar.0);
    } else if let Some(ce) = cube.as_any().downcast_ref::<CrossEntropyWithLogits>() {
        return gpu.run_cross_entropy_forward(input, ce.num_classes);
    }
    panic!("Unknown loss cube for GPU forward");
}

/// Запуск обратного прохода одного кубика на GPU.
fn run_cube_backward_gpu(
    cube: &dyn ElemCube,
    gpu: &GpuCompute,
    input: &Mat<f32>,
    grad_out: &Mat<f32>,
) -> Mat<f32> {
    if let Some(_) = cube.as_any().downcast_ref::<Sub>() {
        let (ga, gb) = gpu.run_sub_backward(grad_out);
        let batch = input.nrows();
        let mut result = Mat::zeros(batch, 2);
        for i in 0..batch {
            result[(i, 0)] = ga[(i, 0)];
            result[(i, 1)] = gb[(i, 0)];
        }
        return result;
    } else if let Some(_) = cube.as_any().downcast_ref::<Square>() {
        return gpu.run_square_backward(input, grad_out);
    } else if let Some(_) = cube.as_any().downcast_ref::<Abs>() {
        return gpu.run_abs_backward(input, grad_out);
    } else if let Some(_) = cube.as_any().downcast_ref::<Log1p>() {
        return gpu.run_log1p_backward(input, grad_out);
    } else if let Some(_) = cube.as_any().downcast_ref::<AbsDiff>() {
        let (ga, gb) = gpu.run_absdiff_backward(
            &input.subcols(0, 1).to_owned(),
            &input.subcols(1, 1).to_owned(),
            grad_out,
        );
        let batch = input.nrows();
        let mut result = Mat::zeros(batch, 2);
        for i in 0..batch {
            result[(i, 0)] = ga[(i, 0)];
            result[(i, 1)] = gb[(i, 0)];
        }
        return result;
    } else if let Some(_) = cube.as_any().downcast_ref::<Log>() {
        return gpu.run_log_backward(input, grad_out);
    } else if let Some(_) = cube.as_any().downcast_ref::<Neg>() {
        return gpu.run_neg_backward(grad_out);
    } else if let Some(_) = cube.as_any().downcast_ref::<Mul>() {
        let (ga, gb) = gpu.run_mul_backward(
            &input.subcols(0, 1).to_owned(),
            &input.subcols(1, 1).to_owned(),
            grad_out,
        );
        let batch = input.nrows();
        let mut result = Mat::zeros(batch, 2);
        for i in 0..batch {
            result[(i, 0)] = ga[(i, 0)];
            result[(i, 1)] = gb[(i, 0)];
        }
        return result;
    } else if let Some(_) = cube.as_any().downcast_ref::<AddScalar>() {
        return gpu.run_addscalar_backward(grad_out);
    } else if let Some(ce) = cube.as_any().downcast_ref::<CrossEntropyWithLogits>() {
        return gpu.run_cross_entropy_backward(input, grad_out, ce.num_classes);
    }
    panic!("Unknown loss cube for GPU backward");
}
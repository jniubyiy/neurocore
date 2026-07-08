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
    let total_tasks = pred.nrows();

    // Строим единую входную матрицу (batch, in_features), объединяя pred и target
    let mut full_input = Mat::zeros(total_tasks, in_features);
    for i in 0..total_tasks {
        for j in 0..pred_feat {
            full_input[(i, j)] = pred[(i, j)];
        }
        for j in 0..target_feat {
            full_input[(i, pred_feat + j)] = target[(i, j)];
        }
    }

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

    // grad имеет размер (batch, in_features). Берем первые pred_feat столбцов как градиент по pred
    let mut grad_pred = Mat::zeros(total_tasks, pred_feat);
    for i in 0..total_tasks {
        for j in 0..pred_feat {
            grad_pred[(i, j)] = grad[(i, j)];
        }
    }

    (loss, grad_pred)
}

/// Запуск прямого прохода одного кубика на GPU.
fn run_cube_forward_gpu(cube: &dyn ElemCube, gpu: &GpuCompute, input: &Mat<f32>) -> Mat<f32> {
    let a = cube.as_any();
    if a.is::<Sub>() {
        let a_col = input.subcols(0, 1).to_owned();
        let b_col = input.subcols(1, 1).to_owned();
        return gpu.run_sub_forward(&a_col, &b_col);
    } else if a.is::<Square>() {
        return gpu.run_square_forward(input);
    } else if a.is::<Abs>() {
        return gpu.run_abs_forward(input);
    } else if a.is::<Log1p>() {
        return gpu.run_log1p_forward(input);
    } else if a.is::<AbsDiff>() {
        let a_col = input.subcols(0, 1).to_owned();
        let b_col = input.subcols(1, 1).to_owned();
        return gpu.run_absdiff_forward(&a_col, &b_col);
    } else if a.is::<Log>() {
        return gpu.run_log_forward(input);
    } else if a.is::<Neg>() {
        return gpu.run_neg_forward(input);
    } else if a.is::<Mul>() {
        let a_col = input.subcols(0, 1).to_owned();
        let b_col = input.subcols(1, 1).to_owned();
        return gpu.run_mul_forward(&a_col, &b_col);
    } else if a.is::<AddScalar>() {
        let addscalar = a.downcast_ref::<AddScalar>().unwrap();
        return gpu.run_addscalar_forward(input, addscalar.0);
    } else if a.is::<CrossEntropyWithLogits>() {
        let ce = a.downcast_ref::<CrossEntropyWithLogits>().unwrap();
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
    let a = cube.as_any();
    if a.is::<Sub>() {
        let (ga, gb) = gpu.run_sub_backward(grad_out);
        let batch = input.nrows();
        let mut result = Mat::zeros(batch, 2);
        for i in 0..batch {
            result[(i, 0)] = ga[(i, 0)];
            result[(i, 1)] = gb[(i, 0)];
        }
        return result;
    } else if a.is::<Square>() {
        return gpu.run_square_backward(input, grad_out);
    } else if a.is::<Abs>() {
        return gpu.run_abs_backward(input, grad_out);
    } else if a.is::<Log1p>() {
        return gpu.run_log1p_backward(input, grad_out);
    } else if a.is::<AbsDiff>() {
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
    } else if a.is::<Log>() {
        return gpu.run_log_backward(input, grad_out);
    } else if a.is::<Neg>() {
        return gpu.run_neg_backward(grad_out);
    } else if a.is::<Mul>() {
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
    } else if a.is::<AddScalar>() {
        return gpu.run_addscalar_backward(grad_out);
    } else if a.is::<CrossEntropyWithLogits>() {
        let ce = a.downcast_ref::<CrossEntropyWithLogits>().unwrap();
        return gpu.run_cross_entropy_backward(input, grad_out, ce.num_classes);
    }
    panic!("Unknown loss cube for GPU backward");
}
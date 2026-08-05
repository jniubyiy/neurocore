// src/plans/loss_plan/gpu_exec.rs

use faer::Mat;
use crate::compute_manager::gpu::compute::GpuCompute;
use crate::loss_plan::CrossEntropyWithLogits;
use super::cubes::*;
use super::expr::LossExpr;

/// Выполняет вычисление потерь и градиентов на GPU.
///
/// Принимает матрицы `pred` и `target` размера `(batch, features)`,
/// где `features` — количество признаков предсказания / цели.
/// Возвращает скалярное значение потерь и матрицу градиентов по `pred`.
pub fn compute_loss_gpu(
    gpu: &GpuCompute,
    expr: &LossExpr,
    pred: &Mat<f32>,
    target: &Mat<f32>,
) -> (f32, Mat<f32>) {
    let pred_feat = expr.pred_features();
    let target_feat = expr.target_features();
    let batch = pred.nrows();
    assert_eq!(batch, target.nrows(), "Pred and target batch mismatch");
    assert_eq!(pred.ncols(), pred_feat, "Pred features mismatch");
    assert_eq!(target.ncols(), target_feat, "Target features mismatch");

    let in_features = pred_feat + target_feat;

    // Формируем матрицу [pred | target]
    let mut full_input = Mat::zeros(batch, in_features);
    for i in 0..batch {
        for j in 0..pred_feat {
            full_input[(i, j)] = pred[(i, j)];
        }
        for j in 0..target_feat {
            full_input[(i, pred_feat + j)] = target[(i, j)];
        }
    }

    // Прямой проход цепочки кубиков на GPU
    let chain = expr.chain();
    let mut current = full_input.clone();
    let mut intermediates: Vec<(Mat<f32>, Mat<f32>)> = Vec::with_capacity(chain.cubes().len());

    for cube in chain.cubes() {
        let input_for_cube = current.clone();
        let output = run_cube_forward_gpu(cube.as_ref(), gpu, &current, pred_feat, target_feat);
        intermediates.push((input_for_cube, output.clone()));
        current = output;
    }

    // current имеет размер (batch, 1) — значения потерь для каждого сэмпла
    let loss_vec: Vec<f32> = (0..batch).map(|i| current[(i, 0)]).collect();
    let loss = expr.aggregate_loss(&loss_vec);

    // Подготавливаем градиент по потерям с учётом агрегации
    let grad_scale = match expr.aggregation() {
        super::expr::Aggregation::Sum => 1.0f32,
        super::expr::Aggregation::Mean => 1.0f32 / batch as f32,
    };
    let grad_loss_mat = Mat::from_fn(batch, 1, |_i, _j| grad_scale);

    // Обратный проход цепочки на GPU
    let mut grad = grad_loss_mat;
    for (cube, (inp, _outp)) in chain.cubes().iter().zip(intermediates.iter()).rev() {
        grad = run_cube_backward_gpu(cube.as_ref(), gpu, inp, &grad, pred_feat, target_feat);
    }

    // grad имеет размер (batch, in_features). Извлекаем градиент по pred.
    let mut grad_pred = Mat::zeros(batch, pred_feat);
    for i in 0..batch {
        for j in 0..pred_feat {
            grad_pred[(i, j)] = grad[(i, j)];
        }
    }

    (loss, grad_pred)
}

/// Запуск прямого прохода одного кубика на GPU.
/// `pred_feat` и `target_feat` — число признаков предсказания и цели,
/// используются для корректного разделения входной матрицы.
fn run_cube_forward_gpu(
    cube: &dyn ElemCube,
    gpu: &GpuCompute,
    input: &Mat<f32>,
    pred_feat: usize,
    target_feat: usize,
) -> Mat<f32> {
    if let Some(_) = cube.as_any().downcast_ref::<Sub>() {
        let pred = input.subcols(0, pred_feat).to_owned();
        let targ = input.subcols(pred_feat, target_feat).to_owned();
        return gpu.run_sub_forward(&pred, &targ);
    } else if let Some(_) = cube.as_any().downcast_ref::<Square>() {
        return gpu.run_square_forward(input);
    } else if let Some(_) = cube.as_any().downcast_ref::<Abs>() {
        return gpu.run_abs_forward(input);
    } else if let Some(_) = cube.as_any().downcast_ref::<Log1p>() {
        return gpu.run_log1p_forward(input);
    } else if let Some(_) = cube.as_any().downcast_ref::<AbsDiff>() {
        let pred = input.subcols(0, pred_feat).to_owned();
        let targ = input.subcols(pred_feat, target_feat).to_owned();
        return gpu.run_absdiff_forward(&pred, &targ);
    } else if let Some(_) = cube.as_any().downcast_ref::<Log>() {
        return gpu.run_log_forward(input);
    } else if let Some(_) = cube.as_any().downcast_ref::<Neg>() {
        return gpu.run_neg_forward(input);
    } else if let Some(_) = cube.as_any().downcast_ref::<Mul>() {
        let pred = input.subcols(0, pred_feat).to_owned();
        let targ = input.subcols(pred_feat, target_feat).to_owned();
        return gpu.run_mul_forward(&pred, &targ);
    } else if let Some(addscalar) = cube.as_any().downcast_ref::<AddScalar>() {
        return gpu.run_addscalar_forward(input, addscalar.0);
    } else if let Some(ce) = cube.as_any().downcast_ref::<CrossEntropyWithLogits>() {
        return gpu.run_cross_entropy_forward(input, ce.num_classes);
    } else if let Some(_) = cube.as_any().downcast_ref::<SumColumns>() {
        // Суммируем столбцы внутри каждой строки, чтобы получить (batch, 1)
        let batch = input.nrows();
        let cols = input.ncols();
        let mut out = Mat::zeros(batch, 1);
        for i in 0..batch {
            let mut sum = 0.0;
            for j in 0..cols {
                sum += input[(i, j)];
            }
            out[(i, 0)] = sum;
        }
        return out;
    }
    panic!("Unknown loss cube for GPU forward");
}

/// Запуск обратного прохода одного кубика на GPU.
fn run_cube_backward_gpu(
    cube: &dyn ElemCube,
    gpu: &GpuCompute,
    input: &Mat<f32>,
    grad_out: &Mat<f32>,
    pred_feat: usize,
    target_feat: usize,
) -> Mat<f32> {
    if let Some(_) = cube.as_any().downcast_ref::<Sub>() {
        let (ga, gb) = gpu.run_sub_backward(grad_out);
        let batch = ga.nrows();
        let mut result = Mat::zeros(batch, pred_feat + target_feat);
        // ga и gb имеют размер (batch, pred_feat)
        for i in 0..batch {
            for j in 0..pred_feat {
                result[(i, j)] = ga[(i, j)];
                result[(i, j + pred_feat)] = gb[(i, j)];
            }
        }
        return result;
    } else if let Some(_) = cube.as_any().downcast_ref::<Square>() {
        return gpu.run_square_backward(input, grad_out);
    } else if let Some(_) = cube.as_any().downcast_ref::<Abs>() {
        return gpu.run_abs_backward(input, grad_out);
    } else if let Some(_) = cube.as_any().downcast_ref::<Log1p>() {
        return gpu.run_log1p_backward(input, grad_out);
    } else if let Some(_) = cube.as_any().downcast_ref::<AbsDiff>() {
        let pred = input.subcols(0, pred_feat).to_owned();
        let targ = input.subcols(pred_feat, target_feat).to_owned();
        let (ga, gb) = gpu.run_absdiff_backward(&pred, &targ, grad_out);
        let batch = ga.nrows();
        let mut result = Mat::zeros(batch, pred_feat + target_feat);
        for i in 0..batch {
            for j in 0..pred_feat {
                result[(i, j)] = ga[(i, j)];
                result[(i, j + pred_feat)] = gb[(i, j)];
            }
        }
        return result;
    } else if let Some(_) = cube.as_any().downcast_ref::<Log>() {
        return gpu.run_log_backward(input, grad_out);
    } else if let Some(_) = cube.as_any().downcast_ref::<Neg>() {
        return gpu.run_neg_backward(grad_out);
    } else if let Some(_) = cube.as_any().downcast_ref::<Mul>() {
        let pred = input.subcols(0, pred_feat).to_owned();
        let targ = input.subcols(pred_feat, target_feat).to_owned();
        let (ga, gb) = gpu.run_mul_backward(&pred, &targ, grad_out);
        let batch = ga.nrows();
        let mut result = Mat::zeros(batch, pred_feat + target_feat);
        for i in 0..batch {
            for j in 0..pred_feat {
                result[(i, j)] = ga[(i, j)];
                result[(i, j + pred_feat)] = gb[(i, j)];
            }
        }
        return result;
    } else if let Some(_) = cube.as_any().downcast_ref::<AddScalar>() {
        return gpu.run_addscalar_backward(grad_out);
    } else if let Some(ce) = cube.as_any().downcast_ref::<CrossEntropyWithLogits>() {
        return gpu.run_cross_entropy_backward(input, grad_out, ce.num_classes);
    } else if let Some(_) = cube.as_any().downcast_ref::<SumColumns>() {
        // При обратном проходе: градиент из (batch,1) дублируем на все столбцы
        let batch = grad_out.nrows();
        let cols = input.ncols(); // исходное число столбцов до SumColumns
        let mut grad = Mat::zeros(batch, cols);
        for i in 0..batch {
            let g = grad_out[(i, 0)];
            for j in 0..cols {
                grad[(i, j)] = g;
            }
        }
        return grad;
    }
    panic!("Unknown loss cube for GPU backward");
}
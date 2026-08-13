// src/plans/loss_plan/gpu_exec.rs

use faer::Mat;
use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBuffer;
use crate::loss_plan::CrossEntropyWithLogits;
use super::cubes::*;
use super::expr::LossExpr;

/// Выполняет вычисление потерь и градиентов на GPU.
///
/// Принимает матрицы `pred` и `target` размера `(batch, features)`,
/// где `features` — количество признаков предсказания / цели.
/// Возвращает скалярное значение потерь и матрицу градиентов по `pred`.
///
/// Старая реализация, оставлена для обратной совместимости.
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

    let mut full_input = Mat::zeros(batch, in_features);
    for i in 0..batch {
        for j in 0..pred_feat {
            full_input[(i, j)] = pred[(i, j)];
        }
        for j in 0..target_feat {
            full_input[(i, pred_feat + j)] = target[(i, j)];
        }
    }

    let chain = expr.chain();
    let mut current = full_input.clone();
    let mut intermediates: Vec<(Mat<f32>, Mat<f32>)> = Vec::with_capacity(chain.cubes().len());

    for cube in chain.cubes() {
        let input_for_cube = current.clone();
        let output = run_cube_forward_gpu(cube.as_ref(), gpu, &current, pred_feat, target_feat);
        intermediates.push((input_for_cube, output.clone()));
        current = output;
    }

    let loss_vec: Vec<f32> = (0..batch).map(|i| current[(i, 0)]).collect();
    let loss = expr.aggregate_loss(&loss_vec);

    let grad_scale = match expr.aggregation() {
        super::expr::Aggregation::Sum => 1.0f32,
        super::expr::Aggregation::Mean => 1.0f32 / batch as f32,
    };
    let grad_loss_mat = Mat::from_fn(batch, 1, |_i, _j| grad_scale);

    let mut grad = grad_loss_mat;
    for (cube, (inp, _outp)) in chain.cubes().iter().zip(intermediates.iter()).rev() {
        grad = run_cube_backward_gpu(cube.as_ref(), gpu, inp, &grad, pred_feat, target_feat);
    }

    let mut grad_pred = Mat::zeros(batch, pred_feat);
    for i in 0..batch {
        for j in 0..pred_feat {
            grad_pred[(i, j)] = grad[(i, j)];
        }
    }

    (loss, grad_pred)
}

/// Запуск прямого прохода одного кубика на GPU (старая матричная версия).
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

/// Запуск обратного прохода одного кубика на GPU (старая матричная версия).
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
        let batch = grad_out.nrows();
        let cols = input.ncols();
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

// ===================================================================
// НОВАЯ РЕАЛИЗАЦИЯ НА MATRIXBUFFER (БЕЗ faer::Mat)
// ===================================================================

/// Выполняет вычисление потерь и градиентов на GPU с использованием управляемых буферов `MatrixBuffer`.
///
/// Принимает GPU‑буферы `pred` и `target` размера `(batch, features)`.
/// Возвращает скалярное значение потерь и GPU‑буфер градиентов по `pred`.
///
/// Внутри используются только `MatrixBuffer` и `Vec<f32>` для временного
/// копирования при необходимости. `faer::Mat` не создаётся.
pub fn compute_loss_gpu_buffered(
    gpu: &GpuCompute,
    expr: &LossExpr,
    pred: &MatrixBuffer,
    target: &MatrixBuffer,
) -> (f32, MatrixBuffer) {
    assert!(pred.is_gpu() && target.is_gpu(), "compute_loss_gpu_buffered requires GPU buffers");

    let pred_feat = expr.pred_features();
    let target_feat = expr.target_features();
    let batch = pred.rows();
    assert_eq!(batch, target.rows(), "Pred and target batch mismatch");
    assert_eq!(pred.cols(), pred_feat, "Pred features mismatch");
    assert_eq!(target.cols(), target_feat, "Target features mismatch");

    // Скачиваем данные во временные Vec (без Mat)
    let pred_vec = gpu.download_gpu_matrix_to_vec(pred);
    let target_vec = gpu.download_gpu_matrix_to_vec(target);

    // Создаём отдельные GPU-буферы для pred и target
    let pred_gpu = gpu.upload_vec_to_gpu_buffer(&pred_vec, batch, pred_feat);
    let target_gpu = gpu.upload_vec_to_gpu_buffer(&target_vec, batch, target_feat);

    let chain = expr.chain();
    let cubes = chain.cubes();

    // Храним все промежуточные буферы в одном векторе.
    // Индексы помогают избежать клонирования MatrixBuffer.
    let mut buffers: Vec<MatrixBuffer> = Vec::with_capacity(cubes.len());
    let mut current_idx: usize = 0; // будет переопределён после первого кубика
    let mut input_indices: Vec<usize> = Vec::with_capacity(cubes.len());
    let mut output_indices: Vec<usize> = Vec::with_capacity(cubes.len());

    // Обработка первого кубика (он может принимать пару pred/target)
    if cubes.is_empty() {
        panic!("Loss chain cannot be empty");
    }
    let first_cube = cubes[0].as_ref();
    let out0 = handle_first_cube_forward_buffered(gpu, first_cube, &pred_gpu, &target_gpu);
    buffers.push(out0);
    current_idx = buffers.len() - 1;

    // Сохраняем индексы для первого кубика отдельно: у него входы pred_gpu и target_gpu,
    // которые не находятся в buffers. Для обратного прохода мы используем их напрямую.
    // Для единообразия можно не сохранять в input_indices/output_indices для первого.

    // Обработка остальных кубиков
    for (idx, cube) in cubes.iter().enumerate().skip(1) {
        let input_idx = current_idx;
        let out = handle_unary_cube_forward_buffered(gpu, cube.as_ref(), &buffers[input_idx]);
        buffers.push(out);
        let output_idx = buffers.len() - 1;
        input_indices.push(input_idx);
        output_indices.push(output_idx);
        current_idx = output_idx;
    }

    // Получаем вектор потерь (размер batch)
    let final_buf = &buffers[current_idx];
    let loss_vec: Vec<f32> = if final_buf.cols() == 1 {
        gpu.download_gpu_matrix_to_vec(final_buf)
    } else {
        // Если SumColumns не был последним, агрегируем на CPU
        let raw = gpu.download_gpu_matrix_to_vec(final_buf);
        let rows = final_buf.rows();
        let cols = final_buf.cols();
        (0..rows)
            .map(|r| (0..cols).map(|c| raw[c * rows + r]).sum())
            .collect()
    };
    let loss = expr.aggregate_loss(&loss_vec);

    // Обратный проход
    let grad_scale = match expr.aggregation() {
        super::expr::Aggregation::Sum => 1.0f32,
        super::expr::Aggregation::Mean => 1.0f32 / batch as f32,
    };
    let mut grad = gpu.allocate_gpu_matrix(batch, 1);
    gpu.fill_gpu_buffer(&mut grad, grad_scale);

    // Идём с конца цепочки
    let mut last_is_first = cubes.len() == 1;
    let mut reverse_idx = cubes.len() - 1;

    if !last_is_first {
        // Проходим унарные кубики в обратном порядке
        for rev_pos in (0..(cubes.len() - 1)).rev() {
            let cube = cubes[rev_pos + 1].as_ref(); // потому что первый кубик мы пропускаем
            let input_idx = input_indices[rev_pos];
            let output_idx = output_indices[rev_pos];
            grad = handle_unary_cube_backward_buffered(
                gpu,
                cube,
                &buffers[input_idx],
                &buffers[output_idx],
                &grad,
                buffers[input_idx].cols(),
            );
        }
    }

    // Теперь обрабатываем первый кубик, получаем градиент по pred
    grad = handle_first_cube_backward_buffered(
        gpu,
        first_cube,
        &pred_gpu,
        &target_gpu,
        &grad,
        pred_feat,
        target_feat,
    );

    (loss, grad)
}

/// Обрабатывает первый кубик цепочки на GPU.
/// Возвращает выходной буфер.
fn handle_first_cube_forward_buffered(
    gpu: &GpuCompute,
    cube: &dyn ElemCube,
    pred: &MatrixBuffer,
    target: &MatrixBuffer,
) -> MatrixBuffer {
    if let Some(_) = cube.as_any().downcast_ref::<Sub>() {
        gpu.run_sub_forward_buffered(pred, target)
    } else if let Some(_) = cube.as_any().downcast_ref::<Mul>() {
        gpu.run_mul_forward_buffered(pred, target)
    } else if let Some(_) = cube.as_any().downcast_ref::<AbsDiff>() {
        gpu.run_absdiff_forward_buffered(pred, target)
    } else if let Some(ce) = cube.as_any().downcast_ref::<CrossEntropyWithLogits>() {
        // Объединяем pred и target (метку класса) на CPU и загружаем обратно
        let pred_vec = gpu.download_gpu_matrix_to_vec(pred);
        let target_vec = gpu.download_gpu_matrix_to_vec(target);
        let mut combined = pred_vec;
        combined.extend_from_slice(&target_vec);
        let combined_gpu = gpu.upload_vec_to_gpu_buffer(
            &combined,
            pred.rows(),
            pred.cols() + target.cols(),
        );
        gpu.run_cross_entropy_forward_buffered(&combined_gpu, ce.num_classes)
    } else {
        panic!("Unsupported first loss cube for GPU buffered");
    }
}

/// Обрабатывает унарный кубик (включая SumColumns).
fn handle_unary_cube_forward_buffered(
    gpu: &GpuCompute,
    cube: &dyn ElemCube,
    input: &MatrixBuffer,
) -> MatrixBuffer {
    if let Some(_) = cube.as_any().downcast_ref::<Square>() {
        gpu.run_square_forward_buffered(input)
    } else if let Some(_) = cube.as_any().downcast_ref::<Abs>() {
        gpu.run_abs_forward_buffered(input)
    } else if let Some(_) = cube.as_any().downcast_ref::<Log1p>() {
        gpu.run_log1p_forward_buffered(input)
    } else if let Some(_) = cube.as_any().downcast_ref::<Log>() {
        gpu.run_log_forward_buffered(input)
    } else if let Some(_) = cube.as_any().downcast_ref::<Neg>() {
        gpu.run_neg_forward_buffered(input)
    } else if let Some(addscalar) = cube.as_any().downcast_ref::<AddScalar>() {
        gpu.run_addscalar_forward_buffered(input, addscalar.0)
    } else if let Some(_) = cube.as_any().downcast_ref::<SumColumns>() {
        gpu.run_sum_columns_forward_buffered(input)
    } else {
        panic!("Unsupported unary loss cube for GPU buffered");
    }
}

/// Обрабатывает обратный проход первого кубика.
/// Возвращает градиент по pred.
fn handle_first_cube_backward_buffered(
    gpu: &GpuCompute,
    cube: &dyn ElemCube,
    pred: &MatrixBuffer,
    target: &MatrixBuffer,
    grad_out: &MatrixBuffer,
    pred_feat: usize,
    _target_feat: usize,
) -> MatrixBuffer {
    if let Some(_) = cube.as_any().downcast_ref::<Sub>() {
        let (ga, _gb) = gpu.run_sub_backward_buffered(grad_out);
        ga
    } else if let Some(_) = cube.as_any().downcast_ref::<Mul>() {
        let (ga, _gb) = gpu.run_mul_backward_buffered(pred, target, grad_out);
        ga
    } else if let Some(_) = cube.as_any().downcast_ref::<AbsDiff>() {
        let (ga, _gb) = gpu.run_absdiff_backward_buffered(pred, target, grad_out);
        ga
    } else if let Some(ce) = cube.as_any().downcast_ref::<CrossEntropyWithLogits>() {
        // pred и target объединены в один буфер
        let combined_gpu = gpu.upload_vec_to_gpu_buffer(
            &{
                let mut v = gpu.download_gpu_matrix_to_vec(pred);
                v.extend_from_slice(&gpu.download_gpu_matrix_to_vec(target));
                v
            },
            pred.rows(),
            pred.cols() + target.cols(),
        );
        let grad_combined = gpu.run_cross_entropy_backward_buffered(&combined_gpu, grad_out, ce.num_classes);
        let grad_vec = gpu.download_gpu_matrix_to_vec(&grad_combined);
        let batch = pred.rows();
        let mut grad_pred_vec = Vec::with_capacity(batch * pred_feat);
        for c in 0..pred_feat {
            let start = c * batch;
            grad_pred_vec.extend_from_slice(&grad_vec[start..start + batch]);
        }
        gpu.upload_vec_to_gpu_buffer(&grad_pred_vec, batch, pred_feat)
    } else {
        panic!("Unsupported first loss cube for GPU backward");
    }
}

/// Обрабатывает обратный проход унарного кубика.
fn handle_unary_cube_backward_buffered(
    gpu: &GpuCompute,
    cube: &dyn ElemCube,
    input: &MatrixBuffer,
    _output: &MatrixBuffer,
    grad_out: &MatrixBuffer,
    original_cols: usize,
) -> MatrixBuffer {
    if let Some(_) = cube.as_any().downcast_ref::<Square>() {
        gpu.run_square_backward_buffered(input, grad_out)
    } else if let Some(_) = cube.as_any().downcast_ref::<Abs>() {
        gpu.run_abs_backward_buffered(input, grad_out)
    } else if let Some(_) = cube.as_any().downcast_ref::<Log1p>() {
        gpu.run_log1p_backward_buffered(input, grad_out)
    } else if let Some(_) = cube.as_any().downcast_ref::<Log>() {
        gpu.run_log_backward_buffered(input, grad_out)
    } else if let Some(_) = cube.as_any().downcast_ref::<Neg>() {
        gpu.run_neg_backward_buffered(grad_out)
    } else if let Some(_) = cube.as_any().downcast_ref::<AddScalar>() {
        gpu.run_addscalar_backward_buffered(grad_out)
    } else if let Some(_) = cube.as_any().downcast_ref::<SumColumns>() {
        gpu.run_sum_columns_backward_buffered(grad_out, original_cols)
    } else {
        panic!("Unsupported unary loss cube for GPU backward");
    }
}
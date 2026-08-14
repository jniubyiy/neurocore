// src/plans/loss_plan/gpu_exec.rs

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::loss_plan::CrossEntropyWithLogits;
use super::cubes::*;
use super::expr::{Aggregation, LossExpr};

/// Вычисляет значение функции потерь и градиент по pred на GPU.
/// Все промежуточные операции выполняются на GPU, без CPU‑fallback.
/// Единственное скачивание на CPU — финальный вектор потерь (допустимо).
pub fn compute_loss_gpu_buffered_handle(
    gpu: &GpuCompute,
    expr: &LossExpr,
    pred: &MatrixBufferHandle,
    target: &MatrixBufferHandle,
) -> (f32, MatrixBufferHandle) {
    assert!(pred.is_gpu() && target.is_gpu(),
        "compute_loss_gpu_buffered_handle requires GPU buffers");

    let pred_feat = expr.pred_features();
    let target_feat = expr.target_features();
    let batch = pred.rows();
    assert_eq!(batch, target.rows(), "Pred and target batch mismatch");
    assert_eq!(pred.cols(), pred_feat, "Pred features mismatch");
    assert_eq!(target.cols(), target_feat, "Target features mismatch");

    let chain = expr.chain();
    let cubes = chain.cubes();
    if cubes.is_empty() {
        panic!("Loss chain cannot be empty");
    }

    // ---------------- Прямой проход ----------------
    let mut buffers: Vec<MatrixBufferHandle> = Vec::with_capacity(cubes.len());
    let mut saved_combined: Option<MatrixBufferHandle> = None;

    // Первый кубик может быть бинарным (Sub/Mul/AbsDiff) или CrossEntropy
    let (out0, combined_opt) = handle_first_cube_forward_buffered(gpu, cubes[0].as_ref(), pred, target);
    buffers.push(out0);
    if let Some(c) = combined_opt {
        saved_combined = Some(c);
    }

    // Остальные кубики (унарные, включая SumColumns)
    let mut current_idx = 0;
    for cube in cubes.iter().skip(1) {
        let out = handle_unary_cube_forward_buffered(gpu, cube.as_ref(), &buffers[current_idx]);
        buffers.push(out);
        current_idx += 1;
    }

    // Финальный буфер – loss (batch, 1) или (batch, features) если SumColumns не последний
    let final_buf = &buffers[current_idx];
    let loss_vec: Vec<f32> = if final_buf.cols() == 1 {
        gpu.download_gpu_handle_to_vec(final_buf)
    } else {
        // Если SumColumns не последний, агрегируем на CPU (нестандартный случай)
        let raw = gpu.download_gpu_handle_to_vec(final_buf);
        let rows = final_buf.rows();
        let cols = final_buf.cols();
        (0..rows)
            .map(|r| (0..cols).map(|c| raw[c * rows + r]).sum())
            .collect()
    };
    let loss = expr.aggregate_loss(&loss_vec);

    // ---------------- Обратный проход ----------------
    let grad_scale = match expr.aggregation() {
        Aggregation::Sum => 1.0f32,
        Aggregation::Mean => 1.0f32 / batch as f32,
    };
    let mut grad = gpu.allocate_gpu_matrix_handle(batch, 1);
    gpu.fill_gpu_handle(&grad, grad_scale);

    // Проходим унарные кубики в обратном порядке (если они есть)
    let num_unary = cubes.len().saturating_sub(1);
    for rev_pos in (0..num_unary).rev() {
        let cube = cubes[rev_pos + 1].as_ref(); // унарный кубик
        let input_idx = rev_pos;
        let output_idx = rev_pos + 1;
        grad = handle_unary_cube_backward_buffered(
            gpu,
            cube,
            &buffers[input_idx],
            &buffers[output_idx],
            &grad,
            buffers[input_idx].cols(),
        );
    }

    // Обрабатываем первый кубик
    grad = handle_first_cube_backward_buffered(
        gpu,
        cubes[0].as_ref(),
        pred,
        target,
        saved_combined.as_ref(),
        &grad,
        pred_feat,
        target_feat,
    );

    (loss, grad)
}

/// Обрабатывает первый кубик цепочки на GPU (handle-версия).
/// Возвращает выходной дескриптор и, для CrossEntropy, объединённый входной дескриптор.
fn handle_first_cube_forward_buffered(
    gpu: &GpuCompute,
    cube: &dyn ElemCube,
    pred: &MatrixBufferHandle,
    target: &MatrixBufferHandle,
) -> (MatrixBufferHandle, Option<MatrixBufferHandle>) {
    if cube.as_any().downcast_ref::<Sub>().is_some() {
        let out = gpu.allocate_gpu_matrix_handle(pred.rows(), pred.cols());
        gpu.run_sub_forward_buffered_handle(pred, target, &out);
        (out, None)
    } else if cube.as_any().downcast_ref::<Mul>().is_some() {
        let out = gpu.allocate_gpu_matrix_handle(pred.rows(), pred.cols());
        gpu.run_mul_forward_buffered_handle(pred, target, &out);
        (out, None)
    } else if cube.as_any().downcast_ref::<AbsDiff>().is_some() {
        let out = gpu.allocate_gpu_matrix_handle(pred.rows(), pred.cols());
        gpu.run_absdiff_forward_buffered_handle(pred, target, &out);
        (out, None)
    } else if let Some(ce) = cube.as_any().downcast_ref::<CrossEntropyWithLogits>() {
        // Объединяем pred и target на GPU без CPU
        let batch = pred.rows();
        let num_classes = ce.num_classes;
        let combined = gpu.allocate_gpu_matrix_handle(batch, num_classes + 1);

        // Копируем столбцы pred в первые num_classes столбцов combined
        for c in 0..num_classes {
            gpu.copy_gpu_handle_region(
                pred,
                &combined,
                c * batch,         // src offset
                c * batch,         // dst offset
                batch,
            );
        }
        // Копируем target в последний столбец
        gpu.copy_gpu_handle_region(
            target,
            &combined,
            0,
            num_classes * batch,
            batch,
        );

        let out = gpu.allocate_gpu_matrix_handle(batch, 1);
        gpu.run_cross_entropy_forward_buffered_handle(&combined, ce.num_classes, &out);
        (out, Some(combined))
    } else {
        panic!("Unsupported first loss cube for GPU buffered handle");
    }
}

/// Обрабатывает унарный кубик (включая SumColumns) на GPU.
fn handle_unary_cube_forward_buffered(
    gpu: &GpuCompute,
    cube: &dyn ElemCube,
    input: &MatrixBufferHandle,
) -> MatrixBufferHandle {
    if cube.as_any().downcast_ref::<Square>().is_some() {
        let out = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());
        gpu.run_square_forward_buffered_handle(input, &out);
        out
    } else if cube.as_any().downcast_ref::<Abs>().is_some() {
        let out = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());
        gpu.run_abs_forward_buffered_handle(input, &out);
        out
    } else if cube.as_any().downcast_ref::<Log1p>().is_some() {
        let out = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());
        gpu.run_log1p_forward_buffered_handle(input, &out);
        out
    } else if cube.as_any().downcast_ref::<Log>().is_some() {
        let out = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());
        gpu.run_log_forward_buffered_handle(input, &out);
        out
    } else if cube.as_any().downcast_ref::<Neg>().is_some() {
        let out = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());
        gpu.run_neg_forward_buffered_handle(input, &out);
        out
    } else if let Some(addscalar) = cube.as_any().downcast_ref::<AddScalar>() {
        let out = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());
        gpu.run_addscalar_forward_buffered_handle(input, addscalar.0, &out);
        out
    } else if cube.as_any().downcast_ref::<SumColumns>().is_some() {
        let out = gpu.allocate_gpu_matrix_handle(input.rows(), 1);
        gpu.run_sum_columns_forward_buffered_handle(input, &out);
        out
    } else {
        panic!("Unsupported unary loss cube for GPU buffered handle");
    }
}

/// Обрабатывает обратный проход первого кубика.
/// Возвращает градиент по pred (GPU-дескриптор).
fn handle_first_cube_backward_buffered(
    gpu: &GpuCompute,
    cube: &dyn ElemCube,
    pred: &MatrixBufferHandle,
    target: &MatrixBufferHandle,
    combined: Option<&MatrixBufferHandle>,
    grad_out: &MatrixBufferHandle,
    pred_feat: usize,
    _target_feat: usize,
) -> MatrixBufferHandle {
    if cube.as_any().downcast_ref::<Sub>().is_some() {
        let ga = gpu.allocate_gpu_matrix_handle(grad_out.rows(), grad_out.cols());
        let gb = gpu.allocate_gpu_matrix_handle(grad_out.rows(), grad_out.cols());
        gpu.run_sub_backward_buffered_handle(grad_out, &ga, &gb);
        ga
    } else if cube.as_any().downcast_ref::<Mul>().is_some() {
        let ga = gpu.allocate_gpu_matrix_handle(grad_out.rows(), grad_out.cols());
        let gb = gpu.allocate_gpu_matrix_handle(grad_out.rows(), grad_out.cols());
        gpu.run_mul_backward_buffered_handle(pred, target, grad_out, &ga, &gb);
        ga
    } else if cube.as_any().downcast_ref::<AbsDiff>().is_some() {
        let ga = gpu.allocate_gpu_matrix_handle(grad_out.rows(), grad_out.cols());
        let gb = gpu.allocate_gpu_matrix_handle(grad_out.rows(), grad_out.cols());
        gpu.run_absdiff_backward_buffered_handle(pred, target, grad_out, &ga, &gb);
        ga
    } else if let Some(ce) = cube.as_any().downcast_ref::<CrossEntropyWithLogits>() {
        let combined = combined.expect("CrossEntropy backward requires combined handle from forward");
        let batch = pred.rows();
        let num_classes = ce.num_classes;

        // Вызываем backward на объединённом буфере
        let grad_combined = gpu.allocate_gpu_matrix_handle(batch, num_classes + 1);
        gpu.run_cross_entropy_backward_buffered_handle(
            combined,
            grad_out,
            ce.num_classes,
            &grad_combined,
        );

        // Извлекаем градиент по pred (первые pred_feat столбцов)
        let grad_pred = gpu.allocate_gpu_matrix_handle(batch, pred_feat);
        for c in 0..pred_feat {
            gpu.copy_gpu_handle_region(
                &grad_combined,
                &grad_pred,
                c * batch,
                c * batch,
                batch,
            );
        }
        grad_pred
    } else {
        panic!("Unsupported first loss cube for GPU backward handle");
    }
}

/// Обрабатывает обратный проход унарного кубика.
fn handle_unary_cube_backward_buffered(
    gpu: &GpuCompute,
    cube: &dyn ElemCube,
    input: &MatrixBufferHandle,
    _output: &MatrixBufferHandle,
    grad_out: &MatrixBufferHandle,
    original_cols: usize,
) -> MatrixBufferHandle {
    if cube.as_any().downcast_ref::<Square>().is_some() {
        let grad_in = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());
        gpu.run_square_backward_buffered_handle(input, grad_out, &grad_in);
        grad_in
    } else if cube.as_any().downcast_ref::<Abs>().is_some() {
        let grad_in = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());
        gpu.run_abs_backward_buffered_handle(input, grad_out, &grad_in);
        grad_in
    } else if cube.as_any().downcast_ref::<Log1p>().is_some() {
        let grad_in = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());
        gpu.run_log1p_backward_buffered_handle(input, grad_out, &grad_in);
        grad_in
    } else if cube.as_any().downcast_ref::<Log>().is_some() {
        let grad_in = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());
        gpu.run_log_backward_buffered_handle(input, grad_out, &grad_in);
        grad_in
    } else if cube.as_any().downcast_ref::<Neg>().is_some() {
        let grad_in = gpu.allocate_gpu_matrix_handle(grad_out.rows(), grad_out.cols());
        gpu.run_neg_backward_buffered_handle(grad_out, &grad_in);
        grad_in
    } else if cube.as_any().downcast_ref::<AddScalar>().is_some() {
        let grad_in = gpu.allocate_gpu_matrix_handle(grad_out.rows(), grad_out.cols());
        gpu.run_addscalar_backward_buffered_handle(grad_out, &grad_in);
        grad_in
    } else if cube.as_any().downcast_ref::<SumColumns>().is_some() {
        let grad_in = gpu.allocate_gpu_matrix_handle(grad_out.rows(), original_cols);
        gpu.run_sum_columns_backward_buffered_handle(grad_out, original_cols, &grad_in);
        grad_in
    } else {
        panic!("Unsupported unary loss cube for GPU backward handle");
    }
}
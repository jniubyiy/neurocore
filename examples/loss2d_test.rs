// examples/loss2d_test.rs
// Тестирование функций потерь для Dim2: MSE, CrossEntropy, difference_loss, diff_smooth_loss.
// Использует буферизованный API MatrixBufferHandle + TempMatrixPool.

use std::sync::{Arc, Mutex};

use faer::Mat;
use neurocore::compute_manager::device_spec::DeviceSpec;
use neurocore::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use neurocore::compute_manager::memory_executor::MemoryExecutor;
use neurocore::loss_plan::{
    Aggregation, Abs, AbsDiff, AddScalar, CrossEntropyWithLogits, ElementChain, Log1p, LossDesc,
    Square, Sub, SumColumns,
};

mod losses {
    use neurocore::loss_plan::{
        Aggregation, Abs, AbsDiff, AddScalar, CrossEntropyWithLogits, ElementChain, Log1p,
        LossDesc, Square, Sub, SumColumns,
    };

    pub fn mse() -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(Sub::new(1)))
            .add(Box::new(Square))
            .add(Box::new(SumColumns));
        LossDesc::from_chain(chain, Aggregation::Mean, 4, 1, 1)
    }

    pub fn cross_entropy() -> LossDesc {
        let num_classes = 4;
        let chain = ElementChain::new()
            .add(Box::new(CrossEntropyWithLogits::new(num_classes)));
        LossDesc::from_chain(chain, Aggregation::Mean, 1, num_classes, 1)
    }

    pub fn difference_loss() -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(Sub::new(1)))
            .add(Box::new(Abs))
            .add(Box::new(AddScalar(1.0)))
            .add(Box::new(Log1p))
            .add(Box::new(SumColumns));
        LossDesc::from_chain(chain, Aggregation::Mean, 4, 1, 1)
    }

    pub fn diff_smooth_loss_h() -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(AbsDiff::new(1)));
        LossDesc::from_chain(chain, Aggregation::Mean, 2, 1, 1)
    }

    pub fn diff_smooth_loss_v() -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(AbsDiff::new(1)));
        LossDesc::from_chain(chain, Aggregation::Mean, 2, 1, 1)
    }
}

/// Создаёт MatrixBufferHandle из Mat (column-major).
fn mat_to_handle(mat: &Mat<f32>, pool: &mut TempMatrixPool) -> MatrixBufferHandle {
    let rows = mat.nrows();
    let cols = mat.ncols();
    let handle = pool.acquire(rows, cols);
    {
        let mut guard = handle.write();
        let dst = guard.as_slice_mut().expect("CPU buffer");
        for c in 0..cols {
            for r in 0..rows {
                dst[c * rows + r] = mat[(r, c)];
            }
        }
    }
    handle
}

/// Преобразует MatrixBufferHandle в Mat (column-major).
fn handle_to_mat(handle: &MatrixBufferHandle) -> Mat<f32> {
    let rows = handle.rows();
    let cols = handle.cols();
    let guard = handle.read();
    let src = guard.as_slice().expect("CPU buffer");
    Mat::from_fn(rows, cols, |r, c| src[c * rows + r])
}

fn main() {
    // Создаём MemoryExecutor и TempMatrixPool
    let mem = Arc::new(Mutex::new(MemoryExecutor::new()));
    mem.lock()
        .unwrap()
        .register_compute_device(DeviceSpec::cpu(0, 1024, 1), None);
    // Устанавливаем ссылку на самого себя
    mem.lock().unwrap().set_self_arc(mem.clone());

    let mut pool = TempMatrixPool::new(mem);

    let mse_expr = losses::mse().build();
    let ce_expr = losses::cross_entropy().build();
    let diff_loss_expr = losses::difference_loss().build();
    let smooth_h_expr = losses::diff_smooth_loss_h().build();
    let smooth_v_expr = losses::diff_smooth_loss_v().build();

    // ==================== MSE ====================
    println!("--- MSE (2D) ---");
    let total_elements = 4;
    let pred = vec![1.0f32, 2.0, 3.0, 4.0];
    let target = vec![1.5, 1.5, 3.5, 4.5];

    let mse_in_size = mse_expr.task_input_size(); // = 2
    let mut full_input = Mat::zeros(total_elements, mse_in_size);
    for i in 0..total_elements {
        full_input[(i, 0)] = pred[i];
        full_input[(i, 1)] = target[i];
    }

    let full_input_handle = mat_to_handle(&full_input, &mut pool);
    let (loss_vec, intermediates) = mse_expr.forward_chunk_buffered(&full_input_handle, &mut pool);
    let loss = mse_expr.aggregate_loss(&loss_vec);
    println!("MSE loss: {:.6}", loss);

    let grad_loss = vec![1.0f32; total_elements];
    let grad_handle = mse_expr.backward_chunk_buffered(&intermediates, &grad_loss, &mut pool);
    let grad_mat = handle_to_mat(&grad_handle);
    let grad: Vec<f32> = (0..total_elements).map(|i| grad_mat[(i, 0)]).collect();
    println!("MSE grad: {:?}", grad);

    pool.release(full_input_handle);
    pool.release(grad_handle);
    for (inp, outp) in intermediates {
        pool.release(inp);
        pool.release(outp);
    }

    // ==================== CrossEntropy ====================
    println!("\n--- CrossEntropy (2D) ---");
    let pred_logits = vec![0.2f32, 0.5, 0.1, 0.2];
    let class_index = 1.0f32;

    let ce_in_size = ce_expr.task_input_size(); // num_classes + 1 = 5
    let mut ce_input = Mat::zeros(1, ce_in_size);
    ce_input[(0, 0)] = pred_logits[0];
    ce_input[(0, 1)] = pred_logits[1];
    ce_input[(0, 2)] = pred_logits[2];
    ce_input[(0, 3)] = pred_logits[3];
    ce_input[(0, 4)] = class_index;

    let ce_input_handle = mat_to_handle(&ce_input, &mut pool);
    let (loss_vec, intermediates) = ce_expr.forward_chunk_buffered(&ce_input_handle, &mut pool);
    let ce_loss = ce_expr.aggregate_loss(&loss_vec);
    println!("CE loss: {:.6}", ce_loss);

    let grad_loss = vec![1.0f32; 1];
    let grad_handle = ce_expr.backward_chunk_buffered(&intermediates, &grad_loss, &mut pool);
    let grad_mat = handle_to_mat(&grad_handle);
    let grad_ce: Vec<f32> = (0..4).map(|j| grad_mat[(0, j)]).collect();
    println!("CE grad (first 4): {:?}", grad_ce);

    pool.release(ce_input_handle);
    pool.release(grad_handle);
    for (inp, outp) in intermediates {
        pool.release(inp);
        pool.release(outp);
    }

    // ==================== Difference Loss ====================
    println!("\n--- Difference Loss (2D) ---");
    let diff_in_size = diff_loss_expr.task_input_size(); // = 2
    let mut diff_input = Mat::zeros(total_elements, diff_in_size);
    for i in 0..total_elements {
        diff_input[(i, 0)] = pred[i];
        diff_input[(i, 1)] = target[i];
    }

    let diff_input_handle = mat_to_handle(&diff_input, &mut pool);
    let (loss_vec, intermediates) = diff_loss_expr.forward_chunk_buffered(&diff_input_handle, &mut pool);
    let loss = diff_loss_expr.aggregate_loss(&loss_vec);
    println!("Diff loss: {:.6}", loss);

    let grad_loss = vec![1.0f32; total_elements];
    let grad_handle = diff_loss_expr.backward_chunk_buffered(&intermediates, &grad_loss, &mut pool);
    let grad_mat = handle_to_mat(&grad_handle);
    let grad: Vec<f32> = (0..total_elements).map(|i| grad_mat[(i, 0)]).collect();
    println!("Diff grad: {:?}", grad);

    pool.release(diff_input_handle);
    pool.release(grad_handle);
    for (inp, outp) in intermediates {
        pool.release(inp);
        pool.release(outp);
    }

    // ==================== Diff Smooth Loss ====================
    println!("\n--- Diff Smooth Loss (2D) ---");
    let error_map = vec![1.0, 0.5, 0.2, 0.9];
    // Горизонтальные пары
    let mut horiz_input = Mat::zeros(2, 2);
    horiz_input[(0, 0)] = error_map[0];
    horiz_input[(0, 1)] = error_map[1];
    horiz_input[(1, 0)] = error_map[2];
    horiz_input[(1, 1)] = error_map[3];

    let horiz_handle = mat_to_handle(&horiz_input, &mut pool);
    let (loss_vec, intermediates_h) = smooth_h_expr.forward_chunk_buffered(&horiz_handle, &mut pool);
    let h_val = smooth_h_expr.aggregate_loss(&loss_vec);

    let grad_loss = vec![1.0f32; 2];
    let grad_handle_h = smooth_h_expr.backward_chunk_buffered(&intermediates_h, &grad_loss, &mut pool);
    let grad_mat_h = handle_to_mat(&grad_handle_h);
    let mut grad_h = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            grad_h.push(grad_mat_h[(i, j)]);
        }
    }
    println!("Smooth H loss: {:.6}, grad: {:?}", h_val, grad_h);

    pool.release(horiz_handle);
    pool.release(grad_handle_h);
    for (inp, outp) in intermediates_h {
        pool.release(inp);
        pool.release(outp);
    }

    // Вертикальные пары
    let mut vert_input = Mat::zeros(2, 2);
    vert_input[(0, 0)] = error_map[0];
    vert_input[(0, 1)] = error_map[2];
    vert_input[(1, 0)] = error_map[1];
    vert_input[(1, 1)] = error_map[3];

    let vert_handle = mat_to_handle(&vert_input, &mut pool);
    let (loss_vec, intermediates_v) = smooth_v_expr.forward_chunk_buffered(&vert_handle, &mut pool);
    let v_val = smooth_v_expr.aggregate_loss(&loss_vec);

    let grad_handle_v = smooth_v_expr.backward_chunk_buffered(&intermediates_v, &grad_loss, &mut pool);
    let grad_mat_v = handle_to_mat(&grad_handle_v);
    let mut grad_v = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            grad_v.push(grad_mat_v[(i, j)]);
        }
    }
    println!("Smooth V loss: {:.6}, grad: {:?}", v_val, grad_v);

    pool.release(vert_handle);
    pool.release(grad_handle_v);
    for (inp, outp) in intermediates_v {
        pool.release(inp);
        pool.release(outp);
    }

    let total_smooth = h_val + v_val;
    println!("Total smooth loss: {:.6}", total_smooth);
}




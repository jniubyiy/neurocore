// examples/loss2d_test.rs
// Тестирование потерь: MSE, CrossEntropy, difference_loss, diff_smooth_loss.
// Использует матричный API LossExpr.

use faer::Mat;
use neurocore::create_losses;

// -------- Область потерь --------
mod losses {
    use neurocore::loss_plan::{
        Aggregation, CrossEntropyWithLogits, ElementChain, LossDesc, Square, Sub,
    };
    use neurocore::loss_plan::{Abs, AddScalar, Log1p, AbsDiff};

    pub fn mse() -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(Sub))
            .add(Box::new(Square));
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
            .add(Box::new(Sub))             // p - t
            .add(Box::new(Abs))             // |p-t|
            .add(Box::new(AddScalar(1.0)))  // 1 + |p-t|
            .add(Box::new(Log1p));          // ln(1 + |p-t|)
        LossDesc::from_chain(chain, Aggregation::Mean, 4, 1, 1)
    }

    pub fn diff_smooth_loss_h() -> LossDesc {
        // горизонтальная составляющая: mean(|p_i - p_{i+1}|)
        let chain = ElementChain::new()
            .add(Box::new(AbsDiff));
        LossDesc::from_chain(chain, Aggregation::Mean, 2, 1, 1) // 2 пары соседей по горизонтали
    }

    pub fn diff_smooth_loss_v() -> LossDesc {
        // вертикальная составляющая: mean(|p_i - p_{i+W}|)
        let chain = ElementChain::new()
            .add(Box::new(AbsDiff));
        LossDesc::from_chain(chain, Aggregation::Mean, 2, 1, 1) // 2 пары соседей по вертикали
    }
}

fn main() {
    let (mse,) = create_losses!(losses::mse);
    let (ce,) = create_losses!(losses::cross_entropy);
    let (diff_loss,) = create_losses!(losses::difference_loss);
    let (smooth_h,) = create_losses!(losses::diff_smooth_loss_h);
    let (smooth_v,) = create_losses!(losses::diff_smooth_loss_v);

    // ==================== MSE ====================
    println!("--- MSE (2D) ---");
    let total_elements = 4;
    let pred = vec![1.0f32, 2.0, 3.0, 4.0];
    let target = vec![1.5, 1.5, 3.5, 4.5];

    let in_size = mse.task_input_size(); // = 2
    let mut full_input = Mat::zeros(total_elements, in_size);
    for i in 0..total_elements {
        full_input[(i, 0)] = pred[i];
        full_input[(i, 1)] = target[i];
    }

    let (loss_vec, intermediates) = mse.forward_chunk(&full_input);
    let loss = mse.aggregate_loss(&loss_vec);
    println!("MSE loss: {:.6}", loss);

    let grad_loss = vec![1.0f32; total_elements];
    let grad_mat = mse.backward_chunk(&intermediates, &grad_loss);
    let mut grad_flat = Vec::with_capacity(total_elements * in_size);
    for i in 0..total_elements {
        for j in 0..in_size {
            grad_flat.push(grad_mat[(i, j)]);
        }
    }
    let grad_agg = mse.aggregate_grad(&grad_flat);
    let grad: Vec<f32> = (0..total_elements)
        .map(|i| grad_agg[i * in_size]) // градиент по pred (первый элемент)
        .collect();
    println!("MSE grad: {:?}", grad);

    // ==================== CrossEntropy ====================
    println!("\n--- CrossEntropy (2D) ---");
    let pred_logits = vec![0.2f32, 0.5, 0.1, 0.2];
    let class_index = 1.0f32; // целевой класс 1

    let in_size = ce.task_input_size(); // num_classes + 1 = 5
    let mut ce_input = Mat::zeros(1, in_size);
    ce_input[(0, 0)] = pred_logits[0];
    ce_input[(0, 1)] = pred_logits[1];
    ce_input[(0, 2)] = pred_logits[2];
    ce_input[(0, 3)] = pred_logits[3];
    ce_input[(0, 4)] = class_index;

    let (loss_vec, intermediates) = ce.forward_chunk(&ce_input);
    let ce_loss = ce.aggregate_loss(&loss_vec);
    println!("CE loss: {:.6}", ce_loss);

    let grad_loss = vec![1.0f32; 1];
    let grad_mat = ce.backward_chunk(&intermediates, &grad_loss);
    let mut grad_flat = Vec::with_capacity(in_size);
    for j in 0..in_size {
        grad_flat.push(grad_mat[(0, j)]);
    }
    let grad_agg = ce.aggregate_grad(&grad_flat);
    // Показываем градиент только по логитам (первые num_classes элементов)
    println!("CE grad (first {}): {:?}", 4, &grad_agg[..4]);

    // ==================== Difference Loss ====================
    println!("\n--- Difference Loss (2D) ---");
    let pred = vec![1.0f32, 2.0, 3.0, 4.0];
    let target = vec![1.5, 1.5, 3.5, 4.5];

    let in_size = diff_loss.task_input_size(); // = 2
    let mut full_input = Mat::zeros(total_elements, in_size);
    for i in 0..total_elements {
        full_input[(i, 0)] = pred[i];
        full_input[(i, 1)] = target[i];
    }

    let (loss_vec, intermediates) = diff_loss.forward_chunk(&full_input);
    let loss = diff_loss.aggregate_loss(&loss_vec);
    println!("Diff loss: {:.6}", loss);

    let grad_loss = vec![1.0f32; total_elements];
    let grad_mat = diff_loss.backward_chunk(&intermediates, &grad_loss);
    let mut grad_flat = Vec::with_capacity(total_elements * in_size);
    for i in 0..total_elements {
        for j in 0..in_size {
            grad_flat.push(grad_mat[(i, j)]);
        }
    }
    let grad_agg = diff_loss.aggregate_grad(&grad_flat);
    let grad: Vec<f32> = (0..total_elements)
        .map(|i| grad_agg[i * in_size]) // градиент по pred
        .collect();
    println!("Diff grad: {:?}", grad);

    // ==================== Diff Smooth Loss ====================
    println!("\n--- Diff Smooth Loss (2D) ---");
    // Представим карту ошибок размером 2x2 (4 пикселя)
    let error_map = vec![1.0, 0.5, 0.2, 0.9]; // 2x2 row-major
    // Горизонтальные пары: (e[0], e[1]), (e[2], e[3])
    let in_size = smooth_h.task_input_size(); // 2
    let mut horiz_input = Mat::zeros(2, in_size); // 2 пары
    horiz_input[(0, 0)] = error_map[0];
    horiz_input[(0, 1)] = error_map[1];
    horiz_input[(1, 0)] = error_map[2];
    horiz_input[(1, 1)] = error_map[3];

    let (loss_vec, _intermediates) = smooth_h.forward_chunk(&horiz_input);
    let h_val = smooth_h.aggregate_loss(&loss_vec);

    // Вертикальные пары: (e[0], e[2]), (e[1], e[3])
    let mut vert_input = Mat::zeros(2, in_size);
    vert_input[(0, 0)] = error_map[0];
    vert_input[(0, 1)] = error_map[2];
    vert_input[(1, 0)] = error_map[1];
    vert_input[(1, 1)] = error_map[3];

    let (loss_vec, intermediates) = smooth_v.forward_chunk(&vert_input);
    let v_val = smooth_v.aggregate_loss(&loss_vec);

    let total_smooth = h_val + v_val;
    println!(
        "Smooth loss: {:.6} (horiz={:.6}, vert={:.6})",
        total_smooth, h_val, v_val
    );

    // Градиенты гладкости (показываем для наглядности)
    let grad_loss = vec![1.0f32; 2];
    let grad_mat_h = smooth_h.backward_chunk(&intermediates, &grad_loss);
    let mut grad_h_flat = Vec::with_capacity(2 * in_size);
    for i in 0..2 {
        for j in 0..in_size {
            grad_h_flat.push(grad_mat_h[(i, j)]);
        }
    }
    let grad_h_agg = smooth_h.aggregate_grad(&grad_h_flat);
    println!("Smooth H grad: {:?}", grad_h_agg);

    // вертикальные градиенты пересчитаем заново
    let (_, intermediates_v) = smooth_v.forward_chunk(&vert_input);
    let grad_mat_v = smooth_v.backward_chunk(&intermediates_v, &grad_loss);
    let mut grad_v_flat = Vec::with_capacity(2 * in_size);
    for i in 0..2 {
        for j in 0..in_size {
            grad_v_flat.push(grad_mat_v[(i, j)]);
        }
    }
    let grad_v_agg = smooth_v.aggregate_grad(&grad_v_flat);
    println!("Smooth V grad: {:?}", grad_v_agg);
}





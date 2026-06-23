// examples/loss_test.rs
// Тестирование потерь: MSE, CrossEntropy, difference_loss, diff_smooth_loss.

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
            .add(Box::new(Sub))
            .add(Box::new(Abs))
            .add(Box::new(AddScalar(1.0)))
            .add(Box::new(Log1p));
        LossDesc::from_chain(chain, Aggregation::Mean, 4, 1, 1)
    }

    pub fn diff_smooth_loss_h() -> LossDesc {
        // горизонтальная составляющая: mean(|p_i - p_{i+1}|)
        let chain = ElementChain::new()
            .add(Box::new(AbsDiff));
        LossDesc::from_chain(chain, Aggregation::Mean, 3, 1, 1) // 3 пары соседей
    }

    pub fn diff_smooth_loss_v() -> LossDesc {
        // вертикальная составляющая
        let chain = ElementChain::new()
            .add(Box::new(AbsDiff));
        LossDesc::from_chain(chain, Aggregation::Mean, 2, 1, 1) // 2 пары соседей
    }
}

fn main() {
    let (mse,) = create_losses!(losses::mse);
    let (ce,) = create_losses!(losses::cross_entropy);
    let (diff_loss,) = create_losses!(losses::difference_loss);
    let (smooth_h,) = create_losses!(losses::diff_smooth_loss_h);
    let (smooth_v,) = create_losses!(losses::diff_smooth_loss_v);

    // ==================== MSE ====================
    println!("--- MSE ---");
    let total_elements = 4;
    let pred = vec![1.0f32, 2.0, 3.0, 4.0];
    let target = vec![1.5, 1.5, 3.5, 4.5];

    let in_size = mse.task_input_size();
    let mut flat_input = Vec::with_capacity(total_elements * in_size);
    for i in 0..total_elements {
        flat_input.push(pred[i]);
        flat_input.push(target[i]);
    }
    let mut out_loss = vec![0.0; total_elements];
    mse.forward_chunk(0, total_elements, &flat_input, &mut out_loss);
    let loss = mse.aggregate_loss(&out_loss);
    println!("MSE loss: {:.6}", loss);

    let mut grad_pred = vec![0.0; total_elements * in_size];
    let grad_loss = vec![1.0; total_elements];
    mse.backward_chunk(0, total_elements, &flat_input, &out_loss, &grad_loss, &mut grad_pred);
    let grad_flat = mse.aggregate_grad(&grad_pred);
    let grad: Vec<f32> = (0..total_elements)
        .map(|i| grad_flat[i * in_size])
        .collect();
    println!("MSE grad: {:?}", grad);

    // ==================== CrossEntropy ====================
    println!("\n--- CrossEntropy ---");
    let pred_logits = vec![0.2f32, 0.5, 0.1, 0.2];
    let class_index = 1.0f32;

    let in_size = ce.task_input_size();
    let mut flat_input = Vec::with_capacity(1 * in_size);
    flat_input.extend_from_slice(&pred_logits);
    flat_input.push(class_index);

    let mut out_loss = vec![0.0; 1];
    ce.forward_chunk(0, 1, &flat_input, &mut out_loss);
    let ce_loss = ce.aggregate_loss(&out_loss);
    println!("CE loss: {:.6}", ce_loss);

    let mut grad_input = vec![0.0; 1 * in_size];
    let grad_loss = vec![1.0; 1];
    ce.backward_chunk(0, 1, &flat_input, &out_loss, &grad_loss, &mut grad_input);
    let grad_flat = ce.aggregate_grad(&grad_input);
    println!("CE grad (first {}): {:?}", 4, &grad_flat[..4]);

    // ==================== Difference Loss ====================
    println!("\n--- Difference Loss (ln(1+|p-t|)) ---");
    let pred = vec![1.0f32, 2.0, 3.0, 4.0];
    let target = vec![1.5, 1.5, 3.5, 4.5];

    let in_size = diff_loss.task_input_size();
    let mut flat_input = Vec::with_capacity(total_elements * in_size);
    for i in 0..total_elements {
        flat_input.push(pred[i]);
        flat_input.push(target[i]);
    }
    let mut out_loss = vec![0.0; total_elements];
    diff_loss.forward_chunk(0, total_elements, &flat_input, &mut out_loss);
    let loss = diff_loss.aggregate_loss(&out_loss);
    println!("Diff loss: {:.6}", loss);

    let mut grad_pred = vec![0.0; total_elements * in_size];
    let grad_loss = vec![1.0; total_elements];
    diff_loss.backward_chunk(0, total_elements, &flat_input, &out_loss, &grad_loss, &mut grad_pred);
    let grad_flat = diff_loss.aggregate_grad(&grad_pred);
    let grad: Vec<f32> = (0..total_elements)
        .map(|i| grad_flat[i * in_size])
        .collect();
    println!("Diff grad: {:?}", grad);

    // ==================== Diff Smooth Loss ====================
    println!("\n--- Diff Smooth Loss (spatial TV) ---");
    // Представим карту ошибок размером 2x2 (4 пикселя)
    let error_map = vec![1.0, 0.5, 0.2, 0.9]; // 2x2 row-major
    // Горизонтальные пары: (e[0], e[1]), (e[2], e[3])
    let mut horiz_input = Vec::with_capacity(3 * smooth_h.task_input_size()); // 2 пары
    horiz_input.push(error_map[0]); horiz_input.push(error_map[1]);
    horiz_input.push(error_map[2]); horiz_input.push(error_map[3]);
    let mut h_loss = vec![0.0; 2];
    smooth_h.forward_chunk(0, 2, &horiz_input, &mut h_loss);
    let h_val = smooth_h.aggregate_loss(&h_loss);

    // Вертикальные пары: (e[0], e[2]), (e[1], e[3])
    let mut vert_input = Vec::with_capacity(2 * smooth_v.task_input_size());
    vert_input.push(error_map[0]); vert_input.push(error_map[2]);
    vert_input.push(error_map[1]); vert_input.push(error_map[3]);
    let mut v_loss = vec![0.0; 2];
    smooth_v.forward_chunk(0, 2, &vert_input, &mut v_loss);
    let v_val = smooth_v.aggregate_loss(&v_loss);

    let total_smooth = h_val + v_val;
    println!("Smooth loss: {:.6} (horiz={:.6}, vert={:.6})", total_smooth, h_val, v_val);

    // Градиенты (здесь для краткости только по одному примеру)
    let mut grad_h = vec![0.0; 2 * smooth_h.task_input_size()];
    let grad_loss = vec![1.0; 2];
    smooth_h.backward_chunk(0, 2, &horiz_input, &h_loss, &grad_loss, &mut grad_h);
    let grad_h_flat = smooth_h.aggregate_grad(&grad_h);
    println!("Smooth H grad: {:?}", grad_h_flat);

    let mut grad_v = vec![0.0; 2 * smooth_v.task_input_size()];
    smooth_v.backward_chunk(0, 2, &vert_input, &v_loss, &grad_loss, &mut grad_v);
    let grad_v_flat = smooth_v.aggregate_grad(&grad_v);
    println!("Smooth V grad: {:?}", grad_v_flat);
}





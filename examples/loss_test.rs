// examples/loss_test.rs
// Тестирование функций потерь для Dim1: MSE, CrossEntropy, difference_loss, diff_smooth_loss.
// Использует матричный API LossExpr.

use faer::Mat;
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

fn main() {
    let mse_expr = losses::mse().build();
    let ce_expr = losses::cross_entropy().build();
    let diff_loss_expr = losses::difference_loss().build();
    let smooth_h_expr = losses::diff_smooth_loss_h().build();
    let smooth_v_expr = losses::diff_smooth_loss_v().build();

    // ==================== MSE ====================
    println!("--- MSE (1D) ---");
    let pred = vec![1.0f32, 2.0, 3.0, 4.0];
    let target = vec![1.5, 1.5, 3.5, 4.5];

    let in_size = mse_expr.task_input_size(); // = 1+1 = 2
    let mut full_input = Mat::zeros(4, in_size);
    for i in 0..4 {
        full_input[(i, 0)] = pred[i];
        full_input[(i, 1)] = target[i];
    }

    let (loss_vec, intermediates) = mse_expr.forward_chunk(&full_input);
    let loss = mse_expr.aggregate_loss(&loss_vec);
    println!("MSE loss: {:.6}", loss);

    let grad_loss = vec![1.0f32; 4];
    let grad_mat = mse_expr.backward_chunk(&intermediates, &grad_loss);
    let grad_pred: Vec<f32> = (0..4).map(|i| grad_mat[(i, 0)]).collect();
    println!("MSE grad: {:?}", grad_pred);

    // ==================== CrossEntropy ====================
    println!("\n--- CrossEntropy (1D) ---");
    let pred_logits = vec![0.2f32, 0.5, 0.1, 0.2];
    let class_index = 1.0f32;

    let in_size = ce_expr.task_input_size(); // 4+1 = 5
    let mut ce_input = Mat::zeros(1, in_size);
    ce_input[(0, 0)] = pred_logits[0];
    ce_input[(0, 1)] = pred_logits[1];
    ce_input[(0, 2)] = pred_logits[2];
    ce_input[(0, 3)] = pred_logits[3];
    ce_input[(0, 4)] = class_index;

    let (loss_vec, intermediates) = ce_expr.forward_chunk(&ce_input);
    let ce_loss = ce_expr.aggregate_loss(&loss_vec);
    println!("CE loss: {:.6}", ce_loss);

    let grad_loss = vec![1.0f32; 1];
    let grad_mat = ce_expr.backward_chunk(&intermediates, &grad_loss);
    let grad_ce: Vec<f32> = (0..4).map(|j| grad_mat[(0, j)]).collect();
    println!("CE grad (first 4): {:?}", grad_ce);

    // ==================== Difference Loss ====================
    println!("\n--- Difference Loss (1D) ---");
    let mut full_input_diff = Mat::zeros(4, 2);
    for i in 0..4 {
        full_input_diff[(i, 0)] = pred[i];
        full_input_diff[(i, 1)] = target[i];
    }

    let (loss_vec, intermediates) = diff_loss_expr.forward_chunk(&full_input_diff);
    let diff_loss_val = diff_loss_expr.aggregate_loss(&loss_vec);
    println!("Diff loss: {:.6}", diff_loss_val);

    let grad_loss = vec![1.0f32; 4];
    let grad_mat = diff_loss_expr.backward_chunk(&intermediates, &grad_loss);
    let grad_diff: Vec<f32> = (0..4).map(|i| grad_mat[(i, 0)]).collect();
    println!("Diff grad: {:?}", grad_diff);

    // ==================== Diff Smooth Loss ====================
    println!("\n--- Diff Smooth Loss (1D) ---");
    let error_map = vec![1.0, 0.5, 0.2, 0.9];
    // Горизонтальные пары
    let mut horiz_input = Mat::zeros(2, 2);
    horiz_input[(0, 0)] = error_map[0];
    horiz_input[(0, 1)] = error_map[1];
    horiz_input[(1, 0)] = error_map[2];
    horiz_input[(1, 1)] = error_map[3];

    let (loss_vec, _) = smooth_h_expr.forward_chunk(&horiz_input);
    let h_val = smooth_h_expr.aggregate_loss(&loss_vec);

    // Вертикальные пары
    let mut vert_input = Mat::zeros(2, 2);
    vert_input[(0, 0)] = error_map[0];
    vert_input[(0, 1)] = error_map[2];
    vert_input[(1, 0)] = error_map[1];
    vert_input[(1, 1)] = error_map[3];

    let (loss_vec, intermediates) = smooth_v_expr.forward_chunk(&vert_input);
    let v_val = smooth_v_expr.aggregate_loss(&loss_vec);

    println!("Smooth loss: {:.6} (horiz={:.6}, vert={:.6})", h_val + v_val, h_val, v_val);

    let grad_loss = vec![1.0f32; 2];
    let grad_mat_h = smooth_h_expr.backward_chunk(&intermediates, &grad_loss);
    // избегаем замыканий с перемещением
    let mut grad_h = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            grad_h.push(grad_mat_h[(i, j)]);
        }
    }
    println!("Smooth H grad: {:?}", grad_h);

    let (_loss_vec_v, intermediates_v) = smooth_v_expr.forward_chunk(&vert_input);
    let grad_mat_v = smooth_v_expr.backward_chunk(&intermediates_v, &grad_loss);
    let mut grad_v = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            grad_v.push(grad_mat_v[(i, j)]);
        }
    }
    println!("Smooth V grad: {:?}", grad_v);
}





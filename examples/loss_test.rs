use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::dispatchers::single::loss::dim1d::SingleLoss1D;
use neurocore::dispatchers::single::loss::dim2d::SingleLoss2D;
use neurocore::dispatchers::common::model_trait::LossDispatch;
use neurocore::tensor::Tensor1D;
use neurocore::tensor::Tensor2D;
use neurocore::jacobian::Jacobian;
use neurocore::jacobian::Jacobian2D;
use std::time::Instant;

fn main() {
    // --- MSE 1D ---
    let pred = Tensor1D::new(vec![1.0, 2.0, 3.0, 4.0]);
    let target = Tensor1D::new(vec![1.5, 1.5, 3.5, 4.5]);
    let mut j_pred = Jacobian::new(4, 2);
    for i in 0..4 {
        j_pred.data[i][0] = 1.0;
        j_pred.data[i][1] = 2.0;
    }

    let mse_plan = LossPlan::new(LossBlueprint::mse()).unwrap();
    let built_mse = mse_plan.build(4, 4).unwrap();
    let single1d = SingleLoss1D::new();

    let start = Instant::now();
    let (loss_single, _) = single1d.compute_loss(&pred, &target, &j_pred, &built_mse);
    println!("MSE 1D: loss={:.6}, time={:?}", loss_single, start.elapsed());

    // --- CrossEntropy 1D ---
    let logits = Tensor1D::new(vec![0.2, 0.5, 0.1, 0.2]);   // 4 класса
    let t = Tensor1D::new(vec![1.0]);
    let mut j_logits = Jacobian::new(4, 1);
    j_logits.data[0][0] = 0.5;
    j_logits.data[1][0] = -0.2;
    j_logits.data[2][0] = 1.0;
    j_logits.data[3][0] = -0.5;

    let ce_plan = LossPlan::new(LossBlueprint::cross_entropy(4)).unwrap();
    let built_ce = ce_plan.build(4, 1).unwrap();

    let start = Instant::now();
    let (loss_ce, _) = single1d.compute_loss(&logits, &t, &j_logits, &built_ce);
    println!("CE 1D: loss={:.6}, time={:?}", loss_ce, start.elapsed());

    // --- CrossEntropy 2D ---
    let logits2d = Tensor2D::new(vec![vec![0.2, 0.5, 0.1, 0.2]]);
    let t2d = Tensor2D::new(vec![vec![1.0]]);
    let mut j_logits2d = Jacobian2D::new(1, 4, 1);
    j_logits2d.data[0][0][0] = 0.5;
    j_logits2d.data[0][1][0] = -0.2;
    j_logits2d.data[0][2][0] = 1.0;
    j_logits2d.data[0][3][0] = -0.5;

    let single2d = SingleLoss2D::new();
    let start = Instant::now();
    let (loss_ce2d, _) = single2d.compute_loss(&logits2d, &t2d, &j_logits2d, &built_ce);
    println!("CE 2D: loss={:.6}, time={:?}", loss_ce2d, start.elapsed());
}






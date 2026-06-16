use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::dispatchers::single::loss::dim4d::SingleLoss4D;
use neurocore::dispatchers::common::model_trait::LossDispatch;
use neurocore::tensor::Tensor4D;
use neurocore::jacobian::Jacobian4D;
use std::time::Instant;

fn main() {
    let pred = Tensor4D::new(vec![vec![vec![vec![1.0, 2.0, 3.0, 4.0]]]]);
    let target = Tensor4D::new(vec![vec![vec![vec![1.0, 2.0, 3.0, 4.0]]]]);
    let mut j = Jacobian4D::new(1, 1, 1, 4, 1);
    j.data[0][0][0][0][0] = 1.0;

    let mse_plan = LossPlan::new(LossBlueprint::mse()).unwrap();
    let built = mse_plan.build(4, 4).unwrap();
    let single = SingleLoss4D::new();

    let start = Instant::now();
    let (loss_s, _) = single.compute_loss(&pred, &target, &j, &built);
    println!("MSE 4D: loss={:.6}, time={:?}", loss_s, start.elapsed());

    // CrossEntropy 4D
    let logits = Tensor4D::new(vec![vec![vec![vec![0.2, 0.5, 0.1, 0.2]]]]);
    let target_ce = Tensor4D::new(vec![vec![vec![vec![1.0]]]]);
    let mut jl = Jacobian4D::new(1, 1, 1, 4, 1);
    jl.data[0][0][0][0][0] = 1.0;

    let ce_plan = LossPlan::new(LossBlueprint::cross_entropy(4)).unwrap();
    let built_ce = ce_plan.build(4, 1).unwrap();

    let start = Instant::now();
    let (loss_ce, _) = single.compute_loss(&logits, &target_ce, &jl, &built_ce);
    println!("CE 4D: loss={:.6}, time={:?}", loss_ce, start.elapsed());
}






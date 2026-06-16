use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::dispatchers::single::loss::dim2d::SingleLoss2D;
use neurocore::dispatchers::common::model_trait::LossDispatch;
use neurocore::tensor::Tensor2D;
use neurocore::jacobian::Jacobian2D;
use std::time::Instant;

fn main() {
    let pred = Tensor2D::new(vec![vec![1.0, 2.0, 3.0, 4.0]]);
    let target = Tensor2D::new(vec![vec![0.8, 1.2, 3.0, 4.0]]);
    let j = Jacobian2D::new(1, 4, 1);

    let mse_plan = LossPlan::new(LossBlueprint::mse()).unwrap();
    let built = mse_plan.build(4, 4).unwrap();
    let single = SingleLoss2D::new();

    let start = Instant::now();
    let (loss_s, _) = single.compute_loss(&pred, &target, &j, &built);
    println!("MSE 2D: loss={:.6}, time={:?}", loss_s, start.elapsed());
}






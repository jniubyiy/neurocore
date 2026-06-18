use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::tensor::Tensor1D;

fn main() {
    // MSE
    let plan = LossPlan::new(LossBlueprint::mse()).unwrap();
    let built = plan.build().unwrap();
    let pred = Tensor1D::new(vec![1.0, 2.0, 3.0, 4.0]);
    let target = Tensor1D::new(vec![1.5, 1.5, 3.5, 4.5]);
    let (loss, delta) = (built.forward)(&pred, &target);
    println!("MSE loss: {:.6}, delta: {:?}", loss, delta.data);

    // CrossEntropy
    let plan_ce = LossPlan::new(LossBlueprint::cross_entropy(4)).unwrap();
    let built_ce = plan_ce.build().unwrap();
    let logits = Tensor1D::new(vec![0.2, 0.5, 0.1, 0.2]);
    let class = Tensor1D::new(vec![1.0]); // target class 1
    let (loss_ce, delta_ce) = (built_ce.forward)(&logits, &class);
    println!("CE loss: {:.6}, delta: {:?}", loss_ce, delta_ce.data);
}






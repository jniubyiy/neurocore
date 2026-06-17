use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::tensor::Tensor2D;

fn main() {
    // MSE 2D
    let plan = LossPlan::new(LossBlueprint::mse()).unwrap();
    let built = plan.build_2d(4, 4).unwrap();
    let pred = Tensor2D::new(vec![vec![1.0, 2.0, 3.0, 4.0]]);
    let target = Tensor2D::new(vec![vec![1.0, 2.0, 3.0, 4.0]]);
    let (loss, delta) = (built.forward)(&pred, &target);
    println!("MSE 2D: loss={:.6}", loss);
    println!("Delta:\n{:?}", delta.data);

    // CrossEntropy 2D
    let ce_plan = LossPlan::new(LossBlueprint::cross_entropy(4)).unwrap();
    let built_ce = ce_plan.build_2d(4, 1).unwrap();
    let logits = Tensor2D::new(vec![vec![0.2, 0.5, 0.1, 0.2]]);
    let class = Tensor2D::new(vec![vec![1.0]]);
    let (loss_ce, delta_ce) = (built_ce.forward)(&logits, &class);
    println!("CE 2D: loss={:.6}", loss_ce);
    println!("Delta:\n{:?}", delta_ce.data);
}






use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::tensor::Tensor4D;

fn main() {
    // MSE 4D
    let plan = LossPlan::new(LossBlueprint::mse()).unwrap();
    let built = plan.build_4d(4, 4).unwrap();
    let pred = Tensor4D::new(vec![vec![vec![vec![1.0, 2.0, 3.0, 4.0]]]]);
    let target = Tensor4D::new(vec![vec![vec![vec![1.0, 2.0, 3.0, 4.0]]]]);
    let (loss, delta) = (built.forward)(&pred, &target);
    println!("MSE 4D: loss={:.6}", loss);
    println!("Delta: {:?}", delta.data);

    // CrossEntropy 4D
    let ce_plan = LossPlan::new(LossBlueprint::cross_entropy(4)).unwrap();
    let built_ce = ce_plan.build_4d(4, 1).unwrap();
    let logits = Tensor4D::new(vec![vec![vec![vec![0.2, 0.5, 0.1, 0.2]]]]);
    let class = Tensor4D::new(vec![vec![vec![vec![1.0]]]]);
    let (loss_ce, delta_ce) = (built_ce.forward)(&logits, &class);
    println!("CE 4D: loss={:.6}", loss_ce);
    println!("Delta: {:?}", delta_ce.data);
}






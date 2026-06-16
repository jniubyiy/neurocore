use neurocore::model_plan::{Plan, LayerBlueprint, Dim};
use neurocore::dispatchers::single::SingleModel2D;
use neurocore::dispatchers::single::loss::dim2d::SingleLoss2D;
use neurocore::dispatchers::common::model_trait::{Model2D, LossDispatch};
use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::tensor::Tensor2D;
use neurocore::jacobian::Jacobian2D;
use std::time::Instant;

fn main() {
    let plan = Plan::new(vec![
        LayerBlueprint::linear(Dim::Dim2, 4, 2),
    ]).expect("Ошибка архитектуры");

    let mut built = plan.build_2d();
    let total = built.store.len();
    for i in 0..total {
        built.store.set_param(i, 0.5);
    }

    let mut model = built.into_single_model();
    let input = Tensor2D::new(vec![vec![1.0, 2.0, 3.0, 4.0]]);
    let target = Tensor2D::new(vec![vec![0.8, 1.2]]);
    let j0 = Jacobian2D::new(1, 4, total);

    let loss_plan = LossPlan::new(LossBlueprint::mse()).unwrap();
    let built_loss = loss_plan.build(2, 2).unwrap();
    let loss = SingleLoss2D::new();

    println!("=== SingleModel2D ===");
    let start = Instant::now();
    for e in 0..1000 {
        let (pred, jp) = model.forward(&input, &j0);
        let (loss_val, grad) = loss.compute_loss(&pred, &target, &jp, &built_loss);
        model.update_params(0.01, &grad);
        if e % 200 == 0 { println!("  Epoch {}: loss={:.6}", e, loss_val); }
    }
    println!("  Time: {:?}", start.elapsed());
    let (pred, _) = model.forward(&input, &j0);
    println!("  Final prediction:\n  {:?}", pred.data);
}






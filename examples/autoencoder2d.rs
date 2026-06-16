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
        LayerBlueprint::sigmoid(Dim::Dim2, 2),
        LayerBlueprint::linear(Dim::Dim2, 2, 4),
    ]).expect("Ошибка архитектуры");

    let mut built = plan.build_2d();
    let total = built.store.len();
    for i in 0..total {
        built.store.set_param(i, 0.3);
    }

    let mut model = built.into_single_model();

    let input = Tensor2D::new(vec![vec![1.0, 2.0, 3.0, 4.0]]);
    let j0 = Jacobian2D::new(1, 4, total);

    let loss_plan = LossPlan::new(LossBlueprint::mse()).unwrap();
    let built_loss = loss_plan.build(4, 4).unwrap();
    let loss_single = SingleLoss2D::new();
    let lr = 0.1;

    println!("\n=== SingleModel2D ===");
    let start = Instant::now();
    for epoch in 0..500 {
        let (pred, j_pred) = model.forward(&input, &j0);
        let (loss, grad) = loss_single.compute_loss(&pred, &input, &j_pred, &built_loss);
        model.update_params(lr, &grad);
        if epoch % 100 == 0 { println!("  Epoch {}: loss={:.6}", epoch, loss); }
    }
    println!("  Time: {:?}", start.elapsed());
}



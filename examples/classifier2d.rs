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
        LayerBlueprint::linear(Dim::Dim2, 2, 2),
        LayerBlueprint::softmax(Dim::Dim2, 2),
    ]).expect("Ошибка архитектуры");

    let mut built = plan.build_2d();
    let total = built.store.len();
    for i in 0..total {
        built.store.set_param(i, 0.2);
    }

    let mut model = built.into_single_model();

    let x1 = Tensor2D::new(vec![vec![1.0, 2.0]]);
    let x2 = Tensor2D::new(vec![vec![2.0, 1.0]]);
    let y1 = Tensor2D::new(vec![vec![0.0]]);
    let y2 = Tensor2D::new(vec![vec![1.0]]);
    let j0 = Jacobian2D::new(1, 2, total);

    let loss_plan = LossPlan::new(LossBlueprint::cross_entropy(2)).unwrap();
    let built_loss = loss_plan.build(2, 1).unwrap();
    let loss = SingleLoss2D::new();

    println!("=== SingleModel2D ===");
    let start = Instant::now();
    for e in 0..200 {
        for (x, y) in [(&x1, &y1), (&x2, &y2)] {
            let (logits, jl) = model.forward(x, &j0);
            let (loss_val, grad) = loss.compute_loss(&logits, y, &jl, &built_loss);
            model.update_params(0.5, &grad);
        }
        if e % 50 == 0 { println!("  Epoch {}", e); }
    }
    println!("  Time: {:?}", start.elapsed());
}






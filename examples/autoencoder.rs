use neurocore::model_plan::{Plan, LayerBlueprint, Dim};
use neurocore::tensor::Tensor1D;
use neurocore::jacobian::Jacobian;
use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::dispatchers::single::loss::dim1d::SingleLoss1D;
use neurocore::dispatchers::common::model_trait::{Model1D, LossDispatch};

fn main() {
    let plan = Plan::new(vec![
        LayerBlueprint::linear(Dim::Dim1, 4, 2),
        LayerBlueprint::sigmoid(Dim::Dim1, 2),
        LayerBlueprint::linear(Dim::Dim1, 2, 4),
    ]).expect("Ошибка архитектуры");

    let mut built = plan.build_1d();
    let total_params = built.store.len();
    for i in 0..total_params {
        built.store.set_param(i, 0.3);
    }

    let mut model = built.into_single_model();

    let x = Tensor1D::new(vec![1.0, 2.0, 3.0, 4.0]);
    let j0 = Jacobian::new(4, total_params);

    let loss_plan = LossPlan::new(LossBlueprint::mse()).unwrap();
    let built_loss = loss_plan.build(4, 4).unwrap();
    let loss_dispatch = SingleLoss1D::new();
    let lr = 0.1;

    for e in 0..500 {
        let (pred, j_pred) = model.forward(&x, &j0);
        let (loss, grad) = loss_dispatch.compute_loss(&pred, &x, &j_pred, &built_loss);
        model.update_params(lr, &grad);
        if e % 100 == 0 { println!("Epoch {}: loss={:.6}", e, loss); }
    }
    println!("Done");
}





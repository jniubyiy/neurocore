use neurocore::model_plan::{Plan, LayerBlueprint, Dim};
use neurocore::dispatchers::single::SingleModel1D;
use neurocore::dispatchers::auto::AutoModel1D;
use neurocore::dispatchers::trained::TrainedModel1D;
use neurocore::dispatchers::single::loss::dim1d::SingleLoss1D;
use neurocore::dispatchers::common::model_trait::{Model1D, LossDispatch};
use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::tensor::Tensor1D;

fn main() {
    let plan = Plan::new(vec![
        LayerBlueprint::linear(Dim::Dim1, 4, 2),
    ]).expect("Ошибка архитектуры");

    let loss_plan = LossPlan::new(LossBlueprint::mse()).unwrap();
    let built_loss = loss_plan.build(2, 2).unwrap();
    let loss_dispatch = SingleLoss1D::new();
    let x = Tensor1D::new(vec![1.0, 2.0, 3.0, 4.0]);
    let target = Tensor1D::new(vec![0.8, 1.5]);
    let lr = 0.01;

    // --- SingleModel1D ---
    println!("=== SingleModel1D ===");
    let mut built = plan.build_1d();
    let total = built.store.len();
    for i in 0..total { built.store.set_param(i, 0.5); }
    let mut model = built.into_single_model();

    for epoch in 0..500 {
        let pred = model.forward(&x);
        let (loss, delta) = loss_dispatch.compute_loss(&pred, &target, &built_loss);
        model.backward(&delta);
        model.update_params(lr);
        if epoch % 100 == 0 { println!("  Epoch {}: loss={:.6}", epoch, loss); }
    }
    println!("  Final prediction: {:?}", model.forward(&x).data);

    // --- AutoModel1D ---
    println!("\n=== AutoModel1D (потоков: 1) ===");
    let mut built2 = plan.build_1d();
    for i in 0..total { built2.store.set_param(i, 0.5); }
    let mut model2 = built2.into_auto_model(1);

    for epoch in 0..500 {
        let pred = model2.forward(&x);
        let (loss, delta) = loss_dispatch.compute_loss(&pred, &target, &built_loss);
        model2.backward(&delta);
        model2.update_params(lr);
        if epoch % 100 == 0 { println!("  Epoch {}: loss={:.6}", epoch, loss); }
    }
    println!("  Final prediction: {:?}", model2.forward(&x).data);

    // --- TrainedModel1D ---
    println!("\n=== TrainedModel1D (потоков: 1) ===");
    let mut built3 = plan.build_1d();
    for i in 0..total { built3.store.set_param(i, 0.5); }
    let mut model3 = built3.into_trained_model(1);

    for epoch in 0..500 {
        let pred = model3.forward(&x);
        let (loss, delta) = loss_dispatch.compute_loss(&pred, &target, &built_loss);
        model3.backward(&delta);
        model3.update_params(lr);
        if epoch % 100 == 0 { println!("  Epoch {}: loss={:.6}", epoch, loss); }
    }
    println!("  Final prediction: {:?}", model3.forward(&x).data);
}





use neurocore::model_plan::{Plan, LayerBlueprint, Dim};
use neurocore::dispatchers::single::SingleModel3D;
use neurocore::dispatchers::auto::AutoModel3D;
use neurocore::dispatchers::trained::TrainedModel3D;
use neurocore::dispatchers::single::loss::dim3d::SingleLoss3D;
use neurocore::dispatchers::common::model_trait::{Model3D, LossDispatch3D};
use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::tensor::Tensor3D;

fn main() {
    let plan = Plan::new(vec![
        LayerBlueprint::linear(Dim::Dim3, 4, 2),
        LayerBlueprint::sigmoid(Dim::Dim3, 2),
        LayerBlueprint::linear(Dim::Dim3, 2, 4),
    ]).expect("Ошибка архитектуры");

    let loss_plan = LossPlan::new(LossBlueprint::mse()).unwrap();
    let built_loss = loss_plan.build_3d(4, 4).unwrap();
    let loss_dispatch = SingleLoss3D::new();
    let input = Tensor3D::new(vec![vec![vec![1.0, 2.0, 3.0, 4.0]]]);
    let target = input.clone();
    let lr = 0.1;

    // --- SingleModel3D ---
    println!("=== SingleModel3D ===");
    let mut built = plan.build_3d();
    let total = built.store.len();
    for i in 0..total { built.store.set_param(i, 0.3); }
    let mut model = built.into_single_model();

    for e in 0..500 {
        let (pred, contexts) = model.forward(&input);
        let (loss, delta) = loss_dispatch.compute_loss(&pred, &target, &built_loss);
        let (_, all_grads) = model.backward(&contexts, &delta);
        model.update_params(lr, &all_grads);
        if e % 100 == 0 { println!("  Epoch {}: loss={:.6}", e, loss); }
    }
    println!("  Done.");

    // --- AutoModel3D (4 потока) ---
    println!("\n=== AutoModel3D (4 потока) ===");
    let mut built2 = plan.build_3d();
    for i in 0..total { built2.store.set_param(i, 0.3); }
    let mut model2 = built2.into_auto_model(4);

    for e in 0..500 {
        let (pred, contexts) = model2.forward(&input);
        let (loss, delta) = loss_dispatch.compute_loss(&pred, &target, &built_loss);
        let (_, all_grads) = model2.backward(&contexts, &delta);
        model2.update_params(lr, &all_grads);
        if e % 100 == 0 { println!("  Epoch {}: loss={:.6}", e, loss); }
    }
    println!("  Done.");

    // --- TrainedModel3D (4 потока) ---
    println!("\n=== TrainedModel3D (4 потока) ===");
    let mut built3 = plan.build_3d();
    for i in 0..total { built3.store.set_param(i, 0.3); }
    let mut model3 = built3.into_trained_model(4);

    for e in 0..500 {
        let (pred, contexts) = model3.forward(&input);
        let (loss, delta) = loss_dispatch.compute_loss(&pred, &target, &built_loss);
        let (_, all_grads) = model3.backward(&contexts, &delta);
        model3.update_params(lr, &all_grads);
        if e % 100 == 0 { println!("  Epoch {}: loss={:.6}", e, loss); }
    }
    println!("  Done.");
}





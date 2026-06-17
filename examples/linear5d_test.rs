use neurocore::model_plan::{Plan, LayerBlueprint, Dim};
use neurocore::dispatchers::single::SingleModel5D;
use neurocore::dispatchers::auto::AutoModel5D;
use neurocore::dispatchers::trained::TrainedModel5D;
use neurocore::dispatchers::single::loss::dim5d::SingleLoss5D;
use neurocore::dispatchers::common::model_trait::{Model5D, LossDispatch5D};
use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::tensor::Tensor5D;

fn main() {
    let plan = Plan::new(vec![
        LayerBlueprint::linear(Dim::Dim5, 4, 2),
    ]).expect("Ошибка архитектуры");

    let loss_plan = LossPlan::new(LossBlueprint::mse()).unwrap();
    let built_loss = loss_plan.build_5d(2, 2).unwrap();
    let loss_dispatch = SingleLoss5D::new();
    let input = Tensor5D::new(vec![vec![vec![vec![vec![1.0, 2.0, 3.0, 4.0]]]]]);
    let target = Tensor5D::new(vec![vec![vec![vec![vec![0.8, 1.2]]]]]);
    let lr = 0.01;

    // --- SingleModel5D ---
    println!("=== SingleModel5D ===");
    let mut built = plan.build_5d();
    let total = built.store.len();
    for i in 0..total { built.store.set_param(i, 0.5); }
    let mut model = built.into_single_model();

    for e in 0..1000 {
        let (pred, contexts) = model.forward(&input);
        let (loss, delta) = loss_dispatch.compute_loss(&pred, &target, &built_loss);
        let (_, all_grads) = model.backward(&contexts, &delta);
        model.update_params(lr, &all_grads);
        if e % 200 == 0 { println!("  Epoch {}: loss={:.6}", e, loss); }
    }
    let (pred, _) = model.forward(&input);
    println!("  Final prediction: {:?}", pred.data);

    // --- AutoModel5D (4 потока) ---
    println!("\n=== AutoModel5D (4 потока) ===");
    let mut built2 = plan.build_5d();
    for i in 0..total { built2.store.set_param(i, 0.5); }
    let mut model2 = built2.into_auto_model(4);

    for e in 0..1000 {
        let (pred, contexts) = model2.forward(&input);
        let (loss, delta) = loss_dispatch.compute_loss(&pred, &target, &built_loss);
        let (_, all_grads) = model2.backward(&contexts, &delta);
        model2.update_params(lr, &all_grads);
        if e % 200 == 0 { println!("  Epoch {}: loss={:.6}", e, loss); }
    }
    let (pred, _) = model2.forward(&input);
    println!("  Final prediction: {:?}", pred.data);

    // --- TrainedModel5D (4 потока) ---
    println!("\n=== TrainedModel5D (4 потока) ===");
    let mut built3 = plan.build_5d();
    for i in 0..total { built3.store.set_param(i, 0.5); }
    let mut model3 = built3.into_trained_model(4);

    for e in 0..1000 {
        let (pred, contexts) = model3.forward(&input);
        let (loss, delta) = loss_dispatch.compute_loss(&pred, &target, &built_loss);
        let (_, all_grads) = model3.backward(&contexts, &delta);
        model3.update_params(lr, &all_grads);
        if e % 200 == 0 { println!("  Epoch {}: loss={:.6}", e, loss); }
    }
    let (pred, _) = model3.forward(&input);
    println!("  Final prediction: {:?}", pred.data);
}





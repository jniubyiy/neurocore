use neurocore::model_plan::{Plan, LayerBlueprint, Dim};
use neurocore::dispatchers::single::loss::dim2d::SingleLoss2D;
use neurocore::dispatchers::common::model_trait::{Model2D, LossDispatch2D};
use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::tensor::Tensor2D;
use std::time::Instant;

fn main() {
    let plan = Plan::new(vec![
        LayerBlueprint::linear(Dim::Dim2, 4, 2),
    ]).expect("Ошибка архитектуры");

    let loss_plan = LossPlan::new(LossBlueprint::mse()).unwrap();
    let built_loss = loss_plan.build_2d(2, 2).unwrap();
    let loss_dispatch = SingleLoss2D::new();
    let input = Tensor2D::new(vec![vec![1.0, 2.0, 3.0, 4.0]]);
    let target = Tensor2D::new(vec![vec![0.8, 1.2]]);
    let lr = 0.01;

    let built_tmp = plan.build_2d();
    let param_mem = built_tmp.store.len() * std::mem::size_of::<f32>();
    let buffer_mem = (4 + 2) * std::mem::size_of::<f32>();
    let total_mem = param_mem + buffer_mem;
    println!("Estimated peak memory: {} bytes ({:.2} KB)", total_mem, total_mem as f64 / 1024.0);

    let epochs = 1000;
    let mut loss_start = 0.0_f32;

    // --- SingleModel2D ---
    println!("\n=== SingleModel2D ===");
    let mut built = plan.build_2d();
    let total = built.store.len();
    for i in 0..total { built.store.set_param(i, 0.5); }
    let mut model = built.into_single_model();

    let start = Instant::now();
    for e in 0..epochs {
        let (pred, contexts) = model.forward(&input);
        let (loss, delta) = loss_dispatch.compute_loss(&pred, &target, &built_loss);
        let (_, all_grads) = model.backward(&contexts, &delta);
        model.update_params(lr, &all_grads);
        if e == 0 { loss_start = loss; }
        if e % 200 == 0 { println!("  Epoch {}: loss={:.6}", e, loss); }
    }
    let duration = start.elapsed();
    let (pred, _) = model.forward(&input);
    let (final_loss, _) = loss_dispatch.compute_loss(&pred, &target, &built_loss);
    println!("  Done. Time: {:?}", duration);
    println!("  Final loss: {:.6}", final_loss);
    if final_loss > 0.0 && loss_start > 0.0 {
        let rate = (loss_start / final_loss).ln() / epochs as f32;
        println!("  Convergence rate (avg log improvement per epoch): {:.6}", rate);
    }

    // --- AutoModel2D (4 потока) ---
    println!("\n=== AutoModel2D (4 потока) ===");
    let mut built2 = plan.build_2d();
    for i in 0..total { built2.store.set_param(i, 0.5); }
    let mut model2 = built2.into_auto_model(4);

    let start = Instant::now();
    for e in 0..epochs {
        let (pred, contexts) = model2.forward(&input);
        let (loss, delta) = loss_dispatch.compute_loss(&pred, &target, &built_loss);
        let (_, all_grads) = model2.backward(&contexts, &delta);
        model2.update_params(lr, &all_grads);
        if e == 0 { loss_start = loss; }
        if e % 200 == 0 { println!("  Epoch {}: loss={:.6}", e, loss); }
    }
    let duration = start.elapsed();
    let (pred, _) = model2.forward(&input);
    let (final_loss, _) = loss_dispatch.compute_loss(&pred, &target, &built_loss);
    println!("  Done. Time: {:?}", duration);
    println!("  Final loss: {:.6}", final_loss);
    if final_loss > 0.0 && loss_start > 0.0 {
        let rate = (loss_start / final_loss).ln() / epochs as f32;
        println!("  Convergence rate (avg log improvement per epoch): {:.6}", rate);
    }

    // --- TrainedModel2D (4 потока) ---
    println!("\n=== TrainedModel2D (4 потока) ===");
    let mut built3 = plan.build_2d();
    for i in 0..total { built3.store.set_param(i, 0.5); }
    let mut model3 = built3.into_trained_model(4);

    let start = Instant::now();
    for e in 0..epochs {
        let (pred, contexts) = model3.forward(&input);
        let (loss, delta) = loss_dispatch.compute_loss(&pred, &target, &built_loss);
        let (_, all_grads) = model3.backward(&contexts, &delta);
        model3.update_params(lr, &all_grads);
        if e == 0 { loss_start = loss; }
        if e % 200 == 0 { println!("  Epoch {}: loss={:.6}", e, loss); }
    }
    let duration = start.elapsed();
    let (pred, _) = model3.forward(&input);
    let (final_loss, _) = loss_dispatch.compute_loss(&pred, &target, &built_loss);
    println!("  Done. Time: {:?}", duration);
    println!("  Final loss: {:.6}", final_loss);
    if final_loss > 0.0 && loss_start > 0.0 {
        let rate = (loss_start / final_loss).ln() / epochs as f32;
        println!("  Convergence rate (avg log improvement per epoch): {:.6}", rate);
    }
}






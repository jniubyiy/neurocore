use neurocore::model_plan::{Plan, LayerBlueprint, Dim};
use neurocore::dispatchers::single::loss::dim1d::SingleLoss1D;
use neurocore::dispatchers::common::model_trait::{Model1D, LossDispatch};
use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::tensor::Tensor1D;
use std::time::Instant;

fn main() {
    let plan = Plan::new(vec![
        LayerBlueprint::linear(Dim::Dim1, 4, 2),
        LayerBlueprint::sigmoid(Dim::Dim1, 2),
        LayerBlueprint::linear(Dim::Dim1, 2, 4),
    ]).expect("Ошибка архитектуры");

    let loss_plan = LossPlan::new(LossBlueprint::mse()).unwrap();
    let built_loss = loss_plan.build(4, 4).unwrap();
    let loss_dispatch = SingleLoss1D::new();
    let x = Tensor1D::new(vec![1.0, 2.0, 3.0, 4.0]);
    let target = x.clone();
    let lr = 0.1;

    // Оценка памяти (байты): параметры + буферы
    let param_mem = plan.build_1d().store.len() * std::mem::size_of::<f32>();
    // Вход 4, скрытый 2, выход 4 – плюс временные буферы
    let buffer_mem = (4 + 2 + 4) * std::mem::size_of::<f32>() * 2; // вход-выход-промежуточные
    let total_mem = param_mem + buffer_mem;
    println!("Estimated peak memory: {} bytes ({:.2} KB)", total_mem, total_mem as f64 / 1024.0);

    let epochs = 500;
    let mut loss_start = 0.0_f32;

    // --- SingleModel1D ---
    println!("\n=== SingleModel1D ===");
    let mut built = plan.build_1d();
    let total = built.store.len();
    for i in 0..total { built.store.set_param(i, 0.3); }
    let mut model = built.into_single_model();

    let start = Instant::now();
    for e in 0..epochs {
        let pred = model.forward(&x);
        let (loss, delta) = loss_dispatch.compute_loss(&pred, &target, &built_loss);
        model.backward(&delta);
        model.update_params(lr);
        if e == 0 { loss_start = loss; }
        if e % 100 == 0 { println!("  Epoch {}: loss={:.6}", e, loss); }
    }
    let duration = start.elapsed();
    let final_pred = model.forward(&x);
    let (final_loss, _) = loss_dispatch.compute_loss(&final_pred, &target, &built_loss);
    println!("  Done. Time: {:?}", duration);
    println!("  Final loss: {:.6}", final_loss);
    if final_loss > 0.0 && loss_start > 0.0 {
        let rate = (loss_start / final_loss).ln() / epochs as f32;
        println!("  Convergence rate (avg log improvement per epoch): {:.6}", rate);
    }

    // --- AutoModel1D ---
    println!("\n=== AutoModel1D (потоков: 1) ===");
    let mut built2 = plan.build_1d();
    for i in 0..total { built2.store.set_param(i, 0.3); }
    let mut model2 = built2.into_auto_model(1);

    let start = Instant::now();
    for e in 0..epochs {
        let pred = model2.forward(&x);
        let (loss, delta) = loss_dispatch.compute_loss(&pred, &target, &built_loss);
        model2.backward(&delta);
        model2.update_params(lr);
        if e == 0 { loss_start = loss; }
        if e % 100 == 0 { println!("  Epoch {}: loss={:.6}", e, loss); }
    }
    let duration = start.elapsed();
    let final_pred = model2.forward(&x);
    let (final_loss, _) = loss_dispatch.compute_loss(&final_pred, &target, &built_loss);
    println!("  Done. Time: {:?}", duration);
    println!("  Final loss: {:.6}", final_loss);
    if final_loss > 0.0 && loss_start > 0.0 {
        let rate = (loss_start / final_loss).ln() / epochs as f32;
        println!("  Convergence rate (avg log improvement per epoch): {:.6}", rate);
    }

    // --- TrainedModel1D ---
    println!("\n=== TrainedModel1D (потоков: 1) ===");
    let mut built3 = plan.build_1d();
    for i in 0..total { built3.store.set_param(i, 0.3); }
    let mut model3 = built3.into_trained_model(1);

    let start = Instant::now();
    for e in 0..epochs {
        let pred = model3.forward(&x);
        let (loss, delta) = loss_dispatch.compute_loss(&pred, &target, &built_loss);
        model3.backward(&delta);
        model3.update_params(lr);
        if e == 0 { loss_start = loss; }
        if e % 100 == 0 { println!("  Epoch {}: loss={:.6}", e, loss); }
    }
    let duration = start.elapsed();
    let final_pred = model3.forward(&x);
    let (final_loss, _) = loss_dispatch.compute_loss(&final_pred, &target, &built_loss);
    println!("  Done. Time: {:?}", duration);
    println!("  Final loss: {:.6}", final_loss);
    if final_loss > 0.0 && loss_start > 0.0 {
        let rate = (loss_start / final_loss).ln() / epochs as f32;
        println!("  Convergence rate (avg log improvement per epoch): {:.6}", rate);
    }
}





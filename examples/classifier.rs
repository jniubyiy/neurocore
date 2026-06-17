use neurocore::model_plan::{Plan, LayerBlueprint, Dim};
use neurocore::dispatchers::single::loss::dim1d::SingleLoss1D;
use neurocore::dispatchers::common::model_trait::{Model1D, LossDispatch};
use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::tensor::Tensor1D;
use std::time::Instant;

fn main() {
    let plan = Plan::new(vec![
        LayerBlueprint::linear(Dim::Dim1, 2, 2),
        LayerBlueprint::softmax(Dim::Dim1, 2),
    ]).expect("Ошибка архитектуры");

    let loss_plan = LossPlan::new(LossBlueprint::cross_entropy(2)).unwrap();
    let built_loss = loss_plan.build(2, 1).unwrap();
    let loss_dispatch = SingleLoss1D::new();
    let x1 = Tensor1D::new(vec![1.0, 2.0]);
    let x2 = Tensor1D::new(vec![2.0, 1.0]);
    let y1 = Tensor1D::new(vec![0.0]);
    let y2 = Tensor1D::new(vec![1.0]);
    let lr = 0.5;

    let built_tmp = plan.build_1d();
    let param_mem = built_tmp.store.len() * std::mem::size_of::<f32>();
    let buffer_mem = (2 + 2) * std::mem::size_of::<f32>();
    let total_mem = param_mem + buffer_mem;
    println!("Estimated peak memory: {} bytes ({:.2} KB)", total_mem, total_mem as f64 / 1024.0);

    let epochs = 200;
    let mut loss_start = 0.0_f32;

    // --- SingleModel1D ---
    println!("\n=== SingleModel1D ===");
    let mut built = plan.build_1d();
    let total = built.store.len();
    for i in 0..total { built.store.set_param(i, 0.2); }
    let mut model = built.into_single_model();

    let start = Instant::now();
    for epoch in 0..epochs {
        for (x, y) in [(&x1, &y1), (&x2, &y2)] {
            let pred = model.forward(x);
            let (_, delta) = loss_dispatch.compute_loss(&pred, y, &built_loss);
            model.backward(&delta);
            model.update_params(lr);
        }
        if epoch % 40 == 0 {
            let pred = model.forward(&x1);
            let (loss, _) = loss_dispatch.compute_loss(&pred, &y1, &built_loss);
            if epoch == 0 { loss_start = loss; }
            println!("  Epoch {}: loss={:.6}", epoch, loss);
        }
    }
    let duration = start.elapsed();
    let pred = model.forward(&x1);
    let (final_loss, _) = loss_dispatch.compute_loss(&pred, &y1, &built_loss);
    println!("  Done. Time: {:?}", duration);
    println!("  Final loss: {:.6}", final_loss);
    if final_loss > 0.0 && loss_start > 0.0 {
        let rate = (loss_start / final_loss).ln() / epochs as f32;
        println!("  Convergence rate (avg log improvement per epoch): {:.6}", rate);
    }

    // --- AutoModel1D ---
    println!("\n=== AutoModel1D (потоков: 1) ===");
    let mut built2 = plan.build_1d();
    for i in 0..total { built2.store.set_param(i, 0.2); }
    let mut model2 = built2.into_auto_model(1);

    let start = Instant::now();
    for epoch in 0..epochs {
        for (x, y) in [(&x1, &y1), (&x2, &y2)] {
            let pred = model2.forward(x);
            let (_, delta) = loss_dispatch.compute_loss(&pred, y, &built_loss);
            model2.backward(&delta);
            model2.update_params(lr);
        }
        if epoch % 40 == 0 {
            let pred = model2.forward(&x1);
            let (loss, _) = loss_dispatch.compute_loss(&pred, &y1, &built_loss);
            if epoch == 0 { loss_start = loss; }
            println!("  Epoch {}: loss={:.6}", epoch, loss);
        }
    }
    let duration = start.elapsed();
    let pred = model2.forward(&x1);
    let (final_loss, _) = loss_dispatch.compute_loss(&pred, &y1, &built_loss);
    println!("  Done. Time: {:?}", duration);
    println!("  Final loss: {:.6}", final_loss);
    if final_loss > 0.0 && loss_start > 0.0 {
        let rate = (loss_start / final_loss).ln() / epochs as f32;
        println!("  Convergence rate (avg log improvement per epoch): {:.6}", rate);
    }

    // --- TrainedModel1D ---
    println!("\n=== TrainedModel1D (потоков: 1) ===");
    let mut built3 = plan.build_1d();
    for i in 0..total { built3.store.set_param(i, 0.2); }
    let mut model3 = built3.into_trained_model(1);

    let start = Instant::now();
    for epoch in 0..epochs {
        for (x, y) in [(&x1, &y1), (&x2, &y2)] {
            let pred = model3.forward(x);
            let (_, delta) = loss_dispatch.compute_loss(&pred, y, &built_loss);
            model3.backward(&delta);
            model3.update_params(lr);
        }
        if epoch % 40 == 0 {
            let pred = model3.forward(&x1);
            let (loss, _) = loss_dispatch.compute_loss(&pred, &y1, &built_loss);
            if epoch == 0 { loss_start = loss; }
            println!("  Epoch {}: loss={:.6}", epoch, loss);
        }
    }
    let duration = start.elapsed();
    let pred = model3.forward(&x1);
    let (final_loss, _) = loss_dispatch.compute_loss(&pred, &y1, &built_loss);
    println!("  Done. Time: {:?}", duration);
    println!("  Final loss: {:.6}", final_loss);
    if final_loss > 0.0 && loss_start > 0.0 {
        let rate = (loss_start / final_loss).ln() / epochs as f32;
        println!("  Convergence rate (avg log improvement per epoch): {:.6}", rate);
    }
}






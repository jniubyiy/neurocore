// examples/linear_test.rs

use std::sync::mpsc;
use std::thread;
use std::time::Instant;
use neurocore::compute_manager::{DevicePlan, DynamicTensor, MixedModel};
use neurocore::tensor::Tensor2D;

mod models {
    use neurocore::model_plan::{Dim, LayerDesc, LayerKind};
    pub fn linear_model() -> Vec<LayerDesc> {
        vec![LayerDesc::new("linear", LayerKind::Linear, Dim::Dim1)
            .input(Dim::Dim1, &[4])
            .output(Dim::Dim1, &[2])]
    }
}

mod losses {
    use neurocore::loss_plan::{Aggregation, ElementChain, LossDesc, Square, Sub};
    pub fn mse() -> LossDesc {
        let chain = ElementChain::new().add(Box::new(Sub)).add(Box::new(Square));
        LossDesc::from_chain(chain, Aggregation::Mean, 2, 1, 1)
    }
}

mod optimizers {
    use neurocore::optimizer_plan::{OptimizerDesc, OptCubeDesc};
    pub fn sgd() -> OptimizerDesc {
        OptimizerDesc::new()
            .add(OptCubeDesc::ScaleGradient(0.01))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

fn run_training(
    device_plan: DevicePlan,
    label: &str,
    initial_params: Option<&[f32]>,
) -> (Vec<f32>, f32) {
    if device_plan.gpu_id().is_some() {
        let device_plan = device_plan.clone();
        let label = label.to_string();
        let initial_params = initial_params.map(|p| p.to_vec());
        let (tx, rx) = mpsc::channel();
        thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(move || {
                let (params, loss) = run_training_inner(device_plan, &label, initial_params.as_deref());
                tx.send((params, loss)).ok();
            })
            .unwrap();
        return rx.recv().unwrap();
    }
    run_training_inner(device_plan, label, initial_params)
}

fn run_training_inner(
    device_plan: DevicePlan,
    label: &str,
    initial_params: Option<&[f32]>,
) -> (Vec<f32>, f32) {
    println!("\n===== {} =====", label);
    let mut model = match MixedModel::from_plan_with_device_plan(
        models::linear_model(),
        device_plan.clone(),
    ) {
        Ok(m) => m,
        Err(e) => {
            println!("[ERROR] Не удалось собрать модель: {}", e);
            return (vec![], 0.0);
        }
    };

    if let Some(p) = initial_params {
        model.param_store().lock().unwrap().set_all_params(p);
    } else {
        let mut store = model.param_store().lock().unwrap();
        for i in 0..store.len() {
            store.set_param(i, rand::random::<f32>() * 0.01);
        }
    }

    let x = Tensor2D::new(vec![vec![1.0, 2.0, 3.0, 4.0]]);
    let target = Tensor2D::new(vec![vec![0.8, 1.5]]);
    let epochs = 500;

    let start = Instant::now();
    for epoch in 0..epochs {
        let (pred, ctxs) = model.forward(DynamicTensor::Dim1(x.clone()));
        let (loss, delta) = model.compute_loss(
            losses::mse(),
            &pred,
            &DynamicTensor::Dim1(target.clone()),
        );
        let (_, grads) = model.backward(&ctxs, delta);
        model.update_params(optimizers::sgd(), &grads[0]);

        if epoch == 0 || epoch % 100 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch, loss);
        }
    }
    let duration = start.elapsed();

    let (final_pred, _) = model.forward(DynamicTensor::Dim1(x.clone()));
    let (final_loss, _) = model.compute_loss(
        losses::mse(),
        &final_pred,
        &DynamicTensor::Dim1(target.clone()),
    );
    println!("Done. Time: {:?}", duration);
    println!("Final loss: {:.6}", final_loss);

    let params = model.param_store().lock().unwrap().all_params_vec();
    (params, final_loss)
}

fn main() {
    // 1. Однопоток
    let (cpu1_params, _) = run_training(
        DevicePlan::new().cpu(1, 8192),
        "CPU (1 поток)",
        None,
    );

    // 2. Многопоток
    run_training(
        DevicePlan::new().cpu(4, 8192),
        "CPU (4 потока)",
        None,
    );

    // 3. GPU
    let (gpu0_params, _) = run_training(
        DevicePlan::new().cpu(2, 8192).gpu(0, 4096),
        "GPU (id 0)",
        None,
    );

    // 4. С CPU на GPU
    run_training(
        DevicePlan::new().cpu(2, 8192).gpu(0, 4096),
        "GPU после CPU",
        Some(&cpu1_params),
    );

    // 5. С GPU на CPU
    run_training(
        DevicePlan::new().cpu(1, 8192),
        "CPU после GPU",
        Some(&gpu0_params),
    );

    // 6. Многопоток с SSD-кэшем
    run_training(
        DevicePlan::new()
            .cpu(4, 8192)
            .ssd_cache("D:\\neurocore_cache", 5000),
        "CPU (4 потока, SSD-кэш)",
        None,
    );
}


// examples/classifier.rs

use std::sync::mpsc;
use std::thread;
use std::time::Instant;
use neurocore::compute_manager::{Device, DynamicTensor};
use neurocore::model_plan::Plan;
use neurocore::tensor::Tensor2D;

mod models {
    use neurocore::model_plan::{Dim, LayerDesc, LayerKind};
    pub fn classifier() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new("fc", LayerKind::Linear, Dim::Dim1)
                .input(Dim::Dim1, &[2])
                .output(Dim::Dim1, &[2]),
            LayerDesc::new("softmax", LayerKind::Softmax, Dim::Dim1)
                .input(Dim::Dim1, &[2])
                .output(Dim::Dim1, &[2]),
        ]
    }
}

mod losses {
    use neurocore::loss_plan::{LossDesc, CrossEntropyWithLogits, ElementChain, Aggregation};
    pub fn cross_entropy() -> LossDesc {
        let chain = ElementChain::new().add(Box::new(CrossEntropyWithLogits::new(2)));
        LossDesc::from_chain(chain, Aggregation::Sum, 1, 2, 1)
    }
}

mod optimizers {
    use neurocore::optimizer_plan::{OptimizerDesc, OptCubeDesc};
    pub fn sgd() -> OptimizerDesc {
        OptimizerDesc::new().add(OptCubeDesc::ScaleGradient(0.5)).add(OptCubeDesc::ApplyUpdate)
    }
}

fn run_training(
    device: Device,
    label: &str,
    initial_params: Option<&[f32]>,
) -> (Vec<f32>, f32) {
    if matches!(device, Device::Gpu { .. }) {
        let device = device.clone();
        let label = label.to_string();
        let initial_params = initial_params.map(|p| p.to_vec());
        let (tx, rx) = mpsc::channel();
        thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(move || {
                let (params, loss) = run_training_inner(device, &label, initial_params.as_deref());
                tx.send((params, loss)).ok();
            })
            .unwrap();
        return rx.recv().unwrap();
    }
    run_training_inner(device, label, initial_params)
}

fn run_training_inner(
    device: Device,
    label: &str,
    initial_params: Option<&[f32]>,
) -> (Vec<f32>, f32) {
    println!("\n===== {} =====", label);
    let mut model = match Plan::from_layer_descs(models::classifier()) {
        Ok(plan) => plan.build_with_device(device),
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

    let x1 = Tensor2D::new(vec![vec![1.0, 2.0]]);
    let x2 = Tensor2D::new(vec![vec![2.0, 1.0]]);
    let y1 = Tensor2D::new(vec![vec![0.0]]);
    let y2 = Tensor2D::new(vec![vec![1.0]]);
    let epochs = 200;

    let start = Instant::now();
    for epoch in 0..epochs {
        for (x, y) in &[(&x1, &y1), (&x2, &y2)] {
            let (pred, ctxs) = model.forward(DynamicTensor::Dim1((*x).clone()));
            let (_, delta) = model.compute_loss(
                losses::cross_entropy(),
                &pred,
                &DynamicTensor::Dim1((*y).clone()),
            );
            let (_, grads) = model.backward(&ctxs, delta);
            model.update_params(optimizers::sgd(), &grads[0]);
        }

        if epoch % 40 == 0 {
            let (pred, _) = model.forward(DynamicTensor::Dim1(x1.clone()));
            let (loss, _) = model.compute_loss(
                losses::cross_entropy(),
                &pred,
                &DynamicTensor::Dim1(y1.clone()),
            );
            println!("Epoch {}: loss = {:.6}", epoch, loss);
        }
    }
    let duration = start.elapsed();

    let (final_pred, _) = model.forward(DynamicTensor::Dim1(x1.clone()));
    let (final_loss, _) = model.compute_loss(
        losses::cross_entropy(),
        &final_pred,
        &DynamicTensor::Dim1(y1.clone()),
    );
    println!("Done. Time: {:?}", duration);
    println!("Final loss: {:.6}", final_loss);

    let params = model.param_store().lock().unwrap().all_params_vec();
    (params, final_loss)
}

fn main() {
    // Стандартные одиночные проходы
    run_training(Device::Cpu { threads: 1 }, "CPU (1 поток)", None);
    run_training(Device::Cpu { threads: 4 }, "CPU (4 потока)", None);
    run_training(Device::Gpu { id: 0 }, "GPU (id 0)", None);

    // CPU → GPU
    let cpu_params = run_training(Device::Cpu { threads: 1 }, "CPU (1 поток)", None).0;
    run_training(Device::Gpu { id: 0 }, "GPU после CPU", Some(&cpu_params));

    // GPU → CPU
    let gpu_params = run_training(Device::Gpu { id: 0 }, "GPU (id 0)", None).0;
    run_training(Device::Cpu { threads: 1 }, "CPU после GPU", Some(&gpu_params));
}






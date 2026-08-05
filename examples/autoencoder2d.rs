// examples/autoencoder2d.rs
// Автоэнкодер с одним скрытым слоем, размерность Dim2 (Tensor3D).
// Демонстрация всех семи режимов обучения через ExecutionShop.

use std::sync::{Arc, Mutex};
use std::time::Instant;
use neurocore::compute_manager::{
    DevicePlan, DynamicTensor, ExecutionShop,
};
use neurocore::compute_manager::execution_shop::plan::ExecutionPlan;
use neurocore::compute_manager::execution_shop::ModelId;
use neurocore::tensor::Tensor3D;

mod models {
    use neurocore::model_plan::{Dim, LayerDesc, LayerKind};
    pub fn encoder() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new("fc1", LayerKind::Linear, Dim::Dim2)
                .input(Dim::Dim2, &[4])
                .output(Dim::Dim2, &[2]),
            LayerDesc::new("sigm", LayerKind::Sigmoid, Dim::Dim2)
                .input(Dim::Dim2, &[2])
                .output(Dim::Dim2, &[2]),
        ]
    }
    pub fn decoder() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new("fc2", LayerKind::Linear, Dim::Dim2)
                .input(Dim::Dim2, &[2])
                .output(Dim::Dim2, &[4]),
        ]
    }
}

mod losses {
    use neurocore::loss_plan::{Aggregation, ElementChain, LossDesc, Square, Sub};
    pub fn mse() -> LossDesc {
        let chain = ElementChain::new().add(Box::new(Sub)).add(Box::new(Square));
        LossDesc::from_chain(chain, Aggregation::Mean, 4, 1, 1)
    }
}

mod optimizers {
    use neurocore::optimizer_plan::{OptimizerDesc, OptCubeDesc};
    pub fn sgd() -> OptimizerDesc {
        OptimizerDesc::new().add(OptCubeDesc::ScaleGradient(0.01)).add(OptCubeDesc::ApplyUpdate)
    }
}

/// Разбивает одну строку (1, N) на N однострочных матриц (N, 1)
fn reshape_batch_to_tasks(tensor: &DynamicTensor) -> DynamicTensor {
    match tensor {
        DynamicTensor::Dim1(t) => {
            let row = &t.data[0];
            let data: Vec<Vec<f32>> = row.iter().map(|&v| vec![v]).collect();
            DynamicTensor::Dim1(neurocore::tensor::Tensor2D::new(data))
        }
        _ => panic!("Expected Dim1"),
    }
}

/// Обратное преобразование: из N однострочных матриц обратно в одну строку (1, N)
fn reshape_tasks_to_batch(tensor: &DynamicTensor) -> DynamicTensor {
    match tensor {
        DynamicTensor::Dim1(t) => {
            let mut combined = Vec::new();
            for row in &t.data {
                combined.push(row[0]);
            }
            DynamicTensor::Dim1(neurocore::tensor::Tensor2D::new(vec![combined]))
        }
        _ => panic!("Expected Dim1"),
    }
}

/// Извлекает плоский вектор из Dim2-тензора [1,1,N] для поэлементной обработки
fn extract_flat_dim2(tensor: &DynamicTensor) -> Vec<f32> {
    match tensor {
        DynamicTensor::Dim2(t) => t.data[0][0].clone(),
        _ => panic!("Expected Dim2"),
    }
}

/// Упаковывает плоский вектор обратно в Dim2 [1,1,N]
fn pack_flat_dim2(values: &[f32]) -> DynamicTensor {
    DynamicTensor::Dim2(Tensor3D::new(vec![vec![values.to_vec()]]))
}

fn run_training_with_shop(
    shop_enc: Arc<Mutex<ExecutionShop>>,
    enc_id: ModelId,
    shop_dec: Arc<Mutex<ExecutionShop>>,
    dec_id: ModelId,
    plan_enc: ExecutionPlan,
    plan_dec: ExecutionPlan,
    label: &str,
    initial_params_enc: Option<&[f32]>,
    initial_params_dec: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>, f32) {
    println!("\n===== {} =====", label);

    {
        let mut enc = shop_enc.lock().unwrap();
        enc.set_execution_plan(enc_id, plan_enc).expect("Failed to set encoder plan");
    }
    {
        let mut dec = shop_dec.lock().unwrap();
        dec.set_execution_plan(dec_id, plan_dec).expect("Failed to set decoder plan");
    }

    {
        let enc = shop_enc.lock().unwrap();
        if let Some(p) = initial_params_enc {
            enc.param_store(enc_id).unwrap().lock().unwrap().set_all_params(p);
        } else {
            let mut store = enc.param_store(enc_id).unwrap().lock().unwrap();
            for i in 0..store.len() { store.set_param(i, rand::random::<f32>() * 0.01); }
        }
    }
    {
        let dec = shop_dec.lock().unwrap();
        if let Some(p) = initial_params_dec {
            dec.param_store(dec_id).unwrap().lock().unwrap().set_all_params(p);
        } else {
            let mut store = dec.param_store(dec_id).unwrap().lock().unwrap();
            for i in 0..store.len() { store.set_param(i, rand::random::<f32>() * 0.01); }
        }
    }

    let x = Tensor3D::new(vec![vec![vec![1.0, 2.0, 3.0, 4.0]]]);
    let target_orig = x.clone();
    let epochs = 500;

    let start = Instant::now();
    for epoch in 0..epochs {
        let (code, ctx_enc) = {
            let enc = shop_enc.lock().unwrap();
            enc.forward(enc_id, DynamicTensor::Dim2(x.clone())).expect("Encoder forward failed")
        };
        let (recon, ctx_dec) = {
            let dec = shop_dec.lock().unwrap();
            dec.forward(dec_id, code).expect("Decoder forward failed")
        };

        let pred_vec = extract_flat_dim2(&recon);
        let target_vec = extract_flat_dim2(&DynamicTensor::Dim2(target_orig.clone()));
        let pred_tasks = reshape_batch_to_tasks(&DynamicTensor::Dim1(neurocore::tensor::Tensor2D::new(vec![pred_vec])));
        let target_tasks = reshape_batch_to_tasks(&DynamicTensor::Dim1(neurocore::tensor::Tensor2D::new(vec![target_vec])));

        let (loss, delta_tasks) = {
            let enc = shop_enc.lock().unwrap();
            enc.compute_loss(enc_id, losses::mse(), &pred_tasks, &target_tasks).expect("Loss failed")
        };

        let delta = reshape_tasks_to_batch(&delta_tasks);
        let delta_flat = match &delta {
            DynamicTensor::Dim1(t) => t.data[0].clone(),
            _ => panic!("Expected Dim1 delta"),
        };
        let delta_dim2 = pack_flat_dim2(&delta_flat);

        let (delta_code, grads_dec) = {
            let dec = shop_dec.lock().unwrap();
            dec.backward(dec_id, &ctx_dec, delta_dim2).expect("Decoder backward failed")
        };
        {
            let dec = shop_dec.lock().unwrap();
            dec.update_params(dec_id, optimizers::sgd(), &grads_dec[0]);
        }

        let (_, grads_enc) = {
            let enc = shop_enc.lock().unwrap();
            enc.backward(enc_id, &ctx_enc, delta_code).expect("Encoder backward failed")
        };
        {
            let enc = shop_enc.lock().unwrap();
            enc.update_params(enc_id, optimizers::sgd(), &grads_enc[0]);
        }

        if epoch == 0 || epoch % 100 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch, loss);
        }
    }
    let duration = start.elapsed();

    let (code, _) = {
        let enc = shop_enc.lock().unwrap();
        enc.forward(enc_id, DynamicTensor::Dim2(x.clone())).expect("Final encoder forward")
    };
    let (final_recon, _) = {
        let dec = shop_dec.lock().unwrap();
        dec.forward(dec_id, code).expect("Final decoder forward")
    };
    let pred_vec = extract_flat_dim2(&final_recon);
    let target_vec = extract_flat_dim2(&DynamicTensor::Dim2(target_orig));
    let pred_tasks = reshape_batch_to_tasks(&DynamicTensor::Dim1(neurocore::tensor::Tensor2D::new(vec![pred_vec])));
    let target_tasks = reshape_batch_to_tasks(&DynamicTensor::Dim1(neurocore::tensor::Tensor2D::new(vec![target_vec])));
    let (final_loss, _) = {
        let enc = shop_enc.lock().unwrap();
        enc.compute_loss(enc_id, losses::mse(), &pred_tasks, &target_tasks).expect("Final loss failed")
    };

    println!("Done. Time: {:?}", duration);
    println!("Final loss: {:.6}", final_loss);

    let enc_params = shop_enc.lock().unwrap().param_store(enc_id).unwrap().lock().unwrap().all_params_vec();
    let dec_params = shop_dec.lock().unwrap().param_store(dec_id).unwrap().lock().unwrap().all_params_vec();
    (enc_params, dec_params, final_loss)
}

fn main() {
    // 1. CPU (1 поток)
    let plan1 = DevicePlan::new().cpu(1, 8192);
    let shop_enc1 = ExecutionShop::new_shared(plan1.clone());
    let shop_dec1 = ExecutionShop::new_shared(plan1.clone());
    let enc_id1 = shop_enc1.lock().unwrap().create_model_with_device_plan(models::encoder(), plan1.clone()).expect("build enc");
    let dec_id1 = shop_dec1.lock().unwrap().create_model_with_device_plan(models::decoder(), plan1).expect("build dec");
    let (enc_cpu1, dec_cpu1, _) = run_training_with_shop(
        shop_enc1, enc_id1,
        shop_dec1, dec_id1,
        ExecutionPlan::single_device(1, 0),
        ExecutionPlan::single_device(1, 0),
        "CPU (1 поток, single‑device план)",
        None, None,
    );

    // 2. CPU (4 потока)
    let plan2 = DevicePlan::new().cpu(4, 8192);
    let shop_enc2 = ExecutionShop::new_shared(plan2.clone());
    let shop_dec2 = ExecutionShop::new_shared(plan2.clone());
    let enc_id2 = shop_enc2.lock().unwrap().create_model_with_device_plan(models::encoder(), plan2.clone()).expect("build enc");
    let dec_id2 = shop_dec2.lock().unwrap().create_model_with_device_plan(models::decoder(), plan2).expect("build dec");
    run_training_with_shop(
        shop_enc2, enc_id2,
        shop_dec2, dec_id2,
        ExecutionPlan::single_device(1, 0),
        ExecutionPlan::single_device(1, 0),
        "CPU (4 потока, single‑device план)",
        None, None,
    );

    // 3. GPU (id 0)
    let plan3 = DevicePlan::new().cpu(2, 8192).gpu(0, 4096);
    let shop_enc3 = ExecutionShop::new_shared(plan3.clone());
    let shop_dec3 = ExecutionShop::new_shared(plan3.clone());
    let enc_id3 = shop_enc3.lock().unwrap().create_model_with_device_plan(models::encoder(), plan3.clone()).expect("build enc");
    let dec_id3 = shop_dec3.lock().unwrap().create_model_with_device_plan(models::decoder(), plan3).expect("build dec");
    let (enc_gpu, dec_gpu, _) = run_training_with_shop(
        shop_enc3, enc_id3,
        shop_dec3, dec_id3,
        ExecutionPlan::single_device(1, 1),
        ExecutionPlan::single_device(1, 1),
        "GPU (id 0, single‑device план)",
        None, None,
    );

    // 4. GPU после CPU
    let plan4 = DevicePlan::new().cpu(2, 8192).gpu(0, 4096);
    let shop_enc4 = ExecutionShop::new_shared(plan4.clone());
    let shop_dec4 = ExecutionShop::new_shared(plan4.clone());
    let enc_id4 = shop_enc4.lock().unwrap().create_model_with_device_plan(models::encoder(), plan4.clone()).expect("build enc");
    let dec_id4 = shop_dec4.lock().unwrap().create_model_with_device_plan(models::decoder(), plan4).expect("build dec");
    run_training_with_shop(
        shop_enc4, enc_id4,
        shop_dec4, dec_id4,
        ExecutionPlan::single_device(1, 1),
        ExecutionPlan::single_device(1, 1),
        "GPU после CPU",
        Some(&enc_cpu1), Some(&dec_cpu1),
    );

    // 5. CPU после GPU
    let plan5 = DevicePlan::new().cpu(1, 8192);
    let shop_enc5 = ExecutionShop::new_shared(plan5.clone());
    let shop_dec5 = ExecutionShop::new_shared(plan5.clone());
    let enc_id5 = shop_enc5.lock().unwrap().create_model_with_device_plan(models::encoder(), plan5.clone()).expect("build enc");
    let dec_id5 = shop_dec5.lock().unwrap().create_model_with_device_plan(models::decoder(), plan5).expect("build dec");
    run_training_with_shop(
        shop_enc5, enc_id5,
        shop_dec5, dec_id5,
        ExecutionPlan::single_device(1, 0),
        ExecutionPlan::single_device(1, 0),
        "CPU после GPU",
        Some(&enc_gpu), Some(&dec_gpu),
    );

    // 6. CPU (4 потока) + SSD кэш
    let plan6 = DevicePlan::new().cpu(4, 8192).ssd_cache("D:\\neurocore_cache", 5000);
    let shop_enc6 = ExecutionShop::new_shared(plan6.clone());
    let shop_dec6 = ExecutionShop::new_shared(plan6.clone());
    let enc_id6 = shop_enc6.lock().unwrap().create_model_with_device_plan(models::encoder(), plan6.clone()).expect("build enc");
    let dec_id6 = shop_dec6.lock().unwrap().create_model_with_device_plan(models::decoder(), plan6).expect("build dec");
    run_training_with_shop(
        shop_enc6, enc_id6,
        shop_dec6, dec_id6,
        ExecutionPlan::single_device(1, 0),
        ExecutionPlan::single_device(1, 0),
        "CPU (4 потока, SSD кэш)",
        None, None,
    );

    // 7. Параллельное обучение на CPU и GPU с автоматическим планом
    let plan7 = DevicePlan::new().cpu(4, 8192).gpu(0, 4096);
    let shop_enc7 = ExecutionShop::new_shared(plan7.clone());
    let shop_dec7 = ExecutionShop::new_shared(plan7.clone());
    let enc_id7 = shop_enc7.lock().unwrap().create_model_with_device_plan(models::encoder(), plan7.clone()).expect("build enc");
    let dec_id7 = shop_dec7.lock().unwrap().create_model_with_device_plan(models::decoder(), plan7).expect("build dec");
    let auto_enc_plan = {
        let enc = shop_enc7.lock().unwrap();
        let graph = enc.model_graph(enc_id7).expect("Graph not built");
        let devices = enc.devices();
        ExecutionPlan::auto(graph, devices)
    };
    let auto_dec_plan = {
        let dec = shop_dec7.lock().unwrap();
        let graph = dec.model_graph(dec_id7).expect("Graph not built");
        let devices = dec.devices();
        ExecutionPlan::auto(graph, devices)
    };
    run_training_with_shop(
        shop_enc7, enc_id7,
        shop_dec7, dec_id7,
        auto_enc_plan,
        auto_dec_plan,
        "CPU + GPU параллельно (auto‑план)",
        None, None,
    );
}
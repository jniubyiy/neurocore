// examples/linear_test.rs
// Обучение одного линейного слоя с демонстрацией всех режимов,
// включая распределённое выполнение на CPU и GPU через ExecutionShop.

use std::sync::{Arc, Mutex};
use std::time::Instant;
use neurocore::compute_manager::{
    DevicePlan, DynamicTensor, ExecutionShop,
};
use neurocore::compute_manager::execution_shop::plan::ExecutionPlan;
use neurocore::compute_manager::execution_shop::ModelId;
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

fn run_training_with_shop(
    shop: Arc<Mutex<ExecutionShop>>,
    model_id: ModelId,
    plan: ExecutionPlan,
    label: &str,
    initial_params: Option<&[f32]>,
    use_gpu: bool,
) -> (Vec<f32>, f32) {
    println!("\n===== {} =====", label);

    {
        let mut shop_guard = shop.lock().unwrap();
        shop_guard.set_execution_plan(model_id, plan).expect("Failed to set execution plan");
    }

    {
        let shop_guard = shop.lock().unwrap();
        if let Some(p) = initial_params {
            shop_guard.param_store(model_id)
                .expect("No param store")
                .lock().unwrap()
                .set_all_params(p);
        } else {
            let mut store = shop_guard.param_store(model_id)
                .expect("No param store")
                .lock().unwrap();
            for i in 0..store.len() {
                store.set_param(i, rand::random::<f32>() * 0.01);
            }
        }
        // Если используется GPU, синхронизируем параметры на GPU
        if use_gpu {
            shop_guard.sync_params_to_gpu(model_id).expect("Failed to sync params to GPU");
        }
    }

    let x = Tensor2D::new(vec![vec![1.0, 2.0, 3.0, 4.0]]);
    let target_orig = Tensor2D::new(vec![vec![0.8, 1.5]]);
    let epochs = 500;

    let start = Instant::now();
    for epoch in 0..epochs {
        let (pred, ctxs) = {
            let shop_guard = shop.lock().unwrap();
            shop_guard.forward(model_id, DynamicTensor::Dim1(x.clone()))
                .expect("Forward failed")
        };

        let (loss, delta) = {
            let shop_guard = shop.lock().unwrap();
            shop_guard.compute_loss(
                model_id,
                losses::mse(),
                &pred,
                &DynamicTensor::Dim1(target_orig.clone()),
            ).expect("Loss failed")
        };

        let (_, grads) = {
            let shop_guard = shop.lock().unwrap();
            shop_guard.backward(model_id, &ctxs, delta)
                .expect("Backward failed")
        };

        // Обновление параметров: CPU или GPU в зависимости от флага
        {
            let shop_guard = shop.lock().unwrap();
            if use_gpu {
                shop_guard.update_params_gpu(model_id, optimizers::sgd(), epoch);
            } else {
                shop_guard.update_params(model_id, optimizers::sgd(), &grads[0]);
            }
        }

        if epoch == 0 || epoch % 100 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch, loss);
        }
    }
    let duration = start.elapsed();

    // Финальный loss
    let (final_pred, _) = {
        let shop_guard = shop.lock().unwrap();
        shop_guard.forward(model_id, DynamicTensor::Dim1(x.clone()))
            .expect("Final forward failed")
    };
    let (final_loss, _) = {
        let shop_guard = shop.lock().unwrap();
        shop_guard.compute_loss(
            model_id,
            losses::mse(),
            &final_pred,
            &DynamicTensor::Dim1(target_orig.clone()),
        ).expect("Final loss failed")
    };
    println!("Done. Time: {:?}", duration);
    println!("Final loss: {:.6}", final_loss);

    // Получаем итоговые параметры
    let params = if use_gpu {
        let shop_guard = shop.lock().unwrap();
        // Для GPU нужно прочитать параметры из GPU-хранилища
        // Пока используем заглушку: получаем с CPU (параметры могли не синхронизироваться обратно)
        // В будущем добавим метод sync_params_from_gpu
        shop_guard.param_store(model_id).unwrap().lock().unwrap().all_params_vec()
    } else {
        let shop_guard = shop.lock().unwrap();
        let param_store = shop_guard.param_store(model_id).unwrap();
        let store = param_store.lock().unwrap();
        store.all_params_vec()
    };
    (params, final_loss)
}

fn main() {
    // 1. CPU (1 поток)
    let plan1 = DevicePlan::new().cpu(1, 8192);
    let shop1 = ExecutionShop::new_shared(plan1.clone());
    let model_id1 = shop1.lock().unwrap().create_model_with_device_plan(models::linear_model(), plan1)
        .expect("build failed");
    let seg_count = shop1.lock().unwrap().model_graph(model_id1).unwrap().len();
    let (cpu1_params, _) = run_training_with_shop(
        shop1,
        model_id1,
        ExecutionPlan::single_device(seg_count, 0),
        "CPU (1 поток, single‑device план)",
        None,
        false,
    );

    // 2. CPU (4 потока)
    let plan2 = DevicePlan::new().cpu(4, 8192);
    let shop2 = ExecutionShop::new_shared(plan2.clone());
    let model_id2 = shop2.lock().unwrap().create_model_with_device_plan(models::linear_model(), plan2)
        .expect("build failed");
    let seg_count2 = shop2.lock().unwrap().model_graph(model_id2).unwrap().len();
    run_training_with_shop(
        shop2,
        model_id2,
        ExecutionPlan::single_device(seg_count2, 0),
        "CPU (4 потока, single‑device план)",
        None,
        false,
    );

    // 3. GPU (id 0)
    let plan3 = DevicePlan::new().cpu(2, 8192).gpu(0, 4096);
    let shop3 = ExecutionShop::new_shared(plan3.clone());
    let model_id3 = shop3.lock().unwrap().create_model_with_device_plan(models::linear_model(), plan3)
        .expect("build failed");
    let seg_count3 = shop3.lock().unwrap().model_graph(model_id3).unwrap().len();
    let (gpu0_params, _) = run_training_with_shop(
        shop3,
        model_id3,
        ExecutionPlan::single_device(seg_count3, 1),
        "GPU (id 0, single‑device план)",
        None,
        true,
    );

    // 4. GPU после CPU (передаём параметры с CPU на GPU)
    let plan4 = DevicePlan::new().cpu(2, 8192).gpu(0, 4096);
    let shop4 = ExecutionShop::new_shared(plan4.clone());
    let model_id4 = shop4.lock().unwrap().create_model_with_device_plan(models::linear_model(), plan4)
        .expect("build failed");
    let seg_count4 = shop4.lock().unwrap().model_graph(model_id4).unwrap().len();
    run_training_with_shop(
        shop4,
        model_id4,
        ExecutionPlan::single_device(seg_count4, 1),
        "GPU после CPU (параметры с CPU, single‑device план)",
        Some(&cpu1_params),
        true,
    );

    // 5. CPU после GPU (передаём параметры с GPU на CPU)
    let plan5 = DevicePlan::new().cpu(1, 8192);
    let shop5 = ExecutionShop::new_shared(plan5.clone());
    let model_id5 = shop5.lock().unwrap().create_model_with_device_plan(models::linear_model(), plan5)
        .expect("build failed");
    let seg_count5 = shop5.lock().unwrap().model_graph(model_id5).unwrap().len();
    run_training_with_shop(
        shop5,
        model_id5,
        ExecutionPlan::single_device(seg_count5, 0),
        "CPU после GPU (параметры с GPU, single‑device план)",
        Some(&gpu0_params),
        false,
    );

    // 6. CPU (4 потока) + SSD кэш
    let plan6 = DevicePlan::new()
        .cpu(4, 8192)
        .ssd_cache("D:\\neurocore_cache", 5000);
    let shop6 = ExecutionShop::new_shared(plan6.clone());
    let model_id6 = shop6.lock().unwrap().create_model_with_device_plan(models::linear_model(), plan6)
        .expect("build failed");
    let seg_count6 = shop6.lock().unwrap().model_graph(model_id6).unwrap().len();
    run_training_with_shop(
        shop6,
        model_id6,
        ExecutionPlan::single_device(seg_count6, 0),
        "CPU (4 потока, SSD кэш, single‑device план)",
        None,
        false,
    );

    // 7. Параллельное обучение на CPU и GPU с автоматическим планом
    let plan7 = DevicePlan::new().cpu(4, 8192).gpu(0, 4096);
    let shop7 = ExecutionShop::new_shared(plan7.clone());
    let model_id7 = shop7.lock().unwrap().create_model_with_device_plan(models::linear_model(), plan7)
        .expect("build failed");
    let auto_plan = {
        let shop_guard = shop7.lock().unwrap();
        let graph = shop_guard.model_graph(model_id7).expect("Graph not built");
        let devices = shop_guard.devices();
        ExecutionPlan::auto(graph, devices)
    };
    // Для смешанного выполнения используем GPU-оптимизатор, так как план может включать GPU
    run_training_with_shop(
        shop7,
        model_id7,
        auto_plan,
        "CPU + GPU параллельно (auto‑план)",
        None,
        true,
    );
}

// examples/soft_sparse_gate_example.rs
// Обучаемый SoftSparseGate: учим пороги обнулять шум.

use std::time::Instant;
use neurocore::compute_manager::DynamicTensor;
use neurocore::tensor::Tensor2D;
use neurocore::create_models;

mod models {
    use neurocore::model_plan::{Dim, LayerDesc, LayerKind};

    pub fn gate_model() -> Vec<LayerDesc> {
        vec![
            // Линейный слой, не меняющий размер (просто для демонстрации)
            LayerDesc::new("linear", LayerKind::Linear, Dim::Dim1)
                .input(Dim::Dim1, &[4usize])
                .output(Dim::Dim1, &[4usize]),
            // SoftSparseGate с обучаемыми порогами
            LayerDesc::new("sparse_gate", LayerKind::SoftSparseGate, Dim::Dim1)
                .input(Dim::Dim1, &[4usize])
                .output(Dim::Dim1, &[4usize])
                .extra(vec![0.5]), // temperature
        ]
    }
}

mod losses {
    use neurocore::loss_plan::{Aggregation, ElementChain, LossDesc, Square, Sub};
    pub fn mse() -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(Sub))
            .add(Box::new(Square));
        LossDesc::from_chain(chain, Aggregation::Mean, 4, 1, 1)
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

fn main() {
    let (mut model,) = create_models!(models::gate_model);
    // Инициализация параметров
    {
        let mut store = model.param_store().lock().unwrap();
        let len = store.len();
        for i in 0..len {
            store.set_param(i, rand::random::<f32>() * 0.1);
        }
    }

    // Вход: полезный сигнал (большие значения) + шум (маленькие значения)
    let x_batch = Tensor2D::new(vec![
        vec![0.0, 1.5, -0.8, 2.0],  // смесь
        vec![-0.2, 0.1, 0.3, -1.2],
    ]);
    // Целевой выход: шум обнулён, полезный сигнал сохранён
    let y_batch = Tensor2D::new(vec![
        vec![0.0, 1.5, -0.8, 2.0],  // нулевой шум уже 0, остальное остаётся
        vec![0.0, 0.0, 0.3, -1.2],  // -0.2 и 0.1 считаем шумом, обнуляем
    ]);

    let epochs = 200;
    let start = Instant::now();
    for epoch in 0..epochs {
        let (pred, ctxs) = model.forward(DynamicTensor::Dim1(x_batch.clone()));
        let (loss, delta) = model.compute_loss(
            losses::mse(),
            &pred,
            &DynamicTensor::Dim1(y_batch.clone()),
        );
        let (_, grads) = model.backward(&ctxs, delta);
        model.update_params(optimizers::sgd(), &grads[0]);

        if epoch % 50 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch, loss);
        }
    }
    let duration = start.elapsed();
    println!("SoftSparseGate обучение завершено за {:?}", duration);
}
// examples/leaky_relu_example.rs
// Сравнение ReLU и LeakyReLU на данных с отрицательными значениями.

use std::time::Instant;
use neurocore::compute_manager::DynamicTensor;
use neurocore::tensor::Tensor2D;
use neurocore::create_models;

mod models {
    use neurocore::model_plan::{Dim, LayerDesc, LayerKind};
    pub fn leaky_relu_model() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new("linear", LayerKind::Linear, Dim::Dim1)
                .input(Dim::Dim1, &[4])
                .output(Dim::Dim1, &[4]),
            LayerDesc::new("leaky_relu", LayerKind::LeakyReLU, Dim::Dim1)
                .input(Dim::Dim1, &[4])
                .output(Dim::Dim1, &[4])
                .extra(vec![0.1]), // alpha = 0.1
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
        OptimizerDesc::new()
            .add(OptCubeDesc::ScaleGradient(0.01))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

fn main() {
    let (mut model,) = create_models!(models::leaky_relu_model);
    {
        let mut store = model.param_store().lock().unwrap();
        for i in 0..store.len() {
            store.set_param(i, rand::random::<f32>() * 0.1);
        }
    }

    // Вход: смесь положительных и отрицательных чисел
    let x = Tensor2D::new(vec![vec![1.0, -2.0, 3.0, -4.0]]);
    // Целевой выход: LeakyReLU(x) = x если x>0, иначе 0.1*x
    let target = Tensor2D::new(vec![vec![1.0, -0.2, 3.0, -0.4]]);

    let epochs = 200;
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

        if epoch % 50 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch, loss);
        }
    }

    let duration = start.elapsed();
    println!("LeakyReLU example done in {:?}", duration);
}
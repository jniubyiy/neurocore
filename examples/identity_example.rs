// examples/identity_example.rs
// Identity слой в остаточной связи: y = x + Linear(x).

use std::time::Instant;
use neurocore::compute_manager::DynamicTensor;
use neurocore::tensor::Tensor2D;
use neurocore::create_models;

mod models {
    use neurocore::model_plan::{Dim, LayerDesc, LayerKind};
    pub fn residual() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new("linear", LayerKind::Linear, Dim::Dim1)
                .input(Dim::Dim1, &[4])
                .output(Dim::Dim1, &[4]),
            LayerDesc::new("identity", LayerKind::Identity, Dim::Dim1)
                .input(Dim::Dim1, &[4])
                .output(Dim::Dim1, &[4]),
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
    let (mut model,) = create_models!(models::residual);
    {
        let mut store = model.param_store().lock().unwrap();
        for i in 0..store.len() {
            store.set_param(i, rand::random::<f32>() * 0.1);
        }
    }

    // Обучаем Linear предсказывать разность target - x
    let x = Tensor2D::new(vec![vec![1.0, 2.0, 3.0, 4.0]]);
    let target = Tensor2D::new(vec![vec![1.2, 1.9, 3.3, 3.8]]);  // x + residual

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
    println!("Identity example done in {:?}", duration);
}
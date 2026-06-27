// examples/classifier2d.rs
// Классификатор на 2 класса, размерность Dim2.
// Используются новые удобные методы compute_loss и update_params.

use std::time::Instant;
use neurocore::compute_manager::DynamicTensor;
use neurocore::tensor::Tensor3D;
use neurocore::create_models;

mod models {
    use neurocore::model_plan::{Dim, LayerDesc, LayerKind};

    pub fn classifier() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new("fc", LayerKind::Linear, Dim::Dim2)
                .input(Dim::Dim2, &[2])
                .output(Dim::Dim2, &[2]),
            LayerDesc::new("softmax", LayerKind::Softmax, Dim::Dim2)
                .input(Dim::Dim2, &[2])
                .output(Dim::Dim2, &[2]),
        ]
    }
}

mod losses {
    use neurocore::loss_plan::{LossDesc, CrossEntropyWithLogits, ElementChain, Aggregation};

    pub fn cross_entropy() -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(CrossEntropyWithLogits::new(2)));
        LossDesc::from_chain(chain, Aggregation::Sum, 1, 2, 1)
    }
}

mod optimizers {
    use neurocore::optimizer_plan::{OptimizerDesc, OptCubeDesc};

    pub fn sgd() -> OptimizerDesc {
        OptimizerDesc::new()
            .add(OptCubeDesc::ScaleGradient(0.5))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

fn main() {
    let (mut model,) = create_models!(models::classifier);

    let x1 = Tensor3D::new(vec![vec![vec![1.0, 2.0]]]);
    let x2 = Tensor3D::new(vec![vec![vec![2.0, 1.0]]]);
    let y1 = Tensor3D::new(vec![vec![vec![0.0]]]); // класс 0
    let y2 = Tensor3D::new(vec![vec![vec![1.0]]]); // класс 1
    let epochs = 200;

    let start = Instant::now();
    for epoch in 0..epochs {
        for (x, y) in &[(&x1, &y1), (&x2, &y2)] {
            let (pred, ctxs) = model.forward(DynamicTensor::Dim2((*x).clone()));
            let (_, delta) = model.compute_loss(
                losses::cross_entropy(),
                &pred,
                &DynamicTensor::Dim2((*y).clone()),
            );
            let (_, grads) = model.backward(&ctxs, delta);
            model.update_params(optimizers::sgd(), &grads[0]);
        }

        if epoch % 50 == 0 {
            let (pred, _) = model.forward(DynamicTensor::Dim2(x1.clone()));
            let (loss, _) = model.compute_loss(
                losses::cross_entropy(),
                &pred,
                &DynamicTensor::Dim2(y1.clone()),
            );
            println!("Epoch {}: loss = {:.6}", epoch, loss);
        }
    }
    let duration = start.elapsed();

    let (final_pred, _) = model.forward(DynamicTensor::Dim2(x1.clone()));
    let (final_loss, _) = model.compute_loss(
        losses::cross_entropy(),
        &final_pred,
        &DynamicTensor::Dim2(y1.clone()),
    );
    println!("Done. Time: {:?}", duration);
    println!("Final loss: {:.6}", final_loss);
}






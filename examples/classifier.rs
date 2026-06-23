// examples/classifier.rs
// Классификатор на 2 класса, размерность Dim1.

use std::time::Instant;
use neurocore::compute_manager::DynamicTensor;
use neurocore::tensor::Tensor2D;
use neurocore::{create_models, create_losses, create_optimizers};

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
    let (loss_expr,) = create_losses!(losses::cross_entropy);
    let (mut opt,) = create_optimizers!((model, optimizers::sgd));

    let x1 = Tensor2D::new(vec![vec![1.0, 2.0]]);
    let x2 = Tensor2D::new(vec![vec![2.0, 1.0]]);
    let y1 = Tensor2D::new(vec![vec![0.0]]); // класс 0
    let y2 = Tensor2D::new(vec![vec![1.0]]); // класс 1
    let epochs = 200;

    let start = Instant::now();
    for epoch in 0..epochs {
        for (x, y) in &[(&x1, &y1), (&x2, &y2)] {
            let (pred, ctxs) = model.forward(DynamicTensor::Dim1((*x).clone()));
            let (_, delta) = model.compute_loss_with_expr(
                loss_expr.clone(),
                &pred,
                &DynamicTensor::Dim1((*y).clone()),
            );
            let (_, grads) = model.backward(&ctxs, delta);
            model.update_params_with_optimizer(&mut opt, &grads[0]);
        }

        if epoch % 40 == 0 {
            let (pred, _) = model.forward(DynamicTensor::Dim1(x1.clone()));
            let (loss, _) = model.compute_loss_with_expr(
                loss_expr.clone(),
                &pred,
                &DynamicTensor::Dim1(y1.clone()),
            );
            println!("Epoch {}: loss = {:.6}", epoch, loss);
        }
    }
    let duration = start.elapsed();

    let (final_pred, _) = model.forward(DynamicTensor::Dim1(x1.clone()));
    let (final_loss, _) = model.compute_loss_with_expr(
        loss_expr,
        &final_pred,
        &DynamicTensor::Dim1(y1.clone()),
    );
    println!("Done. Time: {:?}", duration);
    println!("Final loss: {:.6}", final_loss);
}






// examples/linear3d_test.rs
// Один линейный слой 4 -> 2, размерность Dim3.

use std::time::Instant;
use neurocore::compute_manager::DynamicTensor;
use neurocore::tensor::Tensor4D;
use neurocore::{create_models, create_losses, create_optimizers};

mod models {
    use neurocore::model_plan::{Dim, LayerDesc, LayerKind};

    pub fn linear_model() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new("linear", LayerKind::Linear, Dim::Dim3)
                .input(Dim::Dim3, &[4])
                .output(Dim::Dim3, &[2]),
        ]
    }
}

mod losses {
    use neurocore::loss_plan::{Aggregation, ElementChain, LossDesc, Square, Sub};

    pub fn mse() -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(Sub))
            .add(Box::new(Square));
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

fn main() {
    let (mut model,) = create_models!(models::linear_model);
    let (loss_expr,) = create_losses!(losses::mse);
    let (mut opt,) = create_optimizers!((model, optimizers::sgd));

    let x = Tensor4D::new(vec![vec![vec![vec![1.0, 2.0, 3.0, 4.0]]]]);
    let target = Tensor4D::new(vec![vec![vec![vec![0.8, 1.2]]]]);
    let epochs = 500;

    let start = Instant::now();
    for epoch in 0..epochs {
        let (pred, ctxs) = model.forward(DynamicTensor::Dim3(x.clone()));
        let (loss, delta) = model.compute_loss_with_expr(
            loss_expr.clone(),
            &pred,
            &DynamicTensor::Dim3(target.clone()),
        );
        let (_, grads) = model.backward(&ctxs, delta);
        model.update_params_with_optimizer(&mut opt, &grads[0]);

        if epoch == 0 || epoch % 200 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch, loss);
        }
    }
    let duration = start.elapsed();

    let (final_pred, _) = model.forward(DynamicTensor::Dim3(x.clone()));
    let (final_loss, _) = model.compute_loss_with_expr(
        loss_expr,
        &final_pred,
        &DynamicTensor::Dim3(target.clone()),
    );

    println!("Done. Time: {:?}", duration);
    println!("Final loss: {:.6}", final_loss);
}
// examples/autoencoder3d.rs
// Автоэнкодер с Dim3 (Tensor4D).
// Используются новые удобные методы compute_loss и update_params.

use std::time::Instant;
use neurocore::compute_manager::DynamicTensor;
use neurocore::tensor::Tensor4D;
use neurocore::create_models;

mod models {
    use neurocore::model_plan::{Dim, LayerDesc, LayerKind};
    pub fn encoder() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new("fc1", LayerKind::Linear, Dim::Dim3)
                .input(Dim::Dim3, &[4])
                .output(Dim::Dim3, &[2]),
            LayerDesc::new("sigm", LayerKind::Sigmoid, Dim::Dim3)
                .input(Dim::Dim3, &[2])
                .output(Dim::Dim3, &[2]),
        ]
    }
    pub fn decoder() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new("fc2", LayerKind::Linear, Dim::Dim3)
                .input(Dim::Dim3, &[2])
                .output(Dim::Dim3, &[4]),
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
    pub fn sgd_encoder() -> OptimizerDesc {
        OptimizerDesc::new().add(OptCubeDesc::ScaleGradient(0.01)).add(OptCubeDesc::ApplyUpdate)
    }
    pub fn sgd_decoder() -> OptimizerDesc {
        OptimizerDesc::new().add(OptCubeDesc::ScaleGradient(0.01)).add(OptCubeDesc::ApplyUpdate)
    }
}

fn main() {
    let (mut encoder, mut decoder) = create_models!(models::encoder, models::decoder);

    // Данные: Tensor4D [batch=1, dim2=1, dim3=1, features=4]
    let x = Tensor4D::new(vec![vec![vec![vec![1.0, 2.0, 3.0, 4.0]]]]);
    let target = x.clone();
    let epochs = 500;

    let start = Instant::now();
    for epoch in 0..epochs {
        let (code, ctx_enc) = encoder.forward(DynamicTensor::Dim3(x.clone()));
        let (recon, ctx_dec) = decoder.forward(code);

        let (loss, delta_loss) = encoder.compute_loss(
            losses::mse(),
            &recon,
            &DynamicTensor::Dim3(target.clone()),
        );

        let (delta_code, grads_dec) = decoder.backward(&ctx_dec, delta_loss);
        decoder.update_params(optimizers::sgd_decoder(), &grads_dec[0]);

        let (_, grads_enc) = encoder.backward(&ctx_enc, delta_code);
        encoder.update_params(optimizers::sgd_encoder(), &grads_enc[0]);

        if epoch == 0 || epoch % 100 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch, loss);
        }
    }
    let duration = start.elapsed();

    let (code, _) = encoder.forward(DynamicTensor::Dim3(x.clone()));
    let (final_recon, _) = decoder.forward(code);
    let (final_loss, _) = encoder.compute_loss(
        losses::mse(),
        &final_recon,
        &DynamicTensor::Dim3(target),
    );

    println!("Обучение завершено за {:?}", duration);
    println!("Финальный loss: {:.6}", final_loss);
}




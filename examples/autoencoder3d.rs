// examples/autoencoder3d.rs

use neurocore::model_plan::{Plan, LayerDesc, LayerKind, Dim};
use neurocore::dispatchers::auto_model::{MixedModel, DynamicTensor};
use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::tensor::{Tensor3D, Tensor4D};
use std::time::Instant;

fn main() {
    let descs = vec![
        LayerDesc::new("fc1", LayerKind::Linear, Dim::Dim3)
            .input(Dim::Dim3, &[4])
            .output(Dim::Dim3, &[2]),
        LayerDesc::new("sigm", LayerKind::Sigmoid, Dim::Dim3)
            .input(Dim::Dim3, &[2])
            .output(Dim::Dim3, &[2]),
        LayerDesc::new("fc2", LayerKind::Linear, Dim::Dim3)
            .input(Dim::Dim3, &[2])
            .output(Dim::Dim3, &[4]),
    ];

    let plan = Plan::from_descs(descs).expect("Ошибка плана");
    let mut model = plan.build();

    let loss_plan = LossPlan::new(LossBlueprint::mse()).unwrap();
    let built_loss = loss_plan.build_3d().unwrap();

    let input = Tensor4D::new(vec![vec![vec![vec![1.0, 2.0, 3.0, 4.0]]]]);
    let target = input.clone();
    let lr = 0.1;
    let epochs = 500;

    let start = Instant::now();
    for epoch in 0..epochs {
        let (pred_dyn, ctxs) = model.forward(DynamicTensor::Dim3(Tensor3D::new(input.data[0].clone())));
        let pred = match pred_dyn {
            DynamicTensor::Dim3(t) => t,
            _ => panic!(),
        };
        let (loss, delta) = (built_loss.forward)(&pred, &Tensor3D::new(target.data[0].clone()));
        let (_, grads) = model.backward(&ctxs, DynamicTensor::Dim3(delta));
        model.update_params(lr, &grads);

        if epoch == 0 || epoch % 100 == 0 {
            println!("Epoch {}: loss={:.6}", epoch, loss);
        }
    }
    let duration = start.elapsed();

    let (final_pred_dyn, _) = model.forward(DynamicTensor::Dim3(Tensor3D::new(input.data[0].clone())));
    let final_pred = match final_pred_dyn {
        DynamicTensor::Dim3(t) => t,
        _ => panic!(),
    };
    let (final_loss, _) = (built_loss.forward)(&final_pred, &Tensor3D::new(target.data[0].clone()));

    println!("Done. Time: {:?}", duration);
    println!("Final loss: {:.6}", final_loss);
}





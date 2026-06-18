// examples/linear5d_test.rs

use neurocore::model_plan::{Plan, LayerDesc, LayerKind, Dim};
use neurocore::dispatchers::auto_model::{MixedModel, DynamicTensor};
use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::tensor::Tensor5D;
use std::time::Instant;

fn main() {
    let descs = vec![
        LayerDesc::new("linear", LayerKind::Linear, Dim::Dim5)
            .input(Dim::Dim5, &[4])
            .output(Dim::Dim5, &[2]),
    ];

    let plan = Plan::from_descs(descs).expect("Ошибка плана");
    let mut model = plan.build();

    let loss_plan = LossPlan::new(LossBlueprint::mse()).unwrap();
    let built_loss = loss_plan.build_5d().unwrap();

    let x = Tensor5D::new(vec![vec![vec![vec![vec![1.0, 2.0, 3.0, 4.0]]]]]);
    let target = Tensor5D::new(vec![vec![vec![vec![vec![0.8, 1.2]]]]]);
    let lr = 0.01;
    let epochs = 500;

    let start = Instant::now();
    for epoch in 0..epochs {
        let (pred_dyn, ctxs) = model.forward(DynamicTensor::Dim5(x.clone()));
        let pred = match pred_dyn {
            DynamicTensor::Dim5(t) => t,
            _ => panic!(),
        };
        let (loss, delta) = (built_loss.forward)(&pred, &target);
        let (_, grads) = model.backward(&ctxs, DynamicTensor::Dim5(delta));
        model.update_params(lr, &grads);

        if epoch == 0 || epoch % 200 == 0 {
            println!("Epoch {}: loss={:.6}", epoch, loss);
        }
    }
    let duration = start.elapsed();

    let (final_pred_dyn, _) = model.forward(DynamicTensor::Dim5(x.clone()));
    let final_pred = match final_pred_dyn {
        DynamicTensor::Dim5(t) => t,
        _ => panic!(),
    };
    let (final_loss, _) = (built_loss.forward)(&final_pred, &target);
    println!("Done. Time: {:?}", duration);
    println!("Final loss: {:.6}", final_loss);
}





// examples/classifier4d.rs

use neurocore::model_plan::{Plan, LayerDesc, LayerKind, Dim};
use neurocore::dispatchers::auto_model::{MixedModel, DynamicTensor};
use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::tensor::Tensor4D;
use std::time::Instant;

fn main() {
    let descs = vec![
        LayerDesc::new("fc", LayerKind::Linear, Dim::Dim4)
            .input(Dim::Dim4, &[2])
            .output(Dim::Dim4, &[2]),
        LayerDesc::new("softmax", LayerKind::Softmax, Dim::Dim4)
            .input(Dim::Dim4, &[2])
            .output(Dim::Dim4, &[2]),
    ];

    let plan = Plan::from_descs(descs).expect("Ошибка плана");
    let mut model = plan.build();

    let loss_plan = LossPlan::new(LossBlueprint::cross_entropy(2)).unwrap();
    let built_loss = loss_plan.build_4d().unwrap();

    let x1 = Tensor4D::new(vec![vec![vec![vec![1.0, 2.0]]]]);
    let x2 = Tensor4D::new(vec![vec![vec![vec![2.0, 1.0]]]]);
    let y1 = Tensor4D::new(vec![vec![vec![vec![0.0]]]]);
    let y2 = Tensor4D::new(vec![vec![vec![vec![1.0]]]]);
    let lr = 0.5;
    let epochs = 200;

    let start = Instant::now();
    for epoch in 0..epochs {
        for (x, y) in &[(&x1, &y1), (&x2, &y2)] {
            let (pred_dyn, ctxs) = model.forward(DynamicTensor::Dim4((*x).clone()));
            let pred = match pred_dyn {
                DynamicTensor::Dim4(t) => t,
                _ => panic!(),
            };
            let (_, delta) = (built_loss.forward)(&pred, y);
            let (_, grads) = model.backward(&ctxs, DynamicTensor::Dim4(delta));
            model.update_params(lr, &grads);
        }

        if epoch % 50 == 0 {
            let (pred_dyn, _) = model.forward(DynamicTensor::Dim4(x1.clone()));
            let pred = match pred_dyn {
                DynamicTensor::Dim4(t) => t,
                _ => panic!(),
            };
            let (loss, _) = (built_loss.forward)(&pred, &y1);
            println!("Epoch {}: loss={:.6}", epoch, loss);
        }
    }
    let duration = start.elapsed();

    let (pred_dyn, _) = model.forward(DynamicTensor::Dim4(x1.clone()));
    let pred = match pred_dyn {
        DynamicTensor::Dim4(t) => t,
        _ => panic!(),
    };
    let (final_loss, _) = (built_loss.forward)(&pred, &y1);
    println!("Done. Time: {:?}", duration);
    println!("Final loss: {:.6}", final_loss);
}





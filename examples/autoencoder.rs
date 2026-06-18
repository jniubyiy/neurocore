// examples/autoencoder.rs

use neurocore::model_plan::{Plan, LayerDesc, LayerKind, Dim};
use neurocore::dispatchers::auto_model::{MixedModel, DynamicTensor};
use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::tensor::Tensor1D;
use std::time::Instant;

fn main() {
    // 1. Построение плана через LayerDesc
    let descs = vec![
        LayerDesc::new("fc1", LayerKind::Linear, Dim::Dim1)
            .input(Dim::Dim1, &[4])
            .output(Dim::Dim1, &[2]),
        LayerDesc::new("sigm", LayerKind::Sigmoid, Dim::Dim1)
            .input(Dim::Dim1, &[2])
            .output(Dim::Dim1, &[2]),
        LayerDesc::new("fc2", LayerKind::Linear, Dim::Dim1)
            .input(Dim::Dim1, &[2])
            .output(Dim::Dim1, &[4]),
    ];

    let plan = Plan::from_descs(descs).expect("Ошибка плана");
    let mut model = plan.build();

    // 2. Функция потерь
    let loss_plan = LossPlan::new(LossBlueprint::mse()).unwrap();
    let built_loss = loss_plan.build().unwrap();

    // 3. Данные и гиперпараметры
    let x = Tensor1D::new(vec![1.0, 2.0, 3.0, 4.0]);
    let target = x.clone();
    let lr = 0.1;
    let epochs = 500;

    let start = Instant::now();
    for epoch in 0..epochs {
        // Прямой проход
        let (pred_dyn, ctxs) = model.forward(DynamicTensor::Dim1(x.clone()));
        let pred = match pred_dyn {
            DynamicTensor::Dim1(t) => t,
            _ => panic!(),
        };

        // Вычисление ошибки и градиента
        let (loss, delta) = (built_loss.forward)(&pred, &target);

        // Обратный проход и обновление параметров
        let (_, grads) = model.backward(&ctxs, DynamicTensor::Dim1(delta));
        model.update_params(lr, &grads);

        if epoch == 0 || epoch % 100 == 0 {
            println!("Epoch {}: loss={:.6}", epoch, loss);
        }
    }
    let duration = start.elapsed();

    // Финальное предсказание
    let (final_pred_dyn, _) = model.forward(DynamicTensor::Dim1(x.clone()));
    let final_pred = match final_pred_dyn {
        DynamicTensor::Dim1(t) => t,
        _ => panic!(),
    };
    let (final_loss, _) = (built_loss.forward)(&final_pred, &target);

    println!("Done. Time: {:?}", duration);
    println!("Final loss: {:.6}", final_loss);
}





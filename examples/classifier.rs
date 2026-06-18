// examples/classifier.rs

use neurocore::model_plan::{Plan, LayerDesc, LayerKind, Dim};
use neurocore::dispatchers::auto_model::{MixedModel, DynamicTensor};
use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::tensor::Tensor1D;
use std::time::Instant;

fn main() {
    // Модель: линейный слой 2 -> 2, затем softmax
    let descs = vec![
        LayerDesc::new("fc", LayerKind::Linear, Dim::Dim1)
            .input(Dim::Dim1, &[2])
            .output(Dim::Dim1, &[2]),
        LayerDesc::new("softmax", LayerKind::Softmax, Dim::Dim1)
            .input(Dim::Dim1, &[2])
            .output(Dim::Dim1, &[2]),
    ];

    let plan = Plan::from_descs(descs).expect("Ошибка плана");
    let mut model = plan.build();

    // Функция потерь: кросс-энтропия для 2 классов
    let loss_plan = LossPlan::new(LossBlueprint::cross_entropy(2)).unwrap();
    let built_loss = loss_plan.build().unwrap();

    // Данные и гиперпараметры
    let x1 = Tensor1D::new(vec![1.0, 2.0]);
    let x2 = Tensor1D::new(vec![2.0, 1.0]);
    let y1 = Tensor1D::new(vec![0.0]); // класс 0
    let y2 = Tensor1D::new(vec![1.0]); // класс 1
    let lr = 0.5;
    let epochs = 200;

    let start = Instant::now();
    for epoch in 0..epochs {
        for (x, y) in &[(&x1, &y1), (&x2, &y2)] {
            // Прямой проход
            let (pred_dyn, ctxs) = model.forward(DynamicTensor::Dim1((*x).clone()));
            let pred = match pred_dyn {
                DynamicTensor::Dim1(t) => t,
                _ => panic!(),
            };

            // Вычисление потерь и градиента
            let (_, delta) = (built_loss.forward)(&pred, y);

            // Обратный проход и обновление параметров
            let (_, grads) = model.backward(&ctxs, DynamicTensor::Dim1(delta));
            model.update_params(lr, &grads);
        }

        if epoch % 40 == 0 {
            let (pred_dyn, _) = model.forward(DynamicTensor::Dim1(x1.clone()));
            let pred = match pred_dyn {
                DynamicTensor::Dim1(t) => t,
                _ => panic!(),
            };
            let (loss, _) = (built_loss.forward)(&pred, &y1);
            println!("Epoch {}: loss={:.6}", epoch, loss);
        }
    }
    let duration = start.elapsed();

    // Финальная оценка
    let (pred_dyn, _) = model.forward(DynamicTensor::Dim1(x1.clone()));
    let pred = match pred_dyn {
        DynamicTensor::Dim1(t) => t,
        _ => panic!(),
    };
    let (final_loss, _) = (built_loss.forward)(&pred, &y1);
    println!("Done. Time: {:?}", duration);
    println!("Final loss: {:.6}", final_loss);
}






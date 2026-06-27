// examples/memory_example.rs
// Пример обучения слоя Memory на синтетических данных.
// Используется Tensor2D (размерность Dim1).

use std::time::Instant;
use neurocore::compute_manager::DynamicTensor;
use neurocore::tensor::Tensor2D;
use neurocore::create_models;

mod models {
    use neurocore::model_plan::{Dim, LayerDesc, LayerKind};

    pub fn memory_model() -> Vec<LayerDesc> {
        vec![
            // Входной линейный слой, преобразует признаки к нужной размерности
            LayerDesc::new("linear", LayerKind::Linear, Dim::Dim1)
                .input(Dim::Dim1, &[4usize])
                .output(Dim::Dim1, &[4usize]),
            // Собственно Memory‑слой с тем же числом признаков
            LayerDesc::new("memory", LayerKind::Memory, Dim::Dim1)
                .input(Dim::Dim1, &[4usize])
                .output(Dim::Dim1, &[4usize]),
        ]
    }
}

mod losses {
    use neurocore::loss_plan::{Aggregation, ElementChain, LossDesc, Square, Sub};

    pub fn mse() -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(Sub))
            .add(Box::new(Square));
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
    let (mut model,) = create_models!(models::memory_model);

    // Инициализация параметров малыми случайными числами
    {
        let mut store = model.param_store().lock().unwrap();
        let len = store.len();
        for i in 0..len {
            store.set_param(i, rand::random::<f32>() * 0.1);
        }
    }

    // Синтетические данные: два режима смешивания
    // Входные векторы (4 признака)
    let x_batch = Tensor2D::new(vec![
        vec![1.0, 0.5, -0.2, 0.8],
        vec![0.3, -0.7, 1.2, -0.4],
    ]);
    // Целевые выходы: результат смешивания (например, w0*0.8 + w1*0.2)
    // здесь мы задаём желаемый выход вручную
    let y_batch = Tensor2D::new(vec![
        vec![0.64, 0.35, -0.1, 0.56],  // примерно 0.8*x1 + 0.2*x2? для примера зафиксируем
        vec![0.18, -0.5, 0.9, -0.3],
    ]);

    let epochs = 200;
    let start = Instant::now();

    for epoch in 0..epochs {
        let (pred, ctxs) = model.forward(DynamicTensor::Dim1(x_batch.clone()));
        let (loss, delta) = model.compute_loss(
            losses::mse(),
            &pred,
            &DynamicTensor::Dim1(y_batch.clone()),
        );
        let (_, grads) = model.backward(&ctxs, delta);
        model.update_params(optimizers::sgd(), &grads[0]);

        if epoch % 50 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch, loss);
        }
    }

    let duration = start.elapsed();
    println!("Обучение завершено за {:?}", duration);

    // Проверка на одном примере
    let (final_pred, _) = model.forward(DynamicTensor::Dim1(x_batch.clone()));
    let (final_loss, _) = model.compute_loss(
        losses::mse(),
        &final_pred,
        &DynamicTensor::Dim1(y_batch.clone()),
    );
    println!("Final loss: {:.6}", final_loss);
}
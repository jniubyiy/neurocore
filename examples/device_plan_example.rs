// examples/device_plan_example.rs
// Демонстрация обязательного модуля device_plan с разделением Compute/Storage устройств.
// План автоматически применяется макросом create_models!.

use neurocore::compute_manager::DynamicTensor;
use neurocore::tensor::Tensor2D;
use neurocore::create_models;

// ---------------------------------------------------------------------------
// Обязательный модуль device_plan – задаёт конфигурацию устройств
// ---------------------------------------------------------------------------
mod device_plan {
    use neurocore::device_plan::DevicePlan;

    pub fn plan() -> DevicePlan {
        DevicePlan::empty()
            // Вычислительные устройства
            .cpu(0, 4)            // CPU id=0, 4 потока
            .gpu(0)               // GPU id=0
            // Устройства хранения
            .ram(0, 8192)         // RAM id=0, 8 ГБ
            .vram(0, 0, 4096)     // VRAM id=0, привязана к GPU id=0, 4 ГБ
        //  .ssd(0, "D:/cache", 10000)  // SSD id=0, 10 ГБ (опционально)
    }
}

// ---------------------------------------------------------------------------
// Модель
// ---------------------------------------------------------------------------
mod models {
    use neurocore::model_plan::{Dim, LayerDesc, LayerKind};

    pub fn linear_model() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new("linear", LayerKind::Linear, Dim::Dim1)
                .input(Dim::Dim1, &[4])
                .output(Dim::Dim1, &[2]),
        ]
    }
}

// ---------------------------------------------------------------------------
// Функция потерь
// ---------------------------------------------------------------------------
mod losses {
    use neurocore::loss_plan::{Aggregation, ElementChain, LossDesc, Square, Sub};

    pub fn mse() -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(Sub))
            .add(Box::new(Square));
        LossDesc::from_chain(chain, Aggregation::Mean, 2, 1, 1)
    }
}

// ---------------------------------------------------------------------------
// Оптимизатор
// ---------------------------------------------------------------------------
mod optimizers {
    use neurocore::optimizer_plan::{OptimizerDesc, OptCubeDesc};

    pub fn sgd() -> OptimizerDesc {
        OptimizerDesc::new()
            .add(OptCubeDesc::ScaleGradient(0.01))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

// ---------------------------------------------------------------------------
fn main() {
    // create_models! автоматически подхватывает device_plan::plan()
    let (mut model,) = create_models!(models::linear_model);

    // Данные
    let x = Tensor2D::new(vec![vec![1.0, 2.0, 3.0, 4.0]]);
    let target = Tensor2D::new(vec![vec![0.8, 1.2]]);

    // Обучение
    for epoch in 0..500 {
        let (pred, ctxs) = model.forward(DynamicTensor::Dim1(x.clone()));
        let (loss, delta) = model.compute_loss(
            losses::mse(),
            &pred,
            &DynamicTensor::Dim1(target.clone()),
        );
        let (_, grads) = model.backward(&ctxs, delta);
        model.update_params(optimizers::sgd(), &grads[0]);

        if epoch % 100 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch, loss);
        }
    }

    println!("Обучение завершено.");
}
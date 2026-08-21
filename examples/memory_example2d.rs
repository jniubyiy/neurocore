// examples/memory_example2d.rs
// Пример обучения сети со слоями Memory для очистки зашумлённого сигнала (2D данные).
// Модель: Linear -> Memory -> Linear -> Memory

use neurocore::tensor::Tensor3D;

mod models {
    use neurocore::model_plan::{LayerKind, LayerDesc};
    use neurocore::shape;

    pub fn memory_model() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[4], B[4]))
                .output(shape!(batch, A[4], B[4])),
            LayerDesc::new(LayerKind::Memory)
                .input(shape!(batch, A[4], B[4]))
                .output(shape!(batch, A[4], B[4])),
            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[4], B[4]))
                .output(shape!(batch, A[4], B[4])),
            LayerDesc::new(LayerKind::Memory)
                .input(shape!(batch, A[4], B[4]))
                .output(shape!(batch, A[4], B[4])),
        ]
    }
}

mod losses {
    use neurocore::loss_plan::{
        Aggregation, ElementChain, LossDesc, Square, Sub, SumColumns,
    };

    pub fn mse() -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(Sub::new(16)))
            .add(Box::new(Square))
            .add(Box::new(SumColumns));
        LossDesc::from_chain(chain, Aggregation::Mean, 20, 16, 16)
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

/// Генерирует 2D зашумлённые данные (batch, 4, 4).
fn generate_data2d(num_samples: usize) -> (Tensor3D, Tensor3D) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut noisy = Vec::with_capacity(num_samples);
    let mut clean = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let mut noisy_sample = Vec::with_capacity(4);
        let mut clean_sample = Vec::with_capacity(4);
        for _ in 0..4 {
            let mut noisy_row = Vec::with_capacity(4);
            let mut clean_row = Vec::with_capacity(4);
            for _ in 0..4 {
                let target = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
                let noise = rng.gen_range(-0.5..0.5);
                noisy_row.push(target + noise);
                clean_row.push(target);
            }
            noisy_sample.push(noisy_row);
            clean_sample.push(clean_row);
        }
        noisy.push(noisy_sample);
        clean.push(clean_sample);
    }

    (Tensor3D::new(noisy), Tensor3D::new(clean))
}

mod device_plan {
    use neurocore::device_plan::DevicePlan;

    pub fn plan() -> DevicePlan {
        DevicePlan::empty().cpu(0, 4).ram(0, 8192)
    }
}

mod training_plan {
    use super::*;
    use neurocore::training_plan::plan::{TrainingPlan, DataSource, Initializer};

    pub fn plan() -> TrainingPlan {
        let num_samples = 20;
        let (train_x, train_y) = generate_data2d(num_samples);

        TrainingPlan::new()
            .model(models::memory_model)
            .loss(losses::mse())
            .optimizer(optimizers::sgd())
            .epochs(300)
            .batch_size(num_samples)
            .train_data(DataSource::from_tensor3d(train_x))
            .target_data(DataSource::from_tensor3d(train_y))
            .init_weights(Initializer::RandomUniform {
                min: -0.1,
                max: 0.1,
            })
            .seed(42)
            .output_tensors(vec!["prediction".to_string()])
    }
}

fn main() {
    let result = neurocore::run_training!(
        training_plan::plan,
        device = device_plan::plan
    );
    println!(
        "Memory example 2D done. Final loss: {:.6}, time: {:.3}s, best epoch: {}",
        result.final_loss, result.training_time_secs, result.best_epoch
    );
}
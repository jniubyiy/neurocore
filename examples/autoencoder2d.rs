// examples/autoencoder2d.rs
// Автоэнкодер 16 -> 4 -> 16, размерность Dim2 (Tensor3D).

use neurocore::tensor::Tensor3D;

mod models {
    use neurocore::model_plan::{LayerKind, LayerDesc};
    use neurocore::shape;
    pub fn autoencoder() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[4], B[4]))
                .output(shape!(batch, A[2], B[2])),
            LayerDesc::new(LayerKind::Sigmoid)
                .input(shape!(batch, A[2], B[2]))
                .output(shape!(batch, A[2], B[2])),
            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[2], B[2]))
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
        LossDesc::from_chain(chain, Aggregation::Mean, 1, 16, 16)
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

fn data() -> Tensor3D {
    Tensor3D::new(vec![vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0, 11.0, 12.0],
        vec![13.0, 14.0, 15.0, 16.0],
    ]])
}

mod device_plan {
    use neurocore::device_plan::DevicePlan;
    pub fn plan() -> DevicePlan { DevicePlan::empty().cpu(0, 4).ram(0, 8192) }
}

mod training_plan {
    use super::*;
    use neurocore::training_plan::plan::{TrainingPlan, DataSource, Initializer};
    pub fn plan() -> TrainingPlan {
        TrainingPlan::new()
            .model(models::autoencoder)
            .loss(losses::mse())
            .optimizer(optimizers::sgd())
            .epochs(100)
            .batch_size(1)
            .train_data(DataSource::from_tensor3d(data()))
            .target_data(DataSource::from_tensor3d(data()))
            .init_weights(Initializer::RandomUniform { min: -0.1, max: 0.1 })
            .seed(42)
            .output_tensors(vec!["prediction".to_string()])
    }
}

fn main() {
    let result = neurocore::run_training!(training_plan::plan, device = device_plan::plan);
    println!("Autoencoder2D done. Final loss: {:.6}, time: {:.3}s, best epoch: {}",
        result.final_loss, result.training_time_secs, result.best_epoch);
}
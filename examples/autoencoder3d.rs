// examples/autoencoder3d.rs
// Автоэнкодер 64 -> 8 -> 64, размерность Dim3 (Tensor4D).

use neurocore::tensor::Tensor4D;

mod model {
    use neurocore::model_plan::{LayerKind, LayerDesc};
    pub fn autoencoder() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new(LayerKind::Linear)
                .input((batch, A[4], B[4], C[4]))
                .output((batch, A[2], B[2], C[2])),
            LayerDesc::new(LayerKind::Sigmoid)
                .input((batch, A[2], B[2], C[2]))
                .output((batch, A[2], B[2], C[2])),
            LayerDesc::new(LayerKind::Linear)
                .input((batch, A[2], B[2], C[2]))
                .output((batch, A[4], B[4], C[4])),
        ]
    }
}

mod losses {
    use neurocore::loss_plan::{
        Aggregation, ElementChain, LossDesc, Square, Sub, SumColumns,
    };
    pub fn mse() -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(Sub::new(64)))
            .add(Box::new(Square))
            .add(Box::new(SumColumns));
        LossDesc::from_chain(chain, Aggregation::Mean, 1, 64, 64)
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

fn data() -> Tensor4D {
    let vals: Vec<f32> = (0..64).map(|i| (i as f32 + 1.0) / 64.0).collect();
    let mut idx = 0;
    Tensor4D::new(vec![vec![vec![
        (0..4).map(|_| { let v = vals[idx]; idx+=1; v }).collect(),
        (0..4).map(|_| { let v = vals[idx]; idx+=1; v }).collect(),
        (0..4).map(|_| { let v = vals[idx]; idx+=1; v }).collect(),
        (0..4).map(|_| { let v = vals[idx]; idx+=1; v }).collect(),
    ]; 4]])
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
            .train_data(DataSource::from_tensor4d(data()))
            .target_data(DataSource::from_tensor4d(data()))
            .init_weights(Initializer::RandomUniform { min: -0.1, max: 0.1 })
            .seed(42)
            .output_tensors(vec!["prediction".to_string()])
    }
}

fn main() {
    let result = neurocore::run_training!(training_plan::plan, device = device_plan::plan);
    println!("Autoencoder3D done. Final loss: {:.6}, time: {:.3}s, best epoch: {}",
        result.final_loss, result.training_time_secs, result.best_epoch);
}




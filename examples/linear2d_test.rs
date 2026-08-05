// examples/linear2d_test.rs
// Полноценное обучение автоэнкодера 16 -> 16 (Tensor3D) через TrainingPlan.

use neurocore::training_plan::plan::{TrainingPlan, DataSource, Initializer};

mod device_plan {
    use neurocore::device_plan::DevicePlan;
    pub fn plan() -> DevicePlan { DevicePlan::empty().cpu(0, 4).ram(0, 8192) }
}

mod models {
    use neurocore::model_plan::{LayerKind, LayerDesc};
    use neurocore::shape;
    pub fn linear_model() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[4], B[4]))
                .output(shape!(batch, A[4], B[4])),
        ]
    }
}

mod losses {
    use neurocore::loss_plan::{Aggregation, ElementChain, LossDesc, Square, Sub, SumColumns};
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
            .add(OptCubeDesc::ScaleGradient(0.0001))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

mod training_plan {
    use super::models;
    use super::losses;
    use super::optimizers;
    use neurocore::training_plan::plan::{TrainingPlan, DataSource, Initializer};
    use neurocore::tensor::Tensor3D;

    pub fn plan() -> TrainingPlan {
        let x = Tensor3D::new(vec![
            vec![
                vec![1.0, 2.0, 3.0, 4.0],
                vec![5.0, 6.0, 7.0, 8.0],
                vec![9.0, 10.0, 11.0, 12.0],
                vec![13.0, 14.0, 15.0, 16.0],
            ],
        ]);
        let ds = DataSource::from_tensor3d(x);
        TrainingPlan::new()
            .model(models::linear_model)
            .loss(losses::mse())
            .optimizer(optimizers::sgd())
            .epochs(100)
            .batch_size(1)
            .train_data(ds)
            .init_weights(Initializer::RandomUniform { min: -0.01, max: 0.01 })
    }
}

fn main() {
    let r = neurocore::run_training!(training_plan::plan, device = device_plan::plan);
    println!("Training done. Final loss: {:.6}", r.final_loss);
}





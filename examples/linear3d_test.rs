// examples/linear3d_test.rs
// Полноценное обучение автоэнкодера 64 -> 64 (Tensor4D) через TrainingPlan.

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
                .input(shape!(batch, A[4], B[4], C[4]))
                .output(shape!(batch, A[4], B[4], C[4])),
        ]
    }
}

mod losses {
    use neurocore::loss_plan::{Aggregation, ElementChain, LossDesc, Square, Sub, SumColumns};
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
            .add(OptCubeDesc::ScaleGradient(0.00001))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

mod training_plan {
    use super::models;
    use super::losses;
    use super::optimizers;
    use neurocore::training_plan::plan::{TrainingPlan, DataSource, Initializer};
    use neurocore::tensor::Tensor4D;

    pub fn plan() -> TrainingPlan {
        let x = Tensor4D::new(vec![
            vec![
                vec![
                    vec![1.0, 2.0, 3.0, 4.0],
                    vec![5.0, 6.0, 7.0, 8.0],
                    vec![9.0, 10.0, 11.0, 12.0],
                    vec![13.0, 14.0, 15.0, 16.0],
                ],
                vec![
                    vec![17.0, 18.0, 19.0, 20.0],
                    vec![21.0, 22.0, 23.0, 24.0],
                    vec![25.0, 26.0, 27.0, 28.0],
                    vec![29.0, 30.0, 31.0, 32.0],
                ],
                vec![
                    vec![33.0, 34.0, 35.0, 36.0],
                    vec![37.0, 38.0, 39.0, 40.0],
                    vec![41.0, 42.0, 43.0, 44.0],
                    vec![45.0, 46.0, 47.0, 48.0],
                ],
                vec![
                    vec![49.0, 50.0, 51.0, 52.0],
                    vec![53.0, 54.0, 55.0, 56.0],
                    vec![57.0, 58.0, 59.0, 60.0],
                    vec![61.0, 62.0, 63.0, 64.0],
                ],
            ],
        ]);
        let ds = DataSource::from_tensor4d(x);
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
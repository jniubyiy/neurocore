// examples/linear4d_test.rs
// Полноценное обучение автоэнкодера 256 -> 256 (Tensor5D) через TrainingPlan.
// Данные нормированы в [0, 1] для стабильности.

use neurocore::training_plan::plan::{TrainingPlan, DataSource, Initializer};

mod device_plan {
    use neurocore::device_plan::DevicePlan;
    pub fn plan() -> DevicePlan { DevicePlan::empty().cpu(0, 4).ram(0, 8192) }
}

mod model {
    use neurocore::model_plan::{LayerKind, LayerDesc};
    pub fn linear_model() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new(LayerKind::Linear)
                .input((batch, A[4], B[4], C[4], D[4]))
                .output((batch, A[4], B[4], C[4], D[4])),
        ]
    }
}

mod losses {
    use neurocore::loss_plan::{Aggregation, ElementChain, LossDesc, Square, Sub, SumColumns};
    pub fn mse() -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(Sub::new(256)))
            .add(Box::new(Square))
            .add(Box::new(SumColumns));
        LossDesc::from_chain(chain, Aggregation::Mean, 1, 256, 256)
    }
}

mod optimizers {
    use neurocore::optimizer_plan::{OptimizerDesc, OptCubeDesc};
    pub fn sgd() -> OptimizerDesc {
        OptimizerDesc::new()
            .add(OptCubeDesc::ScaleGradient(0.001))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

mod training_plan {
    use super::models;
    use super::losses;
    use super::optimizers;
    use neurocore::training_plan::plan::{TrainingPlan, DataSource, Initializer};
    use neurocore::tensor::Tensor5D;

    pub fn plan() -> TrainingPlan {
        // Нормированные значения в [0, 1]
        let values: Vec<f32> = (0..256).map(|i| i as f32 / 256.0).collect();
        let mut data = Vec::with_capacity(1);
        let mut dim2 = Vec::with_capacity(4);
        for a in 0..4 {
            let mut dim3 = Vec::with_capacity(4);
            for b in 0..4 {
                let mut dim4 = Vec::with_capacity(4);
                for c in 0..4 {
                    let mut dim5 = Vec::with_capacity(4);
                    for d in 0..4 {
                        let idx = a * 64 + b * 16 + c * 4 + d;
                        dim5.push(values[idx]);
                    }
                    dim4.push(dim5);
                }
                dim3.push(dim4);
            }
            dim2.push(dim3);
        }
        data.push(dim2);
        let x = Tensor5D::new(data);
        let ds = DataSource::from_tensor5d(x);
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





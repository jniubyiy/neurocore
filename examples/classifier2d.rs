// examples/classifier2d.rs
// Классификатор на 2 класса, размерность Dim2 (Tensor3D).
// Вход: 4 признака (2×2), выход: 2 логита.

use neurocore::tensor::Tensor3D;

mod models {
    use neurocore::model_plan::{LayerKind, LayerDesc};
    use neurocore::shape;

    pub fn classifier() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[2], B[2]))   // 2*2 = 4 признака
                .output(shape!(batch, A[2])),       // 2 логита
        ]
    }
}

mod losses {
    use neurocore::loss_plan::{
        Aggregation, CrossEntropyWithLogits, ElementChain, LossDesc,
    };

    pub fn cross_entropy() -> LossDesc {
        let num_classes = 2;
        let chain = ElementChain::new()
            .add(Box::new(CrossEntropyWithLogits::new(num_classes)));
        LossDesc::from_chain(chain, Aggregation::Sum, 1, num_classes, 1)
    }
}

mod optimizers {
    use neurocore::optimizer_plan::{OptimizerDesc, OptCubeDesc};

    pub fn sgd() -> OptimizerDesc {
        OptimizerDesc::new()
            .add(OptCubeDesc::ScaleGradient(0.5))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

// Данные: два сэмпла, каждый размерности [2,2] = 4 признака
fn train_data() -> Tensor3D {
    Tensor3D::new(vec![
        vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ],
        vec![
            vec![4.0, 3.0],
            vec![2.0, 1.0],
        ],
    ])
}

// Целевые метки: [0, 1] в Tensor3D формы [2,1,1]
fn target_data() -> Tensor3D {
    Tensor3D::new(vec![
        vec![vec![0.0]],
        vec![vec![1.0]],
    ])
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
        TrainingPlan::new()
            .model(models::classifier)
            .loss(losses::cross_entropy())
            .optimizer(optimizers::sgd())
            .epochs(200)
            .batch_size(1)
            .train_data(DataSource::from_tensor3d(train_data()))
            .target_data(DataSource::from_tensor3d(target_data()))
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
        "Classifier2D done. Final loss: {:.6}, time: {:.3}s, best epoch: {}",
        result.final_loss, result.training_time_secs, result.best_epoch
    );
}






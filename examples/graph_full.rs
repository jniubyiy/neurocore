// examples/graph_full.rs
// Полный граф с ветвлением: Linear -> Splitter -> две ветви -> Combiner -> Linear.

use neurocore::tensor::Tensor2D;

mod models {
    use neurocore::model_plan::{LayerDesc, LayerKind};
    use neurocore::shape;

    pub fn stage1() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[6]))
                .output(shape!(batch, A[6])),

            LayerDesc::new(LayerKind::Splitter)
                .input(shape!(batch, A[6]))
                .output(shape!(batch, A[2]; batch, A[4])),

            LayerDesc::new(LayerKind::SplitterConnector)
                .input(shape!(batch, A[2]; batch, A[4])),
        ]
    }

    pub fn branch_a() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new(LayerKind::SplitterConnector)
                .output(shape!(batch, A[2])),

            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[2]))
                .output(shape!(batch, A[3])),

            LayerDesc::new(LayerKind::ReLU)
                .input(shape!(batch, A[3]))
                .output(shape!(batch, A[3])),

            LayerDesc::new(LayerKind::CombinerConnector)
                .input(shape!(batch, A[3])),
        ]
    }

    pub fn branch_b() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new(LayerKind::SplitterConnector)
                .output(shape!(batch, A[4])),

            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[4]))
                .output(shape!(batch, A[3])),

            LayerDesc::new(LayerKind::Sigmoid)
                .input(shape!(batch, A[3]))
                .output(shape!(batch, A[3])),

            LayerDesc::new(LayerKind::CombinerConnector)
                .input(shape!(batch, A[3])),
        ]
    }

    pub fn stage2() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new(LayerKind::SplitterConnector)
                .output(shape!(batch, A[3]; batch, A[3])),

            LayerDesc::new(LayerKind::Combiner)
                .input(shape!(batch, A[3]; batch, A[3]))
                .output(shape!(batch, A[6])),

            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[6]))
                .output(shape!(batch, A[3])),

            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[3]))
                .output(shape!(batch, A[1])),
        ]
    }

    pub fn graph_model() -> Vec<LayerDesc> {
        stage1()
            .into_iter()
            .chain(branch_a())
            .chain(branch_b())
            .chain(stage2())
            .collect()
    }
}

mod losses {
    use neurocore::loss_plan::{Aggregation, ElementChain, LossDesc, Square, Sub};

    pub fn mse() -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(Sub::new(1)))
            .add(Box::new(Square));
        LossDesc::from_chain(chain, Aggregation::Mean, 2, 1, 1)
    }
}

mod optimizers {
    use neurocore::optimizer_plan::{OptCubeDesc, OptimizerDesc};

    pub fn sgd() -> OptimizerDesc {
        OptimizerDesc::new()
            .add(OptCubeDesc::ScaleGradient(0.01))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

fn train_data() -> Tensor2D {
    Tensor2D::new(vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ])
}

fn target_data() -> Tensor2D {
    Tensor2D::new(vec![vec![0.8], vec![0.2]])
}

mod device_plan {
    use neurocore::device_plan::DevicePlan;

    pub fn plan() -> DevicePlan {
        DevicePlan::empty().cpu(0, 4).ram(0, 8192)
    }
}

mod training_plan {
    use super::*;
    use neurocore::training_plan::plan::{DataSource, Initializer, TrainingPlan};

    pub fn plan() -> TrainingPlan {
        TrainingPlan::new()
            .model(models::graph_model)
            .loss(losses::mse())
            .optimizer(optimizers::sgd())
            .epochs(200)
            .batch_size(2)
            .train_data(DataSource::from_tensor2d(train_data()))
            .target_data(DataSource::from_tensor2d(target_data()))
            .init_weights(Initializer::RandomUniform { min: -0.1, max: 0.1 })
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
        "GraphFull training done. Final loss: {:.6}, time: {:.3}s, best epoch: {}",
        result.final_loss, result.training_time_secs, result.best_epoch
    );
}
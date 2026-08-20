// examples/graph_full.rs
// Полный граф с ветвлением: Linear -> Splitter -> две ветви -> Combiner -> Linear.
// Используется новый API TrainingPlan и run_training!.
// Модель разделена на четыре части: stage1, branch_a, branch_b, stage2.

use neurocore::tensor::Tensor2D;

// ═══════════════ Модель (четыре части + сборка) ═══════════════
mod models {
    use neurocore::model_plan::{LayerDesc, LayerKind};

    /// Этап 1: входной Linear, Splitter и SplitterConnector.
    pub fn stage1() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new(LayerKind::Linear)
                .input((batch, A[6]))
                .output((batch, A[6])),

            LayerDesc::new(LayerKind::Splitter)
                .input((batch, A[6]))
                .output((batch, A[2]),((batch, A[4]))),

            // SplitterConnector: устанавливает активные порты [2, 4]
            LayerDesc::new(LayerKind::SplitterConnector)
                .input((batch, A[2]),((batch, A[4]))),
        ]
    }

    /// Ветвь A: 2 -> 3 с ReLU.
    pub fn branch_a() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new(LayerKind::SplitterConnector)
                .output((batch, A[2])),

            LayerDesc::new(LayerKind::Linear)
                .input((batch, A[2]))
                .output((batch, A[3])),

            LayerDesc::new(LayerKind::ReLU)
                .input((batch, A[3]))
                .output(s(batch, A[3])),

            // Завершаем ветвь A и переключаемся на ветвь B:
            // активные порты становятся [3]
            LayerDesc::new(LayerKind::CombinerConnector)
                .input((batch, A[3])),
        ]
    }

    /// Ветвь B: 4 -> 3 с Sigmoid.
    pub fn branch_b() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new(LayerKind::SplitterConnector)
                .output((batch, A[4])),

            LayerDesc::new(LayerKind::Linear)
                .input((batch, A[4]))
                .output((batch, A[3])),

            LayerDesc::new(LayerKind::Sigmoid)
                .input((batch, A[3]))
                .output((batch, A[3])),

            // Объединяем ветви: CombinerConnector с двумя входами (3, 3) и одним выходом 3
            LayerDesc::new(LayerKind::CombinerConnector)
                .input((batch, A[3])),
        ]
    }

    /// Этап 2: Combiner и выходной Linear.
    pub fn stage2() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new(LayerKind::SplitterConnector)
                .output((batch, A[3]);((batch, A[3]))),

            LayerDesc::new(LayerKind::Combiner)
                .input((batch, A[3]);((batch, A[3]))),
                .output((batch, A[6])),

            LayerDesc::new(LayerKind::Linear)
                .input((batch, A[6]))
                .output((batch, A[3])),

            LayerDesc::new(LayerKind::Linear)
                .input((batch, A[3]))
                .output((batch, A[1])),
        ]
    }

    /// Полная модель, собирающая все четыре части.
    /// Эта функция не захватывает окружение, поэтому может быть передана в `TrainingPlan::model`.
    pub fn graph_model() -> Vec<LayerDesc> {
        stage1()
            .into_iter()
            .chain(branch_a())
            .chain(branch_b())
            .chain(stage2())
            .collect()
    }
}

// ═══════════════ Функция потерь ═══════════════
mod losses {
    use neurocore::loss_plan::{Aggregation, ElementChain, LossDesc, Square, Sub};

    /// MSE для одного выходного признака.
    pub fn mse() -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(Sub::new(1)))   // Sub с features=1
            .add(Box::new(Square));
        LossDesc::from_chain(chain, Aggregation::Mean, 2, 1, 1)
    }
}

// ═══════════════ Оптимизатор ═══════════════
mod optimizers {
    use neurocore::optimizer_plan::{OptCubeDesc, OptimizerDesc};

    pub fn sgd() -> OptimizerDesc {
        OptimizerDesc::new()
            .add(OptCubeDesc::ScaleGradient(0.01))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

// ═══════════════ Данные ═══════════════
fn train_data() -> Tensor2D {
    Tensor2D::new(vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ])
}

fn target_data() -> Tensor2D {
    Tensor2D::new(vec![vec![0.8], vec![0.2]])
}

// ═══════════════ План устройств ═══════════════
mod device_plan {
    use neurocore::device_plan::DevicePlan;

    pub fn plan() -> DevicePlan {
        DevicePlan::empty().cpu(0, 4).ram(0, 8192)
    }
}

// ═══════════════ План обучения ═══════════════
mod training_plan {
    use super::*;
    use neurocore::training_plan::plan::{DataSource, Initializer, TrainingPlan};

    pub fn plan() -> TrainingPlan {
        TrainingPlan::new()
            .model(models::graph_model)   // передаём fn-указатель
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

// ═══════════════ Точка входа ═══════════════
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
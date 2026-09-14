// examples/linear_test.rs
//
// Диагностический прогон автоэнкодера 4 → 4 с разными learning rate.
//
// Цель: выяснить, зависит ли фактический шаг оптимизатора от заданного lr.
//
// Если lr реально линейный — каждая строка ниже даст различный loss, и
// чем больше lr, тем меньше loss (до определённого предела, потом NaN).
//
// Если lr используется как "boolean" (0 = выкл, всё остальное = вкл с
// фиксированным коэффициентом) — все непустые значения дадут одинаковый
// финальный loss, а 0.0 — константу.

use neurocore::tensor::Tensor2D;

mod models {
    use neurocore::model_plan::{LayerKind, LayerDesc};
    use neurocore::shape;

    pub fn linear_model() -> Vec<LayerDesc> {
        vec![LayerDesc::new(LayerKind::Linear)
            .input(shape!(batch, A[4]))
            .output(shape!(batch, A[4]))]
    }
}

mod losses {
    use neurocore::loss_plan::{Aggregation, ElementChain, LossDesc, Square, Sub, SumColumns};

    pub fn mse() -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(Sub::new(4)))
            .add(Box::new(Square))
            .add(Box::new(SumColumns));
        LossDesc::from_chain(chain, Aggregation::Mean, 1, 4, 4)
    }
}

mod optimizers {
    use neurocore::optimizer_plan::{OptimizerDesc, OptCubeDesc};

    /// SGD с явно заданным learning rate.
    /// `lr` передаётся в ScaleGradient, а не берётся из дефолта.
    pub fn sgd(lr: f32) -> OptimizerDesc {
        OptimizerDesc::new()
            .add(OptCubeDesc::ScaleGradient(lr))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

fn data() -> Tensor2D {
    Tensor2D::new(vec![vec![1.0, 2.0, 3.0, 4.0]])
}

/// Общая фабрика плана обучения с параметром `lr`.
fn make_training_plan(lr: f32) -> neurocore::training_plan::TrainingPlan {
    use neurocore::training_plan::plan::{TrainingPlan, DataSource, Initializer};

    TrainingPlan::new()
        .model(models::linear_model)
        .loss(losses::mse())
        .optimizer(optimizers::sgd(lr))
        .epochs(100)
        .batch_size(1)
        .train_data(DataSource::from_tensor2d(data()))
        .init_weights(Initializer::RandomUniform { min: -0.1, max: 0.1 })
        .seed(42)
        .output_tensors(vec!["prediction".to_string()])
}

// ---------------------------------------------------------------------------
// По одной обёртке на каждое значение lr.
// `run_training!` принимает путь к функции без аргументов, поэтому
// каждый lr закреплён в своём модуле.
// ---------------------------------------------------------------------------

mod training_plan_lr_0 {
    use neurocore::training_plan::TrainingPlan;
    pub fn plan() -> TrainingPlan { super::make_training_plan(0.0) }
}
mod training_plan_lr_1e_9 {
    use neurocore::training_plan::TrainingPlan;
    pub fn plan() -> TrainingPlan { super::make_training_plan(1e-9) }
}
mod training_plan_lr_1e_6 {
    use neurocore::training_plan::TrainingPlan;
    pub fn plan() -> TrainingPlan { super::make_training_plan(1e-6) }
}
mod training_plan_lr_1e_4 {
    use neurocore::training_plan::TrainingPlan;
    pub fn plan() -> TrainingPlan { super::make_training_plan(1e-4) }
}
mod training_plan_lr_1e_2 {
    use neurocore::training_plan::TrainingPlan;
    pub fn plan() -> TrainingPlan { super::make_training_plan(1e-2) }
}
mod training_plan_lr_0p1 {
    use neurocore::training_plan::TrainingPlan;
    pub fn plan() -> TrainingPlan { super::make_training_plan(0.1) }
}
mod training_plan_lr_1p0 {
    use neurocore::training_plan::TrainingPlan;
    pub fn plan() -> TrainingPlan { super::make_training_plan(1.0) }
}

/// Единый device_plan: CPU 2 потока, RAM 8 ГБ.
mod device_plan {
    use neurocore::device_plan::DevicePlan;
    pub fn plan() -> DevicePlan {
        DevicePlan::empty().cpu(0, 2).ram(0, 8192)
    }
}

fn print_result(label: &str, r: &neurocore::training_plan::execution::TrainingResult) {
    println!(
        "{}  time={:.3}s | best_loss={:.6} @ epoch {} | zero_loss_epoch={:?}",
        label, r.training_time_secs, r.best_loss, r.best_epoch, r.zero_loss_epoch
    );
}

fn main() {
    println!();
    println!("=== LINEAR_TEST: lr sweep ===");
    println!();

    let r = neurocore::run_training!(training_plan_lr_0::plan, device = device_plan::plan);
    print_result("lr=0.0    ", &r);

    let r = neurocore::run_training!(training_plan_lr_1e_9::plan, device = device_plan::plan);
    print_result("lr=1e-9   ", &r);

    let r = neurocore::run_training!(training_plan_lr_1e_6::plan, device = device_plan::plan);
    print_result("lr=1e-6   ", &r);

    let r = neurocore::run_training!(training_plan_lr_1e_4::plan, device = device_plan::plan);
    print_result("lr=1e-4   ", &r);

    let r = neurocore::run_training!(training_plan_lr_1e_2::plan, device = device_plan::plan);
    print_result("lr=1e-2   ", &r);

    let r = neurocore::run_training!(training_plan_lr_0p1::plan, device = device_plan::plan);
    print_result("lr=0.1    ", &r);

    let r = neurocore::run_training!(training_plan_lr_1p0::plan, device = device_plan::plan);
    print_result("lr=1.0    ", &r);

    println!();
    println!("=== end lr sweep ===");
}
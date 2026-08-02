// examples/linear_test.rs
// Все 7 вариантов обучения выполняются через макрос run_training!.
// Библиотека автоматически управляет стеком GPU, RUSTFLAGS не требуется.
// Для всех вариантов задан seed = 42, гарантирующий одинаковые начальные веса.
// Выводится время обучения и эпоха с наименьшим (или первым нулевым) loss.

use std::time::Instant;
use neurocore::tensor::Tensor2D;

mod models {
    use neurocore::model_plan::{Dim, LayerDesc, LayerKind};
    pub fn linear_model() -> Vec<LayerDesc> {
        vec![LayerDesc::new("linear", LayerKind::Linear, Dim::Dim1)
            .input(Dim::Dim1, &[4])
            .output(Dim::Dim1, &[2])]
    }
}

mod losses {
    use neurocore::loss_plan::{Aggregation, ElementChain, LossDesc, Square, Sub};
    pub fn mse() -> LossDesc {
        let chain = ElementChain::new().add(Box::new(Sub)).add(Box::new(Square));
        LossDesc::from_chain(chain, Aggregation::Mean, 2, 1, 1)
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

fn data() -> Tensor2D {
    Tensor2D::new(vec![vec![1.0, 2.0, 3.0, 4.0]])
}

fn base_training() -> neurocore::training_plan::TrainingPlan {
    use neurocore::training_plan::plan::{TrainingPlan, DataSource, Initializer};
    TrainingPlan::new()
        .model(models::linear_model)
        .loss(losses::mse())
        .optimizer(optimizers::sgd())
        .epochs(100)
        .batch_size(1)
        .train_data(DataSource::from_tensor2d(data()))
        .init_weights(Initializer::RandomUniform { min: -0.1, max: 0.1 })
        .output_tensors(vec!["prediction".to_string()])
        .seed(42)   // воспроизводимость
}

// Определяем модули для каждого варианта
macro_rules! define_variant {
    ($vis:ident, $train_mod:ident, $dev_mod:ident, $dev_expr:expr) => {
        mod $train_mod {
            use neurocore::training_plan::TrainingPlan;
            pub fn plan() -> TrainingPlan { super::base_training() }
        }
        mod $dev_mod {
            use neurocore::device_plan::DevicePlan;
            pub fn plan() -> DevicePlan { $dev_expr }
        }
    }
}

define_variant!(v1, training_plan_v1, device_plan_v1, DevicePlan::empty().cpu(0, 1).ram(0, 8192));
define_variant!(v2, training_plan_v2, device_plan_v2, DevicePlan::empty().cpu(0, 4).ram(0, 8192));
define_variant!(v3, training_plan_v3, device_plan_v3, DevicePlan::empty().cpu(0, 2).ram(0, 8192).gpu(0).vram(0, 0, 4096));

// для варианта 4 (CPU -> GPU) нужно два плана устройств
mod training_plan_v4 {
    use neurocore::training_plan::TrainingPlan;
    pub fn plan() -> TrainingPlan { super::base_training() }
}
mod device_plan_v4_cpu {
    use neurocore::device_plan::DevicePlan;
    pub fn plan() -> DevicePlan { DevicePlan::empty().cpu(0, 1).ram(0, 8192) }
}
mod device_plan_v4_gpu {
    use neurocore::device_plan::DevicePlan;
    pub fn plan() -> DevicePlan { DevicePlan::empty().cpu(0, 2).ram(0, 8192).gpu(0).vram(0, 0, 4096) }
}

// вариант 5 (GPU -> CPU)
mod training_plan_v5 {
    use neurocore::training_plan::TrainingPlan;
    pub fn plan() -> TrainingPlan { super::base_training() }
}
mod device_plan_v5_gpu {
    use neurocore::device_plan::DevicePlan;
    pub fn plan() -> DevicePlan { DevicePlan::empty().cpu(0, 2).ram(0, 8192).gpu(0).vram(0, 0, 4096) }
}
mod device_plan_v5_cpu {
    use neurocore::device_plan::DevicePlan;
    pub fn plan() -> DevicePlan { DevicePlan::empty().cpu(0, 1).ram(0, 8192) }
}

define_variant!(v6, training_plan_v6, device_plan_v6, DevicePlan::empty().cpu(0, 4).ram(0, 8192).ssd(0, "neurocore_ssd_cache", 5000));
define_variant!(v7, training_plan_v7, device_plan_v7, DevicePlan::empty().cpu(0, 4).ram(0, 8192).gpu(0).vram(0, 0, 4096));

fn run_and_report(label: &str, train_fn: fn() -> neurocore::training_plan::TrainingPlan, dev_fn: fn() -> neurocore::device_plan::DevicePlan) {
    let start = Instant::now();
    let result = neurocore::run_training!(train_fn, device = dev_fn);
    let elapsed = start.elapsed();
    println!("{} - loss: {:.6}, time: {:.2?}", label, result.final_loss, elapsed);

    // найдём эпоху с минимальным loss (или первую нулевую)
    if let Some((epoch, &min_loss)) = result.loss_history.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
        if min_loss == 0.0 {
            // ищем первую эпоху с нулевым loss
            let first_zero = result.loss_history.iter().position(|&l| l == 0.0).unwrap_or(0);
            println!("  -> loss стал равен нулю на эпохе {}", first_zero);
        } else {
            println!("  -> наименьший loss = {:.6} на эпохе {}", min_loss, epoch);
        }
    }
}

fn main() {
    run_and_report("Вариант 1 (CPU 1 поток)", training_plan_v1::plan, device_plan_v1::plan);
    run_and_report("Вариант 2 (CPU 4 потока)", training_plan_v2::plan, device_plan_v2::plan);
    run_and_report("Вариант 3 (GPU)", training_plan_v3::plan, device_plan_v3::plan);

    // вариант 4: CPU затем GPU
    run_and_report("Вариант 4 (CPU этап)", training_plan_v4::plan, device_plan_v4_cpu::plan);
    run_and_report("Вариант 4 (GPU этап)", training_plan_v4::plan, device_plan_v4_gpu::plan);

    // вариант 5: GPU затем CPU
    run_and_report("Вариант 5 (GPU этап)", training_plan_v5::plan, device_plan_v5_gpu::plan);
    run_and_report("Вариант 5 (CPU этап)", training_plan_v5::plan, device_plan_v5_cpu::plan);

    run_and_report("Вариант 6 (CPU + SSD)", training_plan_v6::plan, device_plan_v6::plan);
    run_and_report("Вариант 7 (Параллельное CPU+GPU)", training_plan_v7::plan, device_plan_v7::plan);
}





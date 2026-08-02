// examples/linear_test.rs
// Демонстрация семи вариантов обучения с использованием модульных планов.
// Вариант 7 включает глубокий анализ (ProfileMode::Full) – вывод времени и памяти
// для каждого этапа и устройства.

use neurocore::tensor::Tensor2D;
use neurocore::training_plan::ProfileMode;

// ─── Модель ────────────────────────────────────────────────────────
mod models {
    use neurocore::model_plan::{Dim, LayerDesc, LayerKind};
    pub fn linear_model() -> Vec<LayerDesc> {
        vec![LayerDesc::new("linear", LayerKind::Linear, Dim::Dim1)
            .input(Dim::Dim1, &[4])
            .output(Dim::Dim1, &[2])]
    }
}

// ─── Функция потерь ────────────────────────────────────────────────
mod losses {
    use neurocore::loss_plan::{Aggregation, ElementChain, LossDesc, Square, Sub};
    pub fn mse() -> LossDesc {
        let chain = ElementChain::new().add(Box::new(Sub)).add(Box::new(Square));
        LossDesc::from_chain(chain, Aggregation::Mean, 2, 1, 1)
    }
}

// ─── Оптимизатор ───────────────────────────────────────────────────
mod optimizers {
    use neurocore::optimizer_plan::{OptimizerDesc, OptCubeDesc};
    pub fn sgd() -> OptimizerDesc {
        OptimizerDesc::new()
            .add(OptCubeDesc::ScaleGradient(0.01))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

// ─── Данные ────────────────────────────────────────────────────────
fn data() -> Tensor2D {
    Tensor2D::new(vec![vec![1.0, 2.0, 3.0, 4.0]])
}

// ─── Общий план обучения (общий для всех вариантов) ─────────────
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
        .seed(42)
        .output_tensors(vec!["prediction".to_string()])
}

// ═══════════════════════════════════════════════════════════════════
// Вариант 1: CPU (1 поток)
// ═══════════════════════════════════════════════════════════════════
mod training_plan_v1 {
    use neurocore::training_plan::TrainingPlan;
    pub fn plan() -> TrainingPlan { super::base_training() }
}
mod device_plan_v1 {
    use neurocore::device_plan::DevicePlan;
    pub fn plan() -> DevicePlan {
        DevicePlan::empty().cpu(0, 1).ram(0, 8192)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Вариант 2: CPU (4 потока)
// ═══════════════════════════════════════════════════════════════════
mod training_plan_v2 {
    use neurocore::training_plan::TrainingPlan;
    pub fn plan() -> TrainingPlan { super::base_training() }
}
mod device_plan_v2 {
    use neurocore::device_plan::DevicePlan;
    pub fn plan() -> DevicePlan {
        DevicePlan::empty().cpu(0, 4).ram(0, 8192)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Вариант 3: GPU (id 0)
// ═══════════════════════════════════════════════════════════════════
mod training_plan_v3 {
    use neurocore::training_plan::TrainingPlan;
    pub fn plan() -> TrainingPlan { super::base_training() }
}
mod device_plan_v3 {
    use neurocore::device_plan::DevicePlan;
    pub fn plan() -> DevicePlan {
        DevicePlan::empty().cpu(0, 2).ram(0, 8192).gpu(0).vram(0, 0, 4096)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Вариант 4: CPU (1 поток) → GPU (id 0)
// ═══════════════════════════════════════════════════════════════════
mod training_plan_v4 {
    use neurocore::training_plan::TrainingPlan;
    pub fn plan() -> TrainingPlan { super::base_training() }
}
mod device_plan_v4_cpu {
    use neurocore::device_plan::DevicePlan;
    pub fn plan() -> DevicePlan {
        DevicePlan::empty().cpu(0, 1).ram(0, 8192)
    }
}
mod device_plan_v4_gpu {
    use neurocore::device_plan::DevicePlan;
    pub fn plan() -> DevicePlan {
        DevicePlan::empty().cpu(0, 2).ram(0, 8192).gpu(0).vram(0, 0, 4096)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Вариант 5: GPU (id 0) → CPU (1 поток)
// ═══════════════════════════════════════════════════════════════════
mod training_plan_v5 {
    use neurocore::training_plan::TrainingPlan;
    pub fn plan() -> TrainingPlan { super::base_training() }
}
mod device_plan_v5_gpu {
    use neurocore::device_plan::DevicePlan;
    pub fn plan() -> DevicePlan {
        DevicePlan::empty().cpu(0, 2).ram(0, 8192).gpu(0).vram(0, 0, 4096)
    }
}
mod device_plan_v5_cpu {
    use neurocore::device_plan::DevicePlan;
    pub fn plan() -> DevicePlan {
        DevicePlan::empty().cpu(0, 1).ram(0, 8192)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Вариант 6: CPU (4 потока) + SSD-кэш
// ═══════════════════════════════════════════════════════════════════
mod training_plan_v6 {
    use neurocore::training_plan::TrainingPlan;
    pub fn plan() -> TrainingPlan { super::base_training() }
}
mod device_plan_v6 {
    use neurocore::device_plan::DevicePlan;
    pub fn plan() -> DevicePlan {
        DevicePlan::empty().cpu(0, 4).ram(0, 8192).ssd(0, "neurocore_ssd_cache", 5000)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Вариант 7: Параллельное CPU + GPU (с глубоким профилированием)
// ═══════════════════════════════════════════════════════════════════
mod training_plan_v7 {
    use neurocore::training_plan::{TrainingPlan, ProfileMode};
    pub fn plan() -> TrainingPlan {
        super::base_training()
            .profile(ProfileMode::Full)   // <-- включаем детальный сбор метрик
    }
}
mod device_plan_v7 {
    use neurocore::device_plan::DevicePlan;
    pub fn plan() -> DevicePlan {
        DevicePlan::empty().cpu(0, 4).ram(0, 8192).gpu(0).vram(0, 0, 4096)
    }
}

fn print_result(label: &str, r: &neurocore::training_plan::execution::TrainingResult) {
    println!(
        "{}  time={:.3}s | best_loss={:.6} @ epoch {} | zero_loss_epoch={:?}",
        label,
        r.training_time_secs,
        r.best_loss,
        r.best_epoch,
        r.zero_loss_epoch
    );
    if let Some(ref profile) = r.profile {
        println!("{}", profile.report());
    }
}

fn main() {
    // Вариант 1
    let r1 = neurocore::run_training!(training_plan_v1::plan, device = device_plan_v1::plan);
    print_result("V1 CPU 1t", &r1);

    // Вариант 2
    let r2 = neurocore::run_training!(training_plan_v2::plan, device = device_plan_v2::plan);
    print_result("V2 CPU 4t", &r2);

    // Вариант 3
    let r3 = neurocore::run_training!(training_plan_v3::plan, device = device_plan_v3::plan);
    print_result("V3 GPU   ", &r3);

    // Вариант 4: CPU → GPU
    let r4_cpu = neurocore::run_training!(training_plan_v4::plan, device = device_plan_v4_cpu::plan);
    print_result("V4a CPU ", &r4_cpu);
    let r4_gpu = neurocore::run_training!(training_plan_v4::plan, device = device_plan_v4_gpu::plan);
    print_result("V4b GPU ", &r4_gpu);

    // Вариант 5: GPU → CPU
    let r5_gpu = neurocore::run_training!(training_plan_v5::plan, device = device_plan_v5_gpu::plan);
    print_result("V5a GPU ", &r5_gpu);
    let r5_cpu = neurocore::run_training!(training_plan_v5::plan, device = device_plan_v5_cpu::plan);
    print_result("V5b CPU ", &r5_cpu);

    // Вариант 6
    let r6 = neurocore::run_training!(training_plan_v6::plan, device = device_plan_v6::plan);
    print_result("V6 SSD  ", &r6);

    // Вариант 7 – с детальным профилем
    let r7 = neurocore::run_training!(training_plan_v7::plan, device = device_plan_v7::plan);
    print_result("V7 CPU+GPU (profiled)", &r7);
}

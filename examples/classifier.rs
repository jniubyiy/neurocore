// examples/classifier.rs
// Классификатор на 2 класса (Dim1) с 7 вариантами обучения через run_training!.
// Модель: Linear(2->2) -> CrossEntropyWithLogits(2).
// Вариант 7 включает профилирование.

use neurocore::tensor::Tensor2D;
use neurocore::training_plan::ProfileMode;

// ═══════════════ Модель ═══════════════
mod model {
    use neurocore::model_plan::{LayerKind, LayerDesc};

    pub fn classifier() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new(LayerKind::Linear)
                .input((batch, A[2]))
                .output((batch, A[2])),
        ]
    }
}

// ═══════════════ Потери ═══════════════
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

// ═══════════════ Оптимизатор ═══════════════
mod optimizers {
    use neurocore::optimizer_plan::{OptimizerDesc, OptCubeDesc};

    pub fn sgd() -> OptimizerDesc {
        OptimizerDesc::new()
            .add(OptCubeDesc::ScaleGradient(0.5))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

// ═══════════════ Данные ═══════════════
fn train_data() -> Tensor2D {
    // два сэмпла: [1,2] и [2,1]
    Tensor2D::new(vec![vec![1.0, 2.0], vec![2.0, 1.0]])
}

fn target_data() -> Tensor2D {
    // классы: 0 и 1
    Tensor2D::new(vec![vec![0.0], vec![1.0]])
}

// ═══════════════ Общий план обучения ═══════════════
fn base_training() -> neurocore::training_plan::TrainingPlan {
    use neurocore::training_plan::plan::{TrainingPlan, DataSource, Initializer};

    TrainingPlan::new()
        .model(models::classifier)
        .loss(losses::cross_entropy())
        .optimizer(optimizers::sgd())
        .epochs(200)
        .batch_size(1)
        .train_data(DataSource::from_tensor2d(train_data()))
        .target_data(DataSource::from_tensor2d(target_data()))
        .init_weights(Initializer::RandomUniform {
            min: -0.1,
            max: 0.1,
        })
        .seed(42)
        .output_tensors(vec!["prediction".to_string()])
}

// Версия с профилированием для варианта 7
fn profiled_training() -> neurocore::training_plan::TrainingPlan {
    base_training().profile(ProfileMode::Full)
}

// ═══════════════ Макросы для генерации планов устройств ═══════════════
macro_rules! device_plan_v {
    ($name:ident, $cpu:expr, $ram:expr, $gpu:expr, $vram:expr, $ssd:expr) => {
        mod $name {
            use neurocore::device_plan::DevicePlan;
            pub fn plan() -> DevicePlan {
                let p = DevicePlan::empty()
                    .cpu(0, $cpu)
                    .ram(0, $ram);
                let p = if $gpu { p.gpu(0).vram(0, 0, $vram) } else { p };
                if $ssd {
                    p.ssd(0, "neurocore_ssd_cache", 5000)
                } else {
                    p
                }
            }
        }
    };
}

// Генерируем модули для всех 7 вариантов
device_plan_v!(device_plan_v1, 1, 8192, false, 0, false);
device_plan_v!(device_plan_v2, 4, 8192, false, 0, false);
device_plan_v!(device_plan_v3, 2, 8192, true, 4096, false);
device_plan_v!(device_plan_v4_cpu, 1, 8192, false, 0, false);
device_plan_v!(device_plan_v4_gpu, 2, 8192, true, 4096, false);
device_plan_v!(device_plan_v5_gpu, 2, 8192, true, 4096, false);
device_plan_v!(device_plan_v5_cpu, 1, 8192, false, 0, false);
device_plan_v!(device_plan_v6, 4, 8192, false, 0, true);
device_plan_v!(device_plan_v7, 4, 8192, true, 4096, false);

fn print_result(label: &str, r: &neurocore::training_plan::execution::TrainingResult) {
    println!(
        "{}  time={:.3}s | best_loss={:.6} @ epoch {} | zero_loss_epoch={:?}",
        label, r.training_time_secs, r.best_loss, r.best_epoch, r.zero_loss_epoch
    );
    if let Some(ref profile) = r.profile {
        println!("{}", profile.report());
    }
}

fn main() {
    // Вариант 1: CPU 1 поток
    let r1 = neurocore::run_training!(
        base_training,
        device = device_plan_v1::plan
    );
    print_result("V1 CPU1", &r1);

    // Вариант 2: CPU 4 потока
    let r2 = neurocore::run_training!(
        base_training,
        device = device_plan_v2::plan
    );
    print_result("V2 CPU4", &r2);

    // Вариант 3: GPU
    let r3 = neurocore::run_training!(
        base_training,
        device = device_plan_v3::plan
    );
    print_result("V3 GPU ", &r3);

    // Вариант 4: CPU -> GPU
    let r4a = neurocore::run_training!(
        base_training,
        device = device_plan_v4_cpu::plan
    );
    print_result("V4a CPU", &r4a);
    let r4b = neurocore::run_training!(
        base_training,
        device = device_plan_v4_gpu::plan
    );
    print_result("V4b GPU", &r4b);

    // Вариант 5: GPU -> CPU
    let r5a = neurocore::run_training!(
        base_training,
        device = device_plan_v5_gpu::plan
    );
    print_result("V5a GPU", &r5a);
    let r5b = neurocore::run_training!(
        base_training,
        device = device_plan_v5_cpu::plan
    );
    print_result("V5b CPU", &r5b);

    // Вариант 6: SSD
    let r6 = neurocore::run_training!(
        base_training,
        device = device_plan_v6::plan
    );
    print_result("V6 SSD", &r6);

    // Вариант 7: профилирование
    let r7 = neurocore::run_training!(
        profiled_training,
        device = device_plan_v7::plan
    );
    print_result("V7 Prof", &r7);
}






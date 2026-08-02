// examples/autoencoder.rs
// Автоэнкодер с одним скрытым слоем, Dim1 (Tensor2D).
// Все 7 вариантов обучения выполняются через макрос run_training!.
// Вариант 7 включает глубокий анализ времени и памяти.

use neurocore::tensor::Tensor2D;
use neurocore::training_plan::ProfileMode;

// ═══════════════ Модели ═══════════════
mod models {
    use neurocore::model_plan::{Dim, LayerDesc, LayerKind};
    pub fn encoder() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new("fc1", LayerKind::Linear, Dim::Dim1)
                .input(Dim::Dim1, &[4])
                .output(Dim::Dim1, &[2]),
            LayerDesc::new("sigm", LayerKind::Sigmoid, Dim::Dim1)
                .input(Dim::Dim1, &[2])
                .output(Dim::Dim1, &[2]),
        ]
    }
    pub fn decoder() -> Vec<LayerDesc> {
        vec![LayerDesc::new("fc2", LayerKind::Linear, Dim::Dim1)
            .input(Dim::Dim1, &[2])
            .output(Dim::Dim1, &[4])]
    }
}

// ═══════════════ Потери ═══════════════
mod losses {
    use neurocore::loss_plan::{Aggregation, ElementChain, LossDesc, Square, Sub};
    pub fn mse() -> LossDesc {
        let chain = ElementChain::new().add(Box::new(Sub)).add(Box::new(Square));
        LossDesc::from_chain(chain, Aggregation::Mean, 4, 1, 1)
    }
}

// ═══════════════ Оптимизатор ═══════════════
mod optimizers {
    use neurocore::optimizer_plan::{OptimizerDesc, OptCubeDesc};
    pub fn sgd_encoder() -> OptimizerDesc {
        OptimizerDesc::new().add(OptCubeDesc::ScaleGradient(0.01)).add(OptCubeDesc::ApplyUpdate)
    }
    pub fn sgd_decoder() -> OptimizerDesc {
        OptimizerDesc::new().add(OptCubeDesc::ScaleGradient(0.01)).add(OptCubeDesc::ApplyUpdate)
    }
}

// ═══════════════ Данные ═══════════════
fn data() -> Tensor2D {
    Tensor2D::new(vec![vec![1.0, 2.0, 3.0, 4.0]])
}

// ═══════════════ Общий план обучения ═══════════════
fn base_encoder_training() -> neurocore::training_plan::TrainingPlan {
    use neurocore::training_plan::plan::{TrainingPlan, DataSource, Initializer};
    TrainingPlan::new()
        .model(models::encoder)
        .loss(losses::mse())
        .optimizer(optimizers::sgd_encoder())
        .epochs(500)
        .batch_size(1)
        .train_data(DataSource::from_tensor2d(data()))
        .init_weights(Initializer::RandomUniform { min: -0.1, max: 0.1 })
        .seed(42)
        .output_tensors(vec!["prediction".to_string()])
}
fn base_decoder_training() -> neurocore::training_plan::TrainingPlan {
    use neurocore::training_plan::plan::{TrainingPlan, DataSource, Initializer};
    TrainingPlan::new()
        .model(models::decoder)
        .loss(losses::mse())
        .optimizer(optimizers::sgd_decoder())
        .epochs(500)
        .batch_size(1)
        .train_data(DataSource::from_tensor2d(data()))
        .init_weights(Initializer::RandomUniform { min: -0.1, max: 0.1 })
        .seed(42)
        .output_tensors(vec!["prediction".to_string()])
}

// Версии с профилированием для варианта 7
fn base_encoder_training_profiled() -> neurocore::training_plan::TrainingPlan {
    base_encoder_training().profile(ProfileMode::Full)
}
fn base_decoder_training_profiled() -> neurocore::training_plan::TrainingPlan {
    base_decoder_training().profile(ProfileMode::Full)
}

// Макросы для генерации модулей
macro_rules! training_variants {
    ($v:ident, $encoder_fn:ident, $decoder_fn:ident) => {
        mod $v {
            use neurocore::training_plan::TrainingPlan;
            pub fn encoder_plan() -> TrainingPlan { super::$encoder_fn() }
            pub fn decoder_plan() -> TrainingPlan { super::$decoder_fn() }
        }
    };
}

macro_rules! device_plan_v {
    ($name:ident, $cpu_threads:expr, $ram:expr, $gpu:expr, $vram:expr, $ssd:expr) => {
        mod $name {
            use neurocore::device_plan::DevicePlan;
            pub fn plan() -> DevicePlan {
                let p = DevicePlan::empty()
                    .cpu(0, $cpu_threads)
                    .ram(0, $ram);
                let p = if $gpu { p.gpu(0).vram(0, 0, $vram) } else { p };
                if $ssd { p.ssd(0, "neurocore_ssd_cache", 5000) } else { p }
            }
        }
    };
}

training_variants!(training_plan_v1, base_encoder_training, base_decoder_training);
device_plan_v!(device_plan_v1, 1, 8192, false, 0, false);

training_variants!(training_plan_v2, base_encoder_training, base_decoder_training);
device_plan_v!(device_plan_v2, 4, 8192, false, 0, false);

training_variants!(training_plan_v3, base_encoder_training, base_decoder_training);
device_plan_v!(device_plan_v3, 2, 8192, true, 4096, false);

training_variants!(training_plan_v4, base_encoder_training, base_decoder_training);
device_plan_v!(device_plan_v4_cpu, 1, 8192, false, 0, false);
device_plan_v!(device_plan_v4_gpu, 2, 8192, true, 4096, false);

training_variants!(training_plan_v5, base_encoder_training, base_decoder_training);
device_plan_v!(device_plan_v5_gpu, 2, 8192, true, 4096, false);
device_plan_v!(device_plan_v5_cpu, 1, 8192, false, 0, false);

training_variants!(training_plan_v6, base_encoder_training, base_decoder_training);
device_plan_v!(device_plan_v6, 4, 8192, false, 0, true);

// Вариант 7 с глубоким профилированием
training_variants!(training_plan_v7, base_encoder_training_profiled, base_decoder_training_profiled);
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
    // Вариант 1
    let r1 = neurocore::run_training!(training_plan_v1::encoder_plan, device = device_plan_v1::plan);
    print_result("V1 Enc CPU1", &r1);
    let r1d = neurocore::run_training!(training_plan_v1::decoder_plan, device = device_plan_v1::plan);
    print_result("V1 Dec CPU1", &r1d);

    // Вариант 2
    let r2 = neurocore::run_training!(training_plan_v2::encoder_plan, device = device_plan_v2::plan);
    print_result("V2 Enc CPU4", &r2);
    let r2d = neurocore::run_training!(training_plan_v2::decoder_plan, device = device_plan_v2::plan);
    print_result("V2 Dec CPU4", &r2d);

    // Вариант 3 (GPU)
    let r3 = neurocore::run_training!(training_plan_v3::encoder_plan, device = device_plan_v3::plan);
    print_result("V3 Enc GPU", &r3);
    let r3d = neurocore::run_training!(training_plan_v3::decoder_plan, device = device_plan_v3::plan);
    print_result("V3 Dec GPU", &r3d);

    // Вариант 4: CPU -> GPU
    let r4a = neurocore::run_training!(training_plan_v4::encoder_plan, device = device_plan_v4_cpu::plan);
    print_result("V4a Enc CPU", &r4a);
    let r4b = neurocore::run_training!(training_plan_v4::encoder_plan, device = device_plan_v4_gpu::plan);
    print_result("V4b Enc GPU", &r4b);

    // Вариант 5: GPU -> CPU
    let r5a = neurocore::run_training!(training_plan_v5::encoder_plan, device = device_plan_v5_gpu::plan);
    print_result("V5a Enc GPU", &r5a);
    let r5b = neurocore::run_training!(training_plan_v5::encoder_plan, device = device_plan_v5_cpu::plan);
    print_result("V5b Enc CPU", &r5b);

    // Вариант 6 (SSD)
    let r6 = neurocore::run_training!(training_plan_v6::encoder_plan, device = device_plan_v6::plan);
    print_result("V6 Enc SSD", &r6);

    // Вариант 7 (профилированный)
    let r7 = neurocore::run_training!(training_plan_v7::encoder_plan, device = device_plan_v7::plan);
    print_result("V7 Enc PROFILE", &r7);
    let r7d = neurocore::run_training!(training_plan_v7::decoder_plan, device = device_plan_v7::plan);
    print_result("V7 Dec PROFILE", &r7d);
}


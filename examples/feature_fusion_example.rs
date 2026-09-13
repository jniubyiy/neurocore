// examples/feature_fusion_example.rs
// Пример обучения модели с использованием слоя FeatureFusion.
// FeatureFusion — обучаемое глобальное агрегирование признаков с softmax-весами.
// Вход: (batch, in_features), выход: (batch, out_features).
// В данном примере модель учится предсказывать целевую переменную, которая
// является взвешенной суммой только первых двух признаков (x0, x1),
// игнорируя остальные (x2, x3). FeatureFusion автоматически подбирает веса.
// Демонстрирует несколько вариантов запуска: CPU с разным числом потоков,
// GPU, SSD, а также профилирование.

use neurocore::tensor::Tensor2D;
use neurocore::training_plan::ProfileMode;

mod models {
    use neurocore::model_plan::{LayerKind, LayerDesc};
    use neurocore::shape;

    pub fn feature_fusion_model() -> Vec<LayerDesc> {
        vec![
            // FeatureFusion агрегирует 4 входных признака в один выходной
            LayerDesc::new(LayerKind::FeatureFusion)
                .input(shape!(batch, A[4]))
                .output(shape!(batch, A[1])),
        ]
    }
}

mod losses {
    use neurocore::loss_plan::{
        Aggregation, ElementChain, LossDesc, Square, Sub,
    };

    pub fn mse(batch_size: usize) -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(Sub::new(1)))
            .add(Box::new(Square));
        LossDesc::from_chain(chain, Aggregation::Mean, batch_size, 1, 1)
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

/// Генерирует обучающие данные: входные векторы размерности 4.
/// Целевая переменная = 2.0 * x0 - 3.0 * x1 + небольшой шум.
/// Признаки x2 и x3 не влияют на цель.
fn generate_data(num_samples: usize, seed: u64) -> (Tensor2D, Tensor2D) {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let mut inputs = Vec::with_capacity(num_samples);
    let mut targets = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let x0: f32 = rng.gen_range(-1.0..1.0);
        let x1: f32 = rng.gen_range(-1.0..1.0);
        let x2: f32 = rng.gen_range(-1.0..1.0);
        let x3: f32 = rng.gen_range(-1.0..1.0);
        let noise: f32 = rng.gen_range(-0.1..0.1);
        let target = 2.0 * x0 - 3.0 * x1 + noise;
        inputs.push(vec![x0, x1, x2, x3]);
        targets.push(vec![target]);
    }

    (Tensor2D::new(inputs), Tensor2D::new(targets))
}

fn base_training() -> neurocore::training_plan::TrainingPlan {
    use neurocore::training_plan::plan::{TrainingPlan, DataSource, Initializer};

    let num_samples = 100;
    let batch_size = 20;
    let (train_x, train_y) = generate_data(num_samples, 42);

    TrainingPlan::new()
        .model(models::feature_fusion_model)
        .loss(losses::mse(batch_size))
        .optimizer(optimizers::sgd())
        .epochs(300)
        .batch_size(batch_size)
        .train_data(DataSource::from_tensor2d(train_x))
        .target_data(DataSource::from_tensor2d(train_y))
        .init_weights(Initializer::RandomUniform {
            min: -0.1,
            max: 0.1,
        })
        .seed(42)
        .output_tensors(vec!["prediction".to_string()])
}

fn profiled_training() -> neurocore::training_plan::TrainingPlan {
    base_training().profile(ProfileMode::Full)
}

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

device_plan_v!(device_plan_v1, 2, 8192, false, 0, false);
device_plan_v!(device_plan_v2, 4, 8192, false, 0, false);
device_plan_v!(device_plan_v3, 2, 8192, true, 4096, false);
device_plan_v!(device_plan_v4_cpu, 2, 8192, false, 0, false);
device_plan_v!(device_plan_v4_gpu, 2, 8192, true, 4096, false);
device_plan_v!(device_plan_v5_gpu, 2, 8192, true, 4096, false);
device_plan_v!(device_plan_v5_cpu, 2, 8192, false, 0, false);
device_plan_v!(device_plan_v6, 4, 8192, false, 0, true);
device_plan_v!(device_plan_v7, 4, 8192, true, 4096, false);

/// Печатает только краткую итоговую строку результата.
/// Подробный отчёт профиля (`ProfileResult::report()`) намеренно не выводится,
/// чтобы не засорять консоль. При необходимости отчёт доступен через
/// `r.profile.as_ref().map(|p| p.report())`.
fn print_result(label: &str, r: &neurocore::training_plan::execution::TrainingResult) {
    println!(
        "{}  time={:.3}s | best_loss={:.6} @ epoch {} | zero_loss_epoch={:?}",
        label, r.training_time_secs, r.best_loss, r.best_epoch, r.zero_loss_epoch
    );
}

fn main() {
    let r1 = neurocore::run_training!(
        base_training,
        device = device_plan_v1::plan
    );
    print_result("V1 CPU2", &r1);

    let r2 = neurocore::run_training!(
        base_training,
        device = device_plan_v2::plan
    );
    print_result("V2 CPU4", &r2);

    let r3 = neurocore::run_training!(
        base_training,
        device = device_plan_v3::plan
    );
    print_result("V3 GPU ", &r3);

    let r4a = neurocore::run_training!(
        base_training,
        device = device_plan_v4_cpu::plan
    );
    print_result("V4a CPU2", &r4a);

    let r4b = neurocore::run_training!(
        base_training,
        device = device_plan_v4_gpu::plan
    );
    print_result("V4b GPU", &r4b);

    let r5a = neurocore::run_training!(
        base_training,
        device = device_plan_v5_gpu::plan
    );
    print_result("V5a GPU", &r5a);

    let r5b = neurocore::run_training!(
        base_training,
        device = device_plan_v5_cpu::plan
    );
    print_result("V5b CPU2", &r5b);

    let r6 = neurocore::run_training!(
        base_training,
        device = device_plan_v6::plan
    );
    print_result("V6 SSD", &r6);

    let r7 = neurocore::run_training!(
        profiled_training,
        device = device_plan_v7::plan
    );
    print_result("V7 Prof", &r7);
}
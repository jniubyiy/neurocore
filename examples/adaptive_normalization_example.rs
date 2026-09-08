// examples/adaptive_normalization_example.rs
// Пример обучения классификатора с использованием слоя AdaptiveNormalization.
// Слой комбинирует LayerNorm, RMSNorm и BatchNorm с обучаемыми весами выбора
// для каждого признака. Это позволяет автоматически подобрать подходящий
// тип нормализации для признаков с сильно различающимися масштабами.
// Демонстрирует несколько вариантов запуска: CPU с разным числом потоков,
// GPU, SSD, а также профилирование.

use neurocore::tensor::Tensor2D;
use neurocore::training_plan::ProfileMode;

mod models {
    use neurocore::model_plan::{LayerKind, LayerDesc};
    use neurocore::shape;

    pub fn adaptive_norm_classifier() -> Vec<LayerDesc> {
        vec![
            // Входной линейный слой расширяет признаки
            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[4]))
                .output(shape!(batch, A[8])),

            // Адаптивная нормализация выбирает стратегию для каждого признака
            LayerDesc::new(LayerKind::AdaptiveNormalization)
                .input(shape!(batch, A[8]))
                .output(shape!(batch, A[8])),

            // Нелинейность
            LayerDesc::new(LayerKind::ReLU)
                .input(shape!(batch, A[8]))
                .output(shape!(batch, A[8])),

            // Выходной слой на 2 класса
            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[8]))
                .output(shape!(batch, A[2])),
        ]
    }
}

mod losses {
    use neurocore::loss_plan::{
        Aggregation, CrossEntropyWithLogits, ElementChain, LossDesc,
    };

    pub fn cross_entropy(batch_size: usize) -> LossDesc {
        let num_classes = 2;
        let chain = ElementChain::new()
            .add(Box::new(CrossEntropyWithLogits::new(num_classes)));
        LossDesc::from_chain(chain, Aggregation::Mean, batch_size, num_classes, 1)
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

/// Генерирует обучающие данные: два класса, признаки с разными масштабами.
/// Признак 0 имеет малый масштаб (~0.001), признак 1 – обычный (~1.0),
/// признак 2 – большой (~100.0), признак 3 – средний (~0.1).
/// Класс 0: средние близки к нулю; класс 1: средние сдвинуты пропорционально масштабу.
fn generate_data(num_samples_per_class: usize, seed: u64) -> (Tensor2D, Tensor2D) {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let mut inputs = Vec::with_capacity(num_samples_per_class * 2);
    let mut targets = Vec::with_capacity(num_samples_per_class * 2);

    // Масштабы для каждого из 4 признаков
    let scales = [0.001_f32, 1.0_f32, 100.0_f32, 0.1_f32];
    // Сдвиг для класса 1 (в единицах своего масштаба)
    let shifts = [1.0_f32, 1.0_f32, 1.0_f32, 1.0_f32];

    for class in 0..2 {
        for _ in 0..num_samples_per_class {
            let mut sample = Vec::with_capacity(4);
            for i in 0..4 {
                // Для класса 0 среднее 0, для класса 1 – сдвиг * масштаб
                let mean = if class == 0 { 0.0 } else { shifts[i] * scales[i] };
                // Добавляем шум в пределах половины масштаба
                let noise = rng.gen_range(-0.2..0.2) * scales[i];
                sample.push(mean + noise);
            }
            inputs.push(sample);
            targets.push(vec![class as f32]);
        }
    }

    (Tensor2D::new(inputs), Tensor2D::new(targets))
}

fn base_training() -> neurocore::training_plan::TrainingPlan {
    use neurocore::training_plan::plan::{TrainingPlan, DataSource, Initializer};

    let num_samples_per_class = 50;
    let total_samples = num_samples_per_class * 2;
    let batch_size = 32;
    let (train_x, train_y) = generate_data(num_samples_per_class, 42);

    TrainingPlan::new()
        .model(models::adaptive_norm_classifier)
        .loss(losses::cross_entropy(batch_size))
        .optimizer(optimizers::sgd())
        .epochs(200)
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
    let r1 = neurocore::run_training!(
        base_training,
        device = device_plan_v1::plan
    );
    print_result("V1 CPU1", &r1);

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
    print_result("V4a CPU", &r4a);

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
    print_result("V5b CPU", &r5b);

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
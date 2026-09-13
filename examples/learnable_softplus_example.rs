// examples/learnable_softplus_example.rs
// Пример обучения автоэнкодера с использованием слоя LearnableSoftplus.
// LearnableSoftplus — это Softplus с обучаемыми порогом θ и масштабом β,
// что позволяет адаптировать активацию к диапазону положительных данных.
// Модель учится восстанавливать входные данные с экспоненциальным распределением.
// Демонстрирует несколько вариантов запуска: CPU с разным числом потоков,
// GPU, SSD, а также профилирование.
//
// Переменная окружения NEUROCORE_VARIANT (необязательная):
//   v1  — только V1 CPU2
//   v2  — только V2 CPU4
//   v3  — только V3 GPU
//   v4a — только V4a CPU2
//   v4b — только V4b GPU
//   v5a — только V5a GPU
//   v5b — только V5b CPU2
//   v6  — только V6 SSD
//   v7  — только V7 Prof
//   all (или не задано) — все варианты последовательно.
//
// Изоляция одного варианта позволяет исключить накопление глобального
// состояния между прогонами (счётчики, scheduler, модели mini-model) и
// понять, воспроизводится ли проблема в конкретном варианте изолированно.

use neurocore::tensor::Tensor2D;
use neurocore::training_plan::ProfileMode;

mod models {
    use neurocore::model_plan::{LayerKind, LayerDesc};
    use neurocore::shape;

    pub fn learnable_softplus_autoencoder() -> Vec<LayerDesc> {
        vec![
            // Кодировщик: расширяем размерность
            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[4]))
                .output(shape!(batch, A[8])),

            // Активация LearnableSoftplus с обучаемыми параметрами
            LayerDesc::new(LayerKind::LearnableSoftplus)
                .input(shape!(batch, A[8]))
                .output(shape!(batch, A[8])),

            // Декодер: возвращаем исходную размерность
            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[8]))
                .output(shape!(batch, A[4])),
        ]
    }
}

mod losses {
    use neurocore::loss_plan::{
        Aggregation, ElementChain, LossDesc, Square, Sub, SumColumns,
    };

    pub fn mse(batch_size: usize) -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(Sub::new(4)))
            .add(Box::new(Square))
            .add(Box::new(SumColumns));
        LossDesc::from_chain(chain, Aggregation::Mean, batch_size, 4, 4)
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

/// Генерирует обучающие данные: векторы размерности 4 с положительными
/// элементами, распределёнными примерно по экспоненциальному закону
/// (смещённые вправо). Целевые значения равны входным (автоэнкодер).
fn generate_data(num_samples: usize, seed: u64) -> (Tensor2D, Tensor2D) {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let mut inputs = Vec::with_capacity(num_samples);
    let mut targets = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let mut sample = Vec::with_capacity(4);
        for _ in 0..4 {
            // Простая имитация экспоненциального распределения:
            // используем -ln(1 - u) / lambda, lambda = 1.0
            let u: f32 = rng.gen_range(0.001..0.999);
            let value = -((1.0 - u).ln());
            // Ограничим значения для стабильности обучения
            let value = value.min(5.0).max(0.01);
            sample.push(value);
        }
        inputs.push(sample.clone());
        targets.push(sample);
    }

    (Tensor2D::new(inputs), Tensor2D::new(targets))
}

fn base_training() -> neurocore::training_plan::TrainingPlan {
    use neurocore::training_plan::plan::{TrainingPlan, DataSource, Initializer};

    let num_samples = 40;
    let batch_size = 10;
    let (train_x, train_y) = generate_data(num_samples, 42);

    TrainingPlan::new()
        .model(models::learnable_softplus_autoencoder)
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

fn run_variant(variant: &str) {
    match variant {
        "v1" => {
            let r = neurocore::run_training!(base_training, device = device_plan_v1::plan);
            print_result("V1 CPU2", &r);
        }
        "v2" => {
            let r = neurocore::run_training!(base_training, device = device_plan_v2::plan);
            print_result("V2 CPU4", &r);
        }
        "v3" => {
            let r = neurocore::run_training!(base_training, device = device_plan_v3::plan);
            print_result("V3 GPU ", &r);
        }
        "v4a" => {
            let r = neurocore::run_training!(base_training, device = device_plan_v4_cpu::plan);
            print_result("V4a CPU2", &r);
        }
        "v4b" => {
            let r = neurocore::run_training!(base_training, device = device_plan_v4_gpu::plan);
            print_result("V4b GPU", &r);
        }
        "v5a" => {
            let r = neurocore::run_training!(base_training, device = device_plan_v5_gpu::plan);
            print_result("V5a GPU", &r);
        }
        "v5b" => {
            let r = neurocore::run_training!(base_training, device = device_plan_v5_cpu::plan);
            print_result("V5b CPU2", &r);
        }
        "v6" => {
            let r = neurocore::run_training!(base_training, device = device_plan_v6::plan);
            print_result("V6 SSD", &r);
        }
        "v7" => {
            let r = neurocore::run_training!(profiled_training, device = device_plan_v7::plan);
            print_result("V7 Prof", &r);
        }
        other => panic!(
            "Unknown NEUROCORE_VARIANT = {:?}. \
             Use one of: v1, v2, v3, v4a, v4b, v5a, v5b, v6, v7, all",
            other
        ),
    }
}

fn main() {
    let variant = std::env::var("NEUROCORE_VARIANT")
        .unwrap_or_else(|_| "all".to_string());

    if variant == "all" {
        for v in ["v1", "v2", "v3", "v4a", "v4b", "v5a", "v5b", "v6", "v7"] {
            run_variant(v);
        }
    } else {
        run_variant(&variant);
    }
}
// examples/batch_renorm_example.rs
// Пример обучения автоэнкодера с использованием слоя BatchRenorm1d.
// BatchRenorm1d — улучшенная версия BatchNorm с обучаемыми поправками r и d,
// которые уменьшают расхождение между статистиками train и test.
// Особенно полезен при обучении на маленьких батчах (batch_size = 4),
// где обычный BatchNorm может быть нестабильным.
// Демонстрирует несколько вариантов запуска: CPU с разным числом потоков,
// GPU, SSD, а также профилирование.

use neurocore::tensor::Tensor2D;
use neurocore::training_plan::ProfileMode;

mod models {
    use neurocore::model_plan::{LayerKind, LayerDesc};
    use neurocore::shape;

    pub fn batch_renorm_autoencoder() -> Vec<LayerDesc> {
        vec![
            // Кодировщик: расширяем размерность
            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[4]))
                .output(shape!(batch, A[8])),

            // Нормализация с обучаемыми поправками
            LayerDesc::new(LayerKind::BatchRenorm1d)
                .input(shape!(batch, A[8]))
                .output(shape!(batch, A[8])),

            // Нелинейность
            LayerDesc::new(LayerKind::ReLU)
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

/// Генерирует обучающие данные: случайные векторы размерности 4
/// с элементами из равномерного распределения [-1, 1].
/// Целевые значения равны входным (автоэнкодер).
fn generate_data(num_samples: usize, seed: u64) -> (Tensor2D, Tensor2D) {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let mut inputs = Vec::with_capacity(num_samples);
    let mut targets = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let sample: Vec<f32> = (0..4)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        inputs.push(sample.clone());
        targets.push(sample);
    }

    (Tensor2D::new(inputs), Tensor2D::new(targets))
}

fn base_training() -> neurocore::training_plan::TrainingPlan {
    use neurocore::training_plan::plan::{TrainingPlan, DataSource, Initializer};

    let num_samples = 40;
    // Маленький размер батча, чтобы продемонстрировать преимущества BatchRenorm
    let batch_size = 4;
    let (train_x, train_y) = generate_data(num_samples, 42);

    TrainingPlan::new()
        .model(models::batch_renorm_autoencoder)
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
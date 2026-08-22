// examples/soft_sparse_gate_example.rs
// Обновлённый пример обучения слоя SoftSparseGate.
// Слой учится обнулять слабый шум и пропускать полезный сигнал.
// Демонстрирует несколько вариантов запуска: CPU, GPU, SSD и профилирование.

use neurocore::tensor::Tensor2D;
use neurocore::training_plan::ProfileMode;

mod models {
    use neurocore::model_plan::{LayerKind, LayerDesc};
    use neurocore::shape;

    pub fn soft_sparse_gate_model() -> Vec<LayerDesc> {
        vec![
            // Только SoftSparseGate: обучаемые пороги решают, что считать шумом
            LayerDesc::new(LayerKind::SoftSparseGate)
                .input(shape!(batch, A[4]))
                .output(shape!(batch, A[4]))
                .extra(vec![0.5]),   // температура
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
            .add(OptCubeDesc::ScaleGradient(0.005))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

/// Генерирует обучающие данные: полезный сигнал (большие значения) и шум (малые).
/// Цель: шум обнулён, сигнал сохранён.
fn generate_data(num_samples: usize, seed: u64) -> (Tensor2D, Tensor2D) {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let mut inputs = Vec::with_capacity(num_samples);
    let mut targets = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let mut sample = Vec::with_capacity(4);
        let mut target = Vec::with_capacity(4);
        for _ in 0..4 {
            if rng.gen_bool(0.5) {
                // Полезный сигнал: значение по модулю 2.0..4.0
                let signal = rng.gen_range(2.0..4.0) * if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
                sample.push(signal);
                target.push(signal);
            } else {
                // Шум: значение по модулю < 0.4
                let noise = rng.gen_range(-0.4..0.4);
                sample.push(noise);
                target.push(0.0);
            }
        }
        inputs.push(sample);
        targets.push(target);
    }

    (Tensor2D::new(inputs), Tensor2D::new(targets))
}

fn base_training() -> neurocore::training_plan::TrainingPlan {
    use neurocore::training_plan::plan::{TrainingPlan, DataSource, Initializer};

    let num_samples = 20;
    let (train_x, train_y) = generate_data(num_samples, 42);

    TrainingPlan::new()
        .model(models::soft_sparse_gate_model)
        .loss(losses::mse(num_samples))
        .optimizer(optimizers::sgd())
        .epochs(300)
        .batch_size(num_samples)
        .train_data(DataSource::from_tensor2d(train_x))
        .target_data(DataSource::from_tensor2d(train_y))
        // Пороги инициализируем около 1.0, чтобы с самого начала отделять сигнал от шума
        .init_weights(Initializer::RandomUniform {
            min: 0.5,
            max: 1.5,
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
    let r1 = neurocore::run_training!(base_training, device = device_plan_v1::plan);
    print_result("V1 CPU1", &r1);

    let r2 = neurocore::run_training!(base_training, device = device_plan_v2::plan);
    print_result("V2 CPU4", &r2);

    let r3 = neurocore::run_training!(base_training, device = device_plan_v3::plan);
    print_result("V3 GPU ", &r3);

    let r4a = neurocore::run_training!(base_training, device = device_plan_v4_cpu::plan);
    print_result("V4a CPU", &r4a);

    let r4b = neurocore::run_training!(base_training, device = device_plan_v4_gpu::plan);
    print_result("V4b GPU", &r4b);

    let r5a = neurocore::run_training!(base_training, device = device_plan_v5_gpu::plan);
    print_result("V5a GPU", &r5a);

    let r5b = neurocore::run_training!(base_training, device = device_plan_v5_cpu::plan);
    print_result("V5b CPU", &r5b);

    let r6 = neurocore::run_training!(base_training, device = device_plan_v6::plan);
    print_result("V6 SSD", &r6);

    let r7 = neurocore::run_training!(profiled_training, device = device_plan_v7::plan);
    print_result("V7 Prof", &r7);
}
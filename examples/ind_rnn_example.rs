// examples/ind_rnn_example.rs
// Пример обучения автоэнкодера с использованием слоя IndRNN (Independent RNN).
// IndRNN — рекуррентный слой, в котором каждый скрытый нейрон обновляется
// независимо (без полносвязной рекуррентной матрицы). Это уменьшает проблему
// затухания/взрыва градиента и делает обучение более устойчивым.
// В данном примере модель учится восстанавливать последовательности:
// вход и выход имеют одинаковую размерность (seq_len * input_dim).
// Демонстрирует несколько вариантов запуска: CPU с разным числом потоков,
// GPU, SSD, а также профилирование.
//
// ВАЖНО (о данных):
//   Входы генерируются в [0, 1), а не [-1, 1). Причина — слой IndRNN
//   использует ReLU как встроенную активацию. На данных, содержащих
//   отрицательные значения, identity-задача (target = input) недостижима
//   в принципе: ReLU не может выдать отрицательное число, поэтому MSE
//   упирается в теоретический предел E[max(0,−x)²]·features ≈ 3.33 при
//   равномерном [-1,1]. Наблюдаемый в исходной версии loss ≈ 3.22 —
//   это не «застой из-за бага», а архитектурный предел выбранной модели.
//
//   На положительных данных (x ∈ [0, 1)) этот предел исчезает: identity
//   достигается точно при W = I, u = 0, b = 0, и MSE сходится к 0.

use neurocore::tensor::Tensor2D;
use neurocore::training_plan::ProfileMode;

mod models {
    use neurocore::model_plan::{LayerKind, LayerDesc};
    use neurocore::shape;

    pub fn ind_rnn_autoencoder() -> Vec<LayerDesc> {
        let input_dim = 4;
        let seq_len = 5;

        vec![
            // IndRNN с заданными размерностями
            LayerDesc::new(LayerKind::IndRNN)
                .input(shape!(batch, A[seq_len * input_dim]))   // 20 признаков
                .output(shape!(batch, A[seq_len * input_dim]))  // 20 признаков
                .extra(vec![input_dim as f32, seq_len as f32]),
        ]
    }
}

mod losses {
    use neurocore::loss_plan::{
        Aggregation, ElementChain, LossDesc, Square, Sub, SumColumns,
    };

    pub fn mse(batch_size: usize, feature_count: usize) -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(Sub::new(feature_count)))
            .add(Box::new(Square))
            .add(Box::new(SumColumns));
        LossDesc::from_chain(chain, Aggregation::Mean, batch_size, feature_count, feature_count)
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

/// Генерирует обучающие данные: случайные последовательности.
/// Каждый пример — вектор из seq_len * input_dim элементов в [0, 1).
/// Целевые значения равны входным (автоэнкодер).
///
/// Значения строго неотрицательные — это ключевое условие для
/// сходимости модели с ReLU-активацией (см. комментарий в шапке файла).
fn generate_data(
    num_samples: usize,
    seq_len: usize,
    input_dim: usize,
    seed: u64,
) -> (Tensor2D, Tensor2D) {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let feature_count = seq_len * input_dim;
    let mut inputs = Vec::with_capacity(num_samples);
    let mut targets = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let sample: Vec<f32> = (0..feature_count)
            .map(|_| rng.gen_range(0.0..1.0))
            .collect();
        inputs.push(sample.clone());
        targets.push(sample);
    }

    (Tensor2D::new(inputs), Tensor2D::new(targets))
}

fn base_training() -> neurocore::training_plan::TrainingPlan {
    use neurocore::training_plan::plan::{TrainingPlan, DataSource, Initializer};

    let seq_len = 5;
    let input_dim = 4;
    let feature_count = seq_len * input_dim;
    let num_samples = 40;
    let batch_size = 10;
    let (train_x, train_y) = generate_data(num_samples, seq_len, input_dim, 42);

    TrainingPlan::new()
        .model(models::ind_rnn_autoencoder)
        .loss(losses::mse(batch_size, feature_count))
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

// ВНИМАНИЕ: минимальное число CPU-потоков — 2 (см. DevicePlan::cpu).
// Один поток уходит под управление, второй — на вычисления.
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
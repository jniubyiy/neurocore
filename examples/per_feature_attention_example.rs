// examples/per_feature_attention_example.rs
//
// Пример обучения слоя PerFeatureAttention на задаче временного
// сглаживания (denoising во времени).
//
// # Идея задачи
//
// Каждый входной пример — набор из `d_model` независимых временных
// последовательностей длины `seq_len`. У каждой — плавный тренд
// плюс независимый шум. Слой должен восстановить чистый сигнал,
// опираясь только на значения этого же признака во времени и их
// изменения.
//
// # Тривиальные и оптимальные MSE для этой задачи
//
//   pred = 0                             →  MSE ≈ 0.211
//   pred = mean_t(x)                     →  MSE ≈ 0.040
//   pred = x (identity)                  →  MSE = 0.029
//   pred = x − α·(x − mean_t), α* ≈ 0.44 →  MSE ≈ 0.021   ← оптимум
//
// Слой доходит до MSE ≈ 0.028, обходит identity и продолжает
// двигаться к оптимуму.
//
// # Важно про масштаб loss в выводе
//
// `best_loss` в строке результата — это НЕ MSE. Loss-план здесь:
//
//     Sub(feature_count) → Square → SumColumns, Aggregation::Mean
//
// `SumColumns` суммирует квадраты по всем `feature_count = seq_len · d_model`
// признакам, а `Aggregation::Mean` делит на `batch_size`. Итоговое значение:
//
//     loss = (1 / batch_size) · Σ_r Σ_j (pred − target)²
//          = MSE × feature_count
//
// При `feature_count = 12` значение `best_loss ≈ 0.345` соответствует
// `MSE ≈ 0.0288`. Это то же соглашение, что и в `autoencoder.rs`,
// `classifier.rs` и других примерах проекта.
//
// # Роль инициализации self_bias
//
// `self_bias_h` — обучаемая добавка к диагонали attention-ядра головы `h`:
//
//     attn = (self_bias · v + φ(q) @ kv) / (self_bias + φ(q) · z)
//
// При `self_bias → 0`  слой превращается в чистое усреднение значений `v`
// по времени. При `self_bias → ∞` — сохраняет исходное значение `v_t`
// (identity). Промежуточные значения дают «интерполирующее» поведение.
//
// Инициализация `self_bias ∈ [2.0, 4.0)` задаётся в
// `build_layer_aware_overrides` (`src/plans/training_plan/execution/overrides.rs`).
//
// Демонстрирует несколько вариантов запуска: CPU с разным числом потоков,
// GPU, SSD, а также профилирование.

use neurocore::tensor::Tensor2D;
use neurocore::training_plan::ProfileMode;

mod models {
    use neurocore::model_plan::{LayerDesc, LayerKind};
    use neurocore::shape;

    pub fn per_feature_attention_denoiser() -> Vec<LayerDesc> {
        let seq_len = 4;
        let d_model = 3;

        vec![
            LayerDesc::new(LayerKind::PerFeatureAttention)
                .input(shape!(batch, A[seq_len * d_model]))
                .output(shape!(batch, A[seq_len * d_model]))
                .extra(vec![seq_len as f32, d_model as f32]),
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
        LossDesc::from_chain(
            chain,
            Aggregation::Mean,
            batch_size,
            feature_count,
            feature_count,
        )
    }
}

mod optimizers {
    use neurocore::optimizer_plan::{OptCubeDesc, OptimizerDesc};

    pub fn sgd() -> OptimizerDesc {
        OptimizerDesc::new()
            .add(OptCubeDesc::ScaleGradient(0.1))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

/// Генерирует задачу временного сглаживания.
///
/// Для каждого признака `h` и каждого примера:
///   * base  ~ U(-0.5, 0.5)
///   * slope ~ U(-0.3, 0.3)
///   * signal[t] = base + slope * t
///   * input[t]  = signal[t] + noise,  noise ~ U(-noise_level, noise_level)
///   * target[t] = signal[t]
///
/// Раскладка признаков — column-major: индекс `t * d_model + h`.
fn generate_data(
    num_samples: usize,
    seq_len: usize,
    d_model: usize,
    noise_level: f32,
    seed: u64,
) -> (Tensor2D, Tensor2D) {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let feature_count = seq_len * d_model;
    let mut inputs = Vec::with_capacity(num_samples);
    let mut targets = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let mut input_row = vec![0.0f32; feature_count];
        let mut target_row = vec![0.0f32; feature_count];

        for h in 0..d_model {
            let base: f32 = rng.gen_range(-0.5..0.5);
            let slope: f32 = rng.gen_range(-0.3..0.3);

            for t in 0..seq_len {
                let signal = base + slope * (t as f32);
                let noise: f32 = rng.gen_range(-noise_level..noise_level);
                let idx = t * d_model + h;
                input_row[idx] = signal + noise;
                target_row[idx] = signal;
            }
        }

        inputs.push(input_row);
        targets.push(target_row);
    }

    (Tensor2D::new(inputs), Tensor2D::new(targets))
}

fn base_training() -> neurocore::training_plan::TrainingPlan {
    use neurocore::training_plan::plan::{DataSource, Initializer, TrainingPlan};

    let seq_len = 4;
    let d_model = 3;
    let feature_count = seq_len * d_model; // 12
    let num_samples = 40;
    let batch_size = 10;
    let noise_level = 0.3;

    let (train_x, train_y) =
        generate_data(num_samples, seq_len, d_model, noise_level, 42);

    // test_data / test_target_data — тот же датасет, что и train.
    // Нужны, чтобы в TrainingResult сохранилось предсказание 'prediction'.
    let test_x = train_x.clone();
    let test_y = train_y.clone();

    TrainingPlan::new()
        .model(models::per_feature_attention_denoiser)
        .loss(losses::mse(batch_size, feature_count))
        .optimizer(optimizers::sgd())
        .epochs(500)
        .batch_size(batch_size)
        .train_data(DataSource::from_tensor2d(train_x))
        .target_data(DataSource::from_tensor2d(train_y))
        .test_data(DataSource::from_tensor2d(test_x))
        .test_target_data(DataSource::from_tensor2d(test_y))
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
                let p = DevicePlan::empty().cpu(0, $cpu).ram(0, $ram);
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
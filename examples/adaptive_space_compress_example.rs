// examples/adaptive_space_compress_example.rs
//
// Демонстрация слоя AdaptiveSpaceCompress через TrainingPlan.
//
// # Идея примера
//
// Есть 10 разных примеров. Каждый — вектор разной длины (features)
// и своего характера распределения:
//
//   0) Uniform [-1, 1]           — длина 16
//   1) Gaussian N(0, 1)          — длина 32
//   2) Exponential               — длина 48
//   3) Sine                      — длина 64
//   4) Constant + noise          — длина 96
//   5) Linear trend + noise      — длина 128
//   6) Sparse                    — длина 192
//   7) Binomial (0/1)            — длина 256
//   8) Step function             — длина 384
//   9) Mixture of two Gaussians  — длина 512
//
// Все 10 примеров идут через ОДИН слой AdaptiveSpaceCompress, который
// сжимает каждый вход в вектор фиксированной длины OUT_FEATURES = 10.
//
// # Как работает ragged в TrainingPlan
//
// `Tensor2D::ragged` хранит примеры разной длины: все строки физически
// паддятся нулями до `max_len`, а реальные длины лежат в `sample_lens`.
// `DataSource::batch` сохраняет `sample_lens` в нарезке. `GraphV2::forward`
// передаёт их в job, а `AdaptiveSpaceCompress` обрабатывает только
// реальную часть каждой строки. Из-за разной длины примеров
// `execute_v2` автоматически устанавливает `batch_size = 1` для
// ragged-данных.
//
// # Что показывает пример
//
//   * слой принимает входы разной длины (16…512) без переаллокации
//     параметров — `param_len` не зависит от `in_features`;
//   * выход всегда фиксированной длины (OUT_FEATURES = 10);
//   * после обучения каждое сжатие сходится к своему one-hot таргету;
//   * обычный `TrainingPlan` + `run_training!` работают с ragged-данными
//     без изменений пользовательского API.

use neurocore::tensor::Tensor2D;
use neurocore::training_plan::ProfileMode;

// ============================================================================
// Константы задачи
// ============================================================================

const OUT_FEATURES: usize = 10;
const P_MAX: usize = 8;
const NUM_CLASSES: usize = 10;

/// Длины входов: 10 примеров разной длины.
const LENS: [usize; NUM_CLASSES] = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512];

// ============================================================================
// Модель
// ============================================================================

mod models {
    use neurocore::model_plan::{LayerDesc, LayerKind};
    use neurocore::shape;

    use super::{OUT_FEATURES, P_MAX};

    pub fn adaptive_space_compress_model() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new(LayerKind::AdaptiveSpaceCompress)
                // Номинальный вход — 512 (максимальная длина в датасете).
                // Реальный размер берётся из input.cols() в forward;
                // `param_len` от него не зависит.
                .input(shape!(batch, A[512]))
                .output(shape!(batch, A[OUT_FEATURES]))
                .extra(vec![P_MAX as f32]),
        ]
    }
}

// ============================================================================
// Loss: MSE(сжатие, one-hot)
// ============================================================================

mod losses {
    use neurocore::loss_plan::{
        Aggregation, ElementChain, LossDesc, Square, Sub, SumColumns,
    };

    use super::OUT_FEATURES;

    pub fn mse() -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(Sub::new(OUT_FEATURES)))
            .add(Box::new(Square))
            .add(Box::new(SumColumns));
        // batch_size здесь не важен для векторного LossDesc — итоговое
        // значение в `execute_v2` пересчитывается по фактическому батчу.
        // Ставим 1: при ragged обучение идёт по одному примеру за шаг.
        LossDesc::from_chain(
            chain,
            Aggregation::Mean,
            1,
            OUT_FEATURES,
            OUT_FEATURES,
        )
    }
}

// ============================================================================
// Оптимизатор
// ============================================================================

mod optimizers {
    use neurocore::optimizer_plan::{OptCubeDesc, OptimizerDesc};

    pub fn sgd() -> OptimizerDesc {
        OptimizerDesc::new()
            .add(OptCubeDesc::ScaleGradient(0.05))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

// ============================================================================
// Данные
// ============================================================================

/// Генерирует 10 разных примеров (разные длины, разные распределения)
/// и one-hot таргеты к ним.
fn make_dataset() -> (Tensor2D, Tensor2D) {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42);
    let mut inputs: Vec<Vec<f32>> = Vec::with_capacity(NUM_CLASSES);
    let mut targets: Vec<Vec<f32>> = Vec::with_capacity(NUM_CLASSES);

    for (class, &len) in LENS.iter().enumerate() {
        let mut v = Vec::with_capacity(len);
        match class {
            // 0. Uniform [-1, 1]
            0 => {
                for _ in 0..len { v.push(rng.gen_range(-1.0..1.0)); }
            }
            // 1. Gaussian N(0, 1) через Box-Muller
            1 => {
                for _ in 0..len {
                    let u1: f32 = rng.gen_range(1e-6..1.0);
                    let u2: f32 = rng.gen_range(0.0..1.0);
                    let r = (-2.0 * u1.ln()).sqrt();
                    let theta = 2.0 * std::f32::consts::PI * u2;
                    v.push(r * theta.cos());
                }
            }
            // 2. Exponential (λ = 0.5)
            2 => {
                for _ in 0..len {
                    let u: f32 = rng.gen_range(1e-6..1.0);
                    v.push(-u.ln() * 0.5);
                }
            }
            // 3. Sine
            3 => {
                for i in 0..len {
                    let t = i as f32 / len as f32;
                    v.push((2.0 * std::f32::consts::PI * 3.0 * t).sin());
                }
            }
            // 4. Constant + noise
            4 => {
                for _ in 0..len {
                    v.push(0.5 + rng.gen_range(-0.05..0.05));
                }
            }
            // 5. Linear trend + noise
            5 => {
                for i in 0..len {
                    let t = i as f32 / len as f32;
                    v.push(t + rng.gen_range(-0.02..0.02));
                }
            }
            // 6. Sparse (90% нулей, 10% всплесков)
            6 => {
                for _ in 0..len {
                    if rng.gen::<f32>() < 0.10 {
                        v.push(rng.gen_range(0.5..1.5));
                    } else {
                        v.push(0.0);
                    }
                }
            }
            // 7. Binomial (0/1)
            7 => {
                for _ in 0..len {
                    v.push(if rng.gen_bool(0.3) { 1.0 } else { 0.0 });
                }
            }
            // 8. Step function
            8 => {
                let half = len / 2;
                for i in 0..len {
                    v.push(if i < half { 0.0 } else { 1.0 });
                }
            }
            // 9. Mixture of two Gaussians
            9 => {
                for _ in 0..len {
                    let center = if rng.gen_bool(0.5) { -1.5 } else { 1.5 };
                    let u1: f32 = rng.gen_range(1e-6..1.0);
                    let u2: f32 = rng.gen_range(0.0..1.0);
                    let r = (-2.0 * u1.ln()).sqrt();
                    let theta = 2.0 * std::f32::consts::PI * u2;
                    v.push(center + 0.3 * r * theta.cos());
                }
            }
            _ => unreachable!(),
        }
        inputs.push(v);

        let mut t = vec![0.0f32; OUT_FEATURES];
        t[class] = 1.0;
        targets.push(t);
    }

    (
        Tensor2D::ragged(inputs),
        Tensor2D::new(targets),  // targets dense (одинаковая длина = OUT_FEATURES)
    )
}

// ============================================================================
// TrainingPlan
// ============================================================================

mod training_plan {
    use neurocore::training_plan::plan::{DataSource, Initializer, TrainingPlan};

    use super::*;

    pub fn plan() -> TrainingPlan {
        let (train_x, train_y) = make_dataset();

        // Важно: train_data — ragged (`sample_lens` заполнен). `execute_v2`
        // сам выставит batch_size = 1, потому что примеры разной длины.
        TrainingPlan::new()
            .model(models::adaptive_space_compress_model)
            .loss(losses::mse())
            .optimizer(optimizers::sgd())
            .epochs(400)
            .batch_size(1)
            .train_data(DataSource::from_tensor2d(train_x))
            .target_data(DataSource::from_tensor2d(train_y))
            .init_weights(Initializer::RandomUniform { min: -0.1, max: 0.1 })
            .seed(42)
            .output_tensors(vec!["prediction".to_string()])
    }

    pub fn profiled_plan() -> TrainingPlan {
        plan().profile(ProfileMode::Full)
    }
}

// ============================================================================
// DevicePlan
// ============================================================================

mod device_plan {
    use neurocore::device_plan::DevicePlan;

    pub fn plan() -> DevicePlan {
        DevicePlan::empty().cpu(0, 4).ram(0, 8192)
    }
}

// ============================================================================
// main
// ============================================================================

fn main() {
    println!("=== AdaptiveSpaceCompress example (through TrainingPlan) ===");
    println!(
        "10 samples with different lengths: {:?}",
        LENS
    );
    println!(
        "Model: AdaptiveSpaceCompress(out_features = {}, p_max = {})",
        OUT_FEATURES, P_MAX
    );
    println!();

    let result = neurocore::run_training!(
        training_plan::plan,
        device = device_plan::plan
    );

    println!();
    println!("=== Training finished ===");
    println!(
        "time = {:.3}s, best_loss = {:.6} @ epoch {}",
        result.training_time_secs, result.best_loss, result.best_epoch
    );
    println!();

    // Финальная диагностика: печатаем предсказание (сжатие) для каждого
    // примера и argmax. Цель — убедиться, что разные примеры дают разные
    // сжатия и сходятся к своим one-hot.
    if let Some(pred) = result.tensors.get("prediction") {
        let (inputs, _targets) = make_dataset();
        let sample_lens = inputs.sample_lens().unwrap();

        let pred_flat = pred.to_flat();
        println!("=== Final compressions ===");
        println!();
        for class in 0..NUM_CLASSES {
            let row: Vec<f32> = pred_flat
                [class * OUT_FEATURES..(class + 1) * OUT_FEATURES]
                .to_vec();

            let argmax = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            let target = {
                let mut t = vec![0.0f32; OUT_FEATURES];
                t[class] = 1.0;
                t
            };

            println!(
                "class {} (len = {:4}, type = {}):",
                class,
                sample_lens[class],
                sample_type_name(class)
            );
            println!("  target:      {}", format_row(&target));
            println!("  compression: {}", format_row(&row));
            println!(
                "  argmax = {}, expected = {}{}",
                argmax,
                class,
                if argmax == class { "  ✓" } else { "  ✗" }
            );
            println!();
        }
    }
}

// ============================================================================
// Хелперы вывода
// ============================================================================

fn sample_type_name(class: usize) -> &'static str {
    match class {
        0 => "uniform",
        1 => "gaussian",
        2 => "exponential",
        3 => "sine",
        4 => "const+noise",
        5 => "linear trend",
        6 => "sparse",
        7 => "binomial",
        8 => "step",
        9 => "gaussian mixture",
        _ => "unknown",
    }
}

fn format_row(v: &[f32]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{:+.3}", x)).collect();
    format!("[{}]", parts.join(", "))
}
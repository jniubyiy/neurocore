// examples/linear_attention_example.rs
//
// Пример обучения LinearAttention (multi-head) на задаче денойзинга через
// агрегацию.
//
// # Идея задачи
//
// LinearAttention перераспределяет информацию между токенами, поэтому
// естественная для него задача — не identity (сохранить вход как есть),
// а агрегация: собрать согласованный сигнал из нескольких зашумлённых
// наблюдений.
//
// Каждая выборка — вектор констант c длины d_model. Все seq_len токенов
// содержат одну и ту же константу плюс независимый шум:
//
//     input[r, t, j]  = c[r, j] + noise[r, t, j]
//     target[r, t, j] = c[r, j]
//
// Слой должен научиться усреднять информацию по токенам и подавлять шум.
//
// # Роль multi-head
//
// min_heads = 1, max_heads = 2. d_model = 4, значит d_head = d_model /
// max_heads = 2. Слой стартует с одной активной головы (h_soft ≈ 1) и
// может по мере обучения «вырастить» вторую. Обучаемый self_bias позволяет
// голове при необходимости выучить сохранение уникальной информации.
//
// Ожидаемая динамика loss:
//   Старт:      ~5.8   (все параметры случайны, шум большой)
//   Середина:   ~1.0   (модель начала усреднять)
//   Финал:      ~0.12  (усреднение по 4 токенам даёт 4-кратное подавление
//                       шума по дисперсии)

use neurocore::tensor::Tensor2D;
use neurocore::training_plan::ProfileMode;

mod models {
    use neurocore::model_plan::{LayerKind, LayerDesc};
    use neurocore::shape;

    pub fn linear_attention_denoiser() -> Vec<LayerDesc> {
        let seq_len = 4;
        let d_model = 4;      // должно делиться на max_heads
        let min_heads = 1;
        let max_heads = 2;    // d_head = d_model / max_heads = 2

        vec![
            LayerDesc::new(LayerKind::LinearAttention)
                .input(shape!(batch, A[seq_len * d_model]))
                .output(shape!(batch, A[seq_len * d_model]))
                .extra(vec![
                    seq_len as f32,
                    d_model as f32,
                    min_heads as f32,
                    max_heads as f32,
                ]),
        ]
    }
}

mod losses {
    // MSE без деления на feature_count:
    //   loss = (1/batch_size) · Σ_r Σ_j (pred[r,j] - target[r,j])²
    //
    // То есть loss = MSE_per_element · feature_count.
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
    use neurocore::optimizer_plan::{OptimizerDesc, OptCubeDesc};

    pub fn sgd() -> OptimizerDesc {
        OptimizerDesc::new()
            .add(OptCubeDesc::ScaleGradient(0.05))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

/// Генерирует данные для задачи денойзинга.
///
/// Каждая выборка — вектор констант `c` длины `d_model`. Все токены в
/// примере содержат одну и ту же константу плюс независимый шум.
///
/// Задача слоя — восстановить `c` на всех токенах, используя усреднение
/// информации по последовательности.
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
        // Вектор констант для этой выборки.
        let c: Vec<f32> = (0..d_model).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let mut input_row = Vec::with_capacity(feature_count);
        let mut target_row = Vec::with_capacity(feature_count);

        // Раскладка признаков — column-major: индекс признака `t*d + j`.
        for _t in 0..seq_len {
            for j in 0..d_model {
                let noise: f32 = rng.gen_range(-noise_level..noise_level);
                input_row.push(c[j] + noise);
                target_row.push(c[j]);
            }
        }
        inputs.push(input_row);
        targets.push(target_row);
    }

    (Tensor2D::new(inputs), Tensor2D::new(targets))
}

fn base_training() -> neurocore::training_plan::TrainingPlan {
    use neurocore::training_plan::plan::{TrainingPlan, DataSource, Initializer};

    let seq_len = 4;
    let d_model = 4;
    let feature_count = seq_len * d_model;   // 16
    let num_samples = 100;
    let batch_size = 10;
    let noise_level = 0.3;

    let (train_x, train_y) = generate_data(num_samples, seq_len, d_model, noise_level, 42);

    TrainingPlan::new()
        .model(models::linear_attention_denoiser)
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
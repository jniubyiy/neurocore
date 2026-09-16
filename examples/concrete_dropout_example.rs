// examples/concrete_dropout_example.rs
// Пример обучения автоэнкодера с использованием слоя ConcreteDropout.
// ConcreteDropout — это dropout с обучаемой вероятностью удержания элемента,
// основанный на Concrete (Gumbel-Softmax) релаксации Bernoulli.
// Вероятность удержания p = sigmoid(logit_p) обучается вместе с другими параметрами.
// В данном примере модель склонна к переобучению из-за большого скрытого слоя,
// а ConcreteDropout автоматически подбирает оптимальную силу регуляризации.
// Демонстрирует несколько вариантов запуска: CPU с разным числом потоков,
// GPU, SSD, а также профилирование.
//
// # Диагностика
//
// Переменная окружения NEUROCORE_DEBUG_CDROPOUT_EXAMPLE=1 включает
// дополнительный вывод:
//   * первые 3 строки обучающих данных и таргетов;
//   * сводка (min/max/mean/L2) по предсказаниям модели после каждого
//     варианта обучения;
//   * первые 3 строки предсказаний после каждого варианта.
//
// Переменная окружения NEUROCORE_DEBUG_CDROPOUT=1 (обрабатывается внутри
// слоя ConcreteDropout) включает подробную покомпонентную диагностику
// forward/backward конкретного слоя.
//
// # Параметры конкретного примера
//
// Температура Gumbel-Softmax выбрана 1.0 (а не 0.1). При T = 0.1 аргумент
// сигмоиды a = (logit_p + log(u) - log(1-u)) / T выходит в зону |a| > 50,
// и sigmoid'(a) ≈ 0. Это делает градиент по logit_p практически нулевым
// (см. статью Maddison et al. 2016, Jang et al. 2016 — оптимальная T для
// Gumbel-Softmax релаксации близка к 1.0).
//
// Инициализация ±0.5 — адекватный диапазон для Linear(4→16) с точки зрения
// Xavier-подхода (std ~ sqrt(2/fan_in)). При меньшем масштабе выход первого
// Linear имеет std ~ 0.09, тогда как целевые значения имеют std ~ 0.59 —
// модели требуется долго «раскачивать» веса, чтобы попасть в target.
//
// lr = 0.05 подобран так, чтобы logit_p и веса Linear смещались заметно
// за 300 эпох.

use neurocore::compute_manager::DynamicTensor;
use neurocore::tensor::Tensor2D;
use neurocore::training_plan::ProfileMode;

mod models {
    use neurocore::model_plan::{LayerKind, LayerDesc};
    use neurocore::shape;

    pub fn concrete_dropout_autoencoder() -> Vec<LayerDesc> {
        vec![
            // Кодировщик: сильно расширяем размерность, создавая риск переобучения
            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[4]))
                .output(shape!(batch, A[16])),

            // Dropout с обучаемой вероятностью.
            //
            // extra = [temperature, seed]. Температура 1.0 — оптимальная для
            // градиентного сигнала через Gumbel-Softmax (см. шапку файла).
            LayerDesc::new(LayerKind::ConcreteDropout)
                .input(shape!(batch, A[16]))
                .output(shape!(batch, A[16]))
                .extra(vec![1.0, 42.0]),

            // Нелинейность
            LayerDesc::new(LayerKind::ReLU)
                .input(shape!(batch, A[16]))
                .output(shape!(batch, A[16])),

            // Декодер: возвращаем исходную размерность
            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[16]))
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
            .add(OptCubeDesc::ScaleGradient(0.05))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

/// Генерирует обучающие данные: случайные векторы размерности 4
/// с элементами из диапазона [-1, 1]. Целевые значения равны входным
/// (автоэнкодер). Маленький набор данных провоцирует переобучение.
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

    let num_samples = 30;
    let batch_size = 30; // весь датасет одним батчем
    let (train_x, train_y) = generate_data(num_samples, 42);

    TrainingPlan::new()
        .model(models::concrete_dropout_autoencoder)
        .loss(losses::mse(batch_size))
        .optimizer(optimizers::sgd())
        .epochs(300)
        .batch_size(batch_size)
        .train_data(DataSource::from_tensor2d(train_x))
        .target_data(DataSource::from_tensor2d(train_y))
        .init_weights(Initializer::RandomUniform {
            min: -0.5,
            max: 0.5,
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

// ============================================================================
// Диагностика примерa
// ============================================================================

static CD_EX_DEBUG: once_cell::sync::Lazy<bool> =
    once_cell::sync::Lazy::new(|| {
        std::env::var("NEUROCORE_DEBUG_CDROPOUT_EXAMPLE").is_ok()
    });

/// Компактная статистика по произвольному срезу f32.
struct ExStats {
    len: usize,
    min: f32,
    max: f32,
    mean: f32,
    l2: f64,
    nan: usize,
    inf: usize,
}

fn ex_stats(data: &[f32]) -> ExStats {
    if data.is_empty() {
        return ExStats {
            len: 0,
            min: f32::NAN,
            max: f32::NAN,
            mean: f32::NAN,
            l2: 0.0,
            nan: 0,
            inf: 0,
        };
    }
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sq_sum = 0.0f64;
    let mut nan_cnt = 0usize;
    let mut inf_cnt = 0usize;
    for &v in data {
        if v.is_nan() { nan_cnt += 1; continue; }
        if v.is_infinite() { inf_cnt += 1; continue; }
        if v < mn { mn = v; }
        if v > mx { mx = v; }
        sum += v as f64;
        sq_sum += (v as f64) * (v as f64);
    }
    let finite = data.len().saturating_sub(nan_cnt + inf_cnt);
    let mean = if finite > 0 { (sum / finite as f64) as f32 } else { f32::NAN };
    ExStats {
        len: data.len(),
        min: mn,
        max: mx,
        mean,
        l2: sq_sum.sqrt(),
        nan: nan_cnt,
        inf: inf_cnt,
    }
}

impl ExStats {
    fn to_string(&self) -> String {
        format!(
            "len={}, min={:.6}, max={:.6}, mean={:.6}, l2={:.6}, nan={}, inf={}",
            self.len, self.min, self.max, self.mean, self.l2, self.nan, self.inf
        )
    }
}

/// Печатает только краткую итоговую строку результата.
/// Подробный отчёт профиля (`ProfileResult::report()`) намеренно не выводится,
/// чтобы не засорять консоль. При необходимости отчёт доступен через
/// `r.profile.as_ref().map(|p| p.report())`.
fn print_result(label: &str, r: &neurocore::training_plan::execution::TrainingResult) {
    println!(
        "{}  time={:.3}s | best_loss={:.6} @ epoch {} | zero_loss_epoch={:?}",
        label, r.training_time_secs, r.best_loss, r.best_epoch, r.zero_loss_epoch
    );

    if !*CD_EX_DEBUG {
        return;
    }

    // Дополнительная диагностика предсказаний.
    if let Some(pred) = r.tensors.get("prediction") {
        match pred {
            DynamicTensor::Dim1(t2d) => {
                let flat: Vec<f32> = t2d.data.iter().flatten().copied().collect();
                let s = ex_stats(&flat);
                println!("    [pred stats] {}", s.to_string());
                println!("    [pred first 3 rows]:");
                for row in t2d.data.iter().take(3) {
                    println!("      {:?}", row);
                }
            }
            other => {
                let flat = other.to_flat();
                let s = ex_stats(&flat);
                println!("    [pred stats] {}", s.to_string());
            }
        }
    } else {
        println!("    [pred stats] <no 'prediction' tensor in result>");
    }
}

fn dump_data_and_targets() {
    if !*CD_EX_DEBUG {
        return;
    }
    let (x, y) = generate_data(30, 42);
    let x_flat: Vec<f32> = x.data.iter().flatten().copied().collect();
    let y_flat: Vec<f32> = y.data.iter().flatten().copied().collect();
    let xs = ex_stats(&x_flat);
    let ys = ex_stats(&y_flat);

    println!("=========== DEBUG INPUT DATA ===========");
    println!("  dims: {} x {}", x.dim1, x.dim2);
    println!("  x stats: {}", xs.to_string());
    println!("  y stats: {}", ys.to_string());
    println!("  first 3 rows of x:");
    for row in x.data.iter().take(3) {
        println!("    {:?}", row);
    }
    println!("  first 3 rows of y:");
    for row in y.data.iter().take(3) {
        println!("    {:?}", row);
    }
    println!("========================================");
}

fn main() {
    dump_data_and_targets();

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
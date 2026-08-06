// examples_large/mnist_binary_32x32.rs
// Классификатор MNIST (32x32 бинарных изображений) через TrainingPlan.
// Два варианта обучения: CPU (16 потоков) и GPU (id 0).
// Демонстрирует обучение с разными train_data / target_data,
// а также раздельное тестирование с test_data и test_target_data.

use neurocore::tensor::Tensor2D;
use rand::Rng;

// ═══════════════ Модель ═══════════════
mod models {
    use neurocore::model_plan::{LayerKind, LayerDesc};
    use neurocore::shape;

    pub fn mnist_classifier() -> Vec<LayerDesc> {
        let img_size = 32;
        let input_dim = img_size * img_size;  // 1024
        let hidden1 = 512;
        let hidden2 = 256;
        let num_classes = 10;

        vec![
            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[input_dim]))
                .output(shape!(batch, A[hidden1])),
            LayerDesc::new(LayerKind::ReLU)
                .input(shape!(batch, A[hidden1]))
                .output(shape!(batch, A[hidden1])),
            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[hidden1]))
                .output(shape!(batch, A[hidden2])),
            LayerDesc::new(LayerKind::ReLU)
                .input(shape!(batch, A[hidden2]))
                .output(shape!(batch, A[hidden2])),
            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[hidden2]))
                .output(shape!(batch, A[num_classes])),
            LayerDesc::new(LayerKind::Softmax)
                .input(shape!(batch, A[num_classes]))
                .output(shape!(batch, A[num_classes])),
        ]
    }
}

// ═══════════════ Потери ═══════════════
mod losses {
    use neurocore::loss_plan::{
        Aggregation, CrossEntropyWithLogits, ElementChain, LossDesc,
    };

    /// Кросс‑энтропия для классификации.
    /// pred_features = num_classes (логиты), target_features = 1 (индекс класса)
    pub fn cross_entropy_desc(num_classes: usize, batch_size: usize) -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(CrossEntropyWithLogits::new(num_classes)));
        LossDesc::from_chain(chain, Aggregation::Mean, batch_size, num_classes, 1)
    }
}

// ═══════════════ Оптимизатор ═══════════════
mod optimizers {
    use neurocore::optimizer_plan::{OptimizerDesc, OptCubeDesc};

    pub fn sgd(lr: f32) -> OptimizerDesc {
        OptimizerDesc::new()
            .add(OptCubeDesc::ScaleGradient(lr))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

// ═══════════════ Генерация синтетического датасета ═══════════════
fn generate_dataset(num_samples: usize, img_size: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = rand::thread_rng();
    let mut images = Vec::with_capacity(num_samples);
    let mut labels = Vec::with_capacity(num_samples);

    let templates: Vec<Vec<Vec<f32>>> = (0..10)
        .map(|digit| {
            let mut img = vec![vec![0.0f32; img_size]; img_size];
            match digit {
                0 => {
                    let cx = img_size as f32 / 2.0;
                    let cy = img_size as f32 / 2.0;
                    let r = img_size as f32 / 2.0 - 2.0;
                    for i in 0..img_size {
                        for j in 0..img_size {
                            let dx = i as f32 - cx;
                            let dy = j as f32 - cy;
                            let dist = (dx * dx + dy * dy).sqrt();
                            if (dist - r).abs() < 2.0 {
                                img[i][j] = 1.0;
                            }
                        }
                    }
                }
                1 => {
                    for i in 0..img_size {
                        img[i][img_size / 2] = 1.0;
                    }
                }
                2 => {
                    for j in 0..img_size {
                        img[0][j] = 1.0;
                        img[img_size - 1][j] = 1.0;
                    }
                    for i in 0..img_size {
                        img[i][i] = 1.0;
                    }
                }
                3 => {
                    for i in 0..img_size {
                        img[i][0] = 1.0;
                        img[i][img_size - 1] = 1.0;
                    }
                    for j in 0..img_size {
                        img[img_size / 2][j] = 1.0;
                    }
                }
                4 => {
                    for i in 0..img_size {
                        img[i][img_size / 3] = 1.0;
                    }
                    for j in 0..img_size {
                        img[2 * img_size / 3][j] = 1.0;
                    }
                }
                5 => {
                    for i in 0..img_size {
                        img[i][0] = 1.0;
                    }
                    for j in 0..img_size {
                        img[img_size / 2][j] = 1.0;
                        img[img_size - 1][j] = 1.0;
                    }
                }
                6 => {
                    for i in 0..img_size {
                        img[i][0] = 1.0;
                        img[i][img_size - 1] = 1.0;
                        img[0][i] = 1.0;
                        img[img_size / 2][i] = 1.0;
                    }
                }
                7 => {
                    for j in 0..img_size {
                        img[0][j] = 1.0;
                    }
                    for i in 0..img_size {
                        img[i][img_size - 1 - i] = 1.0;
                    }
                }
                8 => {
                    for i in 0..img_size {
                        img[i][0] = 1.0;
                        img[i][img_size - 1] = 1.0;
                        img[0][i] = 1.0;
                        img[img_size - 1][i] = 1.0;
                    }
                }
                9 => {
                    for i in 0..img_size {
                        img[i][img_size - 1] = 1.0;
                    }
                    for j in 0..img_size {
                        img[0][j] = 1.0;
                        img[img_size / 2][j] = 1.0;
                    }
                    for i in 0..img_size {
                        img[i][i] = 1.0;
                    }
                }
                _ => {}
            }
            img
        })
        .collect();

    for _ in 0..num_samples {
        let digit = rng.gen_range(0..10);
        let mut img = templates[digit].clone();
        // Добавим шум: инвертируем случайные пиксели с вероятностью 10%
        for i in 0..img_size {
            for j in 0..img_size {
                if rng.gen::<f32>() < 0.1 {
                    img[i][j] = 1.0 - img[i][j];
                }
            }
        }
        let flat: Vec<f32> = img.into_iter().flatten().collect();
        images.push(flat);
        labels.push(vec![digit as f32]);
    }
    (images, labels)
}

// ═══════════════ Планы устройств ═══════════════
mod device_plan_cpu {
    use neurocore::device_plan::DevicePlan;
    pub fn plan() -> DevicePlan {
        DevicePlan::empty().cpu(0, 16).ram(0, 8192)
    }
}

mod device_plan_gpu {
    use neurocore::device_plan::DevicePlan;
    pub fn plan() -> DevicePlan {
        DevicePlan::empty()
            .cpu(0, 2)
            .ram(0, 8192)
            .gpu(0)
            .vram(0, 0, 4096)
    }
}

// ═══════════════ План обучения (общий, с 200 эпохами) ═══════════════
mod training_plan {
    use super::*;
    use neurocore::training_plan::plan::{TrainingPlan, DataSource, Initializer};

    pub fn plan() -> TrainingPlan {
        let num_samples = 500;
        let img_size = 32;
        let num_classes = 10;
        let batch_size = 32;

        let (train_x, train_y) = generate_dataset(num_samples, img_size);

        // Тестовые данные – первые 10 примеров
        let test_x = train_x[..10].to_vec();
        let test_y = train_y[..10].to_vec();

        TrainingPlan::new()
            .model(models::mnist_classifier)
            .loss(losses::cross_entropy_desc(num_classes, batch_size))
            .optimizer(optimizers::sgd(0.1))
            .epochs(200)                               // <-- увеличено до 200
            .batch_size(batch_size)
            .train_data(DataSource::from_tensor2d(Tensor2D::new(train_x)))
            .target_data(DataSource::from_tensor2d(Tensor2D::new(train_y)))
            .test_data(DataSource::from_tensor2d(Tensor2D::new(test_x)))
            .test_target_data(DataSource::from_tensor2d(Tensor2D::new(test_y)))
            .init_weights(Initializer::RandomUniform {
                min: -0.01,
                max: 0.01,
            })
            .seed(42)
    }
}

fn main() {
    // Вариант 1: CPU (16 потоков)
    println!("=== Обучение на CPU (16 потоков) ===");
    let r_cpu = neurocore::run_training!(
        training_plan::plan,
        device = device_plan_cpu::plan
    );
    println!(
        "CPU: Final loss (тест) = {:.6}, время = {:.3}с, лучшая эпоха = {}",
        r_cpu.final_loss, r_cpu.training_time_secs, r_cpu.best_epoch
    );

    // Вариант 2: GPU
    println!("\n=== Обучение на GPU ===");
    let r_gpu = neurocore::run_training!(
        training_plan::plan,
        device = device_plan_gpu::plan
    );
    println!(
        "GPU: Final loss (тест) = {:.6}, время = {:.3}с, лучшая эпоха = {}",
        r_gpu.final_loss, r_gpu.training_time_secs, r_gpu.best_epoch
    );
}
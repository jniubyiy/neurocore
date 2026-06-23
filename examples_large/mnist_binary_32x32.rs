// examples_large/mnist_binary_32x32.rs
// Классификатор MNIST (32x32 бинарных изображений) с двумя скрытыми слоями.
// Модель и потери собираются через макросы, оптимизатор – вручную для гибкости.

use std::time::Instant;
use rand::Rng;

use neurocore::compute_manager::DynamicTensor;
use neurocore::loss_plan::LossDesc;
use neurocore::optimizer_plan::OptimizerDesc;
use neurocore::tensor::Tensor2D;
use neurocore::{create_models, create_losses};

// -----------------------------------------------------------------
// Описание моделей, потерь и оптимизаторов (всё как отдельные функции)
// -----------------------------------------------------------------
mod models {
    use neurocore::model_plan::{Dim, LayerDesc, LayerKind};

    pub fn mnist_classifier() -> Vec<LayerDesc> {
        let img_size = 32;
        let input_dim = img_size * img_size;  // 1024
        let hidden1 = 512;
        let hidden2 = 256;
        let num_classes = 10;

        vec![
            LayerDesc::new("fc1", LayerKind::Linear, Dim::Dim1)
                .input(Dim::Dim1, &[input_dim])
                .output(Dim::Dim1, &[hidden1]),
            LayerDesc::new("relu1", LayerKind::ReLU, Dim::Dim1)
                .input(Dim::Dim1, &[hidden1])
                .output(Dim::Dim1, &[hidden1]),
            LayerDesc::new("fc2", LayerKind::Linear, Dim::Dim1)
                .input(Dim::Dim1, &[hidden1])
                .output(Dim::Dim1, &[hidden2]),
            LayerDesc::new("relu2", LayerKind::ReLU, Dim::Dim1)
                .input(Dim::Dim1, &[hidden2])
                .output(Dim::Dim1, &[hidden2]),
            LayerDesc::new("fc3", LayerKind::Linear, Dim::Dim1)
                .input(Dim::Dim1, &[hidden2])
                .output(Dim::Dim1, &[num_classes]),
            LayerDesc::new("softmax", LayerKind::Softmax, Dim::Dim1)
                .input(Dim::Dim1, &[num_classes])
                .output(Dim::Dim1, &[num_classes]),
        ]
    }
}

mod losses {
    use neurocore::loss_plan::{Aggregation, CrossEntropyWithLogits, ElementChain, LossDesc};

    /// Кросс‑энтропия для классификации с указанным числом классов и размером батча.
    pub fn cross_entropy_desc(num_classes: usize, batch_size: usize) -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(CrossEntropyWithLogits::new(num_classes)));
        LossDesc::from_chain(chain, Aggregation::Mean, batch_size, num_classes, 1)
    }
}

mod optimizers {
    use neurocore::optimizer_plan::{OptimizerDesc, OptCubeDesc};

    /// Обычный SGD с заданным learning rate.
    pub fn sgd(lr: f32) -> OptimizerDesc {
        OptimizerDesc::new()
            .add(OptCubeDesc::ScaleGradient(lr))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

// -----------------------------------------------------------------
fn main() {
    // Параметры обучения
    let img_size = 32;
    let num_classes = 10;
    let batch_size = 32;
    let epochs = 20;
    let lr = 0.1;

    // Сборка модели
    let (mut model,) = create_models!(models::mnist_classifier);

    // Сгенерируем синтетический датасет (500 примеров 32x32)
    let (train_x, train_y) = generate_dataset(500, img_size);
    println!("Сгенерировано {} обучающих примеров", train_x.len());

    // Инициализация параметров малыми случайными числами
    {
        let mut store = model.param_store().lock().unwrap();
        let len = store.len();
        for i in 0..len {
            store.set_param(i, rand::random::<f32>() * 0.01);
        }
    }

    // Создаём SGD‑оптимизатор вручную (макрос create_optimizers! не поддерживает параметры)
    let mut opt = model.create_optimizer(optimizers::sgd(lr).build_chain());

    let start = Instant::now();
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        for batch_start in (0..train_x.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_x.len());
            let current_batch_size = batch_end - batch_start;

            // Подготовка батча
            let batch_x = Tensor2D::new(train_x[batch_start..batch_end].to_vec());
            let batch_y = Tensor2D::new(train_y[batch_start..batch_end].to_vec());

            // Прямой проход
            let (pred, contexts) = model.forward(DynamicTensor::Dim1(batch_x.clone()));

            // Функция потерь для текущего размера батча
            let loss_expr = losses::cross_entropy_desc(num_classes, current_batch_size).build();

            // Вычисление потерь и начального градиента
            let (loss, delta) = model.compute_loss_with_expr(
                loss_expr,
                &pred,
                &DynamicTensor::Dim1(batch_y),
            );
            total_loss += loss * current_batch_size as f32;

            // Обратный проход
            let (_, grads) = model.backward(&contexts, delta);
            // Обновление параметров
            model.update_params_with_optimizer(&mut opt, &grads[0]);
        }
        println!(
            "Epoch {}: avg loss = {:.6}",
            epoch,
            total_loss / train_x.len() as f32
        );
    }
    let duration = start.elapsed();
    println!("Обучение завершено за {:?}", duration);

    // Проверка на первых 10 примерах
    let test_x = Tensor2D::new(train_x[..10].to_vec());
    let (pred, _) = model.forward(DynamicTensor::Dim1(test_x));
    let pred_matrix = match pred {
        DynamicTensor::Dim1(t) => t,
        _ => panic!("Ожидался Dim1 (Tensor2D)"),
    };
    for i in 0..10 {
        let pred_class = pred_matrix.data[i]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let true_class = train_y[i][0] as usize;
        println!(
            "Пример {}: предсказано {}, истина {}",
            i, pred_class, true_class
        );
    }
}

/// Генерация синтетического датасета: бинарные изображения 32x32 с шаблонами цифр 0-9.
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
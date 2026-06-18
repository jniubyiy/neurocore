// examples_large/mnist_binary_32x32.rs

use neurocore::model_plan::{Plan, LayerBlueprint, Dim};
use neurocore::dispatchers::common::Model1D;
use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::tensor::{Tensor1D, Tensor2D};
use std::time::Instant;
use rand::Rng; // Добавим rand в Cargo.toml для генерации шума (можно обойтись встроенным)

fn main() {
    // Гиперпараметры
    let img_size = 32;
    let input_dim = img_size * img_size; // 1024
    let hidden1 = 512;
    let hidden2 = 256;
    let num_classes = 10;
    let batch_size = 32;
    let epochs = 20;
    let lr = 0.01;

    // Построение модели
    let plan = Plan::new(vec![
        LayerBlueprint::linear(Dim::Dim1, input_dim, hidden1),
        LayerBlueprint::relu(Dim::Dim1, hidden1),
        LayerBlueprint::linear(Dim::Dim1, hidden1, hidden2),
        LayerBlueprint::relu(Dim::Dim1, hidden2),
        LayerBlueprint::linear(Dim::Dim1, hidden2, num_classes),
        LayerBlueprint::softmax(Dim::Dim1, num_classes),
    ]).expect("Ошибка архитектуры");

    let loss_plan = LossPlan::new(LossBlueprint::cross_entropy(num_classes)).unwrap();
    let built_loss = loss_plan.build().unwrap();

    // Генерация синтетического датасета
    let (train_x, train_y) = generate_dataset(500, img_size);
    println!("Сгенерировано {} обучающих примеров", train_x.len());

    // Инициализация модели
    let mut built = plan.build_1d();
    let total_params = built.store.len();
    // Заполним параметры случайными малыми значениями
    for i in 0..total_params {
        built.store.set_param(i, rand::random::<f32>() * 0.01);
    }
    let mut model = built.into_dispatcher(4); // 4 потока

    // Обучение
    let start = Instant::now();
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        for batch_start in (0..train_x.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_x.len());
            let batch_x = Tensor2D::new(train_x[batch_start..batch_end].to_vec());
            let batch_y = Tensor2D::new(train_y[batch_start..batch_end].to_vec());

            // Forward
            let (pred, contexts) = model.forward(&batch_x);
            // Преобразуем каждый образец в Tensor1D для лосса
            let mut batch_loss = 0.0;
            let mut batch_delta = Vec::with_capacity(pred.dim1);
            for i in 0..pred.dim1 {
                let pred1d = Tensor1D::new(pred.data[i].clone());
                let target1d = Tensor1D::new(batch_y.data[i].clone());
                let (loss, delta1d) = (built_loss.forward)(&pred1d, &target1d);
                batch_loss += loss;
                batch_delta.push(delta1d.data);
            }
            total_loss += batch_loss / (batch_end - batch_start) as f32;

            let delta = Tensor2D::new(batch_delta);
            let (_, grads) = model.backward(&contexts, &delta);
            model.update_params(lr, &grads);
        }
        println!("Epoch {}: avg loss = {:.6}", epoch, total_loss / (train_x.len() as f32 / batch_size as f32));
    }
    let duration = start.elapsed();
    println!("Обучение завершено за {:?}", duration);

    // Проверка на нескольких примерах
    let test_x = Tensor2D::new(train_x[..10].to_vec());
    let (pred, _) = model.forward(&test_x);
    for i in 0..10 {
        let pred_class = pred.data[i].iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let true_class = train_y[i][0] as usize;
        println!("Пример {}: предсказано {}, истина {}", i, pred_class, true_class);
    }
}

/// Генерирует синтетические бинарные изображения цифр.
/// Возвращает векторы (изображение, метка) в формате Tensor2D (образцы × 1024) и метки (образцы × 1).
fn generate_dataset(num_samples: usize, img_size: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = rand::thread_rng();
    let mut images = Vec::with_capacity(num_samples);
    let mut labels = Vec::with_capacity(num_samples);

    // Простые шаблоны цифр 0-9 (упрощённые, можно заменить на более реалистичные)
    let templates: Vec<Vec<Vec<f32>>> = (0..10).map(|digit| {
        let mut img = vec![vec![0.0f32; img_size]; img_size];
        match digit {
            0 => {
                // круг
                let cx = img_size as f32 / 2.0;
                let cy = img_size as f32 / 2.0;
                let r = img_size as f32 / 2.0 - 2.0;
                for i in 0..img_size {
                    for j in 0..img_size {
                        let dx = i as f32 - cx;
                        let dy = j as f32 - cy;
                        let dist = (dx*dx + dy*dy).sqrt();
                        if (dist - r).abs() < 2.0 { img[i][j] = 1.0; }
                    }
                }
            },
            1 => {
                // вертикальная линия
                for i in 0..img_size { img[i][img_size/2] = 1.0; }
            },
            2 => {
                // горизонтальные линии сверху и снизу, диагональ
                for j in 0..img_size { img[0][j] = 1.0; img[img_size-1][j] = 1.0; }
                for i in 0..img_size { img[i][i] = 1.0; }
            },
            // ... аналогично для остальных цифр (просто разные узоры)
            3 => {
                for i in 0..img_size { img[i][0] = 1.0; img[i][img_size-1] = 1.0; }
                for j in 0..img_size { img[img_size/2][j] = 1.0; }
            },
            4 => {
                for i in 0..img_size { img[i][img_size/3] = 1.0; }
                for j in 0..img_size { img[2*img_size/3][j] = 1.0; }
            },
            5 => {
                for i in 0..img_size { img[i][0] = 1.0; }
                for j in 0..img_size { img[img_size/2][j] = 1.0; img[img_size-1][j] = 1.0; }
            },
            6 => {
                for i in 0..img_size { img[i][0] = 1.0; img[i][img_size-1] = 1.0; img[0][i] = 1.0; img[img_size/2][i] = 1.0; }
            },
            7 => {
                for j in 0..img_size { img[0][j] = 1.0; }
                for i in 0..img_size { img[i][img_size-1 - i] = 1.0; }
            },
            8 => {
                for i in 0..img_size { img[i][0] = 1.0; img[i][img_size-1] = 1.0; img[0][i] = 1.0; img[img_size-1][i] = 1.0; }
            },
            9 => {
                for i in 0..img_size { img[i][img_size-1] = 1.0; }
                for j in 0..img_size { img[0][j] = 1.0; img[img_size/2][j] = 1.0; }
                for i in 0..img_size { img[i][i] = 1.0; }
            },
            _ => {}
        }
        img
    }).collect();

    for _ in 0..num_samples {
        let digit = rng.gen_range(0..10);
        let mut img = templates[digit].clone();
        // Добавляем шум (инвертируем некоторые пиксели)
        for i in 0..img_size {
            for j in 0..img_size {
                if rng.gen::<f32>() < 0.1 { // 10% шанс инвертировать
                    img[i][j] = 1.0 - img[i][j];
                }
            }
        }
        // Преобразуем в одномерный вектор
        let flat: Vec<f32> = img.into_iter().flatten().collect();
        images.push(flat);
        labels.push(vec![digit as f32]);
    }
    (images, labels)
}
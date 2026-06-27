```markdown
# neurocore – руководство пользователя

**neurocore** — библиотека глубокого обучения для Rust, которая позволяет с лёгкостью строить, обучать и применять нейронные сети. Благодаря матричным вычислениям на базе `faer` и встроенной многопоточности, она сочетает производительность с простотой использования.

## Установка

Добавьте в `Cargo.toml` вашего проекта:

```toml
[dependencies]
neurocore = { path = "../neurocore" }   # путь к папке библиотеки
```

Все публичные API становятся доступны через `use neurocore::...`.

## Быстрый старт: обучение автоэнкодера

```rust
use neurocore::create_models;
use neurocore::compute_manager::DynamicTensor;
use neurocore::tensor::Tensor2D;

// 1. Описываем модель: вход 4 → скрытый 2 → выход 4
mod models {
    use neurocore::model_plan::{Dim, LayerDesc, LayerKind};
    pub fn autoencoder() -> Vec<LayerDesc> {
        vec![
            LayerDesc::linear(Dim::Dim1, 4, 2),
            LayerDesc::sigmoid(Dim::Dim1, 2),
            LayerDesc::linear(Dim::Dim1, 2, 4),
        ]
    }
}

// 2. Описываем функцию потерь (MSE)
mod losses {
    use neurocore::loss_plan::*;
    pub fn mse() -> LossDesc {
        LossDesc::from_chain(
            ElementChain::new()
                .add(Box::new(Sub))
                .add(Box::new(Square)),
            Aggregation::Mean,
            4, 1, 1
        )
    }
}

// 3. Описываем оптимизатор (SGD с lr=0.01)
mod optimizers {
    use neurocore::optimizer_plan::*;
    pub fn sgd() -> OptimizerDesc {
        OptimizerDesc::new()
            .add(OptCubeDesc::ScaleGradient(0.01))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

fn main() {
    let (mut model,) = create_models!(models::autoencoder);
    let x = Tensor2D::new(vec![vec![1.0, 2.0, 3.0, 4.0]]);
    let target = x.clone();

    for epoch in 0..500 {
        let (pred, ctxs) = model.forward(DynamicTensor::Dim1(x.clone()));
        let (loss, delta) = model.compute_loss(losses::mse(), &pred,
                                               &DynamicTensor::Dim1(target.clone()));
        let (_, grads) = model.backward(&ctxs, delta);
        model.update_params(optimizers::sgd(), &grads[0]);
        if epoch % 100 == 0 { println!("{}: loss={:.6}", epoch, loss); }
    }
}
```

## Основные концепции

### Тензоры

Библиотека предоставляет тензоры для разных размерностей. Вы всегда работаете с ними как с входными/выходными данными.

| Тип | Размерность | Что хранит |
|-----|------------|------------|
| `Tensor2D` | `(batch, features)` | набор векторов |
| `Tensor3D` | `(batch, dim2, dim3)` | например, изображения (каналы×строки) |
| `Tensor4D` | `(batch, dim2, dim3, dim4)` | 3D-данные |
| `Tensor5D` | `(batch, dim2, dim3, dim4, dim5)` | 4D-данные |

Каждый тензор создаётся конструктором `new(data)`, где `data` — вложенный `Vec` соответствующей глубины, или `zeros(...)` для нулевого заполнения.

### Описание модели (`LayerDesc` и `Plan`)

Модель описывается цепочкой слоёв, каждый из которых задаётся через `LayerDesc`. Основные поля:
- `kind` – тип слоя (`LayerKind`)
- `in_features` / `out_features` – размеры входа/выхода
- `extra` – дополнительные параметры (например, коэффициент наклона у `LeakyReLU`)

Используйте готовые конструкторы:
- `LayerDesc::linear(Dim::Dim1, in_features, out_features)`
- `LayerDesc::relu(Dim::Dim1, size)`
- `LayerDesc::sigmoid(Dim::Dim1, size)`
- `LayerDesc::softmax(Dim::Dim1, size)`
- `LayerDesc::tanh(Dim::Dim1, size)`
- `LayerDesc::leaky_relu(Dim::Dim1, size, alpha)`
- `LayerDesc::identity(Dim::Dim1, size)`
- `LayerDesc::soft_sparse_gate(Dim::Dim1, size, temperature)` — обучаемое прореживание
- `LayerDesc::soft_keep_gate(Dim::Dim1, size, temperature)` — обучаемое удержание
- `LayerDesc::dual_anchor(Dim::Dim1, size)` — двуханкерное преобразование

Для изменения формы тензора есть `LayerDesc::unsqueeze(input_dim, target_dims)` и `LayerDesc::reduce_mean(input_dim, target_dims)`. Пример: `unsqueeze(Dim::Dim1, vec![2,2])` превращает вектор длины 4 в матрицу 2×2.

Список слоёв объединяется в `Plan`:
```rust
let plan = Plan::from_layer_descs(vec![...])?;
```

Макрос `create_models!` делает это автоматически, принимая имена функций, возвращающих `Vec<LayerDesc>`, и создавая готовые `MixedModel`:
```rust
let (encoder, decoder) = create_models!(my_encoder, my_decoder);
```

### Модель (`MixedModel`)

Главный объект, который вы будете использовать — `MixedModel`. Основные методы:

- **`forward(input: DynamicTensor) -> (DynamicTensor, Vec<Vec<DynamicContext>>)`**  
  Прямой проход. Возвращает предсказание и контексты, необходимые для обратного прохода.

- **`backward(&contexts, delta: DynamicTensor) -> (DynamicTensor, Vec<Vec<f32>>)`**  
  Обратный проход. Принимает контексты из `forward` и градиент потерь по выходу. Возвращает градиент по входу и градиенты параметров (вектор).

- **`compute_loss(desc: LossDesc, pred, target) -> (f32, DynamicTensor)`**  
  Удобный комбинированный метод: вычисляет потери по описанию, возвращает значение и градиент по `pred`.

- **`update_params(desc: OptimizerDesc, grads: &[f32])`**  
  Применяет градиенты к параметрам, используя описанный оптимизатор.

- **`param_store()`**  
  Даёт доступ к хранилищу параметров (нужно для инициализации весов случайными числами перед обучением). Пример:
  ```rust
  let mut store = model.param_store().lock().unwrap();
  for i in 0..store.len() {
      store.set_param(i, rand::random::<f32>() * 0.01);
  }
  ```

### Динамические тензоры (`DynamicTensor`)

Так как модель может менять размерность внутри, вход/выход `forward` и `compute_loss` оборачиваются в `DynamicTensor`. Создать его просто:
```rust
let dt = DynamicTensor::Dim1(tensor2d);
let dt = DynamicTensor::Dim2(tensor3d);
// ...
```
Извлечь обратно можно через сопоставление с образцом.

### Функции потерь

Описываются цепочкой элементарных операций – «кубиков».  
Создайте описание с помощью `LossDesc::from_chain`:

```rust
LossDesc::from_chain(chain, aggregation, total_tasks, pred_features, target_features)
```

- `chain` – `ElementChain`, полученный вызовами `.add(...)`  
- `aggregation` – `Aggregation::Mean` или `Aggregation::Sum`  
- `total_tasks` – количество задач в одном батче (произведение батча на количество элементов)  
- `pred_features` – сколько признаков в предсказании  
- `target_features` – сколько признаков в цели  

Доступные кубики (все из модуля `neurocore::loss_plan`):
- `Sub` – разность `pred - target`
- `Square` – квадрат
- `Abs` – модуль
- `Log` – логарифм
- `Neg` – смена знака
- `Mul` – произведение двух входов
- `AddScalar(c)` – прибавление константы
- `Log1p` – `ln(1 + x)`
- `AbsDiff` – модуль разности двух чисел (полезно для сглаживающих потерь)
- `CrossEntropyWithLogits::new(num_classes)` – кросс-энтропия (на вход ожидает вектор логитов + индекс класса в последнем столбце).

Пример MSE:
```rust
ElementChain::new().add(Box::new(Sub)).add(Box::new(Square))
```

Макрос `create_losses!` может собрать несколько функций потерь в кортеж.

### Оптимизаторы

Задаются цепочкой действий. Самые востребованные кубики:
- `ScaleGradient(f32)` – умножает градиент (learning rate)
- `Momentum(f32)` – момент
- `Adam { beta1, beta2, eps }` – преобразование Adam (требует ApplyUpdate)
- `ApplyUpdate` – финальное вычитание градиента из параметров (должен идти последним).

Используйте `OptimizerDesc` для построения цепочки:
```rust
OptimizerDesc::new()
    .add(OptCubeDesc::ScaleGradient(0.01))
    .add(OptCubeDesc::ApplyUpdate)
```
Метод `update_params()` модели принимает именно `OptimizerDesc`.

## Дополнительные возможности

### Изменение размерности

Слои `Unsqueeze` и `ReduceMean` позволяют менять форму тензора, например, переходить от вектора к матрице и обратно. Задаются целевыми размерами (без батча).

```rust
// Превращаем вектор длины 4 в 2×2
LayerDesc::unsqueeze(Dim::Dim1, vec![2, 2])
// Обратно: ReduceMean с теми же размерами восстановит вектор
LayerDesc::reduce_mean(Dim::Dim2, vec![2, 2])
```

### Выбор числа потоков

По умолчанию модель создаётся для одного потока. Для многопоточности используйте `Plan::build_with_threads(n)`, где `n` – желаемое количество рабочих потоков.

### Логирование

Библиотека имеет встроенный логгер. Для включения вызова информационных сообщений (например, при создании модели) установите уровень:
```rust
Logger::set_level(1); // info
```
Уровни: 0 – молчание, 1 – info, 2 – debug, 3 – trace.

## Полный список слоёв и их параметров

| Слой | Обучаемые параметры | extra |
|------|---------------------|-------|
| `Linear` | веса + смещения | нет |
| `ReLU` | нет | нет |
| `Sigmoid` | нет | нет |
| `Softmax` | нет | нет |
| `Tanh` | нет | нет |
| `LeakyReLU` | нет | `alpha` |
| `Identity` | нет | нет |
| `SoftSparseGate` | пороги (1 на признак) | `temperature` |
| `SoftKeepGate` | пороги (1 на признак) | `temperature` |
| `DualAnchor` | min + max + alpha | нет |
| `Memory` | нет | нет |
| `Unsqueeze` | нет | target_dims |
| `ReduceMean` | нет | target_dims |

## Рекомендации

1. Для инициализации параметров всегда заполняйте их малыми случайными значениями после создания модели.
2. При использовании `CrossEntropyWithLogits` не добавляйте `Softmax` перед ним – кубик уже включает softmax внутри.
3. `Softmax` в модели должен идти последним слоем (ограничение валидации плана).
4. Для многомерных данных выбирайте соответствующую размерность `Dim` и используйте тензоры `Tensor3D`, `Tensor4D`, `Tensor5D`.

## Где искать примеры

В папке `examples/` находятся небольшие, готовые к запуску сценарии:
- `autoencoder.rs` – автоэнкодер (Dim1)
- `autoencoder2d.rs`, `autoencoder3d.rs`, `autoencoder4d.rs` – то же для старших размерностей
- `classifier.rs`, `classifier2d.rs`, `classifier3d.rs`, `classifier4d.rs` – классификаторы
- `linear_test.rs`, `linear2d_test.rs`, … – обучение одного линейного слоя
- `loss_test.rs`, `loss2d_test.rs`, … – тесты функций потерь
- `graph_full.rs` – граф с разветвлением

Папка `examples_large/` содержит более масштабный пример `mnist_binary_32x32.rs` (классификатор на синтетических изображениях).

---

Теперь вы готовы использовать **neurocore** для создания и тренировки собственных нейросетей на Rust!
```
// src/plans/model_plan/shape.rs

/// Описание формы тензора с учётом многопоточности.
/// Хранит список потоков (streams), каждый поток имеет набор осей (axes).
/// Для одного потока axes содержит размеры всех осей; общее число признаков = произведение axes.
/// Для нескольких потоков каждый поток имеет свои оси и свои веса.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    /// Размер батча (пока информативно, не используется в вычислениях).
    pub batch: usize,
    /// Общее количество признаков для каждого потока.
    pub streams: Vec<usize>,
    /// Оси для каждого потока (длина совпадает с `streams`).
    pub axes: Vec<Vec<usize>>,
}

impl Shape {
    /// Создаёт форму с одним потоком и указанными осями.
    ///
    /// # Пример
    /// ```ignore
    /// let shape = Shape::single_with_axes(vec![2, 4]); // один поток, 2*4=8 признаков
    /// ```
    pub fn single_with_axes(axes: Vec<usize>) -> Self {
        let total: usize = axes.iter().product();
        Shape {
            batch: 0,
            streams: vec![total],
            axes: vec![axes],
        }
    }

    /// Создаёт форму с одним потоком заданного размера признаков (обратная совместимость).
    pub fn single(features: usize) -> Self {
        Shape {
            batch: 0,
            streams: vec![features],
            axes: vec![vec![features]],
        }
    }

    /// Создаёт форму с несколькими потоками, каждый поток задан набором осей.
    ///
    /// # Пример
    /// ```ignore
    /// let shape = Shape::multi_with_axes(vec![vec![2, 4], vec![3]]);
    /// // два потока: первый 8 признаков, второй 3 признака
    /// ```
    pub fn multi_with_axes(streams_axes: Vec<Vec<usize>>) -> Self {
        let streams: Vec<usize> = streams_axes
            .iter()
            .map(|axes| axes.iter().product())
            .collect();
        Shape {
            batch: 0,
            streams,
            axes: streams_axes,
        }
    }

    /// Создаёт форму с несколькими потоками, каждый поток имеет одну ось (обратная совместимость).
    pub fn multi(streams: Vec<usize>) -> Self {
        Shape {
            batch: 0,
            streams: streams.clone(),
            axes: streams.iter().map(|&s| vec![s]).collect(),
        }
    }

    /// Общее количество признаков во всех потоках (информационно, не для вычислений).
    pub fn total_features(&self) -> usize {
        self.streams.iter().sum()
    }

    /// Количество потоков.
    pub fn num_streams(&self) -> usize {
        self.streams.len()
    }
}

/// Макрос для создания `Shape`.
///
/// Поддерживает два основных варианта:
/// 1. **Один поток с несколькими осями**:
///    `shape!(batch, A[2], B[4])` → один поток, оси `[2, 4]`, всего `8` признаков.
/// 2. **Несколько потоков** (разделены `;`):
///    `shape!(batch, A[3]; batch, A[3])` → два потока, каждый с осью `[3]`.
///    `shape!(batch, A[2], B[2]; batch, A[3])` → первый поток с осями `[2,2]`, второй с осью `[3]`.
#[macro_export]
macro_rules! shape {
    // Один поток: batch, A[x], B[y], ...
    (batch $( , $label:ident [ $size:expr ] )+ ) => {
        $crate::plans::model_plan::Shape::single_with_axes(vec![ $( $size ),+ ])
    };

    // Несколько потоков: первый поток, затем `; batch, ...` ещё один или более потоков
    (batch $( , $label:ident [ $size:expr ] )+ ; $( batch $( , $label2:ident [ $size2:expr ] )+ );+ ) => {
        {
            let mut streams_axes = Vec::new();
            streams_axes.push(vec![ $( $size ),+ ]);
            $(
                streams_axes.push(vec![ $( $size2 ),+ ]);
            )+
            $crate::plans::model_plan::Shape::multi_with_axes(streams_axes)
        }
    };
}
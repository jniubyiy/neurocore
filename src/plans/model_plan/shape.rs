// src/plans/model_plan/shape.rs

/// Описание формы тензора с учётом многопоточности.
/// Хранит список размеров признаков для каждого потока (stream).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    /// Размер батча (пока информативно, не используется в вычислениях).
    pub batch: usize,
    /// Количество признаков для каждого потока.
    pub streams: Vec<usize>,
}

impl Shape {
    /// Создаёт форму с одним потоком заданного размера признаков.
    pub fn single(features: usize) -> Self {
        Shape {
            batch: 0,
            streams: vec![features],
        }
    }

    /// Создаёт форму с несколькими потоками (размеры признаков каждого).
    pub fn multi(streams: Vec<usize>) -> Self {
        Shape {
            batch: 0,
            streams,
        }
    }

    /// Общее количество признаков во всех потоках.
    /// Для многомерных тензоров это произведение всех осей, кроме batch.
    pub fn total_features(&self) -> usize {
        self.streams.iter().product()
    }

    /// Количество потоков.
    pub fn num_streams(&self) -> usize {
        self.streams.len()
    }
}

/// Макрос для создания Shape.
/// Принимает ключевое слово `batch` и перечисление потоков вида `A[число]`,
/// например: `shape!(batch, A[4])` или `shape!(batch, A[2], B[2])`.
#[macro_export]
macro_rules! shape {
    (batch $( , $label:ident [ $size:expr ] )+ ) => {
        $crate::plans::model_plan::Shape {
            batch: 0,
            streams: vec![ $( $size ),+ ],
        }
    };
}
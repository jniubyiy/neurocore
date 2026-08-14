// src/layers/buffered_context.rs

use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

/// Контекст, сохраняемый слоями при буферизованном прямом проходе.
///
/// В отличие от `MatContext`, здесь хранятся не `faer::Mat`, а лёгкие
/// дескрипторы `MatrixBufferHandle`, которые ссылаются на данные в
/// `MemoryExecutor`. Дескрипторы можно свободно клонировать, что позволяет
/// разделять один буфер между контекстом и следующим слоем без копирования.
///
/// Используется только в варианте `DynamicContext::Buffered`.
#[derive(Clone)]
pub enum BufferedContext {
    /// Вход линейного слоя.
    Linear {
        input: MatrixBufferHandle,
    },

    /// Вход ReLU.
    ReLU {
        input: MatrixBufferHandle,
    },

    /// Выход Sigmoid (для обратного прохода нужен выход).
    Sigmoid {
        output: MatrixBufferHandle,
    },

    /// Выход Tanh (для обратного прохода нужен выход).
    Tanh {
        output: MatrixBufferHandle,
    },

    /// Выход Softmax (для обратного прохода нужен выход).
    Softmax {
        output: MatrixBufferHandle,
    },

    /// Вход Memory (текущий обратный проход может его не использовать, но сохранён для полноты).
    Memory {
        input: MatrixBufferHandle,
    },

    /// Вход LeakyReLU.
    LeakyReLU {
        input: MatrixBufferHandle,
    },

    /// Вход SoftSparseGate.
    SoftSparseGate {
        input: MatrixBufferHandle,
    },

    /// Вход SoftKeepGate.
    SoftKeepGate {
        input: MatrixBufferHandle,
    },

    /// Вход DualAnchor.
    DualAnchor1D {
        input: MatrixBufferHandle,
    },

    /// Вход Identity.
    Identity {
        input: MatrixBufferHandle,
    },
}
// src/layers/buffered_context.rs

use std::sync::Arc;

use crate::compute_manager::matrix_buffer::MatrixBuffer;

/// Контекст, сохраняемый слоями при буферизованном прямом проходе.
///
/// В отличие от `MatContext`, здесь хранятся не `faer::Mat`, а управляемые
/// буферы `MatrixBuffer`, обёрнутые в `Arc` для безопасного разделения между
/// прямым и обратным проходами.
///
/// Используется только в варианте `DynamicContext::Buffered`.
#[derive(Clone)]
pub enum BufferedContext {
    /// Вход линейного слоя.
    Linear {
        input: Arc<MatrixBuffer>,
    },

    /// Вход ReLU.
    ReLU {
        input: Arc<MatrixBuffer>,
    },

    /// Выход Sigmoid (для обратного прохода нужен выход).
    Sigmoid {
        output: Arc<MatrixBuffer>,
    },

    /// Выход Tanh (для обратного прохода нужен выход).
    Tanh {
        output: Arc<MatrixBuffer>,
    },

    /// Выход Softmax (для обратного прохода нужен выход).
    Softmax {
        output: Arc<MatrixBuffer>,
    },

    /// Вход Memory (текущий обратный проход может его не использовать, но сохранён для полноты).
    Memory {
        input: Arc<MatrixBuffer>,
    },

    /// Вход LeakyReLU.
    LeakyReLU {
        input: Arc<MatrixBuffer>,
    },

    /// Вход SoftSparseGate.
    SoftSparseGate {
        input: Arc<MatrixBuffer>,
    },

    /// Вход SoftKeepGate.
    SoftKeepGate {
        input: Arc<MatrixBuffer>,
    },

    /// Вход DualAnchor.
    DualAnchor1D {
        input: Arc<MatrixBuffer>,
    },

    /// Вход Identity.
    Identity {
        input: Arc<MatrixBuffer>,
    },
}
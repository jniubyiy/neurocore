// src/plans/loss_plan/mod.rs

pub mod cubes;          // оставлен для обратной совместимости (но не используется)
pub mod cross_entropy;  // оставлен для обратной совместимости (но не используется)
pub mod chain;
pub mod expr;
pub mod desc;
pub mod execution;
pub mod gpu_exec;

// Реэкспорт новых кубиков из src/losses
pub use crate::losses::{
    ElemCube, BufferedElemCube,
    Sub, Square, SumColumns, Log, Neg, Mul, Abs, AddScalar, Log1p, AbsDiff,
    CrossEntropyWithLogits,
};

pub use chain::ElementChain;
pub use expr::{Aggregation, LossExpr};
pub use desc::LossDesc;
pub use execution::compute_loss_mat_buffered;
pub use gpu_exec::compute_loss_gpu_buffered_handle;
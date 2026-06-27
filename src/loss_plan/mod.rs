// src/loss_plan/mod.rs

pub mod cubes;
pub mod cross_entropy;
pub mod chain;
pub mod expr;
pub mod desc;
pub mod execution;

pub use cubes::{ElemCube, Sub, Square, Log, Neg, Mul, Abs, AddScalar, Log1p, AbsDiff};
pub use cross_entropy::CrossEntropyWithLogits;
pub use chain::ElementChain;
pub use expr::{Aggregation, LossExpr};
pub use desc::LossDesc;
pub use execution::compute_loss_mat;
// src/model_plan/mod.rs

pub mod blueprint;
pub mod plan;
pub mod param_store;
pub mod apply_gradient;
pub mod dim;
pub mod sequential;
pub mod builder;
pub mod layer_desc;

pub use blueprint::{LayerBlueprint, LayerKind};
pub use plan::Plan;
pub use param_store::{ParamSlice, ParamStore};
pub use apply_gradient::apply_gradient;
pub use dim::Dim;
pub use sequential::Sequential;
pub use builder::{LayerBuilder, LinearLayerBuilder, ReLULayerBuilder, SigmoidLayerBuilder, SoftmaxLayerBuilder};
pub use layer_desc::{LayerDesc, IntoSizes};
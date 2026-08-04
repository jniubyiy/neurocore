// src/model_plan/dim.rs

/// Размерность модели / слоя (количество осей после батча).
/// Dim1 – одна ось (Tensor2D), Dim2 – две оси (Tensor3D), Dim3 – три (Tensor4D), Dim4 – четыре (Tensor5D).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dim {
    Dim1,
    Dim2,
    Dim3,
    Dim4,
}

impl Dim {
    pub fn next(self) -> Option<Dim> {
        match self {
            Dim::Dim1 => Some(Dim::Dim2),
            Dim::Dim2 => Some(Dim::Dim3),
            Dim::Dim3 => Some(Dim::Dim4),
            Dim::Dim4 => None,
        }
    }

    pub fn prev(self) -> Option<Dim> {
        match self {
            Dim::Dim4 => Some(Dim::Dim3),
            Dim::Dim3 => Some(Dim::Dim2),
            Dim::Dim2 => Some(Dim::Dim1),
            Dim::Dim1 => None,
        }
    }
}
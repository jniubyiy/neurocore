// src/model_plan/dim.rs

/// Размерность модели / слоя.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dim {
    Dim1,
    Dim2,
    Dim3,
    Dim4,
    Dim5,
}

impl Dim {
    /// Повышает размерность на 1 (Dim1 -> Dim2, …, Dim4 -> Dim5).
    /// Для Dim5 возвращает None.
    pub fn next(self) -> Option<Dim> {
        match self {
            Dim::Dim1 => Some(Dim::Dim2),
            Dim::Dim2 => Some(Dim::Dim3),
            Dim::Dim3 => Some(Dim::Dim4),
            Dim::Dim4 => Some(Dim::Dim5),
            Dim::Dim5 => None,
        }
    }

    /// Понижает размерность на 1 (Dim2 -> Dim1, …, Dim5 -> Dim4).
    /// Для Dim1 возвращает None.
    pub fn prev(self) -> Option<Dim> {
        match self {
            Dim::Dim5 => Some(Dim::Dim4),
            Dim::Dim4 => Some(Dim::Dim3),
            Dim::Dim3 => Some(Dim::Dim2),
            Dim::Dim2 => Some(Dim::Dim1),
            Dim::Dim1 => None,
        }
    }
}
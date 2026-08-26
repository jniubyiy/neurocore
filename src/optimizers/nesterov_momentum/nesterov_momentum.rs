/// Момент Нестерова (NAG).
/// Состояние: одно число на параметр — скорость `v`.
pub struct NesterovMomentum {
    pub beta: f32,
}

impl NesterovMomentum {
    pub fn new(beta: f32) -> Self {
        Self { beta }
    }
}
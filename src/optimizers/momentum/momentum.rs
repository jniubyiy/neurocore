/// Классический момент (momentum).
/// Состояние: одно число на параметр — скорость `v`.
pub struct Momentum {
    pub beta: f32,
}

impl Momentum {
    pub fn new(beta: f32) -> Self {
        Self { beta }
    }
}
/// Умножает градиент на заданный коэффициент (обычно learning rate).
pub struct ScaleGradient {
    pub factor: f32,
}

impl ScaleGradient {
    pub fn new(factor: f32) -> Self {
        Self { factor }
    }
} 
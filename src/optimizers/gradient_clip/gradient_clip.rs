/// Обрезает градиент по значениям `[min, max]`.
pub struct GradientClip {
    pub min: Option<f32>,
    pub max: Option<f32>,
}

impl GradientClip {
    pub fn new(min: Option<f32>, max: Option<f32>) -> Self {
        Self { min, max }
    }
}
/// Добавляет L2‑регуляризацию к градиенту:
/// `grads[i] += decay * params[i]`
pub struct AddWeightDecay {
    pub decay: f32,
}

impl AddWeightDecay {
    pub fn new(decay: f32) -> Self {
        Self { decay }
    }
}
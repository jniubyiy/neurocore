/// Стабильный softmax (с вычитанием максимума).
pub fn stable_softmax(values: &[f32]) -> Vec<f32> {
    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = values.iter().map(|v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    for e in &mut exps {
        *e /= sum;
    }
    exps
}






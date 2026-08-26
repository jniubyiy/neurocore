/// Применяет накопленный градиент к параметрам: `params[i] -= grads[i]`
/// Этот кубик всегда должен быть последним в цепочке.
pub struct ApplyUpdate;

impl ApplyUpdate {
    pub fn new() -> Self {
        Self
    }
}
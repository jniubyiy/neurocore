use super::param_store::ParamStore;

/// Простейший оптимизатор: градиентный спуск с постоянным lr.
pub fn apply_gradient(param_store: &mut ParamStore, lr: f32, grad: &[f32]) {
    param_store.apply_gradient(lr, grad);
}

// В будущем здесь появятся:
// pub fn adam_step(...)
// pub fn sgd_momentum(...)
// и т.д.
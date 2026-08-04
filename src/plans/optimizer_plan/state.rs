// src/optimizer/state.rs

/// Хранилище состояния оптимизатора.
///
/// Содержит единый плоский буфер для всех кубиков цепочки.
/// Размер буфера равен `num_params * total_state_size_per_param()`.
pub struct OptimizerState {
    data: Vec<f32>,
}

impl OptimizerState {
    /// Создаёт новое состояние, инициализированное нулями.
    ///
    /// # Аргументы
    /// * `num_params` – количество оптимизируемых параметров.
    /// * `state_size_per_param` – суммарный размер состояния на параметр,
    ///   полученный из `OptimizerChain::total_state_size_per_param()`.
    pub fn new(num_params: usize, state_size_per_param: usize) -> Self {
        Self {
            data: vec![0.0; num_params * state_size_per_param],
        }
    }

    /// Возвращает мутабельный срез всех данных состояния.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }
}
// src/compute_manager/gpu/processor/mod.rs

//! GPU-исполнитель прямого и обратного проходов для UniversalProcessor-сегментов.
//!
//! Модуль разбит по категориям слоёв:
//! - `forward`  — оркестратор прямого прохода;
//! - `backward` — оркестратор обратного прохода;
//! - `layers`   — пер-категорийные обработчики (по одному файлу на группу слоёв).
//!
//! Оркестраторы (`forward.rs`, `backward.rs`) последовательно опрашивают
//! обработчики из `layers::*` через `if let Some(r) = handler::forward(...)`.
//! Каждый обработчик сам решает, относится ли текущий слой к его категории,
//! и возвращает `None`, если нет.

mod forward;
mod backward;
mod layers;

pub use forward::process_forward_gpu_buffered;
pub use backward::process_backward_gpu_buffered;
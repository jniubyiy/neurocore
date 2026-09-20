// src/layers/adapter/mod.rs

//! Инфраструктура градиентных адаптеров (MIGRATION_PLAN.md §7, Фазы 3–6).
//!
//! # Роль адаптеров
//!
//! Градиентный адаптер — компонент слоя, реализующий трейт
//! [`GradientAdapter`]. Он живёт в папке слоя (`src/layers/<layer>/adapter/`),
//! знает математику слоя и корректирует **свой собственный** градиент
//! (`grad_params`) между фазами `optimizer_modify_grads` и
//! `optimizer_apply_update`.
//!
//! # Инварианты (MIGRATION_PLAN.md §2)
//!
//!   * **I-3:** адаптер трогает только `grad_params`, не `grad_input`.
//!   * **I-4:** адаптер — часть слоя, живёт в его папке.
//!   * **I-5:** устройство адаптера = устройство слоя (CPU-слой → CPU-адаптер,
//!     GPU-слой → GPU-адаптер, без автоматических roundtrip'ов).
//!   * **I-8:** адаптеры работают через `&self`, состояние — внутри
//!     `Mutex`/`RwLock` (по образцу `BatchRenorm1d::state`).
//!   * **I-9:** каждый слой имеет право на свою формулу и своё состояние.
//!   * **I-10:** в первой итерации адаптер видит только слои своего сегмента.
//!
//! # Состав модуля
//!
//!   * [`GradientAdapter`] — трейт (файл `trait.rs`, подключён через
//!     `#[path]` как приватный модуль `adapter_trait`);
//!   * [`AdapterContext`] — контекст, передаваемый адаптеру в `apply`;
//!   * [`AdapterRegistry`] — реестр типов под save/load;
//!   * [`AdapterSummary`] и связанные — диагностика работы адаптеров
//!     (Фаза 6 плана);
//!   * метод `UniversalLayer::adapter()` с дефолтом `None`
//!     (расширение `src/layers/mod.rs`).

mod context;
mod registry;
mod stats;

// `trait` — зарезервированное слово Rust. Чтобы имя файла совпало
// с MIGRATION_PLAN.md (`trait.rs`), но имя модуля внутри Rust-кода
// осталось читаемым, используем атрибут `#[path]` и переименовываем
// модуль в `adapter_trait`. Публичный API — через `pub use` ниже.
#[path = "trait.rs"]
mod adapter_trait;

pub use context::AdapterContext;
pub use registry::{AdapterRegistry, AdapterTypeInfo};
pub use stats::{
    AdapterAggregate, AdapterCallStats, AdapterPassStats, AdapterSummary,
};
pub use adapter_trait::GradientAdapter;

// ============================================================================
// Юнит-тесты (Фаза 3, точка валидации §7)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_manager::core::device_spec::DeviceSpec;
    use crate::compute_manager::operators_v2::memory_v2::buffer::MatrixBufferHandle;
    use crate::compute_manager::operators_v2::memory_v2::executor::MemoryExecutor;
    use crate::compute_manager::operators_v2::memory_v2::policy::BufferPriority;
    use crate::compute_manager::operators_v2::memory_v2::types::MemoryDeviceKind;
    use crate::model_plan::param_store::ParamSlice;
    use std::sync::{Arc, RwLock};

    fn make_executor() -> Arc<RwLock<MemoryExecutor>> {
        let mem = Arc::new(RwLock::new(MemoryExecutor::new()));
        mem.write()
            .unwrap()
            .register_compute_device(DeviceSpec::cpu(0, 4096, 1), None);
        mem.write().unwrap().set_self_arc(mem.clone());
        mem
    }

    fn make_handle(mem: &Arc<RwLock<MemoryExecutor>>, size: usize) -> MatrixBufferHandle {
        let mut m = mem.write().unwrap();
        m.acquire_matrix_handle(size, 1, MemoryDeviceKind::HostRam, BufferPriority::Medium)
            .expect("acquire matrix handle")
    }

    /// No-op адаптер: ничего не делает. Прогон должен быть идентичен
    /// отсутствию адаптера.
    struct NoopAdapter;

    impl GradientAdapter for NoopAdapter {
        fn apply(&self, _ctx: &AdapterContext<'_>) {
            // Ничего не делаем.
        }
    }

    /// Адаптер, который масштабирует свой градиент. Демонстрирует
    /// read-modify-write-паттерн через `AdapterContext`.
    struct ScaleAdapter {
        factor: f32,
    }

    impl GradientAdapter for ScaleAdapter {
        fn apply(&self, ctx: &AdapterContext<'_>) {
            ctx.modify_own_grads(|g| {
                for v in g.iter_mut() {
                    *v *= self.factor;
                }
            });
        }

        fn name(&self) -> &'static str {
            "scale"
        }
    }

    #[test]
    fn noop_adapter_does_not_change_grads() {
        let mem = make_executor();
        let params = make_handle(&mem, 4);
        let grads = make_handle(&mem, 4);

        params.write_range(0, &[1.0, 2.0, 3.0, 4.0]);
        grads.write_range(0, &[0.1, 0.2, 0.3, 0.4]);

        let all_slices = vec![ParamSlice::new(0, 0, 4)];
        let ctx = AdapterContext {
            segment_params: &params,
            segment_grads: &grads,
            own_slice: ParamSlice::new(0, 0, 4),
            all_slices: &all_slices,
            batch: 1,
            optimizer_applied: true,
            own_state_slice: None,
            adapter_store: None,
            forward_ctx: None,
        };

        NoopAdapter.apply(&ctx);

        let after = grads.read_range(0, 4);
        assert_eq!(after, vec![0.1, 0.2, 0.3, 0.4],
            "no-op adapter must not modify grads");
    }

    #[test]
    fn scale_adapter_modifies_only_own_grads() {
        let mem = make_executor();
        let params = make_handle(&mem, 8);
        let grads = make_handle(&mem, 8);

        // Сегмент из двух «слоёв»: [0..4) и [4..8).
        params.write_range(0, &[0.0; 8]);
        grads.write_range(0, &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);

        let all_slices = vec![
            ParamSlice::new(0, 0, 4),
            ParamSlice::new(0, 4, 4),
        ];
        let ctx = AdapterContext {
            segment_params: &params,
            segment_grads: &grads,
            own_slice: ParamSlice::new(0, 0, 4), // первый «слой»
            all_slices: &all_slices,
            batch: 1,
            optimizer_applied: true,
            own_state_slice: None,
            adapter_store: None,
            forward_ctx: None,
        };

        ScaleAdapter { factor: 2.0 }.apply(&ctx);

        // Первый слой — умножен на 2, второй слой — не тронут.
        let after = grads.read_range(0, 8);
        assert_eq!(
            after,
            vec![2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 30.0, 40.0],
            "scale adapter must modify only its own slice"
        );
    }

    #[test]
    fn adapter_name_default_and_override() {
        assert_eq!(NoopAdapter.name(), "adapter");
        assert_eq!(ScaleAdapter { factor: 1.0 }.name(), "scale");
    }

    #[test]
    fn adapter_state_defaults_are_stateless() {
        assert!(!NoopAdapter.has_state());
        assert_eq!(NoopAdapter.state_size_per_param(), 0);
        assert!(!ScaleAdapter { factor: 1.0 }.has_state());
    }
}
// src/layers/adapter/context.rs

use std::sync::{Arc, Mutex};

use crate::compute_manager::core::dynamic_context::DynamicContext;
use crate::compute_manager::operators_v2::memory_v2::buffer::MatrixBufferHandle;
use crate::model_plan::adapter_store::{AdapterSlice, AdapterStateStore};
use crate::model_plan::param_store::ParamSlice;

/// Контекст, передаваемый адаптеру в [`GradientAdapter::apply`].
///
/// # Жизненный цикл
///
/// Создаётся `GraphV2::adapter_pass` (Фаза 4 плана) на время одного вызова
/// `apply` для конкретного адаптера. Все ссылки валидны на протяжении
/// вызова; после возврата из `apply` контекст уничтожается.
///
/// # Инварианты (MIGRATION_PLAN.md §2)
///
///   * **I-3:** `grad_input` в контексте отсутствует — адаптер не может
///     его исказить, даже если бы захотел.
///
///   * **I-10:** область видимости — один сегмент. `segment_params` и
///     `segment_grads` — буферы сегмента, `all_slices` — все срезы слоёв
///     этого сегмента. Cross-segment доступа нет.
///
///   * **I-5:** никаких автоматических миграций. Адаптер работает на том
///     устройстве, где лежат `segment_params`/`segment_grads`.
pub struct AdapterContext<'a> {
    /// Общий буфер параметров сегмента (все слои сегмента).
    ///
    /// Адаптер может читать любой срез (I-10). Модифицировать не должен:
    /// параметры обновляются только фазой `optimizer_apply_update`.
    pub segment_params: &'a MatrixBufferHandle,

    /// Общий буфер градиентов параметров сегмента (все слои сегмента).
    ///
    /// Адаптер имеет право модифицировать **свой** срез
    /// (`ctx.own_slice`). Чужие срезы — только чтение.
    pub segment_grads: &'a MatrixBufferHandle,

    /// Срез этого адаптера в `segment_params` / `segment_grads`.
    pub own_slice: ParamSlice,

    /// Все срезы слоёв сегмента, в порядке следования слоёв
    /// (включая `own_slice`).
    ///
    /// `all_slices[i]` соответствует i-му слою сегмента. Для adapter_pass
    /// актуален при cross-layer математике внутри сегмента.
    pub all_slices: &'a [ParamSlice],

    /// Размер батча (число строк активаций сегмента на forward).
    pub batch: usize,

    /// `true`, если фаза `optimizer_modify_grads` уже прошла.
    ///
    /// В нормальном цикле обучения adapter_pass вызывается строго
    /// после `modify_grads`, поэтому это поле всегда `true`. Оно
    /// сохранено на случай будущих сценариев (например, «только
    /// адаптеры, без оптимизатора») — адаптер может его проверять
    /// и паниковать, если что-то не так.
    pub optimizer_applied: bool,

    /// Срез состояния этого адаптера в `adapter_store`.
    ///
    /// `None` для stateless-адаптеров и для Фазы 3 (адаптеров нет).
    pub own_state_slice: Option<AdapterSlice>,

    /// Общий `AdapterStateStore` сессии (Фаза 2 плана).
    ///
    /// Клонируется `Arc` — дёшево. Адаптер использует `own_state_slice`
    /// для локализации своего состояния внутри store.
    ///
    /// `None`, если граф по какой-то причине работает без store
    /// (в v2 такое невозможно, но поле оставлено для гибкости).
    pub adapter_store: Option<Arc<Mutex<AdapterStateStore>>>,

    /// Forward-state слоя, если слой его сохранил в `BufferedContext`.
    ///
    /// Адаптер может матчить вариант `BufferedContext` и читать из него
    /// любые дескрипторы (вход, выход, промежуточные буферы слоя).
    ///
    /// `None`, если слой не сохранял контекст либо адаптер вызывается
    /// в контексте, где forward-cache недоступен.
    ///
    /// # Фаза 4+
    ///
    /// На Фазе 3 поле всегда `None` (adapter_pass ещё не реализован).
    /// Фаза 4 заполнит его из forward_cache после backward.
    pub forward_ctx: Option<&'a DynamicContext>,
}

impl<'a> AdapterContext<'a> {
    /// Читает значения параметров этого адаптера.
    ///
    /// Возвращает owned `Vec<f32>` длины `own_slice.len`.
    #[inline]
    pub fn read_own_params(&self) -> Vec<f32> {
        self.segment_params
            .read_range(self.own_slice.start, self.own_slice.len)
    }

    /// Читает значения градиентов этого адаптера.
    ///
    /// Возвращает owned `Vec<f32>` длины `own_slice.len`.
    #[inline]
    pub fn read_own_grads(&self) -> Vec<f32> {
        self.segment_grads
            .read_range(self.own_slice.start, self.own_slice.len)
    }

    /// Записывает значения градиентов этого адаптера in-place.
    ///
    /// # Паника
    /// Паникует, если длина `data` не совпадает с `own_slice.len`.
    #[inline]
    pub fn write_own_grads(&self, data: &[f32]) {
        assert_eq!(
            data.len(),
            self.own_slice.len,
            "AdapterContext::write_own_grads: data length ({}) != own_slice.len ({})",
            data.len(),
            self.own_slice.len
        );
        self.segment_grads
            .write_range(self.own_slice.start, data);
    }

    /// Read-modify-write удобным способом.
    ///
    /// # Пример
    ///
    /// ```ignore
    /// ctx.modify_own_grads(|g| {
    ///     for v in g.iter_mut() { *v *= 2.0; }
    /// });
    /// ```
    #[inline]
    pub fn modify_own_grads<F: FnOnce(&mut [f32])>(&self, f: F) {
        let mut data = self.read_own_grads();
        f(&mut data);
        self.write_own_grads(&data);
    }
}
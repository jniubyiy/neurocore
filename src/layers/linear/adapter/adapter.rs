// src/layers/linear/adapter/adapter.rs

use std::sync::atomic::{AtomicBool, Ordering};

use crate::layers::adapter::{AdapterContext, GradientAdapter};

/// No-op градиентный адаптер слоя `Linear`.
///
/// # Назначение (MIGRATION_PLAN.md §7, Фаза 5 — пилот)
///
/// Пилотный адаптер для проверки **среды адаптеров**. Он не модифицирует
/// градиент — только валидирует контракт [`AdapterContext`] через
/// `debug_assert`. Любое нарушение инвариантов (неверный срез, отсутствие
/// forward-контекста, отсутствие adapter_store, `optimizer_applied == false`)
/// обнаруживается сразу — в debug-сборке через панику, а не где-то позже
/// в процессе обучения.
///
/// # Инварианты, которые валидирует адаптер (MIGRATION_PLAN.md §2)
///
///   * **I-1** (фазовый порядок): `optimizer_applied == true`.
///   * **I-10** (сегмент-local): `own_slice` — легитимный член `all_slices`,
///     его конец не выходит за пределы `segment_params` / `segment_grads`.
///   * **Фаза 4** (жизненный цикл cache): `forward_ctx.is_some()`.
///   * **Интеграция со Фазой 2**: `adapter_store.is_some()`.
///
/// # Модификация градиента
///
/// Отсутствует. Числовые результаты примеров **идентичны** baseline
/// (конец Фазы 4). Это точка валидации §7 Фазы 5.
///
/// # Потокобезопасность
///
/// `LinearAdapter` содержит только `AtomicBool` для однократного
/// диагностического сообщения. Никакого персистентного состояния
/// между шагами нет (адаптер stateless).
pub struct LinearAdapter {
    /// Флаг «первый вызов уже был». Используется только для однократного
    /// диагностического сообщения в debug-сборке.
    ///
    /// В release-сборке всегда `false`, к нему нет обращений по горячему
    /// пути. Поле сохранено, чтобы структура была одинаковой в debug и
    /// release (никаких `#[cfg]` на уровне полей).
    reported: AtomicBool,
}

impl LinearAdapter {
    /// Создаёт no-op адаптер.
    pub fn new() -> Self {
        Self {
            reported: AtomicBool::new(false),
        }
    }
}

impl Default for LinearAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl GradientAdapter for LinearAdapter {
    fn apply(&self, ctx: &AdapterContext<'_>) {
        // ====================================================================
        // Валидация контракта AdapterContext (MIGRATION_PLAN.md §2).
        //
        // Все проверки — `debug_assert`: срабатывают только в debug-сборке,
        // в release вырезаются компилятором. Это суть пилота: убедиться,
        // что среда адаптеров полностью прокинута и передаёт корректные
        // данные, до появления первого настоящего адаптера с формулой.
        // ====================================================================

        // ---- I-10: own_slice лежит внутри segment_params и segment_grads ----
        let params_len = ctx.segment_params.rows() * ctx.segment_params.cols();
        let grads_len = ctx.segment_grads.rows() * ctx.segment_grads.cols();
        debug_assert!(
            ctx.own_slice.end() <= params_len,
            "LinearAdapter: own_slice.end()={} exceeds segment_params len={} \
             (buffer_idx={}, start={}, len={})",
            ctx.own_slice.end(),
            params_len,
            ctx.own_slice.buffer_idx(),
            ctx.own_slice.start,
            ctx.own_slice.len
        );
        debug_assert!(
            ctx.own_slice.end() <= grads_len,
            "LinearAdapter: own_slice.end()={} exceeds segment_grads len={} \
             (buffer_idx={}, start={}, len={})",
            ctx.own_slice.end(),
            grads_len,
            ctx.own_slice.buffer_idx(),
            ctx.own_slice.start,
            ctx.own_slice.len
        );

        // ---- Форма params и grads должна совпадать ----
        // Иначе backprop в принципе не сходится: градиенты разной формы
        // не смогут быть вычтены из параметров в apply_update.
        debug_assert_eq!(
            ctx.segment_params.rows(),
            ctx.segment_grads.rows(),
            "LinearAdapter: segment_params.rows ({}) != segment_grads.rows ({})",
            ctx.segment_params.rows(),
            ctx.segment_grads.rows()
        );
        debug_assert_eq!(
            ctx.segment_params.cols(),
            ctx.segment_grads.cols(),
            "LinearAdapter: segment_params.cols ({}) != segment_grads.cols ({})",
            ctx.segment_params.cols(),
            ctx.segment_grads.cols()
        );

        // ---- batch > 0 ----
        // cache.batch сохраняется из input.rows() при forward;
        // нулевой batch означал бы, что forward не выполнялся.
        debug_assert!(
            ctx.batch > 0,
            "LinearAdapter: batch must be positive, got {}",
            ctx.batch
        );

        // ---- I-1: adapter_pass вызывается после optimizer_modify_grads ----
        // Фазовый порядок (MIGRATION_PLAN.md §2): forward → loss → backward
        // → optimizer_modify_grads → adapter_pass → optimizer_apply_update.
        // Если флаг `false`, значит кто-то нарушил порядок фаз.
        debug_assert!(
            ctx.optimizer_applied,
            "LinearAdapter: optimizer_applied must be true. \
             Phase-order violation (MIGRATION_PLAN.md §2, инвариант I-1): \
             adapter_pass must be called after optimizer_modify_grads."
        );

        // ---- I-10: own_slice — легитимный член all_slices ----
        // `all_slices` содержит срезы слоёв сегмента. `own_slice` адаптера
        // должен быть одним из них — иначе это чужой срез или ошибка индексации.
        debug_assert!(
            ctx.all_slices.iter().any(|s| *s == ctx.own_slice),
            "LinearAdapter: own_slice {:?} is not present in all_slices {:?}",
            ctx.own_slice,
            ctx.all_slices
        );

        // ---- Фаза 4: forward_ctx присутствует ----
        // MIGRATION_PLAN.md §7, Фаза 4: forward_cache живёт до
        // optimizer_apply_update. К моменту adapter_pass он обязан
        // быть заполнен.
        debug_assert!(
            ctx.forward_ctx.is_some(),
            "LinearAdapter: forward_ctx is missing. \
             Likely cause: forward_cache was cleared too early \
             (MIGRATION_PLAN.md §7, Фаза 4: cache must survive until apply_update)."
        );

        // ---- Фаза 2: adapter_store доступен ----
        debug_assert!(
            ctx.adapter_store.is_some(),
            "LinearAdapter: adapter_store is missing. \
             Likely cause: GraphV2::build did not initialize AdapterStateStore \
             or AdapterContext was constructed incorrectly."
        );

        // ====================================================================
        // No-op: адаптер не модифицирует градиент.
        // ====================================================================
        //
        // По решению администратора пилотный адаптер только проверяет
        // среду, не меняет числа. Никакого `ctx.modify_own_grads(...)`
        // здесь нет и не должно быть до следующей фазы.

        // ---- Однократное диагностическое сообщение (только debug) ----
        #[cfg(debug_assertions)]
        {
            if !self.reported.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[adapter] LinearAdapter no-op activated \
                     (pilot phase: validating adapter environment, \
                     grads are NOT modified)"
                );
            }
        }

        // В release-сборке `reported` не читается по горячему пути.
        // Заглушка ниже — только чтобы поле не считалось dead_code.
        #[cfg(not(debug_assertions))]
        {
            let _ = self.reported.load(Ordering::Relaxed);
        }
    }

    fn name(&self) -> &'static str {
        "linear"
    }

    // `has_state` / `state_size_per_param` — дефолт (false / 0):
    // пилотный адаптер stateless.
}
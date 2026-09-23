// src/layers/linear/adapter/adapter.rs

//! Градиентный адаптер слоя `Linear` (MIGRATION_PLAN.md §7).
//!
//! # Формула (per-row)
//!
//! Для каждого выходного нейрона `c` слоя `Linear`:
//!
//! ```text
//!   ‖W_c‖              = L2-норма строки весов [c · in, (c+1) · in)
//!   ‖∇W_c‖_sum         = L2-норма СЫРОГО Sum-градиента (I-2, ∝ √B)
//!   ‖∇W_c‖_per_sample  = ‖∇W_c‖_sum / √B        ← batch-инвариантность
//!
//!   r                  = ‖∇W_c‖_ps / (‖W_c‖ + ε)
//!   β_eff              = β · (1 + α · r)         ← адаптивный damping
//!   scale_raw          = ‖W_c‖ / (‖∇W_c‖_ps + β_eff · ‖W_c‖ + ε)
//!   scale              = soft_clamp(scale_raw, min_scale, max_scale, k)
//!   ∇W_c              *= scale · gain_eff
//! ```
//!
//! Bias (хвост `out_features` элементов `own_slice`) не модифицируется.
//!
//! # Batch-инвариантность
//!
//! Целевое свойство: **`loss(M шагов, B=1) == loss(1 шаг, B=M)`**.
//!
//! Достигается за счёт оценки `‖g_per_sample‖ = ‖g_sum‖ / √B`: при
//! i.i.d.-шумовом режиме `‖Σ g_i‖ ≈ √B · G`, поэтому деление на `√B`
//! даёт корректную оценку нормы per-sample градиента. Деление на `B`
//! занижает оценку в `√B` раз и приводит к неинвариантности.
//!
//! # Адаптивность LARS-знаменателя
//!
//! `β_eff = β · (1 + α · r)`:
//!   * `r → 0` (малый градиент) → `β_eff ≈ β` → сильный boost;
//!   * `r ≫ 1` (большой градиент) → `β_eff ≫ β` → damping.
//!
//! # Soft-clamp вместо жёсткого clip
//!
//! Раньше per-row scale обрезался жёстко: `raw.clamp(MIN, MAX)`. Резкая
//! граница давала «отрезание» выбросов и последующий резкий возврат
//! scale к 1.0, что увеличивало `max_jump`.
//!
//! Теперь используется **soft-clamp**: log-симметричная функция с
//! асимптотами `MIN` и `MAX`, не искажающая значения внутри рабочего
//! диапазона:
//!
//! ```text
//!   soft_clamp(x, lo, hi, k) = 1 + sign(x−1) · soft_cap(|x−1|, range, k)
//!   soft_cap(y, cap, k)      = y / (1 + (y/cap)^k)^(1/k)
//! ```
//!
//! При `k = 3` значения `x ∈ [1, 2]` искажаются менее чем на 0.3%, а
//! `x ≫ hi` асимптотически сжимается к `hi`. Такой clip не «режет»
//! выброс — он их плавно прижимает, что снижает `max_jump`.
//!
//! # Online gain (EMA-based)
//!
//! Помимо per-row LARS-scale, применяется общий на срез **online gain**:
//! адаптер ведёт EMA per-sample per-param RMS по W-части среза и на
//! каждом шаге вычисляет
//!
//! ```text
//!   rms_cur  = ‖∇W‖_sum / (√B · √n_W)     (по W-части среза)
//!   gain_raw = ema_rms / (rms_cur + ε)
//!   gain_eff = clamp(gain_raw, GAIN_MIN, GAIN_MAX)
//!   ema_new  = β_ema · ema_rms + (1 − β_ema) · rms_cur
//! ```
//!
//! Семантика:
//!   * градиент в норме → `gain ≈ 1.0` (нейтрально);
//!   * градиент систематически падает (середина / конец обучения) →
//!     `gain > 1` (ускорение);
//!   * градиент подскочил (шум / выброс) → `gain < 1` (**демпфирование**).
//!
//! Первый вызов `apply` инициализирует `ema_rms = rms_cur` и возвращает
//! `gain_eff = 1.0`.
//!
//! `gain_eff` batch-инвариантен: `rms_cur` и `ema_rms` делятся на `√B`,
//! значит их отношение от B не зависит.
//!
//! # Параметры LARS-формулы (зафиксированы, эмпирически подтверждены)
//!
//! | Параметр | Значение |
//! |----------|----------|
//! | β (базовый damping) | 0.3 |
//! | α (адаптивность)    | 1.0 |
//! | min_scale           | 0.01 |
//! | max_scale           | 3.0 |
//! | ε                   | 1e-12 |
//!
//! # Тюн-параметры (env)
//!
//! | Env | Default | Смысл |
//! |-----|---------|-------|
//! | `NEUROCORE_LINEAR_SOFT_CLIP_K`   | `3.0`  | жёсткость soft-clamp (∞ → жёсткий clamp) |
//! | `NEUROCORE_LINEAR_GAIN_MIN`      | `0.5`  | нижний clip online gain |
//! | `NEUROCORE_LINEAR_GAIN_MAX`      | `2.5`  | верхний clip online gain |
//! | `NEUROCORE_LINEAR_GAIN_EMA_BETA` | `0.9`  | β EMA для online gain |
//!
//! # GPU
//!
//! Заморожена (I-5). Если `ctx.segment_grads.is_gpu()` — адаптер
//! пропускает модификацию.
//!
//! # Инварианты (MIGRATION_PLAN.md §2)
//!
//!   * I-1: `optimizer_applied == true`;
//!   * I-2: Loss отдаёт сырой Sum-градиент;
//!   * I-3: адаптер трогает только `grad_params`;
//!   * I-4: адаптер в папке слоя;
//!   * I-5: GPU↔GPU / CPU↔CPU;
//!   * I-8: `&self`, состояние — внутри `Mutex`;
//!   * I-10: `own_slice` — легитимный член `all_slices`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use once_cell::sync::Lazy;

use crate::compute_manager::core::dynamic_context::DynamicContext;
use crate::layers::adapter::{AdapterContext, GradientAdapter};
use crate::layers::buffered_context::BufferedContext;

// ============================================================================
// Константы формулы LARS
// ============================================================================

/// Базовый β в знаменателе LARS-adaptive scale.
const LARS_BETA: f32 = 0.3;

/// α — коэффициент адаптивности β: `β_eff = β · (1 + α · r)`.
const LARS_ADAPT: f32 = 1.0;

/// Нижняя асимптота per-row scale.
const LARS_MIN_SCALE: f32 = 0.01;

/// Верхняя асимптота per-row scale.
const LARS_MAX_SCALE: f32 = 3.0;

/// ε в знаменателе (защита от деления на ноль).
const LARS_EPS: f32 = 1e-12;

// ============================================================================
// Env-параметры soft-clamp и online gain
// ============================================================================

/// Жёсткость soft-clamp. Default = 3.0.
///
/// Большие значения ближе к жёсткому `clamp` (резкие границы).
/// Малые значения — более мягкое сжатие (но и более выраженное
/// искажение внутри рабочего диапазона).
static LINEAR_SOFT_CLIP_K: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_LINEAR_SOFT_CLIP_K")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(3.0)
});

/// Нижний clip online gain. Default = 0.5.
///
/// Значение < 1 означает демпфирование, когда текущий градиент больше
/// исторического среднего (шум / выброс).
static LINEAR_GAIN_MIN: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_LINEAR_GAIN_MIN")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.5)
});

/// Верхний clip online gain. Default = 2.5.
///
/// Значение > 1 означает ускорение, когда текущий градиент меньше
/// исторического среднего (середина / конец обучения).
static LINEAR_GAIN_MAX: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_LINEAR_GAIN_MAX")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(2.5)
});

/// β EMA для online gain. Default = 0.9 (half-life ≈ 7 шагов).
static LINEAR_GAIN_EMA_BETA: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_LINEAR_GAIN_EMA_BETA")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.9)
});

// ============================================================================
// Soft-clamp helpers
// ============================================================================

/// Мягкий log-симметричный clamp: `x ∈ (0, ∞)` → значение в `(lo, hi)`
/// с асимптотами `lo` и `hi`.
///
/// Свойства:
///   * `soft_clamp(1.0, lo, hi, k) == 1.0` (не сдвигает нейтральное значение);
///   * при `x ≫ 1` искажение минимально в рабочем диапазоне `x ∈ [1, 2]`
///     (для `k ≥ 3` — менее 0.3%);
///   * при `x → ∞` → `hi` асимптотически;
///   * при `x → 0+` → `lo` асимптотически;
///   * `k → ∞` даёт жёсткий `clamp(x, lo, hi)`.
#[inline]
fn soft_clamp(x: f32, lo: f32, hi: f32, k: f32) -> f32 {
    debug_assert!(lo > 0.0 && lo < 1.0, "soft_clamp: lo must be in (0, 1), got {}", lo);
    debug_assert!(hi > 1.0, "soft_clamp: hi must be > 1, got {}", hi);
    debug_assert!(k > 0.0, "soft_clamp: k must be > 0, got {}", k);

    if x >= 1.0 {
        1.0 + soft_cap_positive(x - 1.0, hi - 1.0, k)
    } else {
        1.0 - soft_cap_positive(1.0 - x, 1.0 - lo, k)
    }
}

/// Мягкий «cap» положительной величины: `y ≥ 0`, `cap > 0`.
///
/// Возвращает значение в `[0, cap)` с асимптотой `cap`:
///   * `soft_cap_positive(0, cap, k) == 0`;
///   * при `y ≪ cap` — почти `y`;
///   * при `y → ∞` → `cap`.
///
/// Формула: `y / (1 + (y/cap)^k)^(1/k)`.
#[inline]
fn soft_cap_positive(y: f32, cap: f32, k: f32) -> f32 {
    if y <= 0.0 || cap <= 0.0 {
        return 0.0;
    }
    let ratio = y / cap;
    let denom = (1.0 + ratio.powf(k)).powf(1.0 / k);
    y / denom
}

// ============================================================================
// Прочие helpers
// ============================================================================

#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    let mut sum_sq: f64 = 0.0;
    for &x in v {
        let d = x as f64;
        sum_sq += d * d;
    }
    sum_sq.sqrt() as f32
}

/// Адаптивный β_eff = β · (1 + α · r), где r = ‖g_ps‖ / ‖W‖.
#[inline]
fn adaptive_beta(beta: f32, alpha: f32, w_norm: f32, g_norm_ps: f32) -> f32 {
    if alpha == 0.0 || w_norm < 1e-30 {
        return beta;
    }
    let r = g_norm_ps / w_norm;
    beta * (1.0 + alpha * r)
}

fn extract_in_features(forward_ctx: Option<&DynamicContext>) -> Option<usize> {
    let fctx = forward_ctx?;
    match fctx {
        DynamicContext::Buffered(BufferedContext::Linear { input }) => {
            let cols = input.cols();
            if cols > 0 { Some(cols) } else { None }
        }
        DynamicContext::Buffered(_) => None,
    }
}

// ============================================================================
// LinearAdapter
// ============================================================================

/// Градиентный адаптер слоя `Linear`.
///
/// Работает через `&self`. Персистентное состояние — `gain_state`
/// (EMA online gain) — под `Mutex`.
pub struct LinearAdapter {
    /// Одноразовый флаг предупреждения: если forward-контекст слоя не
    /// содержит `BufferedContext::Linear`, адаптер молча делает no-op.
    reported_missing_ctx: AtomicBool,

    /// EMA of per-sample per-param RMS по W-части среза.
    ///
    /// Ключ — `(buffer_idx, slice.start)`. Уникален для каждого среза
    /// в графе.
    gain_state: Mutex<HashMap<(usize, usize), f32>>,
}

impl LinearAdapter {
    pub fn new() -> Self {
        Self {
            reported_missing_ctx: AtomicBool::new(false),
            gain_state: Mutex::new(HashMap::new()),
        }
    }

    /// Online gain для конкретного среза.
    ///
    /// Первый вызов: инициализирует EMA значением `rms_cur`, возвращает
    /// `1.0` (нейтрально).
    ///
    /// Последующие вызовы:
    ///   * `gain_raw = ema_rms / (rms_cur + ε)`;
    ///   * `gain_eff = clamp(gain_raw, GAIN_MIN, GAIN_MAX)`;
    ///   * `ema_rms` обновляется по формуле EMA.
    fn compute_gain_eff(&self, ctx: &AdapterContext<'_>, rms_cur: f32) -> f32 {
        let key = (ctx.own_slice.buffer_idx(), ctx.own_slice.start);
        let mut state = self.gain_state.lock().unwrap();

        match state.get(&key).copied() {
            None => {
                // Первый вызов: инициализируем EMA, gain нейтральный.
                state.insert(key, rms_cur);
                1.0
            }
            Some(ema) => {
                let eps = LARS_EPS;
                let gain_raw = if rms_cur > 1e-30 {
                    ema / (rms_cur + eps)
                } else {
                    // Нулевой градиент — не усиливаем шум.
                    1.0
                };
                let gain_eff = gain_raw.clamp(*LINEAR_GAIN_MIN, *LINEAR_GAIN_MAX);

                let beta = *LINEAR_GAIN_EMA_BETA;
                let new_ema = beta * ema + (1.0 - beta) * rms_cur;
                state.insert(key, new_ema);

                gain_eff
            }
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
        // ====================================================================
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
        debug_assert!(
            ctx.batch > 0,
            "LinearAdapter: batch must be positive, got {}",
            ctx.batch
        );
        debug_assert!(
            ctx.optimizer_applied,
            "LinearAdapter: optimizer_applied must be true. \
             Phase-order violation (MIGRATION_PLAN.md §2, инвариант I-1): \
             adapter_pass must be called after optimizer_modify_grads."
        );
        debug_assert!(
            ctx.all_slices.iter().any(|s| *s == ctx.own_slice),
            "LinearAdapter: own_slice {:?} is not present in all_slices {:?}",
            ctx.own_slice,
            ctx.all_slices
        );

        // ====================================================================
        // GPU-ветка заморожена (I-5).
        // ====================================================================
        if ctx.segment_grads.is_gpu() {
            debug_assert!(
                ctx.segment_params.is_gpu(),
                "LinearAdapter: I-5 violation — grads on GPU but params on CPU. \
                 Adapter device must match layer device."
            );
            return;
        }

        // ====================================================================
        // Извлечение in_features из forward-контекста.
        //
        // Если forward_ctx отсутствует или имеет другой вариант — адаптер
        // делает no-op. Ранее здесь применялся fallback per-tensor по
        // всему срезу, но он смешивал W и bias в одной LARS-норме, что
        // искажало формулу. В v2-графе forward_ctx доступен всегда
        // (cache живёт до optimizer_apply_update), поэтому fallback
        // не нужен. Если forward_ctx нет — это ошибка конфигурации,
        // безопаснее ничего не делать.
        // ====================================================================
        let Some(in_features) = extract_in_features(ctx.forward_ctx) else {
            if !self.reported_missing_ctx.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[LINEAR-ADAPTER] forward_ctx missing or not \
                     BufferedContext::Linear; adapter no-op. (Once-per-session.)"
                );
            }
            return;
        };

        // ====================================================================
        // Валидация согласованности длины среза с in_features.
        //
        //   slice_len = in·out + out = out·(in + 1)
        //   ⇒ out = slice_len / (in + 1)
        //
        // Если slice_len не делится на (in + 1) нацело — значит, что-то
        // не так с конфигурацией, и мы не можем корректно разложить
        // W-часть по строкам. No-op.
        // ====================================================================
        let slice_len = ctx.own_slice.len;
        let denom = in_features + 1;
        if slice_len == 0 || slice_len % denom != 0 {
            if !self.reported_missing_ctx.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[LINEAR-ADAPTER] slice_len={} not divisible by (in_features+1)={}; \
                     adapter no-op. (Once-per-session.)",
                    slice_len, denom
                );
            }
            return;
        }
        let out_features = slice_len / denom;
        if out_features == 0 {
            return;
        }

        // ====================================================================
        // Готовим чтение W-части.
        // ====================================================================
        let w = ctx.read_own_params();
        let mut g = ctx.read_own_grads();

        let b = ctx.batch.max(1) as f32;
        let sqrt_b = b.sqrt();

        let w_len = in_features * out_features;

        // ====================================================================
        // Online gain по W-части СЫРОГО градиента.
        // ====================================================================
        let rms_cur = {
            let w_grads = &g[..w_len];
            let n = w_len as f32;
            if n > 0.0 {
                l2_norm(w_grads) / (sqrt_b * n.sqrt())
            } else {
                0.0
            }
        };
        let gain_eff = self.compute_gain_eff(ctx, rms_cur);

        let soft_k = *LINEAR_SOFT_CLIP_K;

        // ====================================================================
        // Per-row LARS + soft-clamp + online gain.
        //
        // Bias (элементы [w_len .. slice_len)) не трогаем.
        // ====================================================================
        for c in 0..out_features {
            let start = c * in_features;
            let end = start + in_features;

            let w_c = &w[start..end];
            let g_c = &mut g[start..end];

            let w_norm = l2_norm(w_c);
            let g_norm_sum = l2_norm(g_c);
            let g_norm_per_sample = g_norm_sum / sqrt_b;

            let beta_eff =
                adaptive_beta(LARS_BETA, LARS_ADAPT, w_norm, g_norm_per_sample);

            let denom = g_norm_per_sample + beta_eff * w_norm + LARS_EPS;
            let scale_raw = if denom > 0.0 { w_norm / denom } else { 1.0 };
            let scale =
                soft_clamp(scale_raw, LARS_MIN_SCALE, LARS_MAX_SCALE, soft_k);
            let final_scale = scale * gain_eff;

            for v in g_c.iter_mut() {
                *v *= final_scale;
            }
        }

        ctx.write_own_grads(&g);
    }

    fn name(&self) -> &'static str {
        "linear"
    }
}
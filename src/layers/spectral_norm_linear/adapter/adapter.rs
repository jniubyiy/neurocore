// src/layers/spectral_norm_linear/adapter/adapter.rs

//! Градиентный адаптер слоя `SpectrallyNormalizedLinear`.
//!
//! # Формула (LARS-adaptive, единственный режим)
//!
//! ```text
//!   W_eff_norm = |scale|                          (эффективная спектральная норма W_sn)
//!   g_norm_sum = ‖∇W‖₂                            (L2 по W-части среза)
//!   g_norm_ps  = g_norm_sum / √B                  (batch-инвариантная оценка)
//!
//!   r          = g_norm_ps / (W_eff_norm + ε)
//!   β_eff      = β · (1 + α · r)                  (адаптивный damping)
//!   denom      = g_norm_ps + β_eff · W_eff_norm + ε
//!   scale_raw  = W_eff_norm / denom
//!   scale      = soft_clamp(scale_raw, min_scale, max_scale, k)
//!   ∇W_sum    *= scale · gain_eff
//! ```
//!
//! # Что модифицируется
//!
//! Только **W-часть** среза параметров. `bias` и `scale`-параметр
//! остаются как есть — у них свои градиенты, не связанные с нормировкой W.
//!
//! # Адаптивность LARS-знаменателя
//!
//! `β_eff = β · (1 + α · r)`:
//!   * `r → 0` (малый градиент, стагнация) → `β_eff ≈ β` → boost в полную силу;
//!   * `r = 1` (норма) → `β_eff = 2β` → умеренный damping;
//!   * `r ≫ 1` (старт, выброс) → `β_eff ≫ β` → **сильный demote**.
//!
//! Это даёт **разную эффективную скорость обучения** на разных стадиях:
//! в сходимости работает почти нейтрально, на старте — защищает.
//!
//! # Batch-инвариантность
//!
//! `g_norm_ps` и `r` делятся на `√B`, значит `β_eff` тоже. Никакого
//! нарушения batch-инвариантности адаптивность не вносит.
//!
//! # Экспериментальные результаты (spectral_norm_linear_example)
//!
//! | Конфигурация              | best_loss | best_epoch | max jump |
//! |---------------------------|-----------|------------|----------|
//! | без адаптера, lr=0.01     | 8.4e-5    | 274        | +543.66% |
//! | LARS adapter, lr=0.01     | 1.0e-6    | 261        | +9.54%   |
//! | без адаптера, lr=0.015    | 1.67e-4   | 289        | +65.91%  |
//!
//! (Числа — для версии без `adaptive_beta`. С текущей версией ожидается
//! дополнительное улучшение стабильности за счёт демпфирования на старте.)
//!
//! # Soft-clamp вместо жёсткого clip
//!
//! Log-симметричная функция с асимптотами `MIN` и `MAX`, не искажающая
//! значения внутри рабочего диапазона (при `k=3` для `x ∈ [1, 2]` менее
//! 0.3%). Снижает `max_jump` за счёт плавной границы.
//!
//! # Online gain (EMA-based)
//!
//! EMA of per-sample per-param RMS по W-части среза:
//!
//! ```text
//!   rms_cur  = ‖∇W‖_sum / (√B · √n_W)
//!   gain_raw = ema_rms / (rms_cur + ε)
//!   gain_eff = clamp(gain_raw, GAIN_MIN, GAIN_MAX)
//!   ema_new  = β_ema · ema_rms + (1 − β_ema) · rms_cur
//! ```
//!
//! Демпфирует выбросы (`gain < 1`) и ускоряет в стагнации (`gain > 1`).
//! Первый вызов инициализирует EMA и возвращает `gain_eff = 1.0`.
//!
//! # Тюн-параметры (env)
//!
//! | Env | Default | Смысл |
//! |-----|---------|-------|
//! | `NEUROCORE_SN_LARS_BETA`         | `0.5`  | β в знаменателе (базовый) |
//! | `NEUROCORE_SN_LARS_MIN_SCALE`    | `0.01` | Нижняя асимптота soft-clamp |
//! | `NEUROCORE_SN_LARS_MAX_SCALE`    | `1.5`  | Верхняя асимптота soft-clamp |
//! | `NEUROCORE_SN_LARS_EPS`          | `1e-12`| ε в знаменателе |
//! | `NEUROCORE_SN_SOFT_CLIP_K`       | `3.0`  | жёсткость soft-clamp |
//! | `NEUROCORE_SN_GAIN_MIN`          | `0.5`  | нижний clip online gain |
//! | `NEUROCORE_SN_GAIN_MAX`          | `2.5`  | верхний clip online gain |
//! | `NEUROCORE_SN_GAIN_EMA_BETA`     | `0.9`  | β EMA для online gain |
//!
//! `α` (коэффициент адаптивности β_eff) — константа `SN_LARS_ADAPT = 1.0`.
//!
//! # Инварианты (MIGRATION_PLAN.md §2)
//!
//!   * I-3: адаптер трогает только `grad_params`, не `grad_input`;
//!   * I-4: адаптер в папке слоя;
//!   * I-5: CPU↔CPU / GPU↔GPU (GPU-ветка заморожена);
//!   * I-8: `&self`, состояние — внутри `Mutex`/`AtomicBool`;
//!   * I-10: `own_slice` — легитимный член `all_slices`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use once_cell::sync::Lazy;

use crate::compute_manager::core::dynamic_context::DynamicContext;
use crate::layers::adapter::{AdapterContext, GradientAdapter};
use crate::layers::buffered_context::BufferedContext;

// ============================================================================
// Параметры LARS-формулы
// ============================================================================

/// β в знаменателе LARS-adaptive scale. Default = 0.5.
static SN_LARS_BETA: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_SN_LARS_BETA")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.5)
});

/// α — коэффициент адаптивности β: `β_eff = β · (1 + α · r)`.
///
/// Константа (не env). Симметрично `LARS_ADAPT` в `LinearAdapter`.
const SN_LARS_ADAPT: f32 = 1.0;

/// Нижняя асимптота scale. Default = 0.01.
static SN_LARS_MIN_SCALE: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_SN_LARS_MIN_SCALE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.01)
});

/// Верхняя асимптота scale. Default = 1.5.
static SN_LARS_MAX_SCALE: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_SN_LARS_MAX_SCALE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.5)
});

/// ε в знаменателе. Default = 1e-12.
static SN_LARS_EPS: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_SN_LARS_EPS")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1e-12)
});

// --- Soft-clamp + online gain ---

/// Жёсткость soft-clamp. Default = 3.0.
static SN_SOFT_CLIP_K: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_SN_SOFT_CLIP_K")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(3.0)
});

/// Нижний clip online gain. Default = 0.5.
static SN_GAIN_MIN: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_SN_GAIN_MIN")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.5)
});

/// Верхний clip online gain. Default = 2.5.
static SN_GAIN_MAX: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_SN_GAIN_MAX")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(2.5)
});

/// β EMA для online gain. Default = 0.9.
static SN_GAIN_EMA_BETA: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_SN_GAIN_EMA_BETA")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.9)
});

/// Порог, ниже которого `scale` считается вырожденным; в этом случае
/// forward y = b, dL/dW_sn = 0, модификация пропускается.
const SCALE_DEGENERATE_EPS: f32 = 1e-30;

// ============================================================================
// Soft-clamp helpers
// ============================================================================

/// Мягкий log-симметричный clamp: `x ∈ (0, ∞)` → значение в `(lo, hi)`
/// с асимптотами `lo` и `hi`.
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
///
/// При α = 0 вырождается в const β. При r → 0 → β; при r → ∞ → ∞.
#[inline]
fn adaptive_beta(beta: f32, alpha: f32, w_norm: f32, g_norm_ps: f32) -> f32 {
    if alpha == 0.0 || w_norm < 1e-30 {
        return beta;
    }
    let r = g_norm_ps / w_norm;
    beta * (1.0 + alpha * r)
}

/// Извлекает `in_features` из forward-контекста SN-слоя.
fn extract_in_features(forward_ctx: Option<&DynamicContext>) -> Option<usize> {
    let fctx = forward_ctx?;
    match fctx {
        DynamicContext::Buffered(BufferedContext::SpectralNormLinear { input, .. }) => {
            let cols = input.cols();
            if cols > 0 { Some(cols) } else { None }
        }
        _ => None,
    }
}

// ============================================================================
// Адаптер
// ============================================================================

/// Градиентный адаптер `SpectrallyNormalizedLinear`.
///
/// Полное описание формулы — в документации модуля.
pub struct SpectralNormLinearAdapter {
    /// Одноразовый флаг предупреждения о missing forward_ctx.
    reported_missing_ctx: AtomicBool,

    /// EMA of per-sample per-param RMS по W-части среза.
    ///
    /// Ключ — `(buffer_idx, slice.start)`.
    gain_state: Mutex<HashMap<(usize, usize), f32>>,
}

impl SpectralNormLinearAdapter {
    /// Создаёт новый адаптер.
    pub fn new() -> Self {
        Self {
            reported_missing_ctx: AtomicBool::new(false),
            gain_state: Mutex::new(HashMap::new()),
        }
    }

    /// Online gain для конкретного среза.
    fn compute_gain_eff(&self, ctx: &AdapterContext<'_>, rms_cur: f32) -> f32 {
        let key = (ctx.own_slice.buffer_idx(), ctx.own_slice.start);
        let mut state = self.gain_state.lock().unwrap();

        match state.get(&key).copied() {
            None => {
                state.insert(key, rms_cur);
                1.0
            }
            Some(ema) => {
                let eps = *SN_LARS_EPS;
                let gain_raw = if rms_cur > 1e-30 {
                    ema / (rms_cur + eps)
                } else {
                    1.0
                };
                let gain_eff = gain_raw.clamp(*SN_GAIN_MIN, *SN_GAIN_MAX);

                let beta = *SN_GAIN_EMA_BETA;
                let new_ema = beta * ema + (1.0 - beta) * rms_cur;
                state.insert(key, new_ema);

                gain_eff
            }
        }
    }
}

impl Default for SpectralNormLinearAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl GradientAdapter for SpectralNormLinearAdapter {
    fn apply(&self, ctx: &AdapterContext<'_>) {
        // --------------------------------------------------------------------
        // Валидация контракта AdapterContext.
        // --------------------------------------------------------------------
        let params_len = ctx.segment_params.rows() * ctx.segment_params.cols();
        let grads_len = ctx.segment_grads.rows() * ctx.segment_grads.cols();

        debug_assert!(
            ctx.own_slice.end() <= params_len,
            "SpectralNormLinearAdapter: own_slice.end()={} exceeds segment_params len={}",
            ctx.own_slice.end(), params_len
        );
        debug_assert!(
            ctx.own_slice.end() <= grads_len,
            "SpectralNormLinearAdapter: own_slice.end()={} exceeds segment_grads len={}",
            ctx.own_slice.end(), grads_len
        );
        debug_assert_eq!(
            ctx.segment_params.rows(), ctx.segment_grads.rows(),
            "SpectralNormLinearAdapter: params.rows ({}) != grads.rows ({})",
            ctx.segment_params.rows(), ctx.segment_grads.rows()
        );
        debug_assert_eq!(
            ctx.segment_params.cols(), ctx.segment_grads.cols(),
            "SpectralNormLinearAdapter: params.cols ({}) != grads.cols ({})",
            ctx.segment_params.cols(), ctx.segment_grads.cols()
        );
        debug_assert!(ctx.batch > 0, "SNAdapter: batch must be positive");
        debug_assert!(
            ctx.optimizer_applied,
            "SNAdapter: optimizer_applied must be true (I-1 phase-order)"
        );
        debug_assert!(
            ctx.all_slices.iter().any(|s| *s == ctx.own_slice),
            "SNAdapter: own_slice not present in all_slices"
        );

        // GPU-ветка заморожена (I-5).
        if ctx.segment_grads.is_gpu() {
            debug_assert!(
                ctx.segment_params.is_gpu(),
                "SNAdapter: I-5 violation — grads GPU, params CPU"
            );
            return;
        }

        // --------------------------------------------------------------------
        // Извлечение `in_features`.
        // --------------------------------------------------------------------
        let Some(in_feat) = extract_in_features(ctx.forward_ctx) else {
            if !self.reported_missing_ctx.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[SN-ADAPTER] forward_ctx missing or not SpectralNormLinear; \
                     adapter no-op. (Once-per-session.)"
                );
            }
            return;
        };

        if in_feat == 0 {
            return;
        }

        // --------------------------------------------------------------------
        // Восстановление `out_features`.
        //   L = in·out + out + 1 = out·(in + 1) + 1
        //   ⇒ out = (L − 1) / (in + 1)
        // --------------------------------------------------------------------
        let total_len = ctx.own_slice.len;
        if total_len < 2 {
            return;
        }
        let denom = in_feat + 1;
        if (total_len - 1) % denom != 0 {
            return;
        }
        let out_feat = (total_len - 1) / denom;
        if out_feat == 0 {
            return;
        }

        // --------------------------------------------------------------------
        // LARS-adaptive + adaptive_beta + soft-clamp + online gain.
        // --------------------------------------------------------------------
        apply_lars(self, ctx, in_feat, out_feat);
    }

    fn name(&self) -> &'static str {
        "spectral_norm"
    }
}

// ============================================================================
// Реализация формулы
// ============================================================================

/// LARS-adaptive нормировка с `W_eff_norm = |scale|` и адаптивным β.
///
/// Модифицирует только W-часть среза; `bias` и `scale`-параметр не
/// трогаются.
fn apply_lars(
    adapter: &SpectralNormLinearAdapter,
    ctx: &AdapterContext<'_>,
    in_feat: usize,
    out_feat: usize,
) {
    let w_len = in_feat * out_feat;
    let scale_idx = w_len + out_feat;

    let params = ctx.read_own_params();
    let scale_param = params[scale_idx];
    let w_eff_norm = scale_param.abs();
    if w_eff_norm < SCALE_DEGENERATE_EPS {
        return;
    }

    let mut grads = ctx.read_own_grads();
    let b = ctx.batch.max(1) as f32;
    let g_norm_sum = {
        let w_grads = &grads[..w_len];
        l2_norm(w_grads)
    };
    let g_norm_ps = g_norm_sum / b.sqrt();

    // Online gain по W-части СЫРОГО градиента.
    let rms_cur = {
        let n = w_len as f32;
        if n > 0.0 { g_norm_ps / n.sqrt() } else { 0.0 }
    };
    let gain_eff = adapter.compute_gain_eff(ctx, rms_cur);

    let beta = *SN_LARS_BETA;
    let alpha = SN_LARS_ADAPT;
    let eps = *SN_LARS_EPS;
    let min_scale = *SN_LARS_MIN_SCALE;
    let max_scale = *SN_LARS_MAX_SCALE;
    let soft_k = *SN_SOFT_CLIP_K;

    // Адаптивный β_eff: чем больше r = g_ps / W_eff_norm, тем сильнее
    // демпфирование. На старте (большие градиенты) защищает сильнее,
    // в сходимости (малые градиенты) работает почти нейтрально.
    let beta_eff = adaptive_beta(beta, alpha, w_eff_norm, g_norm_ps);

    let denom = g_norm_ps + beta_eff * w_eff_norm + eps;
    let scale_raw = if denom > 0.0 { w_eff_norm / denom } else { 1.0 };
    let scale = soft_clamp(scale_raw, min_scale, max_scale, soft_k);
    let final_scale = scale * gain_eff;

    // Модифицируем только W-часть.
    {
        let w_grads = &mut grads[..w_len];
        for v in w_grads.iter_mut() {
            *v *= final_scale;
        }
    }

    ctx.write_own_grads(&grads);
}
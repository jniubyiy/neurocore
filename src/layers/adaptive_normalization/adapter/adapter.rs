// src/layers/adaptive_normalization/adapter/adapter.rs
//!
//! Градиентный адаптер слоя `AdaptiveNormalization` (MIGRATION_PLAN.md §7).
//!
//! # Архитектура параметров (7 групп × f)
//!
//! ```text
//!   ln_gamma [0f..1f)  ln_beta [1f..2f)  rms_gamma [2f..3f)
//!   bn_gamma [3f..4f)  bn_beta [4f..5f)  logits_ln [5f..6f)  logits_rms [6f..7f)
//! ```
//!
//! # Формула (per-group LARS + online gain + logits shielding)
//!
//! Единственный режим — per-group LARS с адаптивным damping, online gain (EMA)
//! и защитой logits-групп. Выбран после тестирования всех 9 вариантов
//! (см. MIGRATION_PLAN.md §7) как дающий лучшее соотношение max_jump / best_loss:
//!
//! ```text
//!   sqrt_b     = √B                           (batch-инвариентность)
//!   ‖W_g‖      = L2-норма параметров группы g
//!   ‖∇W_g‖_ps  = ‖∇W_g‖ / sqrt_b             (per-sample оценка)
//!
//!   r_g              = ‖∇W_g‖_ps / (‖W_g‖ + ε)
//!   β_eff            = β_group · (1 + α · r_g)   ← адаптивный damping
//!   scale_raw        = ‖W_g‖ / (‖∇W_g‖_ps + β_eff · ‖W_g‖ + ε)
//!   scale            = soft_clamp(scale_raw, MIN, MAX_group, k)
//!   ∇W_g            *= scale · gain_eff
//! ```
//!
//! Для logits-групп (индексы 5, 6) используются более консервативные
//! параметры (β_logits, MAX_logits), чтобы не искажать выходы модели.
//!
//! # Online gain (EMA-based)
//!
//! Аналогично адаптеру Linear: EMA of per-sample RMS по всему срезу.
//!
//! ```text
//!   rms_cur  = ‖∇W‖_sum / (√B · √n_total)
//!   gain_raw = ema_rms / (rms_cur + ε)
//!   gain_eff = clamp(gain_raw, GAIN_MIN, GAIN_MAX)
//!   ema_new  = β_ema · ema_rms + (1 − β_ema) · rms_cur
//! ```
//!
//! # Batch-инвариантность
//!
//! `‖g_per_sample‖ = ‖g_sum‖ / √B` даёт `loss(M шагов, B=1) == loss(1 шаг, B=M)`.
//!
//! # Soft-clamp вместо жёсткого clip
//!
//! Log-симметричная функция с асимптотами `MIN` и `MAX`. При `k = 3` значения
//! `x ∈ [1, 2]` искажаются менее чем на 0.3%, а `x ≫ MAX` сжимаются к `MAX`.
//!
//! # Инварианты (MIGRATION_PLAN.md §2)
//!
//!   * I-1: `optimizer_applied == true`;
//!   * I-2: Loss отдаёт сыной Sum-градиент;
//!   * I-3: адаптер трогает только `grad_params`;
//!   * I-4: адаптер в папке слоя;
//!   * I-5: CPU↔CPU / GPU↔GPU (GPU-ветка заморожена);
//!   * I-8: `&self`, состояние — внутри `Mutex`;
//!   * I-10: `own_slice` — легитимный член `all_slices`.

use std::collections::HashMap;
use std::sync::Mutex;

use once_cell::sync::Lazy;

use crate::compute_manager::core::dynamic_context::DynamicContext;
use crate::layers::adapter::{AdapterContext, GradientAdapter};
use crate::layers::buffered_context::BufferedContext;

#[cfg(test)]
use crate::compute_manager::core::device_spec::DeviceSpec;
#[cfg(test)]
use crate::compute_manager::operators_v2::memory_v2::buffer::MatrixBufferHandle;
#[cfg(test)]
use crate::compute_manager::operators_v2::memory_v2::executor::MemoryExecutor;
#[cfg(test)]
use crate::compute_manager::operators_v2::memory_v2::policy::BufferPriority;
#[cfg(test)]
use crate::compute_manager::operators_v2::memory_v2::types::MemoryDeviceKind;
#[cfg(test)]
use crate::model_plan::param_store::ParamSlice;
#[cfg(test)]
use std::sync::{Arc, RwLock};

// ============================================================================
// Constants
// ============================================================================

/// Количество групп параметров в AdaptiveNormalization.
const NUM_GROUPS: usize = 7;

/// Индекс группы logits_ln внутри параметров слоя.
const LOGITS_LN: usize = 5;

/// Индекс группы logits_rms внутри параметров слоя.
const LOGITS_RMS: usize = 6;

// ============================================================================
// Env-параметры (тюн-параметры, не режимы)
// ============================================================================

/// Базовый β в знаменателе LARS-adaptive scale (для не-logits групп).
static ADNORM_BETA: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_ADNORM_BETA")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.3)
});

/// α — коэффициент адаптивности β: `β_eff = β · (1 + α · r)`.
static ADNORM_ALPHA: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_ADNORM_ALPHA")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0)
});

/// Нижняя асимптота per-group scale.
static ADNORM_MIN_SCALE: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_ADNORM_MIN_SCALE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.01)
});

/// Верхняя асимптота per-group scale для не-logits групп.
static ADNORM_MAX_SCALE: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_ADNORM_MAX_SCALE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(3.0)
});

/// Верхняя асимптота per-group scale для logits-групп (меньше — защита).
static ADNORM_LOGITS_MAX_SCALE: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_ADNORM_LOGITS_MAX_SCALE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.5)
});

/// Базовый β для logits-групп (меньше — защита).
static ADNORM_LOGITS_BETA: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_ADNORM_LOGITS_BETA")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.5)
});

/// ε в знаменателе (защита от деления на ноль).
static ADNORM_EPS: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_ADNORM_EPS")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1e-12)
});

/// Жёсткость soft-clamp. Default = 3.0 (∞ → жёсткий clamp).
static ADNORM_SOFT_CLIP_K: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_ADNORM_SOFT_CLIP_K")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(3.0)
});

/// Нижний clip online gain. Default = 0.5.
static ADNORM_GAIN_MIN: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_ADNORM_GAIN_MIN")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.5)
});

/// Верхний clip online gain. Default = 2.5.
static ADNORM_GAIN_MAX: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_ADNORM_GAIN_MAX")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(2.5)
});

/// β EMA для online gain. Default = 0.9 (half-life ≈ 7 шагов).
static ADNORM_GAIN_EMA_BETA: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_ADNORM_GAIN_EMA_BETA")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.9)
});

// ============================================================================
// Soft-clamp helpers
// ============================================================================

/// Мягкий log-симметричный clamp: `x ∈ (0, ∞)` → значение в `(lo, hi)`.
#[inline]
fn soft_clamp(x: f32, lo: f32, hi: f32, k: f32) -> f32 {
    if x >= 1.0 {
        1.0 + soft_cap_positive(x - 1.0, hi - 1.0, k)
    } else {
        1.0 - soft_cap_positive(1.0 - x, 1.0 - lo, k)
    }
}

#[inline]
fn soft_cap_positive(y: f32, cap: f32, k: f32) -> f32 {
    if y <= 0.0 || cap <= 0.0 { return 0.0; }
    let ratio = y / cap;
    y / (1.0 + ratio.powf(k)).powf(1.0 / k)
}

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

#[inline]
fn group_bounds(g: usize, f: usize) -> (usize, usize) {
    (g * f, f)
}

#[inline]
fn is_logits_group(g: usize) -> bool {
    g == LOGITS_LN || g == LOGITS_RMS
}

/// Batch-инвариантная оценка per-sample RMS по всему срезу.
#[inline]
fn compute_rms_cur(grads: &[f32], sqrt_b: f32, n_total: f32) -> f32 {
    if n_total > 0.0 {
        l2_norm(grads) / (sqrt_b * n_total.sqrt())
    } else {
        0.0
    }
}

fn extract_in_features(forward_ctx: Option<&DynamicContext>) -> Option<usize> {
    let fctx = forward_ctx?;
    match fctx {
        DynamicContext::Buffered(BufferedContext::AdaptiveNormalization { input }) => {
            let cols = input.cols();
            if cols > 0 { Some(cols) } else { None }
        }
        _ => None,
    }
}

fn prepare_apply(ctx: &AdapterContext<'_>) -> Option<(usize, Vec<f32>, Vec<f32>)> {
    let f = extract_in_features(ctx.forward_ctx)?;
    if f == 0 { return None; }
    let slice_len = ctx.own_slice.len;
    if slice_len != NUM_GROUPS * f { return None; }
    Some((f, ctx.read_own_params(), ctx.read_own_grads()))
}

// ============================================================================
// AdaptiveNormAdapter
// ============================================================================

/// Градиентный адаптер слоя `AdaptiveNormalization`.
///
/// Работает через `&self`. Персистентное состояние — `gain_state`
/// (EMA online gain) — под `Mutex`.
pub struct AdaptiveNormAdapter {
    /// EMA per-sample per-param RMS — ключ `(buffer_idx, slice.start)`.
    gain_state: Mutex<HashMap<(usize, usize), f32>>,
}

impl AdaptiveNormAdapter {
    pub fn new() -> Self {
        Self {
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
                state.insert(key, rms_cur);
                1.0
            }
            Some(ema) => {
                let eps = *ADNORM_EPS;
                let gain_raw = if rms_cur > 1e-30 {
                    ema / (rms_cur + eps)
                } else {
                    // Нулевой градиент — не усиливаем шум.
                    1.0
                };
                let gain_eff = gain_raw.clamp(*ADNORM_GAIN_MIN, *ADNORM_GAIN_MAX);

                let beta = *ADNORM_GAIN_EMA_BETA;
                let new_ema = beta * ema + (1.0 - beta) * rms_cur;
                state.insert(key, new_ema);

                gain_eff
            }
        }
    }
}

impl Default for AdaptiveNormAdapter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// impl GradientAdapter
// ============================================================================

impl GradientAdapter for AdaptiveNormAdapter {
    fn apply(&self, ctx: &AdapterContext<'_>) {
        let Some((f, params, mut grads)) = prepare_apply(ctx) else { return };

        let sqrt_b = ctx.batch.max(1) as f32;
        let n_total = (NUM_GROUPS * f) as f32;
        let rms_cur = compute_rms_cur(&grads, sqrt_b, n_total);
        let gain_eff = self.compute_gain_eff(ctx, rms_cur);

        let eps = *ADNORM_EPS;
        let k = *ADNORM_SOFT_CLIP_K;
        let min_s = *ADNORM_MIN_SCALE;
        let beta_base = *ADNORM_BETA;
        let alpha = *ADNORM_ALPHA;
        let max_s = *ADNORM_MAX_SCALE;
        let logit_max_s = *ADNORM_LOGITS_MAX_SCALE;
        let logit_beta = *ADNORM_LOGITS_BETA;

        for g in 0..NUM_GROUPS {
            let (s, len) = group_bounds(g, f);
            if len == 0 { continue; }

            // Logits-группы (5, 6) — консервативные параметры (shield).
            let (use_max, use_beta) = if is_logits_group(g) {
                (logit_max_s, logit_beta)
            } else {
                (max_s, beta_base)
            };

            let wg = &params[s..s + len];
            let gg = &mut grads[s..s + len];

            let wn = l2_norm(wg);
            let gp = l2_norm(gg) / sqrt_b;
            let be = adaptive_beta(use_beta, alpha, wn, gp);
            let d = gp + be * wn + eps;
            let sr = if d > 0.0 { wn / d } else { 1.0 };
            let sc = soft_clamp(sr, min_s, use_max, k);
            let fs = sc * gain_eff;

            for v in gg.iter_mut() {
                *v *= fs;
            }
        }

        ctx.write_own_grads(&grads);
    }

    #[inline]
    fn name(&self) -> &'static str {
        "adaptive_normalization_adapter"
    }

    #[inline]
    fn has_state(&self) -> bool {
        false
    }
}

// ============================================================================
// Helpers для тестов
// ============================================================================

#[cfg(test)]
fn make_executor() -> Arc<RwLock<MemoryExecutor>> {
    let mem = Arc::new(RwLock::new(MemoryExecutor::new()));
    mem.write().unwrap()
        .register_compute_device(DeviceSpec::cpu(0, 4096, 1), None);
    mem.write().unwrap().set_self_arc(mem.clone());
    mem
}

#[cfg(test)]
fn make_matrix(
    mem: &Arc<RwLock<MemoryExecutor>>,
    rows: usize,
    cols: usize,
) -> MatrixBufferHandle {
    mem.write().unwrap()
        .acquire_matrix_handle(
            rows, cols, MemoryDeviceKind::HostRam, BufferPriority::Medium,
        )
        .expect("acquire handle")
}



// ============================================================================
// Тесты
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const F: usize = 8;

    #[test]
    fn adapter_name_and_state() {
        let a = AdaptiveNormAdapter::new();
        assert_eq!(a.name(), "adaptive_normalization_adapter");
        assert!(!a.has_state());
    }

    #[test]
    fn apply_modifies_grads() {
        let f = 8;
        let total = NUM_GROUPS * f;
        let params: Vec<f32> = (0..total).map(|i| i as f32 * 0.1 + 0.1).collect();
        let grads: Vec<f32> = (0..total).map(|i| (i as f32 - 14.0) * 0.01).collect();

        let mem = make_executor();
        let ph = make_matrix(&mem, total, 1);
        let gh = make_matrix(&mem, total, 1);
        let ih = make_matrix(&mem, 4, f);
        ph.write_range(0, &params);
        gh.write_range(0, &grads);

        let fc = DynamicContext::Buffered(
            BufferedContext::AdaptiveNormalization { input: ih },
        );
        let own_slice = ParamSlice::new(0, 0, total);
        let all_slices = vec![own_slice];
        let ctx = AdapterContext {
            segment_params: &ph,
            segment_grads: &gh,
            own_slice,
            all_slices: &all_slices,
            batch: 4,
            optimizer_applied: true,
            own_state_slice: None,
            adapter_store: None,
            forward_ctx: Some(&fc),
        };

        let adj = AdaptiveNormAdapter::new();
        adj.apply(&ctx);

        let after = gh.read_range(0, total);
        assert_eq!(after.len(), total);
        assert!(
            after.iter().all(|v| v.is_finite()),
            "All gradients must be finite after adapter"
        );
        assert_ne!(after, grads, "Gradients should be modified");
    }

    #[test]
    fn apply_zero_grads_remain_zero() {
        let total = NUM_GROUPS * F;
        let params: Vec<f32> = (0..total).map(|i| i as f32 + 1.0).collect();
        let grads: Vec<f32> = vec![0.0; total];

        let mem = make_executor();
        let ph = make_matrix(&mem, total, 1);
        let gh = make_matrix(&mem, total, 1);
        let ih = make_matrix(&mem, 4, F);
        ph.write_range(0, &params);
        gh.write_range(0, &grads);

        let fc = DynamicContext::Buffered(
            BufferedContext::AdaptiveNormalization { input: ih },
        );
        let own_slice = ParamSlice::new(0, 0, total);
        let all_slices = vec![own_slice];
        let ctx = AdapterContext {
            segment_params: &ph,
            segment_grads: &gh,
            own_slice,
            all_slices: &all_slices,
            batch: 4,
            optimizer_applied: true,
            own_state_slice: None,
            adapter_store: None,
            forward_ctx: Some(&fc),
        };

        let adj = AdaptiveNormAdapter::new();
        adj.apply(&ctx);

        let after = gh.read_range(0, total);
        assert!(
            after.iter().all(|&v| v.abs() < 1e-10),
            "Zero grads should remain ~zero"
        );
    }

    #[test]
    fn prepare_apply_returns_none_without_forward_ctx() {
        let total = NUM_GROUPS * F;
        let params: Vec<f32> = (0..total).map(|i| i as f32 + 1.0).collect();
        let grads: Vec<f32> = (0..total).map(|i| (i as f32 - 14.0) * 0.01).collect();

        let mem = make_executor();
        let ph = make_matrix(&mem, total, 1);
        let gh = make_matrix(&mem, total, 1);
        ph.write_range(0, &params);
        gh.write_range(0, &grads);

        let all_slices = vec![ParamSlice::new(0, 0, total)];
        let ctx = AdapterContext {
            segment_params: &ph,
            segment_grads: &gh,
            own_slice: ParamSlice::new(0, 0, total),
            all_slices: &all_slices,
            batch: 4,
            optimizer_applied: true,
            own_state_slice: None,
            adapter_store: None,
            forward_ctx: None,
        };

        let adj = AdaptiveNormAdapter::new();
        adj.apply(&ctx); // no panic, no-op
        let after = gh.read_range(0, total);
        assert_eq!(after, grads, "Without forward_ctx, no mutation");
    }

    #[test]
    fn prepare_apply_returns_some_with_forward_ctx() {
        let f = 8;
        let total = NUM_GROUPS * f;
        let params: Vec<f32> = (0..total).map(|i| i as f32 + 1.0).collect();
        let grads: Vec<f32> = (0..total).map(|i| (i as f32 - 14.0) * 0.01).collect();

        let mem = make_executor();
        let ph = make_matrix(&mem, total, 1);
        let gh = make_matrix(&mem, total, 1);
        let ih = make_matrix(&mem, 4, f);
        ph.write_range(0, &params);
        gh.write_range(0, &grads);

        let fc = DynamicContext::Buffered(
            BufferedContext::AdaptiveNormalization { input: ih },
        );
        let own_slice = ParamSlice::new(0, 0, total);
        let all_slices = vec![own_slice];
        let ctx = AdapterContext {
            segment_params: &ph,
            segment_grads: &gh,
            own_slice,
            all_slices: &all_slices,
            batch: 4,
            optimizer_applied: true,
            own_state_slice: None,
            adapter_store: None,
            forward_ctx: Some(&fc),
        };

        let adj = AdaptiveNormAdapter::new();
        adj.apply(&ctx);
        let after = gh.read_range(0, total);
        assert_ne!(after, grads, "Gradients should be modified");
    }
}

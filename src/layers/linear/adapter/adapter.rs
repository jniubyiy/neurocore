// src/layers/linear/adapter/adapter.rs

//! Градиентный адаптер слоя `Linear` (MIGRATION_PLAN.md §7, финальная версия).
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
//!   scale              = clamp(scale_raw, min_scale, max_scale)
//!   ∇W_c              *= scale
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
//! # Адаптивность
//!
//! `β_eff = β · (1 + α · r)`:
//!   * `r → 0` (малый градиент) → `β_eff ≈ β` → сильный boost;
//!   * `r ≫ 1` (большой градиент) → `β_eff ≫ β` → damping.
//!
//! # Параметры (зафиксированы, эмпирически подтверждены)
//!
//! | Параметр | Значение |
//! |----------|----------|
//! | β (базовый damping) | 0.3 |
//! | α (адаптивность)    | 1.0 |
//! | min_scale           | 0.01 |
//! | max_scale           | 3.0 |
//! | ε                   | 1e-12 |
//!
//! # GPU
//!
//! Заморожена (I-5). Если `ctx.segment_grads.is_gpu()` — адаптер
//! пропускает модификацию.
//!
//! # Инварианты (MIGRATION_PLAN.md §2)
//!
//!   * I-1: `optimizer_applied == true`.
//!   * I-2: Loss отдаёт сырой Sum-градиент.
//!   * I-3: адаптер трогает только `grad_params`.
//!   * I-4: адаптер в папке слоя.
//!   * I-5: GPU↔GPU / CPU↔CPU.
//!   * I-8: `&self`, без внутреннего состояния.
//!   * I-10: `own_slice` — легитимный член `all_slices`.

use crate::compute_manager::core::dynamic_context::DynamicContext;
use crate::layers::adapter::{AdapterContext, GradientAdapter};
use crate::layers::buffered_context::BufferedContext;

// ============================================================================
// Константы формулы
// ============================================================================

/// Базовый β в знаменателе LARS-adaptive scale.
const LARS_BETA: f32 = 0.3;

/// α — коэффициент адаптивности β: `β_eff = β · (1 + α · r)`.
const LARS_ADAPT: f32 = 1.0;

/// Нижний clip `scale`.
const LARS_MIN_SCALE: f32 = 0.01;

/// Верхний clip `scale`.
const LARS_MAX_SCALE: f32 = 3.0;

/// ε в знаменателе (защита от деления на ноль).
const LARS_EPS: f32 = 1e-12;

// ============================================================================
// LinearAdapter
// ============================================================================

/// Градиентный адаптер слоя `Linear`.
///
/// Stateless. Работает через `&self`.
pub struct LinearAdapter;

impl LinearAdapter {
    pub fn new() -> Self {
        Self
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
        // 7 групп debug_assert из пилотной версии сохранены.
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

        debug_assert!(
            ctx.forward_ctx.is_some(),
            "LinearAdapter: forward_ctx is missing. \
             Likely cause: forward_cache was cleared too early \
             (MIGRATION_PLAN.md §7, Фаза 4: cache must survive until apply_update)."
        );

        debug_assert!(
            ctx.adapter_store.is_some(),
            "LinearAdapter: adapter_store is missing. \
             Likely cause: GraphV2::build did not initialize AdapterStateStore \
             or AdapterContext was constructed incorrectly."
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
        // Режим per_row. Если из forward_ctx не удаётся извлечь in_features
        // или slice_len не согласован — тихо откатываемся на per_tensor.
        // ====================================================================
        match extract_in_features(ctx.forward_ctx) {
            Some(in_features)
                if in_features > 0
                    && ctx.own_slice.len % (in_features + 1) == 0 =>
            {
                apply_per_row(ctx, in_features);
            }
            _ => {
                apply_per_tensor(ctx);
            }
        }
    }

    fn name(&self) -> &'static str {
        "linear"
    }
}

// ============================================================================
// Реализация формул
// ============================================================================

/// Адаптивный β_eff = β · (1 + α · r), где r = ‖g_ps‖ / ‖W‖.
#[inline]
fn adaptive_beta(beta: f32, alpha: f32, w_norm: f32, g_norm_ps: f32) -> f32 {
    if alpha == 0.0 || w_norm < 1e-30 {
        return beta;
    }
    let r = g_norm_ps / w_norm;
    beta * (1.0 + alpha * r)
}

/// Per-tensor LARS. Fallback, если per_row неприменим.
fn apply_per_tensor(ctx: &AdapterContext<'_>) {
    let w = ctx.read_own_params();
    let mut g = ctx.read_own_grads();

    let w_norm = l2_norm(&w);
    let g_norm_sum = l2_norm(&g);

    let b = ctx.batch.max(1) as f32;
    let g_norm_per_sample = g_norm_sum / b.sqrt();

    let beta_eff = adaptive_beta(LARS_BETA, LARS_ADAPT, w_norm, g_norm_per_sample);

    let denom = g_norm_per_sample + beta_eff * w_norm + LARS_EPS;
    let scale_raw = if denom > 0.0 { w_norm / denom } else { 1.0 };
    let scale = scale_raw.clamp(LARS_MIN_SCALE, LARS_MAX_SCALE);

    for v in g.iter_mut() {
        *v *= scale;
    }
    ctx.write_own_grads(&g);
}

/// Per-row LARS. Свой scale на каждую строку W.
fn apply_per_row(ctx: &AdapterContext<'_>, in_features: usize) {
    let slice_len = ctx.own_slice.len;
    let out_features = slice_len / (in_features + 1);

    if out_features == 0 {
        return;
    }

    let w = ctx.read_own_params();
    let mut g = ctx.read_own_grads();

    let b = ctx.batch.max(1) as f32;
    let sqrt_b = b.sqrt();

    for c in 0..out_features {
        let start = c * in_features;
        let end = start + in_features;

        let w_c = &w[start..end];
        let g_c = &mut g[start..end];

        let w_norm = l2_norm(w_c);
        let g_norm_sum = l2_norm(g_c);
        let g_norm_per_sample = g_norm_sum / sqrt_b;

        let beta_eff = adaptive_beta(LARS_BETA, LARS_ADAPT, w_norm, g_norm_per_sample);

        let denom = g_norm_per_sample + beta_eff * w_norm + LARS_EPS;
        let scale_raw = if denom > 0.0 { w_norm / denom } else { 1.0 };
        let scale = scale_raw.clamp(LARS_MIN_SCALE, LARS_MAX_SCALE);

        for v in g_c.iter_mut() {
            *v *= scale;
        }
    }

    ctx.write_own_grads(&g);
}

// ============================================================================
// Вспомогательные функции
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
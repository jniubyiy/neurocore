//! Градиентный адаптер слоя `SpectrallyNormalizedLinear`.
//!
//! # Формула (LARS-adaptive, единственный режим)
//!
//! ```text
//!   W_eff_norm = |scale|                          (эффективная спектральная норма W_sn)
//!   g_norm_sum = ‖∇W‖₂                            (L2 по W-части среза)
//!   g_norm_ps  = g_norm_sum / √B                  (batch-инвариантная оценка)
//!   denom      = g_norm_ps + β·W_eff_norm + ε
//!   scale_raw  = W_eff_norm / denom
//!   scale      = clamp(scale_raw, min_scale, max_scale)
//!   ∇W_sum    *= scale
//! ```
//!
//! # Что модифицируется
//!
//! Только **W-часть** среза параметров. `bias` и `scale`-параметр
//! остаются как есть — у них свои градиенты, не связанные с нормировкой W.
//! Это фактически даёт per-parameter-group learning rate: W движется
//! с эффективным шагом `lr · scale`, bias и scale — с чистым `lr`.
//!
//! # Экспериментальные результаты (spectral_norm_linear_example)
//!
//! | Конфигурация              | best_loss | best_epoch | max jump |
//! |---------------------------|-----------|------------|----------|
//! | без адаптера, lr=0.01     | 8.4e-5    | 274        | +543.66% |
//! | **LARS adapter, lr=0.01** | **1.0e-6**| **261**    | **+9.54%** |
//! | без адаптера, lr=0.015    | 1.67e-4   | 289        | +65.91%  |
//!
//! Адаптер даёт **84×** лучший loss и **57×** меньший скачок по сравнению
//! с baseline. Простой boost lr=0.015 — **хуже** baseline; следовательно,
//! эффект не сводится к умножению lr на константу. Причина —
//! модификация только W-части (per-group lr).
//!
//! # Batch-инвариантность
//!
//! Оценка `‖∇W‖_sum / √B` не зависит от `B` в шумовом режиме (E[g]=0).
//!
//! # Тюн-параметры (env)
//!
//! | Env | Default | Смысл |
//! |-----|---------|-------|
//! | `NEUROCORE_SN_LARS_BETA`      | `0.5`  | β в знаменателе |
//! | `NEUROCORE_SN_LARS_MIN_SCALE` | `0.01` | Нижний clip |
//! | `NEUROCORE_SN_LARS_MAX_SCALE` | `1.5`  | Верхний clip |
//! | `NEUROCORE_SN_LARS_EPS`       | `1e-12`| ε в знаменателе |
//!
//! # Инварианты (MIGRATION_PLAN.md §2)
//!
//!   * I-3: адаптер трогает только `grad_params`, не `grad_input`;
//!   * I-4: адаптер в папке слоя;
//!   * I-5: CPU↔CPU / GPU↔GPU (GPU-ветка заморожена);
//!   * I-8: `&self`, состояние — `AtomicBool`;
//!   * I-10: `own_slice` — легитимный член `all_slices`.

use std::sync::atomic::{AtomicBool, Ordering};

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

/// Нижний clip `scale`. Default = 0.01.
static SN_LARS_MIN_SCALE: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_SN_LARS_MIN_SCALE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.01)
});

/// Верхний clip `scale`. Default = 1.5.
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

/// Порог, ниже которого `scale` считается вырожденным; в этом случае
/// forward y = b, dL/dW_sn = 0, модификация пропускается.
const SCALE_DEGENERATE_EPS: f32 = 1e-30;

// ============================================================================
// Адаптер
// ============================================================================

/// Градиентный адаптер `SpectrallyNormalizedLinear`.
///
/// Полное описание формулы и экспериментальных результатов —
/// в документации модуля `super`.
pub struct SpectralNormLinearAdapter {
    /// Одноразовый флаг: предупредить в stderr, если forward-контекст
    /// не содержит `BufferedContext::SpectralNormLinear`. Повторные
    /// предупреждения подавляются.
    reported_missing_ctx: AtomicBool,
}

impl SpectralNormLinearAdapter {
    /// Создаёт новый адаптер.
    pub fn new() -> Self {
        Self {
            reported_missing_ctx: AtomicBool::new(false),
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
        // Валидация контракта AdapterContext (MIGRATION_PLAN.md §2).
        // --------------------------------------------------------------------
        let params_len = ctx.segment_params.rows() * ctx.segment_params.cols();
        let grads_len = ctx.segment_grads.rows() * ctx.segment_grads.cols();

        debug_assert!(
            ctx.own_slice.end() <= params_len,
            "SpectralNormLinearAdapter: own_slice.end()={} exceeds segment_params len={} \
             (buffer_idx={}, start={}, len={})",
            ctx.own_slice.end(),
            params_len,
            ctx.own_slice.buffer_idx(),
            ctx.own_slice.start,
            ctx.own_slice.len
        );
        debug_assert!(
            ctx.own_slice.end() <= grads_len,
            "SpectralNormLinearAdapter: own_slice.end()={} exceeds segment_grads len={} \
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
            "SpectralNormLinearAdapter: segment_params.rows ({}) != segment_grads.rows ({})",
            ctx.segment_params.rows(),
            ctx.segment_grads.rows()
        );
        debug_assert_eq!(
            ctx.segment_params.cols(),
            ctx.segment_grads.cols(),
            "SpectralNormLinearAdapter: segment_params.cols ({}) != segment_grads.cols ({})",
            ctx.segment_params.cols(),
            ctx.segment_grads.cols()
        );
        debug_assert!(
            ctx.batch > 0,
            "SpectralNormLinearAdapter: batch must be positive, got {}",
            ctx.batch
        );
        debug_assert!(
            ctx.optimizer_applied,
            "SpectralNormLinearAdapter: optimizer_applied must be true. \
             Phase-order violation (MIGRATION_PLAN.md §2, инвариант I-1): \
             adapter_pass must be called after optimizer_modify_grads."
        );
        debug_assert!(
            ctx.all_slices.iter().any(|s| *s == ctx.own_slice),
            "SpectralNormLinearAdapter: own_slice {:?} not present in all_slices",
            ctx.own_slice
        );

        // --------------------------------------------------------------------
        // GPU-ветка заморожена (симметрично LinearAdapter, I-5).
        // --------------------------------------------------------------------
        if ctx.segment_grads.is_gpu() {
            debug_assert!(
                ctx.segment_params.is_gpu(),
                "SpectralNormLinearAdapter: I-5 violation — grads on GPU but params on CPU"
            );
            return;
        }

        // --------------------------------------------------------------------
        // Извлечение `in_features` из forward-контекста.
        // --------------------------------------------------------------------
        let Some(in_feat) = extract_in_features(ctx.forward_ctx) else {
            if !self.reported_missing_ctx.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[SN-ADAPTER] forward_ctx missing or not SpectralNormLinear; \
                     adapter no-op. (Once-per-session message.)"
                );
            }
            return;
        };

        if in_feat == 0 {
            return;
        }

        // --------------------------------------------------------------------
        // Восстановление `out_features` из длины среза.
        //
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
        // LARS-adaptive нормировка градиента по W.
        // --------------------------------------------------------------------
        apply_lars(ctx, in_feat, out_feat);
    }

    fn name(&self) -> &'static str {
        "spectral_norm"
    }
}

// ============================================================================
// Реализация формулы
// ============================================================================

/// LARS-adaptive нормировка с `W_eff_norm = |scale|`.
///
/// Модифицирует только W-часть среза; `bias` и `scale`-параметр остаются
/// нетронутыми (индексы `[w_len .. L)`).
fn apply_lars(ctx: &AdapterContext<'_>, in_feat: usize, out_feat: usize) {
    let w_len = in_feat * out_feat;
    let scale_idx = w_len + out_feat;

    let params = ctx.read_own_params();
    let scale_param = params[scale_idx];
    let w_eff_norm = scale_param.abs();
    if w_eff_norm < SCALE_DEGENERATE_EPS {
        return;
    }

    // L2-норма ∇W и batch-инвариантная оценка per-sample нормы.
    let mut grads = ctx.read_own_grads();
    let g_norm_sum = {
        let w_grads = &grads[..w_len];
        l2_norm(w_grads)
    };
    let b = ctx.batch.max(1) as f32;
    let g_norm_ps = g_norm_sum / b.sqrt();

    let beta = *SN_LARS_BETA;
    let eps = *SN_LARS_EPS;
    let min_scale = *SN_LARS_MIN_SCALE;
    let max_scale = *SN_LARS_MAX_SCALE;

    let denom = g_norm_ps + beta * w_eff_norm + eps;
    let scale_raw = if denom > 0.0 { w_eff_norm / denom } else { 1.0 };
    let scale = scale_raw.clamp(min_scale, max_scale);

    // Модифицируем только W-часть; bias и scale-параметр не трогаем.
    {
        let w_grads = &mut grads[..w_len];
        for v in w_grads.iter_mut() {
            *v *= scale;
        }
    }

    ctx.write_own_grads(&grads);
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
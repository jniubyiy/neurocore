// src/layers/multi_resolution_kan_linear/adapter/adapter.rs

//! Градиентный адаптер слоя `MultiResolutionKANLinear`.
//!
//! # Задача
//!
//! Слой KAN имеет **шесть разнородных групп параметров** с разной
//! математикой и разным масштабом:
//!
//! ```text
//!   группа            длина (на слой)   роль в forward
//!   ----------------  ----------------  ------------------------------------
//!   bias              out               аддитивное смещение выхода
//!   mix_logits        2·in·out          логиты softmax-смеси coarse/fine
//!   mix_temp_raw      in·out            log-температура смеси (T = T_MIN+e^t)
//!   spline_coarse     6·in·out          коэффициенты грубой B-сплайн сетки
//!   spline_fine       11·in·out         коэффициенты точной B-сплайн сетки
//!   base_weight       in·out            вес SiLU-ветки
//! ```
//!
//! Сырой Sum-градиент (I-2) у групп имеет разный порядок величины и
//! разный рост по батчу (√B). Группы `mix_*` содержат множитель 1/T
//! (T→1e-3), что даёт всплески. Единый SGD-шаг обновляет группы с
//! сильно разной эффективной скоростью обучения.
//!
//! # Единственный режим: adaptive group RMS equalize
//!
//! Адаптер всегда работает в одном режиме и сам подстраивается под
//! ситуацию через три независимых механизма. Никаких env-переключателей
//! режимов нет.
//!
//! ## Формула
//!
//! ```text
//!   rms_g       = ‖G_g‖ / (√B · √n_g)
//!   target_rms  = ‖G‖   / (√B · √n_среза)
//!   r_g         = rms_g / (target_rms + ε)
//!   β_eff       = β · (1 + α · r_g)
//!   raw_scale_g = target_rms · (1 + β_eff) / (rms_g + β_eff · target_rms + ε)
//!   scale_g     = soft_clamp(raw_scale_g, MIN, MAX, K)
//!   gain_eff    = clamp(ema_rms / (rms_cur + ε), GAIN_MIN, GAIN_MAX)
//!   G_g        *= scale_g · gain_eff
//! ```
//!
//! `B` — размер батча, `n_g` — число параметров в группе.
//!
//! # Три механизма авто-подстройки
//!
//! ## 1. Adaptive β_rms (пространственный)
//!
//! `β_eff = β · (1 + α · r_g)`:
//!   * `r_g = 1` (норма) → `β_eff = 2β`; `raw_scale = 1` при
//!     `rms_g = target_rms` (независимо от β);
//!   * `r_g → 0` (слабый градиент, стагнация) → `β_eff ≈ β`; boost
//!     группы ограничен сверху значением `(1 + β)/β`;
//!   * `r_g ≫ 1` (сильный градиент, выброс) → `β_eff` растёт, что
//!     усиливает demote выбившейся группы.
//!
//! Это автоматически **уравнивает эффективные скорости обучения групп**,
//! не позволяя ни одной группе доминировать.
//!
//! ## 2. Soft-clamp (границы)
//!
//! `soft_clamp` — log-симметричная функция с асимптотами `MIN` и
//! `MAX`. При `k = 3` значения `x ∈ [1, 2]` искажаются менее чем на
//! 0.3%, а `x ≫ hi` асимптотически сжимается к `hi`. Убирает резкий
//! «отрез» жёсткого clamp, снижая `max_jump`.
//!
//! ## 3. Online gain (временной)
//!
//! Адаптер ведёт EMA per-sample per-param RMS среза:
//!
//! ```text
//!   ema_new = β_ema · ema_old + (1 − β_ema) · rms_cur
//!   gain    = clamp(ema_old / (rms_cur + ε), GAIN_MIN, GAIN_MAX)
//! ```
//!
//! Семантика:
//!   * градиент в норме → `gain ≈ 1.0` (нейтрально);
//!   * градиент систематически падает (середина / конец обучения) →
//!     `gain > 1` (ускорение);
//!   * градиент подскочил (выброс) → `gain < 1` (**демпфирование**).
//!
//! Первый вызов `apply` инициализирует EMA значением `rms_cur` и
//! возвращает `gain_eff = 1.0`.
//!
//! # Batch-инвариантность
//!
//! Все три механизма делят нормы на `√B` (или эквивалентно), поэтому
//! отношения и итоговый scale от `B` не зависят в шумовом режиме.
//!
//! # Тюн-параметры (env)
//!
//! | Env | Default | Смысл |
//! |-----|---------|-------|
//! | `NEUROCORE_KAN_BETA`          | `0.3`   | β в adaptive β_rms |
//! | `NEUROCORE_KAN_ALPHA`         | `1.0`   | α в adaptive β_rms |
//! | `NEUROCORE_KAN_MIN_SCALE`     | `0.1`   | нижняя асимптота soft-clamp |
//! | `NEUROCORE_KAN_MAX_SCALE`     | `6.0`   | верхняя асимптота soft-clamp |
//! | `NEUROCORE_KAN_SOFT_CLIP_K`   | `3.0`   | жёсткость soft-clamp |
//! | `NEUROCORE_KAN_GAIN_MIN`      | `0.5`   | нижний clip online gain |
//! | `NEUROCORE_KAN_GAIN_MAX`      | `2.5`   | верхний clip online gain |
//! | `NEUROCORE_KAN_GAIN_EMA_BETA` | `0.9`   | β EMA online gain |
//! | `NEUROCORE_KAN_EPS`           | `1e-12` | ε в знаменателях |
//!
//! Для отключения адаптера целиком используйте
//! `NEUROCORE_DISABLE_ADAPTERS=1` — глобальный рубильник графа
//! (см. `GraphV2::adapter_pass`).
//!
//! # Что НЕ модифицируется
//!
//! Адаптер трогает весь `own_slice` (все 6 групп). Каждая группа
//! нормируется своим scale, к которому применяется общий online gain.
//!
//! # Инварианты (MIGRATION_PLAN.md §2)
//!
//!   * I-1: `optimizer_applied == true` (assert при `apply`);
//!   * I-2: Loss отдаёт сырой Sum-градиент;
//!   * I-3: адаптер трогает только `grad_params`;
//!   * I-4: адаптер в папке слоя;
//!   * I-5: CPU↔CPU / GPU↔GPU (GPU-ветка заморожена, пропускается);
//!   * I-8: `&self`, состояние — внутри `Mutex`;
//!   * I-10: `own_slice` — легитимный член `all_slices`.

use std::collections::HashMap;
use std::sync::Mutex;

use once_cell::sync::Lazy;

use crate::compute_manager::core::dynamic_context::DynamicContext;
use crate::layers::adapter::{AdapterContext, GradientAdapter};
use crate::layers::buffered_context::BufferedContext;

// ============================================================================
// Env-параметры
// ============================================================================

/// β в adaptive β_rms. Default = 0.3.
static KAN_BETA: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_KAN_BETA")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.3)
});

/// α в adaptive β_rms: `β_eff = β · (1 + α · r_g)`. Default = 1.0.
static KAN_ALPHA: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_KAN_ALPHA")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0)
});

/// Нижняя асимптота soft-clamp. Default = 0.1.
static KAN_MIN_SCALE: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_KAN_MIN_SCALE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.1)
});

/// Верхняя асимптота soft-clamp. Default = 6.0.
static KAN_MAX_SCALE: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_KAN_MAX_SCALE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(6.0)
});

/// Жёсткость soft-clamp. Default = 3.0.
///
/// Большие значения ближе к жёсткому `clamp`.
/// Малые — мягче сжатие, но и более выраженное искажение внутри
/// рабочего диапазона.
static KAN_SOFT_CLIP_K: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_KAN_SOFT_CLIP_K")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(3.0)
});

/// Нижний clip online gain. Default = 0.5.
static KAN_GAIN_MIN: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_KAN_GAIN_MIN")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.5)
});

/// Верхний clip online gain. Default = 2.5.
static KAN_GAIN_MAX: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_KAN_GAIN_MAX")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(2.5)
});

/// β EMA для online gain. Default = 0.9 (half-life ≈ 7 шагов).
static KAN_GAIN_EMA_BETA: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_KAN_GAIN_EMA_BETA")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.9)
});

/// ε в знаменателях. Default = 1e-12.
static KAN_EPS: Lazy<f32> = Lazy::new(|| {
    std::env::var("NEUROCORE_KAN_EPS")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1e-12)
});

// ============================================================================
// Раскладка параметров
// ============================================================================

const MIXTURE_BRANCHES: usize = 2;
const COARSE_NUM_COEFFS: usize = 6;
const FINE_NUM_COEFFS: usize = 11;

/// Описание одной группы параметров внутри `own_slice`.
#[derive(Clone, Copy)]
struct Group {
    start: usize,
    len: usize,
}

/// Разбивает срез длины `out + 21·in·out` на 6 групп.
fn group_layout(in_feat: usize, out_feat: usize) -> [Group; 6] {
    let edges = in_feat * out_feat;
    let bias_start = 0;
    let mix_logits_start = bias_start + out_feat;
    let mix_temp_start = mix_logits_start + edges * MIXTURE_BRANCHES;
    let spline_c_start = mix_temp_start + edges;
    let spline_f_start = spline_c_start + edges * COARSE_NUM_COEFFS;
    let base_w_start = spline_f_start + edges * FINE_NUM_COEFFS;
    [
        Group { start: bias_start, len: out_feat },
        Group { start: mix_logits_start, len: edges * MIXTURE_BRANCHES },
        Group { start: mix_temp_start, len: edges },
        Group { start: spline_c_start, len: edges * COARSE_NUM_COEFFS },
        Group { start: spline_f_start, len: edges * FINE_NUM_COEFFS },
        Group { start: base_w_start, len: edges },
    ]
}

// ============================================================================
// Soft-clamp
// ============================================================================

/// Мягкий log-симметричный clamp с асимптотами `lo` и `hi`.
///
/// При `x = 1.0` возвращает ровно 1.0 (нейтральное значение не
/// сдвигается). При `x ≫ 1` асимптотически сжимается к `hi`; при
/// `x → 0+` — к `lo`. При `k → ∞` вырождается в жёсткий `clamp`.
#[inline]
fn soft_clamp(x: f32, lo: f32, hi: f32, k: f32) -> f32 {
    debug_assert!(lo > 0.0 && lo < 1.0, "soft_clamp: lo must be in (0,1)");
    debug_assert!(hi > 1.0, "soft_clamp: hi must be > 1");
    debug_assert!(k > 0.0, "soft_clamp: k must be > 0");

    if x >= 1.0 {
        1.0 + soft_cap_positive(x - 1.0, hi - 1.0, k)
    } else {
        1.0 - soft_cap_positive(1.0 - x, 1.0 - lo, k)
    }
}

/// Мягкий «cap» положительной величины. Возвращает значение в `[0, cap)`.
#[inline]
fn soft_cap_positive(y: f32, cap: f32, k: f32) -> f32 {
    if y <= 0.0 || cap <= 0.0 {
        return 0.0;
    }
    let ratio = y / cap;
    y / (1.0 + ratio.powf(k)).powf(1.0 / k)
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
        DynamicContext::Buffered(BufferedContext::MultiResolutionKANLinear { input }) => {
            let cols = input.cols();
            if cols > 0 { Some(cols) } else { None }
        }
        _ => None,
    }
}

/// Adaptive group RMS scale.
///
/// Включает в себя adaptive β_rms и soft-clamp. Online gain
/// применяется снаружи (общий на срез).
#[inline]
fn group_scale(
    rms_g: f32,
    target_rms: f32,
    beta: f32,
    alpha: f32,
    min_scale: f32,
    max_scale: f32,
    eps: f32,
    soft_k: f32,
) -> f32 {
    let r_g = if target_rms > 1e-30 {
        rms_g / target_rms
    } else {
        0.0
    };
    let beta_eff = beta * (1.0 + alpha * r_g);
    let num = target_rms * (1.0 + beta_eff);
    let den = rms_g + beta_eff * target_rms + eps;
    let raw = if den > 1e-30 { num / den } else { 1.0 };
    soft_clamp(raw, min_scale, max_scale, soft_k)
}

// ============================================================================
// Адаптер
// ============================================================================

/// Градиентный адаптер `MultiResolutionKANLinear`.
///
/// Единственный режим — adaptive group RMS equalize. Внутри хранит
/// только состояние online gain (EMA по срезу).
pub struct MultiResolutionKANLinearAdapter {
    /// EMA of per-sample per-param RMS среза.
    ///
    /// Ключ — `(buffer_idx, slice.start)`. Уникален для каждого среза
    /// в графе.
    gain_state: Mutex<HashMap<(usize, usize), f32>>,
}

impl MultiResolutionKANLinearAdapter {
    pub fn new() -> Self {
        Self {
            gain_state: Mutex::new(HashMap::new()),
        }
    }

    /// Online gain для конкретного среза.
    ///
    /// Первый вызов: инициализирует EMA значением `rms_cur`,
    /// возвращает `1.0`.
    ///
    /// Последующие: `gain = clamp(ema/rms, GAIN_MIN, GAIN_MAX)`,
    /// обновление EMA.
    fn compute_gain_eff(&self, ctx: &AdapterContext<'_>, rms_cur: f32) -> f32 {
        let key = (ctx.own_slice.buffer_idx(), ctx.own_slice.start);
        let mut state = self.gain_state.lock().unwrap();

        match state.get(&key).copied() {
            None => {
                state.insert(key, rms_cur);
                1.0
            }
            Some(ema) => {
                let eps = *KAN_EPS;
                let gain_raw = if rms_cur > 1e-30 {
                    ema / (rms_cur + eps)
                } else {
                    1.0
                };
                let gain_eff = gain_raw.clamp(*KAN_GAIN_MIN, *KAN_GAIN_MAX);

                let beta = *KAN_GAIN_EMA_BETA;
                let new_ema = beta * ema + (1.0 - beta) * rms_cur;
                state.insert(key, new_ema);

                gain_eff
            }
        }
    }
}

impl Default for MultiResolutionKANLinearAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl GradientAdapter for MultiResolutionKANLinearAdapter {
    fn apply(&self, ctx: &AdapterContext<'_>) {
        // --------------------------------------------------------------------
        // Валидация контракта AdapterContext.
        // --------------------------------------------------------------------
        let params_len = ctx.segment_params.rows() * ctx.segment_params.cols();
        let grads_len = ctx.segment_grads.rows() * ctx.segment_grads.cols();
        debug_assert!(
            ctx.own_slice.end() <= params_len,
            "KANAdapter: own_slice.end()={} exceeds segment_params len={}",
            ctx.own_slice.end(), params_len
        );
        debug_assert!(
            ctx.own_slice.end() <= grads_len,
            "KANAdapter: own_slice.end()={} exceeds segment_grads len={}",
            ctx.own_slice.end(), grads_len
        );
        debug_assert!(ctx.batch > 0, "KANAdapter: batch must be positive");
        debug_assert!(
            ctx.optimizer_applied,
            "KANAdapter: optimizer_applied must be true (I-1 phase-order)"
        );

        // --------------------------------------------------------------------
        // GPU-ветка заморожена (I-5).
        // --------------------------------------------------------------------
        if ctx.segment_grads.is_gpu() {
            debug_assert!(
                ctx.segment_params.is_gpu(),
                "KANAdapter: I-5 violation — grads GPU, params CPU"
            );
            return;
        }

        // --------------------------------------------------------------------
        // Извлечение in_features из forward-контекста.
        // --------------------------------------------------------------------
        let Some(in_feat) = extract_in_features(ctx.forward_ctx) else {
            return;
        };
        if in_feat == 0 {
            return;
        }

        // --------------------------------------------------------------------
        // Восстановление out_features из длины среза:
        //   L = out + 21·in·out = out·(21·in + 1)  ⇒  out = L / (21·in + 1)
        // --------------------------------------------------------------------
        let total_len = ctx.own_slice.len;
        let denom = 21 * in_feat + 1;
        if total_len == 0 || total_len % denom != 0 {
            return;
        }
        let out_feat = total_len / denom;
        if out_feat == 0 {
            return;
        }

        // --------------------------------------------------------------------
        // Online gain: rms_global_ps считается по СЫРОМУ градиенту
        // до применения scale.
        // --------------------------------------------------------------------
        let rms_global_ps = {
            let g_full = ctx.read_own_grads();
            let b = ctx.batch.max(1) as f32;
            let n = g_full.len() as f32;
            if n > 0.0 {
                l2_norm(&g_full) / (b.sqrt() * n.sqrt())
            } else {
                0.0
            }
        };

        // Защита от мёртвых градиентов: если rms среза ~ 0, не трогаем.
        if rms_global_ps < 1e-20 {
            return;
        }

        let gain_eff = self.compute_gain_eff(ctx, rms_global_ps);

        // --------------------------------------------------------------------
        // Основной проход: adaptive group RMS + soft-clamp + online gain.
        // --------------------------------------------------------------------
        let mut g = ctx.read_own_grads();
        let b = ctx.batch.max(1) as f32;
        let sqrt_b = b.sqrt();
        let groups = group_layout(in_feat, out_feat);

        // Per-param per-sample RMS каждой группы.
        let mut rms_arr = [0.0f32; 6];
        for (idx, grp) in groups.iter().enumerate() {
            if grp.len == 0 {
                continue;
            }
            let end = grp.start + grp.len;
            let g_g = &g[grp.start..end];
            rms_arr[idx] = l2_norm(g_g) / (sqrt_b * (grp.len as f32).sqrt());
        }

        let beta = *KAN_BETA;
        let alpha = *KAN_ALPHA;
        let eps = *KAN_EPS;
        let min_scale = *KAN_MIN_SCALE;
        let max_scale = *KAN_MAX_SCALE;
        let soft_k = *KAN_SOFT_CLIP_K;

        for (idx, grp) in groups.iter().enumerate() {
            if grp.len == 0 {
                continue;
            }
            let end = grp.start + grp.len;

            let scale = group_scale(
                rms_arr[idx], rms_global_ps,
                beta, alpha, min_scale, max_scale, eps, soft_k,
            );
            let final_scale = scale * gain_eff;

            let g_g = &mut g[grp.start..end];
            for v in g_g.iter_mut() {
                *v *= final_scale;
            }
        }

        ctx.write_own_grads(&g);
    }

    fn name(&self) -> &'static str {
        "kan"
    }
}
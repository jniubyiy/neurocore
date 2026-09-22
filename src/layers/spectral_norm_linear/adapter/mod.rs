// src/layers/spectral_norm_linear/adapter/mod.rs

//! Градиентный адаптер слоя `SpectrallyNormalizedLinear`
//! (MIGRATION_PLAN.md §7, Фаза 5+).
//!
//! # Пилот — не no-op, в отличие от `LinearAdapter`
//!
//! Слой `SpectrallyNormalizedLinear` уже **имеет** встроенную нормировку
//! весов через `scale`: по построению `‖W_sn‖_spectral = |scale|`.
//! Поэтому «чистая» LARS-нормировка вида `‖W‖ / (…)` здесь **неверна** —
//! масштаб сырых `W` не определён однозначно:
//!
//! ```text
//!   σ(k·W)     = k · σ(W)
//!   W_sn(k·W)  = kW · (scale / (k·σ)) = W · (scale/σ) = W_sn(W)
//! ```
//!
//! Слой не различает `W` и `k·W`. Числитель LARS поэтому берётся не от
//! `‖W‖`, а от **эффективной** спектральной нормы `W_sn`:
//!
//! ```text
//!   W_eff_norm = |scale|
//! ```
//!
//! # Формула (per-tensor, основной режим)
//!
//! ```text
//!   B             = ctx.batch
//!   g_norm_sum    = ‖∇W‖₂                     (L2 норма градиента по W)
//!   g_norm_ps     = g_norm_sum / √B           (batch-инвариантная оценка)
//!   W_eff_norm    = |scale|                   (спектральная норма W_sn)
//!   denom         = g_norm_ps + β·W_eff_norm + ε
//!   scale_raw     = W_eff_norm / denom
//!   scale         = clamp(scale_raw, min, max)
//!   ∇W_sum       *= scale
//! ```
//!
//! `bias` и `scale`-параметр не модифицируются: bias — свободный сдвиг,
//! scale — отдельный обучаемый скаляр, его шаг регулирует оптимизатор.
//!
//! # Инвариантность к масштабу `W`
//!
//! При `W ← k·W` (для любого `k ≠ 0`):
//!   * `σ ← k·σ`;
//!   * `g_norm_sum ← k·g_norm_sum` (градиент линеен по W);
//!   * `scale` — **не меняется** (это отдельный параметр);
//!   * следовательно множитель LARS — **не меняется**.
//!
//! Это существенно отличает SN-адаптер от `LinearAdapter`, где `‖W‖`
//! стоял в числителе и создавал масштабную зависимость.
//!
//! # Batch-инвариантность
//!
//! `‖∇W_batch‖ = ‖Σ_i g_i‖ ≈ √B · ‖g_per_sample‖` в шумовом режиме
//! (E[g]=0, независимые сэмплы). Оценка per-sample нормы — `‖∇W_batch‖/√B`,
//! стандартная практика LARS/K-FAC.
//!
//! # Env-флаги
//!
//! | Env | Default | Смысл |
//! |-----|---------|-------|
//! | `NEUROCORE_SN_ADAPTER_MODE`   | `per_tensor` | `per_tensor` / `per_row` |
//! | `NEUROCORE_SN_LARS_BETA`      | `0.5`  | β в знаменателе |
//! | `NEUROCORE_SN_LARS_MIN_SCALE` | `0.01` | Нижний clip |
//! | `NEUROCORE_SN_LARS_MAX_SCALE` | `1.5`  | Верхний clip |
//! | `NEUROCORE_SN_LARS_EPS`       | `1e-12`| ε в знаменателе |
//! | `NEUROCORE_DEBUG_SN_ADAPTER`  | —      | Per-call трейс |
//!
//! # GPU
//!
//! Заморожена (как у `LinearAdapter`). Если `ctx.segment_grads.is_gpu()`,
//! адаптер пропускает модификацию (I-5).
//!
//! # Инварианты (MIGRATION_PLAN.md §2)
//!
//!   * I-1: `optimizer_applied == true`;
//!   * I-2: Loss отдаёт сырой Sum-градиент;
//!   * I-3: адаптер трогает только `grad_params`;
//!   * I-4: адаптер в папке слоя;
//!   * I-5: CPU↔CPU / GPU↔GPU;
//!   * I-8: `&self`, состояние — `AtomicBool`;
//!   * I-10: `own_slice` — легитимный член `all_slices`.

mod adapter;

pub use adapter::SpectralNormLinearAdapter;
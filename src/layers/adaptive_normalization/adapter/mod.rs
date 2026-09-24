// src/layers/adaptive_normalization/adapter/mod.rs

//! радиентный адаптер слоя \AdaptiveNormalization//! (MIGRATION_PLAN.md §7, аза 5+).
//!
//! # ачем адаптер
//!
//! \AdaptiveNormalization\ объединяет LayerNorm, RMSNorm и BatchNorm с
//! обучаемыми логитами выбора ветки для каждого признака. Сырой градиент
//! имеет сильные парадоксы:
//!
//! - **огиты** (группы 5–6) управляют softmax-весами ветвей; агрессивное
//!   обновление вызывает резкое переключение нормировки → всплески loss.
//! - **BN-параметры** (группы 3–4) имеют batch-dependent статистики →
//!   градиент ∝ √B, нужен пересчёт на per-sample.
//! - **Gamma/Beta** (группы 0–2) масштабируют выход → нужен LARS-масштаб
//!   относительности параметра, чтобы не «перепрыгнуть» через оптимум.
//!
//! # ормула
//!
//! даптер всегда применяет единую схему (single-mode, permanent):
//!
//! \\	ext
//! final_scale = lars_scale · gain_eff
//! \//!
//! де:
//! - \lars_scale\ — per-group LARS-множитель (adaptive β + soft-clamp).
//!   ля logits-групп (5–6) используются более консервативные параметры
//!   (\β_logits\, \MAX_logits\), чтобы защитить выходы модели.
//! - \gain_eff\ — online gain по EMA (демпфирование выбросов, ускорение
//!   в стагнации).
//!
//! # нварианты (MIGRATION_PLAN.md §2)
//!
//! - I-1: \optimizer_applied == true\;
//! - I-2: Loss отдаёт сыной Sum-градиент;
//! - I-3: адаптер трогает только \grad_params\;
//! - I-4: адаптер в папке слоя;
//! - I-5: CPU↔CPU / GPU↔GPU (GPU-ветка заморожена);
//! - I-8: \&self\, состояние — внутри \Mutex\;
//! - I-10: \own_slice\ — легитимный член \ll_slices\.

mod adapter;

pub use adapter::AdaptiveNormAdapter;

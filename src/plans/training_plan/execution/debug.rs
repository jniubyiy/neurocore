// src/plans/training_plan/execution/debug.rs
//
// Отладочные переключатели, константы и утилиты форматирования
// диагностических сообщений.

use once_cell::sync::Lazy;

// ============================================================================
// Отладочные переключатели
// ============================================================================

pub(super) static TRAIN_DEBUG: Lazy<bool> =
    Lazy::new(|| std::env::var("NEUROCORE_DEBUG_TRAIN").is_ok());

pub(super) static LAYER_DEBUG: Lazy<bool> =
    Lazy::new(|| std::env::var("NEUROCORE_DEBUG_LAYERS").is_ok());

pub(super) static NAN_DEBUG: Lazy<bool> =
    Lazy::new(|| std::env::var("NEUROCORE_DEBUG_NAN").is_ok());

// ============================================================================
// Константы логирования и диагностики
// ============================================================================

pub(super) const LOG_FIRST_EPOCHS: usize = 3;
pub(super) const LOG_NORM_EVERY: usize = 10;
pub(super) const ANOMALY_MULTIPLIER: f32 = 2.0;
pub(super) const ANOMALY_MIN_LOSS: f32 = 0.05;
pub(super) const ANOMALY_DIAG_LIMIT: usize = 10;

pub(super) const LOG_LSF_EVO_FIRST: usize = 20;
pub(super) const LOG_LSF_EVO_EVERY: usize = 20;

// ============================================================================
// Утилиты
// ============================================================================

pub(super) fn train_dbg_l2(data: &[f32]) -> f64 {
    let mut s = 0.0f64;
    for &v in data {
        if v.is_finite() {
            s += (v as f64) * (v as f64);
        }
    }
    s.sqrt()
}

pub(super) fn train_dbg_print_stats(name: &str, data: &[f32]) {
    if data.is_empty() {
        println!("    [TRAIN] {}: <empty>", name);
        return;
    }
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut nan_cnt = 0usize;
    let mut inf_cnt = 0usize;
    for &v in data {
        if v.is_nan() { nan_cnt += 1; continue; }
        if v.is_infinite() { inf_cnt += 1; continue; }
        if v < mn { mn = v; }
        if v > mx { mx = v; }
        sum += v as f64;
    }
    let finite = data.len().saturating_sub(nan_cnt + inf_cnt);
    let mean = if finite > 0 { sum / finite as f64 } else { 0.0 };
    println!(
        "    [TRAIN] {}: len={}, min={:.6}, max={:.6}, mean={:.6}, l2={:.6}, nan={}, inf={}",
        name, data.len(), mn, mx, mean, train_dbg_l2(data), nan_cnt, inf_cnt
    );
}

pub(super) fn train_dbg_print_vec(label: &str, data: &[f32]) {
    println!("    [TRAIN] {} (len={}): {:?}", label, data.len(), data);
}

pub(super) fn train_dbg_summary(data: &[f32]) -> String {
    if data.is_empty() {
        return "len=0".to_string();
    }
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut nan_cnt = 0usize;
    let mut inf_cnt = 0usize;
    for &v in data {
        if v.is_nan() { nan_cnt += 1; continue; }
        if v.is_infinite() { inf_cnt += 1; continue; }
        if v < mn { mn = v; }
        if v > mx { mx = v; }
        sum += v as f64;
    }
    let l2 = train_dbg_l2(data);
    format!(
        "len={} sum={:.6e} l2={:.6e} min={:.6e} max={:.6e} nan={} inf={}",
        data.len(), sum, l2, mn, mx, nan_cnt, inf_cnt
    )
}
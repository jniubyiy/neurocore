// src/optimizers/apply_update/cpu/mod.rs

use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};

use once_cell::sync::Lazy;

use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::plans::optimizer_plan::cube::OptimizerCube;

use super::super::apply_update::ApplyUpdate;

// ============================================================================
// Отладочные переключатели для ApplyUpdate (CPU)
// ============================================================================
//
// Логи включаются переменной окружения NEUROCORE_DEBUG_TRAIN=1
// (тот же флаг, что и для execution.rs — чтобы видеть весь след).
//
// Печатает норму градиента ||g|| и норму параметров ||p|| до/после апдейта
// на первых APPLY_LOG_LIMIT вызовах, а также любые вызовы с NaN/Inf.
//
// При обнаружении аномалии печатаются ПОЛНЫЕ срезы g и p (все 92 элемента),
// чтобы можно было точно локализовать источник (индексы, значения).
// ============================================================================

static APPLY_DEBUG: Lazy<bool> =
    Lazy::new(|| std::env::var("NEUROCORE_DEBUG_TRAIN").is_ok());
static APPLY_CALLS: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(0));

const APPLY_LOG_LIMIT: usize = 3;

#[inline]
fn apply_dbg_l2(data: &[f32]) -> f64 {
    let mut s = 0.0f64;
    for &v in data {
        if v.is_finite() {
            s += (v as f64) * (v as f64);
        }
    }
    s.sqrt()
}

impl OptimizerCube for ApplyUpdate {
    fn state_size_per_param(&self) -> usize {
        0
    }

    fn apply_buffered_handle(
        &self,
        params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        _state: &MatrixBufferHandle,
    ) {
        assert!(
            !params.is_gpu() && !grads.is_gpu(),
            "ApplyUpdate: params and grads must be CPU"
        );

        // Решаем, логировать ли текущий вызов.
        let (log_this, call_id) = if *APPLY_DEBUG {
            let n = APPLY_CALLS.fetch_add(1, Ordering::Relaxed);
            (n < APPLY_LOG_LIMIT, n)
        } else {
            (false, 0usize)
        };

        let mut param_guard = params.write();
        let p_slice = param_guard
            .as_slice_mut()
            .expect("ApplyUpdate: expected CPU buffer");
        let grad_guard = grads.read();
        let g_slice = grad_guard
            .as_slice()
            .expect("ApplyUpdate: expected CPU buffer");

        debug_assert_eq!(p_slice.len(), g_slice.len());

        // Проверяем аномалии ДО update.
        let has_anom_before = p_slice.iter().any(|v| !v.is_finite())
            || g_slice.iter().any(|v| !v.is_finite());

        if *APPLY_DEBUG && (log_this || has_anom_before) {
            if has_anom_before && !log_this {
                println!(
                    "[APPLY #{}] ANOMALY DETECTED (non-finite in p or g BEFORE update)",
                    call_id
                );
            }
            println!(
                "[APPLY #{}] len={}, ||g||={:.6}, ||p||_before={:.6}",
                call_id,
                p_slice.len(),
                apply_dbg_l2(g_slice),
                apply_dbg_l2(p_slice),
            );
            if has_anom_before {
                // Полный срез: сколько элементов, все значения.
                // Печатаем по частям (по 16 значений), чтобы строки были читаемы.
                println!("    FULL g (len={}):", g_slice.len());
                for (chunk_idx, chunk) in g_slice.chunks(16).enumerate() {
                    println!("      g[{:>3}..{:>3}] = {:?}", chunk_idx * 16,
                             chunk_idx * 16 + chunk.len(), chunk);
                }
                println!("    FULL p_before (len={}):", p_slice.len());
                for (chunk_idx, chunk) in p_slice.chunks(16).enumerate() {
                    println!("      p[{:>3}..{:>3}] = {:?}", chunk_idx * 16,
                             chunk_idx * 16 + chunk.len(), chunk);
                }
            } else {
                let show = p_slice.len().min(5);
                println!("    first g[:{}] = {:?}", show, &g_slice[..show]);
                println!("    first p[:{}] (before) = {:?}", show, &p_slice[..show]);
            }
        }

        for i in 0..p_slice.len() {
            p_slice[i] -= g_slice[i];
        }

        let has_anom_after = p_slice.iter().any(|v| !v.is_finite());

        if *APPLY_DEBUG && (log_this || has_anom_after) {
            println!(
                "    ||p||_after = {:.6}{}",
                apply_dbg_l2(p_slice),
                if has_anom_after { "  <-- NaN/Inf in p after update" } else { "" }
            );
            if has_anom_after {
                println!("    FULL p_after (len={}):", p_slice.len());
                for (chunk_idx, chunk) in p_slice.chunks(16).enumerate() {
                    println!("      p[{:>3}..{:>3}] = {:?}", chunk_idx * 16,
                             chunk_idx * 16 + chunk.len(), chunk);
                }
            } else {
                let show = p_slice.len().min(5);
                println!("    first p[:{}] (after)  = {:?}", show, &p_slice[..show]);
            }
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
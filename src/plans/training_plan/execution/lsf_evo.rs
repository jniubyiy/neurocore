// src/plans/training_plan/execution/lsf_evo.rs
//
// Диагностика эволюции β и θ слоя LearnableSoftplus.

use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::Model;

use super::debug::train_dbg_l2;

pub(super) struct LsfEvoSnap {
    model_idx: usize,
    layer_idx: usize,
    features: usize,
    raw_betas: Vec<f32>,
    thetas: Vec<f32>,
    grad_raw_betas: Vec<f32>,
    grad_thetas: Vec<f32>,
}

pub(super) fn collect_lsf_evo(model: &MixedModel) -> Vec<LsfEvoSnap> {
    let ps_guard = model.param_store().lock().unwrap();
    let mut out = Vec::new();
    for (m_idx, m) in model.models().iter().enumerate() {
        if let Model::UniversalProcessor(layers, slices, _) = m {
            for (l_idx, layer) in layers.iter().enumerate() {
                if let Some(lsf) = layer.as_learnable_softplus() {
                    let f = lsf.features;
                    let slice = &slices[l_idx];
                    let params_h = ps_guard.params_handle(slice);
                    let grads_h = ps_guard.grads_handle(slice);
                    if params_h.is_gpu() || grads_h.is_gpu() {
                        continue;
                    }
                    let raw_betas = params_h.read_range(slice.start, f);
                    let thetas = params_h.read_range(slice.start + f, f);
                    let grad_raw_betas = grads_h.read_range(slice.start, f);
                    let grad_thetas = grads_h.read_range(slice.start + f, f);
                    out.push(LsfEvoSnap {
                        model_idx: m_idx,
                        layer_idx: l_idx,
                        features: f,
                        raw_betas,
                        thetas,
                        grad_raw_betas,
                        grad_thetas,
                    });
                }
            }
        }
    }
    out
}

pub(super) fn print_lsf_evo(epoch: usize, snaps: &[LsfEvoSnap]) {
    if snaps.is_empty() {
        println!("  [LSF-EVOLUTION ep{}] (no LearnableSoftplus on CPU found)", epoch);
        return;
    }
    for s in snaps {
        let mut beta_min = f32::INFINITY;
        let mut beta_max = f32::NEG_INFINITY;
        let mut beta_sum = 0.0f32;
        let mut beta_dev_inf = 0.0f32;
        let mut n_valid = 0usize;
        for &rb in &s.raw_betas {
            let b = (1.0 + rb).max(1e-3);
            if b < beta_min { beta_min = b; }
            if b > beta_max { beta_max = b; }
            beta_sum += b;
            let dev = (b - 1.0).abs();
            if dev > beta_dev_inf { beta_dev_inf = dev; }
            n_valid += 1;
        }
        let beta_mean = if n_valid > 0 { beta_sum / n_valid as f32 } else { 0.0 };

        let mut theta_min = f32::INFINITY;
        let mut theta_max = f32::NEG_INFINITY;
        let mut theta_sum = 0.0f32;
        let mut theta_abs_inf = 0.0f32;
        for &t in &s.thetas {
            if t < theta_min { theta_min = t; }
            if t > theta_max { theta_max = t; }
            theta_sum += t;
            let a = t.abs();
            if a > theta_abs_inf { theta_abs_inf = a; }
        }
        let theta_mean = if !s.thetas.is_empty() {
            theta_sum / s.thetas.len() as f32
        } else {
            0.0
        };

        let grad_beta_l2 = train_dbg_l2(&s.grad_raw_betas);
        let grad_theta_l2 = train_dbg_l2(&s.grad_thetas);
        let grad_beta_inf = s.grad_raw_betas.iter().map(|v| v.abs()).fold(0.0, f32::max);
        let grad_theta_inf = s.grad_thetas.iter().map(|v| v.abs()).fold(0.0, f32::max);

        println!(
            "  [LSF-EVOLUTION ep{}] m{} l{} F={}",
            epoch, s.model_idx, s.layer_idx, s.features
        );
        println!(
            "    β: min={:.6} max={:.6} mean={:.6} |β−1|_inf={:.6}",
            beta_min, beta_max, beta_mean, beta_dev_inf
        );
        println!(
            "    θ: min={:.6} max={:.6} mean={:.6} |θ|_inf={:.6}",
            theta_min, theta_max, theta_mean, theta_abs_inf
        );
        println!("    grad_raw_β: l2={:.6} |·|_inf={:.6}", grad_beta_l2, grad_beta_inf);
        println!("    grad_θ:     l2={:.6} |·|_inf={:.6}", grad_theta_l2, grad_theta_inf);
    }
}
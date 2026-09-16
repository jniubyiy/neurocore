// src/plans/training_plan/execution/nan_debug.rs
//
// Точечная диагностика NaN/Inf в процессе обучения.
// Включается флагом `NAN_DEBUG` (см. `debug.rs`).

use super::debug::train_dbg_print_stats;

pub(super) struct NanTracker {
    init_reported: bool,
    forward_reported: bool,
    loss_reported: bool,
    backward_reported: bool,
    update_reported: bool,
    pub total_steps_observed: usize,
}

impl NanTracker {
    pub fn new() -> Self {
        Self {
            init_reported: false,
            forward_reported: false,
            loss_reported: false,
            backward_reported: false,
            update_reported: false,
            total_steps_observed: 0,
        }
    }

    pub fn bump(&mut self) {
        self.total_steps_observed += 1;
    }

    pub fn should_report(&self, stage: &str) -> bool {
        match stage {
            "init" => !self.init_reported,
            "forward" => !self.forward_reported,
            "loss" => !self.loss_reported,
            "backward" => !self.backward_reported,
            "update" => !self.update_reported,
            _ => false,
        }
    }

    pub fn mark_reported(&mut self, stage: &str) {
        match stage {
            "init" => self.init_reported = true,
            "forward" => self.forward_reported = true,
            "loss" => self.loss_reported = true,
            "backward" => self.backward_reported = true,
            "update" => self.update_reported = true,
            _ => {}
        }
    }
}

pub(super) fn contains_bad(data: &[f32]) -> bool {
    data.iter().any(|v| !v.is_finite())
}

fn first_bad_indices(data: &[f32], limit: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(limit);
    for (i, v) in data.iter().enumerate() {
        if !v.is_finite() {
            out.push(i);
            if out.len() >= limit {
                break;
            }
        }
    }
    out
}

pub(super) fn report_first_nan(
    stage: &str,
    epoch: usize,
    batch_idx: usize,
    shapes: &[(&str, usize, usize)],
    data: &[f32],
    extra: Option<&str>,
) {
    println!();
    println!("========== [NAN-DEBUG] first NaN at stage = '{}' ==========", stage);
    println!("  epoch = {}, batch_idx = {}", epoch, batch_idx);
    if let Some(e) = extra {
        println!("  extra: {}", e);
    }
    println!("  buffer shapes:");
    for (name, r, c) in shapes {
        println!("    {}: {} x {}", name, r, c);
    }
    println!("  buffer len = {}", data.len());
    train_dbg_print_stats(&format!("{} buffer", stage), data);

    let bad_idx = first_bad_indices(data, 16);
    println!("  first bad indices (up to 16): {:?}", bad_idx);
    println!("  first 16 values:");
    let show = data.len().min(16);
    println!("    {:?}", &data[..show]);

    println!("  last 16 values:");
    if data.len() >= 16 {
        println!("    {:?}", &data[data.len() - 16..]);
    } else {
        println!("    {:?}", data);
    }

    println!("========== [NAN-DEBUG] end ==========");
    println!();
}
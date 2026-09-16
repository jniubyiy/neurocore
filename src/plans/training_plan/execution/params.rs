// src/plans/training_plan/execution/params.rs
//
// Безопасное чтение параметров модели и вспомогательные метрики.

use crate::compute_manager::graph::model::MixedModel;

pub(super) fn read_all_params_safe(model: &MixedModel) -> Option<Vec<f32>> {
    let ps = model.param_store().lock().unwrap();
    if ps.is_empty() {
        return Some(Vec::new());
    }
    let gpu_opt = model.compute_executor().gpu_compute();
    let mut result = Vec::with_capacity(ps.total_params());
    for buffer_idx in 0..ps.num_buffers() {
        let buf = ps.get_param_buffer_by_idx(buffer_idx);
        if buf.params.is_gpu() {
            let gpu = gpu_opt.as_ref()?;
            let data = gpu.download_gpu_handle_to_vec(&buf.params);
            result.extend_from_slice(&data);
        } else {
            let guard = buf.params.read();
            let slice = guard.as_slice()?;
            result.extend_from_slice(slice);
        }
    }
    Some(result)
}

pub(super) fn per_sample_mse(pred_flat: &[f32], target_flat: &[f32]) -> f32 {
    assert_eq!(pred_flat.len(), target_flat.len());
    let n = pred_flat.len();
    if n == 0 { return 0.0; }
    let mut s = 0.0f32;
    for i in 0..n {
        let d = pred_flat[i] - target_flat[i];
        s += d * d;
    }
    s / n as f32
}
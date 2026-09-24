// examples/adaptive_normalization_adapter_example.rs
//
// Demo of the AdaptiveNormalization gradient adapter (single permanent mode).
//
// The adapter is always-on (no mode switching). It combines:
//   - per-group LARS scaling with adaptive beta
//   - soft-clamp (smooth boundaries)
//   - online gain (EMA)
//   - logits shielding (groups 5-6 get gentler scaling)
//
// Run:
//   cargo run --example adaptive_normalization_adapter_example
//
// Tune via env vars (optional):
//   NEUROCORE_ADNORM_BETA, NEUROCORE_ADNORM_ALPHA
//   NEUROCORE_ADNORM_MIN_SCALE, NEUROCORE_ADNORM_MAX_SCALE
//   NEUROCORE_ADNORM_LOGITS_MAX_SCALE, NEUROCORE_ADNORM_LOGITS_BETA
//   NEUROCORE_ADNORM_EPS, NEUROCORE_ADNORM_SOFT_CLIP_K
//   NEUROCORE_ADNORM_GAIN_MIN, NEUROCORE_ADNORM_GAIN_MAX
//   NEUROCORE_ADNORM_GAIN_EMA_BETA

use neurocore::layers::adaptive_normalization::adapter::AdaptiveNormAdapter;
use neurocore::layers::adapter::{AdapterContext, GradientAdapter};
use neurocore::compute_manager::core::device_spec::DeviceSpec;
use neurocore::compute_manager::operators_v2::memory_v2::buffer::MatrixBufferHandle;
use neurocore::compute_manager::operators_v2::memory_v2::executor::MemoryExecutor;
use neurocore::compute_manager::operators_v2::memory_v2::policy::BufferPriority;
use neurocore::compute_manager::operators_v2::memory_v2::types::MemoryDeviceKind;
use neurocore::compute_manager::core::dynamic_context::DynamicContext;
use neurocore::layers::buffered_context::BufferedContext;
use neurocore::model_plan::param_store::ParamSlice;
use std::sync::{Arc, RwLock};

const NUM_GROUPS: usize = 7;
const F: usize = 8; // features per group

fn main() {
    println!("=== AdaptiveNormalization Adapter Demo ===");
    println!("Single permanent mode: per-group LARS + online gain + logits shielding");
    println!();

    // --- Setup memory executor ---
    let mem = Arc::new(RwLock::new(MemoryExecutor::new()));
    mem.write().unwrap()
        .register_compute_device(DeviceSpec::cpu(0, 4096, 1), None);
    mem.write().unwrap().set_self_arc(mem.clone());

    let total = NUM_GROUPS * F; // 56 parameters
    let batch = 4;

    // --- Create param/grad/input handles ---
    let params: Vec<f32> = (0..total).map(|i| i as f32 * 0.1 + 0.1).collect();
    let grads: Vec<f32> = (0..total)
        .map(|i| ((i as f32 - 14.0) * 0.01) as f32)
        .collect();

    let mut mem_write = mem.write().unwrap();
    let ph: MatrixBufferHandle = mem_write
        .acquire_matrix_handle(total, 1, MemoryDeviceKind::HostRam, BufferPriority::Medium)
        .expect("acquire params handle");
    let gh: MatrixBufferHandle = mem_write
        .acquire_matrix_handle(total, 1, MemoryDeviceKind::HostRam, BufferPriority::Medium)
        .expect("acquire grads handle");
    let ih: MatrixBufferHandle = mem_write
        .acquire_matrix_handle(batch, F, MemoryDeviceKind::HostRam, BufferPriority::Medium)
        .expect("acquire input handle");
    drop(mem_write);

    ph.write_range(0, &params);
    gh.write_range(0, &grads);

    // --- Build forward context ---
    let fc = DynamicContext::Buffered(
        BufferedContext::AdaptiveNormalization { input: ih }
    );

    // --- Build adapter context ---
    let own_slice = ParamSlice::new(0, 0, total);
    let all_slices = vec![own_slice];
    let ctx = AdapterContext {
        segment_params: &ph,
        segment_grads: &gh,
        own_slice,
        all_slices: &all_slices,
        batch,
        optimizer_applied: true,
        own_state_slice: None,
        adapter_store: None,
        forward_ctx: Some(&fc),
    };

    // --- Print gradient norms by group before ---
    println!("Feature groups: {} (features per group: {})", NUM_GROUPS, F);
    println!("Batch size: {}", batch);
    println!("Total params: {}", total);
    println!();

    let before = gh.read_range(0, total);
    print_group_norms("BEFORE", &before, F);

    // --- Apply adapter ---
    let adapter = AdaptiveNormAdapter::new();
    println!("Applying adapter: {}", adapter.name());
    adapter.apply(&ctx);
    println!("Adapter applied successfully.");

    // --- Print gradient norms by group after ---
    let after = gh.read_range(0, total);
    print_group_norms("AFTER ", &after, F);

    // --- Summary ---
    let scale_factors: Vec<f32> = after.iter().zip(before.iter())
        .map(|(a, b)| if b.abs() > 1e-30 { a / b } else { 1.0 })
        .collect();
    println!();
    println!("Scale factor stats:");
    let min_sf = scale_factors.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_sf = scale_factors.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let avg_sf: f32 = scale_factors.iter().sum::<f32>() / scale_factors.len() as f32;
    println!("  min: {:.6}", min_sf);
    println!("  max: {:.6}", max_sf);
    println!("  avg: {:.6}", avg_sf);
    println!();

    // --- Verify invariants ---
    let all_finite = after.iter().all(|v| v.is_finite());
    assert!(all_finite, "All gradients must be finite after adapter");
    println!("Invariant check: all gradients finite = PASS");
    println!();
    println!("=== Done ===");
}

fn print_group_norms(label: &str, grads: &[f32], f: usize) {
    let group_names = [
        "ln_gamma ", "ln_beta  ", "rms_gamma", "bn_gamma ",
        "bn_beta  ", "logits_ln", "logits_rms"
    ];
    println!("{} gradients by group:", label);
    for g in 0..NUM_GROUPS {
        let start = g * f;
        let group = &grads[start..start + f];
        let norm: f64 = group.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        let mean: f64 = group.iter().map(|&x| x as f64).sum::<f64>() / f as f64;
        println!("  [{}] L2={:.6}, mean={:.6}", group_names[g], norm, mean);
    }
}

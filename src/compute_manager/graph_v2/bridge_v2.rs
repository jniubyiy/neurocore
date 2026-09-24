// src/compute_manager/graph_v2/bridge_v2.rs
//
// Мост между v2-графом и каноническими per-layer инициализациями.
//
// Задача: после того как `GraphV2::init_params` применит generic
// Initializer ко всем параметрам, для ряда слоёв нужно переопределить
// значения на канонические.
//
//   * `BatchRenorm1d` — γ=1, β=0, r=1, d=0.
//   * `PerFeatureAttention` — `self_bias_h ∈ [2.0, 4.0)`.
//   * `LearnableMish` — λ = 1.0.
//   * `AdaptiveNormalization` — γ=1 для трёх ветвей, β=0, логиты=0.
//   * `AdaptiveSpaceCompress` — `center[p] = p/(p_max−1)`,
//     `b_L[p] = 0.0`, `compress[p] = 1.0`.

use crate::layers::UniversalLayer;

use super::model_v2::GraphV2;
use super::types_v2::SegmentKindV2;

/// Возвращает список `(buffer_idx, start, values)`, которые нужно
/// записать в `ParamStore` поверх generic-инициализации.
pub(crate) fn build_layer_aware_overrides_v2(
    graph: &GraphV2,
) -> Vec<(usize, usize, Vec<f32>)> {
    let mut out = Vec::new();

    for seg in &graph.segments {
        let SegmentKindV2::Universal { layers, slices } = &seg.kind else {
            continue;
        };

        for (i, layer) in layers.iter().enumerate() {
            let Some(slice) = slices.get(i) else {
                continue;
            };

            // ---------- BatchRenorm1d ----------
            if let Some(br) = layer.as_batch_renorm() {
                let f = br.features;
                let mut buf = vec![0.0f32; 4 * f];
                for c in 0..f {
                    buf[0 * f + c] = 1.0; // γ
                    buf[2 * f + c] = 1.0; // r
                }
                out.push((slice.buffer_idx, slice.start, buf));
            }

            // ---------- PerFeatureAttention ----------
            if let Some(pfa) = layer.as_per_feature_attention() {
                let d_model = pfa.d_model;
                let d_head = pfa.d_head;
                let head_pc = 10 * d_head + 2;

                let mut state: u64 = (slice.buffer_idx as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (slice.start as u64)
                        .wrapping_mul(0xBF58_476D_1CE4_E5B9);
                if state == 0 {
                    state = 0xDEAD_BEEF_CAFE_BABE;
                }

                for h in 0..d_model {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;

                    let unit =
                        ((state & 0x00FF_FFFF) as f32) / 0x0100_0000u32 as f32;

                    let sb = 2.0 + 2.0 * unit;
                    let sb_pos = slice.start + h * head_pc + 10 * d_head + 1;
                    out.push((slice.buffer_idx, sb_pos, vec![sb]));
                }
            }

            // ---------- LearnableMish ----------
            if layer.as_learnable_mish().is_some() {
                out.push((slice.buffer_idx, slice.start, vec![1.0f32]));
            }

            // ---------- AdaptiveNormalization ----------
            if let Some(adnorm) = layer.as_adaptive_normalization() {
                let f = adnorm.features;
                let mut buf = vec![0.0f32; 7 * f];
                for c in 0..f {
                    buf[0 * f + c] = 1.0; // γ_ln
                    buf[2 * f + c] = 1.0; // γ_rms
                    buf[3 * f + c] = 1.0; // γ_bn
                }
                out.push((slice.buffer_idx, slice.start, buf));
            }

            // ---------- AdaptiveSpaceCompress ----------
            //
            // Канонические значения:
            //   center[p]   = p / (p_max − 1)   (при p_max == 1 → 0.5)
            //   b_L[p]      = 0.0
            //   compress[p] = 1.0
            //
            // Раскладка:
            //   [center: p_max | b_L: p_max | compress: p_max | ...].
            //
            // Один блок из 3·p_max значений.
            if let Some(asc) = layer.as_adaptive_space_compress() {
                let p_max = asc.p_max;
                let mut buf = vec![0.0f32; 3 * p_max];
                for p in 0..p_max {
                    let pos = if p_max > 1 {
                        p as f32 / (p_max - 1) as f32
                    } else {
                        0.5
                    };
                    buf[p] = pos;
                    // b_L[p] = 0.0 — по умолчанию уже 0.
                    buf[2 * p_max + p] = 1.0; // compress[p] = 1.0
                }
                out.push((slice.buffer_idx, slice.start, buf));
            }
        }
    }

    out
}